"""Interactive inference CLI supporting both model-only and agent-based workflows."""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from supabase import Client, create_client

from models.build import build_model_from_cfg
from ops.agent_utils import load_agent_config, resolve_model_section
from ops.config import load_config
from scripts.load_gpt_weights import load_weights_into_gpt
from models.meta_arch.gpt_oss import GPTOSS, load_gpt_oss_weights
from ops.harmony import create_simple_prompt, parse_response
from ops.alpaca import format_alpaca_prompt, extract_alpaca_response

# Try to import openai_harmony for proper GPT-OSS inference
try:
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Message,
        Role,
        StreamableParser,
        StreamState,
        SystemContent,
        load_harmony_encoding,
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions using only the provided context. "
    "If the context does not contain the answer, reply that you do not know."
)
DEFAULT_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
EXIT_COMMANDS = {"/exit", "/quit", ":q", "q"}

HARMONY_TAG_PATTERN = re.compile(r"<\|[^|>]+?\|>")


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    greedy: bool = False,
) -> int:
    if greedy or temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / max(1e-5, temperature)
    probs = torch.softmax(logits, dim=-1)

    if top_k > 0:
        values, _ = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        threshold = values.min()
        probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        idx = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[idx].item())

    idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item())


def _strip_harmony_tokens(text: str) -> str:
    """Remove Harmony control tokens from decoded text."""
    return HARMONY_TAG_PATTERN.sub("", text)


def _resolve_inference_device(preferred: Optional[str] = None) -> torch.device:
    """Select the best available device, honoring an explicit preference if provided."""
    if preferred:
        try:
            device = torch.device(preferred)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid preferred device '{preferred}'") from exc

        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested CUDA device but torch.cuda.is_available() is False")

        if device.type == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not getattr(mps_backend, "is_available", lambda: False)():
                raise ValueError("Requested MPS device but torch.backends.mps.is_available() is False")

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
        return torch.device("mps")

    return torch.device("cpu")


def _generate_stream(
    model,
    tokenizer,
    prompt: Optional[str] = None,
    device: torch.device = None,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    greedy: bool = False,
    autocast_kwargs: Optional[Dict[str, Any]] = None,
    skip_special_tokens: bool = False,
    input_tokens: Optional[List[int]] = None,
):
    """Generate tokens one at a time, yielding each token ID.

    Args:
        model: The language model
        tokenizer: The tokenizer (used only if prompt is provided)
        prompt: Text prompt to encode (if input_tokens not provided)
        device: Device to run on
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        greedy: Use greedy decoding
        autocast_kwargs: Autocasting options
        skip_special_tokens: Whether to skip Harmony special tokens (deprecated, use parser instead)
        input_tokens: Pre-encoded token IDs (if provided, prompt is ignored)

    Yields:
        Generated token IDs (int)
    """
    model.eval()
    with torch.no_grad():
        # Get input token IDs
        if input_tokens is not None:
            ids = list(input_tokens)
        elif prompt is not None:
            input_ids = tokenizer.encode(prompt)
            ids = list(input_ids)
        else:
            raise ValueError("Either prompt or input_tokens must be provided")

        # GPT-OSS uses multiple EOS tokens
        eos_tokens = {200002, 199999, 200012, 200007}
        fallback_eos = getattr(tokenizer, "eos_id", None) if tokenizer else None
        if fallback_eos is not None:
            eos_tokens.add(fallback_eos)

        max_ctx = int(getattr(model, "max_seq_len", 1024))

        x = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            x_cond = x[:, -max_ctx:]
            if autocast_kwargs:
                with torch.autocast(**autocast_kwargs):
                    logits = model(x_cond)
            else:
                logits = model(x_cond)
            next_logits = logits[0, -1, :]
            next_token = _sample_next_token(
                next_logits.float(),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
            )
            ids.append(next_token)
            x = torch.tensor([ids], dtype=torch.long, device=device)

            # Stop at any EOS token
            if next_token in eos_tokens:
                break

            # Yield the token ID (caller decides how to decode/filter)
            yield next_token


class LocalResponder:
    """Wrapper around a locally hosted model for text generation."""

    def __init__(self, model_cfg: Dict[str, Any], weights_dir: Optional[str], checkpoint: Optional[str]):
        self.cfg = dict(model_cfg)
        if "model" not in self.cfg:
            raise ValueError("Local responder requires a full model config under 'model'.")

        # If --weights_dir is provided, remove config weights to avoid double-loading
        if weights_dir and "weights" in self.cfg.get("model", {}):
            print(f"Overriding config weights with --weights_dir: {weights_dir}")
            self.cfg["model"].pop("weights")

        preferred_device = os.environ.get("INFER_DEVICE") or os.environ.get("DEVICE")
        self.device = _resolve_inference_device(preferred_device)
        self.model_dtype: Optional[torch.dtype] = None
        self.autocast_kwargs: Optional[Dict[str, Any]] = None

        self.model = build_model_from_cfg(self.cfg)

        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None:
            raise AttributeError("Model must expose a tokenizer for inference.")

        # Determine dtype before loading weights
        # Only use reduced precision for GPT-OSS on GPU/MPS (it was trained in bfloat16)
        # For other models (like GPT-2), use full precision to avoid numerical issues
        if isinstance(self.model, GPTOSS):
            if self.device.type == "cuda":
                if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                    self.model_dtype = torch.bfloat16
                else:
                    self.model_dtype = torch.float16
            elif self.device.type == "mps":
                self.model_dtype = torch.float16

        # Move model to device BEFORE loading weights (critical for CUDA)
        if self.model_dtype is not None:
            self.model = self.model.to(self.device, dtype=self.model_dtype)
            self.autocast_kwargs = {"device_type": self.device.type, "dtype": self.model_dtype}
        else:
            self.model = self.model.to(self.device)

        # Load weights AFTER model is on the correct device
        self._load_weights(weights_dir, checkpoint)

        self.model.eval()

        eval_cfg = self.cfg.get("eval", {})
        self.max_new_tokens = int(eval_cfg.get("max_new_tokens", 200))
        self.temperature = float(eval_cfg.get("temperature", 1.0))
        self.top_k = int(eval_cfg.get("top_k", 0))
        self.top_p = float(eval_cfg.get("top_p", 0.0))
        self.greedy = bool(eval_cfg.get("greedy", False))
        self.system_prompt: Optional[str] = None
        self._warned_missing_harmony = False

        # Read prompt format from config - the config determines what format to use
        # Each model config should specify the format it was trained with
        self.prompt_format = eval_cfg.get("prompt_format", None)

    def _load_weights(self, weights_dir: Optional[str], checkpoint: Optional[str]) -> None:
        """Load weights from external source (overrides config weights if provided)."""
        if weights_dir:
            path = Path(weights_dir)

            # Check if path exists
            if not path.exists():
                raise FileNotFoundError(f"Weights path does not exist: {path.absolute()}")

            if isinstance(self.model, GPTOSS):
                load_gpt_oss_weights(self.model.backbone, path, self.device)
                return

            params = None
            if path.is_dir():
                pt_path = path / "params.pt"
                pkl_path = path / "params.pkl"
                if pkl_path.exists():
                    import pickle
                    with pkl_path.open("rb") as handle:
                        params = pickle.load(handle)
                elif pt_path.exists():
                    params = torch.load(pt_path, map_location="cpu", weights_only=False)
                else:
                    raise FileNotFoundError(f"No params.pt or params.pkl found in {path}")
            else:
                if path.suffix == ".pt":
                    params = torch.load(path, map_location="cpu", weights_only=False)
                elif path.suffix == ".pkl":
                    import pickle
                    with path.open("rb") as handle:
                        params = pickle.load(handle)
                else:
                    raise ValueError("weights_dir must point to a directory or .pt/.pkl file")
            load_weights_into_gpt(self.model, params)
        elif checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
        # If neither weights_dir nor checkpoint is provided, weights should have been loaded from config

    def set_system_prompt(self, system_prompt: Optional[str]) -> None:
        self.system_prompt = system_prompt.strip() if system_prompt else None

    def generate(self, prompt: str) -> str:
        """Convenience helper that returns the full generated text."""
        return "".join(self.generate_stream(prompt))

    def _is_gpt_oss_model(self) -> bool:
        """Check if the current model is a GPT-OSS model."""
        return isinstance(self.model, GPTOSS)

    def generate_stream(self, prompt: str):
        """Generate text token by token, yielding each token as it's generated."""
        # Route to appropriate formatting based on config
        # harmony: GPT-OSS models with Harmony library
        # alpaca: Instruction-tuned models (GPT-2, etc.)
        # None/other: Plain text generation

        if self.prompt_format == "harmony" and HARMONY_AVAILABLE:
            # Use official Harmony library with streaming parser
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

            # Build conversation with system and user messages
            messages = []
            if self.system_prompt:
                system_content = SystemContent.new()
                messages.append(Message.from_role_and_content(Role.SYSTEM, system_content))
            messages.append(Message.from_role_and_content(Role.USER, prompt))

            conversation = Conversation.from_messages(messages)
            input_tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

            # Use StreamableParser to parse generated tokens
            parser = StreamableParser(encoding, role=Role.ASSISTANT)

            # Generate tokens and parse them
            for token_id in _generate_stream(
                self.model,
                self.tokenizer,
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                greedy=self.greedy,
                autocast_kwargs=self.autocast_kwargs,
                skip_special_tokens=False,
                input_tokens=input_tokens,
            ):
                # Process token through parser
                parser.process(token_id)

                # Yield content deltas (parser automatically filters structural tokens)
                if parser.last_content_delta:
                    yield parser.last_content_delta
                elif parser.state == StreamState.EXPECT_START:
                    # New message starting - this is normal, continue
                    continue
                else:
                    # Debugging: yield raw decoded token if parser isn't giving us content
                    token_text = self.tokenizer.decode([token_id])
                    # Only yield if it's not a special token
                    if token_id not in {200005, 200006, 200007, 200008}:
                        yield token_text

        elif self.prompt_format == "harmony":
            # Harmony format without library - use fallback string formatting
            if not self._warned_missing_harmony:
                print("\nWarning: openai_harmony not installed. Install with: pip install openai-harmony")
                print("Falling back to Harmony prompt string without streaming parser.\n")
                self._warned_missing_harmony = True

            harmony_prompt = create_simple_prompt(prompt, self.system_prompt)
            generated_tokens: List[str] = []

            for token_id in _generate_stream(
                self.model,
                self.tokenizer,
                prompt=harmony_prompt,
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                greedy=self.greedy,
                autocast_kwargs=self.autocast_kwargs,
                skip_special_tokens=False,
            ):
                token_text = self.tokenizer.decode([token_id])
                generated_tokens.append(token_text)

            generated_suffix = "".join(generated_tokens)
            generated_text = harmony_prompt + generated_suffix
            try:
                parsed = parse_response(generated_text)
                final_answer = parsed.get("final_answer") or ""
                if final_answer:
                    yield final_answer
                else:
                    cleaned = _strip_harmony_tokens(generated_suffix).strip()
                    if cleaned:
                        yield cleaned
            except Exception:
                cleaned = _strip_harmony_tokens(generated_suffix).strip()
                if cleaned:
                    yield cleaned

        elif self.prompt_format == "alpaca":
            # Alpaca instruction format (for instruction-tuned models like fine-tuned GPT-2)
            full_prompt = format_alpaca_prompt(
                instruction=prompt,
                input_text=None,
                system_prompt=self.system_prompt
            )
            for token_id in _generate_stream(
                self.model,
                self.tokenizer,
                prompt=full_prompt,
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                greedy=self.greedy,
                autocast_kwargs=self.autocast_kwargs,
                skip_special_tokens=False,
            ):
                token_text = self.tokenizer.decode([token_id])
                yield token_text

        else:
            # Plain text generation (no formatting)
            full_prompt = prompt
            if self.system_prompt:
                full_prompt = f"{self.system_prompt}\n\n{prompt}".strip()

            for token_id in _generate_stream(
                self.model,
                self.tokenizer,
                prompt=full_prompt,
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                greedy=self.greedy,
                autocast_kwargs=self.autocast_kwargs,
                skip_special_tokens=False,
            ):
                token_text = self.tokenizer.decode([token_id])
                yield token_text


class OpenAIResponder:
    """Adapter for OpenAI hosted chat models."""

    def __init__(self, client: OpenAI, model_name: str, system_prompt: Optional[str], temperature: float = 0.7):
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt.strip() if system_prompt else None
        self.temperature = temperature

    def set_system_prompt(self, system_prompt: Optional[str]) -> None:
        self.system_prompt = system_prompt.strip() if system_prompt else None

    def generate(self, prompt: str) -> str:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=messages,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


def embed_query(client: OpenAI, model_name: str, question: str) -> List[float]:
    response = client.embeddings.create(model=model_name, input=question)
    return response.data[0].embedding


def render_filter(template: Any, context: Dict[str, Any]) -> Any:
    if isinstance(template, dict):
        return {key: render_filter(value, context) for key, value in template.items()}
    if isinstance(template, list):
        return [render_filter(item, context) for item in template]
    if isinstance(template, str):
        try:
            return template.format(**context)
        except KeyError:
            return template
    return template


def retrieve_matches(
    client: Client,
    match_fn: str,
    embedding: List[float],
    match_count: int,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "query_embedding": embedding,
        "match_count": match_count,
    }
    if metadata_filter:
        payload["filter"] = metadata_filter

    response = client.rpc(match_fn, payload).execute()
    return response.data or []


def build_prompt(question: str, matches: List[Dict[str, Any]], template: str, doc_id: str) -> str:
    if matches:
        blocks = []
        for rank, row in enumerate(matches, start=1):
            metadata = row.get("metadata") or {}
            page = metadata.get("page", "?")
            chunk_index = row.get("chunk_index", "?")
            content = row.get("content", "").strip()
            header = f"[Match {rank} | page {page} | chunk {chunk_index}]"
            blocks.append(f"{header}\n{content}")
        context = "\n\n".join(blocks)
    else:
        context = "No relevant context retrieved."

    return template.format(context=context, question=question, doc_id=doc_id).strip()


def print_matches(matches: List[Dict[str, Any]]) -> None:
    if not matches:
        print("(no matches found)")
        return

    for rank, row in enumerate(matches, start=1):
        metadata = row.get("metadata") or {}
        page = metadata.get("page", "?")
        similarity = row.get("similarity")
        sim_str = f"{similarity:.3f}" if isinstance(similarity, (int, float)) else "?"
        preview = textwrap.shorten((row.get("content") or "").replace("\n", " "), width=160)
        print(f"  [{rank}] page {page}  sim={sim_str}  chunk_index={row.get('chunk_index')}")
        print(f"      {preview}")


def validate_vector_store(client: Client, table: str, doc_id: Optional[str]) -> None:
    try:
        _ = client.table(table).select("doc_id").limit(1).execute()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to access Supabase table '{table}': {exc}")

    if doc_id:
        try:
            resp = client.table(table).select("doc_id").eq("doc_id", doc_id).limit(1).execute()
        except Exception:
            return
        if not resp.data:
            print(
                f"⚠️  No rows found for doc_id '{doc_id}' in table '{table}'. Ensure ingestion has been run.",
                file=sys.stderr,
            )


def run_model_mode(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    responder = LocalResponder(cfg, args.weights_dir, args.checkpoint)
    responder.set_system_prompt(None)

    print("Interactive generation ready. Type /exit to quit.\n")
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in EXIT_COMMANDS:
            print("Bye.")
            break

        print("\nOutput> ", end="", flush=True)
        for token in responder.generate_stream(prompt):
            print(token, end="", flush=True)
        print("\n")


def run_agent_mode(args: argparse.Namespace) -> None:
    load_dotenv(find_dotenv(usecwd=True))

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
    )
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not supabase_url or not supabase_key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/ANON_KEY must be set")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY must be set")

    agent_cfg = load_agent_config(args.config)

    embed_cfg = resolve_model_section(agent_cfg.get("embed_model"))
    embed_model_name = embed_cfg.get("name")
    if not embed_model_name:
        raise ValueError("embed_model.name (or model_zoo_id) must be provided in the agent config")

    supabase_cfg = agent_cfg.get("supabase", {})
    table_name = supabase_cfg.get("table", "chunks")
    match_fn = supabase_cfg.get("match_fn", "match_documents")
    filter_template = supabase_cfg.get("filter")

    inference_cfg = agent_cfg.get("inference", {})
    base_doc_id = args.doc_id or inference_cfg.get("doc_id") or agent_cfg.get("doc_id")
    if not base_doc_id:
        config_slug = _slugify(Path(args.config).stem)
        base_doc_id = f"{config_slug}-docs"
    if not base_doc_id:
        raise ValueError("doc_id must be provided either via --doc-id or in the agent config")
    if "{name}" in base_doc_id:
        raise ValueError(
            "doc_id contains placeholder '{name}'. Provide a specific doc_id via --doc-id or agent.inference.doc_id."
        )

    match_count = (
        args.top_k
        or inference_cfg.get("match_count")
        or supabase_cfg.get("match_count")
        or 3
    )

    if filter_template is None:
        filter_template = {"doc_id": "{doc_id}"}

    filter_context: Dict[str, Any] = {"doc_id": base_doc_id}

    source_override = inference_cfg.get("source_path")
    data_source_path = agent_cfg.get("data_source", {}).get("path")

    if source_override:
        filter_context["source_path"] = source_override
    elif data_source_path and Path(data_source_path).is_file():
        filter_context["source_path"] = data_source_path

    if "source_path" in filter_context:
        filter_context.setdefault("source", filter_context["source_path"])

    metadata_filter = render_filter(filter_template, filter_context)

    system_prompt = inference_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    prompt_template = inference_cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)

    supabase_client = create_client(supabase_url, supabase_key)
    openai_client = OpenAI(api_key=openai_key)

    validate_vector_store(supabase_client, table_name, base_doc_id)

    model_section = agent_cfg.get("model", {})
    model_cfg = resolve_model_section(model_section)

    if "model" not in model_cfg:
        config_name = model_section.get("config_name")
        if config_name:
            model_cfg = load_config(config_name)

    if "model" in model_cfg:
        responder: Any = LocalResponder(model_cfg, args.weights_dir, args.checkpoint)
        responder.set_system_prompt(system_prompt)
    elif model_cfg.get("provider") == "openai":
        temperature = float(model_cfg.get("temperature", 0.7))
        model_name = model_cfg.get("name")
        if not model_name:
            raise ValueError("OpenAI model configuration must include a name")
        responder = OpenAIResponder(
            openai_client,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    else:
        raise ValueError("Unsupported model configuration for inference")

    print("Interactive QA ready. Type /exit to quit.\n")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Bye.")
            break

        try:
            embedding = embed_query(openai_client, embed_model_name, question)
            matches = retrieve_matches(
                supabase_client,
                match_fn=match_fn,
                embedding=embedding,
                match_count=match_count,
                metadata_filter=metadata_filter,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error retrieving matches: {exc}", file=sys.stderr)
            continue

        if args.show_context or inference_cfg.get("show_context"):
            print("\nRetrieved matches:")
            print_matches(matches)
            print()

        prompt = build_prompt(question, matches, prompt_template, base_doc_id)

        # Handle streaming for LocalResponder or non-streaming for OpenAI
        if isinstance(responder, LocalResponder):
            print("\nAnswer> ", end="", flush=True)
            for token in responder.generate_stream(prompt):
                print(token, end="", flush=True)
            print("\n")
        else:
            answer = responder.generate(prompt)
            print("\nAnswer> ", answer.strip(), "\n", sep="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified inference CLI")
    parser.add_argument("--config", required=True, help="Path to model or agent YAML configuration")
    parser.add_argument("--checkpoint", help="Optional local checkpoint (.pt) to load weights from")
    parser.add_argument("--weights_dir", help="Directory or file containing converted GPT weights")
    parser.add_argument("--doc-id", dest="doc_id", help="Override doc_id used for retrieval (agent mode)")
    parser.add_argument("--top-k", dest="top_k", type=int, help="Override number of matches to retrieve (agent mode)")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context chunks before answering (agent mode)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    if "agent" in raw_cfg:
        run_agent_mode(args)
    else:
        run_model_mode(args)


if __name__ == "__main__":
    main()
def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    return slug.lower() or "document"
