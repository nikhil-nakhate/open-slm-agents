"""Interactive inference CLI supporting both model-only and agent-based workflows."""

from __future__ import annotations

import argparse
import os
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


def _generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    greedy: bool = False,
) -> str:
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt)
        ids = list(input_ids)
        eos_id: Optional[int] = getattr(tokenizer, "eos_id", None)
        max_ctx = int(getattr(model, "max_seq_len", 1024))

        x = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            x_cond = x[:, -max_ctx:]
            logits = model(x_cond)
            next_logits = logits[0, -1, :]
            next_token = _sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
            )
            ids.append(next_token)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            if eos_id is not None and next_token == eos_id:
                break

        continuation = ids[len(input_ids):]
        return tokenizer.decode(continuation)


class LocalResponder:
    """Wrapper around a locally hosted model for text generation."""

    def __init__(self, model_cfg: Dict[str, Any], weights_dir: Optional[str], checkpoint: Optional[str]):
        self.cfg = dict(model_cfg)
        if "model" not in self.cfg:
            raise ValueError("Local responder requires a full model config under 'model'.")

        self.model = build_model_from_cfg(self.cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None:
            raise AttributeError("Model must expose a tokenizer for inference.")

        self._load_weights(weights_dir, checkpoint)
        self.model.to(self.device)
        self.model.eval()

        eval_cfg = self.cfg.get("eval", {})
        self.max_new_tokens = int(eval_cfg.get("max_new_tokens", 200))
        self.temperature = float(eval_cfg.get("temperature", 1.0))
        self.top_k = int(eval_cfg.get("top_k", 0))
        self.top_p = float(eval_cfg.get("top_p", 0.0))
        self.greedy = bool(eval_cfg.get("greedy", False))
        self.system_prompt: Optional[str] = None

    def _load_weights(self, weights_dir: Optional[str], checkpoint: Optional[str]) -> None:
        if weights_dir:
            path = Path(weights_dir)
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

    def set_system_prompt(self, system_prompt: Optional[str]) -> None:
        self.system_prompt = system_prompt.strip() if system_prompt else None

    def generate(self, prompt: str) -> str:
        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}".strip()
        return _generate(
            self.model,
            self.tokenizer,
            prompt=full_prompt,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            greedy=self.greedy,
        )


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

        output = responder.generate(prompt)
        print("\nOutput> ", output.strip(), "\n", sep="")


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
