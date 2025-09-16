from typing import List, Dict, Any, Optional
import os
import re


class BaseTokenizer:
    def encode(self, text: str) -> List[int]:  # noqa: D401
        """Encodes a string into token ids."""
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:  # noqa: D401
        """Decodes token ids into a string."""
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def pad_id(self) -> int:
        # Default pad id
        return 0

    @property
    def eos_id(self) -> Optional[int]:
        # Default: unknown
        return None


class SimpleCharTokenizer(BaseTokenizer):
    """A minimal character-level tokenizer as a safe default.

    - Builds vocabulary from provided `vocab_chars` or falls back to printable ASCII.
    - Uses id 0 as padding if needed.
    """

    def __init__(self, vocab_chars: str = None):
        import string

        if vocab_chars is None:
            # Basic printable set (excluding control chars)
            vocab_chars = string.printable
        self.itos = list(dict.fromkeys(vocab_chars))
        # Reserve 0 for padding; start tokens at 1
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.itos)}
        self._pad_id = 0

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.stoi.get("?", 1)) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i - 1] if i > 0 and i - 1 < len(self.itos) else "?" for i in ids)

    @property
    def vocab_size(self) -> int:
        # Plus one for pad id 0
        return len(self.itos) + 1

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def eos_id(self) -> Optional[int]:
        return None


class RegexWordTokenizer(BaseTokenizer):
    """A simple regex-based word tokenizer with a small dynamic vocab.

    Not suitable for large corpora, but OK for quick tests.
    """

    def __init__(self):
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]
        self._word_re = re.compile(r"\w+|[^\w\s]")

    def encode(self, text: str) -> List[int]:
        ids = []
        for tok in self._word_re.findall(text):
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
            ids.append(self.stoi.get(tok, 1))
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]
        return " ".join(toks).replace(" <pad>", "").strip()

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


def build_tokenizer(cfg: Dict[str, Any]) -> BaseTokenizer:
    kind = cfg.get("kind", "simple_char")
    params = cfg.get("params", {})

    if kind == "simple_char":
        return SimpleCharTokenizer(**params)
    elif kind == "regex_word":
        return RegexWordTokenizer()
    elif kind in {"hf_gpt2", "huggingface"}:
        # Lazy import to avoid hard dependency
        try:
            from transformers import GPT2TokenizerFast  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("Install transformers to use hf_gpt2 tokenizer") from e
        name = params.get("name", "gpt2")
        add_prefix_space = params.get("add_prefix_space", True)
        tok = GPT2TokenizerFast.from_pretrained(name, add_prefix_space=add_prefix_space)
        return HFTokenizer(tok)

    else:
        raise ValueError(f"Unknown tokenizer kind: {kind}")


class HFTokenizer(BaseTokenizer):
    """Wraps a HuggingFace tokenizer to match BaseTokenizer interface."""

    def __init__(self, tok):
        self.tok = tok

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    @property
    def pad_id(self) -> int:
        return getattr(self.tok, "pad_token_id", 0) or 0

    @property
    def eos_id(self) -> Optional[int]:
        return getattr(self.tok, "eos_token_id", None)
