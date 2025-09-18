from typing import List, Dict, Any, Optional
import os
import re
import tiktoken


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
        # Reserve the last index for padding
        return self.vocab_size - 1

    @property
    def eos_id(self) -> Optional[int]:
        # Default: unknown
        return None


class SimpleCharTokenizer(BaseTokenizer):
    """A minimal character-level tokenizer as a safe default.

    - Builds vocabulary from provided `vocab_chars` or falls back to printable ASCII.
    - Reserves the last id (vocab_size - 1) as padding.
    """

    def __init__(self, vocab_chars: str = None):
        import string

        if vocab_chars is None:
            # Basic printable set (excluding control chars)
            vocab_chars = string.printable
        self.itos = list(dict.fromkeys(vocab_chars))
        # 0-based token ids for characters; last id is reserved for PAD
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, text: str) -> List[int]:
        unk = self.stoi.get("?", 0)
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: List[int]) -> str:
        out = []
        pad = self.pad_id
        for i in ids:
            if i == pad:
                continue
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
            else:
                out.append("?")
        return "".join(out)

    @property
    def vocab_size(self) -> int:
        # +1 slot reserved for PAD at the end
        return len(self.itos) + 1

    @property
    def eos_id(self) -> Optional[int]:
        return None


class RegexWordTokenizer(BaseTokenizer):
    """A simple regex-based word tokenizer with a small dynamic vocab.

    Not suitable for large corpora, but OK for quick tests.
    """

    def __init__(self):
        # Do not bake PAD into vocab; reserve last index for PAD
        self.stoi = {"<unk>": 0}
        self.itos = ["<unk>"]
        self._word_re = re.compile(r"\w+|[^\w\s]")

    def encode(self, text: str) -> List[int]:
        ids = []
        for tok in self._word_re.findall(text):
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
            ids.append(self.stoi.get(tok, 0))
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        pad = self.pad_id
        for i in ids:
            if i == pad:
                continue
            toks.append(self.itos[i] if 0 <= i < len(self.itos) else "<unk>")
        return " ".join(toks).strip()

    @property
    def vocab_size(self) -> int:
        # +1 slot reserved for PAD at the end
        return len(self.itos) + 1


def build_tokenizer(cfg: Dict[str, Any]) -> BaseTokenizer:
    kind = cfg.get("kind", "tiktoken")
    params = cfg.get("params", {})

    if kind == "simple_char":
        return SimpleCharTokenizer(**params)
    elif kind == "regex_word":
        return RegexWordTokenizer()
    elif kind in {"tiktoken", "tiktoken_gpt2"}:
        name = params.get("name", "gpt2")
        enc = tiktoken.get_encoding(name)
        return TiktokenTokenizer(enc)
    elif kind in {"hf_gpt2", "huggingface"}:
        from transformers import GPT2TokenizerFast  # type: ignore
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
        # Always reserve last id for padding
        return self.tok.vocab_size - 1

    @property
    def eos_id(self) -> Optional[int]:
        return getattr(self.tok, "eos_token_id", None)


class TiktokenTokenizer(BaseTokenizer):
    """Wrapper around tiktoken encodings (default: GPT-2)."""

    def __init__(self, encoding):
        self.encoding = encoding
        # Compute eos id via special token
        self._eos = self.encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: List[int]) -> str:
        # Skip pad ids (reserved as the last vocab index)
        pad = self.pad_id
        filtered = [i for i in ids if i != pad]
        return self.encoding.decode(filtered)

    @property
    def vocab_size(self) -> int:
        return int(self.encoding.n_vocab)

    @property
    def eos_id(self) -> Optional[int]:
        return self._eos
