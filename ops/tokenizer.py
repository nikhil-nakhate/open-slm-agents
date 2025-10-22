from typing import List, Dict, Any, Optional
import os
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




def build_tokenizer(cfg: Dict[str, Any]) -> BaseTokenizer:
    kind = cfg.get("kind", "tiktoken")
    params = cfg.get("params", {})

    if kind == "simple_char":
        return SimpleCharTokenizer(**params)
    elif kind == "o200k_harmony":
        return HarmonyTokenizer()
    elif kind == "tiktoken":
        # Always uses gpt2 encoding
        enc = tiktoken.get_encoding("gpt2")
        return TiktokenTokenizer(enc)
    else:
        raise ValueError(f"Unknown tokenizer kind: {kind}")




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


class HarmonyTokenizer(BaseTokenizer):
    """Harmony tokenizer with extended special tokens.

    Based on the o200k_base encoding with additional special tokens for
    structured text generation and control.
    """

    def __init__(self):
        # Get the base o200k encoding
        o200k_base = tiktoken.get_encoding("o200k_base")

        # Create the harmony encoding with extended special tokens
        self.encoding = tiktoken.Encoding(
            name="o200k_harmony",
            pat_str=o200k_base._pat_str,
            mergeable_ranks=o200k_base._mergeable_ranks,
            special_tokens={
                **o200k_base._special_tokens,
                "<|startoftext|>": 199998,
                "<|endoftext|>": 199999,
                "<|reserved_200000|>": 200000,
                "<|reserved_200001|>": 200001,
                "<|return|>": 200002,
                "<|constrain|>": 200003,
                "<|reserved_200004|>": 200004,
                "<|channel|>": 200005,
                "<|start|>": 200006,
                "<|end|>": 200007,
                "<|message|>": 200008,
                "<|reserved_200009|>": 200009,
                "<|reserved_200010|>": 200010,
                "<|reserved_200011|>": 200011,
                "<|call|>": 200012,
            } | {
                f"<|reserved_{i}|>": i for i in range(200013, 201088)
            },
        )

        # Compute eos id via special token
        self._eos = self.encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def encode(self, text: str) -> List[int]:
        # Allow all special tokens during encoding
        return self.encoding.encode(text, allowed_special="all")

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
