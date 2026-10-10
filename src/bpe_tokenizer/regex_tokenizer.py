"""BPE with a regex pre-split (the GPT-2/GPT-4 recipe) and special tokens."""

from __future__ import annotations

from pathlib import Path

import regex as re

from bpe_tokenizer.base import Tokenizer, get_stats, merge

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str = GPT4_SPLIT_PATTERN) -> None:
        super().__init__()
        self.pattern = pattern
        self._compiled = re.compile(pattern)
        self._inverse_special: dict[int, str] = {}

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        self.special_tokens = dict(special_tokens)
        self._inverse_special = {v: k for k, v in special_tokens.items()}
        self.vocab = self._build_vocab()

    def load(self, path: str | Path) -> None:
        super().load(path)
        self._compiled = re.compile(self.pattern)
        self._inverse_special = {v: k for k, v in self.special_tokens.items()}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        chunks = [list(ch.encode("utf-8")) for ch in re.findall(self._compiled, text)]
        merges: dict[tuple[int, int], int] = {}
        for i in range(vocab_size - 256):
            stats: dict[tuple[int, int], int] = {}
            for chunk in chunks:
                get_stats(chunk, stats)
            if not stats:
                break
            pair = max(stats, key=lambda p: stats[p])
            idx = 256 + i
            chunks = [merge(chunk, pair, idx) for chunk in chunks]
            merges[pair] = idx
            if verbose:
                print(f"merge {i + 1}/{vocab_size - 256}: {pair} -> {idx} ({stats[pair]} times)")
        self.merges = merges
        self.vocab = self._build_vocab()

    def decode(self, ids: list[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self._inverse_special:
                parts.append(self._inverse_special[i].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode ignoring special tokens (they are treated as normal text)."""
        ids: list[int] = []
        for chunk in re.findall(self._compiled, text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str | set[str] = "none_raise") -> list[int]:
        """Encode text. `allowed_special` is "all", "none", "none_raise" or a set of tokens."""
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            if any(token in text for token in self.special_tokens):
                raise ValueError("text contains a special token but allowed_special='none_raise'")
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special!r} not understood")
        if not special:
            return self.encode_ordinary(text)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        ids: list[int] = []
        for part in re.split(special_pattern, text):
            if part in special:
                ids.append(special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids
