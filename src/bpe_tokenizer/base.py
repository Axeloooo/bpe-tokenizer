"""Shared pieces: pair counting, merging, and the Tokenizer base class with save/load."""

from __future__ import annotations

import json
from pathlib import Path


def get_stats(
    ids: list[int], counts: dict[tuple[int, int], int] | None = None
) -> dict[tuple[int, int], int]:
    """Count how often each consecutive pair of ids occurs."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:], strict=False):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with the new token `idx`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    """Base class: a tokenizer is its merges (pair -> new id), a split pattern, special tokens."""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.pattern = ""
        self.special_tokens: dict[str, int] = {}
        self.vocab = self._build_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, path: str | Path) -> None:
        """Write merges, pattern and special tokens as JSON (merge order is the token order)."""
        data = {
            "version": 1,
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "merges": [[p0, p1] for (p0, p1) in self.merges],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        if data.get("version") != 1:
            raise ValueError("unsupported tokenizer file version")
        self.pattern = data["pattern"]
        self.special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        self.merges = {(p0, p1): 256 + i for i, (p0, p1) in enumerate(data["merges"])}
        self.vocab = self._build_vocab()
