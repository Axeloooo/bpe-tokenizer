"""Minimal BPE tokenizer working directly on the UTF-8 bytes of the text."""

from __future__ import annotations

from bpe_tokenizer.base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=lambda p: stats[p])
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            if verbose:
                print(f"merge {i + 1}/{vocab_size - 256}: {pair} -> {idx} ({stats[pair]} times)")
        self.merges = merges
        self.vocab = self._build_vocab()

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Apply the earliest-learned merge available (lowest new id).
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
