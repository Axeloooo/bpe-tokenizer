from pathlib import Path

import pytest

from bpe_tokenizer import (
    BasicTokenizer,
    RegexTokenizer,
    Tokenizer,
    get_stats,
    merge,
)

TEXT = (
    "Hello world! Hello, tokenizer. The quick brown fox jumps over the lazy dog. "
    "naïve café 你好 🙂 123456 don't we'll\n\n  indented\n"
) * 20

SAMPLES = [
    "",
    "a",
    "hello world",
    "  leading and trailing  ",
    "naïve café",
    "你好 🙂",
    "x\n\ny\t z",
]


def test_get_stats_and_merge() -> None:
    ids = [1, 2, 3, 1, 2]
    assert get_stats(ids) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert merge(ids, (1, 2), 4) == [4, 3, 4]
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1]  # non-overlapping, left to right


def test_wikipedia_example() -> None:
    tok = BasicTokenizer()
    tok.train("aaabdaaabac", 256 + 3)
    assert tok.merges[(97, 97)] == 256  # "aa" is the most frequent pair
    assert len(tok.merges) == 3
    ids = tok.encode("aaabdaaabac")
    assert len(ids) < len("aaabdaaabac")
    assert tok.decode(ids) == "aaabdaaabac"


@pytest.mark.parametrize("make", [BasicTokenizer, RegexTokenizer])
def test_round_trip(make: type[Tokenizer]) -> None:
    tok = make()
    tok.train(TEXT, 300)
    for s in [TEXT, *SAMPLES]:
        assert tok.decode(tok.encode(s)) == s


@pytest.mark.parametrize("make", [BasicTokenizer, RegexTokenizer])
def test_training_compresses(make: type[Tokenizer]) -> None:
    tok = make()
    tok.train(TEXT, 320)
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8")) * 0.6
    assert len(tok.vocab) == 256 + len(tok.merges)


def test_no_merges_means_raw_bytes() -> None:
    tok = BasicTokenizer()
    tok.train("abc", 256)
    assert tok.encode("héllo") == list("héllo".encode())


def test_vocab_size_below_256_rejected() -> None:
    with pytest.raises(ValueError):
        BasicTokenizer().train("abc", 100)


def test_training_stops_when_no_pairs_left() -> None:
    tok = BasicTokenizer()
    tok.train("ab", 1000)
    assert len(tok.merges) == 1


def test_regex_never_merges_across_chunks() -> None:
    tok = RegexTokenizer()
    tok.train(TEXT, 400)
    for token_bytes in tok.vocab.values():
        text = token_bytes.decode("utf-8", errors="ignore")
        # letters and digits are split apart by the GPT-4 pattern
        assert not (any(c.isalpha() for c in text) and any(c.isdigit() for c in text))


def test_special_tokens() -> None:
    tok = RegexTokenizer()
    tok.train(TEXT, 300)
    tok.register_special_tokens({"<|endoftext|>": 1000})
    text = "hello<|endoftext|>world"
    with pytest.raises(ValueError):
        tok.encode(text)
    ids = tok.encode(text, allowed_special="all")
    assert ids.count(1000) == 1
    assert tok.decode(ids) == text
    plain = tok.encode(text, allowed_special="none")
    assert 1000 not in plain and tok.decode(plain) == text
    assert tok.encode(text, allowed_special={"<|endoftext|>"}) == ids
    with pytest.raises(ValueError):
        tok.decode([999999])


def test_deterministic_training() -> None:
    a, b = RegexTokenizer(), RegexTokenizer()
    a.train(TEXT, 310)
    b.train(TEXT, 310)
    assert a.merges == b.merges


@pytest.mark.parametrize("make", [BasicTokenizer, RegexTokenizer])
def test_save_and_load(make: type[Tokenizer], tmp_path: Path) -> None:
    tok = make()
    tok.train(TEXT, 300)
    if isinstance(tok, RegexTokenizer):
        tok.register_special_tokens({"<|endoftext|>": 5000})
    path = tmp_path / "tok.json"
    tok.save(path)
    other = make()
    other.load(path)
    assert other.merges == tok.merges
    assert other.encode(TEXT) == tok.encode(TEXT)
    assert other.special_tokens == tok.special_tokens
