# bpe-tokenizer

A byte pair encoding (BPE) tokenizer, built from scratch in the style of GPT-4. It follows Andrej
Karpathy's lecture [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE),
part of the Neural Networks: Zero to Hero series.

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

The only runtime dependency is [`regex`](https://pypi.org/project/regex/), needed for the Unicode
classes (`\p{L}`, `\p{N}`) in the GPT-4 split pattern.

## What is here

| Lecture step                                      | Code                                                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Pair counting and merging                         | `get_stats`, `merge` in `base.py`                                                    |
| Minimal BPE on UTF-8 bytes: train, encode, decode | `BasicTokenizer` in `basic.py`                                                       |
| Regex pre-split so merges never cross word edges  | `RegexTokenizer`, `GPT2_SPLIT_PATTERN`, `GPT4_SPLIT_PATTERN` in `regex_tokenizer.py` |
| Special tokens such as `<\|endoftext\|>`          | `register_special_tokens`, `encode(allowed_special=...)`                             |
| Saving and loading a trained tokenizer            | `Tokenizer.save`, `Tokenizer.load` (JSON)                                            |

## Usage

```python
from bpe_tokenizer import RegexTokenizer

tok = RegexTokenizer()                       # GPT-4 split pattern
tok.train(open("corpus.txt").read(), vocab_size=512)
tok.register_special_tokens({"<|endoftext|>": 512})

ids = tok.encode("hello world<|endoftext|>", allowed_special="all")
assert tok.decode(ids) == "hello world<|endoftext|>"

tok.save("tokenizer.json")
```

`allowed_special` is `"all"`, `"none"`, `"none_raise"` (the default: error if the text contains a
special token) or a set of token strings.

## Not included

The lecture also loads OpenAI's pretrained `cl100k_base` merges to match `tiktoken` exactly, and
covers SentencePiece. Those are left out here: this repository trains its own merges and has no
`tiktoken` dependency.

## Tests

```
pytest -q
```

They cover the lecture's `aaabdaaabac` example, round trips on ASCII, accents, CJK and emoji, that
training compresses, that the regex tokenizer never merges across chunks, special tokens, training
determinism and save/load.

## Development

See `CLAUDE.md`. Commits follow Conventional Commits; work lands on `devel` through pull requests.

## License

MIT
