"""Command line: train a tokenizer on a text file, then encode and decode with it."""

from __future__ import annotations

import argparse
import json

from bpe_tokenizer.base import Tokenizer
from bpe_tokenizer.basic import BasicTokenizer
from bpe_tokenizer.regex_tokenizer import RegexTokenizer


def _load(path: str) -> Tokenizer:
    """A saved tokenizer with no split pattern was trained by BasicTokenizer."""
    with open(path, encoding="utf-8") as f:
        has_pattern = bool(json.load(f)["pattern"])
    tok: Tokenizer = RegexTokenizer() if has_pattern else BasicTokenizer()
    tok.load(path)
    return tok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bpe_tokenizer", description="Byte pair encoding tokenizer"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train on a text file and save the tokenizer as JSON")
    train.add_argument("text_file")
    train.add_argument("--vocab-size", type=int, default=512)
    train.add_argument("--kind", choices=["regex", "basic"], default="regex")
    train.add_argument("--out", default="tokenizer.json")
    train.add_argument("--verbose", action="store_true")

    encode = sub.add_parser("encode", help="print the token ids of a string")
    encode.add_argument("--model", required=True)
    encode.add_argument("text")

    decode = sub.add_parser("decode", help="turn token ids back into text")
    decode.add_argument("--model", required=True)
    decode.add_argument("ids", type=int, nargs="+")

    args = parser.parse_args(argv)
    if args.command == "train":
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
        tok: Tokenizer = BasicTokenizer() if args.kind == "basic" else RegexTokenizer()
        tok.train(text, args.vocab_size, verbose=args.verbose)
        tok.save(args.out)
        ratio = len(text.encode("utf-8")) / max(1, len(tok.encode(text)))
        print(f"saved {args.out}: {len(tok.merges)} merges, compression {ratio:.2f}x")
    elif args.command == "encode":
        tok = _load(args.model)
        ids = (
            tok.encode(args.text, allowed_special="all")
            if isinstance(tok, RegexTokenizer)
            else tok.encode(args.text)
        )
        print(*ids)
    else:
        print(_load(args.model).decode(args.ids))


if __name__ == "__main__":
    main()
