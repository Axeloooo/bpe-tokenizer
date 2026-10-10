from pathlib import Path

import pytest

from bpe_tokenizer.__main__ import main

TEXT = "Hello world! Hello, tokenizer. naïve café 你好 🙂\n" * 30


@pytest.mark.parametrize("kind", ["regex", "basic"])
def test_train_encode_decode(kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus, model = tmp_path / "corpus.txt", tmp_path / "tok.json"
    corpus.write_text(TEXT, encoding="utf-8")
    main(["train", str(corpus), "--vocab-size", "300", "--kind", kind, "--out", str(model)])
    assert "compression" in capsys.readouterr().out

    main(["encode", "--model", str(model), "Hello café 🙂"])
    ids = capsys.readouterr().out.split()
    main(["decode", "--model", str(model), *ids])
    assert capsys.readouterr().out.strip() == "Hello café 🙂"
