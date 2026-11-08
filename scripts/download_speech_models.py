#!/usr/bin/env python3
"""Download and verify the exact speech-model revisions used in production."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelFile:
    url: str
    destination: Path
    sha256: str


FILES = (
    ModelFile(
        "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/"
        "ad9dd8518bedf517bd7cbc9f63b8e5c844bf5bc0/fa_IR-mana-medium.onnx",
        ROOT / "TTS/models/mana/fa_IR-mana-medium.onnx",
        "e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126",
    ),
    ModelFile(
        "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/"
        "ad9dd8518bedf517bd7cbc9f63b8e5c844bf5bc0/fa_IR-mana-medium.onnx.json",
        ROOT / "TTS/models/mana/fa_IR-mana-medium.onnx.json",
        "d102276a05436613a05e1fbad8f125abcf8f56759aaf4fd36adf216653949997",
    ),
    ModelFile(
        "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/"
        "ea66dc9353e660b1dbbe0bebebd4c2d8b105483c/converter/checkpoint.pth",
        ROOT / "VC/checkpoints/openvoice-v2/converter/checkpoint.pth",
        "9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e",
    ),
    ModelFile(
        "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/"
        "ea66dc9353e660b1dbbe0bebebd4c2d8b105483c/converter/config.json",
        ROOT / "VC/checkpoints/openvoice-v2/converter/config.json",
        "9dfff60350b8c63f2c664efd92a61b2516efb22671466960f0e5dfebd881fa47",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_file: ModelFile) -> bool:
    return model_file.destination.is_file() and sha256(model_file.destination) == model_file.sha256


def download(model_file: ModelFile) -> None:
    model_file.destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        model_file.url,
        headers={"User-Agent": "MetaRang-speech-model-installer/1"},
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{model_file.destination.name}.",
            dir=model_file.destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(request, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        actual = sha256(temporary_path)
        if actual != model_file.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {model_file.destination.name}: {actual}"
            )
        os.replace(temporary_path, model_file.destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; fail if any exact model file is absent or invalid.",
    )
    args = parser.parse_args()

    failed = False
    for model_file in FILES:
        relative = model_file.destination.relative_to(ROOT)
        if verify(model_file):
            print(f"OK   {relative}")
            continue
        if args.verify_only:
            print(f"FAIL {relative}", file=sys.stderr)
            failed = True
            continue
        print(f"GET  {relative}")
        download(model_file)
        print(f"OK   {relative}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
