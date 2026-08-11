"""Download, lock, and optionally load-check the canonical local embedding.

This downloads model artifacts, not project RAW data.  Every Hub revision and
file digest is fixed below; the resulting model manifest is still a candidate
until ranking gold and resource measurements approve it.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Callable, Sequence

from huggingface_hub import hf_hub_download

from skillchain.tools.embedding import OpenCLIPEmbeddingBackend
from skillchain.tools.model_artifacts import (
    load_model_artifact_manifest,
    publish_model_artifact_manifest,
)
from skillchain.tools.serialization import canonical_json_bytes


_MODEL_REPO = "laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k"
_MODEL_REVISION = "f9ab955287d06ac45c7654f26fecd29301c70a1f"
_TOKENIZER_REPO = "FacebookAI/xlm-roberta-large"
_TOKENIZER_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
_MODEL_ID = "open-clip:xlm-roberta-large-ViT-H-14:frozen_laion5b_s13b_b90k"
_FILES = {
    "weights": (
        _MODEL_REPO,
        _MODEL_REVISION,
        "open_clip_pytorch_model.bin",
        "ee81ce53c8591d48ee5e7b390b2f10d502c721df5920c3096c6bdb54d1459881",
    ),
    "transformer_config": (
        _TOKENIZER_REPO,
        _TOKENIZER_REVISION,
        "config.json",
        "ec7c3a99c58a38ebb702a2297f1c4586944e841b5eb29d201ff87e0f9c9abe0e",
    ),
    "tokenizer_model": (
        _TOKENIZER_REPO,
        _TOKENIZER_REVISION,
        "sentencepiece.bpe.model",
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
    "tokenizer": (
        _TOKENIZER_REPO,
        _TOKENIZER_REVISION,
        "tokenizer.json",
        "a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c",
    ),
    "tokenizer_config": (
        _TOKENIZER_REPO,
        _TOKENIZER_REVISION,
        "tokenizer_config.json",
        "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce",
    ),
}


def prepare_openclip_artifact(
    destination: Path,
    *,
    cache_dir: Path | None = None,
    verify_runtime: bool = True,
    download_file: Callable[..., str] = hf_hub_download,
) -> dict[str, object]:
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"OpenCLIP artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        artifacts: dict[str, Path] = {}
        tokenizer_root = staging / "tokenizer"
        tokenizer_root.mkdir()
        for role, (repo, revision, filename, expected_sha256) in _FILES.items():
            source = Path(
                download_file(
                    repo_id=repo,
                    revision=revision,
                    filename=filename,
                    cache_dir=None if cache_dir is None else str(cache_dir),
                )
            ).resolve()
            if _digest_regular_file(source) != expected_sha256:
                raise ValueError(f"OpenCLIP source digest mismatch: {role}")
            target = (
                staging / "open_clip_pytorch_model.bin"
                if role == "weights"
                else tokenizer_root / filename
            )
            shutil.copyfile(source, target)
            artifacts[role] = target
        config = staging / "openclip-runtime.json"
        config.write_bytes(
            canonical_json_bytes(
                {
                    "device": "cpu",
                    "image_interpolation": "bicubic",
                    "image_mean": [0.48145466, 0.4578275, 0.40821073],
                    "image_resize_mode": "shortest",
                    "image_std": [0.26862954, 0.26130258, 0.27577711],
                    "model_name": "xlm-roberta-large-ViT-H-14",
                    "output_dimension": 1024,
                    "precision": "fp32",
                    "pretrained": "frozen_laion5b_s13b_b90k",
                    "schema_version": 1,
                    "tokenizer_kind": "hf_local",
                    "torch_version": metadata.version("torch"),
                    "transformers_version": metadata.version("transformers"),
                }
            )
        )
        artifacts["config"] = config
        published = publish_model_artifact_manifest(
            staging / "manifest.json",
            artifact_kind="multimodal_embedding",
            model_id=_MODEL_ID,
            backend_name="open-clip-torch",
            backend_version=metadata.version("open_clip_torch"),
            artifacts=artifacts,
        )
        manifest_sha256 = published.manifest.manifest_sha256
        os.replace(staging, destination)
        verified = load_model_artifact_manifest(
            destination / "manifest.json",
            expected_kind="multimodal_embedding",
            expected_manifest_sha256=manifest_sha256,
            verify_files=True,
        )
        runtime_sha256 = None
        if verify_runtime:
            runtime_sha256 = OpenCLIPEmbeddingBackend(
                verified
            ).formal_runtime_binding_sha256
        return {
            "artifact_kind": "multimodal_embedding",
            "manifest_path": str(destination / "manifest.json"),
            "manifest_sha256": manifest_sha256,
            "model_id": _MODEL_ID,
            "runtime_binding_sha256": runtime_sha256,
            "runtime_loaded": verify_runtime,
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if destination.exists():
            shutil.rmtree(destination)
        raise


def _digest_regular_file(path: Path) -> str:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"downloaded model is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        opened = os.fstat(source.fileno())
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        after_read = os.fstat(source.fileno())
    after = path.lstat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if (
        identity
        != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        )
        or identity
        != (
            after_read.st_dev,
            after_read.st_ino,
            after_read.st_size,
            after_read.st_mtime_ns,
        )
        or identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
    ):
        raise ValueError(f"downloaded model changed while hashing: {path}")
    return digest.hexdigest()


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--skip-runtime-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        result = prepare_openclip_artifact(
            arguments.destination,
            cache_dir=arguments.cache_dir,
            verify_runtime=not arguments.skip_runtime_check,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"prepare-openclip-artifact: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
