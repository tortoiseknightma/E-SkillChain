from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from skillchain.tools.embedding import OpenCLIPEmbeddingBackend
from skillchain.tools.model_artifacts import load_model_artifact_manifest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_openclip_artifact",
    ROOT / "scripts" / "prepare_openclip_artifact.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prepare_openclip_artifact_locks_all_local_tokenizer_files(
    tmp_path: Path,
    monkeypatch,
):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    fixture_files = {}
    for role, (_repo, _revision, filename, _digest) in MODULE._FILES.items():
        path = downloads / f"{role}-{filename}"
        path.write_bytes(f"fixture:{role}".encode())
        fixture_files[role] = path
    monkeypatch.setattr(
        MODULE,
        "_FILES",
        {
            role: (
                repo,
                revision,
                filename,
                hashlib.sha256(fixture_files[role].read_bytes()).hexdigest(),
            )
            for role, (repo, revision, filename, _digest) in MODULE._FILES.items()
        },
    )
    monkeypatch.setattr(
        MODULE.metadata,
        "version",
        lambda name: {
            "open_clip_torch": "3.3.0",
            "torch": "2.13.0",
            "transformers": "5.14.1",
        }[name],
    )

    def download_file(*, filename, **_kwargs):
        for role, path in fixture_files.items():
            if MODULE._FILES[role][2] == filename:
                return str(path)
        raise AssertionError(filename)

    destination = tmp_path / "embedding"
    result = MODULE.prepare_openclip_artifact(
        destination,
        verify_runtime=False,
        download_file=download_file,
    )

    verified = load_model_artifact_manifest(
        destination / "manifest.json",
        expected_kind="multimodal_embedding",
        expected_manifest_sha256=result["manifest_sha256"],
        verify_files=True,
    )
    backend = OpenCLIPEmbeddingBackend(verified)
    assert backend.model == MODULE._MODEL_ID
    assert {item.role for item in verified.manifest.artifacts}.issuperset(
        {
            "config",
            "tokenizer",
            "tokenizer_config",
            "tokenizer_model",
            "transformer_config",
            "weights",
        }
    )
