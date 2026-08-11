"""正式种子集的 staging、人工接受与拒绝状态机。"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from skillchain.synthesis.models import SeedDraft, SeedManifest
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    canonical_json_bytes,
    new_staging_directory,
    sha256_bytes,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def stage_seed_batch(
    draft_path: str | Path, queries_root: str | Path, seed_batch_id: str
) -> Path:
    """验证由已确认模型产生的草稿，并只发布到 seed staging。"""

    seed_batch_id = _validate_id(seed_batch_id)
    draft = _load_seed_draft(Path(draft_path))
    queries_root = Path(queries_root)
    destination = queries_root / "seeds" / "staging" / seed_batch_id
    staging = new_staging_directory(destination)
    try:
        seed_bytes = canonical_json_bytes({"examples": draft.examples})
        manifest = SeedManifest(
            seed_batch_id=seed_batch_id,
            provider=draft.provider,
            model_display_name=draft.model_display_name,
            model_claim_source=draft.model_claim_source,
            generated_at=draft.generated_at,
            staged_at=datetime.now(timezone.utc),
            seed_set_sha256=sha256_bytes(seed_bytes),
        )
        atomic_create_file(staging / "seed_examples.json", seed_bytes)
        atomic_create_file(staging / "manifest.json", canonical_json_bytes(manifest))
        return atomic_publish_new_directory(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def accept_seed_batch(
    queries_root: str | Path, seed_batch_id: str, *, confirmation: str
) -> Path:
    if confirmation != "ACCEPT":
        raise ValueError("接受种子集必须提供字面确认 ACCEPT")
    seed_batch_id = _validate_id(seed_batch_id)
    queries_root = Path(queries_root)
    staged = queries_root / "seeds" / "staging" / seed_batch_id
    accepted = queries_root / "seeds" / "accepted"
    if accepted.is_dir():
        manifest = _load_manifest(accepted / "manifest.json")
        if manifest.seed_batch_id == seed_batch_id:
            _verify_seed_directory(accepted, manifest)
            return accepted
        raise FileExistsError("已有 accepted 种子集，拒绝覆盖")
    if not staged.is_dir():
        raise FileNotFoundError(f"seed staging 不存在: {staged}")
    manifest = _load_manifest(staged / "manifest.json")
    if manifest.seed_batch_id != seed_batch_id:
        raise ValueError("seed manifest batch id 与目录不一致")
    _verify_seed_directory(staged, manifest)
    return atomic_publish_new_directory(staged, accepted)


def reject_seed_batch(
    queries_root: str | Path, seed_batch_id: str, *, reason: str
) -> Path:
    seed_batch_id = _validate_id(seed_batch_id)
    reason = reason.strip()
    if not reason:
        raise ValueError("拒绝理由不得为空")
    queries_root = Path(queries_root)
    staged = queries_root / "seeds" / "staging" / seed_batch_id
    if not staged.is_dir():
        raise FileNotFoundError(f"seed staging 不存在: {staged}")
    destination = queries_root / "seeds" / "rejected" / seed_batch_id
    atomic_create_file(staged / "reason.json", canonical_json_bytes({"reason": reason}))
    return atomic_publish_new_directory(staged, destination)


def load_accepted_seed_manifest(queries_root: str | Path) -> SeedManifest:
    accepted = Path(queries_root) / "seeds" / "accepted"
    manifest = _load_manifest(accepted / "manifest.json")
    _verify_seed_directory(accepted, manifest)
    return manifest


def list_staged_seed_manifests(
    queries_root: str | Path,
) -> tuple[SeedManifest, ...]:
    """Return every hash-verified staged seed set in stable ID order."""

    staging_root = Path(queries_root) / "seeds" / "staging"
    if not staging_root.exists():
        return ()
    if staging_root.is_symlink() or not staging_root.is_dir():
        raise ValueError("seed staging root 必须是普通目录")

    manifests: list[SeedManifest] = []
    for staged in sorted(staging_root.iterdir(), key=lambda path: path.name):
        if staged.is_symlink() or not staged.is_dir():
            raise ValueError(f"seed staging entry 必须是普通目录: {staged.name}")
        manifest = _load_manifest(staged / "manifest.json")
        if manifest.seed_batch_id != staged.name:
            raise ValueError("seed manifest batch id 与 staging 目录不一致")
        _verify_seed_directory(staged, manifest)
        manifests.append(manifest)
    return tuple(manifests)


def _load_seed_draft(path: Path) -> SeedDraft:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"seed draft 不存在: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"seed draft 不是有效 JSON: {exc.msg}") from exc
    try:
        return SeedDraft.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"seed draft 校验失败: {exc}") from exc


def _load_manifest(path: Path) -> SeedManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SeedManifest.model_validate(raw)
    except FileNotFoundError:
        raise FileNotFoundError(f"seed manifest 不存在: {path}") from None
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"seed manifest 校验失败: {exc}") from exc


def _verify_seed_directory(directory: Path, manifest: SeedManifest) -> None:
    seed_path = directory / "seed_examples.json"
    try:
        actual = sha256_bytes(seed_path.read_bytes())
    except FileNotFoundError:
        raise FileNotFoundError(f"seed 文件不存在: {seed_path}") from None
    if actual != manifest.seed_set_sha256:
        raise ValueError("seed_set_sha256 哈希不一致")


def _validate_id(value: str) -> str:
    value = value.strip()
    if not _SAFE_ID.fullmatch(value):
        raise ValueError("seed_batch_id 只能包含字母、数字、点、下划线和连字符")
    return value
