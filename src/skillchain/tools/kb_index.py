"""Build and load deterministic, catalog-bound Chinese BM25 indices."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Annotated, Any, Literal, Self
import unicodedata

import bm25s
import jieba
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.data.kb_catalog import (
    ArtifactDescriptor,
    KBCatalog,
    KBEntryV2,
    KBKind,
    artifact_descriptor,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_kb_catalog,
    parse_json_object,
    read_regular_file_snapshot,
    sha256_bytes,
    verify_catalog_unchanged,
    verify_file_snapshot,
)
from skillchain.synthesis.store import atomic_publish_new_directory

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
KB_INDEX_SCHEMA_VERSION = 2
KB_INDEX_POLICY_VERSION = "kb-bm25-index-v2"
KB_TOKENIZER_POLICY_VERSION = "jieba-nfc-exact-no-hmm-v1"
KB_RANKING_POLICY_VERSION = "bm25-full-score-rounded-tie-v1"
_KINDS: tuple[KBKind, ...] = ("encyclopedia", "recipe")
_ENTRY_ARTIFACT = "entries.jsonl"
_TOKEN_ARTIFACT = "tokens.jsonl"
_BM25_ARTIFACTS = (
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "vocab.index.json",
    "params.index.json",
)
_CHILD_ARTIFACTS = (_ENTRY_ARTIFACT, _TOKEN_ARTIFACT, *_BM25_ARTIFACTS)


class KBIndexError(ValueError):
    """A persisted KB index or its binding is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class KBTokenRow(_StrictFrozenModel):
    row_index: int = Field(ge=0)
    entry_id: str = Field(min_length=1)
    leakage_group_id: str = Field(min_length=1)
    tokens: tuple[str, ...]

    @field_validator("tokens", mode="before")
    @classmethod
    def coerce_json_tokens(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("tokens")
    @classmethod
    def validate_tokens(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not token.strip() for token in value):
            raise ValueError("token rows must contain non-blank tokens")
        return value


class TokenizerDescriptor(_StrictFrozenModel):
    policy_version: Literal["jieba-nfc-exact-no-hmm-v1"]
    jieba_version: str
    dictionary_sha256: Sha256
    normalization: Literal["NFC+casefold"]
    cut_all: Literal[False]
    hmm: Literal[False]
    indexed_fields: tuple[Literal["title", "text"], Literal["title", "text"]]

    @field_validator("indexed_fields", mode="before")
    @classmethod
    def coerce_json_fields(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_field_order(self) -> Self:
        if self.indexed_fields != ("title", "text"):
            raise ValueError("indexed_fields must be exactly title then text")
        return self


class BM25Descriptor(_StrictFrozenModel):
    bm25s_version: str
    method: Literal["lucene"]
    idf_method: Literal["lucene"]
    k1: Literal[1.5]
    b: Literal[0.75]
    delta: Literal[0.5]
    dtype: Literal["float32"]
    int_dtype: Literal["int32"]
    backend: Literal["numpy"]


class KBIndexManifest(_StrictFrozenModel):
    schema_version: Literal[2]
    policy_version: Literal["kb-bm25-index-v2"]
    ranking_policy_version: Literal["bm25-full-score-rounded-tie-v1"]
    kind: KBKind
    mode: Literal["verified", "provisional"]
    complete: bool
    catalog_sha256: Sha256
    catalog_entries_sha256: Sha256
    catalog_policy_version: str
    catalog_leakage_policy_version: str
    entry_count: int = Field(ge=1)
    catalog_kind_count: int = Field(ge=1)
    row_binding_sha256: Sha256
    tokenizer: TokenizerDescriptor
    bm25: BM25Descriptor
    artifacts: dict[str, ArtifactDescriptor]
    integrity_sha256: Sha256

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if set(self.artifacts) != set(_CHILD_ARTIFACTS):
            raise ValueError("KB index artifact set is invalid")
        for name, descriptor in self.artifacts.items():
            if descriptor.path != name:
                raise ValueError(f"artifact path for {name} is invalid")
        if self.entry_count > self.catalog_kind_count:
            raise ValueError("entry_count exceeds catalog kind count")
        if self.mode == "verified" and (
            not self.complete or self.entry_count != self.catalog_kind_count
        ):
            raise ValueError("verified KB index must cover its full catalog kind")
        if self.mode == "provisional" and self.complete:
            raise ValueError("provisional KB index must not claim completeness")
        if self.integrity_sha256 != _index_manifest_digest(self):
            raise ValueError("KB index manifest self-hash mismatch")
        return self


class BundleChildDescriptor(_StrictFrozenModel):
    path: str
    entry_count: int = Field(ge=1)
    manifest_sha256: Sha256
    integrity_sha256: Sha256


class KBIndexBundleManifest(_StrictFrozenModel):
    schema_version: Literal[2]
    policy_version: Literal["kb-bm25-index-v2"]
    mode: Literal["verified", "provisional"]
    complete: bool
    catalog_sha256: Sha256
    catalog_entries_sha256: Sha256
    catalog_policy_version: str
    catalog_leakage_policy_version: str
    indices: dict[KBKind, BundleChildDescriptor]
    bundle_sha256: Sha256

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if set(self.indices) != set(_KINDS):
            raise ValueError("bundle must contain both KB indices")
        for kind, descriptor in self.indices.items():
            if descriptor.path != kind:
                raise ValueError(f"bundle path for {kind} is invalid")
        if self.mode == "verified" and not self.complete:
            raise ValueError("verified bundle must be complete")
        if self.mode == "provisional" and self.complete:
            raise ValueError("provisional bundle must not claim completeness")
        if self.bundle_sha256 != _bundle_manifest_digest(self):
            raise ValueError("KB bundle manifest self-hash mismatch")
        return self


@dataclass(frozen=True)
class KBIndex:
    kind: KBKind
    retriever: Any
    entries: tuple[KBEntryV2, ...]
    token_rows: tuple[KBTokenRow, ...]
    manifest: KBIndexManifest


@dataclass(frozen=True)
class KBIndexBundle:
    root: Path
    encyclopedia: KBIndex
    recipe: KBIndex
    manifest: KBIndexBundleManifest
    manifest_bytes: bytes
    external_sha256_verified: bool
    catalog: KBCatalog | None = field(default=None, repr=False, compare=False)

    def for_kind(self, kind: KBKind) -> KBIndex:
        return self.encyclopedia if kind == "encyclopedia" else self.recipe


def build_kb_bundle(
    catalog_dir: Path | str,
    output_dir: Path | str,
    *,
    expected_catalog_sha256: str | None = None,
    limit_per_kind: int | None = None,
    allow_provisional: bool = False,
) -> KBIndexBundle:
    """Build both KB indices in one staging tree and publish them together."""

    if allow_provisional:
        if limit_per_kind is None or limit_per_kind <= 0:
            raise KBIndexError(
                "provisional KB index is restricted to positive limit_per_kind smoke builds"
            )
    elif limit_per_kind is not None:
        raise KBIndexError("limited KB index requires allow_provisional=True")

    output_dir = Path(output_dir)
    if _lexists(output_dir):
        raise FileExistsError(f"KB index destination already exists: {output_dir}")
    if not allow_provisional and expected_catalog_sha256 is None:
        raise KBIndexError(
            "formal KB index build requires an external expected catalog SHA-256"
        )
    catalog = load_kb_catalog(
        catalog_dir,
        allow_provisional=allow_provisional,
        expected_catalog_sha256=expected_catalog_sha256,
    )
    if not allow_provisional and catalog.manifest.mode != "verified":
        raise KBIndexError("formal KB index requires a verified catalog")
    verify_catalog_unchanged(catalog)

    tokenizer = _new_tokenizer()
    tokenizer_descriptor = _tokenizer_descriptor(tokenizer)
    grouped = {
        kind: tuple(entry for entry in catalog.entries if entry.kind == kind)
        for kind in _KINDS
    }
    selected = {
        kind: entries if limit_per_kind is None else entries[:limit_per_kind]
        for kind, entries in grouped.items()
    }
    if any(not entries for entries in selected.values()):
        raise KBIndexError("each KB index must contain at least one entry")

    staging = _create_staging_directory(output_dir)
    try:
        child_manifests: dict[KBKind, KBIndexManifest] = {}
        for kind in _KINDS:
            child_manifests[kind] = _build_child_index(
                staging / kind,
                kind=kind,
                entries=selected[kind],
                catalog=catalog,
                catalog_kind_count=len(grouped[kind]),
                tokenizer=tokenizer,
                tokenizer_descriptor=tokenizer_descriptor,
                mode="provisional" if allow_provisional else "verified",
            )
        unsigned: dict[str, Any] = {
            "schema_version": KB_INDEX_SCHEMA_VERSION,
            "policy_version": KB_INDEX_POLICY_VERSION,
            "mode": "provisional" if allow_provisional else "verified",
            "complete": not allow_provisional,
            "catalog_sha256": catalog.manifest.catalog_sha256,
            "catalog_entries_sha256": sha256_bytes(catalog.entries_bytes),
            "catalog_policy_version": catalog.manifest.policy_version,
            "catalog_leakage_policy_version": catalog.manifest.leakage_policy_version,
            "indices": {
                kind: {
                    "path": kind,
                    "entry_count": child_manifests[kind].entry_count,
                    "manifest_sha256": sha256_bytes(
                        (staging / kind / "manifest.json").read_bytes()
                    ),
                    "integrity_sha256": child_manifests[kind].integrity_sha256,
                }
                for kind in _KINDS
            },
        }
        bundle = KBIndexBundleManifest.model_validate(
            {
                **unsigned,
                "bundle_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            }
        )
        (staging / "manifest.json").write_bytes(
            canonical_json_bytes(bundle.model_dump(mode="json")) + b"\n"
        )
        load_kb_bundle(
            staging,
            allow_provisional=True,
            catalog=catalog,
            expected_bundle_sha256=bundle.bundle_sha256,
        )
        verify_catalog_unchanged(catalog)
        _publish_create_only(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    published = load_kb_bundle(
        output_dir,
        allow_provisional=allow_provisional,
        catalog=catalog,
        expected_bundle_sha256=bundle.bundle_sha256,
    )
    # The builder computed this bundle digest itself.  Return the validated
    # artifact for inspection, but require a later formal load with an
    # independently persisted digest before a formal service may consume it.
    return replace(published, external_sha256_verified=False)


def load_kb_bundle(
    root: Path | str,
    *,
    allow_provisional: bool = False,
    catalog: KBCatalog | None = None,
    expected_bundle_sha256: str | None = None,
) -> KBIndexBundle:
    """Load the two-index bundle after validating every persisted artifact."""

    if (
        expected_bundle_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha256) is None
    ):
        raise KBIndexError(
            "expected bundle SHA-256 must be 64 lowercase hex characters"
        )

    root = Path(root)
    _require_directory(root, "KB index bundle")
    if {item.name for item in root.iterdir()} != {
        "manifest.json",
        "encyclopedia",
        "recipe",
    }:
        raise KBIndexError("KB bundle root artifact set is invalid")
    manifest_snapshot = read_regular_file_snapshot(root / "manifest.json")
    try:
        manifest = KBIndexBundleManifest.model_validate(
            parse_json_object(manifest_snapshot.content, "KB bundle manifest")
        )
    except Exception as error:
        raise KBIndexError("KB bundle manifest is invalid") from error
    if manifest_snapshot.content != (
        canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    ):
        raise KBIndexError("KB bundle manifest must be canonical JSON")
    if manifest.mode == "provisional" and not allow_provisional:
        raise KBIndexError("provisional KB bundle requires explicit opt-in")
    if manifest.mode == "verified":
        if expected_bundle_sha256 is None:
            raise KBIndexError(
                "verified KB bundle requires an external expected bundle SHA-256"
            )
        if catalog is None:
            raise KBIndexError("verified KB bundle requires its verified catalog")
        if not catalog.external_sha256_verified:
            raise KBIndexError(
                "verified KB bundle requires an externally locked catalog"
            )
    if (
        expected_bundle_sha256 is not None
        and manifest.bundle_sha256 != expected_bundle_sha256
    ):
        raise KBIndexError("bundle SHA-256 does not match external expected lock")
    if catalog is not None:
        verify_catalog_unchanged(catalog)
        _verify_bundle_catalog_binding(manifest, catalog)

    loaded: dict[KBKind, KBIndex] = {}
    for kind in _KINDS:
        loaded[kind] = _load_child_index(
            root / kind,
            kind=kind,
            allow_provisional=allow_provisional,
            catalog=catalog,
        )
        descriptor = manifest.indices[kind]
        child_manifest_snapshot = read_regular_file_snapshot(
            root / kind / "manifest.json"
        )
        if descriptor.manifest_sha256 != child_manifest_snapshot.sha256:
            raise KBIndexError(f"{kind} child manifest hash mismatch")
        if descriptor.integrity_sha256 != loaded[kind].manifest.integrity_sha256:
            raise KBIndexError(f"{kind} child integrity binding mismatch")
        if descriptor.entry_count != len(loaded[kind].entries):
            raise KBIndexError(f"{kind} child entry count mismatch")
        if loaded[kind].manifest.mode != manifest.mode:
            raise KBIndexError(f"{kind} child mode differs from bundle mode")
        _verify_child_bundle_binding(loaded[kind].manifest, manifest, kind)

    verify_file_snapshot(manifest_snapshot)
    if catalog is not None:
        verify_catalog_unchanged(catalog)
    return KBIndexBundle(
        root=root,
        encyclopedia=loaded["encyclopedia"],
        recipe=loaded["recipe"],
        manifest=manifest,
        manifest_bytes=manifest_snapshot.content,
        external_sha256_verified=expected_bundle_sha256 is not None,
        catalog=catalog,
    )


def tokenize_kb_text(tokenizer: jieba.Tokenizer, value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFC", value).casefold()
    return tuple(
        token
        for token in tokenizer.lcut(normalized, cut_all=False, HMM=False)
        if token.strip()
    )


def _build_child_index(
    destination: Path,
    *,
    kind: KBKind,
    entries: tuple[KBEntryV2, ...],
    catalog: KBCatalog,
    catalog_kind_count: int,
    tokenizer: jieba.Tokenizer,
    tokenizer_descriptor: TokenizerDescriptor,
    mode: Literal["verified", "provisional"],
) -> KBIndexManifest:
    destination.mkdir()
    group_by_id = catalog.leakage_group_by_entry_id
    token_rows = tuple(
        KBTokenRow(
            row_index=index,
            entry_id=entry.entry_id,
            leakage_group_id=group_by_id[entry.entry_id],
            tokens=tokenize_kb_text(tokenizer, f"{entry.title}\n{entry.text}"),
        )
        for index, entry in enumerate(entries)
    )
    if any(not row.tokens for row in token_rows):
        raise KBIndexError(f"{kind} contains an entry with no indexable tokens")
    (destination / _ENTRY_ARTIFACT).write_bytes(canonical_jsonl_bytes(entries))
    (destination / _TOKEN_ARTIFACT).write_bytes(canonical_jsonl_bytes(token_rows))

    retriever = _new_retriever()
    retriever.index([list(row.tokens) for row in token_rows], show_progress=False)
    retriever.save(destination, allow_pickle=False, show_progress=False)
    actual_files = {item.name for item in destination.iterdir()}
    if actual_files != set(_CHILD_ARTIFACTS):
        raise KBIndexError(f"{kind} BM25 artifact set is unexpected: {actual_files}")

    unsigned: dict[str, Any] = {
        "schema_version": KB_INDEX_SCHEMA_VERSION,
        "policy_version": KB_INDEX_POLICY_VERSION,
        "ranking_policy_version": KB_RANKING_POLICY_VERSION,
        "kind": kind,
        "mode": mode,
        "complete": mode == "verified",
        "catalog_sha256": catalog.manifest.catalog_sha256,
        "catalog_entries_sha256": sha256_bytes(catalog.entries_bytes),
        "catalog_policy_version": catalog.manifest.policy_version,
        "catalog_leakage_policy_version": catalog.manifest.leakage_policy_version,
        "entry_count": len(entries),
        "catalog_kind_count": catalog_kind_count,
        "row_binding_sha256": _row_binding_sha256(entries, token_rows),
        "tokenizer": tokenizer_descriptor.model_dump(mode="json"),
        "bm25": _bm25_descriptor().model_dump(mode="json"),
        "artifacts": {
            name: artifact_descriptor(destination / name).model_dump(mode="json")
            for name in _CHILD_ARTIFACTS
        },
    }
    manifest = KBIndexManifest.model_validate(
        {
            **unsigned,
            "integrity_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )
    (destination / "manifest.json").write_bytes(
        canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    )
    return manifest


def _load_child_index(
    root: Path,
    *,
    kind: KBKind,
    allow_provisional: bool,
    catalog: KBCatalog | None,
) -> KBIndex:
    _require_directory(root, f"{kind} KB index")
    expected_names = {"manifest.json", *_CHILD_ARTIFACTS}
    if {item.name for item in root.iterdir()} != expected_names:
        raise KBIndexError(f"{kind} index artifact set is invalid")
    manifest_snapshot = read_regular_file_snapshot(root / "manifest.json")
    try:
        manifest = KBIndexManifest.model_validate(
            parse_json_object(manifest_snapshot.content, f"{kind} index manifest")
        )
    except Exception as error:
        raise KBIndexError(f"{kind} index manifest is invalid") from error
    if manifest_snapshot.content != (
        canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    ):
        raise KBIndexError(f"{kind} index manifest must be canonical JSON")
    if manifest.kind != kind:
        raise KBIndexError(f"{kind} index manifest has the wrong kind")
    if manifest.mode == "provisional" and not allow_provisional:
        raise KBIndexError(f"provisional {kind} index requires explicit opt-in")

    snapshots = {
        name: read_regular_file_snapshot(root / name) for name in _CHILD_ARTIFACTS
    }
    for name, snapshot in snapshots.items():
        descriptor = manifest.artifacts[name]
        if (
            len(snapshot.content) != descriptor.bytes
            or snapshot.sha256 != descriptor.sha256
        ):
            raise KBIndexError(f"{kind}/{name} does not match its descriptor")

    entries = tuple(
        _parse_canonical_jsonl(
            snapshots[_ENTRY_ARTIFACT].content, KBEntryV2, f"{kind}/{_ENTRY_ARTIFACT}"
        )
    )
    token_rows = tuple(
        _parse_canonical_jsonl(
            snapshots[_TOKEN_ARTIFACT].content,
            KBTokenRow,
            f"{kind}/{_TOKEN_ARTIFACT}",
        )
    )
    if len(entries) != manifest.entry_count or len(token_rows) != len(entries):
        raise KBIndexError(f"{kind} entry/token count mismatch")
    if any(entry.kind != kind for entry in entries):
        raise KBIndexError(f"{kind} index contains a cross-kind entry")
    if len({entry.entry_id for entry in entries}) != len(entries):
        raise KBIndexError(f"{kind} index contains duplicate entry_id values")
    if [row.row_index for row in token_rows] != list(range(len(entries))):
        raise KBIndexError(f"{kind} token row indices are not contiguous")
    if [row.entry_id for row in token_rows] != [entry.entry_id for entry in entries]:
        raise KBIndexError(f"{kind} token rows are not aligned to entries")
    if manifest.row_binding_sha256 != _row_binding_sha256(entries, token_rows):
        raise KBIndexError(f"{kind} row binding hash mismatch")

    tokenizer = _new_tokenizer()
    if _tokenizer_descriptor(tokenizer) != manifest.tokenizer:
        raise KBIndexError(f"{kind} tokenizer identity has drifted")
    expected_tokens = tuple(
        tokenize_kb_text(tokenizer, f"{entry.title}\n{entry.text}") for entry in entries
    )
    if tuple(row.tokens for row in token_rows) != expected_tokens:
        raise KBIndexError(f"{kind} persisted tokens do not match current entry bytes")
    if manifest.bm25 != _bm25_descriptor():
        raise KBIndexError(f"{kind} BM25 identity has drifted")

    try:
        retriever = bm25s.BM25.load(
            root,
            load_corpus=False,
            mmap=False,
            allow_pickle=False,
            load_vocab=True,
        )
    except Exception as error:
        raise KBIndexError(f"{kind} BM25 index cannot be loaded") from error
    _validate_loaded_retriever(retriever, len(entries), kind)
    for snapshot in snapshots.values():
        verify_file_snapshot(snapshot)
    verify_file_snapshot(manifest_snapshot)

    if catalog is not None:
        expected_entries = tuple(
            entry for entry in catalog.entries if entry.kind == kind
        )
        if manifest.catalog_kind_count != len(expected_entries):
            raise KBIndexError(f"{kind} catalog kind count binding is invalid")
        if manifest.mode == "provisional":
            expected_entries = expected_entries[: len(entries)]
        if entries != expected_entries:
            raise KBIndexError(f"{kind} entries do not match the bound catalog rows")
        group_by_id = catalog.leakage_group_by_entry_id
        if any(row.leakage_group_id != group_by_id[row.entry_id] for row in token_rows):
            raise KBIndexError(f"{kind} leakage group binding differs from catalog")

    return KBIndex(
        kind=kind,
        retriever=retriever,
        entries=entries,
        token_rows=token_rows,
        manifest=manifest,
    )


def _new_retriever():
    return bm25s.BM25(
        k1=1.5,
        b=0.75,
        delta=0.5,
        method="lucene",
        idf_method="lucene",
        dtype="float32",
        int_dtype="int32",
        backend="numpy",
    )


@lru_cache(maxsize=1)
def _new_tokenizer() -> jieba.Tokenizer:
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    return tokenizer


def _tokenizer_descriptor(tokenizer: jieba.Tokenizer) -> TokenizerDescriptor:
    dictionary = tokenizer.get_dict_file()
    try:
        dictionary_path = Path(dictionary.name)
    finally:
        dictionary.close()
    return TokenizerDescriptor(
        policy_version=KB_TOKENIZER_POLICY_VERSION,
        jieba_version=jieba.__version__,
        dictionary_sha256=read_regular_file_snapshot(dictionary_path).sha256,
        normalization="NFC+casefold",
        cut_all=False,
        hmm=False,
        indexed_fields=("title", "text"),
    )


def _bm25_descriptor() -> BM25Descriptor:
    return BM25Descriptor(
        bm25s_version=importlib.metadata.version("bm25s"),
        method="lucene",
        idf_method="lucene",
        k1=1.5,
        b=0.75,
        delta=0.5,
        dtype="float32",
        int_dtype="int32",
        backend="numpy",
    )


def _validate_loaded_retriever(
    retriever: Any, expected_rows: int, kind: KBKind
) -> None:
    expected_config = _bm25_descriptor()
    actual_config = {
        "method": getattr(retriever, "method", None),
        "idf_method": getattr(retriever, "idf_method", None),
        "k1": getattr(retriever, "k1", None),
        "b": getattr(retriever, "b", None),
        "delta": getattr(retriever, "delta", None),
        "dtype": getattr(retriever, "dtype", None),
        "int_dtype": getattr(retriever, "int_dtype", None),
        "backend": getattr(retriever, "backend", None),
    }
    if actual_config != expected_config.model_dump(
        exclude={"bm25s_version"}, mode="python"
    ):
        raise KBIndexError(f"{kind} BM25 loaded parameters are invalid")
    if not isinstance(retriever.vocab_dict, dict) or not retriever.vocab_dict:
        raise KBIndexError(f"{kind} BM25 vocabulary is invalid")
    vocabulary_ids = tuple(retriever.vocab_dict.values())
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in vocabulary_ids
        )
        or len(vocabulary_ids) != len(set(vocabulary_ids))
        or min(vocabulary_ids) < 0
    ):
        raise KBIndexError(f"{kind} BM25 vocabulary IDs are invalid")
    scores = getattr(retriever, "scores", None)
    if not isinstance(scores, dict) or scores.get("num_docs") != expected_rows:
        raise KBIndexError(f"{kind} BM25 document count is invalid")
    data = np.asarray(scores.get("data"))
    indices = np.asarray(scores.get("indices"))
    indptr = np.asarray(scores.get("indptr"))
    if data.dtype != np.float32 or not np.all(np.isfinite(data)):
        raise KBIndexError(f"{kind} BM25 score data is invalid")
    if (
        indices.ndim != 1
        or indptr.ndim != 1
        or data.ndim != 1
        or indices.dtype != np.int32
    ):
        raise KBIndexError(f"{kind} BM25 sparse arrays are invalid")
    if len(data) != len(indices) or len(indptr) not in {
        len(retriever.vocab_dict),
        len(retriever.vocab_dict) + 1,
    }:
        raise KBIndexError(f"{kind} BM25 sparse shapes are invalid")
    if len(indices) and (indices.min() < 0 or indices.max() >= expected_rows):
        raise KBIndexError(f"{kind} BM25 row indices are invalid")
    if indptr[0] != 0 or indptr[-1] != len(data) or np.any(np.diff(indptr) < 0):
        raise KBIndexError(f"{kind} BM25 sparse pointers are invalid")


def _row_binding_sha256(
    entries: tuple[KBEntryV2, ...], token_rows: tuple[KBTokenRow, ...]
) -> str:
    rows = (
        {
            "row_index": index,
            "entry_id": entry.entry_id,
            "content_sha256": entry.content_sha256,
            "leakage_group_id": token_row.leakage_group_id,
            "tokens_sha256": sha256_bytes(canonical_json_bytes(list(token_row.tokens))),
        }
        for index, (entry, token_row) in enumerate(
            zip(entries, token_rows, strict=True)
        )
    )
    return sha256_bytes(canonical_jsonl_bytes(rows))


def _verify_bundle_catalog_binding(
    manifest: KBIndexBundleManifest, catalog: KBCatalog
) -> None:
    expected = {
        "catalog_sha256": catalog.manifest.catalog_sha256,
        "catalog_entries_sha256": sha256_bytes(catalog.entries_bytes),
        "catalog_policy_version": catalog.manifest.policy_version,
        "catalog_leakage_policy_version": catalog.manifest.leakage_policy_version,
    }
    actual = {field: getattr(manifest, field) for field in expected}
    if actual != expected:
        raise KBIndexError("KB bundle does not match the supplied catalog")


def _verify_child_bundle_binding(
    child: KBIndexManifest,
    bundle: KBIndexBundleManifest,
    kind: KBKind,
) -> None:
    fields = (
        "catalog_sha256",
        "catalog_entries_sha256",
        "catalog_policy_version",
        "catalog_leakage_policy_version",
    )
    if any(getattr(child, field) != getattr(bundle, field) for field in fields):
        raise KBIndexError(f"{kind} child catalog binding differs from bundle")


def _index_manifest_digest(manifest: KBIndexManifest) -> str:
    unsigned = manifest.model_dump(mode="json", exclude={"integrity_sha256"})
    return sha256_bytes(canonical_json_bytes(unsigned))


def _bundle_manifest_digest(manifest: KBIndexBundleManifest) -> str:
    unsigned = manifest.model_dump(mode="json", exclude={"bundle_sha256"})
    return sha256_bytes(canonical_json_bytes(unsigned))


def _parse_canonical_jsonl(content: bytes, model, label: str):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise KBIndexError(f"{label} must be UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise KBIndexError(f"{label} must contain non-empty JSONL rows")
    values = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=lambda value: _raise_invalid_constant(value),
            )
            values.append(model.model_validate(value))
        except Exception as error:
            raise KBIndexError(f"{label}:{line_number} is invalid") from error
    if content != canonical_jsonl_bytes(values):
        raise KBIndexError(f"{label} must be canonical JSONL")
    return values


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise KBIndexError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _raise_invalid_constant(value: str):
    raise KBIndexError(f"non-finite JSON value: {value}")


def _require_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise KBIndexError(f"{label} is missing: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise KBIndexError(f"{label} must be a non-symlink directory")


def _create_staging_directory(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )


def _publish_create_only(staging: Path, destination: Path) -> None:
    atomic_publish_new_directory(staging, destination)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


__all__ = [
    "BM25Descriptor",
    "KBIndex",
    "KBIndexBundle",
    "KBIndexBundleManifest",
    "KBIndexError",
    "KBIndexManifest",
    "KBTokenRow",
    "KB_INDEX_POLICY_VERSION",
    "KB_INDEX_SCHEMA_VERSION",
    "KB_RANKING_POLICY_VERSION",
    "KB_TOKENIZER_POLICY_VERSION",
    "TokenizerDescriptor",
    "build_kb_bundle",
    "load_kb_bundle",
    "tokenize_kb_text",
]
