"""Deterministic encyclopedia and recipe lookup over a verified KB bundle."""

from __future__ import annotations

import hashlib
import os
from threading import Lock
from types import MappingProxyType
from typing import Annotated, Callable, Literal, Self, cast

import jieba
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    model_validator,
)

from skillchain.data.kb_catalog import KBEntryV2, KBKind, load_kb_catalog
from skillchain.tools.contracts import validate_json_value
from skillchain.tools.kb_index import (
    KBIndex,
    KBIndexBundle,
    load_kb_bundle,
    tokenize_kb_text,
)
from skillchain.tools.settings import KB_CATALOG_DIR, KB_INDEX_DIR, KB_K
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
KB_QUERY_MAX_CHARS = 256


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class KBRetrievalArtifactBinding(_StrictModel):
    """Index and source identity inherited by every KB hit."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    mode: Literal["verified", "provisional"]
    kind: KBKind
    bundle_sha256: Sha256 | None = None
    index_integrity_sha256: Sha256 | None = None
    catalog_sha256: Sha256 | None = None
    catalog_entries_sha256: Sha256 | None = None
    indexed_entries_sha256: Sha256 | None = None
    tokenizer_policy_version: str | None = None
    ranking_policy_version: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        evidence = (
            self.bundle_sha256,
            self.index_integrity_sha256,
            self.catalog_sha256,
            self.catalog_entries_sha256,
            self.indexed_entries_sha256,
            self.tokenizer_policy_version,
            self.ranking_policy_version,
        )
        if self.mode == "verified":
            if any(value is None for value in evidence):
                raise ValueError("verified KB retrieval binding is incomplete")
            if (
                not self.tokenizer_policy_version
                or not self.tokenizer_policy_version.strip()
                or not self.ranking_policy_version
                or not self.ranking_policy_version.strip()
            ):
                raise ValueError("KB retrieval policy identities must not be blank")
        elif any(value is not None for value in evidence):
            raise ValueError(
                "provisional KB binding must not imitate verified evidence"
            )
        return self


class KBCitation(_StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    entry_id: str = Field(min_length=1)
    source_dataset: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    attribution: str | None = None
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    excerpt_sha256: Sha256

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("citation span must be non-empty")
        return self


class KBHit(_StrictModel):
    rank: int = Field(ge=1)
    score: FiniteFloat = Field(gt=0.0)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: KBKind
    origin: Literal["dump", "llm_synth"]
    verification_status: Literal["source_verified", "grounded_verified", "unverified"]
    synth_provider: str | None = None
    synth_model: str | None = None
    citation: KBCitation
    artifact_binding: KBRetrievalArtifactBinding


class KBLookupService:
    """Search one immutable two-index bundle with full-score deterministic ties."""

    def __init__(
        self,
        bundle: KBIndexBundle,
        *,
        allow_provisional: bool = False,
    ) -> None:
        if bundle.manifest.mode == "provisional" and not allow_provisional:
            raise ValueError("provisional KB bundle requires explicit service opt-in")
        if bundle.manifest.mode == "verified" and not bundle.external_sha256_verified:
            raise ValueError("verified KB service requires an externally locked bundle")
        self.bundle = bundle
        self._bindings = MappingProxyType(
            {
                kind: self._artifact_binding(bundle.for_kind(kind))
                for kind in ("encyclopedia", "recipe")
            }
        )

    @property
    def formal_runtime_binding_sha256(self) -> str:
        if (
            self.bundle.manifest.mode != "verified"
            or not self.bundle.external_sha256_verified
        ):
            raise ValueError("formal KB runtime requires an externally locked bundle")
        if self.bundle.catalog is None:
            raise ValueError("formal KB runtime requires its verified source catalog")
        fresh = load_kb_bundle(
            self.bundle.root,
            catalog=self.bundle.catalog,
            expected_bundle_sha256=self.bundle.manifest.bundle_sha256,
        )
        live_state = _kb_bundle_live_state_sha256(self.bundle)
        if live_state != _kb_bundle_live_state_sha256(fresh):
            raise ValueError("formal KB in-memory index differs from locked artifacts")
        expected_bindings = {
            kind: self._artifact_binding(self.bundle.for_kind(kind))
            for kind in ("encyclopedia", "recipe")
        }
        if dict(self._bindings) != expected_bindings:
            raise ValueError("formal KB evidence bindings changed after construction")
        payload = {
            "artifact_bindings": {
                kind: binding.model_dump(mode="json")
                for kind, binding in sorted(expected_bindings.items())
            },
            "bundle_sha256": self.bundle.manifest.bundle_sha256,
            "live_state_sha256": live_state,
            "lookup_policy_version": "kb-lookup-full-score-v1",
            "service": "KBLookupService",
        }
        return sha256_bytes(canonical_json_bytes(payload))

    def artifact_binding_for(self, kind: KBKind) -> KBRetrievalArtifactBinding:
        """Expose the exact verified evidence binding for formal validation."""

        if kind not in self._bindings:
            raise ValueError("unknown KB evidence kind")
        binding = self._bindings[kind]
        if binding.mode != "verified":
            raise ValueError("formal KB evidence binding is provisional")
        return binding

    def encyclopedia_lookup(self, entity: str) -> list[dict]:
        return self._lookup("encyclopedia", entity)

    def recipe_lookup(self, dish: str) -> list[dict]:
        return self._lookup("recipe", dish)

    def _lookup(self, kind: KBKind, query: str) -> list[dict]:
        query = _validate_query(query)
        index = self.bundle.for_kind(kind)
        # A fresh tokenizer prevents mutable per-process user-word state from
        # changing a formal lookup after the runtime snapshot was locked.
        tokenizer = jieba.Tokenizer()
        tokenizer.initialize()
        tokens = tokenize_kb_text(tokenizer, query)
        known_tokens = [
            token for token in tokens if token in index.retriever.vocab_dict
        ]
        if not known_tokens:
            return []
        try:
            scores = np.asarray(
                index.retriever.get_scores(known_tokens), dtype=np.float32
            )
        except Exception as error:
            raise ValueError(f"{kind} BM25 query failed") from error
        if scores.shape != (len(index.entries),):
            raise ValueError(f"{kind} BM25 returned an invalid score shape")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{kind} BM25 returned non-finite scores")

        ranked = [
            (entry, _round_score(float(scores[row])))
            for row, entry in enumerate(index.entries)
            if _round_score(float(scores[row])) > 0.0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0].entry_id))
        return [
            _hit_dict(
                entry,
                score,
                rank,
                self._bindings[kind],
            )
            for rank, (entry, score) in enumerate(ranked[:KB_K], start=1)
        ]

    def _artifact_binding(self, index: KBIndex) -> KBRetrievalArtifactBinding:
        if self.bundle.manifest.mode == "provisional":
            return KBRetrievalArtifactBinding(mode="provisional", kind=index.kind)
        manifest = index.manifest
        return KBRetrievalArtifactBinding(
            mode="verified",
            kind=index.kind,
            bundle_sha256=self.bundle.manifest.bundle_sha256,
            index_integrity_sha256=manifest.integrity_sha256,
            catalog_sha256=manifest.catalog_sha256,
            catalog_entries_sha256=manifest.catalog_entries_sha256,
            indexed_entries_sha256=manifest.artifacts["entries.jsonl"].sha256,
            tokenizer_policy_version=manifest.tokenizer.policy_version,
            ranking_policy_version=manifest.ranking_policy_version,
        )


def _validate_query(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("KB query must be a string")
    value = value.strip()
    if not value:
        raise ValueError("KB query must not be blank")
    if len(value) > KB_QUERY_MAX_CHARS:
        raise ValueError(f"KB query must not exceed {KB_QUERY_MAX_CHARS} characters")
    return value


def _kb_bundle_live_state_sha256(bundle: KBIndexBundle) -> str:
    indices: dict[str, object] = {}
    for kind in ("encyclopedia", "recipe"):
        index = bundle.for_kind(kind)
        scores = getattr(index.retriever, "scores", None)
        if not isinstance(scores, dict):
            raise ValueError(f"{kind} KB retriever scores are unavailable")
        arrays: dict[str, object] = {}
        for name in ("data", "indices", "indptr"):
            array = np.asarray(scores.get(name))
            if array.ndim != 1:
                raise ValueError(f"{kind} KB retriever array is invalid")
            arrays[name] = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(array).tobytes()
                ).hexdigest(),
            }
        vocab = getattr(index.retriever, "vocab_dict", None)
        if not isinstance(vocab, dict):
            raise ValueError(f"{kind} KB retriever vocabulary is invalid")
        indices[kind] = {
            "entries_sha256": sha256_bytes(
                canonical_json_bytes(
                    [item.model_dump(mode="json") for item in index.entries]
                )
            ),
            "manifest": index.manifest.model_dump(mode="json"),
            "retriever": {
                "arrays": arrays,
                "b": getattr(index.retriever, "b", None),
                "backend": getattr(index.retriever, "backend", None),
                "delta": getattr(index.retriever, "delta", None),
                "dtype": getattr(index.retriever, "dtype", None),
                "idf_method": getattr(index.retriever, "idf_method", None),
                "int_dtype": getattr(index.retriever, "int_dtype", None),
                "k1": getattr(index.retriever, "k1", None),
                "method": getattr(index.retriever, "method", None),
                "num_docs": scores.get("num_docs"),
                "vocab": [
                    [key, value]
                    for key, value in sorted(
                        (str(key), value) for key, value in vocab.items()
                    )
                ],
            },
            "token_rows_sha256": sha256_bytes(
                canonical_json_bytes(
                    [item.model_dump(mode="json") for item in index.token_rows]
                )
            ),
        }
    return sha256_bytes(
        canonical_json_bytes(
            {
                "bundle_manifest": bundle.manifest.model_dump(mode="json"),
                "indices": indices,
                "policy_version": "kb-live-state-v1",
            }
        )
    )


def _round_score(value: float) -> float:
    return round(value, 8)


def _hit_dict(
    entry: KBEntryV2,
    score: float,
    rank: int,
    binding: KBRetrievalArtifactBinding,
) -> dict:
    excerpt_hash = hashlib.sha256(entry.text.encode("utf-8")).hexdigest()
    hit = KBHit(
        rank=rank,
        score=score,
        title=entry.title,
        text=entry.text,
        kind=entry.kind,
        origin=entry.origin,
        verification_status=entry.verification_status,
        synth_provider=entry.synth_provider,
        synth_model=entry.synth_model,
        citation=KBCitation(
            entry_id=entry.entry_id,
            source_dataset=entry.source_dataset,
            source_revision=entry.source_revision,
            source_record_id=entry.source_record_id,
            source_uri=entry.source_uri,
            license_id=entry.license_id,
            attribution=entry.attribution,
            char_start=0,
            char_end=len(entry.text),
            excerpt_sha256=excerpt_hash,
        ),
        artifact_binding=binding,
    )
    value = validate_json_value(hit.model_dump(mode="json"))
    if not isinstance(value, dict):
        raise TypeError("KB hit must serialize to a JSON object")
    return cast(dict, value)


type ServiceFactory = Callable[[], KBLookupService]
_configured_factory: ServiceFactory | None = None
_service_singleton: KBLookupService | None = None
_service_lock = Lock()


def configure_kb_lookup_service_factory(factory: ServiceFactory | None) -> None:
    """Set a resettable factory for explicit test/local dependency injection."""

    global _configured_factory, _service_singleton
    with _service_lock:
        _configured_factory = factory
        _service_singleton = None


def _default_service_factory() -> KBLookupService:
    expected_catalog_sha256 = os.environ.get("SKILLCHAIN_KB_CATALOG_SHA256")
    expected_bundle_sha256 = os.environ.get("SKILLCHAIN_KB_BUNDLE_SHA256")
    if expected_catalog_sha256 is None or expected_bundle_sha256 is None:
        raise ValueError(
            "formal KB service requires SKILLCHAIN_KB_CATALOG_SHA256 and "
            "SKILLCHAIN_KB_BUNDLE_SHA256"
        )
    catalog = load_kb_catalog(
        KB_CATALOG_DIR,
        expected_catalog_sha256=expected_catalog_sha256,
    )
    return KBLookupService(
        load_kb_bundle(
            KB_INDEX_DIR,
            catalog=catalog,
            expected_bundle_sha256=expected_bundle_sha256,
        )
    )


def _get_service() -> KBLookupService:
    global _service_singleton
    if _service_singleton is None:
        with _service_lock:
            if _service_singleton is None:
                factory = _configured_factory or _default_service_factory
                _service_singleton = factory()
    return _service_singleton


def encyclopedia_lookup(entity: str) -> list[dict]:
    return _get_service().encyclopedia_lookup(entity)


def recipe_lookup(dish: str) -> list[dict]:
    return _get_service().recipe_lookup(dish)


__all__ = [
    "KBCitation",
    "KBHit",
    "KBLookupService",
    "KBRetrievalArtifactBinding",
    "KB_QUERY_MAX_CHARS",
    "configure_kb_lookup_service_factory",
    "encyclopedia_lookup",
    "recipe_lookup",
]
