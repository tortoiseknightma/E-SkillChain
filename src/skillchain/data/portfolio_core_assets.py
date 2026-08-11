"""Resumable, non-formal clean-asset preparation for the Portfolio core run.

This module intentionally does *not* know how to crawl, extract, or infer
records from any source dataset.  Source-specific adapters write a strict
candidate inventory first.  The inventory is then preflighted, copied into a
clean root one candidate at a time, and turned into ``DatasetAssetDraft``
records suitable for the generic asset-catalog builder.

The workflow is a Portfolio Track convenience layer.  Its manifests say so
explicitly and must never be used as a substitute for formal source review,
human label review, or a formal source lock.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    load_asset_catalog,
)
from skillchain.synthesis.planning import (
    CapabilityAssignment,
    PHASE3_TASK_SPEC_SHA256,
    PHASE3_TASK_SPEC_VERSION,
    load_capability_assignments,
    load_plan,
    validate_capability_binding,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.task_spec import load_mvp_task_specification_v1
from skillchain.taxonomy import (
    DEFAULT_TAXONOMY_SHA256,
    TAXONOMY_VERSION,
    load_default_taxonomy_registry,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
)


PORTFOLIO_CORE_ASSET_POLICY_VERSION = "portfolio-core-asset-preparation-v1"
PORTFOLIO_CORE_SOURCE_POLICY_VERSION = "portfolio-core-source-pool-policy-v3"
PORTFOLIO_CORE_ASSIGNMENT_POLICY_VERSION = "portfolio-core-capability-assignments-v1"
PORTFOLIO_CORE_COMPONENT_FLOOR_CLOSURE_POLICY_VERSION = (
    "portfolio-core-component-floor-closure-v1"
)
PORTFOLIO_CORE_V8_CAPABILITY_BINDING_POLICY_VERSION = (
    "portfolio-core-v8-capability-binding-v1"
)
PORTFOLIO_CORE_V9_MULTI_CLOSURE_POLICY_VERSION = (
    "portfolio-core-v9-rpc-multi-closure-v1"
)

_RUN_MANIFEST_FILE = "portfolio-core-asset-run.json"
_SELECTION_MANIFEST_FILE = "selection-manifest.json"
_DRAFTS_FILE = "dataset-assets.jsonl"
_CHECKPOINT_DIRECTORY = "checkpoints"
_MATERIALIZATION_WRITER_LOCK_FILE = "portfolio-core-materialize.writer.lock"
_IMAGE_ROOT = "query_images"
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_MAX_ASSET_BYTES = 256 * 1024 * 1024
_MAX_METADATA_BYTES = 128 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_CANDIDATE_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{2,159}$")

Pool = Literal[
    "exact_match",
    "multi_product",
    "divergent_rec",
    "encyclopedia",
    "utility",
]
SourceAvailability = Literal[
    "ready",
    "pending_extraction",
    "locked_archive",
    "restricted",
    "metadata_only",
    "permanently_unavailable",
]
SelectionState = Literal["include", "reserve"]

_POOL_ORDER: tuple[Pool, ...] = (
    "exact_match",
    "multi_product",
    "divergent_rec",
    "encyclopedia",
    "utility",
)

# This is deliberately a *capacity floor*, not a claim that a source is a
# formal positive or that these image counts alone can satisfy core planning.
# The values account for the fresh 200-query dev prefix and 1,300-query core
# tail (assuming distinct leakage components): they prevent a dev-mini-sized
# or one-image inventory from being described as ready for 1,500 / 60 batches.
CORE_POOL_ASSET_FLOORS: dict[Pool, int] = {
    "exact_match": 157,
    "multi_product": 105,
    "divergent_rec": 122,
    "encyclopedia": 122,
    "utility": 146,
}

# Portfolio Core r2 fixes the capability-level *unique candidate* contract
# before a plan is emitted.  It is deliberately independent of the 1,500
# query allocation: the planner can make bounded, auditable reuse of a
# content-unique candidate, including its declared cross-intent bindings.
# These floors therefore prevent a small source inventory from being described
# as ready while avoiding an incorrect one-candidate-per-query requirement.
CORE_R2_CAPABILITY_ASSET_FLOORS: dict[str, int] = {
    "product.exact_match": 340,
    "product.multi_search": 105,
    "product.style_recommendation": 150,
    "knowledge.visual_encyclopedia": 152,
    "utility.document_reading": 90,
    "utility.recipe_guidance": 210,
}
CORE_R2_DOCUMENT_CARRY_FORWARD_FLOOR = 30
CORE_R2_DOCUMENT_FRESH_FLOOR = 60
CORE_R2_DOCUMENT_CORD_FRESH_FLOOR = 30
CORE_R2_DOCUMENT_SROIE_FRESH_FLOOR = 30
CORE_R2_DOCUMENT_CORD_RESERVE_FLOOR = 5
CORE_R2_DOCUMENT_SROIE_RESERVE_FLOOR = 5
CORE_R2_V8_ABO_TRIPLET_ADDITION_COUNT = 14
CORE_R2_V8_ISIA_FOOD_PAIR_ADDITION_COUNT = 10
CORE_R2_V9_PARENT_RPC_INCLUDE_COUNT = CORE_POOL_ASSET_FLOORS["multi_product"]
CORE_R2_V9_PARENT_RPC_RESERVE_COUNT = 4
CORE_R2_V9_RPC_STAGING_INCLUDE_COUNT = 125
CORE_R2_V9_RPC_STAGING_RESERVE_COUNT = 4
CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT = 20
CORE_R2_V9_FRESH_RPC_MULTI_COUNT = 16

CoreCapacityProfile = Literal["r1", "r2"]
DocumentSelectionRole = Literal[
    "carry_forward",
    "fresh",
    "fresh_cord",
    "fresh_sroie",
    "reserve",
]
CapabilityBindingPlanProfile = Literal[
    "abo_product_identity_style_knowledge_triplet",
    "isia_food_recipe_knowledge_pair",
]

_DOCUMENT_SOURCE_IDS = frozenset({"wikimedia_commons_documents", "cord", "sroie"})

# The ceiling is the code-level translation of the frozen source portfolio.
# A source-specific policy snapshot may become more restrictive or change a
# source's readiness state, but it cannot silently broaden these roles.
_SOURCE_POOL_CEILINGS: dict[str, tuple[Pool, ...]] = {
    "abo": ("exact_match",),
    "cord": ("utility",),
    "csds": (),
    "deepfashion": (),
    "fashioniq": ("divergent_rec",),
    "inaturalist": ("encyclopedia",),
    "isia_food500": ("utility",),
    "mep3m": (),
    "polyvore": ("divergent_rec",),
    "products_10k": ("exact_match",),
    "recipe1m_plus": ("utility",),
    "rpc": ("multi_product",),
    "sroie": ("utility",),
    "simmc_2_1": (),
    "sku_110k": (),
    "u_need": (),
    "wikimedia_commons_documents": ("utility",),
}

# Pools alone are not a sufficient role boundary because one pool can support
# multiple capabilities (notably ``utility``), while the deliberately bounded
# ABO cross-intent set supports three capabilities from one exact-match image
# pool.  Keep the source/capability contract explicit and fail closed when an
# included candidate attempts to broaden it.
_SOURCE_CAPABILITY_CEILINGS: dict[str, frozenset[tuple[str, str]]] = {
    "abo": frozenset(
        {
            ("exact_match", "product.exact_match"),
            ("divergent_rec", "product.style_recommendation"),
            ("encyclopedia", "knowledge.visual_encyclopedia"),
        }
    ),
    "cord": frozenset({("utility", "utility.document_reading")}),
    "csds": frozenset(),
    # A declared DeepFashion style candidate may remain in an early inventory
    # so preflight can report ``pending_extraction``.  Its empty pool ceiling
    # and non-ready policy still prevent inclusion/materialization.
    "deepfashion": frozenset({("divergent_rec", "product.style_recommendation")}),
    "fashioniq": frozenset({("divergent_rec", "product.style_recommendation")}),
    "inaturalist": frozenset({("encyclopedia", "knowledge.visual_encyclopedia")}),
    "isia_food500": frozenset({("utility", "utility.recipe_guidance")}),
    "mep3m": frozenset(),
    "polyvore": frozenset({("divergent_rec", "product.style_recommendation")}),
    "products_10k": frozenset({("exact_match", "product.exact_match")}),
    "recipe1m_plus": frozenset({("utility", "utility.recipe_guidance")}),
    "rpc": frozenset({("multi_product", "product.multi_search")}),
    "sroie": frozenset({("utility", "utility.document_reading")}),
    "simmc_2_1": frozenset(),
    "sku_110k": frozenset(),
    "u_need": frozenset(),
    "wikimedia_commons_documents": frozenset({("utility", "utility.document_reading")}),
}


class PortfolioCoreAssetError(ValueError):
    """The non-formal portfolio asset preparation inputs are inconsistent."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _clean_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-blank without edge whitespace")
    return value


def _canonical_relative_path(value: str, field_name: str) -> str:
    value = _clean_text(value, field_name)
    if value != unicodedata.normalize("NFC", value) or "\\" in value:
        raise ValueError(f"{field_name} must be normalized POSIX text")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a safe relative path")
    if path.as_posix() != value:
        raise ValueError(f"{field_name} is not a normalized POSIX path")
    return value


def _validate_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def document_component_key(
    *,
    source_id: str = "wikimedia_commons_documents",
    source_record_id: str,
    content_sha256: str,
    component_fingerprint: str | None = None,
) -> str:
    """Return the opaque deterministic component identity for one document.

    Keeping this identity outside the generic candidate schema lets the r2
    carry-forward manifest prove that old and fresh selections are disjoint
    without altering immutable v4 candidate bytes.
    """

    return sha256_bytes(
        canonical_json_bytes(
            {
                "component_fingerprint": component_fingerprint,
                "content_sha256": content_sha256,
                "source_id": source_id,
                "source_record_id": source_record_id,
            }
        )
    )


# Preserve the private spelling for nearby Pydantic validation code while
# exporting the public helper for inventory adapters and callers.
_document_component_key = document_component_key


class CandidateCapabilityBinding(_StrictModel):
    """A source-inventory-declared capability use for one selected image."""

    canonical_intent: Literal[
        "exact_match",
        "multi_product",
        "divergent_rec",
        "encyclopedia",
        "utility",
    ]
    canonical_capability: str

    @field_validator("canonical_capability")
    @classmethod
    def validate_capability(cls, value: str) -> str:
        return _clean_text(value, "canonical_capability")


class PortfolioCoreAssetCandidate(_StrictModel):
    """One declared source file eligible for a Portfolio clean selection.

    ``draft.local_path`` is the path beneath the source root.  On publication
    it is replaced with ``destination_path``; provenance fields are preserved.
    No adapter-specific record parsing happens here.
    """

    schema_version: Literal[1] = 1
    candidate_id: str
    source_id: str
    selection: SelectionState = "include"
    pool: Pool
    source_local_path: str
    destination_path: str
    expected_bytes: int = Field(gt=0)
    expected_sha256: str
    draft: DatasetAssetDraft
    capability_bindings: tuple[CandidateCapabilityBinding, ...] = Field(min_length=1)

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        value = _clean_text(value, "candidate_id")
        if not _CANDIDATE_ID_RE.fullmatch(value):
            raise ValueError("candidate_id has an invalid format")
        return value

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        value = _clean_text(value, "source_id")
        if not _SOURCE_ID_RE.fullmatch(value):
            raise ValueError("source_id has an invalid format")
        return value

    @field_validator("source_local_path", "destination_path")
    @classmethod
    def validate_paths(cls, value: str, info) -> str:
        return _canonical_relative_path(value, info.field_name)

    @field_validator("expected_sha256")
    @classmethod
    def validate_expected_sha256(cls, value: str) -> str:
        return _validate_sha256(value, "expected_sha256")

    @field_validator("capability_bindings", mode="before")
    @classmethod
    def coerce_bindings(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("capability_bindings")
    @classmethod
    def validate_bindings(
        cls, value: tuple[CandidateCapabilityBinding, ...]
    ) -> tuple[CandidateCapabilityBinding, ...]:
        keys = [
            (binding.canonical_intent, binding.canonical_capability)
            for binding in value
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("capability_bindings must not contain duplicates")
        return tuple(
            sorted(
                value,
                key=lambda item: (item.canonical_intent, item.canonical_capability),
            )
        )

    def model_post_init(self, __context: Any) -> None:
        if self.draft.local_path != self.source_local_path:
            raise ValueError("draft.local_path must equal source_local_path")
        if self.draft.source_dataset != self.source_id:
            raise ValueError("draft.source_dataset must equal source_id")
        required_prefix = f"{_IMAGE_ROOT}/{self.pool}/"
        if not self.destination_path.startswith(required_prefix):
            raise ValueError(
                "destination_path must be below the selected pool's query_images directory"
            )
        if PurePosixPath(self.destination_path).suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError("destination_path must name a supported image file")


class PortfolioCoreDocumentSelectionEntry(_StrictModel):
    """One explicit multi-source document selection binding.

    Candidate inventories intentionally remain schema-v1 so the immutable v4
    inventory continues to load byte-for-byte.  This side manifest carries
    r2-only provenance instead of adding a defaulted field that would change
    canonical v4 JSONL serialization.
    """

    candidate_id: str
    source_id: str
    source_record_id: str
    content_sha256: str
    component_fingerprint: str
    component_key: str
    selection_role: DocumentSelectionRole

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        value = _clean_text(value, "candidate_id")
        if not _CANDIDATE_ID_RE.fullmatch(value):
            raise ValueError("candidate_id has an invalid format")
        return value

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        value = _clean_text(value, "source_id")
        if value not in _DOCUMENT_SOURCE_IDS:
            raise ValueError("source_id is not a supported document source")
        return value

    @field_validator("source_record_id")
    @classmethod
    def validate_source_record_id(cls, value: str) -> str:
        return _clean_text(value, "source_record_id")

    @field_validator("content_sha256", "component_key")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("component_fingerprint")
    @classmethod
    def validate_component_fingerprint(cls, value: str) -> str:
        value = _clean_text(value, "component_fingerprint")
        if not re.fullmatch(r"[0-9a-f]{16}", value):
            raise ValueError("component_fingerprint must be a 16-character pHash")
        return value

    def model_post_init(self, __context: Any) -> None:
        expected_component = _document_component_key(
            source_id=self.source_id,
            source_record_id=self.source_record_id,
            content_sha256=self.content_sha256,
            component_fingerprint=self.component_fingerprint,
        )
        if self.component_key != expected_component:
            raise ValueError("component_key does not bind document record content")


class PortfolioCoreDocumentSelectionManifest(_StrictModel):
    """Canonical r2-only manifest for document selection provenance.

    The immutable v4 candidate inventory stays schema-v1.  This v2 side
    manifest is deliberately the only location where the Wikimedia carry
    forward and CORD/SROIE fresh-source roles are joined.
    """

    schema_version: Literal[2] = 2
    selection_policy_version: Literal["portfolio-core-document-multi-source-v2"] = (
        "portfolio-core-document-multi-source-v2"
    )
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    formal_status: Literal["non_formal"] = "non_formal"
    dev_mini_selection_sha256: str
    source_manifest_sha256: dict[str, str] = Field(min_length=1)
    inventory_sha256: str
    selections: tuple[PortfolioCoreDocumentSelectionEntry, ...] = Field(min_length=1)

    @field_validator(
        "dev_mini_selection_sha256",
        "inventory_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("source_manifest_sha256")
    @classmethod
    def validate_source_manifest_sha256(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("source_manifest_sha256 must be non-empty")
        for source_id, digest in value.items():
            if source_id not in _DOCUMENT_SOURCE_IDS:
                raise ValueError("source manifest has an unsupported document source")
            _validate_sha256(digest, f"source_manifest_sha256[{source_id}]")
        return dict(sorted(value.items()))

    @field_validator("selections", mode="before")
    @classmethod
    def coerce_selections(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("selections")
    @classmethod
    def validate_selections(
        cls, value: tuple[PortfolioCoreDocumentSelectionEntry, ...]
    ) -> tuple[PortfolioCoreDocumentSelectionEntry, ...]:
        if len({row.candidate_id for row in value}) != len(value):
            raise ValueError("document selection has duplicate candidate IDs")
        if len({row.source_record_id for row in value}) != len(value):
            raise ValueError("document selection has duplicate source records")
        if len({row.content_sha256 for row in value}) != len(value):
            raise ValueError("document selection has duplicate content")
        if len({row.component_fingerprint for row in value}) != len(value):
            raise ValueError("document selection has duplicate pHash components")
        if len({row.component_key for row in value}) != len(value):
            raise ValueError("document selection has duplicate components")
        role_order = {
            "carry_forward": 0,
            "fresh": 1,
            "fresh_cord": 2,
            "fresh_sroie": 3,
            "reserve": 4,
        }
        ordered = tuple(
            sorted(
                value,
                key=lambda row: (
                    role_order[row.selection_role],
                    row.source_id,
                    row.source_record_id,
                    row.candidate_id,
                ),
            )
        )
        if value != ordered:
            raise ValueError("document selection rows are not in canonical order")
        return value

    def model_post_init(self, __context: Any) -> None:
        missing = {row.source_id for row in self.selections} - set(
            self.source_manifest_sha256
        )
        if missing:
            raise ValueError("document selection lacks a source-manifest binding")


class PortfolioCoreReserveActivationEntry(_StrictModel):
    """One reserve asset intentionally made available to the planner."""

    candidate_id: str
    source_id: str
    destination_path: str
    expected_sha256: str
    predicted_component_delta: Literal[1] = 1

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        value = _clean_text(value, "candidate_id")
        if not _CANDIDATE_ID_RE.fullmatch(value):
            raise ValueError("candidate_id has an invalid format")
        return value

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        value = _clean_text(value, "source_id")
        if not _SOURCE_ID_RE.fullmatch(value):
            raise ValueError("source_id has an invalid format")
        return value

    @field_validator("destination_path")
    @classmethod
    def validate_destination_path(cls, value: str) -> str:
        return _canonical_relative_path(value, "destination_path")

    @field_validator("expected_sha256")
    @classmethod
    def validate_expected_sha256(cls, value: str) -> str:
        return _validate_sha256(value, "expected_sha256")


class PortfolioCoreReserveActivationManifest(_StrictModel):
    """Canonical component-floor closure plan for previously reserved assets."""

    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-component-floor-closure-v1"] = (
        PORTFOLIO_CORE_COMPONENT_FLOOR_CLOSURE_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    formal_status: Literal["non_formal"] = "non_formal"
    parent_inventory_sha256: str
    parent_selection_manifest_sha256: str
    parent_catalog_sha256: str
    baseline_component_count: int = Field(ge=0)
    required_component_floor: int = Field(ge=1)
    closure_margin: int = Field(ge=0)
    predicted_component_count: int = Field(ge=1)
    reason: str
    selections: tuple[PortfolioCoreReserveActivationEntry, ...] = Field(min_length=1)

    @field_validator(
        "parent_inventory_sha256",
        "parent_selection_manifest_sha256",
        "parent_catalog_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _clean_text(value, "reason")

    @field_validator("selections", mode="before")
    @classmethod
    def coerce_selections(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("selections")
    @classmethod
    def validate_selections(
        cls,
        value: tuple[PortfolioCoreReserveActivationEntry, ...],
    ) -> tuple[PortfolioCoreReserveActivationEntry, ...]:
        ids = [entry.candidate_id for entry in value]
        destinations = [entry.destination_path for entry in value]
        if len(ids) != len(set(ids)):
            raise ValueError("reserve activation repeats candidate IDs")
        if len(destinations) != len(set(destinations)):
            raise ValueError("reserve activation repeats destinations")
        ordered = tuple(
            sorted(
                value,
                key=lambda entry: (
                    entry.source_id,
                    entry.destination_path,
                    entry.candidate_id,
                ),
            )
        )
        if value != ordered:
            raise ValueError("reserve activation entries are not in canonical order")
        return value

    def model_post_init(self, __context: Any) -> None:
        expected = self.baseline_component_count + sum(
            entry.predicted_component_delta for entry in self.selections
        )
        if self.predicted_component_count != expected:
            raise ValueError(
                "predicted_component_count does not match activation deltas"
            )
        if self.predicted_component_count < (
            self.required_component_floor + self.closure_margin
        ):
            raise ValueError(
                "component-floor closure does not meet the required margin"
            )


class PortfolioCoreCapabilityBindingPlanEntry(_StrictModel):
    """One fixed v8 semantic expansion bound to an existing v7 asset."""

    binding_profile: CapabilityBindingPlanProfile
    candidate_id: str
    candidate_sha256: str
    source_id: str
    source_record_id: str
    destination_path: str
    asset_id: str
    component_id: str
    expected_sha256: str

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        value = _clean_text(value, "candidate_id")
        if not _CANDIDATE_ID_RE.fullmatch(value):
            raise ValueError("candidate_id has an invalid format")
        return value

    @field_validator("candidate_sha256", "expected_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        value = _clean_text(value, "source_id")
        if not _SOURCE_ID_RE.fullmatch(value):
            raise ValueError("source_id has an invalid format")
        return value

    @field_validator("source_record_id", "asset_id", "component_id")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return _clean_text(value, info.field_name)

    @field_validator("destination_path")
    @classmethod
    def validate_destination_path(cls, value: str) -> str:
        return _canonical_relative_path(value, "destination_path")

    def model_post_init(self, __context: Any) -> None:
        required_source = {
            "abo_product_identity_style_knowledge_triplet": "abo",
            "isia_food_recipe_knowledge_pair": "isia_food500",
        }[self.binding_profile]
        if self.source_id != required_source:
            raise ValueError("binding_profile does not match the approved source role")


class PortfolioCoreCapabilityBindingPlanManifest(_StrictModel):
    """Create-only v8 plan for two bounded cross-capability binding packs.

    The plan deliberately does not accept free-form capabilities.  Its two
    profile literals are the whole authority surface: a catalog-bound ABO
    product image may receive the fixed exact/style/knowledge triplet, and a
    catalog-bound ISIA food image may receive the fixed recipe/knowledge pair.
    """

    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-v8-capability-binding-v1"] = (
        PORTFOLIO_CORE_V8_CAPABILITY_BINDING_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    formal_status: Literal["non_formal"] = "non_formal"
    parent_inventory_sha256: str
    parent_selection_manifest_sha256: str
    parent_catalog_sha256: str
    parent_catalog_manifest_sha256: str
    parent_assignment_sha256: str
    parent_r1_core_plan_sha256: str
    dev_prefix_query_count: Literal[200] = 200
    asset_count: int = Field(gt=0)
    component_count: int = Field(gt=0)
    baseline_global_abo_triplet_component_count: int = Field(ge=0)
    baseline_dev_abo_triplet_component_count: int = Field(ge=0)
    baseline_non_dev_abo_triplet_component_count: int = Field(ge=0)
    target_non_dev_abo_triplet_component_count: Literal[20] = 20
    baseline_global_isia_food_pair_component_count: int = Field(ge=0)
    baseline_dev_isia_food_pair_component_count: int = Field(ge=0)
    baseline_non_dev_isia_food_pair_component_count: int = Field(ge=0)
    target_non_dev_isia_food_pair_component_count: Literal[10] = 10
    selection_algorithm: Literal[
        "v8-r1-tail-singleton-component-destination-path-candidate-id-v1"
    ] = "v8-r1-tail-singleton-component-destination-path-candidate-id-v1"
    reason: str
    selections: tuple[PortfolioCoreCapabilityBindingPlanEntry, ...] = Field(
        min_length=1
    )

    @field_validator(
        "parent_inventory_sha256",
        "parent_selection_manifest_sha256",
        "parent_catalog_sha256",
        "parent_catalog_manifest_sha256",
        "parent_assignment_sha256",
        "parent_r1_core_plan_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _clean_text(value, "reason")

    @field_validator("selections", mode="before")
    @classmethod
    def coerce_selections(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("selections")
    @classmethod
    def validate_selections(
        cls,
        value: tuple[PortfolioCoreCapabilityBindingPlanEntry, ...],
    ) -> tuple[PortfolioCoreCapabilityBindingPlanEntry, ...]:
        candidate_ids = [entry.candidate_id for entry in value]
        asset_ids = [entry.asset_id for entry in value]
        component_ids = [entry.component_id for entry in value]
        destinations = [entry.destination_path for entry in value]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("binding plan repeats candidate IDs")
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("binding plan repeats asset IDs")
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("binding plan repeats component IDs")
        if len(destinations) != len(set(destinations)):
            raise ValueError("binding plan repeats destinations")
        ordered = tuple(
            sorted(
                value,
                key=lambda entry: (
                    entry.binding_profile,
                    entry.destination_path,
                    entry.candidate_id,
                ),
            )
        )
        if value != ordered:
            raise ValueError("binding plan entries are not in canonical order")
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.baseline_global_abo_triplet_component_count != (
            self.baseline_dev_abo_triplet_component_count
            + self.baseline_non_dev_abo_triplet_component_count
        ):
            raise ValueError("ABO dev/non-dev baseline counts do not add up")
        if self.baseline_global_isia_food_pair_component_count != (
            self.baseline_dev_isia_food_pair_component_count
            + self.baseline_non_dev_isia_food_pair_component_count
        ):
            raise ValueError("ISIA dev/non-dev baseline counts do not add up")
        profile_counts = Counter(entry.binding_profile for entry in self.selections)
        if profile_counts != Counter(
            {
                "abo_product_identity_style_knowledge_triplet": (
                    CORE_R2_V8_ABO_TRIPLET_ADDITION_COUNT
                ),
                "isia_food_recipe_knowledge_pair": (
                    CORE_R2_V8_ISIA_FOOD_PAIR_ADDITION_COUNT
                ),
            }
        ):
            raise ValueError("binding plan does not contain the fixed v8 profiles")
        if (
            self.baseline_non_dev_abo_triplet_component_count
            + profile_counts["abo_product_identity_style_knowledge_triplet"]
            != self.target_non_dev_abo_triplet_component_count
        ):
            raise ValueError("ABO plan does not reach the non-dev triplet target")
        if (
            self.baseline_non_dev_isia_food_pair_component_count
            + profile_counts["isia_food_recipe_knowledge_pair"]
            != self.target_non_dev_isia_food_pair_component_count
        ):
            raise ValueError("ISIA plan does not reach the non-dev pair target")


class PortfolioCoreV9MultiClosureInventoryManifest(_StrictModel):
    """Immutable lineage for the bounded RPC Multi capacity closure.

    The v9 inventory is intentionally a merge, rather than a mutation of the
    v6 inventory: the original 105 selected RPC records survive unchanged,
    four recorded reserves are promoted by the deterministic adapter rerun,
    and only the next sixteen fresh records become selectable.  The remaining
    four staged records stay reserves for a later, separately approved run.
    """

    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-v9-rpc-multi-closure-v1"] = (
        PORTFOLIO_CORE_V9_MULTI_CLOSURE_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    formal_status: Literal["non_formal"] = "non_formal"
    parent_inventory_sha256: str
    parent_reserve_activation_manifest_sha256: str
    rebased_reserve_activation_manifest_sha256: str
    parent_v8_assignment_sha256: str
    rpc_staging_candidates_sha256: str
    rpc_staging_adapter_manifest_sha256: str
    source_policy_sha256: str
    staging_source_policy_sha256: str
    source_lock_sha256: str
    merged_inventory_sha256: str
    parent_v8_assignment_count: int = Field(gt=0)
    parent_rpc_candidate_count: Literal[109] = 109
    staged_rpc_candidate_count: Literal[129] = 129
    promoted_reserve_candidate_ids: tuple[str, ...] = Field(min_length=4)
    fresh_candidate_ids: tuple[str, ...] = Field(min_length=16)
    remaining_reserve_candidate_ids: tuple[str, ...] = Field(min_length=4)
    selection_delta_candidate_ids: tuple[str, ...] = Field(min_length=20)
    target_new_nondev_multi_component_count: Literal[20] = (
        CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT
    )
    reason: str

    @field_validator(
        "parent_inventory_sha256",
        "parent_reserve_activation_manifest_sha256",
        "rebased_reserve_activation_manifest_sha256",
        "parent_v8_assignment_sha256",
        "rpc_staging_candidates_sha256",
        "rpc_staging_adapter_manifest_sha256",
        "source_policy_sha256",
        "staging_source_policy_sha256",
        "source_lock_sha256",
        "merged_inventory_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator(
        "promoted_reserve_candidate_ids",
        "fresh_candidate_ids",
        "remaining_reserve_candidate_ids",
        "selection_delta_candidate_ids",
        mode="before",
    )
    @classmethod
    def coerce_candidate_ids(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "promoted_reserve_candidate_ids",
        "fresh_candidate_ids",
        "remaining_reserve_candidate_ids",
        "selection_delta_candidate_ids",
    )
    @classmethod
    def validate_candidate_ids(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} repeats candidate IDs")
        if tuple(sorted(value)) != value:
            raise ValueError(f"{info.field_name} is not canonically ordered")
        for candidate_id in value:
            if not _CANDIDATE_ID_RE.fullmatch(
                _clean_text(candidate_id, info.field_name)
            ):
                raise ValueError(f"{info.field_name} has an invalid candidate ID")
        return value

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _clean_text(value, "reason")

    def model_post_init(self, __context: Any) -> None:
        if len(self.promoted_reserve_candidate_ids) != 4:
            raise ValueError("v9 closure must promote exactly four RPC reserves")
        if len(self.fresh_candidate_ids) != CORE_R2_V9_FRESH_RPC_MULTI_COUNT:
            raise ValueError(
                "v9 closure must contain exactly sixteen fresh RPC candidates"
            )
        if len(self.remaining_reserve_candidate_ids) != 4:
            raise ValueError("v9 closure must retain exactly four RPC reserves")
        expected_delta = set(self.promoted_reserve_candidate_ids) | set(
            self.fresh_candidate_ids
        )
        if len(expected_delta) != CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT:
            raise ValueError("v9 closure selection delta has duplicate candidates")
        if tuple(sorted(expected_delta)) != self.selection_delta_candidate_ids:
            raise ValueError(
                "v9 closure selection delta does not bind promoted and fresh candidates"
            )
        if expected_delta & set(self.remaining_reserve_candidate_ids):
            raise ValueError("v9 closure promotes a candidate retained as reserve")


class PortfolioCoreV9MultiClosureCatalogReceipt(_StrictModel):
    """Post-catalog evidence that the v9 RPC closure added usable components."""

    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-v9-rpc-multi-closure-v1"] = (
        PORTFOLIO_CORE_V9_MULTI_CLOSURE_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    formal_status: Literal["non_formal"] = "non_formal"
    closure_inventory_manifest_sha256: str
    parent_inventory_sha256: str
    parent_selection_manifest_sha256: str
    parent_catalog_sha256: str
    parent_catalog_manifest_sha256: str
    parent_v8_assignment_sha256: str
    parent_r1_core_plan_sha256: str
    v9_inventory_sha256: str
    v9_selection_manifest_sha256: str
    v9_catalog_sha256: str
    v9_catalog_manifest_sha256: str
    parent_v8_assignment_count: int = Field(gt=0)
    parent_asset_count: int = Field(gt=0)
    parent_component_count: int = Field(gt=0)
    v9_asset_count: int = Field(gt=0)
    v9_component_count: int = Field(gt=0)
    component_count_delta: int = Field(ge=0)
    dev_prefix_query_count: Literal[200] = 200
    selection_delta_candidate_ids: tuple[str, ...] = Field(min_length=20)
    selection_delta_asset_ids: tuple[str, ...] = Field(min_length=20)
    new_multi_component_ids: tuple[str, ...] = Field(min_length=20)
    new_nondev_multi_component_ids: tuple[str, ...] = Field(min_length=20)
    minimum_new_nondev_multi_component_count: Literal[20] = (
        CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT
    )

    @field_validator(
        "closure_inventory_manifest_sha256",
        "parent_inventory_sha256",
        "parent_selection_manifest_sha256",
        "parent_catalog_sha256",
        "parent_catalog_manifest_sha256",
        "parent_v8_assignment_sha256",
        "parent_r1_core_plan_sha256",
        "v9_inventory_sha256",
        "v9_selection_manifest_sha256",
        "v9_catalog_sha256",
        "v9_catalog_manifest_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator(
        "selection_delta_candidate_ids",
        "selection_delta_asset_ids",
        "new_multi_component_ids",
        "new_nondev_multi_component_ids",
        mode="before",
    )
    @classmethod
    def coerce_identifiers(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "selection_delta_candidate_ids",
        "selection_delta_asset_ids",
        "new_multi_component_ids",
        "new_nondev_multi_component_ids",
    )
    @classmethod
    def validate_identifiers(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} repeats identifiers")
        if tuple(sorted(value)) != value:
            raise ValueError(f"{info.field_name} is not canonically ordered")
        if any(not _clean_text(item, info.field_name) for item in value):
            raise ValueError(f"{info.field_name} has a blank identifier")
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.component_count_delta != (
            self.v9_component_count - self.parent_component_count
        ):
            raise ValueError("v9 component count delta is inconsistent")
        if self.v9_asset_count != (
            self.parent_asset_count + len(self.selection_delta_asset_ids)
        ):
            raise ValueError("v9 asset count does not match the closure delta")
        if len(self.selection_delta_candidate_ids) != (
            CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT
        ):
            raise ValueError(
                "v9 receipt must contain exactly twenty selected candidates"
            )
        if len(self.selection_delta_asset_ids) != (
            CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT
        ):
            raise ValueError("v9 receipt must contain exactly twenty selected assets")
        if not set(self.new_nondev_multi_component_ids) <= set(
            self.new_multi_component_ids
        ):
            raise ValueError(
                "non-dev components must be a subset of new multi components"
            )
        if len(self.new_nondev_multi_component_ids) < (
            CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT
        ):
            raise ValueError("v9 receipt does not meet the new non-dev Multi floor")
        if self.component_count_delta < CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT:
            raise ValueError("v9 receipt does not meet the component delta floor")


class PortfolioCoreSourcePolicy(_StrictModel):
    source_id: str
    availability: SourceAvailability
    allowed_pools: tuple[Pool, ...]
    note: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        value = _clean_text(value, "source_id")
        if not _SOURCE_ID_RE.fullmatch(value):
            raise ValueError("source_id has an invalid format")
        return value

    @field_validator("allowed_pools", mode="before")
    @classmethod
    def coerce_allowed_pools(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("allowed_pools")
    @classmethod
    def validate_allowed_pools(cls, value: tuple[Pool, ...]) -> tuple[Pool, ...]:
        if value != tuple(sorted(set(value), key=_POOL_ORDER.index)):
            raise ValueError("allowed_pools must be unique and use policy order")
        return value

    @field_validator("note")
    @classmethod
    def validate_note(cls, value: str) -> str:
        return _clean_text(value, "note")

    def model_post_init(self, __context: Any) -> None:
        if (
            self.availability
            in {
                "restricted",
                "metadata_only",
                "permanently_unavailable",
            }
            and self.allowed_pools
        ):
            raise ValueError(
                "restricted, metadata-only, or permanently unavailable sources must not expose image pools"
            )


class PortfolioCoreSourcePolicyDocument(_StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal[
        "portfolio-core-source-pool-policy-v2",
        "portfolio-core-source-pool-policy-v3",
    ] = PORTFOLIO_CORE_SOURCE_POLICY_VERSION
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    formal_status: Literal["non_formal"] = "non_formal"
    sources: tuple[PortfolioCoreSourcePolicy, ...] = Field(min_length=1)

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("sources")
    @classmethod
    def validate_sources(
        cls, value: tuple[PortfolioCoreSourcePolicy, ...]
    ) -> tuple[PortfolioCoreSourcePolicy, ...]:
        if len({source.source_id for source in value}) != len(value):
            raise ValueError("source policy contains duplicate source_id")
        return tuple(sorted(value, key=lambda item: item.source_id))


DEFAULT_PORTFOLIO_CORE_SOURCE_POLICY = PortfolioCoreSourcePolicyDocument(
    sources=(
        PortfolioCoreSourcePolicy(
            source_id="abo",
            availability="ready",
            allowed_pools=("exact_match",),
            note="Reusable dev_mini exact-match source; preserve the existing source exclusions.",
        ),
        PortfolioCoreSourcePolicy(
            source_id="csds",
            availability="metadata_only",
            allowed_pools=(),
            note="Dialogue metadata only; it is not an image materialization pool.",
        ),
        PortfolioCoreSourcePolicy(
            source_id="cord",
            availability="ready",
            allowed_pools=("utility",),
            note=(
                "Fixed local CORD v2 Parquet image bytes for document-reading "
                "inputs only; do not claim its OCR/KIE labels as task gold here."
            ),
        ),
        PortfolioCoreSourcePolicy(
            source_id="deepfashion",
            availability="pending_extraction",
            allowed_pools=(),
            note=(
                "Candidate-gallery/same-item-exclusion only; wait for verified "
                "extraction and a separate source-specific selection policy."
            ),
        ),
        PortfolioCoreSourcePolicy(
            source_id="fashioniq",
            availability="ready",
            allowed_pools=("divergent_rec",),
            note="Reusable dev_mini relative-language style source; never use it as exact-match gold.",
        ),
        PortfolioCoreSourcePolicy(
            source_id="inaturalist",
            availability="ready",
            allowed_pools=("encyclopedia",),
            note="Reusable challenge image source for declared encyclopedia candidates, not a general primary claim.",
        ),
        PortfolioCoreSourcePolicy(
            source_id="isia_food500",
            availability="ready",
            allowed_pools=("utility",),
            note=(
                "Recipe-guidance images only; this source cannot be broadened "
                "to retrieval, style, encyclopedia, or document-reading roles."
            ),
        ),
        PortfolioCoreSourcePolicy(
            source_id="mep3m",
            availability="restricted",
            allowed_pools=(),
            note=(
                "Long-tail/hard-negative candidate only; it must not be promoted "
                "to a generic positive retrieval pool by this workflow."
            ),
        ),
        PortfolioCoreSourcePolicy(
            source_id="polyvore",
            availability="ready",
            allowed_pools=("divergent_rec",),
            note=(
                "Use the owner-selected local image-bearing snapshot only for the "
                "declared outfit-compatibility/divergent inventory."
            ),
        ),
        PortfolioCoreSourcePolicy(
            source_id="products_10k",
            availability="locked_archive",
            allowed_pools=("exact_match",),
            note=(
                "Exact-match extension only; the verified multi-volume archive "
                "remains locked until an explicit extraction/materialization step."
            ),
        ),
        PortfolioCoreSourcePolicy(
            source_id="recipe1m_plus",
            availability="ready",
            allowed_pools=("utility",),
            note="Use only the owner-approved bounded selected-image inventory.",
        ),
        PortfolioCoreSourcePolicy(
            source_id="rpc",
            availability="ready",
            allowed_pools=("multi_product",),
            note="Reusable dev_mini multi-product source; preserve its SKU and object-level contracts.",
        ),
        PortfolioCoreSourcePolicy(
            source_id="sroie",
            availability="ready",
            allowed_pools=("utility",),
            note=(
                "Owner-acquired official image-only SROIE package for document "
                "inputs; license/use-scope disclosure remains pending."
            ),
        ),
        PortfolioCoreSourcePolicy(
            source_id="simmc_2_1",
            availability="metadata_only",
            allowed_pools=(),
            note="Dialogue data only; it is not an image materialization pool.",
        ),
        PortfolioCoreSourcePolicy(
            source_id="sku_110k",
            availability="restricted",
            allowed_pools=(),
            note=(
                "Dense-detection challenge only; it must not be converted into "
                "generic exact or multi-product retrieval positives here."
            ),
        ),
        PortfolioCoreSourcePolicy(
            source_id="u_need",
            availability="permanently_unavailable",
            allowed_pools=(),
            note="Permanently unavailable by owner decision; never plan or materialize it.",
        ),
        PortfolioCoreSourcePolicy(
            source_id="wikimedia_commons_documents",
            availability="ready",
            allowed_pools=("utility",),
            note="Reusable document-layout challenge source; do not claim field-level OCR/KIE gold from this policy.",
        ),
    )
)


class PortfolioCoreAssetRunManifest(_StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-asset-preparation-v1"] = (
        PORTFOLIO_CORE_ASSET_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    formal_status: Literal["non_formal"] = "non_formal"
    inventory_sha256: str
    source_policy_sha256: str
    selected_candidate_count: int = Field(ge=0)
    selected_source_counts: dict[str, int]

    @field_validator("inventory_sha256", "source_policy_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)


class PortfolioCoreAssetCheckpoint(_StrictModel):
    schema_version: Literal[1] = 1
    candidate_id: str
    candidate_sha256: str
    destination_path: str
    expected_bytes: int = Field(gt=0)
    expected_sha256: str
    draft: DatasetAssetDraft

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        if not _CANDIDATE_ID_RE.fullmatch(_clean_text(value, "candidate_id")):
            raise ValueError("candidate_id has an invalid format")
        return value

    @field_validator("candidate_sha256", "expected_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("destination_path")
    @classmethod
    def validate_destination_path(cls, value: str) -> str:
        return _canonical_relative_path(value, "destination_path")


class PortfolioCoreSelectionRow(_StrictModel):
    candidate_id: str
    candidate_sha256: str
    destination_path: str
    expected_bytes: int = Field(gt=0)
    expected_sha256: str
    source_id: str
    pool: Pool
    draft: DatasetAssetDraft
    capability_bindings: tuple[CandidateCapabilityBinding, ...]

    @field_validator("candidate_id", "source_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        pattern = (
            _CANDIDATE_ID_RE if info.field_name == "candidate_id" else _SOURCE_ID_RE
        )
        value = _clean_text(value, info.field_name)
        if not pattern.fullmatch(value):
            raise ValueError(f"{info.field_name} has an invalid format")
        return value

    @field_validator("candidate_sha256", "expected_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("destination_path")
    @classmethod
    def validate_destination_path(cls, value: str) -> str:
        return _canonical_relative_path(value, "destination_path")

    @field_validator("capability_bindings", mode="before")
    @classmethod
    def coerce_bindings(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class PortfolioCoreSelectionManifest(_StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-asset-preparation-v1"] = (
        PORTFOLIO_CORE_ASSET_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    formal_status: Literal["non_formal"] = "non_formal"
    state: Literal["complete"] = "complete"
    inventory_sha256: str
    source_policy_sha256: str
    dataset_assets_sha256: str
    dataset_asset_count: int = Field(ge=0)
    selections: tuple[PortfolioCoreSelectionRow, ...]

    @field_validator(
        "inventory_sha256", "source_policy_sha256", "dataset_assets_sha256"
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("selections", mode="before")
    @classmethod
    def coerce_rows(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    def model_post_init(self, __context: Any) -> None:
        if self.dataset_asset_count != len(self.selections):
            raise ValueError("dataset_asset_count does not match selections")
        candidate_ids = [row.candidate_id for row in self.selections]
        destinations = [row.destination_path for row in self.selections]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("selection manifest has duplicate candidate ids")
        if len(destinations) != len(set(destinations)):
            raise ValueError("selection manifest has duplicate destination paths")


@dataclass(frozen=True)
class PortfolioCoreAssetPreflight:
    inventory_sha256: str
    source_policy_sha256: str
    selected_candidate_count: int
    reserve_candidate_count: int
    activated_reserve_candidate_count: int
    ready_candidate_count: int
    blocked_candidates: dict[str, tuple[str, ...]]
    source_statuses: dict[str, str]
    selected_pool_counts: dict[str, int]
    selected_unique_content_counts: dict[str, int]
    selected_capability_counts: dict[str, int]
    capacity_shortfalls: dict[str, int]
    capacity_profile: CoreCapacityProfile = "r1"
    capability_capacity_shortfalls: dict[str, int] = field(default_factory=dict)
    document_selection_shortfalls: dict[str, int] = field(default_factory=dict)
    document_selection_manifest_sha256: str | None = None
    document_selection_counts: dict[str, int] = field(default_factory=dict)
    document_selection_role_source_counts: dict[str, int] = field(default_factory=dict)
    reserve_activation_manifest_sha256: str | None = None

    @property
    def source_ready(self) -> bool:
        return not self.blocked_candidates

    @property
    def capacity_ready(self) -> bool:
        return (
            not self.capacity_shortfalls
            and not self.capability_capacity_shortfalls
            and not self.document_selection_shortfalls
        )

    @property
    def ready(self) -> bool:
        return self.source_ready and self.capacity_ready

    def as_json(self) -> dict[str, Any]:
        return {
            "blocked_candidates": {
                source: list(values)
                for source, values in sorted(self.blocked_candidates.items())
            },
            "command": "preflight",
            "activated_reserve_candidate_count": self.activated_reserve_candidate_count,
            "formal_eligible": False,
            "formal_status": "non_formal",
            "inventory_sha256": self.inventory_sha256,
            "ready": self.ready,
            "source_ready": self.source_ready,
            "capacity_ready": self.capacity_ready,
            "capacity_shortfalls": dict(sorted(self.capacity_shortfalls.items())),
            "capacity_profile": self.capacity_profile,
            "capability_capacity_shortfalls": dict(
                sorted(self.capability_capacity_shortfalls.items())
            ),
            "document_selection_manifest_sha256": self.document_selection_manifest_sha256,
            "document_selection_counts": dict(
                sorted(self.document_selection_counts.items())
            ),
            "document_selection_shortfalls": dict(
                sorted(self.document_selection_shortfalls.items())
            ),
            "document_selection_role_source_counts": dict(
                sorted(self.document_selection_role_source_counts.items())
            ),
            "minimum_pool_asset_floors": dict(
                (pool, CORE_POOL_ASSET_FLOORS[pool]) for pool in _POOL_ORDER
            ),
            "minimum_capability_asset_floors": (
                dict(sorted(CORE_R2_CAPABILITY_ASSET_FLOORS.items()))
                if self.capacity_profile == "r2"
                else {}
            ),
            "ready_candidate_count": self.ready_candidate_count,
            "reserve_candidate_count": self.reserve_candidate_count,
            "reserve_activation_manifest_sha256": self.reserve_activation_manifest_sha256,
            "selected_capability_counts": dict(
                sorted(self.selected_capability_counts.items())
            ),
            "selected_candidate_count": self.selected_candidate_count,
            "selected_pool_counts": dict(sorted(self.selected_pool_counts.items())),
            "selected_unique_content_counts": dict(
                sorted(self.selected_unique_content_counts.items())
            ),
            "source_policy_sha256": self.source_policy_sha256,
            "source_statuses": dict(sorted(self.source_statuses.items())),
            "track": "portfolio",
        }


@dataclass(frozen=True)
class PortfolioCoreAssetMaterializationResult:
    output_root: Path
    checkpointed_candidate_count: int
    copied_this_call: int
    complete: bool
    dataset_assets_path: Path | None
    selection_manifest_path: Path | None

    def as_json(self) -> dict[str, Any]:
        return {
            "checkpointed_candidate_count": self.checkpointed_candidate_count,
            "complete": self.complete,
            "copied_this_call": self.copied_this_call,
            "dataset_assets_path": (
                str(self.dataset_assets_path) if self.dataset_assets_path else None
            ),
            "formal_eligible": False,
            "formal_status": "non_formal",
            "output_root": str(self.output_root),
            "selection_manifest_path": (
                str(self.selection_manifest_path)
                if self.selection_manifest_path
                else None
            ),
            "track": "portfolio",
        }


@dataclass(frozen=True)
class PortfolioCoreAssignmentResult:
    output_path: Path
    output_sha256: str
    assignment_count: int
    selection_manifest_sha256: str
    binding_plan_manifest_sha256: str | None = None
    multi_closure_catalog_receipt_sha256: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "assignment_count": self.assignment_count,
            "assignment_sha256": self.output_sha256,
            "formal_eligible": False,
            "formal_status": "non_formal",
            "output_path": str(self.output_path),
            "binding_plan_manifest_sha256": self.binding_plan_manifest_sha256,
            "multi_closure_catalog_receipt_sha256": (
                self.multi_closure_catalog_receipt_sha256
            ),
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "track": "portfolio",
        }


@dataclass(frozen=True)
class PortfolioCoreCapabilityBindingPlanResult:
    output_path: Path
    output_sha256: str
    abo_candidate_ids: tuple[str, ...]
    isia_food_candidate_ids: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "abo_triplet_addition_count": len(self.abo_candidate_ids),
            "formal_eligible": False,
            "formal_status": "non_formal",
            "isia_food_pair_addition_count": len(self.isia_food_candidate_ids),
            "output_path": str(self.output_path),
            "output_sha256": self.output_sha256,
            "track": "portfolio",
        }


@dataclass(frozen=True)
class PortfolioCoreV9MultiClosureInventoryResult:
    inventory_path: Path
    inventory_sha256: str
    reserve_activation_manifest_path: Path
    reserve_activation_manifest_sha256: str
    manifest_path: Path
    manifest_sha256: str
    selection_delta_candidate_ids: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "formal_eligible": False,
            "formal_status": "non_formal",
            "inventory_path": str(self.inventory_path),
            "inventory_sha256": self.inventory_sha256,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "reserve_activation_manifest_path": str(
                self.reserve_activation_manifest_path
            ),
            "reserve_activation_manifest_sha256": (
                self.reserve_activation_manifest_sha256
            ),
            "selection_delta_candidate_ids": list(self.selection_delta_candidate_ids),
            "track": "portfolio",
        }


@dataclass(frozen=True)
class PortfolioCoreV9MultiClosureCatalogReceiptResult:
    output_path: Path
    output_sha256: str
    component_count_delta: int
    new_nondev_multi_component_ids: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "component_count_delta": self.component_count_delta,
            "formal_eligible": False,
            "formal_status": "non_formal",
            "new_nondev_multi_component_ids": list(self.new_nondev_multi_component_ids),
            "output_path": str(self.output_path),
            "output_sha256": self.output_sha256,
            "track": "portfolio",
        }


def default_source_policy_bytes() -> bytes:
    """Return the create-only policy snapshot used to start a core selection."""

    return canonical_json_bytes(DEFAULT_PORTFOLIO_CORE_SOURCE_POLICY)


def write_default_source_policy(path: str | Path) -> Path:
    """Create the deliberately conservative default source/pool policy."""

    return _publish_idempotent_file(Path(path), default_source_policy_bytes())


def load_source_policy(
    path: str | Path,
) -> tuple[PortfolioCoreSourcePolicyDocument, str]:
    content = _read_metadata(Path(path), "Portfolio core source policy")
    try:
        policy = PortfolioCoreSourcePolicyDocument.model_validate_json(content)
    except ValidationError as exc:
        raise PortfolioCoreAssetError("source policy violates schema") from exc
    if content != canonical_json_bytes(policy):
        raise PortfolioCoreAssetError("source policy must be canonical JSON")
    _validate_source_policy_ceiling(policy)
    return policy, sha256_bytes(content)


def load_candidate_inventory(
    path: str | Path,
) -> tuple[tuple[PortfolioCoreAssetCandidate, ...], str]:
    """Load a canonical source-specific candidate inventory without inference."""

    content = _read_metadata(Path(path), "Portfolio core candidate inventory")
    try:
        rows = parse_canonical_jsonl(
            content, label="Portfolio core candidate inventory"
        )
    except ArtifactFormatError as exc:
        raise PortfolioCoreAssetError(str(exc)) from exc
    if not rows:
        raise PortfolioCoreAssetError(
            "candidate inventory must contain at least one row"
        )
    candidates: list[PortfolioCoreAssetCandidate] = []
    for line_number, row in enumerate(rows, start=1):
        try:
            candidates.append(
                PortfolioCoreAssetCandidate.model_validate(row, strict=True)
            )
        except ValidationError as exc:
            raise PortfolioCoreAssetError(
                f"candidate inventory line {line_number} violates schema"
            ) from exc
    if content != canonical_jsonl_bytes(candidates):
        raise PortfolioCoreAssetError("candidate inventory must be canonical JSONL")
    _validate_candidate_set(candidates)
    return tuple(candidates), sha256_bytes(content)


def load_portfolio_core_document_selection_manifest(
    path: str | Path,
) -> tuple[PortfolioCoreDocumentSelectionManifest, str]:
    """Load the explicit r2 document carry-forward/fresh selection manifest."""

    content = _read_metadata(Path(path), "Portfolio core document selection manifest")
    try:
        value = parse_canonical_json(
            content,
            label="Portfolio core document selection manifest",
        )
        manifest = PortfolioCoreDocumentSelectionManifest.model_validate(
            value,
            strict=True,
        )
    except (ArtifactFormatError, ValidationError) as exc:
        raise PortfolioCoreAssetError(
            "document selection manifest violates schema"
        ) from exc
    if content != canonical_json_bytes(manifest):
        raise PortfolioCoreAssetError(
            "document selection manifest must be canonical JSON"
        )
    return manifest, sha256_bytes(content)


def load_portfolio_core_reserve_activation_manifest(
    path: str | Path,
) -> tuple[PortfolioCoreReserveActivationManifest, str]:
    """Load the canonical reserve-to-planner activation sidecar."""

    content = _read_metadata(Path(path), "Portfolio core reserve activation manifest")
    try:
        value = parse_canonical_json(
            content,
            label="Portfolio core reserve activation manifest",
        )
        manifest = PortfolioCoreReserveActivationManifest.model_validate(
            value,
            strict=True,
        )
    except (ArtifactFormatError, ValidationError) as exc:
        raise PortfolioCoreAssetError(
            "reserve activation manifest violates schema"
        ) from exc
    if content != canonical_json_bytes(manifest):
        raise PortfolioCoreAssetError(
            "reserve activation manifest must be canonical JSON"
        )
    return manifest, sha256_bytes(content)


def load_portfolio_core_capability_binding_plan_manifest(
    path: str | Path,
) -> tuple[PortfolioCoreCapabilityBindingPlanManifest, str]:
    """Load the fixed, catalog-bound v8 semantic binding plan."""

    content = _read_metadata(Path(path), "Portfolio core capability binding plan")
    try:
        value = parse_canonical_json(
            content,
            label="Portfolio core capability binding plan",
        )
        manifest = PortfolioCoreCapabilityBindingPlanManifest.model_validate(
            value,
            strict=True,
        )
    except (ArtifactFormatError, ValidationError) as exc:
        raise PortfolioCoreAssetError(
            "capability binding plan violates schema"
        ) from exc
    if content != canonical_json_bytes(manifest):
        raise PortfolioCoreAssetError("capability binding plan must be canonical JSON")
    return manifest, sha256_bytes(content)


def load_portfolio_core_v9_multi_closure_inventory_manifest(
    path: str | Path,
) -> tuple[PortfolioCoreV9MultiClosureInventoryManifest, str]:
    """Load the canonical v9 RPC inventory lineage sidecar."""

    content = _read_metadata(Path(path), "Portfolio core v9 Multi closure inventory")
    try:
        value = parse_canonical_json(
            content,
            label="Portfolio core v9 Multi closure inventory",
        )
        manifest = PortfolioCoreV9MultiClosureInventoryManifest.model_validate(
            value,
            strict=True,
        )
    except (ArtifactFormatError, ValidationError) as exc:
        raise PortfolioCoreAssetError(
            "v9 Multi closure inventory manifest violates schema"
        ) from exc
    if content != canonical_json_bytes(manifest):
        raise PortfolioCoreAssetError(
            "v9 Multi closure inventory manifest must be canonical JSON"
        )
    return manifest, sha256_bytes(content)


def load_portfolio_core_v9_multi_closure_catalog_receipt(
    path: str | Path,
) -> tuple[PortfolioCoreV9MultiClosureCatalogReceipt, str]:
    """Load the canonical post-catalog v9 RPC component receipt."""

    content = _read_metadata(Path(path), "Portfolio core v9 Multi closure receipt")
    try:
        value = parse_canonical_json(
            content,
            label="Portfolio core v9 Multi closure receipt",
        )
        receipt = PortfolioCoreV9MultiClosureCatalogReceipt.model_validate(
            value,
            strict=True,
        )
    except (ArtifactFormatError, ValidationError) as exc:
        raise PortfolioCoreAssetError(
            "v9 Multi closure catalog receipt violates schema"
        ) from exc
    if content != canonical_json_bytes(receipt):
        raise PortfolioCoreAssetError(
            "v9 Multi closure catalog receipt must be canonical JSON"
        )
    return receipt, sha256_bytes(content)


def _resolve_activated_reserve_candidate_ids(
    *,
    candidates: Sequence[PortfolioCoreAssetCandidate],
    inventory_sha256: str,
    reserve_activation_manifest_path: str | Path | None,
) -> tuple[frozenset[str], str | None]:
    if reserve_activation_manifest_path is None:
        return frozenset(), None
    manifest, manifest_sha256 = load_portfolio_core_reserve_activation_manifest(
        reserve_activation_manifest_path
    )
    if manifest.parent_inventory_sha256 != inventory_sha256:
        raise PortfolioCoreAssetError(
            "reserve activation manifest does not bind the candidate inventory"
        )
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    activated: set[str] = set()
    for entry in manifest.selections:
        candidate = candidates_by_id.get(entry.candidate_id)
        if candidate is None:
            raise PortfolioCoreAssetError(
                "reserve activation candidate is absent from the inventory: "
                + entry.candidate_id
            )
        if candidate.selection != "reserve":
            raise PortfolioCoreAssetError(
                "reserve activation candidate is not reserve: " + entry.candidate_id
            )
        if (
            candidate.source_id != entry.source_id
            or candidate.destination_path != entry.destination_path
            or candidate.expected_sha256 != entry.expected_sha256
        ):
            raise PortfolioCoreAssetError(
                "reserve activation candidate binding drifted: " + entry.candidate_id
            )
        ceiling = _SOURCE_CAPABILITY_CEILINGS.get(candidate.source_id)
        requested = {
            (binding.canonical_intent, binding.canonical_capability)
            for binding in candidate.capability_bindings
        }
        if ceiling is None or not requested <= ceiling:
            raise PortfolioCoreAssetError(
                "reserve activation candidate broadens the source role: "
                + entry.candidate_id
            )
        activated.add(candidate.candidate_id)
    return frozenset(activated), manifest_sha256


def preflight_portfolio_core_assets(
    *,
    inventory_path: str | Path,
    source_policy_path: str | Path,
    source_roots: Mapping[str, str | Path],
    capacity_profile: CoreCapacityProfile = "r1",
    document_selection_manifest_path: str | Path | None = None,
    reserve_activation_manifest_path: str | Path | None = None,
) -> PortfolioCoreAssetPreflight:
    """Check declared candidates without copying or changing raw source bytes.

    Pending, locked, metadata-only, and permanently unavailable sources appear
    in the report as blockers.  This lets a source-specific inventory be
    written early without accidentally treating a raw archive as a usable
    clean image pool.
    """

    candidates, inventory_sha256 = load_candidate_inventory(inventory_path)
    activated_reserve_ids, reserve_activation_manifest_sha256 = (
        _resolve_activated_reserve_candidate_ids(
            candidates=candidates,
            inventory_sha256=inventory_sha256,
            reserve_activation_manifest_path=reserve_activation_manifest_path,
        )
    )
    if capacity_profile not in {"r1", "r2"}:
        raise PortfolioCoreAssetError("capacity_profile must be one of: r1, r2")
    if capacity_profile == "r1" and document_selection_manifest_path is not None:
        raise PortfolioCoreAssetError(
            "document selection manifest is only valid for the r2 capacity profile"
        )

    document_manifest: PortfolioCoreDocumentSelectionManifest | None = None
    document_manifest_sha256: str | None = None
    document_selection_counts: Counter[str] = Counter()
    document_selection_role_source_counts: Counter[str] = Counter()
    document_selection_shortfalls: dict[str, int] = {}
    if capacity_profile == "r2":
        if document_selection_manifest_path is None:
            raise PortfolioCoreAssetError(
                "r2 capacity preflight requires a document selection manifest"
            )
        document_manifest, document_manifest_sha256 = (
            load_portfolio_core_document_selection_manifest(
                document_selection_manifest_path
            )
        )
        (
            document_selection_counts,
            document_selection_role_source_counts,
        ) = _validate_document_selection_against_candidates(
            document_manifest, candidates
        )
        document_selection_shortfalls = {
            "carry_forward": CORE_R2_DOCUMENT_CARRY_FORWARD_FLOOR
            - document_selection_counts["carry_forward"],
            "fresh": CORE_R2_DOCUMENT_FRESH_FLOOR
            - (
                document_selection_counts["fresh_cord"]
                + document_selection_counts["fresh_sroie"]
            ),
            "fresh_cord": CORE_R2_DOCUMENT_CORD_FRESH_FLOOR
            - document_selection_counts["fresh_cord"],
            "fresh_sroie": CORE_R2_DOCUMENT_SROIE_FRESH_FLOOR
            - document_selection_counts["fresh_sroie"],
            "reserve_cord": CORE_R2_DOCUMENT_CORD_RESERVE_FLOOR
            - document_selection_role_source_counts["cord:reserve"],
            "reserve_sroie": CORE_R2_DOCUMENT_SROIE_RESERVE_FLOOR
            - document_selection_role_source_counts["sroie:reserve"],
        }
        document_selection_shortfalls = {
            role: missing
            for role, missing in document_selection_shortfalls.items()
            if missing > 0
        }

    policy, source_policy_sha256 = load_source_policy(source_policy_path)
    policy_by_source = {entry.source_id: entry for entry in policy.sources}
    normalized_roots = _normalize_source_roots(source_roots)

    blocked: dict[str, list[str]] = defaultdict(list)
    pool_counts: Counter[str] = Counter()
    unique_hashes_by_pool: dict[str, set[str]] = defaultdict(set)
    capability_counts: Counter[str] = Counter()
    unique_hashes_by_capability: dict[str, set[str]] = defaultdict(set)
    ready_count = 0
    for candidate in candidates:
        if (
            candidate.selection != "include"
            and candidate.candidate_id not in activated_reserve_ids
        ):
            continue
        pool_counts[candidate.pool] += 1
        unique_hashes_by_pool[candidate.pool].add(candidate.expected_sha256)
        for binding in candidate.capability_bindings:
            capability_counts[binding.canonical_capability] += 1
            unique_hashes_by_capability[binding.canonical_capability].add(
                candidate.expected_sha256
            )
        source = policy_by_source.get(candidate.source_id)
        if source is None:
            blocked[candidate.source_id].append("source_not_registered_in_policy")
            continue
        if source.availability != "ready":
            blocked[candidate.source_id].append(source.availability)
            continue
        if candidate.pool not in source.allowed_pools:
            blocked[candidate.source_id].append(f"pool_not_allowed:{candidate.pool}")
            continue
        root = normalized_roots.get(candidate.source_id)
        if root is None:
            blocked[candidate.source_id].append("source_root_missing")
            continue
        try:
            content = _read_and_verify_source_candidate(candidate, root)
        except PortfolioCoreAssetError as exc:
            blocked[candidate.source_id].append(str(exc))
            continue
        if len(content) != candidate.expected_bytes:  # defensive, helper checks too
            blocked[candidate.source_id].append("source_bytes_mismatch")
            continue
        ready_count += 1

    capacity_shortfalls = {
        pool: floor - len(unique_hashes_by_pool.get(pool, set()))
        for pool, floor in CORE_POOL_ASSET_FLOORS.items()
        if len(unique_hashes_by_pool.get(pool, set())) < floor
    }
    capability_capacity_shortfalls = (
        {
            capability: floor - len(unique_hashes_by_capability.get(capability, set()))
            for capability, floor in CORE_R2_CAPABILITY_ASSET_FLOORS.items()
            if len(unique_hashes_by_capability.get(capability, set())) < floor
        }
        if capacity_profile == "r2"
        else {}
    )

    return PortfolioCoreAssetPreflight(
        inventory_sha256=inventory_sha256,
        source_policy_sha256=source_policy_sha256,
        selected_candidate_count=sum(
            item.selection == "include" or item.candidate_id in activated_reserve_ids
            for item in candidates
        ),
        reserve_candidate_count=sum(
            item.selection == "reserve"
            and item.candidate_id not in activated_reserve_ids
            for item in candidates
        ),
        activated_reserve_candidate_count=len(activated_reserve_ids),
        ready_candidate_count=ready_count,
        blocked_candidates={
            source: tuple(sorted(set(reasons))) for source, reasons in blocked.items()
        },
        source_statuses={
            entry.source_id: entry.availability for entry in policy.sources
        },
        selected_pool_counts=dict(sorted(pool_counts.items())),
        selected_unique_content_counts={
            pool: len(hashes) for pool, hashes in sorted(unique_hashes_by_pool.items())
        },
        selected_capability_counts=dict(sorted(capability_counts.items())),
        capacity_shortfalls=dict(sorted(capacity_shortfalls.items())),
        capacity_profile=capacity_profile,
        capability_capacity_shortfalls=dict(
            sorted(capability_capacity_shortfalls.items())
        ),
        document_selection_shortfalls=dict(
            sorted(document_selection_shortfalls.items())
        ),
        document_selection_manifest_sha256=document_manifest_sha256,
        document_selection_counts=dict(sorted(document_selection_counts.items())),
        document_selection_role_source_counts=dict(
            sorted(document_selection_role_source_counts.items())
        ),
        reserve_activation_manifest_sha256=reserve_activation_manifest_sha256,
    )


def _validate_document_selection_against_candidates(
    manifest: PortfolioCoreDocumentSelectionManifest,
    candidates: Sequence[PortfolioCoreAssetCandidate],
) -> tuple[Counter[str], Counter[str]]:
    """Verify r2 document roles against the inventory being preflighted.

    ``inventory_sha256`` binds the document adapter's direct output.  The
    later merged r2 inventory has a different whole-file digest, so this
    check instead requires every declared document binding to survive the
    merge unchanged and rejects undeclared document candidates.
    """

    documents = {
        candidate.candidate_id: candidate
        for candidate in candidates
        if candidate.source_id in _DOCUMENT_SOURCE_IDS
    }
    manifest_ids = {row.candidate_id for row in manifest.selections}
    if manifest_ids != set(documents):
        missing = sorted(manifest_ids - set(documents))
        undeclared = sorted(set(documents) - manifest_ids)
        raise PortfolioCoreAssetError(
            "document selection manifest does not match inventory "
            + json.dumps(
                {"missing_candidates": missing, "undeclared_candidates": undeclared},
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    allowed_role_sources = {
        "carry_forward": {"wikimedia_commons_documents"},
        "fresh": {"wikimedia_commons_documents"},
        "fresh_cord": {"cord"},
        "fresh_sroie": {"sroie"},
        "reserve": _DOCUMENT_SOURCE_IDS,
    }
    counts: Counter[str] = Counter()
    role_source_counts: Counter[str] = Counter()
    for row in manifest.selections:
        candidate = documents[row.candidate_id]
        if (
            candidate.source_id != row.source_id
            or candidate.source_id not in allowed_role_sources[row.selection_role]
            or candidate.draft.source_record_id != row.source_record_id
            or candidate.expected_sha256 != row.content_sha256
        ):
            raise PortfolioCoreAssetError(
                "document selection binding drifted for candidate " + row.candidate_id
            )
        expected_selection = "reserve" if row.selection_role == "reserve" else "include"
        if candidate.selection != expected_selection:
            raise PortfolioCoreAssetError(
                "document selection role disagrees with inventory selection for "
                + row.candidate_id
            )
        counts[row.selection_role] += 1
        role_source_counts[f"{row.source_id}:{row.selection_role}"] += 1
    return counts, role_source_counts


def materialize_portfolio_core_assets(
    *,
    inventory_path: str | Path,
    source_policy_path: str | Path,
    source_roots: Mapping[str, str | Path],
    output_root: str | Path,
    max_items: int | None = None,
    allow_incomplete_capacity: bool = False,
    capacity_profile: CoreCapacityProfile = "r1",
    document_selection_manifest_path: str | Path | None = None,
    reserve_activation_manifest_path: str | Path | None = None,
) -> PortfolioCoreAssetMaterializationResult:
    """Copy eligible candidates with durable per-candidate checkpoints.

    Every individual image is written with ``atomic_create_file`` and then has
    an immutable checkpoint written.  A later call verifies both before doing
    more work, so interruption between the image write and checkpoint is safe
    and recoverable.  No RAW input is deleted, renamed, extracted, or altered.
    """

    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive when supplied")
    preflight = preflight_portfolio_core_assets(
        inventory_path=inventory_path,
        source_policy_path=source_policy_path,
        source_roots=source_roots,
        capacity_profile=capacity_profile,
        document_selection_manifest_path=document_selection_manifest_path,
        reserve_activation_manifest_path=reserve_activation_manifest_path,
    )
    if not preflight.source_ready:
        rendered = "; ".join(
            f"{source}={','.join(reasons)}"
            for source, reasons in sorted(preflight.blocked_candidates.items())
        )
        raise PortfolioCoreAssetError(
            "materialization blocked by source preflight: " + rendered
        )
    if not preflight.capacity_ready and not allow_incomplete_capacity:
        capacity_parts = [
            f"pool:{pool}+{missing}"
            for pool, missing in sorted(preflight.capacity_shortfalls.items())
        ]
        capacity_parts.extend(
            f"capability:{capability}+{missing}"
            for capability, missing in sorted(
                preflight.capability_capacity_shortfalls.items()
            )
        )
        capacity_parts.extend(
            f"document:{role}+{missing}"
            for role, missing in sorted(preflight.document_selection_shortfalls.items())
        )
        rendered = ", ".join(capacity_parts)
        raise PortfolioCoreAssetError(
            "materialization blocked by Portfolio core capacity floor: " + rendered
        )
    candidates, inventory_sha256 = load_candidate_inventory(inventory_path)
    activated_reserve_ids, _reserve_activation_manifest_sha256 = (
        _resolve_activated_reserve_candidate_ids(
            candidates=candidates,
            inventory_sha256=inventory_sha256,
            reserve_activation_manifest_path=reserve_activation_manifest_path,
        )
    )
    policy, policy_sha256 = load_source_policy(source_policy_path)
    if (
        inventory_sha256 != preflight.inventory_sha256
        or policy_sha256 != preflight.source_policy_sha256
    ):
        raise PortfolioCoreAssetError("source inputs changed during preflight")
    selected = _selected_candidates(
        candidates,
        activated_reserve_candidate_ids=activated_reserve_ids,
    )
    roots = _normalize_source_roots(source_roots)
    output_root = Path(output_root).absolute()
    _require_disjoint_output_root(
        output_root,
        tuple(
            roots[source_id]
            for source_id in sorted({candidate.source_id for candidate in selected})
        ),
    )
    _ensure_real_directory(output_root, create=True, label="portfolio core output root")
    with _exclusive_materialization_writer_lock(output_root):
        _initialize_or_verify_run(
            output_root=output_root,
            inventory_sha256=inventory_sha256,
            source_policy_sha256=policy_sha256,
            selected=selected,
        )
        checkpoint_root = output_root / _CHECKPOINT_DIRECTORY
        _ensure_real_directory(
            checkpoint_root, create=True, label="checkpoint directory"
        )

        copied_this_call = 0
        checkpointed = 0
        for candidate in selected:
            checkpoint_path = checkpoint_root / f"{_candidate_digest(candidate)}.json"
            if checkpoint_path.exists():
                _verify_existing_checkpoint(output_root, checkpoint_path, candidate)
                checkpointed += 1
                continue
            if max_items is not None and copied_this_call >= max_items:
                continue
            source_root = roots[candidate.source_id]
            content = _read_and_verify_source_candidate(candidate, source_root)
            destination = _resolve_output_path(output_root, candidate.destination_path)
            _publish_or_verify_asset(destination, content, candidate)
            checkpoint = PortfolioCoreAssetCheckpoint(
                candidate_id=candidate.candidate_id,
                candidate_sha256=_candidate_digest(candidate),
                destination_path=candidate.destination_path,
                expected_bytes=candidate.expected_bytes,
                expected_sha256=candidate.expected_sha256,
                draft=_published_draft(candidate),
            )
            _publish_idempotent_file(checkpoint_path, canonical_json_bytes(checkpoint))
            checkpointed += 1
            copied_this_call += 1

        complete = checkpointed == len(selected)
        drafts_path: Path | None = None
        selection_path: Path | None = None
        if complete:
            drafts_path, selection_path = _finalize_materialization(
                output_root=output_root,
                selected=selected,
                inventory_sha256=inventory_sha256,
                source_policy_sha256=policy_sha256,
            )
        return PortfolioCoreAssetMaterializationResult(
            output_root=output_root,
            checkpointed_candidate_count=checkpointed,
            copied_this_call=copied_this_call,
            complete=complete,
            dataset_assets_path=drafts_path,
            selection_manifest_path=selection_path,
        )


def _load_selection_catalog_context(
    *,
    selection_manifest_path: str | Path,
    asset_catalog_dir: str | Path,
    asset_root: str | Path,
) -> tuple[
    PortfolioCoreSelectionManifest,
    str,
    Any,
    dict[str, DatasetAssetDraft],
    str,
]:
    """Load and cross-check the immutable selection, drafts, and catalog."""

    selection_path = Path(selection_manifest_path)
    selection_bytes = _read_metadata(
        selection_path, "Portfolio core selection manifest"
    )
    try:
        selection = PortfolioCoreSelectionManifest.model_validate_json(selection_bytes)
    except ValidationError as exc:
        raise PortfolioCoreAssetError("selection manifest violates schema") from exc
    if selection_bytes != canonical_json_bytes(selection):
        raise PortfolioCoreAssetError("selection manifest must be canonical JSON")
    selection_sha256 = sha256_bytes(selection_bytes)
    drafts = _load_published_drafts(
        selection_path.parent / _DRAFTS_FILE,
        expected_sha256=selection.dataset_assets_sha256,
    )
    if len(drafts) != selection.dataset_asset_count:
        raise PortfolioCoreAssetError("published draft count does not match selection")
    draft_by_path = {draft.local_path: draft for draft in drafts}
    if len(draft_by_path) != len(drafts):
        raise PortfolioCoreAssetError("published drafts have duplicate local paths")

    catalog_directory = Path(asset_catalog_dir)
    catalog = load_asset_catalog(catalog_directory, asset_root, verify_files=True)
    catalog.require_verified_files()
    catalog_manifest_sha256 = sha256_bytes(
        _read_metadata(catalog_directory / "manifest.json", "asset catalog manifest")
    )
    for row in selection.selections:
        draft = draft_by_path.get(row.destination_path)
        if draft != row.draft:
            raise PortfolioCoreAssetError(
                f"selection/draft binding drifted: {row.destination_path}"
            )
        _verify_catalog_asset(row, catalog.resolve_path(row.destination_path).asset)
    return (
        selection,
        selection_sha256,
        catalog,
        draft_by_path,
        catalog_manifest_sha256,
    )


def _build_selection_capability_assignments(
    *,
    selection: PortfolioCoreSelectionManifest,
    selection_sha256: str,
    catalog: Any,
    draft_by_path: Mapping[str, DatasetAssetDraft],
) -> tuple[CapabilityAssignment, ...]:
    """Rebuild the baseline assignment bytes from the v7 selection binding."""

    taxonomy = load_default_taxonomy_registry()
    task_spec = load_mvp_task_specification_v1()
    assignments: list[CapabilityAssignment] = []
    for row in selection.selections:
        if draft_by_path.get(row.destination_path) != row.draft:
            raise PortfolioCoreAssetError(
                f"selection/draft binding drifted: {row.destination_path}"
            )
        resolution = catalog.resolve_path(row.destination_path)
        asset = resolution.asset
        _verify_catalog_asset(row, asset)
        for binding in row.capability_bindings:
            capability = taxonomy.capabilities_by_id.get(binding.canonical_capability)
            task = task_spec.capabilities_by_id.get(binding.canonical_capability)
            if capability is None or task is None:
                raise PortfolioCoreAssetError(
                    "unknown capability in candidate inventory: "
                    + binding.canonical_capability
                )
            validate_capability_binding(
                taxonomy_version=TAXONOMY_VERSION,
                task_spec_version=PHASE3_TASK_SPEC_VERSION,
                canonical_intent=binding.canonical_intent,
                canonical_capability=binding.canonical_capability,
                acceptable_capabilities=(binding.canonical_capability,),
                requires_card=capability.requires_card,
                allowed_tools=task.allowed_tools,
                taxonomy=taxonomy,
                task_specification=task_spec,
            )
            identity = {
                "asset_id": asset.asset_id,
                "binding": binding.model_dump(mode="json"),
                "candidate_id": row.candidate_id,
                "policy_version": PORTFOLIO_CORE_ASSIGNMENT_POLICY_VERSION,
                "selection_manifest_sha256": selection_sha256,
            }
            assignments.append(
                CapabilityAssignment(
                    assignment_id=(
                        "portfolio-core.assignment."
                        + sha256_bytes(canonical_json_bytes(identity))
                    ),
                    asset_id=asset.asset_id,
                    image_path=row.destination_path,
                    canonical_intent=binding.canonical_intent,
                    canonical_capability=binding.canonical_capability,
                    acceptable_capabilities=(binding.canonical_capability,),
                    requires_card=capability.requires_card,
                    allowed_tools=task.allowed_tools,
                    taxonomy_version=TAXONOMY_VERSION,
                    taxonomy_sha256=DEFAULT_TAXONOMY_SHA256,
                    task_spec_version=PHASE3_TASK_SPEC_VERSION,
                    task_spec_sha256=PHASE3_TASK_SPEC_SHA256,
                    source_ref="project-choice:portfolio-core-asset-selection-v1",
                    source_artifact_sha256=selection_sha256,
                    rationale=(
                        "Portfolio candidate inventory declared this "
                        "image-to-capability binding."
                    ),
                )
            )
    assignments.sort(
        key=lambda item: (
            item.image_path,
            item.canonical_intent,
            item.canonical_capability,
        )
    )
    if not assignments:
        raise PortfolioCoreAssetError(
            "selection did not produce any capability assignments"
        )
    return tuple(assignments)


def _v8_profile_added_bindings(
    profile: CapabilityBindingPlanProfile,
) -> tuple[CandidateCapabilityBinding, ...]:
    if profile == "abo_product_identity_style_knowledge_triplet":
        return (
            CandidateCapabilityBinding(
                canonical_intent="divergent_rec",
                canonical_capability="product.style_recommendation",
            ),
            CandidateCapabilityBinding(
                canonical_intent="encyclopedia",
                canonical_capability="knowledge.visual_encyclopedia",
            ),
        )
    if profile == "isia_food_recipe_knowledge_pair":
        return (
            CandidateCapabilityBinding(
                canonical_intent="encyclopedia",
                canonical_capability="knowledge.visual_encyclopedia",
            ),
        )
    raise AssertionError(f"unknown v8 binding profile: {profile}")


def _v8_profile_baseline_bindings(
    profile: CapabilityBindingPlanProfile,
) -> tuple[CandidateCapabilityBinding, ...]:
    if profile == "abo_product_identity_style_knowledge_triplet":
        return (
            CandidateCapabilityBinding(
                canonical_intent="exact_match",
                canonical_capability="product.exact_match",
            ),
        )
    if profile == "isia_food_recipe_knowledge_pair":
        return (
            CandidateCapabilityBinding(
                canonical_intent="utility",
                canonical_capability="utility.recipe_guidance",
            ),
        )
    raise AssertionError(f"unknown v8 binding profile: {profile}")


def _v8_profile_rationale(profile: CapabilityBindingPlanProfile) -> str:
    if profile == "abo_product_identity_style_knowledge_triplet":
        return (
            "Approved bounded v8 counterfactual binding for product identity, "
            "style comparison, and product knowledge."
        )
    if profile == "isia_food_recipe_knowledge_pair":
        return (
            "Approved bounded v8 counterfactual binding for recipe guidance "
            "and food knowledge."
        )
    raise AssertionError(f"unknown v8 binding profile: {profile}")


def build_portfolio_core_v8_capability_binding_plan(
    *,
    selection_manifest_path: str | Path,
    asset_catalog_dir: str | Path,
    asset_root: str | Path,
    parent_assignments_path: str | Path,
    parent_assignment_sha256: str,
    parent_r1_core_plan_path: str | Path,
    output_path: str | Path,
) -> PortfolioCoreCapabilityBindingPlanResult:
    """Select the two approved v8 packs from verified v7 metadata only.

    No image is opened here.  Candidate eligibility is limited to catalog and
    selection metadata, singleton leakage components, and the r1 plan's
    resolved tail membership.  The latter makes the non-dev requirement
    executable rather than a narrative assertion.
    """

    _validate_sha256(parent_assignment_sha256, "parent_assignment_sha256")
    (
        selection,
        selection_sha256,
        catalog,
        draft_by_path,
        catalog_manifest_sha256,
    ) = _load_selection_catalog_context(
        selection_manifest_path=selection_manifest_path,
        asset_catalog_dir=asset_catalog_dir,
        asset_root=asset_root,
    )
    baseline_assignments = _build_selection_capability_assignments(
        selection=selection,
        selection_sha256=selection_sha256,
        catalog=catalog,
        draft_by_path=draft_by_path,
    )
    parent_assignments = load_capability_assignments(
        parent_assignments_path,
        expected_sha256=parent_assignment_sha256,
    )
    if canonical_jsonl_bytes(parent_assignments) != canonical_jsonl_bytes(
        baseline_assignments
    ):
        raise PortfolioCoreAssetError(
            "parent assignments do not exactly match the v7 selection/catalog binding"
        )

    try:
        r1_plan, r1_manifest = load_plan(parent_r1_core_plan_path)
    except (OSError, ValueError) as exc:
        raise PortfolioCoreAssetError("r1 core plan binding is invalid") from exc
    if r1_plan.scope != "core" or len(r1_plan.queries) != 1500:
        raise PortfolioCoreAssetError("r1 core plan must contain exactly 1,500 rows")
    if r1_manifest.plan_sha256 != sha256_bytes(
        _read_metadata(Path(parent_r1_core_plan_path), "r1 core plan")
    ):
        raise PortfolioCoreAssetError("r1 core plan manifest hash drifted")

    component_member_counts = {
        component.component_id: len(component.asset_ids)
        for component in catalog.components
    }
    component_by_path = {
        asset.local_path: catalog.resolve_path(asset.local_path).leakage_group_id
        for asset in catalog.assets
    }
    dev_components: set[str] = set()
    tail_components: set[str] = set()
    for index, query in enumerate(r1_plan.queries):
        component_id = component_by_path.get(query.image_path)
        if component_id is None:
            if query.canonical_capability != "utility.document_reading":
                raise PortfolioCoreAssetError(
                    "r1 core plan path is absent from v7 catalog outside the "
                    "explicitly replaced document role: " + query.image_path
                )
            continue
        (dev_components if index < 200 else tail_components).add(component_id)
    if dev_components & tail_components:
        raise PortfolioCoreAssetError("r1 dev and tail components overlap")

    component_capabilities: dict[str, set[str]] = defaultdict(set)
    component_sources: dict[str, set[str]] = defaultdict(set)
    for assignment in parent_assignments:
        component_id = component_by_path.get(assignment.image_path)
        if component_id is None:
            raise PortfolioCoreAssetError(
                "parent assignment path is absent from the v7 catalog: "
                + assignment.image_path
            )
        component_capabilities[component_id].add(assignment.canonical_capability)
        component_sources[component_id].add(
            catalog.resolve_path(assignment.image_path).asset.source_dataset
        )

    abo_triplet_capabilities = {
        "product.exact_match",
        "product.style_recommendation",
        "knowledge.visual_encyclopedia",
    }
    isia_food_pair_capabilities = {
        "utility.recipe_guidance",
        "knowledge.visual_encyclopedia",
    }
    baseline_abo_triplets = {
        component_id
        for component_id, capabilities in component_capabilities.items()
        if abo_triplet_capabilities <= capabilities
        and component_sources[component_id] == {"abo"}
    }
    baseline_isia_pairs = {
        component_id
        for component_id, capabilities in component_capabilities.items()
        if isia_food_pair_capabilities <= capabilities
        and component_sources[component_id] == {"isia_food500"}
    }

    def select_profile(
        profile: CapabilityBindingPlanProfile,
        count: int,
    ) -> tuple[PortfolioCoreCapabilityBindingPlanEntry, ...]:
        baseline_bindings = _v8_profile_baseline_bindings(profile)
        expected_source = (
            "abo"
            if profile == "abo_product_identity_style_knowledge_triplet"
            else "isia_food500"
        )
        expected_pool = "exact_match" if expected_source == "abo" else "utility"
        expected_capabilities = {
            binding.canonical_capability for binding in baseline_bindings
        }
        chosen: list[PortfolioCoreCapabilityBindingPlanEntry] = []
        chosen_components: set[str] = set()
        for row in sorted(
            selection.selections,
            key=lambda item: (item.destination_path, item.candidate_id),
        ):
            if (
                row.source_id != expected_source
                or row.pool != expected_pool
                or row.capability_bindings != baseline_bindings
            ):
                continue
            resolution = catalog.resolve_path(row.destination_path)
            component_id = resolution.leakage_group_id
            if (
                component_id in dev_components
                or component_id not in tail_components
                or component_member_counts.get(component_id) != 1
                or component_capabilities[component_id] != expected_capabilities
                or component_sources[component_id] != {expected_source}
                or component_id in chosen_components
            ):
                continue
            if expected_source == "abo" and row.draft.product_id is None:
                continue
            chosen.append(
                PortfolioCoreCapabilityBindingPlanEntry(
                    binding_profile=profile,
                    candidate_id=row.candidate_id,
                    candidate_sha256=row.candidate_sha256,
                    source_id=row.source_id,
                    source_record_id=row.draft.source_record_id,
                    destination_path=row.destination_path,
                    asset_id=resolution.asset.asset_id,
                    component_id=component_id,
                    expected_sha256=row.expected_sha256,
                )
            )
            chosen_components.add(component_id)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise PortfolioCoreAssetError(
                "v8 binding plan lacks eligible non-dev singleton components for "
                + profile
            )
        return tuple(chosen)

    abo_entries = select_profile(
        "abo_product_identity_style_knowledge_triplet",
        CORE_R2_V8_ABO_TRIPLET_ADDITION_COUNT,
    )
    isia_entries = select_profile(
        "isia_food_recipe_knowledge_pair",
        CORE_R2_V8_ISIA_FOOD_PAIR_ADDITION_COUNT,
    )
    manifest = PortfolioCoreCapabilityBindingPlanManifest(
        parent_inventory_sha256=selection.inventory_sha256,
        parent_selection_manifest_sha256=selection_sha256,
        parent_catalog_sha256=catalog.manifest.catalog_sha256,
        parent_catalog_manifest_sha256=catalog_manifest_sha256,
        parent_assignment_sha256=parent_assignment_sha256,
        parent_r1_core_plan_sha256=r1_manifest.plan_sha256,
        asset_count=catalog.manifest.asset_count,
        component_count=catalog.manifest.component_count,
        baseline_global_abo_triplet_component_count=len(baseline_abo_triplets),
        baseline_dev_abo_triplet_component_count=len(
            baseline_abo_triplets & dev_components
        ),
        baseline_non_dev_abo_triplet_component_count=len(
            baseline_abo_triplets - dev_components
        ),
        baseline_global_isia_food_pair_component_count=len(baseline_isia_pairs),
        baseline_dev_isia_food_pair_component_count=len(
            baseline_isia_pairs & dev_components
        ),
        baseline_non_dev_isia_food_pair_component_count=len(
            baseline_isia_pairs - dev_components
        ),
        reason=(
            "Close the frozen non-dev counterfactual component targets with "
            "two fixed, metadata-bound v8 capability packs; no v7 asset or "
            "catalog bytes are changed."
        ),
        selections=tuple(
            sorted(
                (*abo_entries, *isia_entries),
                key=lambda entry: (
                    entry.binding_profile,
                    entry.destination_path,
                    entry.candidate_id,
                ),
            )
        ),
    )
    output = Path(output_path).absolute()
    content = canonical_json_bytes(manifest)
    _publish_idempotent_file(output, content)
    loaded, output_sha256 = load_portfolio_core_capability_binding_plan_manifest(output)
    if loaded != manifest:
        raise PortfolioCoreAssetError("published v8 binding plan changed on reload")
    return PortfolioCoreCapabilityBindingPlanResult(
        output_path=output,
        output_sha256=output_sha256,
        abo_candidate_ids=tuple(entry.candidate_id for entry in abo_entries),
        isia_food_candidate_ids=tuple(entry.candidate_id for entry in isia_entries),
    )


def build_portfolio_core_capability_assignments(
    *,
    selection_manifest_path: str | Path,
    asset_catalog_dir: str | Path,
    asset_root: str | Path,
    output_path: str | Path,
) -> PortfolioCoreAssignmentResult:
    """Bind published Portfolio drafts to a verified generic AssetCatalog."""

    (
        selection,
        selection_sha256,
        catalog,
        draft_by_path,
        _catalog_manifest_sha256,
    ) = _load_selection_catalog_context(
        selection_manifest_path=selection_manifest_path,
        asset_catalog_dir=asset_catalog_dir,
        asset_root=asset_root,
    )
    assignments = _build_selection_capability_assignments(
        selection=selection,
        selection_sha256=selection_sha256,
        catalog=catalog,
        draft_by_path=draft_by_path,
    )
    assignment_bytes = canonical_jsonl_bytes(assignments)
    output = Path(output_path).absolute()
    _publish_idempotent_file(output, assignment_bytes)
    return PortfolioCoreAssignmentResult(
        output_path=output,
        output_sha256=sha256_bytes(assignment_bytes),
        assignment_count=len(assignments),
        selection_manifest_sha256=selection_sha256,
    )


def build_portfolio_core_v8_capability_assignments(
    *,
    selection_manifest_path: str | Path,
    asset_catalog_dir: str | Path,
    asset_root: str | Path,
    parent_assignments_path: str | Path,
    parent_r1_core_plan_path: str | Path,
    binding_plan_manifest_path: str | Path,
    output_path: str | Path,
) -> PortfolioCoreAssignmentResult:
    """Publish v8 assignments after replaying the fixed binding-plan selection."""

    binding_plan, binding_plan_sha256 = (
        load_portfolio_core_capability_binding_plan_manifest(binding_plan_manifest_path)
    )
    replay = build_portfolio_core_v8_capability_binding_plan(
        selection_manifest_path=selection_manifest_path,
        asset_catalog_dir=asset_catalog_dir,
        asset_root=asset_root,
        parent_assignments_path=parent_assignments_path,
        parent_assignment_sha256=binding_plan.parent_assignment_sha256,
        parent_r1_core_plan_path=parent_r1_core_plan_path,
        output_path=binding_plan_manifest_path,
    )
    if replay.output_sha256 != binding_plan_sha256:
        raise PortfolioCoreAssetError("v8 binding plan replay changed its bytes")

    (
        selection,
        selection_sha256,
        catalog,
        draft_by_path,
        catalog_manifest_sha256,
    ) = _load_selection_catalog_context(
        selection_manifest_path=selection_manifest_path,
        asset_catalog_dir=asset_catalog_dir,
        asset_root=asset_root,
    )
    if (
        binding_plan.parent_inventory_sha256 != selection.inventory_sha256
        or binding_plan.parent_selection_manifest_sha256 != selection_sha256
        or binding_plan.parent_catalog_sha256 != catalog.manifest.catalog_sha256
        or binding_plan.parent_catalog_manifest_sha256 != catalog_manifest_sha256
    ):
        raise PortfolioCoreAssetError("v8 binding plan parent artifact binding drifted")
    assignments = list(
        _build_selection_capability_assignments(
            selection=selection,
            selection_sha256=selection_sha256,
            catalog=catalog,
            draft_by_path=draft_by_path,
        )
    )
    parent_assignments = load_capability_assignments(
        parent_assignments_path,
        expected_sha256=binding_plan.parent_assignment_sha256,
    )
    if canonical_jsonl_bytes(parent_assignments) != canonical_jsonl_bytes(assignments):
        raise PortfolioCoreAssetError(
            "parent assignments do not exactly match the v7 selection/catalog binding"
        )

    rows_by_candidate = {row.candidate_id: row for row in selection.selections}
    existing_keys = {
        (item.asset_id, item.image_path, item.canonical_intent) for item in assignments
    }
    taxonomy = load_default_taxonomy_registry()
    task_spec = load_mvp_task_specification_v1()
    for entry in binding_plan.selections:
        row = rows_by_candidate.get(entry.candidate_id)
        if row is None:
            raise PortfolioCoreAssetError(
                "v8 binding plan candidate is absent from selection: "
                + entry.candidate_id
            )
        resolution = catalog.resolve_path(row.destination_path)
        if (
            row.candidate_sha256 != entry.candidate_sha256
            or row.source_id != entry.source_id
            or row.draft.source_record_id != entry.source_record_id
            or row.destination_path != entry.destination_path
            or row.expected_sha256 != entry.expected_sha256
            or resolution.asset.asset_id != entry.asset_id
            or resolution.leakage_group_id != entry.component_id
        ):
            raise PortfolioCoreAssetError(
                "v8 binding plan entry drifted: " + entry.candidate_id
            )
        for binding in _v8_profile_added_bindings(entry.binding_profile):
            key = (
                resolution.asset.asset_id,
                row.destination_path,
                binding.canonical_intent,
            )
            if key in existing_keys:
                raise PortfolioCoreAssetError(
                    "v8 binding plan duplicates an existing assignment: "
                    + entry.candidate_id
                )
            capability = taxonomy.capabilities_by_id.get(binding.canonical_capability)
            task = task_spec.capabilities_by_id.get(binding.canonical_capability)
            if capability is None or task is None:
                raise PortfolioCoreAssetError(
                    "v8 binding profile has an unknown capability: "
                    + binding.canonical_capability
                )
            validate_capability_binding(
                taxonomy_version=TAXONOMY_VERSION,
                task_spec_version=PHASE3_TASK_SPEC_VERSION,
                canonical_intent=binding.canonical_intent,
                canonical_capability=binding.canonical_capability,
                acceptable_capabilities=(binding.canonical_capability,),
                requires_card=capability.requires_card,
                allowed_tools=task.allowed_tools,
                taxonomy=taxonomy,
                task_specification=task_spec,
            )
            identity = {
                "asset_id": resolution.asset.asset_id,
                "binding": binding.model_dump(mode="json"),
                "binding_plan_manifest_sha256": binding_plan_sha256,
                "candidate_id": entry.candidate_id,
                "policy_version": PORTFOLIO_CORE_V8_CAPABILITY_BINDING_POLICY_VERSION,
            }
            assignments.append(
                CapabilityAssignment(
                    assignment_id=(
                        "portfolio-core.assignment."
                        + sha256_bytes(canonical_json_bytes(identity))
                    ),
                    asset_id=resolution.asset.asset_id,
                    image_path=row.destination_path,
                    canonical_intent=binding.canonical_intent,
                    canonical_capability=binding.canonical_capability,
                    acceptable_capabilities=(binding.canonical_capability,),
                    requires_card=capability.requires_card,
                    allowed_tools=task.allowed_tools,
                    taxonomy_version=TAXONOMY_VERSION,
                    taxonomy_sha256=DEFAULT_TAXONOMY_SHA256,
                    task_spec_version=PHASE3_TASK_SPEC_VERSION,
                    task_spec_sha256=PHASE3_TASK_SPEC_SHA256,
                    source_ref=(
                        "project-choice:portfolio-core-v8-capability-binding-v1"
                    ),
                    source_artifact_sha256=binding_plan_sha256,
                    rationale=_v8_profile_rationale(entry.binding_profile),
                )
            )
            existing_keys.add(key)
    assignments.sort(
        key=lambda item: (
            item.image_path,
            item.canonical_intent,
            item.canonical_capability,
        )
    )
    assignment_bytes = canonical_jsonl_bytes(assignments)
    output = Path(output_path).absolute()
    _publish_idempotent_file(output, assignment_bytes)
    output_sha256 = sha256_bytes(assignment_bytes)
    loaded = load_capability_assignments(output, expected_sha256=output_sha256)
    if canonical_jsonl_bytes(loaded) != assignment_bytes:
        raise PortfolioCoreAssetError("published v8 assignments changed on reload")
    return PortfolioCoreAssignmentResult(
        output_path=output,
        output_sha256=output_sha256,
        assignment_count=len(assignments),
        selection_manifest_sha256=selection_sha256,
        binding_plan_manifest_sha256=binding_plan_sha256,
    )


def _load_v9_rpc_staging_adapter_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], str]:
    """Load only the stable adapter fields needed to bind a v9 merge."""

    content = _read_metadata(Path(path), "v9 RPC staging adapter manifest")
    try:
        value = parse_canonical_json(
            content,
            label="v9 RPC staging adapter manifest",
        )
    except ArtifactFormatError as exc:
        raise PortfolioCoreAssetError(
            "v9 RPC staging adapter manifest is invalid"
        ) from exc
    if not isinstance(value, dict) or content != canonical_json_bytes(value):
        raise PortfolioCoreAssetError(
            "v9 RPC staging adapter manifest must be canonical JSON"
        )
    expected = {
        "schema_version": 1,
        "policy_version": "portfolio-core-selected-source-adapters-v1",
        "track": "portfolio",
        "formal_eligible": False,
        "formal_status": "non_formal",
        "source_id": "rpc",
        "include_count": CORE_R2_V9_RPC_STAGING_INCLUDE_COUNT,
        "reserve_count": CORE_R2_V9_RPC_STAGING_RESERVE_COUNT,
        "raw_mutation_performed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise PortfolioCoreAssetError(
                f"v9 RPC staging adapter manifest field drifted: {key}"
            )
    for key in (
        "source_lock_sha256",
        "source_policy_sha256",
        "dev_mini_selection_sha256",
    ):
        try:
            _validate_sha256(value[key], f"v9 RPC staging adapter {key}")
        except (KeyError, ValueError) as exc:
            raise PortfolioCoreAssetError(
                f"v9 RPC staging adapter manifest lacks {key}"
            ) from exc
    archives = value.get("archives")
    if not isinstance(archives, list) or len(archives) != 1:
        raise PortfolioCoreAssetError(
            "v9 RPC staging adapter manifest must bind exactly one archive"
        )
    archive = archives[0]
    if (
        not isinstance(archive, dict)
        or not isinstance(archive.get("logical_path"), str)
        or not isinstance(archive.get("bytes"), int)
        or archive["bytes"] <= 0
    ):
        raise PortfolioCoreAssetError("v9 RPC staging archive identity is invalid")
    try:
        _validate_sha256(archive["sha256"], "v9 RPC staging archive sha256")
    except (KeyError, ValueError) as exc:
        raise PortfolioCoreAssetError("v9 RPC staging archive hash is invalid") from exc
    return value, sha256_bytes(content)


def _require_expected_digest(
    actual: str,
    expected: str,
    *,
    label: str,
) -> None:
    _validate_sha256(expected, f"expected {label} SHA-256")
    if actual != expected:
        raise PortfolioCoreAssetError(
            f"{label} SHA-256 drifted: expected={expected} actual={actual}"
        )


def _require_rpc_multi_candidate(
    candidate: PortfolioCoreAssetCandidate,
    *,
    label: str,
) -> None:
    expected_bindings = (
        CandidateCapabilityBinding(
            canonical_intent="multi_product",
            canonical_capability="product.multi_search",
        ),
    )
    if (
        candidate.source_id != "rpc"
        or candidate.pool != "multi_product"
        or candidate.capability_bindings != expected_bindings
    ):
        raise PortfolioCoreAssetError(f"{label} is not a valid RPC Multi candidate")


def build_portfolio_core_v9_multi_closure_inventory(
    *,
    parent_inventory_path: str | Path,
    parent_inventory_sha256: str,
    parent_reserve_activation_manifest_path: str | Path,
    parent_v8_assignments_path: str | Path,
    parent_v8_assignment_sha256: str,
    v9_source_policy_path: str | Path,
    rpc_staging_candidates_path: str | Path,
    rpc_staging_adapter_manifest_path: str | Path,
    output_inventory_path: str | Path,
    output_reserve_activation_manifest_path: str | Path,
    output_manifest_path: str | Path,
) -> PortfolioCoreV9MultiClosureInventoryResult:
    """Merge the approved RPC staging rerun into a new create-only inventory.

    This function does not copy image bytes.  It only validates immutable
    candidate metadata, rebinds the existing CORD reserve sidecar to the new
    inventory digest, and publishes the three small metadata artifacts needed
    by the existing materializer.
    """

    parent_candidates, actual_parent_inventory_sha256 = load_candidate_inventory(
        parent_inventory_path
    )
    _require_expected_digest(
        actual_parent_inventory_sha256,
        parent_inventory_sha256,
        label="parent inventory",
    )
    parent_activation, parent_activation_sha256 = (
        load_portfolio_core_reserve_activation_manifest(
            parent_reserve_activation_manifest_path
        )
    )
    if parent_activation.parent_inventory_sha256 != actual_parent_inventory_sha256:
        raise PortfolioCoreAssetError(
            "parent reserve activation does not bind the parent inventory"
        )
    parent_v8_assignments = load_capability_assignments(
        parent_v8_assignments_path,
        expected_sha256=parent_v8_assignment_sha256,
    )
    _validate_sha256(parent_v8_assignment_sha256, "parent_v8_assignment_sha256")
    _v9_source_policy, v9_source_policy_sha256 = load_source_policy(
        v9_source_policy_path
    )

    staging_path = Path(rpc_staging_candidates_path).absolute()
    staging_manifest_path = Path(rpc_staging_adapter_manifest_path).absolute()
    if staging_path.parent != staging_manifest_path.parent:
        raise PortfolioCoreAssetError(
            "v9 RPC staging candidates and adapter manifest must share an output root"
        )
    staging_manifest, staging_manifest_sha256 = _load_v9_rpc_staging_adapter_manifest(
        staging_manifest_path
    )
    staged_candidates, staged_candidates_sha256 = load_candidate_inventory(staging_path)
    if len(staged_candidates) != (
        CORE_R2_V9_RPC_STAGING_INCLUDE_COUNT + CORE_R2_V9_RPC_STAGING_RESERVE_COUNT
    ):
        raise PortfolioCoreAssetError("v9 RPC staging candidate count drifted")
    expected_selections = ("include",) * CORE_R2_V9_RPC_STAGING_INCLUDE_COUNT + (
        "reserve",
    ) * CORE_R2_V9_RPC_STAGING_RESERVE_COUNT
    if (
        tuple(candidate.selection for candidate in staged_candidates)
        != expected_selections
    ):
        raise PortfolioCoreAssetError("v9 RPC staging selection states drifted")
    for candidate in staged_candidates:
        _require_rpc_multi_candidate(candidate, label="v9 RPC staging candidate")
    if len({candidate.expected_sha256 for candidate in staged_candidates}) != len(
        staged_candidates
    ):
        raise PortfolioCoreAssetError(
            "v9 RPC staging candidates are not content-unique"
        )
    if len(
        {candidate.draft.source_record_id for candidate in staged_candidates}
    ) != len(staged_candidates):
        raise PortfolioCoreAssetError("v9 RPC staging candidates repeat source records")

    parent_rpc_candidates = tuple(
        candidate for candidate in parent_candidates if candidate.source_id == "rpc"
    )
    if len(parent_rpc_candidates) != (
        CORE_R2_V9_PARENT_RPC_INCLUDE_COUNT + CORE_R2_V9_PARENT_RPC_RESERVE_COUNT
    ):
        raise PortfolioCoreAssetError(
            "parent inventory does not contain 105+4 RPC rows"
        )
    for candidate in parent_rpc_candidates:
        _require_rpc_multi_candidate(candidate, label="parent RPC candidate")
    if tuple(candidate.selection for candidate in parent_rpc_candidates) != (
        ("include",) * CORE_R2_V9_PARENT_RPC_INCLUDE_COUNT
        + ("reserve",) * CORE_R2_V9_PARENT_RPC_RESERVE_COUNT
    ):
        raise PortfolioCoreAssetError("parent RPC selection states drifted")

    parent_rpc_by_id = {
        candidate.candidate_id: candidate for candidate in parent_rpc_candidates
    }
    staged_original = staged_candidates[: len(parent_rpc_candidates)]
    if {candidate.candidate_id for candidate in staged_original} != set(
        parent_rpc_by_id
    ):
        raise PortfolioCoreAssetError(
            "v9 RPC staging does not start with the parent RPC candidate set"
        )
    for index, staged in enumerate(staged_original):
        parent = parent_rpc_by_id[staged.candidate_id]
        expected_parent = staged.model_copy(
            update={"selection": "reserve"}
            if index >= CORE_R2_V9_PARENT_RPC_INCLUDE_COUNT
            else {}
        )
        if parent != expected_parent:
            raise PortfolioCoreAssetError(
                "v9 RPC staging changed a parent candidate beyond the approved "
                "reserve-to-include transition: " + staged.candidate_id
            )

    promoted = staged_candidates[
        CORE_R2_V9_PARENT_RPC_INCLUDE_COUNT : CORE_R2_V9_PARENT_RPC_INCLUDE_COUNT
        + CORE_R2_V9_PARENT_RPC_RESERVE_COUNT
    ]
    fresh = staged_candidates[
        CORE_R2_V9_PARENT_RPC_INCLUDE_COUNT
        + CORE_R2_V9_PARENT_RPC_RESERVE_COUNT : CORE_R2_V9_RPC_STAGING_INCLUDE_COUNT
    ]
    remaining_reserves = staged_candidates[CORE_R2_V9_RPC_STAGING_INCLUDE_COUNT:]
    if len(fresh) != CORE_R2_V9_FRESH_RPC_MULTI_COUNT:
        raise AssertionError("v9 RPC staging constants are inconsistent")
    parent_candidate_ids = {candidate.candidate_id for candidate in parent_candidates}
    if any(candidate.candidate_id in parent_candidate_ids for candidate in fresh):
        raise PortfolioCoreAssetError("v9 RPC fresh candidate already exists in parent")
    if any(
        candidate.candidate_id in parent_candidate_ids
        for candidate in remaining_reserves
    ):
        raise PortfolioCoreAssetError(
            "v9 RPC reserve candidate already exists in parent"
        )
    if any(candidate.selection != "include" for candidate in (*promoted, *fresh)):
        raise PortfolioCoreAssetError(
            "v9 RPC closure selected candidate is not include"
        )
    if any(candidate.selection != "reserve" for candidate in remaining_reserves):
        raise PortfolioCoreAssetError(
            "v9 RPC closure trailing candidate is not reserve"
        )

    merged: list[PortfolioCoreAssetCandidate] = []
    inserted_staging = False
    for candidate in parent_candidates:
        if candidate.source_id == "rpc":
            if not inserted_staging:
                merged.extend(staged_candidates)
                inserted_staging = True
            continue
        merged.append(candidate)
    if not inserted_staging:
        raise PortfolioCoreAssetError("parent inventory has no RPC candidate block")
    if len(merged) != len(parent_candidates) + CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT:
        raise PortfolioCoreAssetError(
            "v9 merged inventory candidate count is inconsistent"
        )
    _validate_candidate_set(merged)
    merged_inventory_bytes = canonical_jsonl_bytes(merged)
    merged_inventory_sha256 = sha256_bytes(merged_inventory_bytes)
    output_inventory = _publish_idempotent_file(
        Path(output_inventory_path).absolute(),
        merged_inventory_bytes,
    )
    loaded_merged, loaded_merged_sha256 = load_candidate_inventory(output_inventory)
    if (
        loaded_merged != tuple(merged)
        or loaded_merged_sha256 != merged_inventory_sha256
    ):
        raise PortfolioCoreAssetError("published v9 merged inventory changed on reload")

    rebased_activation = PortfolioCoreReserveActivationManifest(
        parent_inventory_sha256=merged_inventory_sha256,
        parent_selection_manifest_sha256=(
            parent_activation.parent_selection_manifest_sha256
        ),
        parent_catalog_sha256=parent_activation.parent_catalog_sha256,
        baseline_component_count=parent_activation.baseline_component_count,
        required_component_floor=parent_activation.required_component_floor,
        closure_margin=parent_activation.closure_margin,
        predicted_component_count=parent_activation.predicted_component_count,
        reason=(
            "Carry forward the immutable CORD reserve activation bindings onto "
            "the v9 RPC Multi closure inventory; no CORD role, bytes, or "
            "selection entry is changed."
        ),
        selections=parent_activation.selections,
    )
    rebased_activation_bytes = canonical_json_bytes(rebased_activation)
    output_activation = _publish_idempotent_file(
        Path(output_reserve_activation_manifest_path).absolute(),
        rebased_activation_bytes,
    )
    loaded_activation, rebased_activation_sha256 = (
        load_portfolio_core_reserve_activation_manifest(output_activation)
    )
    if loaded_activation != rebased_activation:
        raise PortfolioCoreAssetError(
            "published v9 reserve activation changed on reload"
        )

    promoted_ids = tuple(sorted(candidate.candidate_id for candidate in promoted))
    fresh_ids = tuple(sorted(candidate.candidate_id for candidate in fresh))
    remaining_reserve_ids = tuple(
        sorted(candidate.candidate_id for candidate in remaining_reserves)
    )
    delta_ids = tuple(sorted((*promoted_ids, *fresh_ids)))
    manifest = PortfolioCoreV9MultiClosureInventoryManifest(
        parent_inventory_sha256=actual_parent_inventory_sha256,
        parent_reserve_activation_manifest_sha256=parent_activation_sha256,
        rebased_reserve_activation_manifest_sha256=rebased_activation_sha256,
        parent_v8_assignment_sha256=parent_v8_assignment_sha256,
        rpc_staging_candidates_sha256=staged_candidates_sha256,
        rpc_staging_adapter_manifest_sha256=staging_manifest_sha256,
        source_policy_sha256=v9_source_policy_sha256,
        staging_source_policy_sha256=staging_manifest["source_policy_sha256"],
        source_lock_sha256=staging_manifest["source_lock_sha256"],
        merged_inventory_sha256=merged_inventory_sha256,
        parent_v8_assignment_count=len(parent_v8_assignments),
        promoted_reserve_candidate_ids=promoted_ids,
        fresh_candidate_ids=fresh_ids,
        remaining_reserve_candidate_ids=remaining_reserve_ids,
        selection_delta_candidate_ids=delta_ids,
        reason=(
            "Promote the four v6 RPC reserves and add the next sixteen "
            "deterministic RPC scenes, preserving the frozen source role and "
            "leaving the trailing four staged candidates as reserves."
        ),
    )
    manifest_bytes = canonical_json_bytes(manifest)
    output_manifest = _publish_idempotent_file(
        Path(output_manifest_path).absolute(),
        manifest_bytes,
    )
    loaded_manifest, manifest_sha256 = (
        load_portfolio_core_v9_multi_closure_inventory_manifest(output_manifest)
    )
    if loaded_manifest != manifest:
        raise PortfolioCoreAssetError(
            "published v9 closure inventory changed on reload"
        )
    return PortfolioCoreV9MultiClosureInventoryResult(
        inventory_path=output_inventory,
        inventory_sha256=merged_inventory_sha256,
        reserve_activation_manifest_path=output_activation,
        reserve_activation_manifest_sha256=rebased_activation_sha256,
        manifest_path=output_manifest,
        manifest_sha256=manifest_sha256,
        selection_delta_candidate_ids=delta_ids,
    )


def _load_verified_r1_core_plan(path: str | Path) -> tuple[Any, Any]:
    try:
        plan, manifest = load_plan(path)
    except (OSError, ValueError) as exc:
        raise PortfolioCoreAssetError("r1 core plan binding is invalid") from exc
    if plan.scope != "core" or len(plan.queries) != 1_500:
        raise PortfolioCoreAssetError("r1 core plan must contain exactly 1,500 rows")
    actual_sha256 = sha256_bytes(_read_metadata(Path(path), "r1 core plan"))
    if manifest.plan_sha256 != actual_sha256:
        raise PortfolioCoreAssetError("r1 core plan manifest hash drifted")
    return plan, manifest


def build_portfolio_core_v9_multi_closure_catalog_receipt(
    *,
    closure_inventory_manifest_path: str | Path,
    parent_selection_manifest_path: str | Path,
    parent_asset_catalog_dir: str | Path,
    parent_asset_root: str | Path,
    parent_v8_assignments_path: str | Path,
    parent_r1_core_plan_path: str | Path,
    v9_selection_manifest_path: str | Path,
    v9_asset_catalog_dir: str | Path,
    v9_asset_root: str | Path,
    output_path: str | Path,
) -> PortfolioCoreV9MultiClosureCatalogReceiptResult:
    """Publish post-catalog proof that v9 adds independent non-dev Multi assets."""

    closure, closure_manifest_sha256 = (
        load_portfolio_core_v9_multi_closure_inventory_manifest(
            closure_inventory_manifest_path
        )
    )
    (
        parent_selection,
        parent_selection_sha256,
        parent_catalog,
        _parent_drafts,
        parent_catalog_manifest_sha256,
    ) = _load_selection_catalog_context(
        selection_manifest_path=parent_selection_manifest_path,
        asset_catalog_dir=parent_asset_catalog_dir,
        asset_root=parent_asset_root,
    )
    if parent_selection.inventory_sha256 != closure.parent_inventory_sha256:
        raise PortfolioCoreAssetError(
            "v9 closure parent inventory does not bind the parent selection"
        )
    parent_v8_assignments = load_capability_assignments(
        parent_v8_assignments_path,
        expected_sha256=closure.parent_v8_assignment_sha256,
    )
    if len(parent_v8_assignments) != closure.parent_v8_assignment_count:
        raise PortfolioCoreAssetError("v9 closure parent v8 assignment count drifted")

    (
        v9_selection,
        v9_selection_sha256,
        v9_catalog,
        _v9_drafts,
        v9_catalog_manifest_sha256,
    ) = _load_selection_catalog_context(
        selection_manifest_path=v9_selection_manifest_path,
        asset_catalog_dir=v9_asset_catalog_dir,
        asset_root=v9_asset_root,
    )
    if v9_selection.inventory_sha256 != closure.merged_inventory_sha256:
        raise PortfolioCoreAssetError(
            "v9 selection does not bind the v9 closure inventory"
        )
    if v9_selection.source_policy_sha256 != closure.source_policy_sha256:
        raise PortfolioCoreAssetError("v9 selection source policy drifted")

    parent_rows_by_id = {row.candidate_id: row for row in parent_selection.selections}
    v9_rows_by_id = {row.candidate_id: row for row in v9_selection.selections}
    missing_parent_rows = sorted(set(parent_rows_by_id) - set(v9_rows_by_id))
    if missing_parent_rows:
        raise PortfolioCoreAssetError(
            "v9 selection omitted parent rows: " + ",".join(missing_parent_rows)
        )
    for candidate_id, parent_row in parent_rows_by_id.items():
        if v9_rows_by_id[candidate_id] != parent_row:
            raise PortfolioCoreAssetError(
                "v9 selection changed a parent selection row: " + candidate_id
            )
    delta_candidate_ids = tuple(sorted(set(v9_rows_by_id) - set(parent_rows_by_id)))
    if delta_candidate_ids != closure.selection_delta_candidate_ids:
        raise PortfolioCoreAssetError(
            "v9 selection delta does not match the closure inventory manifest"
        )
    if len(delta_candidate_ids) != CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT:
        raise PortfolioCoreAssetError(
            "v9 selection did not add exactly twenty RPC rows"
        )

    expected_multi_bindings = (
        CandidateCapabilityBinding(
            canonical_intent="multi_product",
            canonical_capability="product.multi_search",
        ),
    )
    for candidate_id in delta_candidate_ids:
        row = v9_rows_by_id[candidate_id]
        if (
            row.source_id != "rpc"
            or row.pool != "multi_product"
            or row.capability_bindings != expected_multi_bindings
        ):
            raise PortfolioCoreAssetError(
                "v9 selection delta broadened the frozen RPC Multi role: "
                + candidate_id
            )

    parent_asset_ids: set[str] = set()
    for row in parent_selection.selections:
        parent_asset = parent_catalog.resolve_path(row.destination_path).asset
        v9_asset = v9_catalog.resolve_path(row.destination_path).asset
        if v9_asset.asset_id != parent_asset.asset_id:
            raise PortfolioCoreAssetError(
                "v9 catalog changed a parent asset identity: " + row.destination_path
            )
        parent_asset_ids.add(parent_asset.asset_id)
    delta_asset_ids = tuple(
        sorted(
            v9_catalog.resolve_path(
                v9_rows_by_id[candidate_id].destination_path
            ).asset.asset_id
            for candidate_id in delta_candidate_ids
        )
    )
    if len(delta_asset_ids) != len(set(delta_asset_ids)):
        raise PortfolioCoreAssetError("v9 selection delta has duplicate catalog assets")

    component_asset_ids = {
        component.component_id: set(component.asset_ids)
        for component in v9_catalog.components
    }
    delta_component_ids = {
        v9_catalog.resolve_path(
            v9_rows_by_id[candidate_id].destination_path
        ).leakage_group_id
        for candidate_id in delta_candidate_ids
    }
    pure_new_multi_component_ids = tuple(
        sorted(
            component_id
            for component_id in delta_component_ids
            if component_asset_ids[component_id] <= set(delta_asset_ids)
            and not component_asset_ids[component_id] & parent_asset_ids
        )
    )
    if len(pure_new_multi_component_ids) < CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT:
        raise PortfolioCoreAssetError(
            "v9 RPC closure post-catalog independent Multi component shortfall: "
            f"actual={len(pure_new_multi_component_ids)} "
            f"required={CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT}"
        )
    component_count_delta = (
        v9_catalog.manifest.component_count - parent_catalog.manifest.component_count
    )
    if component_count_delta < CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT:
        raise PortfolioCoreAssetError(
            "v9 RPC closure post-catalog component delta shortfall: "
            f"actual={component_count_delta} "
            f"required={CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT}"
        )

    r1_plan, r1_manifest = _load_verified_r1_core_plan(parent_r1_core_plan_path)
    component_by_path = {
        asset.local_path: v9_catalog.resolve_path(asset.local_path).leakage_group_id
        for asset in v9_catalog.assets
    }
    dev_components: set[str] = set()
    for query in r1_plan.queries[:200]:
        component_id = component_by_path.get(query.image_path)
        if component_id is None:
            if query.canonical_capability != "utility.document_reading":
                raise PortfolioCoreAssetError(
                    "r1 dev plan path is absent from v9 catalog outside the "
                    "explicit document rebase: " + query.image_path
                )
            continue
        dev_components.add(component_id)
    nondev_multi_component_ids = tuple(
        component_id
        for component_id in pure_new_multi_component_ids
        if component_id not in dev_components
    )
    if len(nondev_multi_component_ids) < CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT:
        raise PortfolioCoreAssetError(
            "v9 RPC closure has a dev-overlapping Multi component shortfall: "
            f"actual={len(nondev_multi_component_ids)} "
            f"required={CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT}"
        )

    receipt = PortfolioCoreV9MultiClosureCatalogReceipt(
        closure_inventory_manifest_sha256=closure_manifest_sha256,
        parent_inventory_sha256=parent_selection.inventory_sha256,
        parent_selection_manifest_sha256=parent_selection_sha256,
        parent_catalog_sha256=parent_catalog.manifest.catalog_sha256,
        parent_catalog_manifest_sha256=parent_catalog_manifest_sha256,
        parent_v8_assignment_sha256=closure.parent_v8_assignment_sha256,
        parent_r1_core_plan_sha256=r1_manifest.plan_sha256,
        v9_inventory_sha256=v9_selection.inventory_sha256,
        v9_selection_manifest_sha256=v9_selection_sha256,
        v9_catalog_sha256=v9_catalog.manifest.catalog_sha256,
        v9_catalog_manifest_sha256=v9_catalog_manifest_sha256,
        parent_v8_assignment_count=len(parent_v8_assignments),
        parent_asset_count=parent_catalog.manifest.asset_count,
        parent_component_count=parent_catalog.manifest.component_count,
        v9_asset_count=v9_catalog.manifest.asset_count,
        v9_component_count=v9_catalog.manifest.component_count,
        component_count_delta=component_count_delta,
        selection_delta_candidate_ids=delta_candidate_ids,
        selection_delta_asset_ids=delta_asset_ids,
        new_multi_component_ids=pure_new_multi_component_ids,
        new_nondev_multi_component_ids=nondev_multi_component_ids,
    )
    receipt_bytes = canonical_json_bytes(receipt)
    output = _publish_idempotent_file(Path(output_path).absolute(), receipt_bytes)
    loaded_receipt, receipt_sha256 = (
        load_portfolio_core_v9_multi_closure_catalog_receipt(output)
    )
    if loaded_receipt != receipt:
        raise PortfolioCoreAssetError("published v9 closure receipt changed on reload")
    return PortfolioCoreV9MultiClosureCatalogReceiptResult(
        output_path=output,
        output_sha256=receipt_sha256,
        component_count_delta=component_count_delta,
        new_nondev_multi_component_ids=nondev_multi_component_ids,
    )


def build_portfolio_core_v9_multi_capability_assignments(
    *,
    closure_catalog_receipt_path: str | Path,
    selection_manifest_path: str | Path,
    asset_catalog_dir: str | Path,
    asset_root: str | Path,
    parent_v8_assignments_path: str | Path,
    output_path: str | Path,
) -> PortfolioCoreAssignmentResult:
    """Append only v9 RPC Multi assignments to the immutable v8 assignment set."""

    receipt, receipt_sha256 = load_portfolio_core_v9_multi_closure_catalog_receipt(
        closure_catalog_receipt_path
    )
    (
        selection,
        selection_sha256,
        catalog,
        _draft_by_path,
        catalog_manifest_sha256,
    ) = _load_selection_catalog_context(
        selection_manifest_path=selection_manifest_path,
        asset_catalog_dir=asset_catalog_dir,
        asset_root=asset_root,
    )
    if (
        selection.inventory_sha256 != receipt.v9_inventory_sha256
        or selection_sha256 != receipt.v9_selection_manifest_sha256
        or catalog.manifest.catalog_sha256 != receipt.v9_catalog_sha256
        or catalog_manifest_sha256 != receipt.v9_catalog_manifest_sha256
    ):
        raise PortfolioCoreAssetError(
            "v9 assignment inputs do not bind the closure catalog receipt"
        )
    parent_assignments = load_capability_assignments(
        parent_v8_assignments_path,
        expected_sha256=receipt.parent_v8_assignment_sha256,
    )
    if len(parent_assignments) != receipt.parent_v8_assignment_count:
        raise PortfolioCoreAssetError("parent v8 assignment count drifted")

    rows_by_candidate = {row.candidate_id: row for row in selection.selections}
    expected_ids = set(receipt.selection_delta_candidate_ids)
    if not expected_ids <= set(rows_by_candidate):
        raise PortfolioCoreAssetError("v9 assignment selection delta row is missing")
    expected_binding = CandidateCapabilityBinding(
        canonical_intent="multi_product",
        canonical_capability="product.multi_search",
    )
    taxonomy = load_default_taxonomy_registry()
    task_spec = load_mvp_task_specification_v1()
    capability = taxonomy.capabilities_by_id[expected_binding.canonical_capability]
    task = task_spec.capabilities_by_id[expected_binding.canonical_capability]
    validate_capability_binding(
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=PHASE3_TASK_SPEC_VERSION,
        canonical_intent=expected_binding.canonical_intent,
        canonical_capability=expected_binding.canonical_capability,
        acceptable_capabilities=(expected_binding.canonical_capability,),
        requires_card=capability.requires_card,
        allowed_tools=task.allowed_tools,
        taxonomy=taxonomy,
        task_specification=task_spec,
    )

    assignments = list(parent_assignments)
    existing_keys = {
        (item.asset_id, item.image_path, item.canonical_intent) for item in assignments
    }
    existing_assignment_ids = {item.assignment_id for item in assignments}
    parent_assignment_ids = set(existing_assignment_ids)
    added_asset_ids: set[str] = set()
    for candidate_id in receipt.selection_delta_candidate_ids:
        row = rows_by_candidate[candidate_id]
        if (
            row.source_id != "rpc"
            or row.pool != "multi_product"
            or row.capability_bindings != (expected_binding,)
        ):
            raise PortfolioCoreAssetError(
                "v9 assignment candidate drifted from the frozen RPC Multi role: "
                + candidate_id
            )
        resolution = catalog.resolve_path(row.destination_path)
        asset = resolution.asset
        if asset.asset_id not in receipt.selection_delta_asset_ids:
            raise PortfolioCoreAssetError(
                "v9 assignment candidate asset is absent from the closure receipt: "
                + candidate_id
            )
        added_asset_ids.add(asset.asset_id)
        key = (asset.asset_id, row.destination_path, expected_binding.canonical_intent)
        if key in existing_keys:
            raise PortfolioCoreAssetError(
                "v9 Multi closure duplicates an inherited assignment: " + candidate_id
            )
        identity = {
            "asset_id": asset.asset_id,
            "binding": expected_binding.model_dump(mode="json"),
            "candidate_id": candidate_id,
            "closure_catalog_receipt_sha256": receipt_sha256,
            "policy_version": PORTFOLIO_CORE_V9_MULTI_CLOSURE_POLICY_VERSION,
        }
        assignment = CapabilityAssignment(
            assignment_id=(
                "portfolio-core.assignment."
                + sha256_bytes(canonical_json_bytes(identity))
            ),
            asset_id=asset.asset_id,
            image_path=row.destination_path,
            canonical_intent=expected_binding.canonical_intent,
            canonical_capability=expected_binding.canonical_capability,
            acceptable_capabilities=(expected_binding.canonical_capability,),
            requires_card=capability.requires_card,
            allowed_tools=task.allowed_tools,
            taxonomy_version=TAXONOMY_VERSION,
            taxonomy_sha256=DEFAULT_TAXONOMY_SHA256,
            task_spec_version=PHASE3_TASK_SPEC_VERSION,
            task_spec_sha256=PHASE3_TASK_SPEC_SHA256,
            source_ref="project-choice:portfolio-core-v9-rpc-multi-closure-v1",
            source_artifact_sha256=receipt_sha256,
            rationale=(
                "Approved bounded v9 RPC Multi capacity closure; inherited v8 "
                "bindings remain byte-for-byte unchanged."
            ),
        )
        if assignment.assignment_id in existing_assignment_ids:
            raise PortfolioCoreAssetError(
                "v9 Multi closure duplicates an inherited assignment ID: "
                + candidate_id
            )
        assignments.append(assignment)
        existing_keys.add(key)
        existing_assignment_ids.add(assignment.assignment_id)
    if set(receipt.selection_delta_asset_ids) != added_asset_ids:
        raise PortfolioCoreAssetError(
            "v9 Multi assignment assets do not exactly match the closure receipt"
        )
    assignments.sort(
        key=lambda item: (
            item.image_path,
            item.canonical_intent,
            item.canonical_capability,
        )
    )
    assignment_bytes = canonical_jsonl_bytes(assignments)
    output = _publish_idempotent_file(Path(output_path).absolute(), assignment_bytes)
    output_sha256 = sha256_bytes(assignment_bytes)
    loaded = load_capability_assignments(output, expected_sha256=output_sha256)
    if canonical_jsonl_bytes(loaded) != assignment_bytes:
        raise PortfolioCoreAssetError("published v9 assignments changed on reload")
    loaded_by_id = {item.assignment_id: item for item in loaded}
    parent_by_id = {item.assignment_id: item for item in parent_assignments}
    if not parent_assignment_ids <= set(loaded_by_id) or any(
        loaded_by_id[item_id] != parent for item_id, parent in parent_by_id.items()
    ):
        raise PortfolioCoreAssetError(
            "v9 assignments did not retain all parent v8 rows"
        )
    return PortfolioCoreAssignmentResult(
        output_path=output,
        output_sha256=output_sha256,
        assignment_count=len(assignments),
        selection_manifest_sha256=selection_sha256,
        multi_closure_catalog_receipt_sha256=receipt_sha256,
    )


def _validate_candidate_set(candidates: Sequence[PortfolioCoreAssetCandidate]) -> None:
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    destinations = [candidate.destination_path for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise PortfolioCoreAssetError(
            "candidate inventory contains duplicate candidate_id"
        )
    if len(destinations) != len(set(destinations)):
        raise PortfolioCoreAssetError(
            "candidate inventory contains duplicate destination_path"
        )
    for candidate in candidates:
        if candidate.selection == "include":
            ceiling = _SOURCE_CAPABILITY_CEILINGS.get(candidate.source_id)
            if ceiling is None:
                raise PortfolioCoreAssetError(
                    f"candidate {candidate.candidate_id} uses an unregistered Portfolio source"
                )
            requested = {
                (binding.canonical_intent, binding.canonical_capability)
                for binding in candidate.capability_bindings
            }
            if not requested <= ceiling:
                raise PortfolioCoreAssetError(
                    f"candidate {candidate.candidate_id} broadens the frozen source/capability role"
                )
        for binding in candidate.capability_bindings:
            try:
                validate_capability_binding(
                    taxonomy_version=TAXONOMY_VERSION,
                    task_spec_version=PHASE3_TASK_SPEC_VERSION,
                    canonical_intent=binding.canonical_intent,
                    canonical_capability=binding.canonical_capability,
                    acceptable_capabilities=(binding.canonical_capability,),
                    requires_card=(
                        load_default_taxonomy_registry()
                        .capabilities_by_id[binding.canonical_capability]
                        .requires_card
                    ),
                )
            except (KeyError, ValueError) as exc:
                raise PortfolioCoreAssetError(
                    f"candidate {candidate.candidate_id} capability binding is invalid"
                ) from exc


def _validate_source_policy_ceiling(
    policy: PortfolioCoreSourcePolicyDocument,
) -> None:
    if set(_SOURCE_CAPABILITY_CEILINGS) != set(_SOURCE_POOL_CEILINGS):
        raise PortfolioCoreAssetError(
            "code-level source pool and capability ceilings are inconsistent"
        )
    for entry in policy.sources:
        ceiling = _SOURCE_POOL_CEILINGS.get(entry.source_id)
        if ceiling is None:
            raise PortfolioCoreAssetError(
                f"source policy contains an unregistered Portfolio source: {entry.source_id}"
            )
        if not set(entry.allowed_pools) <= set(ceiling):
            raise PortfolioCoreAssetError(
                f"source policy broadens the frozen role for {entry.source_id}"
            )
        if (
            entry.source_id == "u_need"
            and entry.availability != "permanently_unavailable"
        ):
            raise PortfolioCoreAssetError("u_need must remain permanently unavailable")


def _selected_candidates(
    candidates: Iterable[PortfolioCoreAssetCandidate],
    *,
    activated_reserve_candidate_ids: frozenset[str] = frozenset(),
) -> tuple[PortfolioCoreAssetCandidate, ...]:
    return tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if (
                    candidate.selection == "include"
                    or candidate.candidate_id in activated_reserve_candidate_ids
                )
            ),
            key=lambda candidate: (
                _POOL_ORDER.index(candidate.pool),
                candidate.destination_path,
                candidate.candidate_id,
            ),
        )
    )


def _normalize_source_roots(
    source_roots: Mapping[str, str | Path],
) -> dict[str, Path]:
    normalized: dict[str, Path] = {}
    for source_id, root in source_roots.items():
        if not _SOURCE_ID_RE.fullmatch(source_id):
            raise PortfolioCoreAssetError(
                f"source root has invalid source id: {source_id}"
            )
        if source_id in normalized:
            raise PortfolioCoreAssetError(f"duplicate source root: {source_id}")
        normalized[source_id] = Path(root).absolute()
    return normalized


def _require_disjoint_output_root(
    output_root: Path,
    source_roots: Sequence[Path],
) -> None:
    """Reject every source/output ancestor overlap before creating output bytes."""

    try:
        resolved_output = output_root.resolve(strict=False)
    except OSError as exc:
        raise PortfolioCoreAssetError(
            f"cannot resolve portfolio core output root: {output_root}"
        ) from exc
    seen_sources: set[Path] = set()
    for source_root in source_roots:
        try:
            resolved_source = source_root.resolve(strict=True)
        except OSError as exc:
            raise PortfolioCoreAssetError(
                f"cannot resolve source root for containment check: {source_root}"
            ) from exc
        if resolved_source in seen_sources:
            continue
        seen_sources.add(resolved_source)
        if _paths_overlap(resolved_output, resolved_source):
            raise PortfolioCoreAssetError(
                "portfolio core output root must be disjoint from every source root: "
                f"{source_root}"
            )


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _read_and_verify_source_candidate(
    candidate: PortfolioCoreAssetCandidate,
    source_root: Path,
) -> bytes:
    path = _resolve_beneath_root(
        source_root,
        candidate.source_local_path,
        label=f"source candidate {candidate.candidate_id}",
    )
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise PortfolioCoreAssetError(
            f"candidate {candidate.candidate_id} source has an unsupported image suffix"
        )
    try:
        content = read_stable_regular_file(
            path,
            label=f"source candidate {candidate.candidate_id}",
            max_bytes=_MAX_ASSET_BYTES,
        )
    except ArtifactFormatError as exc:
        raise PortfolioCoreAssetError(str(exc)) from exc
    if len(content) != candidate.expected_bytes:
        raise PortfolioCoreAssetError(
            f"candidate {candidate.candidate_id} source_bytes_mismatch"
        )
    if sha256_bytes(content) != candidate.expected_sha256:
        raise PortfolioCoreAssetError(
            f"candidate {candidate.candidate_id} source_sha256_mismatch"
        )
    return content


def _resolve_beneath_root(root: Path, local_path: str, *, label: str) -> Path:
    _ensure_real_directory(root, create=False, label=f"{label} root")
    normalized = _canonical_relative_path(local_path, "local path")
    current = root
    parts = PurePosixPath(normalized).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise PortfolioCoreAssetError(
                f"{label} is unavailable: {local_path}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PortfolioCoreAssetError(
                f"{label} path contains a symlink: {local_path}"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise PortfolioCoreAssetError(
                f"{label} parent is not a directory: {local_path}"
            )
    if not stat.S_ISREG(current.lstat().st_mode):
        raise PortfolioCoreAssetError(f"{label} is not a regular file: {local_path}")
    return current


def _resolve_output_path(output_root: Path, destination_path: str) -> Path:
    normalized = _canonical_relative_path(destination_path, "destination_path")
    destination = output_root.joinpath(*PurePosixPath(normalized).parts)
    parent = destination.parent
    _ensure_real_directory(parent, create=True, label="destination parent")
    return destination


def _ensure_real_directory(path: Path, *, create: bool, label: str) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PortfolioCoreAssetError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PortfolioCoreAssetError(f"{label} must be a real directory: {path}")


@contextmanager
def _exclusive_materialization_writer_lock(output_root: Path) -> Iterator[None]:
    """Fail closed when a second materializer targets the same output root.

    An invocation timeout in a supervising shell does not prove that its child
    worker stopped.  The lock is therefore deliberately retained if a worker
    is killed before normal cleanup.  An operator must inspect the recorded
    PID's command line, confirm that worker exited, and explicitly resolve a
    retained lock before resuming.
    """

    lock_path = output_root / _MATERIALIZATION_WRITER_LOCK_FILE
    owner = canonical_json_bytes(
        {
            "owner_pid": os.getpid(),
            "owner_token": os.urandom(16).hex(),
        }
    )
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
    except FileExistsError as exc:
        raise PortfolioCoreAssetError(
            "materialization writer lock already exists: "
            f"{lock_path}; do not resume after a caller timeout until the "
            "lock owner PID is absent from the process list (inspect its "
            "command line) and the retained lock has been explicitly resolved"
        ) from exc
    except OSError as exc:
        raise PortfolioCoreAssetError(
            f"cannot acquire materialization writer lock: {lock_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(owner)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Preserve a partially written lock rather than risking a concurrent
        # writer after a failed acquisition.
        raise

    try:
        yield
    finally:
        try:
            retained = read_stable_regular_file(
                lock_path,
                label="materialization writer lock",
                max_bytes=_MAX_METADATA_BYTES,
            )
        except ArtifactFormatError as exc:
            raise PortfolioCoreAssetError(
                "materialization writer lock is unreadable; preserving it "
                f"for explicit recovery: {lock_path}"
            ) from exc
        if retained != owner:
            raise PortfolioCoreAssetError(
                "materialization writer lock ownership changed; preserving it "
                f"for explicit recovery: {lock_path}"
            )
        try:
            lock_path.unlink()
        except OSError as exc:
            raise PortfolioCoreAssetError(
                "cannot release materialization writer lock; preserving it "
                f"for explicit recovery: {lock_path}"
            ) from exc


def _candidate_digest(candidate: PortfolioCoreAssetCandidate) -> str:
    return sha256_bytes(canonical_json_bytes(candidate))


def _published_draft(candidate: PortfolioCoreAssetCandidate) -> DatasetAssetDraft:
    return candidate.draft.model_copy(
        update={
            "local_path": candidate.destination_path,
            "derivation_parent_asset_ids": [],
            "derivation_parent_asset_id": None,
        }
    )


def _initialize_or_verify_run(
    *,
    output_root: Path,
    inventory_sha256: str,
    source_policy_sha256: str,
    selected: Sequence[PortfolioCoreAssetCandidate],
) -> None:
    manifest = PortfolioCoreAssetRunManifest(
        inventory_sha256=inventory_sha256,
        source_policy_sha256=source_policy_sha256,
        selected_candidate_count=len(selected),
        selected_source_counts=dict(
            sorted(Counter(candidate.source_id for candidate in selected).items())
        ),
    )
    _publish_idempotent_file(
        output_root / _RUN_MANIFEST_FILE,
        canonical_json_bytes(manifest),
    )


def _publish_or_verify_asset(
    destination: Path,
    content: bytes,
    candidate: PortfolioCoreAssetCandidate,
) -> None:
    if os.path.lexists(destination):
        _verify_existing_asset(destination, candidate)
        return
    try:
        atomic_create_file(destination, content)
    except FileExistsError:
        _publish_or_verify_asset(destination, content, candidate)


def _verify_existing_asset(
    destination: Path,
    candidate: PortfolioCoreAssetCandidate,
) -> None:
    if destination.is_symlink() or not destination.is_file():
        raise FileExistsError(f"destination is not a regular file: {destination}")
    try:
        existing = read_stable_regular_file(
            destination,
            label=f"existing destination {candidate.candidate_id}",
            max_bytes=_MAX_ASSET_BYTES,
        )
    except ArtifactFormatError as exc:
        raise PortfolioCoreAssetError(str(exc)) from exc
    if (
        len(existing) != candidate.expected_bytes
        or sha256_bytes(existing) != candidate.expected_sha256
    ):
        raise FileExistsError(
            f"existing destination conflicts with candidate: {destination}"
        )


def _verify_existing_checkpoint(
    output_root: Path,
    checkpoint_path: Path,
    candidate: PortfolioCoreAssetCandidate,
) -> None:
    content = _read_metadata(checkpoint_path, f"checkpoint {candidate.candidate_id}")
    try:
        checkpoint = PortfolioCoreAssetCheckpoint.model_validate_json(content)
    except ValidationError as exc:
        raise PortfolioCoreAssetError(
            f"checkpoint is invalid: {checkpoint_path}"
        ) from exc
    if content != canonical_json_bytes(checkpoint):
        raise PortfolioCoreAssetError(f"checkpoint is not canonical: {checkpoint_path}")
    expected = PortfolioCoreAssetCheckpoint(
        candidate_id=candidate.candidate_id,
        candidate_sha256=_candidate_digest(candidate),
        destination_path=candidate.destination_path,
        expected_bytes=candidate.expected_bytes,
        expected_sha256=candidate.expected_sha256,
        draft=_published_draft(candidate),
    )
    if checkpoint != expected:
        raise PortfolioCoreAssetError(
            f"checkpoint does not match immutable candidate: {candidate.candidate_id}"
        )
    destination = _resolve_output_path(output_root, candidate.destination_path)
    if not os.path.lexists(destination):
        raise PortfolioCoreAssetError(
            f"checkpointed destination is missing: {candidate.destination_path}"
        )
    _verify_existing_asset(destination, candidate)


def _finalize_materialization(
    *,
    output_root: Path,
    selected: Sequence[PortfolioCoreAssetCandidate],
    inventory_sha256: str,
    source_policy_sha256: str,
) -> tuple[Path, Path]:
    checkpoint_root = output_root / _CHECKPOINT_DIRECTORY
    drafts: list[DatasetAssetDraft] = []
    rows: list[PortfolioCoreSelectionRow] = []
    for candidate in selected:
        checkpoint_path = checkpoint_root / f"{_candidate_digest(candidate)}.json"
        _verify_existing_checkpoint(output_root, checkpoint_path, candidate)
        draft = _published_draft(candidate)
        drafts.append(draft)
        rows.append(
            PortfolioCoreSelectionRow(
                candidate_id=candidate.candidate_id,
                candidate_sha256=_candidate_digest(candidate),
                destination_path=candidate.destination_path,
                expected_bytes=candidate.expected_bytes,
                expected_sha256=candidate.expected_sha256,
                source_id=candidate.source_id,
                pool=candidate.pool,
                draft=draft,
                capability_bindings=candidate.capability_bindings,
            )
        )
    drafts_bytes = canonical_jsonl_bytes(drafts)
    drafts_path = _publish_idempotent_file(output_root / _DRAFTS_FILE, drafts_bytes)
    manifest = PortfolioCoreSelectionManifest(
        inventory_sha256=inventory_sha256,
        source_policy_sha256=source_policy_sha256,
        dataset_assets_sha256=sha256_bytes(drafts_bytes),
        dataset_asset_count=len(drafts),
        selections=tuple(rows),
    )
    selection_path = _publish_idempotent_file(
        output_root / _SELECTION_MANIFEST_FILE,
        canonical_json_bytes(manifest),
    )
    return drafts_path, selection_path


def _load_published_drafts(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[DatasetAssetDraft, ...]:
    content = _read_metadata(path, "published DatasetAssetDraft JSONL")
    if sha256_bytes(content) != expected_sha256:
        raise PortfolioCoreAssetError("published draft digest does not match selection")
    try:
        rows = parse_canonical_jsonl(content, label="published DatasetAssetDraft JSONL")
        drafts = tuple(
            DatasetAssetDraft.model_validate(row, strict=True) for row in rows
        )
    except (ArtifactFormatError, ValidationError) as exc:
        raise PortfolioCoreAssetError("published drafts violate schema") from exc
    if content != canonical_jsonl_bytes(drafts):
        raise PortfolioCoreAssetError("published drafts must be canonical JSONL")
    return drafts


def _verify_catalog_asset(row: PortfolioCoreSelectionRow, asset: Any) -> None:
    if (
        asset.local_path != row.destination_path
        or asset.sha256 != row.expected_sha256
        or asset.source_dataset != row.draft.source_dataset
        or asset.source_revision != row.draft.source_revision
        or asset.source_record_id != row.draft.source_record_id
        or asset.transform_policy_version != row.draft.transform_policy_version
        or asset.product_id != row.draft.product_id
        or asset.license_id != row.draft.license_id
        or asset.source_url != row.draft.source_url
        or asset.attribution != row.draft.attribution
        or asset.cloud_upload_allowed != row.draft.cloud_upload_allowed
        or asset.public_demo_allowed != row.draft.public_demo_allowed
    ):
        raise PortfolioCoreAssetError(
            f"asset catalog binding drifted: {row.destination_path}"
        )


def _read_metadata(path: Path, label: str) -> bytes:
    try:
        return read_stable_regular_file(
            path, label=label, max_bytes=_MAX_METADATA_BYTES
        )
    except ArtifactFormatError as exc:
        raise PortfolioCoreAssetError(str(exc)) from exc


def _publish_idempotent_file(path: Path, content: bytes) -> Path:
    if os.path.lexists(path):
        existing = _read_metadata(path, f"existing artifact {path.name}")
        if existing != content:
            raise FileExistsError(
                f"existing artifact conflicts with deterministic output: {path}"
            )
        return path
    try:
        return atomic_create_file(path, content)
    except FileExistsError:
        return _publish_idempotent_file(path, content)


def _parse_source_root_args(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        source_id, separator, raw_path = value.partition("=")
        if not separator or not source_id or not raw_path:
            raise ValueError("--source-root must use source_id=PATH")
        if source_id in roots:
            raise ValueError(f"duplicate --source-root source id: {source_id}")
        roots[source_id] = Path(raw_path)
    return roots


def _emit(payload: Mapping[str, Any], *, stream) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        file=stream,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable Portfolio-only core clean-asset preparation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    policy = subparsers.add_parser("write-default-policy")
    policy.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--inventory", type=Path, required=True)
    preflight.add_argument("--source-policy", type=Path, required=True)
    preflight.add_argument("--source-root", action="append", default=[])
    preflight.add_argument("--capacity-profile", choices=("r1", "r2"), default="r1")
    preflight.add_argument("--document-selection-manifest", type=Path)
    preflight.add_argument("--reserve-activation-manifest", type=Path)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--inventory", type=Path, required=True)
    materialize.add_argument("--source-policy", type=Path, required=True)
    materialize.add_argument("--source-root", action="append", default=[])
    materialize.add_argument("--capacity-profile", choices=("r1", "r2"), default="r1")
    materialize.add_argument("--document-selection-manifest", type=Path)
    materialize.add_argument("--reserve-activation-manifest", type=Path)
    materialize.add_argument("--output-root", type=Path, required=True)
    materialize.add_argument("--max-items", type=int)
    materialize.add_argument(
        "--allow-incomplete-capacity",
        action="store_true",
        help="materialize a bounded trial inventory without declaring it core-ready",
    )

    assignments = subparsers.add_parser("build-assignments")
    assignments.add_argument("--selection-manifest", type=Path, required=True)
    assignments.add_argument("--asset-catalog", type=Path, required=True)
    assignments.add_argument("--asset-root", type=Path, required=True)
    assignments.add_argument("--output", type=Path, required=True)

    binding_plan = subparsers.add_parser("build-v8-binding-plan")
    binding_plan.add_argument("--selection-manifest", type=Path, required=True)
    binding_plan.add_argument("--asset-catalog", type=Path, required=True)
    binding_plan.add_argument("--asset-root", type=Path, required=True)
    binding_plan.add_argument("--parent-assignments", type=Path, required=True)
    binding_plan.add_argument("--parent-assignment-sha256", required=True)
    binding_plan.add_argument("--parent-r1-core-plan", type=Path, required=True)
    binding_plan.add_argument("--output", type=Path, required=True)

    v8_assignments = subparsers.add_parser("build-v8-assignments")
    v8_assignments.add_argument("--selection-manifest", type=Path, required=True)
    v8_assignments.add_argument("--asset-catalog", type=Path, required=True)
    v8_assignments.add_argument("--asset-root", type=Path, required=True)
    v8_assignments.add_argument("--parent-assignments", type=Path, required=True)
    v8_assignments.add_argument("--parent-r1-core-plan", type=Path, required=True)
    v8_assignments.add_argument("--binding-plan-manifest", type=Path, required=True)
    v8_assignments.add_argument("--output", type=Path, required=True)

    v9_inventory = subparsers.add_parser("build-v9-multi-inventory")
    v9_inventory.add_argument("--parent-inventory", type=Path, required=True)
    v9_inventory.add_argument("--parent-inventory-sha256", required=True)
    v9_inventory.add_argument(
        "--parent-reserve-activation-manifest", type=Path, required=True
    )
    v9_inventory.add_argument("--parent-v8-assignments", type=Path, required=True)
    v9_inventory.add_argument("--parent-v8-assignment-sha256", required=True)
    v9_inventory.add_argument("--source-policy", type=Path, required=True)
    v9_inventory.add_argument("--rpc-staging-candidates", type=Path, required=True)
    v9_inventory.add_argument(
        "--rpc-staging-adapter-manifest", type=Path, required=True
    )
    v9_inventory.add_argument("--output-inventory", type=Path, required=True)
    v9_inventory.add_argument(
        "--output-reserve-activation-manifest", type=Path, required=True
    )
    v9_inventory.add_argument("--output-manifest", type=Path, required=True)

    v9_receipt = subparsers.add_parser("build-v9-multi-receipt")
    v9_receipt.add_argument("--closure-inventory-manifest", type=Path, required=True)
    v9_receipt.add_argument("--parent-selection-manifest", type=Path, required=True)
    v9_receipt.add_argument("--parent-asset-catalog", type=Path, required=True)
    v9_receipt.add_argument("--parent-asset-root", type=Path, required=True)
    v9_receipt.add_argument("--parent-v8-assignments", type=Path, required=True)
    v9_receipt.add_argument("--parent-r1-core-plan", type=Path, required=True)
    v9_receipt.add_argument("--v9-selection-manifest", type=Path, required=True)
    v9_receipt.add_argument("--v9-asset-catalog", type=Path, required=True)
    v9_receipt.add_argument("--v9-asset-root", type=Path, required=True)
    v9_receipt.add_argument("--output", type=Path, required=True)

    v9_assignments = subparsers.add_parser("build-v9-multi-assignments")
    v9_assignments.add_argument("--closure-catalog-receipt", type=Path, required=True)
    v9_assignments.add_argument("--selection-manifest", type=Path, required=True)
    v9_assignments.add_argument("--asset-catalog", type=Path, required=True)
    v9_assignments.add_argument("--asset-root", type=Path, required=True)
    v9_assignments.add_argument("--parent-v8-assignments", type=Path, required=True)
    v9_assignments.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "write-default-policy":
            path = write_default_source_policy(args.output)
            _emit(
                {
                    "formal_eligible": False,
                    "formal_status": "non_formal",
                    "output": str(path),
                    "track": "portfolio",
                },
                stream=os.sys.stdout,
            )
            return 0
        if args.command == "preflight":
            result = preflight_portfolio_core_assets(
                inventory_path=args.inventory,
                source_policy_path=args.source_policy,
                source_roots=_parse_source_root_args(args.source_root),
                capacity_profile=args.capacity_profile,
                document_selection_manifest_path=args.document_selection_manifest,
                reserve_activation_manifest_path=args.reserve_activation_manifest,
            )
            _emit(result.as_json(), stream=os.sys.stdout)
            return 0 if result.ready else 2
        if args.command == "materialize":
            result = materialize_portfolio_core_assets(
                inventory_path=args.inventory,
                source_policy_path=args.source_policy,
                source_roots=_parse_source_root_args(args.source_root),
                output_root=args.output_root,
                max_items=args.max_items,
                allow_incomplete_capacity=args.allow_incomplete_capacity,
                capacity_profile=args.capacity_profile,
                document_selection_manifest_path=args.document_selection_manifest,
                reserve_activation_manifest_path=args.reserve_activation_manifest,
            )
            _emit(result.as_json(), stream=os.sys.stdout)
            return 0
        if args.command == "build-assignments":
            result = build_portfolio_core_capability_assignments(
                selection_manifest_path=args.selection_manifest,
                asset_catalog_dir=args.asset_catalog,
                asset_root=args.asset_root,
                output_path=args.output,
            )
            _emit(result.as_json(), stream=os.sys.stdout)
            return 0
        if args.command == "build-v8-binding-plan":
            result = build_portfolio_core_v8_capability_binding_plan(
                selection_manifest_path=args.selection_manifest,
                asset_catalog_dir=args.asset_catalog,
                asset_root=args.asset_root,
                parent_assignments_path=args.parent_assignments,
                parent_assignment_sha256=args.parent_assignment_sha256,
                parent_r1_core_plan_path=args.parent_r1_core_plan,
                output_path=args.output,
            )
            _emit(result.as_json(), stream=os.sys.stdout)
            return 0
        if args.command == "build-v8-assignments":
            result = build_portfolio_core_v8_capability_assignments(
                selection_manifest_path=args.selection_manifest,
                asset_catalog_dir=args.asset_catalog,
                asset_root=args.asset_root,
                parent_assignments_path=args.parent_assignments,
                parent_r1_core_plan_path=args.parent_r1_core_plan,
                binding_plan_manifest_path=args.binding_plan_manifest,
                output_path=args.output,
            )
            _emit(result.as_json(), stream=os.sys.stdout)
            return 0
        if args.command == "build-v9-multi-inventory":
            result = build_portfolio_core_v9_multi_closure_inventory(
                parent_inventory_path=args.parent_inventory,
                parent_inventory_sha256=args.parent_inventory_sha256,
                parent_reserve_activation_manifest_path=(
                    args.parent_reserve_activation_manifest
                ),
                parent_v8_assignments_path=args.parent_v8_assignments,
                parent_v8_assignment_sha256=args.parent_v8_assignment_sha256,
                v9_source_policy_path=args.source_policy,
                rpc_staging_candidates_path=args.rpc_staging_candidates,
                rpc_staging_adapter_manifest_path=(args.rpc_staging_adapter_manifest),
                output_inventory_path=args.output_inventory,
                output_reserve_activation_manifest_path=(
                    args.output_reserve_activation_manifest
                ),
                output_manifest_path=args.output_manifest,
            )
            _emit(result.as_json(), stream=os.sys.stdout)
            return 0
        if args.command == "build-v9-multi-receipt":
            result = build_portfolio_core_v9_multi_closure_catalog_receipt(
                closure_inventory_manifest_path=args.closure_inventory_manifest,
                parent_selection_manifest_path=args.parent_selection_manifest,
                parent_asset_catalog_dir=args.parent_asset_catalog,
                parent_asset_root=args.parent_asset_root,
                parent_v8_assignments_path=args.parent_v8_assignments,
                parent_r1_core_plan_path=args.parent_r1_core_plan,
                v9_selection_manifest_path=args.v9_selection_manifest,
                v9_asset_catalog_dir=args.v9_asset_catalog,
                v9_asset_root=args.v9_asset_root,
                output_path=args.output,
            )
            _emit(result.as_json(), stream=os.sys.stdout)
            return 0
        if args.command == "build-v9-multi-assignments":
            result = build_portfolio_core_v9_multi_capability_assignments(
                closure_catalog_receipt_path=args.closure_catalog_receipt,
                selection_manifest_path=args.selection_manifest,
                asset_catalog_dir=args.asset_catalog,
                asset_root=args.asset_root,
                parent_v8_assignments_path=args.parent_v8_assignments,
                output_path=args.output,
            )
            _emit(result.as_json(), stream=os.sys.stdout)
            return 0
        raise AssertionError(f"unhandled command: {args.command}")
    except (OSError, ValueError, PortfolioCoreAssetError) as exc:
        _emit({"error": str(exc), "status": "error"}, stream=os.sys.stderr)
        return 2


__all__ = [
    "CORE_R2_CAPABILITY_ASSET_FLOORS",
    "CORE_R2_DOCUMENT_CARRY_FORWARD_FLOOR",
    "CORE_R2_DOCUMENT_CORD_FRESH_FLOOR",
    "CORE_R2_DOCUMENT_CORD_RESERVE_FLOOR",
    "CORE_R2_DOCUMENT_FRESH_FLOOR",
    "CORE_R2_DOCUMENT_SROIE_FRESH_FLOOR",
    "CORE_R2_DOCUMENT_SROIE_RESERVE_FLOOR",
    "CORE_R2_V8_ABO_TRIPLET_ADDITION_COUNT",
    "CORE_R2_V8_ISIA_FOOD_PAIR_ADDITION_COUNT",
    "CORE_R2_V9_FRESH_RPC_MULTI_COUNT",
    "CORE_R2_V9_MULTI_CLOSURE_DELTA_COUNT",
    "CORE_R2_V9_PARENT_RPC_INCLUDE_COUNT",
    "CORE_R2_V9_PARENT_RPC_RESERVE_COUNT",
    "CORE_R2_V9_RPC_STAGING_INCLUDE_COUNT",
    "CORE_R2_V9_RPC_STAGING_RESERVE_COUNT",
    "DEFAULT_PORTFOLIO_CORE_SOURCE_POLICY",
    "PORTFOLIO_CORE_ASSET_POLICY_VERSION",
    "PORTFOLIO_CORE_ASSIGNMENT_POLICY_VERSION",
    "PORTFOLIO_CORE_COMPONENT_FLOOR_CLOSURE_POLICY_VERSION",
    "PORTFOLIO_CORE_SOURCE_POLICY_VERSION",
    "PORTFOLIO_CORE_V8_CAPABILITY_BINDING_POLICY_VERSION",
    "PORTFOLIO_CORE_V9_MULTI_CLOSURE_POLICY_VERSION",
    "CapabilityBindingPlanProfile",
    "CandidateCapabilityBinding",
    "CoreCapacityProfile",
    "DocumentSelectionRole",
    "PortfolioCoreAssetCandidate",
    "PortfolioCoreAssetError",
    "PortfolioCoreAssetMaterializationResult",
    "PortfolioCoreAssetPreflight",
    "PortfolioCoreAssignmentResult",
    "PortfolioCoreCapabilityBindingPlanEntry",
    "PortfolioCoreCapabilityBindingPlanManifest",
    "PortfolioCoreCapabilityBindingPlanResult",
    "PortfolioCoreDocumentSelectionEntry",
    "PortfolioCoreDocumentSelectionManifest",
    "PortfolioCoreReserveActivationEntry",
    "PortfolioCoreReserveActivationManifest",
    "PortfolioCoreSourcePolicy",
    "PortfolioCoreSourcePolicyDocument",
    "PortfolioCoreV9MultiClosureCatalogReceipt",
    "PortfolioCoreV9MultiClosureCatalogReceiptResult",
    "PortfolioCoreV9MultiClosureInventoryManifest",
    "PortfolioCoreV9MultiClosureInventoryResult",
    "build_portfolio_core_capability_assignments",
    "build_portfolio_core_v9_multi_capability_assignments",
    "build_portfolio_core_v9_multi_closure_catalog_receipt",
    "build_portfolio_core_v9_multi_closure_inventory",
    "build_portfolio_core_v8_capability_assignments",
    "build_portfolio_core_v8_capability_binding_plan",
    "default_source_policy_bytes",
    "document_component_key",
    "load_candidate_inventory",
    "load_portfolio_core_capability_binding_plan_manifest",
    "load_portfolio_core_document_selection_manifest",
    "load_portfolio_core_reserve_activation_manifest",
    "load_portfolio_core_v9_multi_closure_catalog_receipt",
    "load_portfolio_core_v9_multi_closure_inventory_manifest",
    "load_source_policy",
    "main",
    "materialize_portfolio_core_assets",
    "preflight_portfolio_core_assets",
    "write_default_source_policy",
]
