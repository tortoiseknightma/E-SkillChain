"""Independent selected48 AIFast runtime for Portfolio S1 Feedback.

The historical Core remote-processing authorization remains byte-for-byte
unchanged and continues to authorize only its three DashScope processors.
This adapter binds a new owner-approved selected48 scope to the already
verified Static corpus and admits only the exact query/asset/image triples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.data.portfolio_remote_processing import (
    VerifiedPortfolioRemoteProcessingRuntime,
    require_verified_portfolio_remote_processing_runtime,
)
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackAuthorizationV1,
    PortfolioS1FeedbackAuthorizationV2,
    PortfolioS1FeedbackControlV1,
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackSelectionV1,
    VerifiedStaticFeedbackSource,
    build_verified_static_feedback_sources,
    require_verified_static_feedback_source,
    validate_portfolio_s1_feedback_control,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import (
    VerifiedStaticGCSCorpus,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    PortfolioS1QwenFeedbackAuthorizationV3,
)
from skillchain.evaluation.visual_runtime import (
    SelectedFeedbackImageBinding,
    VerifiedSelectedFeedbackRemoteRuntime,
    _make_verified_selected_feedback_remote_runtime,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import read_stable_regular_file


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SELECTED_FEEDBACK_REMOTE_POLICY_VERSION = (
    "portfolio-s1-feedback-selected48-remote-runtime-v1"
)
SELECTED_KIMI_FEEDBACK_REMOTE_POLICY_VERSION = (
    "portfolio-s1-feedback-selected48-remote-runtime-v2"
)
SELECTED_QWEN37_FEEDBACK_REMOTE_POLICY_VERSION = (
    "portfolio-s1-feedback-selected48-remote-runtime-v3"
)


class SelectedFeedbackRemoteRuntimeError(PortfolioS1FeedbackError):
    """The selected48 remote scope or its runtime receipt is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _self_hash(model: BaseModel, field: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(model.model_dump(mode="json", exclude={field}))
    )


class SelectedFeedbackRemoteBindingV1(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=48)
    selection_entry_sha256: Sha256
    query_id: str
    asset_id: str
    image_sha256: Sha256


class PortfolioS1FeedbackRemoteRuntimeReceiptV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-selected48-remote-runtime"] = (
        "portfolio-s1-feedback-selected48-remote-runtime"
    )
    policy_version: Literal["portfolio-s1-feedback-selected48-remote-runtime-v1"] = (
        SELECTED_FEEDBACK_REMOTE_POLICY_VERSION
    )
    processor: Literal["aifast-gemini-feedback"] = "aifast-gemini-feedback"
    selection_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    control_sha256: Sha256
    corpus_sha256: Sha256
    catalog_sha256: Sha256
    selected_count: Literal[48] = 48
    selected_bindings: tuple[SelectedFeedbackRemoteBindingV1, ...]
    selected_binding_set_sha256: Sha256
    receipt_sha256: Sha256

    @field_validator("selected_bindings", mode="before")
    @classmethod
    def _bindings_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if len(self.selected_bindings) != 48 or tuple(
            item.selection_ordinal for item in self.selected_bindings
        ) != tuple(range(1, 49)):
            raise ValueError("selected Feedback runtime must bind ordered48")
        if (
            len({item.query_id for item in self.selected_bindings}) != 48
            or len({item.asset_id for item in self.selected_bindings}) != 48
        ):
            raise ValueError("selected Feedback runtime identities repeat")
        payload = [item.model_dump(mode="json") for item in self.selected_bindings]
        if self.selected_binding_set_sha256 != sha256_bytes(
            canonical_json_bytes(payload)
        ):
            raise ValueError("selected Feedback binding set hash mismatch")
        if self.receipt_sha256 != _self_hash(self, "receipt_sha256"):
            raise ValueError("selected Feedback remote receipt self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioS1KimiFeedbackRemoteRuntimeReceiptV2(_StrictFrozenModel):
    """Selected48 membership derived from the genuine Core Kimi authority."""

    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-feedback-selected48-remote-runtime"] = (
        "portfolio-s1-feedback-selected48-remote-runtime"
    )
    policy_version: Literal["portfolio-s1-feedback-selected48-remote-runtime-v2"] = (
        SELECTED_KIMI_FEEDBACK_REMOTE_POLICY_VERSION
    )
    processor: Literal["dashscope-kimi-feedback"] = "dashscope-kimi-feedback"
    selection_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    control_sha256: Sha256
    corpus_sha256: Sha256
    catalog_sha256: Sha256
    parent_remote_authorization_id: str
    parent_remote_authorization_file_sha256: Sha256
    parent_remote_receipt_file_sha256: Sha256
    parent_remote_receipt_sha256: Sha256
    selected_count: Literal[48] = 48
    selected_bindings: tuple[SelectedFeedbackRemoteBindingV1, ...]
    selected_binding_set_sha256: Sha256
    receipt_sha256: Sha256

    @field_validator("selected_bindings", mode="before")
    @classmethod
    def _bindings_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("parent_remote_authorization_id")
    @classmethod
    def _authorization_id(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("parent remote authorization ID must be canonical")
        return value

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if len(self.selected_bindings) != 48 or tuple(
            item.selection_ordinal for item in self.selected_bindings
        ) != tuple(range(1, 49)):
            raise ValueError("selected Kimi Feedback runtime must bind ordered48")
        if (
            len({item.query_id for item in self.selected_bindings}) != 48
            or len({item.asset_id for item in self.selected_bindings}) != 48
        ):
            raise ValueError("selected Kimi Feedback runtime identities repeat")
        payload = [item.model_dump(mode="json") for item in self.selected_bindings]
        if self.selected_binding_set_sha256 != sha256_bytes(
            canonical_json_bytes(payload)
        ):
            raise ValueError("selected Kimi Feedback binding set hash mismatch")
        if self.receipt_sha256 != _self_hash(self, "receipt_sha256"):
            raise ValueError("selected Kimi Feedback remote receipt self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioS1QwenFeedbackRemoteRuntimeReceiptV3(_StrictFrozenModel):
    """Selected48 Qwen role authority plus verified Core image membership."""

    schema_version: Literal[3] = 3
    kind: Literal["portfolio-s1-feedback-selected48-remote-runtime"] = (
        "portfolio-s1-feedback-selected48-remote-runtime"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-selected48-remote-runtime-v3"
    ] = SELECTED_QWEN37_FEEDBACK_REMOTE_POLICY_VERSION
    processor: Literal["dashscope-qwen37-feedback"] = "dashscope-qwen37-feedback"
    selection_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    control_sha256: Sha256
    corpus_sha256: Sha256
    catalog_sha256: Sha256
    parent_membership_processor: Literal["dashscope-qwen-assistant"] = (
        "dashscope-qwen-assistant"
    )
    parent_remote_authorization_id: str
    parent_remote_authorization_file_sha256: Sha256
    parent_remote_receipt_file_sha256: Sha256
    parent_remote_receipt_sha256: Sha256
    selected_count: Literal[48] = 48
    selected_bindings: tuple[SelectedFeedbackRemoteBindingV1, ...]
    selected_binding_set_sha256: Sha256
    receipt_sha256: Sha256

    @field_validator("selected_bindings", mode="before")
    @classmethod
    def _bindings_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("parent_remote_authorization_id")
    @classmethod
    def _authorization_id(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("parent remote authorization ID must be canonical")
        return value

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if len(self.selected_bindings) != 48 or tuple(
            item.selection_ordinal for item in self.selected_bindings
        ) != tuple(range(1, 49)):
            raise ValueError("selected Qwen Feedback runtime must bind ordered48")
        if (
            len({item.query_id for item in self.selected_bindings}) != 48
            or len({item.asset_id for item in self.selected_bindings}) != 48
        ):
            raise ValueError("selected Qwen Feedback runtime identities repeat")
        payload = [item.model_dump(mode="json") for item in self.selected_bindings]
        if self.selected_binding_set_sha256 != sha256_bytes(
            canonical_json_bytes(payload)
        ):
            raise ValueError("selected Qwen Feedback binding set hash mismatch")
        if self.receipt_sha256 != _self_hash(self, "receipt_sha256"):
            raise ValueError("selected Qwen Feedback remote receipt self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _expected_receipt(
    corpus: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV1,
    authorization: PortfolioS1FeedbackAuthorizationV1,
    control: PortfolioS1FeedbackControlV1,
    sources: tuple[VerifiedStaticFeedbackSource, ...] | None = None,
) -> tuple[
    PortfolioS1FeedbackRemoteRuntimeReceiptV1,
    tuple[SelectedFeedbackImageBinding, ...],
    VerifiedStaticGCSCorpus,
]:
    validate_portfolio_s1_feedback_control(control, selection, authorization)
    selected_sources = sources or build_verified_static_feedback_sources(
        corpus, selection, control
    )
    if (
        len(selected_sources) != 48
        or tuple(item.selection_entry_sha256 for item in selected_sources)
        != tuple(item.entry_sha256 for item in selection.entries)
        or any(
            item.selection_sha256 != selection.selection_sha256
            or item.control_sha256 != control.control_sha256
            for item in selected_sources
        )
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "preverified selected Feedback sources drifted"
        )
    for entry, source in zip(selection.entries, selected_sources, strict=True):
        require_verified_static_feedback_source(source, selection, control, entry)
    verified = selected_sources[0].corpus
    catalog = verified.core_inputs.runtime_asset_catalog()
    catalog.require_verified_files()
    receipt_bindings: list[SelectedFeedbackRemoteBindingV1] = []
    runtime_bindings: list[SelectedFeedbackImageBinding] = []
    for entry, source in zip(selection.entries, selected_sources, strict=True):
        row = source.row
        resolution = catalog.verify_reference(
            entry.asset_id,
            row.query.image_path,
            entry.leakage_group_id,
        )
        if (
            resolution.asset.asset_id != entry.asset_id
            or resolution.asset.sha256 != entry.image_sha256
            or resolution.asset.cloud_upload_allowed is not True
            or source.packet.query_id != entry.query_id
            or source.packet.image.sha256 != entry.image_sha256
        ):
            raise SelectedFeedbackRemoteRuntimeError(
                "selected Feedback query/asset/image membership drifted"
            )
        receipt_bindings.append(
            SelectedFeedbackRemoteBindingV1(
                selection_ordinal=entry.selection_ordinal,
                selection_entry_sha256=entry.entry_sha256,
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )
        runtime_bindings.append(
            SelectedFeedbackImageBinding(
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )
    binding_payload = [item.model_dump(mode="json") for item in receipt_bindings]
    draft = PortfolioS1FeedbackRemoteRuntimeReceiptV1.model_construct(
        selection_sha256=selection.selection_sha256,
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        control_sha256=control.control_sha256,
        corpus_sha256=verified.corpus_sha256,
        catalog_sha256=catalog.catalog_sha256,
        selected_bindings=tuple(receipt_bindings),
        selected_binding_set_sha256=sha256_bytes(canonical_json_bytes(binding_payload)),
        receipt_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"receipt_sha256"})
    receipt = PortfolioS1FeedbackRemoteRuntimeReceiptV1.model_validate_json(
        canonical_json_bytes(
            {
                **unsigned,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            }
        ),
        strict=True,
    )
    return receipt, tuple(runtime_bindings), verified


def _expected_kimi_receipt(
    corpus: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV1,
    authorization: PortfolioS1FeedbackAuthorizationV2,
    control: PortfolioS1FeedbackControlV1,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    sources: tuple[VerifiedStaticFeedbackSource, ...] | None = None,
) -> tuple[
    PortfolioS1KimiFeedbackRemoteRuntimeReceiptV2,
    tuple[SelectedFeedbackImageBinding, ...],
    VerifiedPortfolioRemoteProcessingRuntime,
]:
    validate_portfolio_s1_feedback_control(control, selection, authorization)
    parent = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-kimi-feedback",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    parent_bindings = (
        authorization.parent_remote_authorization_id,
        authorization.parent_remote_authorization_file_sha256,
        authorization.parent_remote_receipt_file_sha256,
        authorization.parent_remote_receipt_sha256,
        authorization.parent_remote_catalog_sha256,
    )
    runtime_bindings = (
        parent.authorization.authorization_id,
        parent.authorization_file_sha256,
        parent.receipt_file_sha256,
        parent.receipt.receipt_sha256,
        parent.catalog.catalog_sha256,
    )
    if parent_bindings != runtime_bindings:
        raise SelectedFeedbackRemoteRuntimeError(
            "selected Kimi Feedback authorization parent runtime drifted"
        )

    selected_sources = sources or build_verified_static_feedback_sources(
        corpus, selection, control
    )
    if (
        len(selected_sources) != 48
        or tuple(item.selection_entry_sha256 for item in selected_sources)
        != tuple(item.entry_sha256 for item in selection.entries)
        or any(
            item.selection_sha256 != selection.selection_sha256
            or item.control_sha256 != control.control_sha256
            for item in selected_sources
        )
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "preverified selected Kimi Feedback sources drifted"
        )
    for entry, source in zip(selection.entries, selected_sources, strict=True):
        require_verified_static_feedback_source(source, selection, control, entry)

    verified = selected_sources[0].corpus
    parent.catalog.require_verified_files()
    if (
        verified.core_inputs.expected_output_catalog_sha256
        != parent.catalog.catalog_sha256
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "selected Kimi Feedback corpus differs from the Core remote catalog"
        )

    receipt_bindings: list[SelectedFeedbackRemoteBindingV1] = []
    image_bindings: list[SelectedFeedbackImageBinding] = []
    for entry, source in zip(selection.entries, selected_sources, strict=True):
        row = source.row
        resolution = parent.catalog.verify_reference(
            entry.asset_id,
            row.query.image_path,
            entry.leakage_group_id,
        )
        if (
            resolution.asset.asset_id != entry.asset_id
            or resolution.asset.sha256 != entry.image_sha256
            or resolution.asset.cloud_upload_allowed is not True
            or source.packet.query_id != entry.query_id
            or source.packet.image.sha256 != entry.image_sha256
        ):
            raise SelectedFeedbackRemoteRuntimeError(
                "selected Kimi Feedback membership drifted"
            )
        receipt_bindings.append(
            SelectedFeedbackRemoteBindingV1(
                selection_ordinal=entry.selection_ordinal,
                selection_entry_sha256=entry.entry_sha256,
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )
        image_bindings.append(
            SelectedFeedbackImageBinding(
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )

    binding_payload = [item.model_dump(mode="json") for item in receipt_bindings]
    draft = PortfolioS1KimiFeedbackRemoteRuntimeReceiptV2.model_construct(
        selection_sha256=selection.selection_sha256,
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        control_sha256=control.control_sha256,
        corpus_sha256=verified.corpus_sha256,
        catalog_sha256=parent.catalog.catalog_sha256,
        parent_remote_authorization_id=parent.authorization.authorization_id,
        parent_remote_authorization_file_sha256=parent.authorization_file_sha256,
        parent_remote_receipt_file_sha256=parent.receipt_file_sha256,
        parent_remote_receipt_sha256=parent.receipt.receipt_sha256,
        selected_bindings=tuple(receipt_bindings),
        selected_binding_set_sha256=sha256_bytes(canonical_json_bytes(binding_payload)),
        receipt_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"receipt_sha256"})
    receipt = PortfolioS1KimiFeedbackRemoteRuntimeReceiptV2.model_validate_json(
        canonical_json_bytes(
            {
                **unsigned,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            }
        ),
        strict=True,
    )
    return receipt, tuple(image_bindings), parent


def _expected_qwen_receipt(
    corpus: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV1,
    authorization: PortfolioS1QwenFeedbackAuthorizationV3,
    control: PortfolioS1FeedbackControlV1,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    sources: tuple[VerifiedStaticFeedbackSource, ...] | None = None,
) -> tuple[
    PortfolioS1QwenFeedbackRemoteRuntimeReceiptV3,
    tuple[SelectedFeedbackImageBinding, ...],
    VerifiedPortfolioRemoteProcessingRuntime,
]:
    validate_portfolio_s1_feedback_control(control, selection, authorization)
    parent = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-qwen-assistant",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    if (
        authorization.parent_remote_authorization_id,
        authorization.parent_remote_authorization_file_sha256,
        authorization.parent_remote_receipt_file_sha256,
        authorization.parent_remote_receipt_sha256,
        authorization.parent_remote_catalog_sha256,
    ) != (
        parent.authorization.authorization_id,
        parent.authorization_file_sha256,
        parent.receipt_file_sha256,
        parent.receipt.receipt_sha256,
        parent.catalog.catalog_sha256,
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "selected Qwen Feedback authorization parent membership drifted"
        )

    selected_sources = sources or build_verified_static_feedback_sources(
        corpus, selection, control
    )
    if (
        len(selected_sources) != 48
        or tuple(item.selection_entry_sha256 for item in selected_sources)
        != tuple(item.entry_sha256 for item in selection.entries)
        or any(
            item.selection_sha256 != selection.selection_sha256
            or item.control_sha256 != control.control_sha256
            for item in selected_sources
        )
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "preverified selected Qwen Feedback sources drifted"
        )
    for entry, source in zip(selection.entries, selected_sources, strict=True):
        require_verified_static_feedback_source(source, selection, control, entry)

    verified = selected_sources[0].corpus
    parent.catalog.require_verified_files()
    if (
        verified.core_inputs.expected_output_catalog_sha256
        != parent.catalog.catalog_sha256
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "selected Qwen Feedback corpus differs from Core catalog membership"
        )

    receipt_bindings: list[SelectedFeedbackRemoteBindingV1] = []
    image_bindings: list[SelectedFeedbackImageBinding] = []
    for entry, source in zip(selection.entries, selected_sources, strict=True):
        resolution = parent.catalog.verify_reference(
            entry.asset_id,
            source.row.query.image_path,
            entry.leakage_group_id,
        )
        if (
            resolution.asset.asset_id != entry.asset_id
            or resolution.asset.sha256 != entry.image_sha256
            or resolution.asset.cloud_upload_allowed is not True
            or source.packet.query_id != entry.query_id
            or source.packet.image.sha256 != entry.image_sha256
        ):
            raise SelectedFeedbackRemoteRuntimeError(
                "selected Qwen Feedback membership drifted"
            )
        receipt_bindings.append(
            SelectedFeedbackRemoteBindingV1(
                selection_ordinal=entry.selection_ordinal,
                selection_entry_sha256=entry.entry_sha256,
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )
        image_bindings.append(
            SelectedFeedbackImageBinding(
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )

    binding_payload = [item.model_dump(mode="json") for item in receipt_bindings]
    draft = PortfolioS1QwenFeedbackRemoteRuntimeReceiptV3.model_construct(
        selection_sha256=selection.selection_sha256,
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        control_sha256=control.control_sha256,
        corpus_sha256=verified.corpus_sha256,
        catalog_sha256=parent.catalog.catalog_sha256,
        parent_remote_authorization_id=parent.authorization.authorization_id,
        parent_remote_authorization_file_sha256=parent.authorization_file_sha256,
        parent_remote_receipt_file_sha256=parent.receipt_file_sha256,
        parent_remote_receipt_sha256=parent.receipt.receipt_sha256,
        selected_bindings=tuple(receipt_bindings),
        selected_binding_set_sha256=sha256_bytes(
            canonical_json_bytes(binding_payload)
        ),
        receipt_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"receipt_sha256"})
    receipt = PortfolioS1QwenFeedbackRemoteRuntimeReceiptV3.model_validate_json(
        canonical_json_bytes(
            {
                **unsigned,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            }
        ),
        strict=True,
    )
    return receipt, tuple(image_bindings), parent


def prepare_selected_feedback_remote_runtime(
    corpus: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV1,
    authorization: PortfolioS1FeedbackAuthorizationV1,
    control: PortfolioS1FeedbackControlV1,
    *,
    receipt_path: str | Path,
    verified_sources: tuple[VerifiedStaticFeedbackSource, ...] | None = None,
) -> VerifiedSelectedFeedbackRemoteRuntime:
    """Create or exactly resume one selected48 AIFast runtime receipt."""

    expected, bindings, verified = _expected_receipt(
        corpus, selection, authorization, control, verified_sources
    )
    target = Path(receipt_path)
    if target.exists():
        content = read_stable_regular_file(
            target,
            label="selected Feedback remote runtime receipt",
            max_bytes=2 * 1024 * 1024,
        )
        try:
            receipt = PortfolioS1FeedbackRemoteRuntimeReceiptV1.model_validate_json(
                content, strict=True
            )
        except ValueError as error:
            raise SelectedFeedbackRemoteRuntimeError(
                "selected Feedback remote receipt is invalid"
            ) from error
        if receipt.canonical_bytes() != content or receipt != expected:
            raise SelectedFeedbackRemoteRuntimeError(
                "selected Feedback remote receipt resume conflict"
            )
    else:
        atomic_create_file(target, expected.canonical_bytes())
        receipt = expected
    content = read_stable_regular_file(
        target,
        label="selected Feedback remote runtime receipt",
        max_bytes=2 * 1024 * 1024,
    )
    catalog = verified.core_inputs.runtime_asset_catalog()
    return _make_verified_selected_feedback_remote_runtime(
        authorization=authorization,
        receipt=receipt,
        catalog=catalog,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        receipt_file_sha256=sha256_bytes(content),
        selected_bindings=bindings,
    )


def prepare_selected_kimi_feedback_remote_runtime(
    corpus: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV1,
    authorization: PortfolioS1FeedbackAuthorizationV2,
    control: PortfolioS1FeedbackControlV1,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    receipt_path: str | Path,
    verified_sources: tuple[VerifiedStaticFeedbackSource, ...] | None = None,
) -> VerifiedSelectedFeedbackRemoteRuntime:
    """Create/resume selected48 membership under the Core Kimi authority."""

    if type(authorization) is not PortfolioS1FeedbackAuthorizationV2:
        raise SelectedFeedbackRemoteRuntimeError(
            "selected Kimi Feedback runtime requires authorization v2"
        )
    expected, bindings, parent = _expected_kimi_receipt(
        corpus,
        selection,
        authorization,
        control,
        parent_remote_runtime,
        verified_sources,
    )
    target = Path(receipt_path)
    if target.exists():
        content = read_stable_regular_file(
            target,
            label="selected Kimi Feedback remote runtime receipt",
            max_bytes=2 * 1024 * 1024,
        )
        try:
            receipt = PortfolioS1KimiFeedbackRemoteRuntimeReceiptV2.model_validate_json(
                content, strict=True
            )
        except ValueError as error:
            raise SelectedFeedbackRemoteRuntimeError(
                "selected Kimi Feedback remote receipt is invalid"
            ) from error
        if receipt.canonical_bytes() != content or receipt != expected:
            raise SelectedFeedbackRemoteRuntimeError(
                "selected Kimi Feedback remote receipt resume conflict"
            )
    else:
        atomic_create_file(target, expected.canonical_bytes())
        receipt = expected
    content = read_stable_regular_file(
        target,
        label="selected Kimi Feedback remote runtime receipt",
        max_bytes=2 * 1024 * 1024,
    )
    return _make_verified_selected_feedback_remote_runtime(
        authorization=authorization,
        receipt=receipt,
        catalog=parent.catalog,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        receipt_file_sha256=sha256_bytes(content),
        selected_bindings=bindings,
        processor="dashscope-kimi-feedback",
    )


def prepare_selected_qwen_feedback_remote_runtime(
    corpus: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV1,
    authorization: PortfolioS1QwenFeedbackAuthorizationV3,
    control: PortfolioS1FeedbackControlV1,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    receipt_path: str | Path,
    verified_sources: tuple[VerifiedStaticFeedbackSource, ...] | None = None,
) -> VerifiedSelectedFeedbackRemoteRuntime:
    """Create/resume selected48 Qwen role authority and image membership."""

    if type(authorization) is not PortfolioS1QwenFeedbackAuthorizationV3:
        raise SelectedFeedbackRemoteRuntimeError(
            "selected Qwen Feedback runtime requires authorization v3"
        )
    expected, bindings, parent = _expected_qwen_receipt(
        corpus,
        selection,
        authorization,
        control,
        parent_remote_runtime,
        verified_sources,
    )
    target = Path(receipt_path)
    if target.exists():
        content = read_stable_regular_file(
            target,
            label="selected Qwen Feedback remote runtime receipt",
            max_bytes=2 * 1024 * 1024,
        )
        try:
            receipt = PortfolioS1QwenFeedbackRemoteRuntimeReceiptV3.model_validate_json(
                content, strict=True
            )
        except ValueError as error:
            raise SelectedFeedbackRemoteRuntimeError(
                "selected Qwen Feedback remote receipt is invalid"
            ) from error
        if receipt.canonical_bytes() != content or receipt != expected:
            raise SelectedFeedbackRemoteRuntimeError(
                "selected Qwen Feedback remote receipt resume conflict"
            )
    else:
        atomic_create_file(target, expected.canonical_bytes())
        receipt = expected
    content = read_stable_regular_file(
        target,
        label="selected Qwen Feedback remote runtime receipt",
        max_bytes=2 * 1024 * 1024,
    )
    return _make_verified_selected_feedback_remote_runtime(
        authorization=authorization,
        receipt=receipt,
        catalog=parent.catalog,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        receipt_file_sha256=sha256_bytes(content),
        selected_bindings=bindings,
        processor="dashscope-qwen37-feedback",
    )


__all__ = [
    "PortfolioS1KimiFeedbackRemoteRuntimeReceiptV2",
    "PortfolioS1QwenFeedbackRemoteRuntimeReceiptV3",
    "PortfolioS1FeedbackRemoteRuntimeReceiptV1",
    "SELECTED_KIMI_FEEDBACK_REMOTE_POLICY_VERSION",
    "SELECTED_QWEN37_FEEDBACK_REMOTE_POLICY_VERSION",
    "SELECTED_FEEDBACK_REMOTE_POLICY_VERSION",
    "SelectedFeedbackRemoteBindingV1",
    "SelectedFeedbackRemoteRuntimeError",
    "prepare_selected_feedback_remote_runtime",
    "prepare_selected_kimi_feedback_remote_runtime",
    "prepare_selected_qwen_feedback_remote_runtime",
]
