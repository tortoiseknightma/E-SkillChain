"""Load evaluator image bytes only from a verified remote-processing catalog."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import re
from pathlib import Path
from typing import Iterable

from skillchain.data.portfolio_remote_processing import (
    PortfolioProcessor,
    VerifiedPortfolioRemoteProcessingRuntime,
    require_verified_portfolio_remote_processing_runtime,
)
from skillchain.evaluation.packets import EvaluationImage


class EvaluatorImageLoadError(ValueError):
    """The requested visual evidence is not a stable authorized catalog file."""


_SELECTED_FEEDBACK_RUNTIME_TOKEN = object()
_SELECTED_FEEDBACK_PROCESSORS = frozenset(
    {
        "aifast-gemini-feedback",
        "dashscope-kimi-feedback",
        "dashscope-qwen37-feedback",
        "dashscope-qwen38-feedback",
    }
)


@dataclass(frozen=True)
class SelectedFeedbackImageBinding:
    query_id: str
    asset_id: str
    image_sha256: str


@dataclass(frozen=True)
class VerifiedSelectedFeedbackRemoteRuntime:
    """Independent exact-selection authority; never aliases the Core scope."""

    authorization: object
    receipt: object
    processor: PortfolioProcessor
    catalog: object = field(repr=False, compare=False)
    authorization_file_sha256: str
    receipt_file_sha256: str
    selected_bindings: tuple[SelectedFeedbackImageBinding, ...]
    _verification_token: object | None = field(default=None, repr=False, compare=False)


def _make_verified_selected_feedback_remote_runtime(
    *,
    authorization: object,
    receipt: object,
    catalog: object,
    authorization_file_sha256: str,
    receipt_file_sha256: str,
    selected_bindings: tuple[SelectedFeedbackImageBinding, ...],
    processor: PortfolioProcessor = "aifast-gemini-feedback",
    expected_binding_count: int = 48,
) -> VerifiedSelectedFeedbackRemoteRuntime:
    if (
        type(expected_binding_count) is not int
        or expected_binding_count <= 0
        or len(selected_bindings) != expected_binding_count
    ):
        raise EvaluatorImageLoadError(
            "selected Feedback runtime binding count differs from its exact authority"
        )
    if (
        len({item.query_id for item in selected_bindings}) != expected_binding_count
        or len({item.asset_id for item in selected_bindings}) != expected_binding_count
    ):
        raise EvaluatorImageLoadError(
            "selected Feedback runtime query/asset identities must be unique"
        )
    if processor not in _SELECTED_FEEDBACK_PROCESSORS:
        raise EvaluatorImageLoadError(
            "selected Feedback runtime processor is not an evaluator Feedback role"
        )
    return VerifiedSelectedFeedbackRemoteRuntime(
        authorization=authorization,
        receipt=receipt,
        processor=processor,
        catalog=catalog,
        authorization_file_sha256=authorization_file_sha256,
        receipt_file_sha256=receipt_file_sha256,
        selected_bindings=selected_bindings,
        _verification_token=_SELECTED_FEEDBACK_RUNTIME_TOKEN,
    )


_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_DATA_URL_PATTERN = re.compile(
    r"data:[^\s,;]{1,128}(?:;[^\s,;]{1,128})*;base64,",
    re.IGNORECASE,
)


def contains_base64_data_url(*texts: str) -> bool:
    """Return whether any string contains an inline Base64 data URL."""

    return any(_DATA_URL_PATTERN.search(text) is not None for text in texts)


def contains_encoded_image_echo(
    *,
    response_text: str,
    tool_argument_texts: Iterable[str],
    image_bytes: bytes,
    mime_type: str,
) -> bool:
    """Detect image encodings that must never enter persisted receipts."""

    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    exact_data_url = f"data:{mime_type};base64,{encoded_image}"
    for text in (response_text, *tuple(tool_argument_texts)):
        if (
            exact_data_url in text
            or encoded_image in text
            or contains_base64_data_url(text)
        ):
            return True
    return False


def load_verified_evaluator_image(
    remote_runtime: (
        VerifiedPortfolioRemoteProcessingRuntime | VerifiedSelectedFeedbackRemoteRuntime
    ),
    *,
    processor: PortfolioProcessor,
    image: EvaluationImage,
    query_id: str | None = None,
) -> tuple[
    VerifiedPortfolioRemoteProcessingRuntime | VerifiedSelectedFeedbackRemoteRuntime,
    bytes,
]:
    """Reverify and read one content-addressed image for a transient API wire."""

    selected_binding: SelectedFeedbackImageBinding | None = None
    if type(remote_runtime) is VerifiedSelectedFeedbackRemoteRuntime:
        if (
            remote_runtime._verification_token is not _SELECTED_FEEDBACK_RUNTIME_TOKEN
            or processor not in _SELECTED_FEEDBACK_PROCESSORS
            or remote_runtime.processor != processor
            or query_id is None
        ):
            raise EvaluatorImageLoadError(
                "selected Feedback image requires its verified query scope"
            )
        selected_by_query = {
            item.query_id: item for item in remote_runtime.selected_bindings
        }
        selected_binding = selected_by_query.get(query_id)
        if selected_binding is None or selected_binding.image_sha256 != image.sha256:
            raise EvaluatorImageLoadError(
                "evaluator image is outside the exact "
                f"selected{len(remote_runtime.selected_bindings)} Feedback authority"
            )
        verified_runtime = remote_runtime
        matches = tuple(
            asset
            for asset in verified_runtime.catalog.assets
            if asset.asset_id == selected_binding.asset_id
            and asset.sha256 == selected_binding.image_sha256
            and asset.cloud_upload_allowed is True
        )
    else:
        verified_runtime = require_verified_portfolio_remote_processing_runtime(
            remote_runtime,
            processor=processor,
            catalog_sha256=remote_runtime.catalog.catalog_sha256,
        )
        matches = tuple(
            asset
            for asset in verified_runtime.catalog.assets
            if asset.sha256 == image.sha256 and asset.cloud_upload_allowed is True
        )
    if not matches:
        raise EvaluatorImageLoadError(
            "evaluator image is outside the verified remote-processing catalog"
        )
    asset = sorted(
        matches,
        key=lambda item: (str(item.local_path), str(item.asset_id)),
    )[0]
    expected_mime = _MIME_BY_SUFFIX.get(Path(asset.local_path).suffix.casefold())
    if expected_mime != image.mime_type:
        raise EvaluatorImageLoadError(
            "evaluator image MIME type does not match the verified catalog path"
        )

    catalog = verified_runtime.catalog
    catalog.verify_asset_ids([asset.asset_id])
    asset_root = catalog.asset_root.resolve()
    image_path = catalog.asset_root / asset.local_path
    resolved_path = image_path.resolve()
    if (
        not resolved_path.is_relative_to(asset_root)
        or image_path.is_symlink()
        or (hasattr(image_path, "is_junction") and image_path.is_junction())
    ):
        raise EvaluatorImageLoadError(
            "evaluator image must be a real file inside the verified catalog root"
        )
    metadata_before = image_path.lstat()
    image_bytes = image_path.read_bytes()
    metadata_after = image_path.lstat()
    identity_before = (
        metadata_before.st_dev,
        metadata_before.st_ino,
        metadata_before.st_size,
        metadata_before.st_mtime_ns,
    )
    identity_after = (
        metadata_after.st_dev,
        metadata_after.st_ino,
        metadata_after.st_size,
        metadata_after.st_mtime_ns,
    )
    if identity_before != identity_after or not image_bytes:
        raise EvaluatorImageLoadError("evaluator image changed while being loaded")
    if hashlib.sha256(image_bytes).hexdigest() != image.sha256:
        raise EvaluatorImageLoadError(
            "evaluator image bytes do not match the prompt commitment"
        )
    catalog.verify_asset_ids([asset.asset_id])
    return verified_runtime, image_bytes


__all__ = [
    "contains_base64_data_url",
    "contains_encoded_image_echo",
    "EvaluatorImageLoadError",
    "SelectedFeedbackImageBinding",
    "VerifiedSelectedFeedbackRemoteRuntime",
    "load_verified_evaluator_image",
]
