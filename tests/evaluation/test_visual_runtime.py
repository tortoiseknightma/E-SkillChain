from __future__ import annotations

import base64
import hashlib
from types import SimpleNamespace

from skillchain.evaluation.packets import EvaluationImage
from skillchain.evaluation.visual_runtime import (
    SelectedFeedbackImageBinding,
    _make_verified_selected_feedback_remote_runtime,
    contains_encoded_image_echo,
    load_verified_evaluator_image,
)


def test_visual_runtime_loads_only_committed_catalog_bytes(
    monkeypatch,
    tmp_path,
) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    image_path = asset_root / "input.png"
    image_bytes = b"verified-image-bytes"
    image_path.write_bytes(image_bytes)
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    verified_ids: list[tuple[str, ...]] = []
    asset = SimpleNamespace(
        asset_id="asset-001",
        local_path="input.png",
        sha256=image_sha256,
        cloud_upload_allowed=True,
    )
    catalog = SimpleNamespace(
        catalog_sha256="a" * 64,
        asset_root=asset_root,
        assets=(asset,),
        verify_asset_ids=lambda values: verified_ids.append(tuple(values)),
    )
    runtime = SimpleNamespace(catalog=catalog)
    preflights: list[tuple[object, str, str]] = []

    def fake_require(value, *, processor, catalog_sha256):
        preflights.append((value, processor, catalog_sha256))
        return value

    monkeypatch.setattr(
        "skillchain.evaluation.visual_runtime."
        "require_verified_portfolio_remote_processing_runtime",
        fake_require,
    )
    verified, loaded = load_verified_evaluator_image(
        runtime,
        processor="dashscope-kimi-judge",
        image=EvaluationImage(mime_type="image/png", sha256=image_sha256),
    )

    assert verified is runtime
    assert loaded == image_bytes
    assert preflights == [(runtime, "dashscope-kimi-judge", "a" * 64)]
    assert verified_ids == [("asset-001",), ("asset-001",)]


def test_encoded_image_echo_detection_checks_tool_arguments() -> None:
    image_bytes = b"tool-argument-image"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    assert contains_encoded_image_echo(
        response_text="{}",
        tool_argument_texts=('{"image":"data:image/png;base64,' + encoded + '"}',),
        image_bytes=image_bytes,
        mime_type="image/png",
    )


def test_selected_feedback_runtime_supports_qwen38_forward_exact240_authority() -> None:
    bindings = tuple(
        SelectedFeedbackImageBinding(
            query_id=f"query-{index:03d}",
            asset_id=f"asset-{index:03d}",
            image_sha256=hashlib.sha256(f"image-{index}".encode()).hexdigest(),
        )
        for index in range(240)
    )

    runtime = _make_verified_selected_feedback_remote_runtime(
        authorization=object(),
        receipt=object(),
        catalog=object(),
        authorization_file_sha256="a" * 64,
        receipt_file_sha256="b" * 64,
        selected_bindings=bindings,
        processor="dashscope-qwen38-feedback",
        expected_binding_count=240,
    )

    assert len(runtime.selected_bindings) == 240
    assert runtime.processor == "dashscope-qwen38-feedback"
