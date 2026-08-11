from __future__ import annotations

import json
from pathlib import Path

from skillchain.evaluation.evaluator_isolation import (
    make_active_portfolio_evaluator_isolation_lock,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


def test_role_selection_v7_is_self_hashed_and_preserves_v5_v6() -> None:
    historical = ROOT / "specs/authoring/model-role-selection-v5.json"
    historical_v6 = ROOT / "specs/authoring/model-role-selection-v6.json"
    active = ROOT / "specs/authoring/model-role-selection-v7.json"

    assert sha256_bytes(historical.read_bytes()) == (
        "7fc6b0ba20ab542478ce5e3d5d53974ff619333a657c3b806eca88584ac62e8c"
    )
    assert sha256_bytes(historical_v6.read_bytes()) == (
        "86d5b7763dfee3087d8a9df05659387ca1033093fc021d806643864080e770f9"
    )
    historical_v6_payload = json.loads(historical_v6.read_text(encoding="utf-8"))
    assert historical_v6_payload["selection_sha256"] == (
        "940f886e40fd0438197384b4de9042efe4f620d8b7a8557060e36f407b0294a0"
    )
    payload = json.loads(active.read_text(encoding="utf-8"))
    unsigned = {
        key: value for key, value in payload.items() if key != "selection_sha256"
    }
    assert active.read_bytes() == canonical_json_bytes(payload)
    assert payload["selection_sha256"] == sha256_bytes(canonical_json_bytes(unsigned))
    assert payload["schema_version"] == 7
    assert payload["feedback_evaluator"]["provider"] == "kimi"
    assert payload["feedback_evaluator"]["model"] == "kimi-k2.6"
    assert payload["feedback_evaluator"]["cache_namespace"] == (
        "feedback-evaluator-v8"
    )
    assert payload["feedback_evaluator"]["prompt_policy_version"] == (
        "visual-feedback-response-schema-v1-prompt-v5"
    )
    assert payload["feedback_evaluator"]["transport_policy_version"] == (
        "visual-feedback-kimi-dashscope-plain-json-v3"
    )
    assert payload["offline_judge"]["provider"] == "gemini"
    assert payload["offline_judge"]["model"] == "gemini-3.6-flash"
    assert payload["authorization_status"] == {
        "aifast_gemini_judge": (
            "new_core_owner_authorization_and_create_only_overlay_receipt_"
            "required_before_live_use"
        ),
        "dashscope_kimi_feedback": (
            "covered_by_existing_core_v9_owner_authorization_v1"
        ),
        "historical_authorizations_reusable_for_new_roles": False,
    }


def test_active_evaluator_isolation_follows_swapped_runtime_roles() -> None:
    lock = make_active_portfolio_evaluator_isolation_lock()

    assert (lock.feedback.provider, lock.feedback.model) == (
        "qwen",
        "qwen3.8-max",
    )
    assert (lock.final.provider, lock.final.model) == (
        "gemini",
        "gemini-3.6-flash",
    )
    assert lock.feedback.model_family == lock.feedback.model
    assert lock.final.model_family == lock.final.model
    assert lock.feedback.endpoint != lock.final.endpoint
    assert lock.shared_model_runtime is False
