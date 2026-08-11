"""Build create-only evidence for the 25-query Core Gate 0 provider canary.

The command has three phases.  It never calls a provider directly; the middle
phase invokes the frozen provider-enabled matrix command against an already
complete canary, which must therefore reuse checkpoints without a new call:

``snapshot``
    Validate the completed first Core ``dev_mini`` batch and capture the
    provider ledger plus every checkpoint/terminal artifact before a rerun.

``rerun``
    Require the snapshot and execute the exact frozen matrix argv.  Capture a
    create-only invocation receipt without persisting command output text.

``verify``
    Require that invocation receipt, recompute the validated state, and prove
    that provider-call identities/counts and artifact bytes did not change.

Both outputs are canonical, content-addressed, and create-only.  The validator
uses the existing batch auditor and matrix analyzer, so Full rollback aliases,
checkpoint semantics, and hard-error classification have one implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.analyze_portfolio_matrix import (  # noqa: E402
    CAPABILITIES,
    analyze_portfolio_matrix,
)
from scripts.audit_portfolio_batch import audit_portfolio_batch  # noqa: E402
from scripts.run_portfolio_shard import (  # noqa: E402
    _ROUTE_CONTRACT_RETRYABLE_FAILURE_SUBTYPES,
    _ROUTE_CONTRACT_RETRYABLE_PAYLOAD_STATUSES,
    _is_retryable_route_failure_shape,
    _load_control,
    _route_contract_attempt_failure_subtype,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_BUDGET_POLICY_VERSION,
    PORTFOLIO_FAILURE_POLICY_VERSION,
    PORTFOLIO_FAILURE_POLICY_VERSION_V3,
    PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY,
    PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS,
    PortfolioAttemptReceipt,
    load_portfolio_budget_ledger,
)
from skillchain.evaluation.final_runtime import (  # noqa: E402
    FINAL_JUDGE_CACHE_NAMESPACE,
    FINAL_JUDGE_RESULT_SCHEMA_VERSION,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    MAIN_CONFIG_ORDER,
    load_portfolio_launch_package,
)
from skillchain.evaluation.portfolio_parallel import (  # noqa: E402
    PortfolioParallelProfile,
    validate_parallel_profile_sources,
)
from skillchain.synthesis.store import atomic_create_file  # noqa: E402
from skillchain.runners.assistant import (  # noqa: E402
    PORTFOLIO_ROUTER_CONTRACT_SHA256,
    PORTFOLIO_ROUTER_CONTRACT_VERSION,
    SHARED_STAGE2_ROUTE_POLICY_VERSION,
    SharedStage2RouteArtifact,
    assistant_route_output_json_schema,
    assistant_router_contract_payload,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


LEGACY_POLICY_VERSION = "portfolio-core-gate0-canary-evidence-v1"
PREVIOUS_POLICY_VERSION = "portfolio-core-gate0-canary-evidence-v2"
POLICY_VERSION = "portfolio-core-gate0-canary-evidence-v3"
_SUPPORTED_POLICY_VERSIONS = frozenset(
    {LEGACY_POLICY_VERSION, PREVIOUS_POLICY_VERSION, POLICY_VERSION}
)
_FREEZE_POLICY_VERSION_V1 = "portfolio-core-gate0-freeze-v1"
_FREEZE_POLICY_VERSION_V2 = "portfolio-core-gate0-freeze-v2"
_FREEZE_POLICY_VERSION_V3 = "portfolio-core-gate0-freeze-v3"
SNAPSHOT_KIND = "portfolio-core-gate0-canary-rerun-snapshot"
RERUN_KIND = "portfolio-core-gate0-canary-rerun-invocation"
RECEIPT_KIND = "portfolio-core-gate0-canary-final-receipt"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OVERALL_LIMIT = Decimal("0.005")
_CAPABILITY_LIMIT = Decimal("0.01")
_MAX_EVIDENCE_FILE_BYTES = 64 * 1024 * 1024
_ROUTE_REPAIR_ELIGIBLE_FAILURES = (
    "length",
    "empty",
    "invalid_json",
    "non_object",
    "unexpected_keys",
    "schema_invalid",
)
_ROUTE_REPAIR_ELIGIBLE_SHAPES = (
    ("route_contract_length", "length", "not_examined"),
    ("route_contract_empty", "response_empty_text", "empty"),
    ("route_contract_invalid_json", "invalid_route_json", "invalid_json"),
    ("route_contract_non_object", "invalid_route_json", "non_object"),
    ("route_contract_unexpected_keys", "invalid_route_json", "unexpected_keys"),
    ("route_contract_schema_invalid", "invalid_route_json", "schema_invalid"),
)
_ROUTE_REPAIR_INELIGIBLE_SHAPES = (
    ("out_of_enum", "out_of_enum"),
    ("response_tool_calls", "not_examined"),
    ("response_finish_reason", "not_examined"),
    ("response_contract", "not_examined"),
    ("route_budget", "not_examined"),
)
_V2_RUNTIME_ROUTER_FIELDS = (
    "shared_stage2_route_policy_version",
    "shared_stage2_route_schema_sha256",
    "portfolio_router_contract_version",
    "portfolio_router_contract_sha256",
    "portfolio_failure_policy_version",
    "portfolio_router_request_max_output_tokens",
    "portfolio_router_pricing_reservation_max_output_tokens",
)
_V3_BUDGET_FREEZE = {
    "policy_version": "portfolio-call-hard-cap-v2",
    "accounting_policy": (
        "settled_actual_plus_forfeited_full_reserve_plus_unresolved_reserve"
    ),
    "exception_policy": (
        "append_full_reserve_forfeit_without_captured_response"
    ),
    "terminal_outcome_policy": "exactly_one_of_settlement_or_forfeit",
    "forfeit_disposition": "charged_full_reserve_unknown_actual_usage",
}
_V3_FINAL_JUDGE_FREEZE = {
    "result_schema_version": FINAL_JUDGE_RESULT_SCHEMA_VERSION,
    "cache_namespace": FINAL_JUDGE_CACHE_NAMESPACE,
    "provider_pre_response_terminal_event": "full_reserve_forfeit",
    "forfeit_binding_fields": [
        "forfeited_reservation_sha256",
        "budget_forfeit_sha256",
    ],
}
_V3_PAUSE_RESUME_FORFEIT_FREEZE = {
    "provider_exception_terminal_event": (
        "append_create_only_full_reserve_forfeit_before_attempt_receipt"
    ),
    "dead_owner_orphan_recovery": (
        "append_one_full_reserve_forfeit_after_confirmed_owner_exit"
    ),
    "unresolved_provider_reservation_policy": (
        "forfeit_full_reserve_after_failure_or_confirmed_dead_owner_"
        "never_release_unknown_cost"
    ),
}


class Gate0CanaryEvidenceError(ValueError):
    """The canary evidence is incomplete, inconsistent, or mutable."""


def _require_sha256(value: str, label: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise Gate0CanaryEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _hash_payload(value: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(value)))


def _content_address(
    payload: Mapping[str, object],
    *,
    field: str,
) -> dict[str, object]:
    unsigned = dict(payload)
    return {**unsigned, field: _hash_payload(unsigned)}


def _canonical_object(
    path: Path,
    *,
    label: str,
    expected_file_sha256: str | None = None,
) -> tuple[dict[str, object], bytes]:
    content = read_stable_regular_file(
        path,
        label=label,
        max_bytes=_MAX_EVIDENCE_FILE_BYTES,
    )
    if (
        expected_file_sha256 is not None
        and sha256_bytes(content)
        != _require_sha256(expected_file_sha256, f"{label} file SHA-256")
    ):
        raise Gate0CanaryEvidenceError(f"{label} file digest mismatch")
    try:
        raw = parse_canonical_json(content, label=label)
    except ValueError as error:
        raise Gate0CanaryEvidenceError(f"{label} is not canonical JSON") from error
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise Gate0CanaryEvidenceError(
            f"{label} must contain one canonical JSON object"
        )
    return dict(raw), content


def _router_freeze_contract(
    *,
    failure_policy_version: str,
) -> dict[str, object]:
    """Return the exact Router-v6 surface for one frozen attempt policy."""

    expected_payload_statuses = frozenset(
        {"invalid_json", "non_object", "unexpected_keys", "schema_invalid"}
    )
    expected_failure_subtypes = frozenset(
        {"invalid_route_json", "length", "response_empty_text"}
    )
    if (
        _ROUTE_CONTRACT_RETRYABLE_PAYLOAD_STATUSES != expected_payload_statuses
        or _ROUTE_CONTRACT_RETRYABLE_FAILURE_SUBTYPES
        != expected_failure_subtypes
    ):
        raise Gate0CanaryEvidenceError(
            "active Router repair eligibility differs from Gate0"
        )
    repair_shapes = [
        {
            "attempt_failure_subtype": attempt_failure_subtype,
            "failure_subtype": failure_subtype,
            "payload_status": payload_status,
        }
        for attempt_failure_subtype, failure_subtype, payload_status in (
            _ROUTE_REPAIR_ELIGIBLE_SHAPES
        )
    ]
    candidate_statuses = {
        *expected_payload_statuses,
        "empty",
        "not_examined",
    }
    observed_shapes = {
        (failure_subtype, payload_status)
        for failure_subtype in expected_failure_subtypes
        for payload_status in candidate_statuses
        if _is_retryable_route_failure_shape(
            failure_subtype,
            SimpleNamespace(payload_status=payload_status),
        )
    }
    expected_shapes = {
        (failure_subtype, payload_status)
        for _attempt, failure_subtype, payload_status in (
            _ROUTE_REPAIR_ELIGIBLE_SHAPES
        )
    }
    if observed_shapes != expected_shapes or any(
        _route_contract_attempt_failure_subtype(
            failure_subtype,
            SimpleNamespace(payload_status=payload_status),
        )
        != attempt_failure_subtype
        for attempt_failure_subtype, failure_subtype, payload_status in (
            _ROUTE_REPAIR_ELIGIBLE_SHAPES
        )
    ):
        raise Gate0CanaryEvidenceError(
            "active Router repair shape mapping differs from Gate0"
        )
    if any(
        _is_retryable_route_failure_shape(
            failure_subtype,
            SimpleNamespace(payload_status=payload_status),
        )
        for failure_subtype, payload_status in _ROUTE_REPAIR_INELIGIBLE_SHAPES
    ):
        raise Gate0CanaryEvidenceError(
            "active Router repairs a Gate0 terminal failure"
        )
    payload = assistant_router_contract_payload()
    if (
        payload.get("route_user_input_fields") != ["turns"]
        or payload.get("route_image_attachment") is not False
        or payload.get("parser") != "full_strict_json_no_prefix_recovery"
        or payload.get("ignored_top_level_response_keys")
        != ["asset_id", "description", "text", "turns"]
        or payload.get("unknown_top_level_response_key_policy") != "reject"
        or payload.get("route_request_max_output_tokens")
        != PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS
        or payload.get("repair_wire_policy")
        != "fixed_prompt_must_differ_from_initial"
    ):
        raise Gate0CanaryEvidenceError(
            "active Router contract differs from the Gate0 behavior"
        )
    return {
        "shared_stage2_route_policy_version": SHARED_STAGE2_ROUTE_POLICY_VERSION,
        "shared_stage2_route_schema_sha256": sha256_bytes(
            canonical_json_bytes(SharedStage2RouteArtifact.model_json_schema())
        ),
        "core_route_output_json_schema_sha256": sha256_bytes(
            canonical_json_bytes(
                assistant_route_output_json_schema(tuple(CAPABILITIES))
            )
        ),
        "portfolio_router_contract_version": PORTFOLIO_ROUTER_CONTRACT_VERSION,
        "portfolio_router_contract_sha256": PORTFOLIO_ROUTER_CONTRACT_SHA256,
        "portfolio_router_contract_payload": payload,
        "portfolio_failure_policy_version": failure_policy_version,
        "portfolio_router_request_max_output_tokens": (
            PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS
        ),
        "portfolio_router_pricing_reservation_max_output_tokens": (
            PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
        ),
        "eligible_repair_failures": list(_ROUTE_REPAIR_ELIGIBLE_FAILURES),
        "eligible_repair_failure_shapes": repair_shapes,
        "non_repairable_failure_shapes": [
            {
                "failure_subtype": failure_subtype,
                "payload_status": payload_status,
            }
            for failure_subtype, payload_status in (
                _ROUTE_REPAIR_INELIGIBLE_SHAPES
            )
        ],
        "response_prefix_recovery": False,
    }


def _active_router_freeze_contract() -> dict[str, object]:
    """Return Router-v6 plus the active attempt-v4 policy for Gate0 v3."""

    if PORTFOLIO_FAILURE_POLICY_VERSION != "portfolio-shard-attempt-v4":
        raise Gate0CanaryEvidenceError(
            "active Portfolio attempt policy differs from Gate0 v3"
        )
    return _router_freeze_contract(
        failure_policy_version=PORTFOLIO_FAILURE_POLICY_VERSION,
    )


def _legacy_v2_router_freeze_contract() -> dict[str, object]:
    """Reconstruct the immutable Router-v6/attempt-v3 Gate0 v2 contract."""

    if PORTFOLIO_FAILURE_POLICY_VERSION_V3 != "portfolio-shard-attempt-v3":
        raise Gate0CanaryEvidenceError(
            "legacy Portfolio attempt-v3 identity is unavailable"
        )
    return _router_freeze_contract(
        failure_policy_version=PORTFOLIO_FAILURE_POLICY_VERSION_V3,
    )


def _active_route_retry_freeze() -> dict[str, object]:
    return {
        "eligible_failure_subtypes": sorted(
            _ROUTE_CONTRACT_RETRYABLE_FAILURE_SUBTYPES
        ),
        "eligible_failures": list(_ROUTE_REPAIR_ELIGIBLE_FAILURES),
        "eligible_failure_shapes": [
            {
                "attempt_failure_subtype": attempt_failure_subtype,
                "failure_subtype": failure_subtype,
                "payload_status": payload_status,
            }
            for attempt_failure_subtype, failure_subtype, payload_status in (
                _ROUTE_REPAIR_ELIGIBLE_SHAPES
            )
        ],
        "eligible_payload_statuses": sorted(
            _ROUTE_CONTRACT_RETRYABLE_PAYLOAD_STATUSES
        ),
        "max_attempts": PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY,
        "repair_request_variant": "fixed_repair",
        "response_prefix_recovery": False,
        "retry_count": 1,
        "wire_policy": "fixed_prompt_must_differ_from_initial",
    }


def _freeze_binding(path: Path, expected_file_sha256: str) -> dict[str, object]:
    raw, content = _canonical_object(
        path,
        label="Core Gate 0 freeze",
        expected_file_sha256=expected_file_sha256,
    )
    supplied = raw.get("contract_sha256")
    unsigned = dict(raw)
    unsigned.pop("contract_sha256", None)
    if supplied != _hash_payload(unsigned):
        raise Gate0CanaryEvidenceError("Core Gate 0 freeze self hash mismatch")
    evaluation = raw.get("evaluation")
    test_unseal = raw.get("test_unseal")
    runtime = raw.get("runtime")
    concurrency = runtime.get("concurrency") if isinstance(runtime, Mapping) else None
    retries = runtime.get("retries") if isinstance(runtime, Mapping) else None
    operational = (
        evaluation.get("operational_gate")
        if isinstance(evaluation, Mapping)
        else None
    )
    if not isinstance(operational, Mapping) or dict(operational) != {
        "assistant_and_judge_hard_error_each_capability_strictly_less_than": (
            "0.010000"
        ),
        "assistant_and_judge_hard_error_overall_strictly_less_than": "0.005000",
    }:
        raise Gate0CanaryEvidenceError("Core Gate 0 hard-error thresholds drifted")
    scope = raw.get("scope")
    policy_version = raw.get("policy_version")
    supported_freezes = {
        _FREEZE_POLICY_VERSION_V1,
        _FREEZE_POLICY_VERSION_V2,
        _FREEZE_POLICY_VERSION_V3,
    }
    expected_concurrency = (
        (1, 2, 20)
        if policy_version == _FREEZE_POLICY_VERSION_V3
        else (2, 8, 40)
    )
    if (
        raw.get("kind") != "portfolio-core-gate0-freeze"
        or policy_version not in supported_freezes
        or raw.get("status") != "frozen_contract_only_not_execution_authority"
        or raw.get("model_calls_performed") != 0
        or not isinstance(scope, Mapping)
        or scope.get("dataset_profile") != "core"
        or tuple(scope.get("config_order", ())) != MAIN_CONFIG_ORDER
        or tuple(scope.get("capabilities", ())) != tuple(CAPABILITIES)
        or not isinstance(test_unseal, Mapping)
        or test_unseal.get("state") != "sealed"
        or not isinstance(concurrency, Mapping)
        or concurrency.get("qwen_inflight") != expected_concurrency[0]
        or concurrency.get("kimi_inflight") != expected_concurrency[1]
        or concurrency.get("qwen_requests_per_minute") != expected_concurrency[2]
    ):
        raise Gate0CanaryEvidenceError("Core Gate 0 freeze contract is incompatible")
    binding = {
        "file_sha256": sha256_bytes(content),
        "contract_sha256": supplied,
        "policy_version": raw["policy_version"],
        "overall_hard_error_limit_exclusive": "0.005000",
        "capability_hard_error_limit_exclusive": "0.010000",
        "assistant_concurrency": concurrency["qwen_inflight"],
        "final_judge_concurrency": concurrency["kimi_inflight"],
        "qwen_requests_per_minute_cap": concurrency[
            "qwen_requests_per_minute"
        ],
        "test_state": "sealed",
    }
    if policy_version in {
        _FREEZE_POLICY_VERSION_V2,
        _FREEZE_POLICY_VERSION_V3,
    }:
        router_contract = runtime.get("router_contract")
        route_retry = retries.get("route_format_retry") if isinstance(
            retries, Mapping
        ) else None
        models = raw.get("models_and_limits")
        assistant = models.get("assistant") if isinstance(models, Mapping) else None
        expected_router = (
            _active_router_freeze_contract()
            if policy_version == _FREEZE_POLICY_VERSION_V3
            else _legacy_v2_router_freeze_contract()
        )
        if (
            not isinstance(router_contract, Mapping)
            or dict(router_contract) != expected_router
            or not isinstance(route_retry, Mapping)
            or dict(route_retry) != _active_route_retry_freeze()
            or not isinstance(assistant, Mapping)
            or assistant.get("route_request_max_output_tokens")
            != PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS
            or assistant.get("route_pricing_reservation_max_output_tokens")
            != PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
            or "route_max_output_tokens" in assistant
        ):
            raise Gate0CanaryEvidenceError(
                f"Core Gate 0 {policy_version} Router contract is incompatible"
            )
        binding["router_contract"] = expected_router
        binding["route_format_retry"] = _active_route_retry_freeze()
    if policy_version == _FREEZE_POLICY_VERSION_V3:
        budget = raw.get("budget")
        final_judge = runtime.get("final_judge_contract")
        pause_resume = runtime.get("pause_resume")
        if (
            not isinstance(budget, Mapping)
            or {
                field: budget.get(field) for field in _V3_BUDGET_FREEZE
            }
            != _V3_BUDGET_FREEZE
            or not isinstance(final_judge, Mapping)
            or dict(final_judge) != _V3_FINAL_JUDGE_FREEZE
            or not isinstance(pause_resume, Mapping)
            or {
                field: pause_resume.get(field)
                for field in _V3_PAUSE_RESUME_FORFEIT_FREEZE
            }
            != _V3_PAUSE_RESUME_FORFEIT_FREEZE
            or concurrency.get("qwen_minimum_start_interval_seconds")
            != "3.000000"
        ):
            raise Gate0CanaryEvidenceError(
                "Core Gate 0 v3 forfeit/final-Judge contract is incompatible"
            )
        binding["budget_forfeit_contract"] = dict(_V3_BUDGET_FREEZE)
        binding["final_judge_contract"] = dict(_V3_FINAL_JUDGE_FREEZE)
        binding["provider_forfeit_recovery"] = dict(
            _V3_PAUSE_RESUME_FORFEIT_FREEZE
        )
    return binding


def _runtime_lock_binding(
    *,
    runtime_root: Path,
    control: Mapping[str, object],
    freeze: Mapping[str, object],
) -> dict[str, object]:
    """Parse and bind the runtime lock, including versioned Gate0 contracts."""

    expected_file_sha256 = _require_sha256(
        str(control.get("runtime_lock_file_sha256")),
        "Core runtime-lock file SHA-256",
    )
    raw, content = _canonical_object(
        runtime_root / "runtime-lock.json",
        label="Core canary runtime lock",
        expected_file_sha256=expected_file_sha256,
    )
    supplied = raw.get("runtime_lock_sha256")
    unsigned = dict(raw)
    unsigned.pop("runtime_lock_sha256", None)
    if (
        raw.get("kind") != "portfolio-assistant-runtime-lock"
        or supplied != _hash_payload(unsigned)
        or supplied != control.get("runtime_lock_sha256")
    ):
        raise Gate0CanaryEvidenceError(
            "Core runtime-lock identity or self hash mismatch"
        )
    binding: dict[str, object] = {
        "runtime_lock_file_sha256": sha256_bytes(content),
        "runtime_lock_sha256": supplied,
        "treatment_chain_sha256": control.get("treatment_chain_sha256"),
    }
    freeze_policy = freeze.get("policy_version")
    if freeze_policy in {
        _FREEZE_POLICY_VERSION_V2,
        _FREEZE_POLICY_VERSION_V3,
    }:
        expected_router = freeze.get("router_contract")
        if not isinstance(expected_router, Mapping):
            raise Gate0CanaryEvidenceError(
                "Core Gate 0 freeze lacks its Router contract"
            )
        observed_router = {
            field: raw.get(field) for field in _V2_RUNTIME_ROUTER_FIELDS
        }
        frozen_router = {
            field: expected_router.get(field) for field in _V2_RUNTIME_ROUTER_FIELDS
        }
        if observed_router != frozen_router:
            raise Gate0CanaryEvidenceError(
                "Core runtime-lock Router contract differs from Gate0 freeze"
            )
        binding["router_contract"] = {
            **observed_router,
            "exact_match_to_freeze": True,
        }
    if freeze_policy == _FREEZE_POLICY_VERSION_V3:
        observed_budget = raw.get("portfolio_budget_policy_version")
        observed_final_judge = {
            "result_schema_version": raw.get("final_judge_result_schema_version"),
            "cache_namespace": raw.get("final_judge_cache_namespace"),
        }
        expected_final_judge = {
            field: _V3_FINAL_JUDGE_FREEZE[field]
            for field in ("result_schema_version", "cache_namespace")
        }
        if (
            observed_budget != PORTFOLIO_BUDGET_POLICY_VERSION
            or observed_budget != _V3_BUDGET_FREEZE["policy_version"]
            or observed_final_judge != expected_final_judge
        ):
            raise Gate0CanaryEvidenceError(
                "Core runtime-lock budget or final-Judge contract differs from Gate0 v3"
            )
        binding["budget_forfeit_contract"] = {
            "policy_version": observed_budget,
            "exact_match_to_freeze": True,
        }
        binding["final_judge_contract"] = {
            **observed_final_judge,
            "exact_match_to_freeze": True,
        }
    return binding


def _require_v2_evidence_bindings(invariants: Mapping[str, object]) -> None:
    """Require a complete freeze-v2/runtime-lock Router proof for evidence-v2."""

    freeze = invariants.get("freeze")
    runtime = invariants.get("runtime")
    if (
        not isinstance(freeze, Mapping)
        or freeze.get("policy_version") != _FREEZE_POLICY_VERSION_V2
        or not isinstance(runtime, Mapping)
    ):
        raise Gate0CanaryEvidenceError(
            "evidence-v2 requires Gate0 freeze-v2 and its runtime-lock proof"
        )
    frozen_router = freeze.get("router_contract")
    observed_router = runtime.get("router_contract")
    expected_router = _legacy_v2_router_freeze_contract()
    expected_runtime_router = {
        **{
            field: expected_router[field]
            for field in _V2_RUNTIME_ROUTER_FIELDS
        },
        "exact_match_to_freeze": True,
    }
    if (
        not isinstance(frozen_router, Mapping)
        or dict(frozen_router) != expected_router
        or not isinstance(observed_router, Mapping)
        or dict(observed_router) != expected_runtime_router
    ):
        raise Gate0CanaryEvidenceError(
            "evidence-v2 Router freeze/runtime-lock proof is incomplete or drifted"
        )


def _require_v3_evidence_bindings(invariants: Mapping[str, object]) -> None:
    """Require the active attempt-v4/budget-v2 forfeit proof end to end."""

    freeze = invariants.get("freeze")
    runtime = invariants.get("runtime")
    ledger = invariants.get("provider_ledger")
    if (
        not isinstance(freeze, Mapping)
        or freeze.get("policy_version") != _FREEZE_POLICY_VERSION_V3
        or not isinstance(runtime, Mapping)
        or not isinstance(ledger, Mapping)
    ):
        raise Gate0CanaryEvidenceError(
            "evidence-v3 requires Gate0 freeze-v3, runtime-lock, and provider ledger"
        )

    expected_router = _active_router_freeze_contract()
    expected_runtime_router = {
        **{
            field: expected_router[field]
            for field in _V2_RUNTIME_ROUTER_FIELDS
        },
        "exact_match_to_freeze": True,
    }
    expected_runtime_budget = {
        "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
        "exact_match_to_freeze": True,
    }
    expected_runtime_judge = {
        "result_schema_version": FINAL_JUDGE_RESULT_SCHEMA_VERSION,
        "cache_namespace": FINAL_JUDGE_CACHE_NAMESPACE,
        "exact_match_to_freeze": True,
    }
    expected_ledger_contract = {
        "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
        "accounting_policy": _V3_BUDGET_FREEZE["accounting_policy"],
        "exception_policy": _V3_BUDGET_FREEZE["exception_policy"],
        "terminal_outcome_policy": "exactly_one_of_settlement_or_forfeit",
    }
    reservation_count = ledger.get("reservation_count")
    settlement_count = ledger.get("settlement_count")
    forfeit_count = ledger.get("forfeit_count")
    terminal_count = ledger.get("terminal_outcome_count")
    forfeit_binding = ledger.get("forfeit_attempt_receipt_binding")
    bindings = (
        forfeit_binding.get("bindings")
        if isinstance(forfeit_binding, Mapping)
        else None
    )
    binding_set_sha256 = (
        forfeit_binding.get("binding_set_sha256")
        if isinstance(forfeit_binding, Mapping)
        else None
    )
    bindings_valid = isinstance(bindings, list) and all(
        isinstance(item, Mapping)
        and all(
            _SHA256.fullmatch(str(item.get(field))) is not None
            for field in (
                "reservation_sha256",
                "forfeit_sha256",
                "attempt_receipt_sha256",
            )
        )
        and isinstance(item.get("config"), str)
        and isinstance(item.get("stage"), str)
        and item.get("reason")
        in {
            "provider_call_ended_without_captured_response",
            "orphan_recovered_after_owner_exit",
        }
        for item in (bindings or [])
    )
    binding_pairs = (
        {
            (item["reservation_sha256"], item["forfeit_sha256"])
            for item in bindings
        }
        if bindings_valid
        else set()
    )
    observed_forfeit_counts = {
        "forfeit_counts_by_config": dict(
            sorted(Counter(item["config"] for item in (bindings or [])).items())
        ),
        "forfeit_counts_by_stage": dict(
            sorted(Counter(item["stage"] for item in (bindings or [])).items())
        ),
        "forfeit_counts_by_reason": dict(
            sorted(Counter(item["reason"] for item in (bindings or [])).items())
        ),
    }
    try:
        forfeited_cost = Decimal(str(ledger.get("forfeited_reserved_cost_cny")))
    except (InvalidOperation, TypeError, ValueError):
        forfeited_cost = Decimal("-1")
    if (
        freeze.get("router_contract") != expected_router
        or freeze.get("budget_forfeit_contract") != _V3_BUDGET_FREEZE
        or freeze.get("final_judge_contract") != _V3_FINAL_JUDGE_FREEZE
        or freeze.get("provider_forfeit_recovery")
        != _V3_PAUSE_RESUME_FORFEIT_FREEZE
        or runtime.get("router_contract") != expected_runtime_router
        or runtime.get("budget_forfeit_contract") != expected_runtime_budget
        or runtime.get("final_judge_contract") != expected_runtime_judge
        or {
            field: ledger.get(field) for field in expected_ledger_contract
        }
        != expected_ledger_contract
        or type(reservation_count) is not int
        or type(settlement_count) is not int
        or type(forfeit_count) is not int
        or type(terminal_count) is not int
        or ledger.get("unresolved_reservation_count") != 0
        or reservation_count != settlement_count + forfeit_count
        or terminal_count != settlement_count + forfeit_count
        or ledger.get("last_event_index")
        != reservation_count + terminal_count
        or ledger.get("launch_state_model_call_count") != reservation_count
        or not isinstance(forfeit_binding, Mapping)
        or forfeit_binding.get("attempt_policy_version")
        != PORTFOLIO_FAILURE_POLICY_VERSION
        or forfeit_binding.get("bound_forfeit_count") != forfeit_count
        or forfeit_binding.get("binding_fields")
        != ["forfeited_reservation_sha256", "budget_forfeit_sha256"]
        or forfeit_binding.get("exact_bidirectional_binding") is not True
        or not bindings_valid
        or len(bindings) != forfeit_count
        or len(binding_pairs) != forfeit_count
        or binding_set_sha256 != sha256_bytes(canonical_json_bytes(bindings))
        or any(
            ledger.get(field) != expected
            for field, expected in observed_forfeit_counts.items()
        )
        or forfeited_cost < 0
        or (forfeit_count == 0 and forfeited_cost != 0)
        or (forfeit_count > 0 and forfeited_cost <= 0)
        or not isinstance(ledger.get("ledger_inventory"), Mapping)
    ):
        raise Gate0CanaryEvidenceError(
            "evidence-v3 forfeit freeze/runtime/ledger proof is incomplete or drifted"
        )


def _repository_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise Gate0CanaryEvidenceError(f"{label} is missing")
    path = Path(value)
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    try:
        return candidate.resolve(strict=True)
    except OSError as error:
        raise Gate0CanaryEvidenceError(f"{label} is unavailable: {candidate}") from error


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_external_output(
    output: Path,
    *,
    scoped_roots: Sequence[Path],
) -> None:
    target = output.resolve(strict=False)
    if any(_is_within(target, root.resolve(strict=True)) for root in scoped_roots):
        raise Gate0CanaryEvidenceError(
            "evidence output must be outside execution, launch, and runtime roots"
        )


def _parallel_profile_binding(
    path: Path,
    *,
    expected_file_sha256: str,
    launch: object,
    freeze: Mapping[str, object],
) -> dict[str, object]:
    raw, content = _canonical_object(
        path,
        label="Core canary parallel profile",
        expected_file_sha256=expected_file_sha256,
    )
    try:
        profile = PortfolioParallelProfile.model_validate(raw, strict=True)
        validate_parallel_profile_sources(profile, repository_root=REPOSITORY_ROOT)
    except ValueError as error:
        raise Gate0CanaryEvidenceError(
            "Core canary parallel profile is invalid or source-drifted"
        ) from error
    if (
        profile.matrix_run_id != launch.plan.matrix_run_id
        or profile.launch_plan_sha256 != launch.plan.launch_plan_sha256
        or profile.execution_scope != "core_canary"
        or profile.assistant_concurrency != freeze["assistant_concurrency"]
        or profile.final_judge_concurrency != freeze["final_judge_concurrency"]
        or profile.qwen_requests_per_minute_cap
        != freeze["qwen_requests_per_minute_cap"]
        or profile.qwen_rate_limit_policy != "smooth_start_v1"
    ):
        raise Gate0CanaryEvidenceError(
            "Core canary parallel profile differs from the frozen execution"
        )
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": sha256_bytes(content),
        "profile_sha256": profile.profile_sha256,
        "matrix_run_id": profile.matrix_run_id,
        "launch_plan_sha256": profile.launch_plan_sha256,
        "execution_scope": profile.execution_scope,
        "worker_count": profile.worker_count,
        "assistant_concurrency": profile.assistant_concurrency,
        "final_judge_concurrency": profile.final_judge_concurrency,
        "qwen_requests_per_minute_cap": profile.qwen_requests_per_minute_cap,
        "qwen_rate_limit_policy": profile.qwen_rate_limit_policy,
        "scheduler_file_sha256": profile.scheduler_file_sha256,
        "parallel_module_file_sha256": profile.parallel_module_file_sha256,
    }


def _directory_files_once(root: Path, directory: Path) -> list[dict[str, object]]:
    root = root.absolute()
    directory = directory.absolute()
    if not _is_within(directory, root):
        raise Gate0CanaryEvidenceError("inventory directory escapes its root")
    try:
        root_stat = directory.lstat()
    except OSError as error:
        raise Gate0CanaryEvidenceError(
            f"inventory directory is unavailable: {directory}"
        ) from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise Gate0CanaryEvidenceError(
            f"inventory root must be a real directory: {directory}"
        )

    files: list[dict[str, object]] = []
    pending = [directory]
    while pending:
        current = pending.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError as error:
            raise Gate0CanaryEvidenceError(
                f"unable to enumerate evidence directory: {current}"
            ) from error
        for entry in entries:
            try:
                entry_stat = entry.lstat()
            except OSError as error:
                raise Gate0CanaryEvidenceError(
                    f"unable to inspect evidence entry: {entry}"
                ) from error
            if stat.S_ISLNK(entry_stat.st_mode):
                raise Gate0CanaryEvidenceError(
                    f"evidence tree contains a symlink: {entry}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                pending.append(entry)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise Gate0CanaryEvidenceError(
                    f"evidence tree contains a non-regular file: {entry}"
                )
            content = read_stable_regular_file(
                entry,
                label=f"Gate 0 canary artifact {entry.name}",
                max_bytes=_MAX_EVIDENCE_FILE_BYTES,
            )
            files.append(
                {
                    "relative_path": entry.relative_to(root).as_posix(),
                    "bytes": len(content),
                    "file_sha256": sha256_bytes(content),
                }
            )
    return sorted(files, key=lambda item: str(item["relative_path"]))


def _stable_directory_inventory(
    root: Path,
    directories: Sequence[Path],
) -> dict[str, object]:
    first: list[dict[str, object]] = []
    for directory in directories:
        first.extend(_directory_files_once(root, directory))
    first.sort(key=lambda item: str(item["relative_path"]))
    if len({str(item["relative_path"]) for item in first}) != len(first):
        raise Gate0CanaryEvidenceError("evidence inventory contains duplicate paths")

    second: list[dict[str, object]] = []
    for directory in directories:
        second.extend(_directory_files_once(root, directory))
    second.sort(key=lambda item: str(item["relative_path"]))
    if second != first:
        raise Gate0CanaryEvidenceError(
            "evidence artifacts changed while the snapshot was collected"
        )
    return {
        "file_count": len(first),
        "files": first,
        "artifact_set_sha256": sha256_bytes(canonical_json_bytes(first)),
    }


def _rate_record(count: int, denominator: int, limit: Decimal) -> dict[str, object]:
    if count < 0 or denominator <= 0 or count > denominator:
        raise Gate0CanaryEvidenceError("hard-error count/denominator is invalid")
    rate = Decimal(count) / Decimal(denominator)
    return {
        "hard_error_count": count,
        "denominator": denominator,
        "rate": format(rate.quantize(Decimal("0.000000000001")), "f"),
        "limit_exclusive": format(limit, "f"),
        "passed": rate < limit,
    }


def build_hard_error_gate(analysis: Mapping[str, object]) -> dict[str, object]:
    """Calculate the frozen Assistant and Judge operational gate exactly."""

    if (
        analysis.get("kind") != "portfolio-matrix-analysis"
        or analysis.get("dataset_profile") != "core"
        or analysis.get("row_count") != 125
        or analysis.get("selected_query_count") != 25
        or tuple(analysis.get("config_order", ())) != MAIN_CONFIG_ORDER
        or tuple(analysis.get("capabilities", ())) != tuple(CAPABILITIES)
        or analysis.get("model_calls_performed") != 0
    ):
        raise Gate0CanaryEvidenceError("Core canary analysis geometry is invalid")
    config_rows = analysis.get("config_summary")
    capability_rows = analysis.get("capability_summary")
    if not isinstance(config_rows, list) or not isinstance(capability_rows, list):
        raise Gate0CanaryEvidenceError("Core canary analysis summaries are missing")
    if (
        len(config_rows) != len(MAIN_CONFIG_ORDER)
        or {row.get("config") for row in config_rows if isinstance(row, Mapping)}
        != set(MAIN_CONFIG_ORDER)
        or any(
            not isinstance(row, Mapping) or row.get("query_count") != 25
            for row in config_rows
        )
    ):
        raise Gate0CanaryEvidenceError("Core canary config summaries are not 25 x 5")

    expected_pairs = {
        (config, capability)
        for config in MAIN_CONFIG_ORDER
        for capability in CAPABILITIES
    }
    observed_pairs: set[tuple[object, object]] = set()
    for row in capability_rows:
        if not isinstance(row, Mapping):
            raise Gate0CanaryEvidenceError("Core capability summary row is invalid")
        observed_pairs.add((row.get("config"), row.get("canonical_capability")))
        if row.get("evaluator_anomaly_count") != row.get("judge_non_scored_count"):
            raise Gate0CanaryEvidenceError(
                "Judge hard-error aliases differ in capability summary"
            )
    if observed_pairs != expected_pairs:
        raise Gate0CanaryEvidenceError(
            "first Core canary batch does not cover all six capabilities per config"
        )

    assistant_total = sum(int(row["assistant_hard_error_count"]) for row in config_rows)
    judge_total = sum(int(row["evaluator_anomaly_count"]) for row in config_rows)
    overall = {
        "assistant": _rate_record(assistant_total, 125, _OVERALL_LIMIT),
        "judge": _rate_record(judge_total, 125, _OVERALL_LIMIT),
    }
    by_capability: dict[str, object] = {}
    for capability in CAPABILITIES:
        selected = [
            row
            for row in capability_rows
            if row.get("canonical_capability") == capability
        ]
        denominator = sum(int(row["query_count"]) for row in selected)
        assistant = sum(int(row["assistant_hard_error_count"]) for row in selected)
        judge = sum(int(row["evaluator_anomaly_count"]) for row in selected)
        by_capability[capability] = {
            "assistant": _rate_record(
                assistant,
                denominator,
                _CAPABILITY_LIMIT,
            ),
            "judge": _rate_record(judge, denominator, _CAPABILITY_LIMIT),
        }
    passed = all(item["passed"] for item in overall.values()) and all(
        record["passed"]
        for capability in by_capability.values()
        for record in capability.values()
    )
    return {
        "policy": (
            "assistant_and_judge_separate_full_125_row_denominators_strict_limits"
        ),
        "judge_hard_error_semantics": (
            "assistant_success_and_final_judge_status_not_scored"
        ),
        "overall": overall,
        "by_capability": by_capability,
        "all_six_capabilities_covered": True,
        "any_single_error_fails_this_batch": True,
        "passed": passed,
    }


def _first_batch_geometry(
    *,
    control: Mapping[str, object],
    launch: object,
) -> tuple[str, tuple[object, ...]]:
    plan = launch.plan
    if (
        control.get("execution_scope") != "core_canary"
        or plan.kind != "portfolio-core-split-x5-launch-plan"
        or plan.dataset_profile != "core"
        or tuple(plan.selected_splits) != ("dev_mini",)
        or not plan.execution_ready
        or plan.status != "prepared_ready"
    ):
        raise Gate0CanaryEvidenceError(
            "execution is not a ready dev_mini-only Core canary"
        )
    batch_order = tuple(
        dict.fromkeys(str(shard.accepted_batch_id) for shard in plan.shards)
    )
    if not batch_order:
        raise Gate0CanaryEvidenceError("Core launch contains no frozen batches")
    first_batch = batch_order[0]
    selected = tuple(
        shard for shard in plan.shards if str(shard.accepted_batch_id) == first_batch
    )
    if (
        len(selected) != 5
        or tuple(str(shard.config) for shard in selected) != MAIN_CONFIG_ORDER
        or any(int(shard.query_count) != 25 for shard in selected)
        or len({tuple(shard.query_ids) for shard in selected}) != 1
        or control.get("authorized_batch_id") != first_batch
        or tuple(control.get("authorized_shard_ids", ()))
        != tuple(str(shard.shard_id) for shard in selected)
        or control.get("local_authorized_shard_count") != 5
        or control.get("comparison_shard_count") != 5
    ):
        raise Gate0CanaryEvidenceError(
            "Core canary authority is not the first 25-query x5 batch"
        )
    completed = set(launch.state.completed_shard_ids)
    selected_ids = {str(shard.shard_id) for shard in selected}
    if completed != selected_ids:
        raise Gate0CanaryEvidenceError(
            "Core canary launch state is not exactly the authorized first batch"
        )
    return first_batch, selected


def _full_alias_binding(
    *,
    control: Mapping[str, object],
    audit: Mapping[str, object],
    selected_shards: Sequence[object],
) -> dict[str, object]:
    aliases = control.get("execution_artifact_aliases")
    if not isinstance(aliases, list) or len(aliases) != 1:
        raise Gate0CanaryEvidenceError("Core canary requires exactly one Full alias")
    alias = aliases[0]
    if (
        not isinstance(alias, Mapping)
        or alias.get("target_config") != "full"
        or alias.get("source_config") != "s1s2"
        or alias.get("stage_decision") != "rolled_back"
        or alias.get("provider_model_call_count") != 0
        or alias.get("source_bank_sha256") != alias.get("target_bank_sha256")
        or alias.get("source_bank_file_sha256")
        != alias.get("target_bank_file_sha256")
        or control.get("execution_artifact_alias_provider_model_call_count") != 0
    ):
        raise Gate0CanaryEvidenceError("Full alias is not an exact zero-call rollback")
    completion = audit.get("completion")
    bindings = audit.get("bindings")
    if not isinstance(completion, Mapping) or not isinstance(bindings, Mapping):
        raise Gate0CanaryEvidenceError("batch audit lacks alias evidence")
    full = completion.get("full")
    internal = bindings.get("internal_artifact_aliases")
    if (
        not isinstance(full, Mapping)
        or full.get("artifact_source") != "accepted_parent_artifact_alias"
        or full.get("assistant_checkpoint_count") != 25
        or full.get("final_checkpoint_count") != 25
        or not isinstance(internal, list)
        or len(internal) != 1
        or not isinstance(internal[0], Mapping)
        or internal[0].get("source_config") != "s1s2"
        or internal[0].get("target_config") != "full"
        or internal[0].get("provider_model_call_count") != 0
    ):
        raise Gate0CanaryEvidenceError("physical Full alias evidence is incomplete")
    source = next(item for item in selected_shards if item.config == "s1s2")
    target = next(item for item in selected_shards if item.config == "full")
    if full.get("physical_shard_id") != source.shard_id:
        raise Gate0CanaryEvidenceError("Full alias physical source differs from S1+S2")
    return {
        "policy_version": control["execution_artifact_alias_policy_version"],
        "alias_sha256": alias.get("alias_sha256"),
        "source_config": "s1s2",
        "target_config": "full",
        "source_shard_id": source.shard_id,
        "target_shard_id": target.shard_id,
        "provider_model_call_count": 0,
        "physical_artifact_evidence": internal[0],
    }


def _forfeit_attempt_receipt_binding(
    *,
    execution_root: Path,
    selected_shards: Sequence[object],
    ledger: object,
) -> dict[str, object]:
    """Prove every budget forfeit is referenced by exactly one attempt-v4 receipt."""

    reservations_by_sha = {
        item.reservation_sha256: item for item in ledger.reservations
    }
    forfeits_by_sha = {item.forfeit_sha256: item for item in ledger.forfeits}
    references: list[dict[str, object]] = []
    for shard in selected_shards:
        config = str(getattr(shard, "config", ""))
        if config == "full":
            continue
        output_relpath = getattr(shard, "output_relpath", None)
        if not isinstance(output_relpath, (str, Path)):
            raise Gate0CanaryEvidenceError(
                "Core canary shard lacks an attempt-receipt location"
            )
        receipt_root = execution_root / output_relpath / "attempt-receipts"
        if not receipt_root.exists():
            continue
        for path in sorted(receipt_root.glob("*.json")):
            content = read_stable_regular_file(
                path,
                label=f"Core canary attempt receipt {path.name}",
                max_bytes=_MAX_EVIDENCE_FILE_BYTES,
            )
            try:
                receipt = PortfolioAttemptReceipt.model_validate_json(
                    content,
                    strict=True,
                )
            except ValueError as error:
                raise Gate0CanaryEvidenceError(
                    "Core canary attempt receipt is invalid"
                ) from error
            if canonical_json_bytes(receipt.model_dump(mode="json")) != content:
                raise Gate0CanaryEvidenceError(
                    "Core canary attempt receipt is not canonical JSON"
                )
            if (
                receipt.policy_version != PORTFOLIO_FAILURE_POLICY_VERSION
                or receipt.shard_id != str(getattr(shard, "shard_id", ""))
                or receipt.config != config
            ):
                raise Gate0CanaryEvidenceError(
                    "Core canary attempt receipt differs from active shard policy"
                )
            reservation_sha = receipt.forfeited_reservation_sha256
            forfeit_sha = receipt.budget_forfeit_sha256
            if reservation_sha is None and forfeit_sha is None:
                continue
            if reservation_sha is None or forfeit_sha is None:
                raise Gate0CanaryEvidenceError(
                    "attempt receipt contains a partial forfeit binding"
                )
            reservation = reservations_by_sha.get(reservation_sha)
            forfeit = forfeits_by_sha.get(forfeit_sha)
            if reservation is None or forfeit is None:
                raise Gate0CanaryEvidenceError(
                    "attempt receipt references an unknown reservation or forfeit"
                )
            identity = reservation.identity
            if receipt.failure_subtype == "orphaned_provider_call":
                # Recovery can find either an unresolved reservation, which it
                # forfeits itself, or a provider-side forfeit that was already
                # persisted before the owning process exited.  Both are valid
                # terminal outcomes for the subsequently recovered receipt.
                failure_matches = forfeit.reason in {
                    "provider_call_ended_without_captured_response",
                    "orphan_recovered_after_owner_exit",
                }
            else:
                failure_matches = (
                    receipt.failure_subtype.startswith("provider_pre_response")
                    and forfeit.reason
                    == "provider_call_ended_without_captured_response"
                )
            if (
                forfeit.reservation_sha256 != reservation_sha
                or forfeit.identity_sha256 != identity.identity_sha256
                or receipt.matrix_run_id != identity.matrix_run_id
                or receipt.shard_id != identity.shard_id
                or receipt.config != identity.config
                or receipt.query_id != identity.query_id
                or receipt.instance_sha256 != identity.instance_sha256
                or receipt.request_sha256 != identity.request_sha256
                or receipt.failure_stage != identity.stage
                or receipt.attempt_index != identity.attempt_index
                or not failure_matches
            ):
                raise Gate0CanaryEvidenceError(
                    "attempt receipt and budget forfeit call identity differ"
                )
            references.append(
                {
                    "reservation_sha256": reservation_sha,
                    "forfeit_sha256": forfeit_sha,
                    "attempt_receipt_sha256": receipt.receipt_sha256,
                    "config": identity.config,
                    "stage": identity.stage,
                    "reason": forfeit.reason,
                }
            )

    references.sort(
        key=lambda item: (
            str(item["reservation_sha256"]),
            str(item["forfeit_sha256"]),
            str(item["attempt_receipt_sha256"]),
        )
    )
    reference_counts = Counter(
        (item["reservation_sha256"], item["forfeit_sha256"])
        for item in references
    )
    expected_pairs = {
        (item.reservation_sha256, item.forfeit_sha256)
        for item in ledger.forfeits
    }
    if (
        set(reference_counts) != expected_pairs
        or any(count != 1 for count in reference_counts.values())
    ):
        raise Gate0CanaryEvidenceError(
            "budget forfeits do not have exact bidirectional attempt-v4 bindings"
        )
    return {
        "attempt_policy_version": PORTFOLIO_FAILURE_POLICY_VERSION,
        "binding_fields": [
            "forfeited_reservation_sha256",
            "budget_forfeit_sha256",
        ],
        "bound_forfeit_count": len(references),
        "binding_set_sha256": sha256_bytes(canonical_json_bytes(references)),
        "bindings": references,
        "exact_bidirectional_binding": True,
    }


def _ledger_binding(
    *,
    execution_root: Path,
    control: Mapping[str, object],
    selected_shards: Sequence[object],
    expected_model_call_count: int,
) -> dict[str, object]:
    if control.get("budget_ledger_relpath") != "budget-ledger":
        raise Gate0CanaryEvidenceError("Core canary budget ledger layout drifted")
    ledger_root = execution_root / "budget-ledger"
    ledger = load_portfolio_budget_ledger(ledger_root)
    matrix_run_id = control.get("matrix_run_id")
    terminal_count = len(ledger.settlements) + len(ledger.forfeits)
    if (
        not isinstance(matrix_run_id, str)
        or not matrix_run_id.strip()
        or ledger.authority.matrix_run_id != matrix_run_id
        or ledger.authority.policy_version != PORTFOLIO_BUDGET_POLICY_VERSION
        or ledger.authority.accounting_policy
        != _V3_BUDGET_FREEZE["accounting_policy"]
        or ledger.authority.exception_policy
        != _V3_BUDGET_FREEZE["exception_policy"]
        or str(ledger.authority.phase_cap_cny)
        != str(Decimal(str(control["phase_cumulative_cap_cny"])))
        or ledger.unresolved_reservations
        or len(ledger.reservations) != terminal_count
        or type(expected_model_call_count) is not int
        or expected_model_call_count <= 0
        or len(ledger.reservations) != expected_model_call_count
    ):
        raise Gate0CanaryEvidenceError(
            "Core canary provider ledger is unresolved or differs from authority"
        )
    authorized_by_id = {str(item.shard_id): item for item in selected_shards}
    identities = tuple(item.identity for item in ledger.reservations)
    if any(
        identity.matrix_run_id != matrix_run_id
        or identity.shard_id not in authorized_by_id
        or identity.config
        != str(getattr(authorized_by_id.get(identity.shard_id), "config", ""))
        or identity.query_id
        not in tuple(
            getattr(authorized_by_id.get(identity.shard_id), "query_ids", ())
        )
        or identity.config == "full"
        for identity in identities
    ):
        raise Gate0CanaryEvidenceError(
            "provider ledger contains an out-of-scope or Full-alias call"
        )
    identity_hashes = sorted(identity.identity_sha256 for identity in identities)
    counts_by_config = Counter(identity.config for identity in identities)
    counts_by_stage = Counter(identity.stage for identity in identities)
    reservations_by_sha = {
        item.reservation_sha256: item for item in ledger.reservations
    }
    forfeit_counts_by_config = Counter(
        reservations_by_sha[item.reservation_sha256].identity.config
        for item in ledger.forfeits
    )
    forfeit_counts_by_stage = Counter(
        reservations_by_sha[item.reservation_sha256].identity.stage
        for item in ledger.forfeits
    )
    forfeit_counts_by_reason = Counter(item.reason for item in ledger.forfeits)
    forfeit_binding = _forfeit_attempt_receipt_binding(
        execution_root=execution_root,
        selected_shards=selected_shards,
        ledger=ledger,
    )
    inventory = _stable_directory_inventory(execution_root, (ledger_root,))
    return {
        "authority_sha256": ledger.authority.authority_sha256,
        "policy_version": ledger.authority.policy_version,
        "accounting_policy": ledger.authority.accounting_policy,
        "exception_policy": ledger.authority.exception_policy,
        "terminal_outcome_policy": "exactly_one_of_settlement_or_forfeit",
        "reservation_count": len(ledger.reservations),
        "settlement_count": len(ledger.settlements),
        "forfeit_count": len(ledger.forfeits),
        "terminal_outcome_count": terminal_count,
        "unresolved_reservation_count": 0,
        "provider_call_identity_count": len(identity_hashes),
        "launch_state_model_call_count": expected_model_call_count,
        "provider_call_identity_set_sha256": sha256_bytes(
            canonical_json_bytes(identity_hashes)
        ),
        "provider_call_counts_by_config": dict(sorted(counts_by_config.items())),
        "provider_call_counts_by_stage": dict(sorted(counts_by_stage.items())),
        "forfeit_counts_by_config": dict(
            sorted(forfeit_counts_by_config.items())
        ),
        "forfeit_counts_by_stage": dict(sorted(forfeit_counts_by_stage.items())),
        "forfeit_counts_by_reason": dict(
            sorted(forfeit_counts_by_reason.items())
        ),
        "full_provider_call_count": counts_by_config.get("full", 0),
        "last_event_index": ledger.last_event_index,
        "last_event_sha256": ledger.last_event_sha256,
        "settled_actual_cost_cny": format(
            ledger.settled_actual_cost_cny,
            "f",
        ),
        "forfeited_reserved_cost_cny": format(
            ledger.forfeited_reserved_cost_cny,
            "f",
        ),
        "accountable_cost_cny": format(ledger.accountable_cost_cny, "f"),
        "forfeit_attempt_receipt_binding": forfeit_binding,
        "ledger_inventory": inventory,
    }


def _collect_canary_state(
    *,
    execution_root: Path,
    freeze_path: Path,
    freeze_file_sha256: str,
    parallel_profile_path: Path,
    parallel_profile_file_sha256: str,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    try:
        execution_root = execution_root.resolve(strict=True)
    except OSError as error:
        raise Gate0CanaryEvidenceError(
            f"Core canary execution root is unavailable: {execution_root}"
        ) from error
    control = _load_control(execution_root)
    launch_root = _repository_path(control.get("launch_root"), "launch root")
    runtime_root = _repository_path(control.get("runtime_root"), "runtime root")
    launch = load_portfolio_launch_package(
        launch_root,
        expected_plan_file_sha256=str(control.get("launch_plan_file_sha256")),
    )
    if (
        control.get("launch_plan_sha256") != launch.plan.launch_plan_sha256
        or control.get("matrix_run_id") != launch.plan.matrix_run_id
    ):
        raise Gate0CanaryEvidenceError("execution control differs from Core launch")
    freeze = _freeze_binding(freeze_path, freeze_file_sha256)
    parallel_profile = _parallel_profile_binding(
        parallel_profile_path,
        expected_file_sha256=parallel_profile_file_sha256,
        launch=launch,
        freeze=freeze,
    )
    first_batch, selected_shards = _first_batch_geometry(
        control=control,
        launch=launch,
    )
    audit = audit_portfolio_batch(execution_root, batch_id=first_batch)
    if (
        audit.get("status") not in {"passed_clean", "review_required"}
        or audit.get("blockers") != []
        or audit.get("dataset_profile") != "core"
        or audit.get("selected_splits") != ["dev_mini"]
        or audit.get("batch_split") != "dev_mini"
        or audit.get("query_count") != 25
        or audit.get("row_count") != 125
        or tuple(audit.get("config_order", ())) != MAIN_CONFIG_ORDER
        or audit.get("model_calls_performed") != 0
    ):
        raise Gate0CanaryEvidenceError("Core canary batch audit is unusable")
    completion = audit.get("completion")
    if not isinstance(completion, Mapping) or any(
        not isinstance(completion.get(config), Mapping)
        or completion[config].get("assistant_checkpoint_count") != 25
        or completion[config].get("final_checkpoint_count") != 25
        for config in MAIN_CONFIG_ORDER
    ):
        raise Gate0CanaryEvidenceError("Core canary checkpoints are incomplete")
    alias = _full_alias_binding(
        control=control,
        audit=audit,
        selected_shards=selected_shards,
    )

    with tempfile.TemporaryDirectory(prefix="portfolio-core-gate0-analysis-") as temp:
        analysis = analyze_portfolio_matrix(
            execution_root,
            batch_id=first_batch,
            output_dir=Path(temp) / "analysis",
        )
    if (
        analysis.get("selected_batch_ids") != [first_batch]
        or analysis.get("selected_splits") != ["dev_mini"]
    ):
        raise Gate0CanaryEvidenceError("Core canary analysis selected another batch")
    hard_error_gate = build_hard_error_gate(analysis)

    control_content = read_stable_regular_file(
        execution_root / "execution-control.json",
        label="Core canary execution control",
        max_bytes=_MAX_EVIDENCE_FILE_BYTES,
    )
    launch_files = {
        name: sha256_bytes(
            read_stable_regular_file(
                launch_root / name,
                label=f"Core launch {name}",
                max_bytes=_MAX_EVIDENCE_FILE_BYTES,
            )
        )
        for name in ("launch-plan.json", "instances.jsonl", "run-state.json")
    }
    runtime_lock = _runtime_lock_binding(
        runtime_root=runtime_root,
        control=control,
        freeze=freeze,
    )

    shard_directories = tuple(
        execution_root / str(shard.output_relpath) for shard in selected_shards
    )
    artifact_inventory = _stable_directory_inventory(
        execution_root,
        (
            *shard_directories,
            execution_root / "shared-routes" / first_batch,
        ),
    )
    ledger = _ledger_binding(
        execution_root=execution_root,
        control=control,
        selected_shards=selected_shards,
        expected_model_call_count=launch.state.model_calls_performed,
    )
    shard_bindings = [
        {
            "shard_id": str(shard.shard_id),
            "shard_ordinal": int(shard.shard_ordinal),
            "shard_sha256": str(shard.shard_sha256),
            "config": str(shard.config),
            "accepted_batch_id": str(shard.accepted_batch_id),
            "query_count": int(shard.query_count),
        }
        for shard in selected_shards
    ]
    invariants = {
        "freeze": freeze,
        "execution": {
            "control_file_sha256": sha256_bytes(control_content),
            "control_sha256": control.get("control_sha256"),
            "matrix_run_id": launch.plan.matrix_run_id,
            "execution_scope": "core_canary",
        },
        "launch": {
            "plan_file_sha256": launch.plan_file_sha256,
            "plan_sha256": launch.plan.launch_plan_sha256,
            "instances_file_sha256": launch.plan.instances_file_sha256,
            "package_file_sha256s": launch_files,
            "selected_splits": ["dev_mini"],
        },
        "runtime": runtime_lock,
        "parallel_profile": parallel_profile,
        "rerun_command": _frozen_rerun_command_binding(
            execution_root=execution_root,
            parallel_profile_path=parallel_profile_path,
        ),
        "canary_geometry": {
            "batch_id": first_batch,
            "query_count": 25,
            "logical_row_count": 125,
            "config_order": list(MAIN_CONFIG_ORDER),
            "capabilities": list(CAPABILITIES),
            "logical_shards": shard_bindings,
        },
        "full_alias": alias,
        "audit": {
            "status": audit.get("status"),
            "audit_sha256": audit.get("audit_sha256"),
            "artifact_set_sha256": audit.get("artifact_set_sha256"),
            "review_flags": audit.get("review_flags"),
        },
        "analysis": {
            "analysis_sha256": analysis.get("analysis_sha256"),
            "row_set_sha256": analysis.get("row_set_sha256"),
            "selected_batch_ids": analysis.get("selected_batch_ids"),
            "selected_query_count": analysis.get("selected_query_count"),
            "row_count": analysis.get("row_count"),
        },
        "hard_error_gate": hard_error_gate,
        "provider_ledger": ledger,
        "checkpoint_terminal_inventory": artifact_inventory,
    }
    return invariants, (execution_root, launch_root, runtime_root)


def _snapshot_payload(invariants: Mapping[str, object]) -> dict[str, object]:
    _require_v3_evidence_bindings(invariants)
    return _content_address(
        {
            "schema_version": 1,
            "kind": SNAPSHOT_KIND,
            "policy_version": POLICY_VERSION,
            "track": "portfolio",
            "formal_eligible": False,
            "phase": "pre_rerun",
            "status": "captured",
            "invariants": dict(invariants),
            "hard_error_gate_passed": invariants["hard_error_gate"]["passed"],
            "model_calls_performed": 0,
        },
        field="snapshot_sha256",
    )


def _load_snapshot(path: Path, expected_file_sha256: str) -> tuple[dict, bytes]:
    raw, content = _canonical_object(
        path,
        label="Core Gate 0 canary pre-rerun snapshot",
        expected_file_sha256=expected_file_sha256,
    )
    supplied = raw.get("snapshot_sha256")
    unsigned = dict(raw)
    unsigned.pop("snapshot_sha256", None)
    if (
        supplied != _hash_payload(unsigned)
        or raw.get("kind") != SNAPSHOT_KIND
        or raw.get("policy_version") not in _SUPPORTED_POLICY_VERSIONS
        or raw.get("phase") != "pre_rerun"
        or raw.get("status") != "captured"
        or raw.get("model_calls_performed") != 0
        or not isinstance(raw.get("invariants"), Mapping)
    ):
        raise Gate0CanaryEvidenceError("pre-rerun snapshot is invalid")
    if raw.get("policy_version") == PREVIOUS_POLICY_VERSION:
        _require_v2_evidence_bindings(raw["invariants"])
    elif raw.get("policy_version") == POLICY_VERSION:
        _require_v3_evidence_bindings(raw["invariants"])
    return raw, content


def _utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Gate0CanaryEvidenceError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise Gate0CanaryEvidenceError(f"{label} is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise Gate0CanaryEvidenceError(f"{label} must use UTC")
    return parsed


def _exact_rerun_argv(
    *,
    execution_root: Path,
    parallel_profile_path: Path,
) -> tuple[str, ...]:
    return (
        str(Path(sys.executable).resolve(strict=True)),
        str((REPOSITORY_ROOT / "scripts" / "run_portfolio_matrix.py").resolve(strict=True)),
        "--execution-root",
        str(execution_root.resolve(strict=True)),
        "--execute",
        "--parallel-profile",
        str(parallel_profile_path.resolve(strict=True)),
    )


def _frozen_rerun_command_binding(
    *,
    execution_root: Path,
    parallel_profile_path: Path,
) -> dict[str, object]:
    argv = _exact_rerun_argv(
        execution_root=execution_root,
        parallel_profile_path=parallel_profile_path,
    )
    run_matrix_sha256 = sha256_bytes(
        read_stable_regular_file(
            REPOSITORY_ROOT / "scripts" / "run_portfolio_matrix.py",
            label="Core canary matrix runner",
            max_bytes=_MAX_EVIDENCE_FILE_BYTES,
        )
    )
    return {
        "working_directory": str(REPOSITORY_ROOT.resolve(strict=True)),
        "argv": list(argv),
        "argv_sha256": sha256_bytes(canonical_json_bytes(list(argv))),
        "run_matrix_file_sha256": run_matrix_sha256,
    }


def _rerun_binding_sections(
    invariants: Mapping[str, object],
) -> dict[str, object]:
    required = (
        "execution",
        "launch",
        "runtime",
        "parallel_profile",
        "rerun_command",
    )
    if any(not isinstance(invariants.get(name), Mapping) for name in required):
        raise Gate0CanaryEvidenceError("rerun command bindings are incomplete")
    return {name: dict(invariants[name]) for name in required}


def _rerun_payload(
    *,
    snapshot: Mapping[str, object],
    snapshot_file_sha256: str,
    pre_run_invariants: Mapping[str, object],
    argv: Sequence[str],
    started_at_utc: str,
    finished_at_utc: str,
    exit_code: int,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, object]:
    before = snapshot.get("invariants")
    evidence_policy_version = snapshot.get("policy_version")
    if not isinstance(before, Mapping) or dict(before) != dict(pre_run_invariants):
        raise Gate0CanaryEvidenceError(
            "pre-rerun state differs from the bound snapshot"
        )
    if evidence_policy_version not in _SUPPORTED_POLICY_VERSIONS:
        raise Gate0CanaryEvidenceError("snapshot evidence policy is unsupported")
    if evidence_policy_version == PREVIOUS_POLICY_VERSION:
        _require_v2_evidence_bindings(before)
    elif evidence_policy_version == POLICY_VERSION:
        _require_v3_evidence_bindings(before)
    started = _parse_utc_timestamp(started_at_utc, "rerun start")
    finished = _parse_utc_timestamp(finished_at_utc, "rerun finish")
    if finished < started:
        raise Gate0CanaryEvidenceError("rerun finish precedes its start")
    if type(exit_code) is not int or not isinstance(stdout, bytes) or not isinstance(
        stderr,
        bytes,
    ):
        raise Gate0CanaryEvidenceError("rerun process result is invalid")
    command = tuple(argv)
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise Gate0CanaryEvidenceError("rerun argv is invalid")
    run_matrix_content = read_stable_regular_file(
        REPOSITORY_ROOT / "scripts" / "run_portfolio_matrix.py",
        label="Core canary matrix runner",
        max_bytes=_MAX_EVIDENCE_FILE_BYTES,
    )
    run_matrix_sha256 = sha256_bytes(run_matrix_content)
    parallel = pre_run_invariants["parallel_profile"]
    frozen_command = pre_run_invariants["rerun_command"]
    if (
        not isinstance(parallel, Mapping)
        or parallel.get("scheduler_file_sha256") != run_matrix_sha256
        or not isinstance(frozen_command, Mapping)
        or tuple(frozen_command.get("argv", ())) != command
        or frozen_command.get("argv_sha256")
        != sha256_bytes(canonical_json_bytes(list(command)))
        or frozen_command.get("working_directory")
        != str(REPOSITORY_ROOT.resolve(strict=True))
        or frozen_command.get("run_matrix_file_sha256") != run_matrix_sha256
    ):
        raise Gate0CanaryEvidenceError(
            "rerun command differs from the pre-rerun frozen command"
        )
    payload = {
        "schema_version": 1,
        "kind": RERUN_KIND,
        "policy_version": evidence_policy_version,
        "track": "portfolio",
        "formal_eligible": False,
        "phase": "frozen_command_rerun",
        "status": "completed_success" if exit_code == 0 else "completed_failure",
        "pre_rerun_snapshot_file_sha256": _require_sha256(
            snapshot_file_sha256,
            "pre-rerun snapshot file SHA-256",
        ),
        "pre_rerun_snapshot_sha256": snapshot.get("snapshot_sha256"),
        "pre_run_invariants_sha256": _hash_payload(dict(pre_run_invariants)),
        "bindings": _rerun_binding_sections(pre_run_invariants),
        "command": {
            "working_directory": str(REPOSITORY_ROOT.resolve(strict=True)),
            "argv": list(command),
            "argv_sha256": sha256_bytes(canonical_json_bytes(list(command))),
            "run_matrix_file_sha256": run_matrix_sha256,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "exit_code": exit_code,
            "stdout_sha256": sha256_bytes(stdout),
            "stderr_sha256": sha256_bytes(stderr),
            "captured_stream_policy": "sha256_only_no_stream_content_persisted",
        },
        "provider_call_delta_disposition": "deferred_to_final_verify",
    }
    return _content_address(payload, field="rerun_receipt_sha256")


def _load_rerun_receipt(
    path: Path,
    expected_file_sha256: str,
    *,
    snapshot: Mapping[str, object],
    snapshot_file_sha256: str,
    execution_root: Path,
    parallel_profile_path: Path,
) -> tuple[dict[str, object], bytes]:
    raw, content = _canonical_object(
        path,
        label="Core Gate 0 frozen-command rerun receipt",
        expected_file_sha256=expected_file_sha256,
    )
    supplied = raw.get("rerun_receipt_sha256")
    unsigned = dict(raw)
    unsigned.pop("rerun_receipt_sha256", None)
    command = raw.get("command")
    before = snapshot.get("invariants")
    if not isinstance(before, Mapping):
        raise Gate0CanaryEvidenceError("pre-rerun snapshot invariants are missing")
    expected_argv = _exact_rerun_argv(
        execution_root=execution_root,
        parallel_profile_path=parallel_profile_path,
    )
    expected_source_sha256 = sha256_bytes(
        read_stable_regular_file(
            REPOSITORY_ROOT / "scripts" / "run_portfolio_matrix.py",
            label="Core canary matrix runner",
            max_bytes=_MAX_EVIDENCE_FILE_BYTES,
        )
    )
    expected_policy_version = snapshot.get("policy_version")
    if (
        supplied != _hash_payload(unsigned)
        or raw.get("schema_version") != 1
        or raw.get("kind") != RERUN_KIND
        or expected_policy_version not in _SUPPORTED_POLICY_VERSIONS
        or raw.get("policy_version") != expected_policy_version
        or raw.get("track") != "portfolio"
        or raw.get("formal_eligible") is not False
        or raw.get("phase") != "frozen_command_rerun"
        or raw.get("status") != "completed_success"
        or raw.get("pre_rerun_snapshot_file_sha256")
        != _require_sha256(
            snapshot_file_sha256,
            "pre-rerun snapshot file SHA-256",
        )
        or raw.get("pre_rerun_snapshot_sha256") != snapshot.get("snapshot_sha256")
        or raw.get("pre_run_invariants_sha256") != _hash_payload(dict(before))
        or raw.get("bindings") != _rerun_binding_sections(before)
        or not isinstance(command, Mapping)
        or command.get("working_directory")
        != str(REPOSITORY_ROOT.resolve(strict=True))
        or tuple(command.get("argv", ())) != expected_argv
        or command.get("argv_sha256")
        != sha256_bytes(canonical_json_bytes(list(expected_argv)))
        or command.get("run_matrix_file_sha256") != expected_source_sha256
        or type(command.get("exit_code")) is not int
        or command.get("exit_code") != 0
        or command.get("captured_stream_policy")
        != "sha256_only_no_stream_content_persisted"
        or raw.get("provider_call_delta_disposition")
        != "deferred_to_final_verify"
    ):
        raise Gate0CanaryEvidenceError(
            "frozen-command rerun receipt is invalid or command-drifted"
        )
    _require_sha256(str(command.get("stdout_sha256")), "rerun stdout SHA-256")
    _require_sha256(str(command.get("stderr_sha256")), "rerun stderr SHA-256")
    started = _parse_utc_timestamp(command.get("started_at_utc"), "rerun start")
    finished = _parse_utc_timestamp(command.get("finished_at_utc"), "rerun finish")
    if finished < started:
        raise Gate0CanaryEvidenceError("rerun finish precedes its start")
    parallel = before.get("parallel_profile")
    if (
        not isinstance(parallel, Mapping)
        or parallel.get("path") != str(parallel_profile_path.resolve(strict=True))
        or parallel.get("scheduler_file_sha256") != expected_source_sha256
    ):
        raise Gate0CanaryEvidenceError(
            "rerun receipt differs from the frozen parallel profile"
        )
    return raw, content


def _difference_keys(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> list[str]:
    return sorted(
        key
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    )


def build_final_receipt(
    *,
    snapshot: Mapping[str, object],
    snapshot_file_sha256: str,
    rerun_receipt: Mapping[str, object],
    rerun_receipt_file_sha256: str,
    current_invariants: Mapping[str, object],
) -> dict[str, object]:
    before = snapshot.get("invariants")
    evidence_policy_version = snapshot.get("policy_version")
    if not isinstance(before, Mapping):
        raise Gate0CanaryEvidenceError("snapshot invariants are missing")
    if evidence_policy_version not in _SUPPORTED_POLICY_VERSIONS:
        raise Gate0CanaryEvidenceError("snapshot evidence policy is unsupported")
    if evidence_policy_version == PREVIOUS_POLICY_VERSION:
        _require_v2_evidence_bindings(before)
        _require_v2_evidence_bindings(current_invariants)
    elif evidence_policy_version == POLICY_VERSION:
        _require_v3_evidence_bindings(before)
        _require_v3_evidence_bindings(current_invariants)
    rerun_unsigned = dict(rerun_receipt)
    rerun_supplied = rerun_unsigned.pop("rerun_receipt_sha256", None)
    rerun_command = rerun_receipt.get("command")
    frozen_command = before.get("rerun_command")
    rerun_succeeded = (
        rerun_supplied == _hash_payload(rerun_unsigned)
        and rerun_receipt.get("kind") == RERUN_KIND
        and rerun_receipt.get("policy_version") == evidence_policy_version
        and rerun_receipt.get("status") == "completed_success"
        and rerun_receipt.get("pre_rerun_snapshot_file_sha256")
        == _require_sha256(
            snapshot_file_sha256,
            "pre-rerun snapshot file SHA-256",
        )
        and rerun_receipt.get("pre_rerun_snapshot_sha256")
        == snapshot.get("snapshot_sha256")
        and rerun_receipt.get("pre_run_invariants_sha256")
        == _hash_payload(dict(before))
        and rerun_receipt.get("bindings") == _rerun_binding_sections(before)
        and isinstance(rerun_command, Mapping)
        and isinstance(frozen_command, Mapping)
        and all(
            rerun_command.get(field) == frozen_command.get(field)
            for field in (
                "working_directory",
                "argv",
                "argv_sha256",
                "run_matrix_file_sha256",
            )
        )
        and rerun_command.get("exit_code") == 0
    )
    if not rerun_succeeded:
        raise Gate0CanaryEvidenceError(
            "successful frozen-command rerun receipt is required"
        )
    differences = _difference_keys(before, current_invariants)
    before_ledger = before.get("provider_ledger")
    after_ledger = current_invariants.get("provider_ledger")
    before_artifacts = before.get("checkpoint_terminal_inventory")
    after_artifacts = current_invariants.get("checkpoint_terminal_inventory")
    if not all(
        isinstance(value, Mapping)
        for value in (before_ledger, after_ledger, before_artifacts, after_artifacts)
    ):
        raise Gate0CanaryEvidenceError("snapshot rerun evidence is incomplete")
    provider_count_delta = int(after_ledger["reservation_count"]) - int(
        before_ledger["reservation_count"]
    )
    settlement_count_delta = int(after_ledger["settlement_count"]) - int(
        before_ledger["settlement_count"]
    )
    forfeit_count_delta = int(after_ledger.get("forfeit_count", 0)) - int(
        before_ledger.get("forfeit_count", 0)
    )
    before_terminal_count = int(
        before_ledger.get(
            "terminal_outcome_count",
            int(before_ledger["settlement_count"])
            + int(before_ledger.get("forfeit_count", 0)),
        )
    )
    after_terminal_count = int(
        after_ledger.get(
            "terminal_outcome_count",
            int(after_ledger["settlement_count"])
            + int(after_ledger.get("forfeit_count", 0)),
        )
    )
    terminal_count_delta = after_terminal_count - before_terminal_count
    ledger_event_delta = int(after_ledger["last_event_index"]) - int(
        before_ledger["last_event_index"]
    )
    ledger_inventory_unchanged = (
        before_ledger.get("ledger_inventory")
        == after_ledger.get("ledger_inventory")
    )
    provider_unchanged = (
        before_ledger == after_ledger
        and provider_count_delta == 0
        and settlement_count_delta == 0
        and forfeit_count_delta == 0
        and terminal_count_delta == 0
        and ledger_event_delta == 0
        and ledger_inventory_unchanged
    )
    artifacts_unchanged = before_artifacts == after_artifacts
    hard_error_gate = current_invariants.get("hard_error_gate")
    hard_error_passed = (
        isinstance(hard_error_gate, Mapping) and hard_error_gate.get("passed") is True
    )
    passed = (
        rerun_succeeded
        and not differences
        and provider_unchanged
        and artifacts_unchanged
        and hard_error_passed
    )
    status = (
        "passed"
        if passed
        else (
            "failed_rerun_drift"
            if differences or not provider_unchanged or not artifacts_unchanged
            else "failed_hard_error_gate"
        )
    )
    return _content_address(
        {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "policy_version": evidence_policy_version,
            "track": "portfolio",
            "formal_eligible": False,
            "phase": "post_rerun_verification",
            "status": status,
            "gate0_canary_passed": passed,
            "pre_rerun_snapshot_file_sha256": _require_sha256(
                snapshot_file_sha256,
                "pre-rerun snapshot file SHA-256",
            ),
            "pre_rerun_snapshot_sha256": snapshot.get("snapshot_sha256"),
            "rerun_receipt_file_sha256": _require_sha256(
                rerun_receipt_file_sha256,
                "frozen-command rerun receipt file SHA-256",
            ),
            "rerun_receipt_sha256": rerun_supplied,
            "rerun": {
                "frozen_command_invocation_proved": rerun_succeeded,
                "command": dict(rerun_command),
                "changed_invariant_sections": differences,
                "provider_call_count_delta": provider_count_delta,
                "reservation_count_delta": provider_count_delta,
                "settlement_count_delta": settlement_count_delta,
                "forfeit_count_delta": forfeit_count_delta,
                "terminal_outcome_count_delta": terminal_count_delta,
                "ledger_event_count_delta": ledger_event_delta,
                "ledger_inventory_unchanged": ledger_inventory_unchanged,
                "provider_call_identity_and_count_unchanged": provider_unchanged,
                "checkpoint_and_terminal_artifact_bytes_unchanged": (
                    artifacts_unchanged
                ),
                "zero_incremental_provider_calls_proved": (
                    provider_unchanged and provider_count_delta == 0
                ),
            },
            "hard_error_gate": hard_error_gate,
            "bindings": {
                "freeze": current_invariants.get("freeze"),
                "execution": current_invariants.get("execution"),
                "launch": current_invariants.get("launch"),
                "runtime": current_invariants.get("runtime"),
                "parallel_profile": current_invariants.get("parallel_profile"),
                "rerun_command": current_invariants.get("rerun_command"),
                "canary_geometry": current_invariants.get("canary_geometry"),
                "full_alias": current_invariants.get("full_alias"),
                "audit": current_invariants.get("audit"),
                "analysis": current_invariants.get("analysis"),
                "provider_ledger": current_invariants.get("provider_ledger"),
                "checkpoint_terminal_inventory": current_invariants.get(
                    "checkpoint_terminal_inventory"
                ),
            },
            "model_calls_performed": 0,
        },
        field="receipt_sha256",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("snapshot", "rerun", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--execution-root", type=Path, required=True)
        command.add_argument("--freeze", type=Path, required=True)
        command.add_argument("--freeze-file-sha256", required=True)
        command.add_argument("--parallel-profile", type=Path, required=True)
        command.add_argument("--parallel-profile-file-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
        if name in {"rerun", "verify"}:
            command.add_argument("--snapshot", type=Path, required=True)
            command.add_argument("--snapshot-file-sha256", required=True)
        if name == "verify":
            command.add_argument("--rerun-receipt", type=Path, required=True)
            command.add_argument("--rerun-receipt-file-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        invariants, scoped_roots = _collect_canary_state(
            execution_root=args.execution_root,
            freeze_path=args.freeze,
            freeze_file_sha256=args.freeze_file_sha256,
            parallel_profile_path=args.parallel_profile,
            parallel_profile_file_sha256=(
                args.parallel_profile_file_sha256
            ),
        )
        _require_external_output(args.output, scoped_roots=scoped_roots)
        if os.path.lexists(args.output):
            raise FileExistsError(f"evidence output already exists: {args.output}")
        if args.command == "snapshot":
            payload = _snapshot_payload(invariants)
            hash_field = "snapshot_sha256"
            exit_code = 0
        elif args.command == "rerun":
            snapshot, snapshot_content = _load_snapshot(
                args.snapshot,
                args.snapshot_file_sha256,
            )
            if snapshot.get("invariants") != invariants:
                raise Gate0CanaryEvidenceError(
                    "current complete canary differs from the pre-rerun snapshot"
                )
            frozen_argv = _exact_rerun_argv(
                execution_root=args.execution_root,
                parallel_profile_path=args.parallel_profile,
            )
            started_at_utc = _utc_now_text()
            completed = subprocess.run(
                frozen_argv,
                cwd=REPOSITORY_ROOT.resolve(strict=True),
                capture_output=True,
                check=False,
            )
            finished_at_utc = _utc_now_text()
            payload = _rerun_payload(
                snapshot=snapshot,
                snapshot_file_sha256=sha256_bytes(snapshot_content),
                pre_run_invariants=invariants,
                argv=frozen_argv,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            hash_field = "rerun_receipt_sha256"
            exit_code = 0 if completed.returncode == 0 else 3
        else:
            snapshot, snapshot_content = _load_snapshot(
                args.snapshot,
                args.snapshot_file_sha256,
            )
            rerun_receipt, rerun_content = _load_rerun_receipt(
                args.rerun_receipt,
                args.rerun_receipt_file_sha256,
                snapshot=snapshot,
                snapshot_file_sha256=sha256_bytes(snapshot_content),
                execution_root=args.execution_root,
                parallel_profile_path=args.parallel_profile,
            )
            payload = build_final_receipt(
                snapshot=snapshot,
                snapshot_file_sha256=sha256_bytes(snapshot_content),
                rerun_receipt=rerun_receipt,
                rerun_receipt_file_sha256=sha256_bytes(rerun_content),
                current_invariants=invariants,
            )
            hash_field = "receipt_sha256"
            exit_code = 0 if payload["gate0_canary_passed"] else 3
        atomic_create_file(args.output, canonical_json_bytes(payload))
    except (
        FileExistsError,
        Gate0CanaryEvidenceError,
        InvalidOperation,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"core-gate0-canary-evidence: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "kind": payload["kind"],
                "status": payload["status"],
                hash_field: payload[hash_field],
                "output": str(args.output.absolute()),
                "model_calls_performed": payload.get(
                    "model_calls_performed",
                    "deferred_to_final_verify",
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
