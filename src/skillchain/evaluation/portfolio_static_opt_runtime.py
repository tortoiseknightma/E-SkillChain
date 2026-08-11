"""Fail-closed runtime package for the Core Static opt800 rollout.

This contract intentionally has no treatment-chain compatibility mode.  A
valid package owns exactly one ``llm_static`` Bank and binds the deterministic
contract-refresh receipt that produced it.  S1/S2/S3 Banks, execution aliases,
and legacy Final-Judge identities are outside this runtime identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path, PurePosixPath
import re
from typing import Literal, Mapping

from skillchain.evaluation.portfolio_gcs import (
    GCS_V2_POLICY_SHA256,
    GCS_V2_POLICY_VERSION,
)
from skillchain.evaluation.portfolio_gcs_evidence import (
    GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION,
)
from skillchain.static_authoring import (
    AuthoringContractError,
    AuthoringInput,
    StaticBankArtifact,
    load_authoring_packet,
)
from skillchain.task_spec import load_mvp_task_specification_v1
from skillchain.tools.portfolio_runtime import (
    PORTFOLIO_SYSTEM_PROMPT,
    PORTFOLIO_TOOL_RUNTIME_POLICY,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


STATIC_OPT_RUNTIME_POLICY_VERSION = "portfolio-static-opt-runtime-v1"
STATIC_CONTRACT_REFRESH_POLICY_VERSION = "portfolio-static-contract-refresh-v1"
STATIC_OPT_RUNTIME_KIND = "portfolio-static-opt-runtime-lock"
STATIC_OPT_BANK_FILE = "bank-llm_static.json"
STATIC_OPT_REFRESH_RECEIPT_FILE = "static-contract-refresh-receipt.json"
STATIC_OPT_CORE_RECEIPT_FILE = "core-runtime-sources/receipt.json"
STATIC_OPT_SYSTEM_PROMPT_FILE = "system-prompt.txt"
STATIC_OPT_SEMANTIC_INPUT_FILE = "semantic-authoring-input.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED_STATIC_OPT_RUNTIME = object()
_ACTIVE_EXECUTION_SCOPE = "active_execution"
_IMMUTABLE_EVIDENCE_SCOPE = "immutable_evidence"

# The completed opt800 corpus is bound to this exact runtime-v8 lock.  Its
# source hashes intentionally describe the code that produced the immutable
# checkpoints; they must not be compared with later live source bytes when the
# package is read only as evidence.  The exact file hash, self hash, source
# tuple, and execution contract are all frozen here.  This is not an execution
# compatibility mode: runners continue to require the active scope below.
HISTORICAL_STATIC_OPT_RUNTIME_V8_FILE_SHA256 = (
    "281fc972806f2b7bd087643c7350fd5f221d2a2df664c00a4ffd6a30f9d89756"
)
_HISTORICAL_STATIC_OPT_RUNTIME_V8_SELF_SHA256 = (
    "17733fcc50f10c1cb3c4bfb1d71c30995ff520f7f3c2ad4246ebe226b86860bf"
)
_HISTORICAL_STATIC_OPT_RUNTIME_V8_SOURCE_HASHES = {
    "runner_file_sha256": (
        "da7cab45f63bbade415e9c2286d125bee4cf661ce9e801045840461571862800"
    ),
    "shard_runner_file_sha256": (
        "b8bf5855af4242735e897a512ba8234479b4f63b0a74efe1e9b8df4a63d2cd4c"
    ),
    "assistant_runs_file_sha256": (
        "1fef25284939b3f288c728d01f3734f07e411409cd97a50fa6300ef193cd83a2"
    ),
    "portfolio_execution_file_sha256": (
        "a1c2c5481b595b112947849379b165c5927ec977304b7ff9b44768942442c664"
    ),
    "portfolio_launch_file_sha256": (
        "3b78a4f83f8c3e6b671326476bffefa137ccb4a88efbb38ba3f24b870ad897a4"
    ),
    "portfolio_parallel_file_sha256": (
        "4c1f6e31959091872d350ebf64eaabcd88d83e6d5718b985c2dbc44c5e23d38a"
    ),
    "portfolio_core_inputs_file_sha256": (
        "d8b1a3e9c4c5f82f5d242daa91f6f4ebc7c6491028aee24d94d033d1d4d18fe1"
    ),
    "portfolio_core_runtime_sources_file_sha256": (
        "44d98d1a4afc23fbfc9f87c936c0509a6e09aff0e93c50597a3ada69491698ef"
    ),
    "portfolio_gcs_file_sha256": (
        "57c17d597201969900edc6a30204be317df222f4b948a5d3363c324451962148"
    ),
    "portfolio_gcs_evidence_file_sha256": (
        "fd912572940529f524eab5120383a523c5f36c197af51a357755e6b055f33f84"
    ),
    "tool_registry_file_sha256": (
        "6ef8e0de07ca7a73053285603f6809759b438afb83a6838183f122e20764a5e6"
    ),
    "portfolio_tool_runtime_file_sha256": (
        "1888190e4e958101b456d660a75735c91a743bcd1f8d8083e12c211c708bb328"
    ),
    "llm_adapter_file_sha256": (
        "7278598bb74932cc981748ffb1d49599041e47f58d82f1849589e5131aeef5e3"
    ),
    "task_spec_file_sha256": (
        "045646e3eb845de4affff3deb73e71e65f9f27212f530b14de9b04aac4f13616"
    ),
    "portfolio_static_opt_runtime_file_sha256": (
        "b24c740e8c35fc9c01344aa3ebacf2bc062bbcee5f618783a430912048e74779"
    ),
}
_HISTORICAL_STATIC_OPT_RUNTIME_V8_EXECUTION_CONTRACT = {
    "portfolio_budget_policy_version": "portfolio-call-hard-cap-v2",
    "portfolio_budget_policy_sha256": (
        "4eb2f3cede57595b0ea7f53d142410aea54eb166d15b929f80516185e78142eb"
    ),
    "provider_pricing_contract_version": "portfolio-provider-pricing-contract-v2",
    "provider_pricing_contract_sha256": (
        "e10b3ede5fc5c52636181ff912b56808bbc494990b57018ef78e488a2611b10c"
    ),
    "portfolio_router_contract_version": "portfolio-assistant-router-v6",
    "portfolio_router_contract_sha256": (
        "dd6b73405e0507605234e3cdf3621c9361bd2105bb9e49f46c5bdf3b81d1b355"
    ),
    "gcs_v2_model_response_contract_version": (
        "portfolio-gcs-v2-model-visible-response-contract-v1"
    ),
    "gcs_v2_model_response_contract_sha256": (
        "5528c28d375dfbd2830e2015f45ec2cf40881a10c766df20870aa1f76a41c882"
    ),
    "portfolio_router_request_max_output_tokens": 64,
    "portfolio_router_pricing_reservation_max_output_tokens": 512,
    "portfolio_failure_policy_version": "portfolio-shard-attempt-v4",
    "portfolio_circuit_breaker_threshold": 2,
    "portfolio_max_retryable_attempts_per_query": 2,
}
_FORBIDDEN_LOCK_FIELDS = frozenset(
    {
        "treatment_chain_manifest_file",
        "treatment_chain_manifest_file_sha256",
        "treatment_chain_sha256",
        "treatment_chain_status",
        "treatment_record_sha256s",
        "runtime_compatibility_rebind_file",
        "runtime_compatibility_rebind_file_sha256",
        "runtime_compatibility_rebind_sha256",
        "execution_artifact_alias_policy_version",
    }
)
_FORBIDDEN_BANK_CONFIGS = frozenset({"s1", "s1s2", "full", "noskill"})
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"


def _active_source_paths() -> dict[str, Path]:
    return {
        "runner_file_sha256": (
            _SOURCE_ROOT / "skillchain" / "runners" / "assistant.py"
        ),
        "shard_runner_file_sha256": (
            _REPOSITORY_ROOT / "scripts" / "run_portfolio_shard.py"
        ),
        "assistant_runs_file_sha256": (
            _SOURCE_ROOT / "skillchain" / "evaluation" / "assistant_runs.py"
        ),
        "portfolio_execution_file_sha256": (
            _SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_execution.py"
        ),
        "portfolio_launch_file_sha256": (
            _SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_launch.py"
        ),
        "portfolio_parallel_file_sha256": (
            _SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_parallel.py"
        ),
        "portfolio_core_inputs_file_sha256": (
            _SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_core_inputs.py"
        ),
        "portfolio_core_runtime_sources_file_sha256": (
            _SOURCE_ROOT
            / "skillchain"
            / "evaluation"
            / "portfolio_core_runtime_sources.py"
        ),
        "portfolio_gcs_file_sha256": (
            _SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_gcs.py"
        ),
        "portfolio_gcs_evidence_file_sha256": (
            _SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_gcs_evidence.py"
        ),
        "tool_registry_file_sha256": (
            _SOURCE_ROOT / "skillchain" / "tools" / "registry.py"
        ),
        "portfolio_tool_runtime_file_sha256": (
            _SOURCE_ROOT / "skillchain" / "tools" / "portfolio_runtime.py"
        ),
        "llm_adapter_file_sha256": _SOURCE_ROOT / "skillchain" / "llm.py",
        "task_spec_file_sha256": (
            _REPOSITORY_ROOT / "specs" / "task_specs" / "ecommerce-task-spec-v1.json"
        ),
        "portfolio_static_opt_runtime_file_sha256": Path(__file__).resolve(),
    }


def _active_source_hashes() -> dict[str, str]:
    return {
        field_name: sha256_bytes(path.read_bytes())
        for field_name, path in _active_source_paths().items()
    }


def _active_execution_contract() -> dict[str, object]:
    # Local imports avoid coupling module import order to the Assistant runner.
    from skillchain.evaluation.portfolio_execution import (
        PORTFOLIO_BUDGET_POLICY_VERSION,
        PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD,
        PORTFOLIO_FAILURE_POLICY_VERSION,
        PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY,
        PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
        PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS,
    )
    from skillchain.evaluation.portfolio_launch import (
        PORTFOLIO_BUDGET_POLICY_SHA256,
        PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256,
        PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION,
    )
    from skillchain.runners.assistant import (
        GCS_V2_MODEL_RESPONSE_CONTRACT_SHA256,
        GCS_V2_MODEL_RESPONSE_CONTRACT_VERSION,
        PORTFOLIO_ROUTER_CONTRACT_SHA256,
        PORTFOLIO_ROUTER_CONTRACT_VERSION,
    )

    return {
        "portfolio_budget_policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
        "portfolio_budget_policy_sha256": PORTFOLIO_BUDGET_POLICY_SHA256,
        "provider_pricing_contract_version": (
            PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
        ),
        "provider_pricing_contract_sha256": (
            PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
        ),
        "portfolio_router_contract_version": PORTFOLIO_ROUTER_CONTRACT_VERSION,
        "portfolio_router_contract_sha256": PORTFOLIO_ROUTER_CONTRACT_SHA256,
        "gcs_v2_model_response_contract_version": (
            GCS_V2_MODEL_RESPONSE_CONTRACT_VERSION
        ),
        "gcs_v2_model_response_contract_sha256": (
            GCS_V2_MODEL_RESPONSE_CONTRACT_SHA256
        ),
        "portfolio_router_request_max_output_tokens": (
            PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS
        ),
        "portfolio_router_pricing_reservation_max_output_tokens": (
            PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
        ),
        "portfolio_failure_policy_version": PORTFOLIO_FAILURE_POLICY_VERSION,
        "portfolio_circuit_breaker_threshold": PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD,
        "portfolio_max_retryable_attempts_per_query": (
            PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
        ),
    }


class PortfolioStaticOptRuntimeError(ValueError):
    """The Static opt runtime is absent, drifted, or mixed with a treatment."""


@dataclass(frozen=True)
class VerifiedPortfolioStaticOptRuntime:
    root: Path
    runtime_lock: Mapping[str, object]
    runtime_lock_file_sha256: str
    bank: StaticBankArtifact
    bank_file_sha256: str
    semantic_authoring_input: AuthoringInput
    semantic_authoring_input_file_sha256: str
    refresh_receipt: Mapping[str, object]
    refresh_receipt_file_sha256: str
    core_source_receipt: Mapping[str, object]
    core_source_receipt_file_sha256: str
    verification_scope: Literal["active_execution", "immutable_evidence"]
    _marker: object = field(repr=False, compare=False, default=None)


def _safe_file(root: Path, relative: str, *, label: str) -> Path:
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != relative
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise PortfolioStaticOptRuntimeError(f"{label} path is unsafe")
    candidate = root.joinpath(*parsed.parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved_parent = candidate.parent.resolve(strict=True)
    except OSError as error:
        raise PortfolioStaticOptRuntimeError(
            f"{label} parent is unavailable"
        ) from error
    if (
        resolved_parent != resolved_root
        and resolved_root not in resolved_parent.parents
    ):
        raise PortfolioStaticOptRuntimeError(f"{label} escapes the runtime root")
    if candidate.is_symlink() or not candidate.is_file():
        raise PortfolioStaticOptRuntimeError(f"{label} must be a regular file")
    return candidate


def _canonical_object(path: Path, *, label: str) -> tuple[bytes, dict]:
    try:
        content = path.read_bytes()
        raw = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortfolioStaticOptRuntimeError(f"{label} cannot be read") from error
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise PortfolioStaticOptRuntimeError(f"{label} is not a canonical object")
    return content, raw


def _self_hash(raw: Mapping[str, object], field_name: str, *, label: str) -> str:
    supplied = raw.get(field_name)
    unsigned = dict(raw)
    unsigned.pop(field_name, None)
    if (
        not isinstance(supplied, str)
        or _SHA256.fullmatch(supplied) is None
        or supplied != sha256_bytes(canonical_json_bytes(unsigned))
    ):
        raise PortfolioStaticOptRuntimeError(f"{label} self hash mismatch")
    return supplied


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PortfolioStaticOptRuntimeError(f"{label} must be SHA-256")
    return value


def _validate_refresh_receipt(
    raw: dict,
    *,
    bank: StaticBankArtifact,
    bank_file_sha256: str,
    semantic_input: AuthoringInput,
    semantic_input_file_sha256: str,
) -> str:
    receipt_sha256 = _self_hash(
        raw,
        "receipt_sha256",
        label="Static contract-refresh receipt",
    )
    unchanged = raw.get("unchanged_skill_sha256s")
    unchanged_rows = raw.get("unchanged_skills")
    non_style_skills = tuple(
        skill
        for skill in bank.skills
        if skill.capability_id != "product.style_recommendation"
    )
    style_skills = tuple(
        skill
        for skill in bank.skills
        if skill.capability_id == "product.style_recommendation"
    )
    expected_unchanged = {skill.slug: skill.skill_sha256 for skill in non_style_skills}
    expected_rows = [
        {
            "canonical_bytes_sha256": sha256_bytes(
                canonical_json_bytes(skill.model_dump(mode="json"))
            ),
            "capability_id": skill.capability_id,
            "new_skill_sha256": skill.skill_sha256,
            "old_skill_sha256": skill.skill_sha256,
            "slug": skill.slug,
        }
        for skill in non_style_skills
    ]
    if (
        raw.get("schema_version") != 1
        or raw.get("kind") != "portfolio-static-contract-refresh-receipt"
        or raw.get("policy_version") != STATIC_CONTRACT_REFRESH_POLICY_VERSION
        or raw.get("status") != "completed_zero_call"
        or raw.get("provider_calls") != 0
        or raw.get("new_bank_sha256") != bank.bank_sha256
        or raw.get("new_bank_file_sha256") != bank_file_sha256
        or raw.get("new_bank_file") != STATIC_OPT_BANK_FILE
        or raw.get("new_semantic_authoring_input_file")
        != STATIC_OPT_SEMANTIC_INPUT_FILE
        or raw.get("new_semantic_authoring_input_file_sha256")
        != semantic_input_file_sha256
        or raw.get("new_semantic_authoring_input_sha256") != semantic_input.input_sha256
        or not isinstance(unchanged, dict)
        or unchanged != expected_unchanged
        or unchanged_rows != expected_rows
        or len(non_style_skills) != 5
        or len(style_skills) != 1
        or raw.get("new_style_skill_sha256") != style_skills[0].skill_sha256
        or raw.get("tool_registry_sha256") != bank.tool_registry_sha256
        or raw.get("tool_registry_runtime_sha256") != bank.tool_registry_runtime_sha256
        or raw.get("task_spec_sha256")
        != semantic_input.task_specification.identity_sha256
        or raw.get("refresh_scope") != "style_skill_only"
        or "old_style_skill_sha256" not in raw
        or "new_style_skill_sha256" not in raw
    ):
        raise PortfolioStaticOptRuntimeError(
            "Static contract-refresh receipt does not prove one Style refresh "
            "and five unchanged Skills"
        )
    old_bank = _sha(raw.get("old_bank_sha256"), label="old Static Bank")
    old_style = _sha(raw.get("old_style_skill_sha256"), label="old Style Skill")
    new_style = _sha(raw.get("new_style_skill_sha256"), label="new Style Skill")
    if old_bank == bank.bank_sha256 or old_style == new_style:
        raise PortfolioStaticOptRuntimeError(
            "Static contract refresh must change the Bank and Style Skill"
        )
    return receipt_sha256


def _validate_core_source_receipt(raw: dict) -> tuple[str, str]:
    receipt_sha256 = _self_hash(
        raw,
        "receipt_sha256",
        label="Core runtime-source receipt",
    )
    runtime_data_sha256 = _sha(
        raw.get("runtime_data_sha256"), label="Core runtime data"
    )
    source_sha256s = raw.get("runtime_source_sha256s")
    if (
        raw.get("provider_call_count") != 0
        or not isinstance(source_sha256s, list)
        or not source_sha256s
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in source_sha256s
        )
        or runtime_data_sha256
        != sha256_bytes(
            canonical_json_bytes(
                {
                    "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
                    "source_sha256s": source_sha256s,
                }
            )
        )
    ):
        raise PortfolioStaticOptRuntimeError(
            "Core runtime-source receipt has invalid zero-call source identity"
        )
    return receipt_sha256, runtime_data_sha256


def _verify_core_source_files(root: Path, receipt: Mapping[str, object]) -> None:
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise PortfolioStaticOptRuntimeError(
            "Core runtime-source receipt has no materialized outputs"
        )
    expected_files = {"receipt.json"}
    for relative, descriptor in outputs.items():
        if not isinstance(relative, str) or not isinstance(descriptor, dict):
            raise PortfolioStaticOptRuntimeError(
                "Core runtime-source output descriptor is invalid"
            )
        path = _safe_file(root, relative, label=f"Core runtime source {relative}")
        content = path.read_bytes()
        if descriptor.get("sha256") != sha256_bytes(content) or descriptor.get(
            "bytes"
        ) != len(content):
            raise PortfolioStaticOptRuntimeError(
                f"Core runtime source differs from receipt: {relative}"
            )
        expected_files.add(relative)
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise PortfolioStaticOptRuntimeError(
            "Core runtime-source directory has missing or extra files"
        )


def _validate_static_root_layout(root: Path, *, include_lock: bool) -> None:
    if root.is_symlink() or not root.is_dir():
        raise PortfolioStaticOptRuntimeError(
            "Static opt runtime root must be a real directory"
        )
    if any(path.is_symlink() for path in root.iterdir()):
        raise PortfolioStaticOptRuntimeError(
            "Static opt runtime root contains a symlink"
        )
    expected_files = {
        STATIC_OPT_BANK_FILE,
        STATIC_OPT_SEMANTIC_INPUT_FILE,
        STATIC_OPT_REFRESH_RECEIPT_FILE,
        STATIC_OPT_SYSTEM_PROMPT_FILE,
        *(("runtime-lock.json",) if include_lock else ()),
    }
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    actual_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    unexpected = sorted(actual_files - expected_files)
    if unexpected:
        raise PortfolioStaticOptRuntimeError(
            "Static opt runtime contains forbidden treatment or unbound artifacts: "
            + ", ".join(unexpected)
        )
    if actual_files != expected_files or actual_dirs != {"core-runtime-sources"}:
        raise PortfolioStaticOptRuntimeError(
            "Static opt runtime root layout is incomplete or unbound"
        )


def _validate_semantic_input(
    semantic_input: AuthoringInput,
    *,
    bank: StaticBankArtifact,
) -> None:
    task_spec = load_mvp_task_specification_v1()
    expected_capabilities = {
        capability.capability_id for capability in task_spec.capabilities
    }
    if (
        semantic_input.task_specification.version != task_spec.task_spec_version
        or semantic_input.task_specification.identity_sha256
        != task_spec.task_spec_sha256
        or semantic_input.tool_registry.identity_sha256 != bank.tool_registry_sha256
        or (
            semantic_input.tool_registry_runtime_sha256 is not None
            and semantic_input.tool_registry_runtime_sha256
            != bank.tool_registry_runtime_sha256
        )
        or {skill.capability_id for skill in bank.skills} != expected_capabilities
    ):
        raise PortfolioStaticOptRuntimeError(
            "semantic AuthoringInput differs from the active TaskSpec or Static Bank"
        )


def _validate_lock_shape(
    lock: dict,
    *,
    expected_execution_contract: Mapping[str, object] | None = None,
) -> None:
    bank_sha256s = lock.get("bank_sha256s")
    bank_files = lock.get("bank_source_file_sha256s")
    if (
        lock.get("schema_version") != 1
        or lock.get("kind") != STATIC_OPT_RUNTIME_KIND
        or lock.get("policy_version") != STATIC_OPT_RUNTIME_POLICY_VERSION
        or lock.get("execution_mode") != "static_opt_rollout"
        or lock.get("track") != "portfolio"
        or lock.get("formal_eligible") is not False
        or lock.get("static_opt_rollout_eligible") is not True
        or lock.get("provider_calls") != 0
        or lock.get("model_calls_performed") != 0
        or lock.get("bank_policy") != STATIC_CONTRACT_REFRESH_POLICY_VERSION
        or lock.get("bank_file") != STATIC_OPT_BANK_FILE
        or lock.get("static_contract_refresh_receipt_file")
        != STATIC_OPT_REFRESH_RECEIPT_FILE
        or lock.get("core_runtime_sources_receipt_file") != STATIC_OPT_CORE_RECEIPT_FILE
        or lock.get("system_prompt_file") != STATIC_OPT_SYSTEM_PROMPT_FILE
        or not isinstance(bank_sha256s, dict)
        or set(bank_sha256s) != {"llm_static"}
        or not isinstance(bank_files, dict)
        or set(bank_files) != {"llm_static"}
        or lock.get("semantic_authoring_input_file") != STATIC_OPT_SEMANTIC_INPUT_FILE
        or lock.get("execution_artifact_aliases") != []
        or lock.get("execution_artifact_alias_provider_model_call_count") != 0
        or any(field in lock for field in _FORBIDDEN_LOCK_FIELDS)
        or any(config in bank_sha256s for config in _FORBIDDEN_BANK_CONFIGS)
        or any(config in bank_files for config in _FORBIDDEN_BANK_CONFIGS)
    ):
        raise PortfolioStaticOptRuntimeError(
            "Static opt runtime mixes a treatment-chain or has invalid scope"
        )
    for digest_field in (
        "bank_file_sha256",
        "bank_sha256",
        "semantic_authoring_input_file_sha256",
        "semantic_authoring_input_sha256",
        "static_contract_refresh_receipt_file_sha256",
        "static_contract_refresh_receipt_sha256",
        "core_runtime_sources_receipt_file_sha256",
        "core_runtime_sources_receipt_sha256",
        "runtime_data_sha256",
        "system_prompt_file_sha256",
        "system_prompt_sha256",
        "tool_registry_sha256",
        "tool_registry_runtime_sha256",
        "gcs_policy_sha256",
        "portfolio_gcs_file_sha256",
        "portfolio_gcs_evidence_file_sha256",
        "task_spec_file_sha256",
        "task_spec_sha256",
        "runner_file_sha256",
        "shard_runner_file_sha256",
        "assistant_runs_file_sha256",
        "portfolio_execution_file_sha256",
        "portfolio_launch_file_sha256",
        "portfolio_parallel_file_sha256",
        "portfolio_core_inputs_file_sha256",
        "portfolio_core_runtime_sources_file_sha256",
        "tool_registry_file_sha256",
        "portfolio_tool_runtime_file_sha256",
        "llm_adapter_file_sha256",
        "portfolio_static_opt_runtime_file_sha256",
    ):
        _sha(lock.get(digest_field), label=f"runtime lock {digest_field}")
    if (
        lock.get("assistant_checkpoint_schema_version") != 2
        or lock.get("gcs_policy_version") != GCS_V2_POLICY_VERSION
        or lock.get("gcs_policy_sha256") != GCS_V2_POLICY_SHA256
        or lock.get("gcs_scorer_evidence_policy_version")
        != GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
        or lock.get("gcs_scorer_evidence_schema_version")
        != GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION
        or lock.get("task_spec_version")
        != load_mvp_task_specification_v1().task_spec_version
        or lock.get("task_spec_sha256")
        != load_mvp_task_specification_v1().task_spec_sha256
    ):
        raise PortfolioStaticOptRuntimeError(
            "Static opt runtime lacks the active GCS v2 or TaskSpec contract"
        )
    execution_contract = (
        _active_execution_contract()
        if expected_execution_contract is None
        else expected_execution_contract
    )
    if any(
        lock.get(field_name) != expected
        for field_name, expected in execution_contract.items()
    ):
        raise PortfolioStaticOptRuntimeError(
            "Static opt runtime lacks the active Router, failure, or budget contract"
        )


def build_portfolio_static_opt_runtime_lock(
    root: str | Path,
    *,
    active_contract: Mapping[str, object],
) -> dict[str, object]:
    """Build (but do not publish) the self-hashed one-Bank runtime lock.

    ``active_contract`` carries runtime-specific Router, budget, tool-binding,
    and permission identities.  Source, Bank, GCS, TaskSpec, and receipt hashes
    are derived here from exact bytes and cannot be overridden by the caller.
    """

    runtime_root = Path(root).absolute()
    _validate_static_root_layout(runtime_root, include_lock=False)
    bank_path = _safe_file(runtime_root, STATIC_OPT_BANK_FILE, label="Static Bank")
    bank_content = bank_path.read_bytes()
    try:
        bank = StaticBankArtifact.model_validate_json(bank_content, strict=True)
    except Exception as error:
        raise PortfolioStaticOptRuntimeError("Static Bank is invalid") from error
    if bank.canonical_bytes() != bank_content:
        raise PortfolioStaticOptRuntimeError("Static Bank is not canonical")
    bank_file_sha256 = sha256_bytes(bank_content)

    semantic_path = _safe_file(
        runtime_root,
        STATIC_OPT_SEMANTIC_INPUT_FILE,
        label="semantic AuthoringInput",
    )
    semantic_content = semantic_path.read_bytes()
    semantic_file_sha256 = sha256_bytes(semantic_content)
    try:
        semantic_input = load_authoring_packet(
            semantic_path,
            expected_file_sha256=semantic_file_sha256,
        )
    except AuthoringContractError as error:
        raise PortfolioStaticOptRuntimeError(
            "semantic AuthoringInput is invalid"
        ) from error
    _validate_semantic_input(semantic_input, bank=bank)

    refresh_path = _safe_file(
        runtime_root,
        STATIC_OPT_REFRESH_RECEIPT_FILE,
        label="Static contract-refresh receipt",
    )
    refresh_content, refresh = _canonical_object(
        refresh_path,
        label="Static contract-refresh receipt",
    )
    refresh_sha256 = _validate_refresh_receipt(
        refresh,
        bank=bank,
        bank_file_sha256=bank_file_sha256,
        semantic_input=semantic_input,
        semantic_input_file_sha256=semantic_file_sha256,
    )

    core_path = _safe_file(
        runtime_root,
        STATIC_OPT_CORE_RECEIPT_FILE,
        label="Core runtime-source receipt",
    )
    core_content, core_receipt = _canonical_object(
        core_path,
        label="Core runtime-source receipt",
    )
    core_receipt_sha256, runtime_data_sha256 = _validate_core_source_receipt(
        core_receipt
    )
    _verify_core_source_files(core_path.parent, core_receipt)
    prompt_path = _safe_file(
        runtime_root,
        STATIC_OPT_SYSTEM_PROMPT_FILE,
        label="system prompt",
    )
    prompt = prompt_path.read_bytes()
    try:
        prompt_text = prompt.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PortfolioStaticOptRuntimeError("system prompt is not UTF-8") from error
    if prompt_text != PORTFOLIO_SYSTEM_PROMPT:
        raise PortfolioStaticOptRuntimeError(
            "system prompt differs from active runtime"
        )

    derived_contract = _active_source_hashes()
    task_spec = load_mvp_task_specification_v1()
    derived_contract.update(
        {
            "assistant_checkpoint_schema_version": 2,
            "gcs_policy_version": GCS_V2_POLICY_VERSION,
            "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
            "gcs_scorer_evidence_policy_version": (
                GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
            ),
            "gcs_scorer_evidence_schema_version": (
                GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION
            ),
            "task_spec_version": task_spec.task_spec_version,
            "task_spec_sha256": task_spec.task_spec_sha256,
        }
    )
    derived_contract.update(_active_execution_contract())
    fixed_keys = {
        "schema_version",
        "kind",
        "policy_version",
        "execution_mode",
        "track",
        "formal_eligible",
        "static_opt_rollout_eligible",
        "provider_calls",
        "model_calls_performed",
        "bank_policy",
        "bank_file",
        "bank_file_sha256",
        "bank_sha256",
        "bank_sha256s",
        "bank_source_file_sha256s",
        "semantic_authoring_input_file",
        "semantic_authoring_input_file_sha256",
        "semantic_authoring_input_sha256",
        "static_contract_refresh_receipt_file",
        "static_contract_refresh_receipt_file_sha256",
        "static_contract_refresh_receipt_sha256",
        "core_runtime_sources_dir",
        "core_runtime_sources_receipt_file",
        "core_runtime_sources_receipt_file_sha256",
        "core_runtime_sources_receipt_sha256",
        "runtime_data_sha256",
        "system_prompt_file",
        "system_prompt_file_sha256",
        "system_prompt_sha256",
        "tool_registry_sha256",
        "tool_registry_runtime_sha256",
        "execution_artifact_aliases",
        "execution_artifact_alias_provider_model_call_count",
        *derived_contract,
        "runtime_lock_sha256",
    }
    if (
        not isinstance(active_contract, Mapping)
        or set(active_contract) & (fixed_keys | _FORBIDDEN_LOCK_FIELDS)
        or any(config in active_contract for config in _FORBIDDEN_BANK_CONFIGS)
    ):
        raise PortfolioStaticOptRuntimeError(
            "active contract overrides the one-Bank Static runtime identity"
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": STATIC_OPT_RUNTIME_KIND,
        "policy_version": STATIC_OPT_RUNTIME_POLICY_VERSION,
        "execution_mode": "static_opt_rollout",
        "track": "portfolio",
        "formal_eligible": False,
        "static_opt_rollout_eligible": True,
        "provider_calls": 0,
        "model_calls_performed": 0,
        "bank_policy": STATIC_CONTRACT_REFRESH_POLICY_VERSION,
        "bank_file": STATIC_OPT_BANK_FILE,
        "bank_file_sha256": bank_file_sha256,
        "bank_sha256": bank.bank_sha256,
        "bank_sha256s": {"llm_static": bank.bank_sha256},
        "bank_source_file_sha256s": {"llm_static": bank_file_sha256},
        "semantic_authoring_input_file": STATIC_OPT_SEMANTIC_INPUT_FILE,
        "semantic_authoring_input_file_sha256": semantic_file_sha256,
        "semantic_authoring_input_sha256": semantic_input.input_sha256,
        "static_contract_refresh_receipt_file": STATIC_OPT_REFRESH_RECEIPT_FILE,
        "static_contract_refresh_receipt_file_sha256": sha256_bytes(refresh_content),
        "static_contract_refresh_receipt_sha256": refresh_sha256,
        "core_runtime_sources_dir": "core-runtime-sources",
        "core_runtime_sources_receipt_file": STATIC_OPT_CORE_RECEIPT_FILE,
        "core_runtime_sources_receipt_file_sha256": sha256_bytes(core_content),
        "core_runtime_sources_receipt_sha256": core_receipt_sha256,
        "runtime_data_sha256": runtime_data_sha256,
        "system_prompt_file": STATIC_OPT_SYSTEM_PROMPT_FILE,
        "system_prompt_file_sha256": sha256_bytes(prompt),
        "system_prompt_sha256": sha256_bytes(prompt),
        "tool_registry_sha256": bank.tool_registry_sha256,
        "tool_registry_runtime_sha256": bank.tool_registry_runtime_sha256,
        "execution_artifact_aliases": [],
        "execution_artifact_alias_provider_model_call_count": 0,
        **derived_contract,
        **dict(active_contract),
    }
    _validate_lock_shape(payload)
    return {
        **payload,
        "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def load_verified_portfolio_static_opt_runtime(
    root: str | Path,
    *,
    expected_runtime_lock_file_sha256: str,
) -> VerifiedPortfolioStaticOptRuntime:
    """Deeply load the one-Bank Static opt runtime without a treatment chain."""

    return _load_verified_portfolio_static_opt_runtime(
        root,
        expected_runtime_lock_file_sha256=expected_runtime_lock_file_sha256,
        verification_scope=_ACTIVE_EXECUTION_SCOPE,
    )


def load_verified_portfolio_static_opt_runtime_evidence(
    root: str | Path,
    *,
    expected_runtime_lock_file_sha256: str,
) -> VerifiedPortfolioStaticOptRuntime:
    """Deeply load an active or the exact immutable runtime-v8 for evidence.

    The historical branch is deliberately restricted to the one published
    opt800 runtime lock.  Objects returned in that scope cannot pass the
    execution capability boundary.
    """

    expected = _sha(
        expected_runtime_lock_file_sha256,
        label="expected Static opt runtime-lock file",
    )
    scope = (
        _IMMUTABLE_EVIDENCE_SCOPE
        if expected == HISTORICAL_STATIC_OPT_RUNTIME_V8_FILE_SHA256
        else _ACTIVE_EXECUTION_SCOPE
    )
    return _load_verified_portfolio_static_opt_runtime(
        root,
        expected_runtime_lock_file_sha256=expected,
        verification_scope=scope,
    )


def _load_verified_portfolio_static_opt_runtime(
    root: str | Path,
    *,
    expected_runtime_lock_file_sha256: str,
    verification_scope: Literal["active_execution", "immutable_evidence"],
) -> VerifiedPortfolioStaticOptRuntime:

    expected_lock_file_sha256 = _sha(
        expected_runtime_lock_file_sha256,
        label="expected Static opt runtime-lock file",
    )
    runtime_root = Path(root).absolute()
    _validate_static_root_layout(runtime_root, include_lock=True)
    lock_path = _safe_file(runtime_root, "runtime-lock.json", label="runtime lock")
    lock_content, lock = _canonical_object(lock_path, label="Static opt runtime lock")
    if sha256_bytes(lock_content) != expected_lock_file_sha256:
        raise PortfolioStaticOptRuntimeError("Static opt runtime-lock file drifted")
    lock_self_sha256 = _self_hash(
        lock, "runtime_lock_sha256", label="Static opt runtime lock"
    )
    immutable_evidence = verification_scope == _IMMUTABLE_EVIDENCE_SCOPE
    expected_execution_contract = (
        _HISTORICAL_STATIC_OPT_RUNTIME_V8_EXECUTION_CONTRACT
        if immutable_evidence
        else _active_execution_contract()
    )
    expected_source_hashes = (
        _HISTORICAL_STATIC_OPT_RUNTIME_V8_SOURCE_HASHES
        if immutable_evidence
        else _active_source_hashes()
    )
    if immutable_evidence and (
        expected_lock_file_sha256 != HISTORICAL_STATIC_OPT_RUNTIME_V8_FILE_SHA256
        or lock_self_sha256 != _HISTORICAL_STATIC_OPT_RUNTIME_V8_SELF_SHA256
    ):
        raise PortfolioStaticOptRuntimeError(
            "immutable Static opt evidence runtime identity is not recognized"
        )
    _validate_lock_shape(
        lock,
        expected_execution_contract=expected_execution_contract,
    )
    if any(
        lock.get(field_name) != expected
        for field_name, expected in expected_source_hashes.items()
    ):
        raise PortfolioStaticOptRuntimeError(
            "Static opt runtime source identity drifted"
        )

    bank_path = _safe_file(runtime_root, STATIC_OPT_BANK_FILE, label="Static Bank")
    bank_content = bank_path.read_bytes()
    bank_file_sha256 = sha256_bytes(bank_content)
    if bank_file_sha256 != lock["bank_file_sha256"]:
        raise PortfolioStaticOptRuntimeError("Static Bank file digest mismatch")
    try:
        bank = StaticBankArtifact.model_validate_json(bank_content, strict=True)
    except Exception as error:
        raise PortfolioStaticOptRuntimeError("Static Bank is invalid") from error
    if (
        bank.canonical_bytes() != bank_content
        or bank.bank_sha256 != lock["bank_sha256"]
        or lock["bank_sha256s"] != {"llm_static": bank.bank_sha256}
        or lock["bank_source_file_sha256s"] != {"llm_static": bank_file_sha256}
        or bank.tool_registry_sha256 != lock["tool_registry_sha256"]
        or bank.tool_registry_runtime_sha256 != lock["tool_registry_runtime_sha256"]
    ):
        raise PortfolioStaticOptRuntimeError("Static Bank differs from runtime lock")

    semantic_path = _safe_file(
        runtime_root,
        STATIC_OPT_SEMANTIC_INPUT_FILE,
        label="semantic AuthoringInput",
    )
    semantic_content = semantic_path.read_bytes()
    semantic_file_sha256 = sha256_bytes(semantic_content)
    if semantic_file_sha256 != lock["semantic_authoring_input_file_sha256"]:
        raise PortfolioStaticOptRuntimeError(
            "semantic AuthoringInput file digest mismatch"
        )
    try:
        semantic_input = load_authoring_packet(
            semantic_path,
            expected_file_sha256=semantic_file_sha256,
        )
    except AuthoringContractError as error:
        raise PortfolioStaticOptRuntimeError(
            "semantic AuthoringInput is invalid"
        ) from error
    _validate_semantic_input(semantic_input, bank=bank)
    if semantic_input.input_sha256 != lock["semantic_authoring_input_sha256"]:
        raise PortfolioStaticOptRuntimeError(
            "semantic AuthoringInput content digest mismatch"
        )

    refresh_path = _safe_file(
        runtime_root,
        STATIC_OPT_REFRESH_RECEIPT_FILE,
        label="Static contract-refresh receipt",
    )
    refresh_content, refresh = _canonical_object(
        refresh_path,
        label="Static contract-refresh receipt",
    )
    refresh_file_sha256 = sha256_bytes(refresh_content)
    refresh_sha256 = _validate_refresh_receipt(
        refresh,
        bank=bank,
        bank_file_sha256=bank_file_sha256,
        semantic_input=semantic_input,
        semantic_input_file_sha256=semantic_file_sha256,
    )
    if (
        refresh_file_sha256 != lock["static_contract_refresh_receipt_file_sha256"]
        or refresh_sha256 != lock["static_contract_refresh_receipt_sha256"]
    ):
        raise PortfolioStaticOptRuntimeError(
            "Static contract-refresh receipt differs from runtime lock"
        )

    core_path = _safe_file(
        runtime_root,
        STATIC_OPT_CORE_RECEIPT_FILE,
        label="Core runtime-source receipt",
    )
    core_content, core_receipt = _canonical_object(
        core_path,
        label="Core runtime-source receipt",
    )
    core_file_sha256 = sha256_bytes(core_content)
    core_receipt_sha256, runtime_data_sha256 = _validate_core_source_receipt(
        core_receipt
    )
    _verify_core_source_files(core_path.parent, core_receipt)
    if (
        core_file_sha256 != lock["core_runtime_sources_receipt_file_sha256"]
        or core_receipt_sha256 != lock["core_runtime_sources_receipt_sha256"]
        or runtime_data_sha256 != lock["runtime_data_sha256"]
    ):
        raise PortfolioStaticOptRuntimeError(
            "Core runtime-source receipt differs from runtime lock"
        )

    system_prompt_path = _safe_file(
        runtime_root,
        STATIC_OPT_SYSTEM_PROMPT_FILE,
        label="system prompt",
    )
    system_prompt = system_prompt_path.read_bytes()
    if (
        sha256_bytes(system_prompt) != lock["system_prompt_file_sha256"]
        or sha256_bytes(system_prompt) != lock["system_prompt_sha256"]
    ):
        raise PortfolioStaticOptRuntimeError("system prompt differs from runtime lock")
    try:
        system_prompt_text = system_prompt.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PortfolioStaticOptRuntimeError("system prompt is not UTF-8") from error
    if system_prompt_text != PORTFOLIO_SYSTEM_PROMPT:
        raise PortfolioStaticOptRuntimeError(
            "system prompt differs from active runtime"
        )

    return VerifiedPortfolioStaticOptRuntime(
        root=runtime_root,
        runtime_lock=lock,
        runtime_lock_file_sha256=expected_lock_file_sha256,
        bank=bank,
        bank_file_sha256=bank_file_sha256,
        semantic_authoring_input=semantic_input,
        semantic_authoring_input_file_sha256=semantic_file_sha256,
        refresh_receipt=refresh,
        refresh_receipt_file_sha256=refresh_file_sha256,
        core_source_receipt=core_receipt,
        core_source_receipt_file_sha256=core_file_sha256,
        verification_scope=verification_scope,
        _marker=_VERIFIED_STATIC_OPT_RUNTIME,
    )


def require_verified_portfolio_static_opt_runtime(
    value: object,
) -> VerifiedPortfolioStaticOptRuntime:
    if (
        type(value) is not VerifiedPortfolioStaticOptRuntime
        or value._marker is not _VERIFIED_STATIC_OPT_RUNTIME
        or value.verification_scope != _ACTIVE_EXECUTION_SCOPE
    ):
        raise TypeError("Static opt execution requires a verified runtime")
    return load_verified_portfolio_static_opt_runtime(
        value.root,
        expected_runtime_lock_file_sha256=value.runtime_lock_file_sha256,
    )


def require_verified_portfolio_static_opt_runtime_evidence(
    value: object,
) -> VerifiedPortfolioStaticOptRuntime:
    if (
        type(value) is not VerifiedPortfolioStaticOptRuntime
        or value._marker is not _VERIFIED_STATIC_OPT_RUNTIME
        or value.verification_scope
        not in {_ACTIVE_EXECUTION_SCOPE, _IMMUTABLE_EVIDENCE_SCOPE}
    ):
        raise TypeError("Static opt evidence requires a verified runtime")
    return load_verified_portfolio_static_opt_runtime_evidence(
        value.root,
        expected_runtime_lock_file_sha256=value.runtime_lock_file_sha256,
    )


def validate_portfolio_static_opt_execution_control(
    control: Mapping[str, object],
    runtime: VerifiedPortfolioStaticOptRuntime,
) -> None:
    """Bind one schema-v2 execution control to the exact one-Bank runtime."""

    verified = require_verified_portfolio_static_opt_runtime(runtime)
    lock = verified.runtime_lock
    expected = {
        "execution_scope": "static_opt_rollout",
        "runtime_lock_file_sha256": verified.runtime_lock_file_sha256,
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "evaluation_stages": ["assistant", "gcs_v2"],
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": GCS_V2_POLICY_VERSION,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "gcs_scorer_evidence_policy_version": (GCS_SCORER_EVIDENCE_V2_POLICY_VERSION),
        "static_bank_file_sha256": verified.bank_file_sha256,
        "static_bank_sha256": verified.bank.bank_sha256,
        "static_contract_refresh_receipt_file_sha256": (
            verified.refresh_receipt_file_sha256
        ),
        "static_contract_refresh_receipt_sha256": verified.refresh_receipt[
            "receipt_sha256"
        ],
        "semantic_authoring_input_file_sha256": (
            verified.semantic_authoring_input_file_sha256
        ),
        "semantic_authoring_input_sha256": (
            verified.semantic_authoring_input.input_sha256
        ),
        "core_runtime_sources_receipt_file_sha256": (
            verified.core_source_receipt_file_sha256
        ),
        "core_runtime_sources_receipt_sha256": verified.core_source_receipt[
            "receipt_sha256"
        ],
        "runtime_data_sha256": lock["runtime_data_sha256"],
        "task_spec_version": lock["task_spec_version"],
        "task_spec_sha256": lock["task_spec_sha256"],
        "task_spec_file_sha256": lock["task_spec_file_sha256"],
        "execution_artifact_aliases": [],
        "execution_artifact_alias_provider_model_call_count": 0,
        "pairwise_judge_enabled": False,
        "legacy_final_judge_enabled": False,
        "analyzer_provider_call_count": 0,
    }
    forbidden = (
        "treatment_chain_manifest_file_sha256",
        "treatment_chain_sha256",
        "treatment_record_sha256s",
        "execution_artifact_alias_policy_version",
        "rubric_path",
        "rubric_file_sha256",
        "rubric_content_sha256",
    )
    if (
        not isinstance(control, Mapping)
        or any(
            control.get(field_name) != value for field_name, value in expected.items()
        )
        or any(field_name in control for field_name in forbidden)
    ):
        raise PortfolioStaticOptRuntimeError(
            "execution control differs from the verified Static opt runtime"
        )


def validate_portfolio_static_opt_evidence_control(
    control: Mapping[str, object],
    runtime: VerifiedPortfolioStaticOptRuntime,
) -> None:
    """Validate a completed control without granting Assistant execution."""

    verified = require_verified_portfolio_static_opt_runtime_evidence(runtime)
    _validate_portfolio_static_opt_control_binding(control, verified)


def _validate_portfolio_static_opt_control_binding(
    control: Mapping[str, object],
    verified: VerifiedPortfolioStaticOptRuntime,
) -> None:
    lock = verified.runtime_lock
    expected = {
        "execution_scope": "static_opt_rollout",
        "runtime_lock_file_sha256": verified.runtime_lock_file_sha256,
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "evaluation_stages": ["assistant", "gcs_v2"],
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": GCS_V2_POLICY_VERSION,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "gcs_scorer_evidence_policy_version": (GCS_SCORER_EVIDENCE_V2_POLICY_VERSION),
        "static_bank_file_sha256": verified.bank_file_sha256,
        "static_bank_sha256": verified.bank.bank_sha256,
        "static_contract_refresh_receipt_file_sha256": (
            verified.refresh_receipt_file_sha256
        ),
        "static_contract_refresh_receipt_sha256": verified.refresh_receipt[
            "receipt_sha256"
        ],
        "semantic_authoring_input_file_sha256": (
            verified.semantic_authoring_input_file_sha256
        ),
        "semantic_authoring_input_sha256": (
            verified.semantic_authoring_input.input_sha256
        ),
        "core_runtime_sources_receipt_file_sha256": (
            verified.core_source_receipt_file_sha256
        ),
        "core_runtime_sources_receipt_sha256": verified.core_source_receipt[
            "receipt_sha256"
        ],
        "runtime_data_sha256": lock["runtime_data_sha256"],
        "task_spec_version": lock["task_spec_version"],
        "task_spec_sha256": lock["task_spec_sha256"],
        "task_spec_file_sha256": lock["task_spec_file_sha256"],
        "execution_artifact_aliases": [],
        "execution_artifact_alias_provider_model_call_count": 0,
        "pairwise_judge_enabled": False,
        "legacy_final_judge_enabled": False,
        "analyzer_provider_call_count": 0,
    }
    forbidden = (
        "treatment_chain_manifest_file_sha256",
        "treatment_chain_sha256",
        "treatment_record_sha256s",
        "execution_artifact_alias_policy_version",
        "rubric_path",
        "rubric_file_sha256",
        "rubric_content_sha256",
    )
    if (
        not isinstance(control, Mapping)
        or any(
            control.get(field_name) != value for field_name, value in expected.items()
        )
        or any(field_name in control for field_name in forbidden)
    ):
        raise PortfolioStaticOptRuntimeError(
            "execution control differs from the verified Static opt runtime"
        )


__all__ = [
    "PortfolioStaticOptRuntimeError",
    "HISTORICAL_STATIC_OPT_RUNTIME_V8_FILE_SHA256",
    "STATIC_CONTRACT_REFRESH_POLICY_VERSION",
    "STATIC_OPT_BANK_FILE",
    "STATIC_OPT_CORE_RECEIPT_FILE",
    "STATIC_OPT_REFRESH_RECEIPT_FILE",
    "STATIC_OPT_SEMANTIC_INPUT_FILE",
    "STATIC_OPT_RUNTIME_KIND",
    "STATIC_OPT_RUNTIME_POLICY_VERSION",
    "STATIC_OPT_SYSTEM_PROMPT_FILE",
    "VerifiedPortfolioStaticOptRuntime",
    "build_portfolio_static_opt_runtime_lock",
    "load_verified_portfolio_static_opt_runtime",
    "load_verified_portfolio_static_opt_runtime_evidence",
    "require_verified_portfolio_static_opt_runtime",
    "require_verified_portfolio_static_opt_runtime_evidence",
    "validate_portfolio_static_opt_evidence_control",
    "validate_portfolio_static_opt_execution_control",
]
