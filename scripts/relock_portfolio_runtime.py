"""Relock an accepted Portfolio runtime around the current execution code.

This is a zero-model-call migration.  It preserves every accepted Bank and
evidence byte, rebuilds the public-data tool runtime from the frozen inputs,
and binds the new execution policies without re-authoring any Skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

from pydantic import ValidationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.evaluation.evaluator_outputs import (  # noqa: E402
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
)
from skillchain.evaluation.final_runtime import (  # noqa: E402
    CARD_REQUIREMENT_GUARD_POLICY_SHA256,
    CARD_REQUIREMENT_GUARD_POLICY_VERSION,
    FINAL_JUDGE_CACHE_NAMESPACE,
    FINAL_JUDGE_MAX_ATTEMPTS,
    FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS,
    FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS,
    FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE,
    FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE,
    FINAL_JUDGE_RESULT_SCHEMA_VERSION,
    FINAL_JUDGE_RETRY_POLICY_SHA256,
    FINAL_JUDGE_RETRY_POLICY_VERSION,
    FINAL_JUDGE_THINKING_BUDGET,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    PORTFOLIO_BUDGET_POLICY_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_BUDGET_POLICY_VERSION,
    PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD,
    PORTFOLIO_FAILURE_POLICY_VERSION,
    PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY,
    PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS,
)
from skillchain.evaluation.portfolio_treatment_io import (  # noqa: E402
    load_verified_portfolio_treatment_runtime,
)
from skillchain.runners.assistant import (  # noqa: E402
    NOSKILL_EXECUTION_CONTRACT_SHA256,
    NOSKILL_EXECUTION_POLICY_VERSION,
    PORTFOLIO_ROUTER_CONTRACT_SHA256,
    PORTFOLIO_ROUTER_CONTRACT_VERSION,
    SHARED_STAGE2_ROUTE_POLICY_VERSION,
    SharedStage2RouteArtifact,
)
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.synthesis.store import (  # noqa: E402
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_SYSTEM_PROMPT,
    PORTFOLIO_TOOL_RUNTIME_POLICY,
    PortfolioRuntimeSources,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import (  # noqa: E402
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BANK_NAMES = ("llm_static", "s1", "s1s2", "full")
_REPLACED_FILES = {"runtime-lock.json", "summary.json"}
_TOOL_SOURCE_FILES = {
    "portfolio_tool_runtime_file_sha256": (
        SOURCE_ROOT / "skillchain" / "tools" / "portfolio_runtime.py"
    ),
    "tool_registry_file_sha256": (
        SOURCE_ROOT / "skillchain" / "tools" / "registry.py"
    ),
}


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_tool_source_file_sha256s() -> dict[str, str]:
    """Return the direct live-source identities that relock cannot migrate."""

    return {
        field: _file_sha(path)
        for field, path in _TOOL_SOURCE_FILES.items()
    }


def _require_parent_tool_source_bindings(
    lock: dict,
    *,
    expected: dict[str, object] | None = None,
) -> None:
    if expected is None:
        expected = _current_tool_source_file_sha256s()
    if any(lock.get(field) != digest for field, digest in expected.items()):
        raise ValueError(
            "source runtime lock does not directly bind the current Portfolio "
            "tool runtime and registry sources"
        )


def _active_execution_contract() -> dict[str, object]:
    code_files = {
        "config_file_sha256": SOURCE_ROOT / "skillchain" / "config.py",
        "runner_file_sha256": (
            SOURCE_ROOT / "skillchain" / "runners" / "assistant.py"
        ),
        "llm_adapter_file_sha256": SOURCE_ROOT / "skillchain" / "llm.py",
        "evaluator_outputs_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "evaluator_outputs.py"
        ),
        "final_runtime_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "final_runtime.py"
        ),
        "packets_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "packets.py"
        ),
        "evaluator_isolation_file_sha256": (
            SOURCE_ROOT
            / "skillchain"
            / "evaluation"
            / "evaluator_isolation.py"
        ),
        "shard_runner_file_sha256": (
            REPOSITORY_ROOT / "scripts" / "run_portfolio_shard.py"
        ),
        "portfolio_execution_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_execution.py"
        ),
        **_TOOL_SOURCE_FILES,
    }
    return {
        **{name: _file_sha(path) for name, path in code_files.items()},
        "final_judge_parser_policy_version": (
            FINAL_JUDGE_PARSER_POLICY_VERSION_V4
        ),
        "final_judge_parser_policy_sha256": FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
        "final_judge_result_schema_version": FINAL_JUDGE_RESULT_SCHEMA_VERSION,
        "final_judge_cache_namespace": FINAL_JUDGE_CACHE_NAMESPACE,
        "final_judge_max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
        "final_judge_retry_policy_version": FINAL_JUDGE_RETRY_POLICY_VERSION,
        "final_judge_retry_policy_sha256": FINAL_JUDGE_RETRY_POLICY_SHA256,
        "final_judge_thinking_budget": FINAL_JUDGE_THINKING_BUDGET,
        "final_judge_max_billable_input_tokens": (
            FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
        ),
        "final_judge_max_billable_output_tokens": (
            FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
        ),
        "final_judge_provider_input_token_reserve": (
            FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE
        ),
        "final_judge_provider_output_token_reserve": (
            FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE
        ),
        "portfolio_budget_policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
        "portfolio_budget_policy_sha256": PORTFOLIO_BUDGET_POLICY_SHA256,
        "provider_pricing_contract_version": (
            PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
        ),
        "provider_pricing_contract_sha256": (
            PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
        ),
        "card_requirement_guard_policy_version": (
            CARD_REQUIREMENT_GUARD_POLICY_VERSION
        ),
        "card_requirement_guard_policy_sha256": (
            CARD_REQUIREMENT_GUARD_POLICY_SHA256
        ),
    }


def _canonical_object(path: Path, *, label: str) -> tuple[dict, bytes]:
    content = path.read_bytes()
    value = parse_canonical_json(content, label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value, content


def _validate_self_hash(value: dict, *, field: str, label: str) -> str:
    unsigned = dict(value)
    observed = unsigned.pop(field, None)
    if not isinstance(observed, str) or not _SHA256_RE.fullmatch(observed):
        raise ValueError(f"{label} lacks a valid {field}")
    if observed != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError(f"{label} {field} mismatch")
    return observed


def _validate_recipe_evidence(content: bytes) -> int:
    if not content or not content.endswith(b"\n"):
        raise ValueError("recipe evidence must be non-empty canonical JSONL")
    rows = 0
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line:
            # The accepted v9 evidence uses one visual separator between
            # canonical rows. It is ignored by the runtime loader and must be
            # preserved byte-for-byte by relock.
            continue
        value = parse_canonical_json(
            line + b"\n",
            label=f"recipe evidence line {line_number}",
        )
        if not isinstance(value, dict):
            raise ValueError("recipe evidence rows must be objects")
        if canonical_json_bytes(value) != line + b"\n":
            raise ValueError("recipe evidence contains a non-canonical row")
        rows += 1
    if rows == 0:
        raise ValueError("recipe evidence must contain at least one row")
    return rows


def _runtime_sources(recipe_evidence: Path) -> PortfolioRuntimeSources:
    clean = REPOSITORY_ROOT / "data" / "clean"
    return PortfolioRuntimeSources(
        selection_manifest=clean / "query_images" / "selection-manifest.json",
        dataset_assets=clean / "query_images" / "dataset-assets.jsonl",
        runtime_catalog_assets=(
            clean / "portfolio-mini-asset-catalog-v3" / "assets.jsonl"
        ),
        rpc_scenes=clean / "rpc-multi-product-query-v1" / "scenes.jsonl",
        inaturalist_manifest=(
            clean
            / "portfolio-source-pools"
            / "encyclopedia-inaturalist"
            / "manifest.jsonl"
        ),
        recipe_evidence=recipe_evidence,
    )


def _validate_parent(
    source_root: Path,
    *,
    expected_lock_file_sha256: str,
) -> tuple[dict, str, dict[str, str], int]:
    if not source_root.is_dir():
        raise ValueError("source runtime root must be a directory")
    lock_path = source_root / "runtime-lock.json"
    lock, lock_bytes = _canonical_object(lock_path, label="source runtime lock")
    lock_file_sha256 = sha256_bytes(lock_bytes)
    if lock_file_sha256 != expected_lock_file_sha256:
        raise ValueError("source runtime lock file SHA-256 mismatch")
    lock_content_sha256 = _validate_self_hash(
        lock,
        field="runtime_lock_sha256",
        label="source runtime lock",
    )
    if (
        lock.get("kind") != "portfolio-assistant-runtime-lock"
        or lock.get("track") != "portfolio"
        or lock.get("formal_eligible") is not False
        or lock.get("policy_version") != PORTFOLIO_TOOL_RUNTIME_POLICY
    ):
        raise ValueError("source runtime lock is not the compatible Portfolio lock")
    _require_parent_tool_source_bindings(lock)
    load_verified_portfolio_treatment_runtime(
        source_root,
        expected_runtime_lock_file_sha256=expected_lock_file_sha256,
    )

    system_prompt_path = source_root / "system-prompt.txt"
    try:
        with system_prompt_path.open(
            "r", encoding="utf-8", newline=None
        ) as stream:
            normalized_system_prompt = stream.read()
    except UnicodeDecodeError as error:
        raise ValueError("source system prompt is not UTF-8") from error
    if normalized_system_prompt != PORTFOLIO_SYSTEM_PROMPT:
        raise ValueError("source system prompt differs from the live frozen prompt")
    if lock.get("system_prompt_sha256") != sha256_bytes(
        normalized_system_prompt.encode("utf-8")
    ):
        raise ValueError("source system prompt is not bound by the runtime lock")

    recipe_content = (source_root / "recipe-evidence.jsonl").read_bytes()
    recipe_rows = _validate_recipe_evidence(recipe_content)

    report, _ = _canonical_object(
        source_root / "iteration-report.json",
        label="source iteration report",
    )
    report_sha256 = _validate_self_hash(
        report,
        field="report_sha256",
        label="source iteration report",
    )
    if lock.get("iteration_report_sha256") != report_sha256:
        raise ValueError("source iteration report is not bound by the runtime lock")

    declared_banks = lock.get("bank_sha256s")
    if not isinstance(declared_banks, dict) or set(declared_banks) != set(_BANK_NAMES):
        raise ValueError("source runtime lock must bind exactly four skilled Banks")
    bank_source_file_sha256s: dict[str, str] = {}
    capability_sets: list[tuple[str, ...]] = []
    for name in _BANK_NAMES:
        bank_path = source_root / f"bank-{name}.json"
        bank_content = bank_path.read_bytes()
        try:
            bank = StaticBankArtifact.model_validate_json(bank_content, strict=True)
        except ValidationError as error:
            raise ValueError(f"source {name} Bank is invalid") from error
        if bank.canonical_bytes() != bank_content:
            raise ValueError(f"source {name} Bank bytes are not canonical")
        if bank.bank_sha256 != declared_banks[name]:
            raise ValueError(f"source {name} Bank is not bound by the runtime lock")
        if (
            bank.tool_registry_sha256 != lock.get("tool_registry_sha256")
            or bank.tool_registry_runtime_sha256
            != lock.get("tool_registry_runtime_sha256")
        ):
            raise ValueError(f"source {name} Bank registry binding drifted")
        capability_sets.append(
            tuple(item.capability_id for item in bank.capability_map)
        )
        bank_source_file_sha256s[name] = sha256_bytes(bank_content)
    if len(set(capability_sets)) != 1:
        raise ValueError("source skilled Banks do not share one capability set")

    runtime = build_portfolio_tool_runtime(
        _runtime_sources(source_root / "recipe-evidence.jsonl")
    )
    if (
        runtime.registry.registry_sha256 != lock.get("tool_registry_sha256")
        or runtime.registry.registry_runtime_sha256
        != lock.get("tool_registry_runtime_sha256")
        or runtime.index.runtime_data_sha256 != lock.get("runtime_data_sha256")
        or list(runtime.index.source_sha256s) != lock.get("source_sha256s")
    ):
        raise ValueError("rebuilt Portfolio tool runtime differs from the source lock")
    return lock, lock_content_sha256, bank_source_file_sha256s, recipe_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime-root", type=Path, required=True)
    parser.add_argument(
        "--expected-runtime-lock-file-sha256",
        required=True,
        help="Externally recorded SHA-256 of the source runtime-lock.json bytes.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    expected_lock_file_sha256 = arguments.expected_runtime_lock_file_sha256
    if not _SHA256_RE.fullmatch(expected_lock_file_sha256):
        print(
            "relock-portfolio-runtime: invalid expected lock SHA-256",
            file=sys.stderr,
        )
        return 2
    if arguments.output_dir.exists():
        print("relock-portfolio-runtime: output directory exists", file=sys.stderr)
        return 2

    staging: Path | None = None
    try:
        (
            parent_lock,
            parent_content_sha256,
            bank_source_file_sha256s,
            recipe_rows,
        ) = _validate_parent(
            arguments.source_runtime_root,
            expected_lock_file_sha256=expected_lock_file_sha256,
        )
        staging = new_staging_directory(arguments.output_dir)
        for source in sorted(arguments.source_runtime_root.rglob("*")):
            if not source.is_file():
                continue
            if source.is_symlink():
                raise ValueError(
                    "source runtime contains a symlink: "
                    + source.relative_to(arguments.source_runtime_root).as_posix()
                )
            relative = source.relative_to(arguments.source_runtime_root)
            if relative.as_posix() in _REPLACED_FILES:
                continue
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            if _file_sha(destination) != _file_sha(source):
                raise ValueError(
                    f"preserved artifact copy mismatch: {relative.as_posix()}"
                )

        route_schema_sha256 = sha256_bytes(
            canonical_json_bytes(SharedStage2RouteArtifact.model_json_schema())
        )
        active_execution_contract = _active_execution_contract()
        _require_parent_tool_source_bindings(
            parent_lock,
            expected={
                field: active_execution_contract[field]
                for field in _TOOL_SOURCE_FILES
            },
        )
        lock_payload = {
            "schema_version": 2,
            "kind": "portfolio-assistant-runtime-lock",
            "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
            "track": "portfolio",
            "formal_eligible": False,
            "formal_ineligible_reason": parent_lock["formal_ineligible_reason"],
            "stage": "execution_policy_relock",
            "parent_runtime_lock_file_sha256": expected_lock_file_sha256,
            "parent_runtime_lock_sha256": parent_content_sha256,
            "compatible_frozen_noskill_runtime_lock_sha256": parent_content_sha256,
            "tool_registry_sha256": parent_lock["tool_registry_sha256"],
            "tool_registry_runtime_sha256": parent_lock[
                "tool_registry_runtime_sha256"
            ],
            "runtime_data_sha256": parent_lock["runtime_data_sha256"],
            "source_sha256s": parent_lock["source_sha256s"],
            "system_prompt_sha256": parent_lock["system_prompt_sha256"],
            "recipe_evidence_rows": recipe_rows,
            "iteration_report_sha256": parent_lock["iteration_report_sha256"],
            "bank_sha256s": parent_lock["bank_sha256s"],
            "bank_source_file_sha256s": bank_source_file_sha256s,
            "bank_policy": parent_lock["bank_policy"],
            "official_matrix_eligible": True,
            "treatment_chain_status": "ready_for_matrix",
            "treatment_chain_manifest_file": parent_lock[
                "treatment_chain_manifest_file"
            ],
            "treatment_chain_manifest_file_sha256": parent_lock[
                "treatment_chain_manifest_file_sha256"
            ],
            "treatment_chain_sha256": parent_lock[
                "treatment_chain_sha256"
            ],
            "treatment_record_sha256s": parent_lock[
                "treatment_record_sha256s"
            ],
            "shared_stage2_route_policy_version": (
                SHARED_STAGE2_ROUTE_POLICY_VERSION
            ),
            "shared_stage2_route_schema_sha256": route_schema_sha256,
            "portfolio_router_contract_version": (
                PORTFOLIO_ROUTER_CONTRACT_VERSION
            ),
            "portfolio_router_contract_sha256": PORTFOLIO_ROUTER_CONTRACT_SHA256,
            "portfolio_router_request_max_output_tokens": (
                PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS
            ),
            "portfolio_router_pricing_reservation_max_output_tokens": (
                PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
            ),
            "portfolio_failure_policy_version": PORTFOLIO_FAILURE_POLICY_VERSION,
            "portfolio_circuit_breaker_threshold": (
                PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD
            ),
            "portfolio_max_retryable_attempts_per_query": (
                PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
            ),
            "noskill_execution_policy_version": NOSKILL_EXECUTION_POLICY_VERSION,
            "noskill_execution_contract_sha256": (
                NOSKILL_EXECUTION_CONTRACT_SHA256
            ),
            **active_execution_contract,
            "model_calls_performed": 0,
        }
        lock = {
            **lock_payload,
            "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(lock_payload)),
        }
        (staging / "runtime-lock.json").write_bytes(canonical_json_bytes(lock))
        files = {
            path.relative_to(staging).as_posix(): _file_sha(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        summary = {
            "formal_eligible": False,
            "model_calls_performed": 0,
            "output_dir": str(arguments.output_dir),
            "parent_runtime_lock_file_sha256": expected_lock_file_sha256,
            "parent_runtime_lock_sha256": parent_content_sha256,
            "runtime_lock_sha256": lock["runtime_lock_sha256"],
            "tool_registry_sha256": lock["tool_registry_sha256"],
            "tool_registry_runtime_sha256": lock["tool_registry_runtime_sha256"],
            "files": files,
        }
        (staging / "summary.json").write_bytes(canonical_json_bytes(summary))
        atomic_publish_new_directory(staging, arguments.output_dir)
        staging = None
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    except (
        ArtifactFormatError,
        FileExistsError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"relock-portfolio-runtime: {error}", file=sys.stderr)
        return 2
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
