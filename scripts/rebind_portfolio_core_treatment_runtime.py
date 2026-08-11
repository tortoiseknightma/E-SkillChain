"""Rebind one accepted Portfolio treatment chain to the verified Core runtime.

The operation is create-only and performs no model/provider calls.  It keeps a
byte-exact copy of the accepted parent runtime, rebinds only the Bank carrier's
registry commitments, and writes an explicit compatibility lineage receipt.
It does not claim that S1, S2, or S3 ran again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.run_portfolio_shard import (  # noqa: E402
    _ACTIVE_BUDGET_CONTRACT,
    _ACTIVE_EVALUATOR_SOURCE_CONTRACT,
    _ACTIVE_FINAL_RESULT_CONTRACT,
    _ACTIVE_GCS_SOURCE_CONTRACT,
)
from skillchain.evaluation.evaluator_outputs import (  # noqa: E402
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD,
    PORTFOLIO_FAILURE_POLICY_VERSION,
    PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY,
    PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    load_portfolio_launch_package,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_core_runtime_sources import (  # noqa: E402
    load_verified_portfolio_core_runtime_sources,
)
from skillchain.evaluation.portfolio_treatment_io import (  # noqa: E402
    load_verified_portfolio_treatment_runtime,
)
from skillchain.evaluation.portfolio_treatments import (  # noqa: E402
    PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION,
    PORTFOLIO_RUNTIME_COMPATIBILITY_REBIND_POLICY_VERSION,
    PortfolioRuntimeCompatibilityRebind,
    PortfolioRuntimeFileBinding,
    PortfolioRuntimeRebindBankBinding,
    rebind_portfolio_bank_runtime,
    verify_portfolio_runtime_compatibility_rebind_chain,
)
from skillchain.runners.assistant import (  # noqa: E402
    GCS_V2_MODEL_RESPONSE_CONTRACT_SHA256,
    GCS_V2_MODEL_RESPONSE_CONTRACT_VERSION,
    NOSKILL_EXECUTION_CONTRACT_SHA256,
    NOSKILL_EXECUTION_POLICY_VERSION,
    PORTFOLIO_ROUTER_CONTRACT_SHA256,
    PORTFOLIO_ROUTER_CONTRACT_VERSION,
    SHARED_STAGE2_ROUTE_POLICY_VERSION,
    SharedStage2RouteArtifact,
)
from skillchain.synthesis.store import (  # noqa: E402
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_SYSTEM_PROMPT,
    PORTFOLIO_TOOL_RUNTIME_POLICY,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_ORDER = ("llm_static", "s1", "s1s2", "full")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_object(path: Path, label: str) -> tuple[dict, bytes]:
    content = path.read_bytes()
    value = parse_canonical_json(content, label=label)
    if not isinstance(value, dict) or canonical_json_bytes(value) != content:
        raise ValueError(f"{label} must be a canonical object")
    return value, content


def _self_hash(value: dict, field: str, label: str) -> str:
    unsigned = dict(value)
    supplied = unsigned.pop(field, None)
    expected = sha256_bytes(canonical_json_bytes(unsigned))
    if supplied != expected:
        raise ValueError(f"{label} {field} mismatch")
    return expected


def _copy_tree_exact(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"source must be a real directory: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source tree contains a symlink: {path}")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if _file_sha(target) != _file_sha(path):
            raise ValueError(f"copied file digest mismatch: {relative.as_posix()}")


def _parent_file_bindings(root: Path) -> tuple[PortfolioRuntimeFileBinding, ...]:
    return tuple(
        PortfolioRuntimeFileBinding(
            relative_path=path.relative_to(root).as_posix(),
            file_sha256=_file_sha(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _active_code_contract() -> dict[str, object]:
    code_files = {
        "runner_file_sha256": SOURCE_ROOT / "skillchain" / "runners" / "assistant.py",
        "llm_adapter_file_sha256": SOURCE_ROOT / "skillchain" / "llm.py",
        "evaluator_outputs_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "evaluator_outputs.py"
        ),
        "final_runtime_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "final_runtime.py"
        ),
        "shard_runner_file_sha256": REPOSITORY_ROOT
        / "scripts"
        / "run_portfolio_shard.py",
        "portfolio_execution_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_execution.py"
        ),
        "portfolio_treatment_io_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_treatment_io.py"
        ),
        "assistant_runs_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "assistant_runs.py"
        ),
        "portfolio_launch_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_launch.py"
        ),
        "portfolio_treatments_file_sha256": (
            SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_treatments.py"
        ),
        "core_runtime_sources_file_sha256": (
            SOURCE_ROOT
            / "skillchain"
            / "evaluation"
            / "portfolio_core_runtime_sources.py"
        ),
        "core_runtime_rebind_file_sha256": Path(__file__),
    }
    return {
        **{name: _file_sha(path) for name, path in code_files.items()},
        **_ACTIVE_FINAL_RESULT_CONTRACT,
        **_ACTIVE_EVALUATOR_SOURCE_CONTRACT,
        **_ACTIVE_GCS_SOURCE_CONTRACT,
        **_ACTIVE_BUDGET_CONTRACT,
        "final_judge_parser_policy_version": FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
        "final_judge_parser_policy_sha256": FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
        "shared_stage2_route_policy_version": SHARED_STAGE2_ROUTE_POLICY_VERSION,
        "shared_stage2_route_schema_sha256": sha256_bytes(
            canonical_json_bytes(SharedStage2RouteArtifact.model_json_schema())
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
        "noskill_execution_policy_version": NOSKILL_EXECUTION_POLICY_VERSION,
        "noskill_execution_contract_sha256": NOSKILL_EXECUTION_CONTRACT_SHA256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime-root", type=Path, required=True)
    parser.add_argument("--source-runtime-lock-file-sha256", required=True)
    parser.add_argument("--core-launch-root", type=Path, required=True)
    parser.add_argument("--core-launch-plan-file-sha256", required=True)
    parser.add_argument("--core-runtime-sources-root", type=Path, required=True)
    parser.add_argument("--core-runtime-sources-receipt-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    digests = (
        args.source_runtime_lock_file_sha256,
        args.core_launch_plan_file_sha256,
        args.core_runtime_sources_receipt_file_sha256,
    )
    if any(_SHA256.fullmatch(value) is None for value in digests):
        print("rebind-core-runtime: invalid external SHA-256", file=sys.stderr)
        return 2
    if args.output_dir.exists():
        print("rebind-core-runtime: output directory exists", file=sys.stderr)
        return 2

    staging: Path | None = None
    try:
        parent = load_verified_portfolio_treatment_runtime(
            args.source_runtime_root,
            expected_runtime_lock_file_sha256=(args.source_runtime_lock_file_sha256),
            _allow_legacy_missing_execution_aliases=True,
        )
        if parent.chain.compatibility_rebind is not None:
            raise ValueError("source runtime must be an accepted authoring runtime")
        if (
            len(parent.chain.manifest.optimization_query_ids) != 25
            or len(parent.chain.manifest.evaluation_query_ids) != 175
        ):
            raise ValueError(
                "source runtime must preserve the accepted optimization25/evaluation175 lineage"
            )
        parent_lock_sha256 = parent.runtime_lock.get("runtime_lock_sha256")
        if not isinstance(parent_lock_sha256, str):
            raise ValueError("source runtime lock lacks its content identity")

        launch = load_portfolio_launch_package(
            args.core_launch_root,
            expected_plan_file_sha256=args.core_launch_plan_file_sha256,
            _allow_legacy_budget_contract=True,
        )
        core_inputs = reconstruct_verified_portfolio_core_inputs(launch.plan)
        verified_sources = load_verified_portfolio_core_runtime_sources(
            core_inputs,
            output_dir=args.core_runtime_sources_root,
            expected_receipt_file_sha256=(
                args.core_runtime_sources_receipt_file_sha256
            ),
        )
        if verified_sources.provider_call_count != 0:
            raise ValueError("Core runtime-source materialization used a provider")
        core_receipt, core_receipt_bytes = _canonical_object(
            verified_sources.receipt_path,
            "Core runtime-source receipt",
        )
        core_receipt_sha256 = _self_hash(
            core_receipt,
            "receipt_sha256",
            "Core runtime-source receipt",
        )
        runtime = build_portfolio_tool_runtime(verified_sources.sources)

        staging = new_staging_directory(args.output_dir)
        _copy_tree_exact(args.source_runtime_root, staging / "parent-runtime")
        _copy_tree_exact(
            args.core_runtime_sources_root,
            staging / "core-runtime-sources",
        )
        manifest_bytes = (
            args.source_runtime_root / "treatment-chain-manifest.json"
        ).read_bytes()
        (staging / "treatment-chain-manifest.json").write_bytes(manifest_bytes)
        (staging / "system-prompt.txt").write_text(
            PORTFOLIO_SYSTEM_PROMPT,
            encoding="utf-8",
            newline="",
        )

        outputs = {}
        candidates = {}
        bindings: list[PortfolioRuntimeRebindBankBinding] = []
        for config in _CONFIG_ORDER:
            for role, source_map, target_map, relative in (
                (
                    "output",
                    parent.chain.output_banks,
                    outputs,
                    f"bank-{config}.json",
                ),
                (
                    "candidate",
                    parent.chain.candidate_banks,
                    candidates,
                    f"candidates/bank-{config}.json",
                ),
            ):
                source_bank = source_map[config]
                rebound = rebind_portfolio_bank_runtime(
                    source_bank,
                    runtime.registry,
                )
                target = staging.joinpath(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                content = rebound.canonical_bytes()
                target.write_bytes(content)
                target_map[config] = rebound
                bindings.append(
                    PortfolioRuntimeRebindBankBinding(
                        config=config,
                        role=role,
                        source_bank_sha256=source_bank.bank_sha256,
                        source_bank_file_sha256=sha256_bytes(
                            source_bank.canonical_bytes()
                        ),
                        rebound_bank_file=relative,
                        rebound_bank_sha256=rebound.bank_sha256,
                        rebound_bank_file_sha256=sha256_bytes(content),
                    )
                )

        receipt_payload = {
            "schema_version": 1,
            "artifact_kind": "portfolio-runtime-compatibility-rebind",
            "policy_version": (PORTFOLIO_RUNTIME_COMPATIBILITY_REBIND_POLICY_VERSION),
            "source_runtime_dir": "parent-runtime",
            "source_runtime_lock_file_sha256": (args.source_runtime_lock_file_sha256),
            "source_runtime_lock_sha256": parent_lock_sha256,
            "source_treatment_manifest_file_sha256": parent.manifest_file_sha256,
            "source_treatment_chain_sha256": parent.chain.manifest.chain_sha256,
            "source_optimization_query_count": len(
                parent.chain.manifest.optimization_query_ids
            ),
            "source_evaluation_query_count": len(
                parent.chain.manifest.evaluation_query_ids
            ),
            "source_optimization_query_ids_sha256": sha256_bytes(
                canonical_json_bytes(list(parent.chain.manifest.optimization_query_ids))
            ),
            "source_evaluation_query_ids_sha256": sha256_bytes(
                canonical_json_bytes(list(parent.chain.manifest.evaluation_query_ids))
            ),
            "source_runtime_files": [
                item.model_dump(mode="json")
                for item in _parent_file_bindings(staging / "parent-runtime")
            ],
            "core_runtime_sources_dir": "core-runtime-sources",
            "core_runtime_sources_receipt_file_sha256": sha256_bytes(
                core_receipt_bytes
            ),
            "core_runtime_sources_receipt_sha256": core_receipt_sha256,
            "target_tool_registry_sha256": runtime.registry.registry_sha256,
            "target_tool_registry_runtime_sha256": (
                runtime.registry.registry_runtime_sha256
            ),
            "target_runtime_data_sha256": runtime.index.runtime_data_sha256,
            "target_source_sha256s": list(runtime.index.source_sha256s),
            "semantic_fields_preserved": [
                "schema_version",
                "baseline_kind",
                "construction_identity_sha256",
                "construction_identity_policy",
                "runtime_binding_policy",
                "compiler",
                "skills",
                "capability_map",
            ],
            "compatibility_changes": [
                "tool_registry_sha256",
                "tool_registry_runtime_sha256",
                "bank_sha256",
            ],
            "algorithm_decisions_preserved": True,
            "model_evidence_bytes_preserved": True,
            "provider_model_call_count": 0,
            "bank_bindings": [item.model_dump(mode="json") for item in bindings],
        }
        receipt = PortfolioRuntimeCompatibilityRebind.model_validate(
            {
                **receipt_payload,
                "rebind_sha256": sha256_bytes(canonical_json_bytes(receipt_payload)),
            },
            strict=True,
        )
        receipt_bytes = receipt.canonical_bytes()
        (staging / "compatibility-rebind.json").write_bytes(receipt_bytes)
        rebound_chain = verify_portfolio_runtime_compatibility_rebind_chain(
            rebind=receipt,
            source_chain=parent.chain,
            output_banks=outputs,
            candidate_banks=candidates,
        )
        aliases = [
            item.model_dump(mode="json")
            for item in rebound_chain.execution_artifact_aliases
        ]
        # The accepted source decisions and 25/175 lineage remain untouched.
        if [
            item.target_config for item in rebound_chain.execution_artifact_aliases
        ] != ["full"]:
            raise ValueError("accepted source does not yield the expected Full alias")

        tool_bindings = [
            {
                "tool_name": spec.name,
                "tool_spec_sha256": spec.spec_sha256,
                "runtime_binding_sha256": runtime.registry.runtime_binding_sha256(
                    spec.name
                ),
            }
            for spec in runtime.registry.specs()
        ]
        lock_payload = {
            "schema_version": 2,
            "kind": "portfolio-assistant-runtime-lock",
            "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
            "track": "portfolio",
            "formal_eligible": False,
            "formal_ineligible_reason": (
                "core-public-data-runtime-is-portfolio-diagnostic-only"
            ),
            "stage": "core_runtime_compatibility_rebind",
            "zero_authoring_rebind": True,
            "parent_runtime_lock_file_sha256": (args.source_runtime_lock_file_sha256),
            "parent_runtime_lock_sha256": parent_lock_sha256,
            "compatible_frozen_noskill_runtime_lock_sha256": parent_lock_sha256,
            "tool_registry_sha256": runtime.registry.registry_sha256,
            "tool_registry_runtime_sha256": (runtime.registry.registry_runtime_sha256),
            "runtime_data_sha256": runtime.index.runtime_data_sha256,
            "source_sha256s": list(runtime.index.source_sha256s),
            "tool_bindings": tool_bindings,
            "system_prompt_sha256": sha256_bytes(
                PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")
            ),
            "bank_sha256s": {
                config: outputs[config].bank_sha256 for config in _CONFIG_ORDER
            },
            "bank_source_file_sha256s": {
                config: sha256_bytes(outputs[config].canonical_bytes())
                for config in _CONFIG_ORDER
            },
            "bank_policy": parent.runtime_lock["bank_policy"],
            "official_matrix_eligible": True,
            "treatment_chain_status": "ready_for_matrix",
            "treatment_chain_manifest_file": "treatment-chain-manifest.json",
            "treatment_chain_manifest_file_sha256": parent.manifest_file_sha256,
            "treatment_chain_sha256": parent.chain.manifest.chain_sha256,
            "treatment_record_sha256s": {
                item.config: item.receipt_sha256
                for item in parent.chain.manifest.records
            },
            "execution_artifact_alias_policy_version": (
                PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION
            ),
            "execution_artifact_aliases": aliases,
            "execution_artifact_alias_provider_model_call_count": 0,
            "runtime_compatibility_rebind_file": "compatibility-rebind.json",
            "runtime_compatibility_rebind_file_sha256": sha256_bytes(receipt_bytes),
            "runtime_compatibility_rebind_sha256": receipt.rebind_sha256,
            "core_runtime_sources_dir": "core-runtime-sources",
            "core_runtime_sources_receipt_file_sha256": sha256_bytes(
                core_receipt_bytes
            ),
            **_active_code_contract(),
            "model_calls_performed": 0,
        }
        lock = {
            **lock_payload,
            "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(lock_payload)),
        }
        lock_bytes = canonical_json_bytes(lock)
        (staging / "runtime-lock.json").write_bytes(lock_bytes)

        summary = {
            "kind": "portfolio-core-runtime-compatibility-rebind-summary",
            "formal_eligible": False,
            "model_calls_performed": 0,
            "output_dir": str(args.output_dir.absolute()),
            "parent_runtime_lock_file_sha256": (args.source_runtime_lock_file_sha256),
            "parent_treatment_chain_sha256": parent.chain.manifest.chain_sha256,
            "core_launch_plan_file_sha256": args.core_launch_plan_file_sha256,
            "core_runtime_sources_receipt_file_sha256": sha256_bytes(
                core_receipt_bytes
            ),
            "compatibility_rebind_sha256": receipt.rebind_sha256,
            "runtime_lock_sha256": lock["runtime_lock_sha256"],
            "runtime_lock_file_sha256": sha256_bytes(lock_bytes),
            "tool_registry_sha256": runtime.registry.registry_sha256,
            "tool_registry_runtime_sha256": runtime.registry.registry_runtime_sha256,
            "full_execution_alias_source": "s1s2",
        }
        (staging / "summary.json").write_bytes(canonical_json_bytes(summary))
        staged_verified = load_verified_portfolio_treatment_runtime(
            staging,
            expected_runtime_lock_file_sha256=sha256_bytes(lock_bytes),
        )
        if staged_verified.chain.compatibility_rebind != receipt:
            raise ValueError("staged runtime rebind failed strict verification")
        atomic_publish_new_directory(staging, args.output_dir)
        staging = None
        verified = load_verified_portfolio_treatment_runtime(
            args.output_dir,
            expected_runtime_lock_file_sha256=sha256_bytes(lock_bytes),
        )
        if verified.chain.compatibility_rebind != receipt:
            raise ValueError("published runtime rebind failed strict reload")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    except (FileExistsError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"rebind-core-runtime: {error}", file=sys.stderr)
        return 2
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
