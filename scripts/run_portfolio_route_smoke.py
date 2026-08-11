"""Run the first 25 Portfolio queries through the repaired Stage-2 router only.

This smoke invokes no action tools and no Judge.  It uses the same frozen
Qwen backbone, public query images, S1+S2 descriptions, and hard-budget ledger
as the matrix runner.  Provider-response failures are retried at most once;
every returned route is persisted without raw model text or arguments.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.prepare_portfolio_launch import _load_active_inputs  # noqa: E402
from skillchain import config  # noqa: E402
from skillchain.data.asset_catalog import load_asset_catalog  # noqa: E402
from skillchain.evaluation.assistant_runs import (  # noqa: E402
    AssistantRequestSnapshot,
    MatrixTreatment,
    _registry_lock,
    make_backbone_lock,
    make_inference_budget,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_QWEN_MODEL,
    PortfolioBudgetExceededError,
    initialize_portfolio_budget_ledger,
    load_portfolio_budget_ledger,
)
from skillchain.runners.assistant import (  # noqa: E402
    AssistantProviderPreResponseError,
    PortfolioAssistantBudgetContext,
    PortfolioAssistantRunner,
    require_portfolio_assistant_runner,
)
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.synthesis.store import (  # noqa: E402
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_SYSTEM_PROMPT,
    PortfolioRuntimeSources,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes  # noqa: E402


_SMOKE_SHARD_ID = "route-contract-smoke-001"
_MAX_PROVIDER_ATTEMPTS = 2


def _hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _model(value: object) -> object:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _request(
    *,
    matrix_run_id: str,
    query_ordinal: int,
    query,
    treatment: MatrixTreatment,
    backbone,
    budget,
    registry_lock,
) -> AssistantRequestSnapshot:
    unsigned = {
        "schema_version": 1,
        "matrix_run_id": matrix_run_id,
        "config": "s1s2",
        "query_ordinal": query_ordinal,
        "query": query,
        "treatment": treatment,
        "backbone": backbone,
        "budget": budget,
        "registry": registry_lock,
    }
    return AssistantRequestSnapshot.model_validate(
        {
            **unsigned,
            "request_sha256": _hash(
                {key: _model(value) for key, value in unsigned.items()}
            ),
        },
        strict=True,
    )


def _runtime_sources(runtime_root: Path) -> PortfolioRuntimeSources:
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
        recipe_evidence=runtime_root / "recipe-evidence.jsonl",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-lock-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--phase-cap-cny", type=Decimal, required=True)
    parser.add_argument(
        "--inter-call-delay-seconds",
        type=float,
        default=2.0,
        help="Bounded throttle before each provider attempt (0..10 seconds).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        not args.matrix_run_id
        or args.matrix_run_id != args.matrix_run_id.strip()
        or args.phase_cap_cny <= 0
        or not 0 <= args.inter_call_delay_seconds <= 10
    ):
        print("portfolio-route-smoke: invalid matrix id or phase cap", file=sys.stderr)
        return 2

    lock_path = args.runtime_root / "runtime-lock.json"
    lock_bytes = lock_path.read_bytes()
    if sha256_bytes(lock_bytes) != args.runtime_lock_file_sha256:
        print("portfolio-route-smoke: runtime lock digest mismatch", file=sys.stderr)
        return 2
    runtime_lock = json.loads(lock_bytes)
    runtime = build_portfolio_tool_runtime(_runtime_sources(args.runtime_root))
    banks = {
        name: StaticBankArtifact.model_validate_json(
            (args.runtime_root / f"bank-{name}.json").read_bytes(),
            strict=True,
        )
        for name in ("llm_static", "s1", "s1s2", "full")
    }
    clean = REPOSITORY_ROOT / "data" / "clean"
    catalog = load_asset_catalog(
        clean / "portfolio-mini-asset-catalog-v3",
        clean,
        verify_files=True,
    )
    runner = require_portfolio_assistant_runner(
        PortfolioAssistantRunner(
            registry=runtime.registry,
            system_prompt=PORTFOLIO_SYSTEM_PROMPT,
            banks=banks,
            asset_catalog=catalog,
            runtime_lock=runtime_lock,
            runtime_lock_file_sha256=args.runtime_lock_file_sha256,
        )
    )

    inputs = _load_active_inputs()
    planned = tuple(item for item in inputs.plan.queries if item.batch_id == "dev-mini-001")
    if len(planned) != 25:
        raise ValueError("first frozen Portfolio batch must contain exactly 25 queries")
    queries_by_id = {item.query_id: item for item in inputs.assistant_queries}
    expected_by_id = {item.plan_id: item.canonical_capability for item in planned}
    query_ids = tuple(item.plan_id for item in planned)
    if tuple(sorted(query_ids)) != tuple(f"dm-{index:03d}" for index in range(1, 26)):
        raise ValueError("route smoke query identity differs from dm-001..dm-025")

    backbone = make_backbone_lock(
        provider="qwen",
        model=PORTFOLIO_QWEN_MODEL,
        endpoint=config.PROVIDER_ENDPOINTS["qwen"],
        temperature=0.0,
        top_p=1.0,
        seed=None,
        system_prompt_sha256=sha256_bytes(
            PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")
        ),
    )
    budget = make_inference_budget(
        max_input_tokens=32_768,
        max_output_tokens=4_096,
        max_tool_calls=3,
        max_turns=5,
        timeout_ms=180_000,
    )
    treatment = MatrixTreatment(
        config="s1s2",
        bank_sha256=banks["s1s2"].bank_sha256,
        router_stage="s2",
        body_stage="s1",
    )
    registry_lock = _registry_lock(runtime.registry)

    staging = new_staging_directory(args.output_dir)
    artifact_root = staging / "routes"
    artifact_root.mkdir()
    ledger_root = staging / "budget-ledger"
    initialize_portfolio_budget_ledger(
        ledger_root,
        matrix_run_id=args.matrix_run_id,
        phase_cap_cny=args.phase_cap_cny,
    )

    rows: list[dict[str, object]] = []
    for ordinal, query_id in enumerate(query_ids):
        query = queries_by_id[query_id]
        request = _request(
            matrix_run_id=args.matrix_run_id,
            query_ordinal=ordinal,
            query=query,
            treatment=treatment,
            backbone=backbone,
            budget=budget,
            registry_lock=registry_lock,
        )
        artifact = None
        provider_failures: list[dict[str, str]] = []
        terminal_exception: str | None = None
        attempt_count = 0
        for attempt_index in range(1, _MAX_PROVIDER_ATTEMPTS + 1):
            attempt_count = attempt_index
            context = PortfolioAssistantBudgetContext(
                ledger_root=ledger_root,
                shard_id=_SMOKE_SHARD_ID,
                instance_sha256=query.query_sha256,
                attempt_index=attempt_index,
            )
            if args.inter_call_delay_seconds:
                time.sleep(args.inter_call_delay_seconds)
            try:
                artifact = runner.prepare_shared_stage2_route(
                    request,
                    budget_context=context,
                )
            except AssistantProviderPreResponseError as error:
                provider_failures.append(
                    {
                        "failure_stage": error.failure_stage,
                        "exception_type": type(error.__cause__).__name__,
                    }
                )
                continue
            except PortfolioBudgetExceededError:
                terminal_exception = "PortfolioBudgetExceededError"
                break
            except Exception as error:  # fail closed without storing messages
                terminal_exception = type(error).__name__
                break
            else:
                break

        selected = None if artifact is None else artifact.selected_capability
        row = {
            "schema_version": 1,
            "kind": "portfolio-route-contract-smoke-row",
            "query_id": query_id,
            "query_ordinal": ordinal,
            "expected_capability": expected_by_id[query_id],
            "selected_capability": selected,
            "semantic_match": selected == expected_by_id[query_id],
            "provider_attempt_count": attempt_count,
            "provider_pre_response_failures": list(provider_failures),
            "terminal_exception": terminal_exception,
            "route_artifact": None if artifact is None else artifact.model_dump(mode="json"),
        }
        row["row_sha256"] = _hash(row)
        rows.append(row)
        atomic_create_file(
            artifact_root / f"{ordinal:02d}-{query_id}.json",
            canonical_json_bytes(row),
        )
        print(
            json.dumps(
                {
                    "completed": ordinal + 1,
                    "query_id": query_id,
                    "status": None if artifact is None else artifact.status,
                    "selected_capability": selected,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    ledger = load_portfolio_budget_ledger(ledger_root)
    route_error_ids = [
        row["query_id"]
        for row in rows
        if isinstance(row["route_artifact"], dict)
        and row["route_artifact"].get("status") == "terminal_route_error"
    ]
    missing_ids = [
        row["query_id"] for row in rows if row["route_artifact"] is None
    ]
    mismatch_ids = [
        row["query_id"]
        for row in rows
        if row["route_artifact"] is not None and not row["semantic_match"]
    ]
    smoke_passed = (
        len(rows) == 25
        and not route_error_ids
        and not missing_ids
        and not ledger.unresolved_reservations
    )
    summary = {
        "schema_version": 1,
        "kind": "portfolio-route-contract-smoke-summary",
        "matrix_run_id": args.matrix_run_id,
        "runtime_lock_file_sha256": args.runtime_lock_file_sha256,
        "s1s2_bank_sha256": banks["s1s2"].bank_sha256,
        "query_count": len(rows),
        "route_selected_count": sum(
            1
            for row in rows
            if isinstance(row["route_artifact"], dict)
            and row["route_artifact"].get("status") == "selected"
        ),
        "route_contract_error_count": len(route_error_ids),
        "route_contract_error_ids": route_error_ids,
        "missing_route_count": len(missing_ids),
        "missing_route_ids": missing_ids,
        "semantic_mismatch_count": len(mismatch_ids),
        "semantic_mismatch_ids": mismatch_ids,
        "provider_pre_response_failure_count": sum(
            len(row["provider_pre_response_failures"]) for row in rows
        ),
        "phase_cap_cny": format(args.phase_cap_cny, ".12f"),
        "settled_actual_cost_cny": format(
            ledger.settled_actual_cost_cny, ".12f"
        ),
        "unresolved_reserved_cost_cny": format(
            ledger.unresolved_reserved_cost_cny, ".12f"
        ),
        "accountable_cost_cny": format(ledger.accountable_cost_cny, ".12f"),
        "smoke_passed": smoke_passed,
    }
    summary["summary_sha256"] = _hash(summary)
    atomic_create_file(
        staging / "results.jsonl",
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows),
    )
    atomic_create_file(staging / "summary.json", canonical_json_bytes(summary))
    atomic_publish_new_directory(staging, args.output_dir)
    print(canonical_json_bytes(summary).decode("utf-8"), flush=True)
    return 0 if smoke_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
