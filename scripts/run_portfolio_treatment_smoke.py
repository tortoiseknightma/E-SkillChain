"""Run the frozen optimization25 smoke for one real treatment Bank.

The default mode is a zero-call dry run.  ``--execute`` is the only path that
constructs the live Portfolio runner and calls the Qwen Assistant or active
final Judge.  Outputs are diagnostic-only: this script never creates a
matrix-ready treatment-chain manifest and never upgrades an unverified or
scaffold Bank into an official Creator/Optimizer/Refiner claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import secrets
import shutil
import sys
import tempfile
import time
from typing import Any, Literal, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.prepare_portfolio_launch import _load_active_inputs  # noqa: E402
from skillchain import config  # noqa: E402
from skillchain import llm as llm_module  # noqa: E402
from skillchain.data.asset_catalog import load_asset_catalog  # noqa: E402
from skillchain.evaluation.assistant_runs import (  # noqa: E402
    AssistantRequestSnapshot,
    BackboneLock,
    InferenceBudget,
    MatrixTreatment,
    _registry_lock,
)
from skillchain.evaluation.evaluator_isolation import (  # noqa: E402
    make_active_portfolio_evaluator_isolation_lock,
)
from skillchain.evaluation.final_runtime import (  # noqa: E402
    FinalJudgeBudgetContext,
    run_visual_final_judge,
)
from skillchain.evaluation.packets import (  # noqa: E402
    AssistantResult,
    HiddenEvaluationIdentityError,
    RubricSnapshot,
    build_final_evaluation_packet,
)
from skillchain.evaluation.portfolio_treatments import (  # noqa: E402
    PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
    PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION,
    calculate_portfolio_skill_adherence,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
    PortfolioBudgetExceededError,
    initialize_portfolio_budget_ledger,
    open_portfolio_budget_ledger_session,
)
from skillchain.runners import assistant as assistant_module  # noqa: E402
from skillchain.runners.assistant import (  # noqa: E402
    PortfolioAssistantBudgetContext,
    PortfolioAssistantRunner,
    SharedStage2RouteArtifact,
    require_portfolio_assistant_runner,
)
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.synthesis.store import atomic_create_file  # noqa: E402
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_SYSTEM_PROMPT,
    PortfolioRuntimeSources,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
)


TreatmentConfig = Literal["llm_static", "s1", "s1s2", "full"]

DEFAULT_QUERY_IDS = tuple(f"dm-{index:03d}" for index in range(1, 26))
REQUIRED_SMOKE_CAPABILITY_COUNTS = {
    "knowledge.visual_encyclopedia": 4,
    "product.exact_match": 5,
    "product.multi_search": 5,
    "product.style_recommendation": 4,
    "utility.document_reading": 4,
    "utility.recipe_guidance": 3,
}
REQUIRED_SMOKE_CAPABILITIES = frozenset(REQUIRED_SMOKE_CAPABILITY_COUNTS)


def _emit_progress(payload: Mapping[str, object]) -> None:
    """Best-effort progress logging must never invalidate completed model work."""

    try:
        print(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    except OSError:
        # A detached caller can close stdout while this create-only run remains
        # active. Canonical artifacts, not the console, are the experiment record.
        return


RUBRIC_PATH = (
    REPOSITORY_ROOT / "specs" / "evaluation" / "portfolio-final-rubric-v1.json"
)
QWEN_INPUT_CNY_PER_MILLION = 0.15
QWEN_OUTPUT_CNY_PER_MILLION = 1.5
KIMI_INPUT_CNY_PER_MILLION = 6.5
KIMI_OUTPUT_CNY_PER_MILLION = 27.0
SMOKE_POLICY_VERSION = PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION


@dataclass(frozen=True)
class PreparedTreatmentSmoke:
    config: TreatmentConfig
    query_ids: tuple[str, ...]
    target_bank: StaticBankArtifact
    parent_bank: StaticBankArtifact | None
    banks: Mapping[str, StaticBankArtifact]
    target_bank_file_sha256: str
    parent_bank_file_sha256: str | None
    runtime: Any
    catalog: Any
    runner: PortfolioAssistantRunner
    runtime_lock: Mapping[str, object]
    runtime_lock_file_sha256: str
    public_by_id: Mapping[str, Any]
    private_by_id: Mapping[str, Any]
    query_artifact_sha256: str
    plan_sha256: str
    judge_runtime: Any
    rubric: RubricSnapshot
    rubric_file_sha256: str


def _hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _model(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json")


def _self_model(model_type, payload: dict[str, object], field: str):
    return model_type.model_validate(
        {**payload, field: _hash(payload)},
        strict=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=("llm_static", "s1", "s1s2", "full"),
        required=True,
    )
    parser.add_argument("--target-bank", type=Path, required=True)
    parser.add_argument("--parent-bank", type=Path)
    parser.add_argument("--runtime-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--paired-route-source-root",
        type=Path,
        help=(
            "Required for Full execution: reuse the exact shared-routes files "
            "from its S1+S2 parent smoke."
        ),
    )
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        metavar="QUERY_ID",
        help=(
            "Exact query to run; repeat for frozen optimization25. Defaults to "
            + ",".join(DEFAULT_QUERY_IDS)
            + "."
        ),
    )
    parser.add_argument(
        "--prior-cost-cny",
        type=float,
        default=0.0,
        help="Observed DashScope cost carried into this smoke phase.",
    )
    parser.add_argument(
        "--phase-cap-cny",
        type=float,
        required=True,
        help="Cumulative prior-plus-smoke DashScope cap.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform real Qwen Assistant and Kimi final-Judge calls.",
    )
    return parser


def _exact_query_ids(values: list[str] | None) -> tuple[str, ...]:
    query_ids = tuple(values) if values is not None else DEFAULT_QUERY_IDS
    if query_ids != DEFAULT_QUERY_IDS:
        raise ValueError(
            "treatment smoke requires exact frozen dm-001..dm-025 query order"
        )
    return query_ids


def _validate_cost_authority(prior_cost_cny: float, phase_cap_cny: float) -> None:
    if prior_cost_cny < 0 or phase_cap_cny <= 0 or prior_cost_cny >= phase_cap_cny:
        raise ValueError("cost authority requires 0 <= prior cost < positive phase cap")


def _development_bank_mapping(
    config_name: TreatmentConfig,
    target_bank: Any,
    parent_bank: Any | None,
) -> dict[str, Any]:
    """Project one target and optional direct parent into the runner's four slots."""

    parent = target_bank if parent_bank is None else parent_bank
    if config_name == "llm_static":
        return {name: target_bank for name in ("llm_static", "s1", "s1s2", "full")}
    if config_name == "s1":
        return {
            "llm_static": parent,
            "s1": target_bank,
            "s1s2": target_bank,
            "full": target_bank,
        }
    if config_name == "s1s2":
        return {
            "llm_static": parent,
            "s1": parent,
            "s1s2": target_bank,
            "full": target_bank,
        }
    if config_name == "full":
        return {
            "llm_static": parent,
            "s1": parent,
            "s1s2": parent,
            "full": target_bank,
        }
    raise ValueError(f"unknown treatment config: {config_name}")


def _validate_query_capability_coverage(
    private_by_id: Mapping[str, Any],
    query_ids: tuple[str, ...],
) -> tuple[str, ...]:
    missing = tuple(query_id for query_id in query_ids if query_id not in private_by_id)
    if missing:
        raise ValueError("unknown smoke query IDs: " + ",".join(missing))
    capabilities = tuple(
        private_by_id[query_id].canonical_capability for query_id in query_ids
    )
    if dict(Counter(capabilities)) != REQUIRED_SMOKE_CAPABILITY_COUNTS:
        raise ValueError("smoke capability counts differ from frozen optimization25")
    return capabilities


def calculate_smoke_cost(
    *,
    assistant_input_tokens: int,
    assistant_output_tokens: int,
    shared_route_input_tokens: int,
    shared_route_output_tokens: int,
    judge_input_tokens: int,
    judge_output_tokens: int,
    prior_cost_cny: float = 0.0,
    phase_cap_cny: float | None = None,
) -> dict[str, object]:
    token_values = (
        assistant_input_tokens,
        assistant_output_tokens,
        shared_route_input_tokens,
        shared_route_output_tokens,
        judge_input_tokens,
        judge_output_tokens,
    )
    if any(type(value) is not int or value < 0 for value in token_values):
        raise ValueError("smoke token counts must be nonnegative integers")
    if prior_cost_cny < 0 or (phase_cap_cny is not None and phase_cap_cny <= 0):
        raise ValueError("smoke cost inputs are invalid")
    qwen_input = assistant_input_tokens + shared_route_input_tokens
    qwen_output = assistant_output_tokens + shared_route_output_tokens
    qwen_cost = (
        qwen_input * QWEN_INPUT_CNY_PER_MILLION
        + qwen_output * QWEN_OUTPUT_CNY_PER_MILLION
    ) / 1_000_000
    kimi_cost = (
        judge_input_tokens * KIMI_INPUT_CNY_PER_MILLION
        + judge_output_tokens * KIMI_OUTPUT_CNY_PER_MILLION
    ) / 1_000_000
    observed = qwen_cost + kimi_cost
    cumulative = prior_cost_cny + observed
    return {
        "assistant_usage": {
            "input_tokens": assistant_input_tokens,
            "output_tokens": assistant_output_tokens,
        },
        "shared_route_usage": {
            "input_tokens": shared_route_input_tokens,
            "output_tokens": shared_route_output_tokens,
        },
        "judge_usage": {
            "input_tokens": judge_input_tokens,
            "output_tokens": judge_output_tokens,
        },
        "qwen_cost_cny": round(qwen_cost, 12),
        "kimi_cost_cny": round(kimi_cost, 12),
        "observed_cost_cny": round(observed, 12),
        "prior_cost_cny": prior_cost_cny,
        "cumulative_cost_cny": round(cumulative, 12),
        "phase_cap_cny": phase_cap_cny,
        "cap_reached": (phase_cap_cny is not None and cumulative >= phase_cap_cny),
    }


def _accumulate_judge_accounting(
    totals: dict[str, int],
    judge: Any,
) -> None:
    """Account for both captured responses when the final Judge retries once."""

    usage = judge.aggregate_usage
    totals["judge_model_calls"] += judge.attempts
    totals["judge_captured_response_count"] += judge.captured_response_count
    totals["judge_input_tokens"] += usage.input_tokens
    totals["judge_output_tokens"] += usage.output_tokens


def _load_bank(path: Path, label: str) -> tuple[StaticBankArtifact, str]:
    content = read_stable_regular_file(path, label=label, max_bytes=4 * 1024 * 1024)
    bank = StaticBankArtifact.model_validate_json(content, strict=True)
    if bank.canonical_bytes() != content:
        raise ValueError(f"{label} is not canonical")
    return bank, sha256_bytes(content)


def _development_runtime_lock(
    *,
    registry: Any,
    banks: Mapping[str, StaticBankArtifact],
    target_config: TreatmentConfig,
) -> tuple[dict[str, object], str]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "portfolio-treatment-smoke-runtime-lock",
        "policy_version": SMOKE_POLICY_VERSION,
        "status": "development_only",
        "track": "portfolio",
        "formal_eligible": False,
        "matrix_ready": False,
        "official_treatment_claim_allowed": False,
        "target_config": target_config,
        "tool_registry_sha256": registry.registry_sha256,
        "tool_registry_runtime_sha256": registry.registry_runtime_sha256,
        "system_prompt_sha256": sha256_bytes(PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")),
        "runner_file_sha256": sha256_bytes(
            Path(assistant_module.__file__).read_bytes()
        ),
        "llm_adapter_file_sha256": sha256_bytes(Path(llm_module.__file__).read_bytes()),
        "smoke_runner_file_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "bank_sha256s": {
            name: bank.bank_sha256 for name, bank in sorted(banks.items())
        },
    }
    lock = {
        **payload,
        "runtime_lock_sha256": _hash(payload),
    }
    return lock, sha256_bytes(canonical_json_bytes(lock))


def _treatment(config_name: TreatmentConfig, bank_sha256: str) -> MatrixTreatment:
    stages = {
        "llm_static": ("llm_static", "llm_static"),
        "s1": ("s1", "s1"),
        "s1s2": ("s2", "s1"),
        "full": ("s2", "s3"),
    }
    router_stage, body_stage = stages[config_name]
    return MatrixTreatment(
        config=config_name,
        bank_sha256=bank_sha256,
        router_stage=router_stage,
        body_stage=body_stage,
    )


def _prepare_smoke(
    arguments: argparse.Namespace,
    query_ids: tuple[str, ...],
) -> PreparedTreatmentSmoke:
    target, target_file_sha = _load_bank(arguments.target_bank, "target Bank")
    parent: StaticBankArtifact | None = None
    parent_file_sha: str | None = None
    if arguments.parent_bank is not None:
        parent, parent_file_sha = _load_bank(arguments.parent_bank, "parent Bank")
    banks = _development_bank_mapping(arguments.config, target, parent)

    clean = REPOSITORY_ROOT / "data" / "clean"
    sources = PortfolioRuntimeSources(
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
        recipe_evidence=arguments.runtime_data_root / "recipe-evidence.jsonl",
    )
    runtime = build_portfolio_tool_runtime(sources)
    catalog = load_asset_catalog(
        clean / "portfolio-mini-asset-catalog-v3",
        clean,
        verify_files=True,
    )
    runtime_lock, runtime_lock_file_sha = _development_runtime_lock(
        registry=runtime.registry,
        banks=banks,
        target_config=arguments.config,
    )
    runner = require_portfolio_assistant_runner(
        PortfolioAssistantRunner(
            registry=runtime.registry,
            system_prompt=PORTFOLIO_SYSTEM_PROMPT,
            banks=banks,
            asset_catalog=catalog,
            runtime_lock=runtime_lock,
            runtime_lock_file_sha256=runtime_lock_file_sha,
        )
    )

    inputs = _load_active_inputs()
    public_by_id = {item.query_id: item for item in inputs.assistant_queries}
    private_by_id = {item.query_id: item for item in inputs.queries}
    capabilities = _validate_query_capability_coverage(private_by_id, query_ids)
    if any(query_id not in public_by_id for query_id in query_ids):
        raise ValueError("one or more smoke queries lack public Assistant input")
    if len(capabilities) != len(query_ids):  # defensive, coverage checks above
        raise ValueError("smoke query coverage is incomplete")
    rubric_bytes = read_stable_regular_file(
        RUBRIC_PATH,
        label="Portfolio final rubric",
        max_bytes=1024 * 1024,
    )
    rubric = RubricSnapshot.model_validate_json(rubric_bytes, strict=True)
    rubric_file_sha256 = sha256_bytes(rubric_bytes)
    if rubric_file_sha256 != PORTFOLIO_FINAL_RUBRIC_FILE_SHA256:
        raise ValueError("Portfolio final rubric differs from the frozen digest")
    return PreparedTreatmentSmoke(
        config=arguments.config,
        query_ids=query_ids,
        target_bank=target,
        parent_bank=parent,
        banks=banks,
        target_bank_file_sha256=target_file_sha,
        parent_bank_file_sha256=parent_file_sha,
        runtime=runtime,
        catalog=catalog,
        runner=runner,
        runtime_lock=runtime_lock,
        runtime_lock_file_sha256=runtime_lock_file_sha,
        public_by_id=public_by_id,
        private_by_id=private_by_id,
        query_artifact_sha256=inputs.expected_query_artifact_sha256,
        plan_sha256=inputs.expected_plan_sha256,
        judge_runtime=inputs.runtime_for("aifast-gemini-judge"),
        rubric=rubric,
        rubric_file_sha256=rubric_file_sha256,
    )


def _backbone_and_budget(prepared: PreparedTreatmentSmoke):
    backbone_payload = {
        "provider": "qwen",
        "model": "qwen3-vl-flash-2026-01-22",
        "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": None,
        "system_prompt_sha256": sha256_bytes(PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")),
    }
    backbone = _self_model(BackboneLock, backbone_payload, "identity_sha256")
    budget_payload = {
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "max_tool_calls": 3,
        "max_turns": 5,
        "timeout_ms": 180000,
    }
    budget = _self_model(InferenceBudget, budget_payload, "budget_sha256")
    return backbone, budget, _registry_lock(prepared.runtime.registry)


def _request(
    *,
    prepared: PreparedTreatmentSmoke,
    query: Any,
    ordinal: int,
    run_id: str,
    backbone: BackboneLock,
    budget: InferenceBudget,
    registry_lock: Any,
) -> AssistantRequestSnapshot:
    payload = {
        "schema_version": 1,
        "matrix_run_id": run_id,
        "config": prepared.config,
        "query_ordinal": ordinal,
        "query": query,
        "treatment": _treatment(
            prepared.config,
            prepared.target_bank.bank_sha256,
        ),
        "backbone": backbone,
        "budget": budget,
        "registry": registry_lock,
    }
    hash_payload = {
        key: _model(value) if hasattr(value, "model_dump") else value
        for key, value in payload.items()
    }
    return AssistantRequestSnapshot.model_validate(
        {
            **payload,
            "request_sha256": _hash(hash_payload),
        },
        strict=True,
    )


def _assistant_result(
    prepared: PreparedTreatmentSmoke,
    request: AssistantRequestSnapshot,
    response: Any,
    run_id: str,
) -> AssistantResult:
    routed = response.selected_capability is not None
    return AssistantResult(
        run_id=run_id,
        query_id=request.query.query_id,
        config=request.config,
        response_text=response.response_text,
        visible_cards=response.visible_cards,
        visible_tool_evidence=response.visible_tool_evidence,
        tool_trace=response.tool_trace,
        selected_capability=response.selected_capability,
        skill_slug=response.skill_slug,
        bank_sha256=request.treatment.bank_sha256 if routed else None,
        route_trace_sha256=response.route_trace_sha256,
        query_artifact_sha256=prepared.query_artifact_sha256,
        split_manifest_sha256=prepared.plan_sha256,
        registry_sha256=response.registry_sha256,
        registry_runtime_sha256=response.registry_runtime_sha256,
        backbone_provider=response.backbone_provider,
        backbone_model=response.backbone_model,
        backbone_request_id=response.backbone_request_id,
        usage=response.usage,
        latency_ms=response.latency_ms,
        error_code=response.error_code,
    )


def _write_payload(path: Path, payload: Mapping[str, object]) -> tuple[str, bytes]:
    content = canonical_json_bytes(payload)
    atomic_create_file(path, content)
    return sha256_bytes(content), content


def _shared_stage2_smoke_run_id(prepared: PreparedTreatmentSmoke) -> str:
    """Use the S1+S2 Bank identity so parent and Full can reuse one route."""

    return "portfolio-treatment-smoke-stage2-" + prepared.banks["s1s2"].bank_sha256[:16]


def _load_paired_shared_route(
    source_root: Path,
    query_id: str,
) -> tuple[SharedStage2RouteArtifact, bytes]:
    route_path = source_root / "shared-routes" / f"{query_id}.json"
    content = read_stable_regular_file(
        route_path,
        label=f"paired parent shared route {query_id}",
        max_bytes=4 * 1024 * 1024,
    )
    route = SharedStage2RouteArtifact.model_validate_json(content, strict=True)
    if (
        canonical_json_bytes(route.model_dump(mode="json")) != content
        or route.query_id != query_id
    ):
        raise ValueError("paired parent shared route is not canonical and query-bound")
    return route, content


def _zero_final(query_id: str, error_code: str) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "kind": "portfolio-treatment-smoke-final-fixed-zero",
        "query_id": query_id,
        "assistant_error_code": error_code,
        "judge_invoked": False,
        "j_project": 0.0,
    }
    return {**payload, "result_sha256": _hash(payload)}


def _execute_treatment_smoke(
    arguments: argparse.Namespace,
    query_ids: tuple[str, ...],
) -> dict[str, object]:
    output_dir = arguments.output_dir.absolute()
    paired_route_source = arguments.paired_route_source_root
    if arguments.config == "full":
        if paired_route_source is None:
            raise ValueError(
                "Full smoke requires --paired-route-source-root for common routing"
            )
        paired_route_source = paired_route_source.absolute().resolve(strict=True)
        if not paired_route_source.is_dir():
            raise ValueError("paired route source root is not a directory")
    elif paired_route_source is not None:
        raise ValueError("only Full smoke may reuse a paired route source")
    if output_dir.exists():
        raise ValueError("output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    started = time.time()
    prepared: PreparedTreatmentSmoke | None = None
    rows: list[dict[str, object]] = []
    totals = {
        "assistant_input_tokens": 0,
        "assistant_output_tokens": 0,
        "shared_route_input_tokens": 0,
        "shared_route_output_tokens": 0,
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
        "assistant_model_calls": 0,
        "shared_route_model_calls": 0,
        "judge_model_calls": 0,
        "judge_captured_response_count": 0,
    }
    try:
        prepared = _prepare_smoke(arguments, query_ids)
        backbone, budget, registry_lock = _backbone_and_budget(prepared)
        isolation = make_active_portfolio_evaluator_isolation_lock()
        run_id = (
            _shared_stage2_smoke_run_id(prepared)
            if prepared.config in {"s1s2", "full"}
            else (
                f"portfolio-treatment-smoke-{prepared.config}-"
                f"{prepared.target_bank.bank_sha256[:16]}"
            )
        )
        blinding_key = secrets.token_bytes(32)
        atomic_create_file(staging / "blinding-key.bin", blinding_key)
        (staging / "assistant").mkdir()
        (staging / "final").mkdir()
        (staging / "shared-routes").mkdir()
        ledger_root = staging / "budget-ledger"
        initialize_portfolio_budget_ledger(
            ledger_root,
            matrix_run_id=run_id,
            phase_cap_cny=Decimal(str(arguments.phase_cap_cny)),
            prior_observed_cost_cny=Decimal(str(arguments.prior_cost_cny)),
        )
        budget_session = open_portfolio_budget_ledger_session(ledger_root)

        status = "complete"
        for ordinal, query_id in enumerate(query_ids):
            current = calculate_smoke_cost(
                assistant_input_tokens=totals["assistant_input_tokens"],
                assistant_output_tokens=totals["assistant_output_tokens"],
                shared_route_input_tokens=totals["shared_route_input_tokens"],
                shared_route_output_tokens=totals["shared_route_output_tokens"],
                judge_input_tokens=totals["judge_input_tokens"],
                judge_output_tokens=totals["judge_output_tokens"],
                prior_cost_cny=arguments.prior_cost_cny,
                phase_cap_cny=arguments.phase_cap_cny,
            )
            if current["cap_reached"]:
                status = "budget_cap_reached"
                break
            request = _request(
                prepared=prepared,
                query=prepared.public_by_id[query_id],
                ordinal=ordinal,
                run_id=run_id,
                backbone=backbone,
                budget=budget,
                registry_lock=registry_lock,
            )
            assistant_budget_context = PortfolioAssistantBudgetContext(
                ledger_root=ledger_root,
                shard_id=f"portfolio-treatment-smoke-{prepared.config}",
                instance_sha256=request.request_sha256,
                attempt_index=1,
            )
            shared_route = None
            shared_route_file: str | None = None
            shared_route_file_sha: str | None = None
            if prepared.config in {"s1s2", "full"}:
                shared_route_file = f"shared-routes/{query_id}.json"
                if prepared.config == "full":
                    if paired_route_source is None:  # pragma: no cover - preflight
                        raise ValueError("Full paired route source disappeared")
                    shared_route, shared_route_bytes = _load_paired_shared_route(
                        paired_route_source,
                        query_id,
                    )
                    atomic_create_file(
                        staging / shared_route_file,
                        shared_route_bytes,
                    )
                    shared_route_file_sha = sha256_bytes(shared_route_bytes)
                else:
                    try:
                        shared_route = prepared.runner.prepare_shared_stage2_route(
                            request,
                            budget_context=assistant_budget_context,
                        )
                    except PortfolioBudgetExceededError:
                        status = "budget_cap_reached"
                        break
                    shared_route_file_sha, _ = _write_payload(
                        staging / shared_route_file,
                        shared_route.model_dump(mode="json"),
                    )
                    totals["shared_route_input_tokens"] += (
                        shared_route.route_call.input_tokens
                    )
                    totals["shared_route_output_tokens"] += (
                        shared_route.route_call.output_tokens
                    )
                    totals["shared_route_model_calls"] += 1

            try:
                execution = prepared.runner.execute(
                    request,
                    shared_stage2_route=shared_route,
                    budget_context=assistant_budget_context,
                )
            except PortfolioBudgetExceededError:
                status = "budget_cap_reached"
                break
            response = execution.response
            receipt = execution.receipt
            totals["assistant_input_tokens"] += response.usage.input_tokens
            totals["assistant_output_tokens"] += response.usage.output_tokens
            totals["assistant_model_calls"] += len(receipt.model_calls)
            assistant_payload = {
                "schema_version": 1,
                "kind": "portfolio-treatment-smoke-assistant",
                "policy_version": SMOKE_POLICY_VERSION,
                "query_id": query_id,
                "config": prepared.config,
                "request": _model(request),
                "response": _model(response),
                "receipt": _model(receipt),
                "shared_route_file": shared_route_file,
                "shared_route_file_sha256": shared_route_file_sha,
            }
            assistant_row = {
                **assistant_payload,
                "row_sha256": _hash(assistant_payload),
            }
            assistant_file = f"assistant/{query_id}.json"
            assistant_file_sha, _ = _write_payload(
                staging / assistant_file,
                assistant_row,
            )

            judge = None
            effective_error_code = response.error_code
            if response.error_code is None:
                try:
                    packet = build_final_evaluation_packet(
                        prepared.private_by_id[query_id],
                        _assistant_result(prepared, request, response, run_id),
                        asset_catalog=prepared.catalog,
                        rubric=prepared.rubric,
                        blinding_key=blinding_key,
                    )
                except HiddenEvaluationIdentityError:
                    effective_error_code = "hidden_evaluation_identity"
                    final_payload = _zero_final(
                        query_id,
                        effective_error_code,
                    )
                    j_project = 0.0
                    final_status = "assistant_public_identity_leak_fixed_zero"
                else:
                    try:
                        judge = run_visual_final_judge(
                            packet,
                            isolation,
                            remote_runtime=prepared.judge_runtime,
                            budget_context=FinalJudgeBudgetContext(
                                ledger_root=ledger_root,
                                matrix_run_id=run_id,
                                shard_id=(
                                    f"portfolio-treatment-smoke-{prepared.config}"
                                ),
                                config=prepared.config,
                                query_id=query_id,
                                instance_sha256=request.request_sha256,
                                request_sha256=request.request_sha256,
                                attempt_index=1,
                            ),
                            max_tokens=2048,
                            record_usage=True,
                        )
                    except PortfolioBudgetExceededError:
                        status = "budget_cap_reached"
                        break
                    final_payload = judge.model_dump(mode="json")
                    _accumulate_judge_accounting(totals, judge)
                    j_project = judge.outcome.scores.j_project
                    final_status = judge.outcome.status
            else:
                final_payload = _zero_final(query_id, response.error_code)
                j_project = 0.0
                final_status = "assistant_fixed_zero"
            final_file = f"final/{query_id}.json"
            final_file_sha, _ = _write_payload(
                staging / final_file,
                final_payload,
            )

            shared_in = (
                shared_route.route_call.input_tokens
                if shared_route is not None and prepared.config != "full"
                else 0
            )
            shared_out = (
                shared_route.route_call.output_tokens
                if shared_route is not None and prepared.config != "full"
                else 0
            )
            judge_in = judge.aggregate_usage.input_tokens if judge is not None else 0
            judge_out = judge.aggregate_usage.output_tokens if judge is not None else 0
            row_cost = calculate_smoke_cost(
                assistant_input_tokens=response.usage.input_tokens,
                assistant_output_tokens=response.usage.output_tokens,
                shared_route_input_tokens=shared_in,
                shared_route_output_tokens=shared_out,
                judge_input_tokens=judge_in,
                judge_output_tokens=judge_out,
            )
            assistant_hard_error = effective_error_code is not None
            evaluator_anomaly = not assistant_hard_error and final_status != "scored"
            result_payload = {
                "schema_version": 1,
                "kind": "portfolio-treatment-smoke-result",
                "policy_version": SMOKE_POLICY_VERSION,
                "query_id": query_id,
                "canonical_capability": prepared.private_by_id[
                    query_id
                ].canonical_capability,
                "config": prepared.config,
                "target_bank_sha256": prepared.target_bank.bank_sha256,
                "assistant_file": assistant_file,
                "assistant_file_sha256": assistant_file_sha,
                "final_file": final_file,
                "final_file_sha256": final_file_sha,
                "shared_route_file": shared_route_file,
                "shared_route_file_sha256": shared_route_file_sha,
                "selected_capability": response.selected_capability,
                "route_correct": (
                    response.selected_capability
                    == prepared.private_by_id[query_id].canonical_capability
                ),
                "assistant_error_code": effective_error_code,
                "final_status": final_status,
                # Retained as the v3 wire-compatible combined projection for
                # existing attribution readers.  Gate v4 derives the two
                # disjoint classes from assistant_error_code/final_status.
                "hard_error": assistant_hard_error or evaluator_anomaly,
                "skill_adherence": calculate_portfolio_skill_adherence(
                    bank=prepared.target_bank,
                    selected_capability=response.selected_capability,
                    skill_slug=response.skill_slug,
                    response_text=response.response_text,
                    visible_cards=response.visible_cards,
                    hard_error=assistant_hard_error or evaluator_anomaly,
                ),
                "j_project": j_project,
                "usage_and_cost": row_cost,
            }
            rows.append(
                {
                    **result_payload,
                    "result_sha256": _hash(result_payload),
                }
            )
            aggregate = calculate_smoke_cost(
                assistant_input_tokens=totals["assistant_input_tokens"],
                assistant_output_tokens=totals["assistant_output_tokens"],
                shared_route_input_tokens=totals["shared_route_input_tokens"],
                shared_route_output_tokens=totals["shared_route_output_tokens"],
                judge_input_tokens=totals["judge_input_tokens"],
                judge_output_tokens=totals["judge_output_tokens"],
                prior_cost_cny=arguments.prior_cost_cny,
                phase_cap_cny=arguments.phase_cap_cny,
            )
            budget_ledger = budget_session.state
            _emit_progress(
                {
                    "completed": len(rows),
                    "query_id": query_id,
                    "cumulative_cost_cny": aggregate["cumulative_cost_cny"],
                    "phase_cap_cny": arguments.phase_cap_cny,
                    "budget_accountable_cost_cny": format(
                        budget_ledger.accountable_cost_cny, ".12f"
                    ),
                    "budget_remaining_cost_cny": format(
                        budget_ledger.remaining_cost_cny, ".12f"
                    ),
                }
            )
            if aggregate["cap_reached"] and len(rows) < len(query_ids):
                status = "budget_cap_reached"
                break

        results_content = b"".join(canonical_json_bytes(row) for row in rows)
        atomic_create_file(staging / "results.jsonl", results_content)
        aggregate = calculate_smoke_cost(
            assistant_input_tokens=totals["assistant_input_tokens"],
            assistant_output_tokens=totals["assistant_output_tokens"],
            shared_route_input_tokens=totals["shared_route_input_tokens"],
            shared_route_output_tokens=totals["shared_route_output_tokens"],
            judge_input_tokens=totals["judge_input_tokens"],
            judge_output_tokens=totals["judge_output_tokens"],
            prior_cost_cny=arguments.prior_cost_cny,
            phase_cap_cny=arguments.phase_cap_cny,
        )
        budget_ledger = budget_session.state
        summary_payload: dict[str, object] = {
            "schema_version": 1,
            "kind": "portfolio-treatment-development-smoke-summary",
            "policy_version": SMOKE_POLICY_VERSION,
            "status": status,
            "track": "portfolio",
            "formal_eligible": False,
            "matrix_ready": False,
            "official_treatment_claim_allowed": False,
            "bank_provenance_disposition": (
                "runtime_behavior_only_not_creator_optimizer_refiner_evidence"
            ),
            "scaffold_claim_disposition": "forbidden",
            "treatment_chain_manifest_created": False,
            "config": prepared.config,
            "query_ids_requested": list(query_ids),
            "query_ids_completed": [row["query_id"] for row in rows],
            "query_ids_incomplete": [
                query_id
                for query_id in query_ids
                if query_id not in {row["query_id"] for row in rows}
            ],
            "incomplete_score_disposition": (
                "not_scored_no_fixed_zero" if status != "complete" else "not_applicable"
            ),
            "capabilities_completed": [row["canonical_capability"] for row in rows],
            "query_count_requested": len(query_ids),
            "query_count_completed": len(rows),
            "target_bank_sha256": prepared.target_bank.bank_sha256,
            "target_bank_file_sha256": prepared.target_bank_file_sha256,
            "parent_bank_sha256": (
                prepared.parent_bank.bank_sha256
                if prepared.parent_bank is not None
                else None
            ),
            "parent_bank_file_sha256": prepared.parent_bank_file_sha256,
            "runtime_lock_sha256": prepared.runtime_lock["runtime_lock_sha256"],
            "runtime_lock_file_sha256": prepared.runtime_lock_file_sha256,
            "results_file": "results.jsonl",
            "results_file_sha256": sha256_bytes(results_content),
            "assistant_model_calls": totals["assistant_model_calls"],
            "shared_route_model_calls": totals["shared_route_model_calls"],
            "shared_route_origin": (
                "reused_exact_parent"
                if prepared.config == "full"
                else "generated"
                if prepared.config == "s1s2"
                else "not_applicable"
            ),
            "judge_model_calls": totals["judge_model_calls"],
            "judge_captured_response_count": totals["judge_captured_response_count"],
            "usage_and_cost": aggregate,
            "budget_authority_sha256": budget_ledger.authority.authority_sha256,
            "budget_ledger_last_event_index": budget_ledger.last_event_index,
            "budget_ledger_last_event_sha256": budget_ledger.last_event_sha256,
            "budget_ledger_settled_actual_cost_cny": format(
                budget_ledger.settled_actual_cost_cny, ".12f"
            ),
            "budget_ledger_unresolved_reserved_cost_cny": format(
                budget_ledger.unresolved_reserved_cost_cny, ".12f"
            ),
            "budget_ledger_accountable_cost_cny": format(
                budget_ledger.accountable_cost_cny, ".12f"
            ),
            "budget_ledger_remaining_phase_cap_cny": format(
                budget_ledger.remaining_cost_cny, ".12f"
            ),
            "mean_j_project": (
                sum(float(row["j_project"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "route_accuracy": (
                sum(bool(row["route_correct"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "hard_error_count": sum(bool(row["hard_error"]) for row in rows),
            "assistant_hard_error_count": sum(
                row["assistant_error_code"] is not None for row in rows
            ),
            "evaluator_anomaly_count": sum(
                row["assistant_error_code"] is None and row["final_status"] != "scored"
                for row in rows
            ),
            "mean_skill_adherence": (
                sum(float(row["skill_adherence"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "blinding_key_sha256": sha256_bytes(blinding_key),
            "rubric_file_sha256": prepared.rubric_file_sha256,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        summary = {
            **summary_payload,
            "summary_sha256": _hash(summary_payload),
        }
        _write_payload(staging / "summary.json", summary)
        staging.replace(output_dir)
        return summary
    except Exception as error:
        failure_cost = calculate_smoke_cost(
            assistant_input_tokens=totals["assistant_input_tokens"],
            assistant_output_tokens=totals["assistant_output_tokens"],
            shared_route_input_tokens=totals["shared_route_input_tokens"],
            shared_route_output_tokens=totals["shared_route_output_tokens"],
            judge_input_tokens=totals["judge_input_tokens"],
            judge_output_tokens=totals["judge_output_tokens"],
            prior_cost_cny=arguments.prior_cost_cny,
            phase_cap_cny=arguments.phase_cap_cny,
        )
        failure_payload = {
            "schema_version": 1,
            "kind": "portfolio-treatment-development-smoke-failure",
            "policy_version": SMOKE_POLICY_VERSION,
            "formal_eligible": False,
            "matrix_ready": False,
            "official_treatment_claim_allowed": False,
            "error_type": type(error).__qualname__,
            "error": str(error),
            "model_calls_may_have_occurred": prepared is not None,
            "query_ids_completed": [row["query_id"] for row in rows],
            "assistant_model_calls": totals["assistant_model_calls"],
            "shared_route_model_calls": totals["shared_route_model_calls"],
            "judge_model_calls": totals["judge_model_calls"],
            "judge_captured_response_count": totals["judge_captured_response_count"],
            "usage_and_cost": failure_cost,
        }
        try:
            _write_payload(
                staging / "failure.json",
                {
                    **failure_payload,
                    "failure_sha256": _hash(failure_payload),
                },
            )
            if not output_dir.exists():
                staging.replace(output_dir)
            else:
                shutil.rmtree(staging, ignore_errors=True)
        except OSError:
            # Preserve already-written call evidence even when the atomic
            # directory rename is unavailable on the host filesystem.
            if not output_dir.exists():
                try:
                    shutil.copytree(staging, output_dir)
                except OSError:
                    pass
        raise


def _dry_run_payload(
    arguments: argparse.Namespace,
    query_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "status": "dry_run_only",
        "policy_version": SMOKE_POLICY_VERSION,
        "model_calls_performed": 0,
        "execute_required_for_calls": True,
        "formal_eligible": False,
        "matrix_ready": False,
        "official_treatment_claim_allowed": False,
        "scaffold_claim_disposition": "forbidden",
        "config": arguments.config,
        "query_ids": list(query_ids),
        "target_bank": arguments.target_bank.as_posix(),
        "parent_bank": (
            arguments.parent_bank.as_posix()
            if arguments.parent_bank is not None
            else None
        ),
        "runtime_data_root": arguments.runtime_data_root.as_posix(),
        "output_dir": arguments.output_dir.as_posix(),
        "prior_cost_cny": arguments.prior_cost_cny,
        "phase_cap_cny": arguments.phase_cap_cny,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        query_ids = _exact_query_ids(arguments.query_ids)
        _validate_cost_authority(
            arguments.prior_cost_cny,
            arguments.phase_cap_cny,
        )
        if not arguments.execute:
            print(
                json.dumps(
                    _dry_run_payload(arguments, query_ids),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        _require_active_judge_execute_ready()
        summary = _execute_treatment_smoke(arguments, query_ids)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["status"] == "complete" else 4
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"portfolio-treatment-smoke: {error}", file=sys.stderr)
        return 2


def _require_active_judge_execute_ready() -> None:
    raise ValueError(
        "active AIFast Gemini Judge execution is disabled while "
        f"{PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
