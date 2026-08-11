"""Run one locked 25-query Portfolio Assistant shard without evaluator calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

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
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    load_portfolio_launch_package,
)
from skillchain.runners.assistant import (  # noqa: E402
    PortfolioAssistantRunner,
    require_portfolio_assistant_runner,
)
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_SYSTEM_PROMPT,
    PortfolioRuntimeSources,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes  # noqa: E402


def _hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _model(value):
    return value.model_dump(mode="json")


def _self_model(model_type, payload: dict, field: str):
    return model_type.model_validate(
        {**payload, field: _hash(payload)}, strict=True
    )


def _treatment(config_name: str, bank_sha256: str | None) -> MatrixTreatment:
    stages = {
        "noskill": ("disabled", "disabled"),
        "llm_static": ("llm_static", "llm_static"),
        "s1": ("s1", "s1"),
        "s1s2": ("s2", "s1"),
        "full": ("s2", "s3"),
    }
    router, body = stages[config_name]
    return MatrixTreatment(
        config=config_name,
        bank_sha256=bank_sha256,
        router_stage=router,
        body_stage=body,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-root", type=Path, required=True)
    parser.add_argument("--launch-plan-sha256", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--query-limit",
        type=int,
        choices=range(1, 26),
        default=25,
        metavar="1..25",
    )
    parser.add_argument(
        "--query-offset",
        type=int,
        choices=range(0, 25),
        default=0,
        metavar="0..24",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        print("portfolio-assistant-smoke: output directory exists", file=sys.stderr)
        return 2
    launch = load_portfolio_launch_package(
        args.launch_root,
        expected_plan_file_sha256=args.launch_plan_sha256,
    )
    if not launch.plan.execution_ready:
        print("portfolio-assistant-smoke: launch package is blocked", file=sys.stderr)
        return 2
    try:
        shard = next(
            item for item in launch.plan.shards if item.shard_id == args.shard_id
        )
    except StopIteration:
        print("portfolio-assistant-smoke: unknown shard", file=sys.stderr)
        return 2
    all_members = tuple(
        item for item in launch.instances if item.shard_id == shard.shard_id
    )
    members = all_members[
        args.query_offset : args.query_offset + args.query_limit
    ]
    if len(members) != args.query_limit:
        print("portfolio-assistant-smoke: shard coverage is incomplete", file=sys.stderr)
        return 2
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
        recipe_evidence=args.runtime_root / "recipe-evidence.jsonl",
    )
    runtime = build_portfolio_tool_runtime(sources)
    banks = {
        name: StaticBankArtifact.model_validate_json(
            (args.runtime_root / f"bank-{name}.json").read_bytes(),
            strict=True,
        )
        for name in ("llm_static", "s1", "s1s2", "full")
    }
    lock_bytes = (args.runtime_root / "runtime-lock.json").read_bytes()
    if sha256_bytes(lock_bytes) != args.runtime_lock_sha256:
        print("portfolio-assistant-smoke: runtime-lock digest mismatch", file=sys.stderr)
        return 2
    lock = json.loads(lock_bytes)
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
            runtime_lock=lock,
            runtime_lock_file_sha256=args.runtime_lock_sha256,
        )
    )
    backbone_payload = {
        "provider": "qwen",
        "model": "qwen3-vl-flash-2026-01-22",
        "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": None,
        "system_prompt_sha256": sha256_bytes(
            PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")
        ),
    }
    backbone = _self_model(
        BackboneLock, backbone_payload, "identity_sha256"
    )
    budget_payload = {
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "max_tool_calls": 3,
        "max_turns": 5,
        "timeout_ms": 180000,
    }
    budget = _self_model(InferenceBudget, budget_payload, "budget_sha256")
    registry_lock = _registry_lock(runtime.registry)
    inputs = _load_active_inputs()
    public_by_id = {item.query_id: item for item in inputs.assistant_queries}
    bank_sha = None if shard.config == "noskill" else banks[shard.config].bank_sha256
    treatment = _treatment(shard.config, bank_sha)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.",
            dir=args.output_dir.parent,
        )
    )
    rows: list[dict] = []
    raw_model_responses: list[dict] = []
    original_chat = llm_module.chat

    def captured_chat(*chat_args, **chat_kwargs):
        response = original_chat(*chat_args, **chat_kwargs)
        raw_model_responses.append(response.model_dump(mode="json"))
        return response

    llm_module.chat = captured_chat
    started = time.time()
    try:
        for ordinal, member in enumerate(members):
            call_start = len(raw_model_responses)
            query = public_by_id[member.query_id]
            request_payload = {
                "schema_version": 1,
                "matrix_run_id": launch.plan.matrix_run_id + "-preflight-smoke",
                "config": shard.config,
                "query_ordinal": ordinal,
                "query": query,
                "treatment": treatment,
                "backbone": backbone,
                "budget": budget,
                "registry": registry_lock,
            }
            request_hash_payload = {
                key: _model(value) if hasattr(value, "model_dump") else value
                for key, value in request_payload.items()
            }
            request = AssistantRequestSnapshot.model_validate(
                {
                    **request_payload,
                    "request_sha256": _hash(request_hash_payload),
                },
                strict=True,
            )
            execution = runner.execute(request)
            row = {
                "schema_version": 1,
                "query_id": member.query_id,
                "config": shard.config,
                "request": _model(request),
                "response": _model(execution.response),
                "receipt": _model(execution.receipt),
                "raw_model_responses": raw_model_responses[call_start:],
            }
            row["row_sha256"] = _hash(row)
            rows.append(row)
            (staging / f"{ordinal:02d}-{member.query_id}.json").write_bytes(
                canonical_json_bytes(row)
            )
            print(
                json.dumps(
                    {
                        "completed": ordinal + 1,
                        "error_code": execution.response.error_code,
                        "query_id": member.query_id,
                        "tool_calls": len(execution.response.tool_trace),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        results = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        (staging / "results.jsonl").write_bytes(results)
        input_tokens = sum(
            row["response"]["usage"]["input_tokens"] for row in rows
        )
        output_tokens = sum(
            row["response"]["usage"]["output_tokens"] for row in rows
        )
        calls = sum(len(row["receipt"]["model_calls"]) for row in rows)
        errors = sum(row["response"]["error_code"] is not None for row in rows)
        observed_cost = (
            input_tokens * 0.15 / 1_000_000
            + output_tokens * 1.5 / 1_000_000
        )
        summary_payload = {
            "schema_version": 1,
            "kind": "portfolio-assistant-shard-smoke",
            "track": "portfolio",
            "formal_eligible": False,
            "launch_plan_sha256": launch.plan.launch_plan_sha256,
            "shard_id": shard.shard_id,
            "config": shard.config,
            "query_count": len(rows),
            "model_calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "dashscope_observed_cost_cny": observed_cost,
            "error_count": errors,
            "success_count": len(rows) - errors,
            "elapsed_seconds": round(time.time() - started, 3),
            "results_file_sha256": sha256_bytes(results),
            "runtime_lock_sha256": lock["runtime_lock_sha256"],
            "bank_sha256": bank_sha,
        }
        summary = {
            **summary_payload,
            "summary_sha256": _hash(summary_payload),
        }
        (staging / "summary.json").write_bytes(canonical_json_bytes(summary))
        staging.replace(args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if errors == 0 else 3
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"portfolio-assistant-smoke: {error}", file=sys.stderr)
        return 2
    finally:
        llm_module.chat = original_chat


if __name__ == "__main__":
    raise SystemExit(main())
