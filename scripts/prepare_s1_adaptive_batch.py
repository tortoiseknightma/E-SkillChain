#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import prepare_s1_counterfactual_cycle as counterfactual  # noqa: E402
from skillchain.evaluation.core_fast.models import (  # noqa: E402
    AssistantObservation,
    CAPABILITIES,
)
from skillchain.schemas import Query  # noqa: E402
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes  # noqa: E402


PLAN_KIND = "core-fast-s1-adaptive-batch-plan"
BATCH_KIND = "core-fast-s1-adaptive-batch-definition"
ROUND_KIND = "core-fast-s1-adaptive-round-definition"


def _load_plan(path: Path) -> dict[str, object]:
    plan = counterfactual._load_json(path)  # noqa: SLF001
    rounds = plan.get("rounds")
    if (
        plan.get("kind") != PLAN_KIND
        or plan.get("schema_version") != 1
        or not isinstance(plan.get("campaign_id"), str)
        or not isinstance(plan.get("batch_id"), str)
        or not isinstance(rounds, list)
        or not 2 <= len(rounds) <= 5
    ):
        raise ValueError("adaptive batch plan identity or size is invalid")
    seen: set[str] = set()
    for row in rounds:
        if not isinstance(row, dict):
            raise ValueError("adaptive round definition must be an object")
        round_id = row.get("round_id")
        capability = row.get("capability")
        surface = row.get("surface")
        gains = row.get("gain_seeds")
        regressions = row.get("regressions")
        memory = row.get("memory")
        if (
            not isinstance(round_id, str)
            or not round_id.startswith("r")
            or not round_id[1:].isdigit()
            or round_id in seen
            or capability not in CAPABILITIES
            or surface not in {"action-policy", "response-policy"}
            or not isinstance(gains, list)
            or gains != sorted(set(gains))
            or len(gains) < 2
            or not isinstance(regressions, list)
            or regressions != sorted(set(regressions))
            or len(regressions) < 3
            or not isinstance(memory, list)
            or not memory
            or any(
                not isinstance(item, dict) or item.get("capability") != capability
                for item in memory
            )
        ):
            raise ValueError(f"adaptive round definition is invalid: {round_id}")
        seen.add(round_id)
    return plan


def _parent_inputs(
    *,
    base_spec_path: Path,
    bootstrap_receipt_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    StaticBankArtifact,
    dict[str, AssistantObservation],
    tuple[Query, ...],
    str,
]:
    parent = counterfactual._verify_r12()  # noqa: SLF001
    receipt = counterfactual._load_json(bootstrap_receipt_path)  # noqa: SLF001
    if (
        receipt.get("kind") != "core-fast-s1-parent-opt800-bootstrap"
        or receipt.get("source_round_id") != "r12"
        or receipt.get("parent_bank_sha256") != counterfactual.R12_BANK_SHA
        or receipt.get("row_count") != 800
    ):
        raise ValueError("adaptive batch requires the accepted R12 opt800 bootstrap")
    opt_path = Path(str(receipt["opt_results_path"]))
    opt_sha = str(receipt["opt_results_sha256"])
    if counterfactual._sha(opt_path) != opt_sha:  # noqa: SLF001
        raise ValueError("adaptive batch parent opt800 SHA-256 differs")
    fixed_samples = receipt.get("fixed_samples")
    if not isinstance(fixed_samples, dict):
        raise ValueError("adaptive batch parent bootstrap lacks fixed samples")
    binding = {
        **parent,
        "opt_results_path": str(opt_path),
        "opt_results_sha256": opt_sha,
    }
    base = counterfactual._load_json(base_spec_path)  # noqa: SLF001
    paths = base.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("base spec paths are missing")
    query_rows = counterfactual._read_jsonl(  # noqa: SLF001
        counterfactual._resolve_spec_path(  # noqa: SLF001
            base_spec_path, str(paths["queries"])
        )
    )
    counterfactual._verify_parent_opt800(  # noqa: SLF001
        opt_path=opt_path,
        base=base,
        queries=query_rows,
    )
    bank = StaticBankArtifact.model_validate_json(
        Path(str(binding["bank_path"])).read_bytes(), strict=True
    )
    observations = {
        item.query_id: item
        for item in (
            AssistantObservation.model_validate(row, strict=True)
            for row in counterfactual._read_jsonl(opt_path)  # noqa: SLF001
        )
    }
    queries = tuple(Query.model_validate(row, strict=True) for row in query_rows)
    return base, binding, fixed_samples, bank, observations, queries, opt_sha


def _definition(
    row: dict[str, object], memory: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "capability": row["capability"],
        "surface": row["surface"],
        "gain_seeds": tuple(row["gain_seeds"]),
        "regressions": tuple(row["regressions"]),
        "memory": tuple(memory),
    }


def _preliminary_specs(
    *,
    plan: dict[str, object],
    base: dict[str, object],
    binding: dict[str, object],
    fixed_samples: dict[str, object],
    preflight_path: Path,
) -> dict[str, dict[str, object]]:
    rounds = plan["rounds"]
    assert isinstance(rounds, list)
    cycle_id = f"{plan['campaign_id']}-{plan['batch_id']}"
    return {
        str(row["round_id"]): counterfactual._round_spec(  # noqa: SLF001
            base,
            round_id=str(row["round_id"]),
            parent_binding=binding,
            definition=_definition(row, row["memory"]),
            cycle_id=cycle_id,
            fixed_samples=fixed_samples,
            typed_contract=True,
            preflight_path=preflight_path,
            preflight_sha256="0" * 64,
        )
        for row in rounds
        if isinstance(row, dict) and isinstance(row.get("memory"), list)
    }


def freeze_batch(args: argparse.Namespace) -> int:
    plan_path = args.plan.resolve()
    plan = _load_plan(plan_path)
    root = args.batch_root.resolve()
    preflight_path = root / "batch-preflight.json"
    (
        base,
        binding,
        fixed_samples,
        bank,
        observations,
        queries,
        opt_sha,
    ) = _parent_inputs(
        base_spec_path=args.base_spec.resolve(),
        bootstrap_receipt_path=args.bootstrap_receipt.resolve(),
    )
    preliminary = _preliminary_specs(
        plan=plan,
        base=base,
        binding=binding,
        fixed_samples=fixed_samples,
        preflight_path=preflight_path,
    )
    round_ids = tuple(str(row["round_id"]) for row in plan["rounds"])
    cycle_id = f"{plan['campaign_id']}-{plan['batch_id']}"
    preflight = counterfactual._build_cycle_preflight(  # noqa: SLF001
        preliminary_specs=preliminary,
        parent=bank,
        parent_opt=observations,
        queries=queries,
        parent_opt_sha256=opt_sha,
        round_ids=round_ids,
        cycle_id=cycle_id,
    )
    counterfactual._write_create_only(preflight_path, preflight)  # noqa: SLF001
    if preflight["passed"] is not True:
        raise ValueError("adaptive batch preflight failed; no provider call is allowed")
    # Five worst-case single-surface rounds remain below the stage CNY 10 ceiling:
    # 18 Feedback attempts and 700 Assistant outer calls per round.
    projected_cny = len(round_ids) * (18 * 0.007 + 700 * 0.0025)
    if projected_cny > 10.0:
        raise ValueError("adaptive batch worst-case DashScope estimate exceeds CNY 10")
    unsigned = {
        "schema_version": 1,
        "kind": BATCH_KIND,
        "campaign_id": plan["campaign_id"],
        "batch_id": plan["batch_id"],
        "cycle_id": cycle_id,
        "plan_path": str(plan_path),
        "plan_sha256": counterfactual._sha(plan_path),  # noqa: SLF001
        "parent_round_id": "r12",
        "parent_bank_sha256": bank.bank_sha256,
        "parent_opt_sha256": opt_sha,
        "preflight_path": str(preflight_path),
        "preflight_sha256": counterfactual._sha(preflight_path),  # noqa: SLF001
        "round_order": list(round_ids),
        "maximum_creator_sessions": len(round_ids),
        "dashscope_stage_budget_cny": 10.0,
        "projected_dashscope_cny": projected_cny,
        "round_specs_are_frozen_sequentially": True,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    path = root / "batch-definition.json"
    counterfactual._write_create_only(path, receipt)  # noqa: SLF001
    print(
        json.dumps(
            {"batch_definition": str(path), "preflight": str(preflight_path)}, indent=2
        )
    )
    return 0


def _verify_batch(path: Path, plan_path: Path) -> dict[str, object]:
    batch = counterfactual._load_json(path)  # noqa: SLF001
    unsigned = {key: value for key, value in batch.items() if key != "receipt_sha256"}
    if (
        batch.get("kind") != BATCH_KIND
        or batch.get("receipt_sha256") != sha256_bytes(canonical_json_bytes(unsigned))
        or batch.get("plan_path") != str(plan_path)
        or batch.get("plan_sha256") != counterfactual._sha(plan_path)  # noqa: SLF001
        or counterfactual._sha(Path(str(batch["preflight_path"])))  # noqa: SLF001
        != batch.get("preflight_sha256")
    ):
        raise ValueError("adaptive batch definition drifted")
    return batch


def freeze_round(args: argparse.Namespace) -> int:
    plan_path = args.plan.resolve()
    plan = _load_plan(plan_path)
    batch_path = args.batch_definition.resolve()
    batch = _verify_batch(batch_path, plan_path)
    rows = plan["rounds"]
    assert isinstance(rows, list)
    row = next((item for item in rows if item.get("round_id") == args.round_id), None)
    if not isinstance(row, dict):
        raise ValueError(f"round is not pre-registered in this batch: {args.round_id}")
    memory = list(row["memory"])
    memory_sha: str | None = None
    if args.memory is not None:
        memory_path = args.memory.resolve()
        memory_payload = counterfactual._load_json(memory_path)  # noqa: SLF001
        additions = memory_payload.get("memory")
        if (
            memory_payload.get("kind") != "core-fast-s1-adaptive-round-memory"
            or memory_payload.get("round_id") != args.round_id
            or not isinstance(additions, list)
            or any(
                not isinstance(item, dict)
                or item.get("capability") != row["capability"]
                for item in additions
            )
        ):
            raise ValueError("adaptive round memory identity differs")
        memory.extend(additions)
        memory_sha = counterfactual._sha(memory_path)  # noqa: SLF001
    (
        base,
        binding,
        fixed_samples,
        bank,
        observations,
        queries,
        opt_sha,
    ) = _parent_inputs(
        base_spec_path=args.base_spec.resolve(),
        bootstrap_receipt_path=args.bootstrap_receipt.resolve(),
    )
    preflight_path = Path(str(batch["preflight_path"]))
    preliminary = _preliminary_specs(
        plan=plan,
        base=base,
        binding=binding,
        fixed_samples=fixed_samples,
        preflight_path=preflight_path,
    )
    round_ids = tuple(str(item["round_id"]) for item in rows)
    active_preflight = counterfactual._build_cycle_preflight(  # noqa: SLF001
        preliminary_specs=preliminary,
        parent=bank,
        parent_opt=observations,
        queries=queries,
        parent_opt_sha256=opt_sha,
        round_ids=round_ids,
        cycle_id=str(batch["cycle_id"]),
    )
    if canonical_json_bytes(active_preflight) != preflight_path.read_bytes():
        raise ValueError("active mechanism changed the frozen adaptive batch preflight")
    spec = counterfactual._round_spec(  # noqa: SLF001
        base,
        round_id=args.round_id,
        parent_binding=binding,
        definition=_definition(row, memory),
        cycle_id=str(batch["cycle_id"]),
        fixed_samples=fixed_samples,
        typed_contract=True,
        preflight_path=preflight_path,
        preflight_sha256=str(batch["preflight_sha256"]),
    )
    spec_path = batch_path.parent / "specs" / f"{args.round_id}.json"
    counterfactual._write_create_only(spec_path, spec)  # noqa: SLF001
    unsigned = {
        "schema_version": 1,
        "kind": ROUND_KIND,
        "campaign_id": plan["campaign_id"],
        "batch_id": plan["batch_id"],
        "round_id": args.round_id,
        "capability": row["capability"],
        "surface": row["surface"],
        "batch_definition_sha256": counterfactual._sha(batch_path),  # noqa: SLF001
        "memory_file_sha256": memory_sha,
        "spec_path": str(spec_path),
        "spec_sha256": counterfactual._sha(spec_path),  # noqa: SLF001
        "run_root": str(batch_path.parent / "runs" / args.round_id),
    }
    receipt = {
        **unsigned,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    receipt_path = batch_path.parent / "rounds" / f"{args.round_id}.json"
    counterfactual._write_create_only(receipt_path, receipt)  # noqa: SLF001
    print(json.dumps({"round_definition": str(receipt_path), **unsigned}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one adaptive R12 S1 batch")
    sub = parser.add_subparsers(dest="command", required=True)
    batch = sub.add_parser("freeze-batch")
    batch.add_argument("--plan", type=Path, required=True)
    batch.add_argument("--batch-root", type=Path, required=True)
    batch.add_argument("--bootstrap-receipt", type=Path, required=True)
    batch.add_argument(
        "--base-spec",
        type=Path,
        default=REPOSITORY_ROOT / "specs" / "core-experiment-fast-v1.json",
    )
    round_parser = sub.add_parser("freeze-round")
    round_parser.add_argument("--plan", type=Path, required=True)
    round_parser.add_argument("--batch-definition", type=Path, required=True)
    round_parser.add_argument("--bootstrap-receipt", type=Path, required=True)
    round_parser.add_argument("--round-id", required=True)
    round_parser.add_argument("--memory", type=Path)
    round_parser.add_argument(
        "--base-spec",
        type=Path,
        default=REPOSITORY_ROOT / "specs" / "core-experiment-fast-v1.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return freeze_batch(args) if args.command == "freeze-batch" else freeze_round(args)


if __name__ == "__main__":
    raise SystemExit(main())
