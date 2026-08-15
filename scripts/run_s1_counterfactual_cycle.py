#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.evaluation.core_fast import (  # noqa: E402
    CoreFastEngine,
    FastPathError,
    load_core_fast_spec,
)
from skillchain.evaluation.core_fast.adapters import load_adapter  # noqa: E402
from skillchain.evaluation.core_fast.models import CAPABILITIES, StageDecision  # noqa: E402
from skillchain.evaluation.core_fast.store import atomic_write_json, load_json  # noqa: E402
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_or_verify(path: Path, payload: object, *, label: str) -> None:
    if path.exists():
        if load_json(path) != payload:
            raise FastPathError(f"{label} differs on resume")
        return
    atomic_write_json(path, payload)


def _load_cycle(path: Path) -> dict[str, object]:
    cycle = load_json(path)
    if not isinstance(cycle, dict):
        raise FastPathError("cycle definition is invalid")
    receipt = cycle.get("receipt_sha256")
    unsigned = {key: value for key, value in cycle.items() if key != "receipt_sha256"}
    if receipt != sha256_bytes(canonical_json_bytes(unsigned)):
        raise FastPathError("cycle definition receipt SHA-256 drifted")
    if cycle.get("kind") not in {
        "core-fast-s1-counterfactual-cycle-definition",
        "core-fast-s1-adaptive-campaign-definition",
    }:
        raise FastPathError("cycle definition kind differs")
    budget = cycle.get("budget_estimate")
    if (
        not isinstance(budget, dict)
        or budget.get("within_cny10") is not True
        or float(budget.get("projected_total_dashscope_cny", 11.0)) > 10.0
    ):
        raise FastPathError("cycle definition does not prove the CNY 10 budget")
    return cycle


def _recorded_cost(root: Path) -> float:
    total = 0.0
    if not root.exists():
        return total
    for path in root.rglob("*.result.json"):
        row = load_json(path)
        if isinstance(row, dict):
            value = row.get("cost_cny")
            if isinstance(value, (int, float)):
                total += float(value)
    return total


def _require_remaining_budget(
    cycle: dict[str, object], *, cycle_root: Path, remaining_assistant_outer: int
) -> None:
    rounds = cycle.get("round_specs")
    assert isinstance(rounds, dict)
    budget = cycle["budget_estimate"]
    assert isinstance(budget, dict)
    observed = float(budget["parent_opt800_observed_cny"])
    finalization_rounds = cycle.get("finalization_round_ids", ["r31", "r32"])
    if not isinstance(finalization_rounds, list):
        raise FastPathError("cycle finalization round IDs are invalid")
    for round_id in finalization_rounds:
        row = rounds.get(round_id)
        if isinstance(row, dict):
            observed += _recorded_cost(Path(str(row["run_root"])))
    observed += _recorded_cost(cycle_root / "runs" / "r33-fanin")
    projected = observed + remaining_assistant_outer * float(
        budget["assistant_cny_per_outer_ceiling"]
    )
    if projected > float(cycle["dashscope_stage_budget_cny"]):
        raise FastPathError(
            f"remaining counterfactual cycle estimate exceeds CNY 10: {projected:.6f}"
        )


def _load_engine(
    cycle: dict[str, object], round_id: str, output_root: Path
) -> CoreFastEngine:
    rounds = cycle.get("round_specs")
    if not isinstance(rounds, dict) or not isinstance(rounds.get(round_id), dict):
        raise FastPathError(f"cycle lacks frozen {round_id} spec")
    row = rounds[round_id]
    assert isinstance(row, dict)
    spec_path = Path(str(row["path"]))
    if _sha(spec_path) != row.get("sha256"):
        raise FastPathError(f"frozen {round_id} spec SHA-256 drifted")
    spec = load_core_fast_spec(spec_path)
    adapter = load_adapter(spec, cwd=REPOSITORY_ROOT, spec_path=spec_path)
    return CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=output_root,
        adapter=adapter,
    )


def _load_accepted_branch(
    cycle: dict[str, object], round_id: str
) -> dict[str, object] | None:
    rounds = cycle["round_specs"]
    assert isinstance(rounds, dict) and isinstance(rounds[round_id], dict)
    run_root = Path(str(rounds[round_id]["run_root"]))
    decision_path = run_root / "decisions" / "s1.json"
    if not decision_path.exists():
        return None
    try:
        decision = StageDecision.model_validate_json(
            decision_path.read_bytes(), strict=True
        )
    except ValueError as error:
        raise FastPathError(f"{round_id} decision is invalid") from error
    if not decision.accepted:
        return None
    artifact_path = run_root / "accepted-branch.json"
    bank_path = run_root / "banks" / "s1-selected.json"
    artifact = load_json(artifact_path)
    bank = StaticBankArtifact.model_validate_json(bank_path.read_bytes(), strict=True)
    if not isinstance(artifact, dict):
        raise FastPathError(f"{round_id} accepted branch artifact is invalid")
    receipt = artifact.get("receipt_sha256")
    unsigned = {
        key: value for key, value in artifact.items() if key != "receipt_sha256"
    }
    expected_spec = rounds[round_id]
    assert isinstance(expected_spec, dict)
    if (
        receipt != sha256_bytes(canonical_json_bytes(unsigned))
        or artifact.get("kind") != "core-fast-s1-accepted-branch"
        or artifact.get("round_id") != round_id
        or artifact.get("parent_bank_sha256") != cycle.get("parent_bank_sha256")
        or artifact.get("candidate_bank_sha256") != bank.bank_sha256
        or decision.selected_bank != bank.bank_sha256
        or decision.alias_of is not None
        or artifact.get("capability") != expected_spec.get("target_capability")
        or artifact.get("surface") != expected_spec.get("target_surface")
    ):
        raise FastPathError(f"{round_id} accepted branch lineage drifted")
    return {
        "round_id": round_id,
        "run_root": str(run_root),
        "artifact_path": str(artifact_path),
        "artifact_file_sha256": _sha(artifact_path),
        "bank_path": str(bank_path),
        "bank_file_sha256": _sha(bank_path),
        "artifact": artifact,
        "bank": bank,
    }


def _compose(
    parent: StaticBankArtifact, branches: list[dict[str, object]]
) -> StaticBankArtifact:
    parent_skills = {item.capability_id: item for item in parent.skills}
    selected = dict(parent_skills)
    capabilities: list[str] = []
    for branch in branches:
        artifact = branch["artifact"]
        bank = branch["bank"]
        assert isinstance(artifact, dict) and isinstance(bank, StaticBankArtifact)
        capability = str(artifact["capability"])
        if capability in capabilities:
            raise FastPathError("fan-in branches overlap one capability")
        capabilities.append(capability)
        branch_skills = {item.capability_id: item for item in bank.skills}
        for other in CAPABILITIES:
            if other == capability:
                continue
            if canonical_json_bytes(
                branch_skills[other].model_dump(mode="json")
            ) != canonical_json_bytes(parent_skills[other].model_dump(mode="json")):
                raise FastPathError(
                    f"fan-in branch changed protected capability: {other}"
                )
        selected[capability] = branch_skills[capability]
    payload = parent.model_dump(mode="json")
    payload["construction_identity_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                "policy_version": "s1-counterfactual-accepted-branch-fanin-v1",
                "parent_bank_sha256": parent.bank_sha256,
                "branches": [
                    {
                        "round_id": branch["round_id"],
                        "artifact_file_sha256": branch["artifact_file_sha256"],
                    }
                    for branch in branches
                ],
            }
        )
    )
    payload["skills"] = [
        selected[item.capability_id].model_dump(mode="json") for item in parent.skills
    ]
    payload.pop("bank_sha256", None)
    payload["bank_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return StaticBankArtifact.model_validate(payload, strict=True)


def _rank_key(branch: dict[str, object]) -> tuple[float, float, float, float, str]:
    artifact = branch["artifact"]
    assert isinstance(artifact, dict)
    body = artifact["body_gate"]
    replay = artifact["replay_gate"]
    assert isinstance(body, dict) and isinstance(replay, dict)
    return (
        -1.0,
        -float(body["macro_delta_pp"]),
        -float(replay["macro_delta_pp"]),
        float(body["hard_error_delta_pp"]),
        str(branch["round_id"]),
    )


def finalize(args: argparse.Namespace) -> int:
    cycle_path = args.cycle_definition.resolve()
    cycle = _load_cycle(cycle_path)
    cycle_root = cycle_path.parent
    finalization_rounds = cycle.get("finalization_round_ids", ["r31", "r32"])
    if not isinstance(finalization_rounds, list) or not finalization_rounds:
        raise FastPathError("cycle has no frozen finalization rounds")
    accepted = [
        branch
        for raw_round_id in finalization_rounds
        if (branch := _load_accepted_branch(cycle, str(raw_round_id))) is not None
    ]
    _require_remaining_budget(
        cycle,
        cycle_root=cycle_root,
        remaining_assistant_outer=(
            1150 if len(accepted) == 2 else 600 if accepted else 0
        ),
    )
    engine = _load_engine(
        cycle,
        str(finalization_rounds[0]),
        cycle_root / "runs" / "r33-fanin",
    )
    parent = engine.s1_parent_bank()
    r33_manifest = {
        "schema_version": 1,
        "kind": "core-fast-s1-counterfactual-r33",
        "cycle_definition_sha256": _sha(cycle_path),
        "parent_bank_sha256": parent.bank_sha256,
        "accepted_branch_artifact_sha256s": [
            branch["artifact_file_sha256"] for branch in accepted
        ],
        "creator_calls": 0,
    }
    _write_or_verify(
        engine.output_root / "manifest.json",
        r33_manifest,
        label="R33 manifest",
    )
    candidates: list[dict[str, object]] = list(accepted)
    combination: dict[str, object] | None = None
    if len(accepted) == 2:
        engine.validate(require_runtime=True)
        combined = _compose(parent, accepted)
        engine.output_root.mkdir(parents=True, exist_ok=True)
        combined_path = engine.output_root / "banks" / "r33-combined.json"
        _write_or_verify(
            combined_path,
            combined.model_dump(mode="json"),
            label="R33 combined Bank",
        )
        replay_queries = tuple(
            query
            for query in engine.queries()
            if query.split == "opt_pool"
            and engine.opt_fold_roles()[query.query_id] == "replay"
        )
        parent_replay = engine._assistant_many(  # noqa: SLF001
            split="r33-replay200",
            config="s1-parent",
            queries=replay_queries,
            bank=parent,
        )
        candidate_replay = engine._assistant_many(  # noqa: SLF001
            split="r33-replay200",
            config="s1-combined",
            queries=replay_queries,
            bank=combined,
        )
        replay_ok, replay_reasons, replay_metrics = engine._s1_gate(  # noqa: SLF001
            parent_replay,
            candidate_replay,
            replay_queries,
            phase="replay200",
            treated_capabilities=frozenset(
                str(branch["artifact"]["capability"])
                for branch in accepted  # type: ignore[index]
            ),
        )
        body_ok = False
        body_reasons: tuple[str, ...] = ()
        body_metrics: dict[str, object] | None = None
        if replay_ok:
            body_queries = engine._queries_for_val_gate("body_gate")  # noqa: SLF001
            parent_body = engine._assistant_many(  # noqa: SLF001
                split="r33-body-gate75",
                config="s1-parent",
                queries=body_queries,
                bank=parent,
            )
            candidate_body = engine._assistant_many(  # noqa: SLF001
                split="r33-body-gate75",
                config="s1-combined",
                queries=body_queries,
                bank=combined,
            )
            body_ok, body_reasons, body_metrics = engine._s1_gate(  # noqa: SLF001
                parent_body,
                candidate_body,
                body_queries,
                phase="body_gate75",
                treated_capabilities=frozenset(
                    str(branch["artifact"]["capability"])
                    for branch in accepted  # type: ignore[index]
                ),
            )
        combination = {
            "attempted": True,
            "accepted": replay_ok and body_ok,
            "bank_path": str(combined_path),
            "bank_file_sha256": _sha(combined_path),
            "bank_sha256": combined.bank_sha256,
            "replay_reasons": list(replay_reasons),
            "replay_gate": replay_metrics,
            "body_reasons": list(body_reasons),
            "body_gate": body_metrics,
        }
        if replay_ok and body_ok and body_metrics is not None:
            candidates.append(
                {
                    "round_id": "r33",
                    "run_root": str(engine.output_root),
                    "artifact_path": None,
                    "artifact_file_sha256": None,
                    "bank_path": str(combined_path),
                    "bank_file_sha256": _sha(combined_path),
                    "bank": combined,
                    "artifact": {
                        "capability": "recipe+multi",
                        "replay_gate": replay_metrics,
                        "body_gate": body_metrics,
                    },
                    "coverage": 2,
                }
            )

    if candidates:
        for item in candidates:
            item.setdefault("coverage", 1)
        finalist = min(
            candidates,
            key=lambda item: (
                -int(item["coverage"]),
                *_rank_key(item)[1:],
            ),
        )
        finalist_bank = finalist["bank"]
        assert isinstance(finalist_bank, StaticBankArtifact)
        finalist_payload: dict[str, object] | None = {
            "source_round_id": finalist["round_id"],
            "capability_count": finalist["coverage"],
            "bank_path": finalist["bank_path"],
            "bank_file_sha256": finalist["bank_file_sha256"],
            "bank_sha256": finalist_bank.bank_sha256,
            "body_macro_delta_pp": finalist["artifact"]["body_gate"]["macro_delta_pp"],  # type: ignore[index]
            "replay_macro_delta_pp": finalist["artifact"]["replay_gate"][
                "macro_delta_pp"
            ],  # type: ignore[index]
            "body_hard_error_delta_pp": finalist["artifact"]["body_gate"][
                "hard_error_delta_pp"
            ],  # type: ignore[index]
        }
    else:
        finalist_payload = None
    unsigned = {
        "schema_version": 1,
        "kind": "core-fast-s1-cycle-finalist",
        "cycle_id": cycle["cycle_id"],
        "cycle_definition_sha256": _sha(cycle_path),
        "parent_round_id": "r12",
        "parent_bank_sha256": parent.bank_sha256,
        "accepted_branch_round_ids": [branch["round_id"] for branch in accepted],
        "combination": combination,
        "finalist": finalist_payload,
        "test300_allowed": finalist_payload is not None,
        "ranking_policy": [
            "capability_count_desc",
            "body_macro_desc",
            "replay_macro_desc",
            "hard_error_delta_asc",
            "round_id_asc",
        ],
    }
    receipt = {
        **unsigned,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    receipt_path = cycle_root / "cycle-finalist.json"
    _write_or_verify(receipt_path, receipt, label="cycle finalist receipt")
    print(
        json.dumps(
            {"cycle_finalist": str(receipt_path), "finalist": finalist_payload},
            indent=2,
        )
    )
    return 0


def freeze_adaptive(args: argparse.Namespace) -> int:
    cycle_root = args.cycle_root.resolve()
    spec_path = args.spec.resolve()
    run_root = args.run_root.resolve()
    artifact_path = run_root / "accepted-branch.json"
    decision_path = run_root / "decisions" / "s1.json"
    bank_path = run_root / "banks" / "s1-selected.json"
    artifact = load_json(artifact_path)
    decision = StageDecision.model_validate_json(
        decision_path.read_bytes(), strict=True
    )
    bank = StaticBankArtifact.model_validate_json(bank_path.read_bytes(), strict=True)
    if (
        not isinstance(artifact, dict)
        or artifact.get("kind") != "core-fast-s1-accepted-branch"
        or artifact.get("round_id") != args.round_id
        or artifact.get("parent_round_id") != "r12"
        or artifact.get("parent_bank_sha256")
        != "e70ed907825833a0bbb97ccde068fd38d4a6342824cd9dcdfb543723feb096cd"
        or artifact.get("candidate_bank_sha256") != bank.bank_sha256
        or not decision.accepted
        or decision.selected_bank != bank.bank_sha256
        or decision.alias_of is not None
    ):
        raise FastPathError("adaptive finalist branch is not a valid R12 descendant")
    unsigned = {
        "schema_version": 1,
        "kind": "core-fast-s1-adaptive-campaign-definition",
        "cycle_id": args.cycle_id,
        "parent_round_id": "r12",
        "parent_bank_sha256": artifact["parent_bank_sha256"],
        "dashscope_stage_budget_cny": 10.0,
        "finalization_round_ids": [args.round_id],
        "round_specs": {
            args.round_id: {
                "path": str(spec_path),
                "sha256": _sha(spec_path),
                "run_root": str(run_root),
                "target_capability": artifact["capability"],
                "target_surface": artifact["surface"],
            }
        },
        "budget_estimate": {
            "within_cny10": True,
            "projected_total_dashscope_cny": 2.0,
            "parent_opt800_observed_cny": 0.0,
            "assistant_cny_per_outer_ceiling": 0.01,
        },
        "test300_consumption_policy": "one-create-only-paired-r12-vs-finalist",
    }
    cycle = {
        **unsigned,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    path = cycle_root / "cycle-definition.json"
    _write_or_verify(path, cycle, label="adaptive cycle definition")
    print(json.dumps({"cycle_definition": str(path)}, indent=2))
    return 0


def finalist_test(args: argparse.Namespace) -> int:
    cycle_path = args.cycle_definition.resolve()
    cycle = _load_cycle(cycle_path)
    finalist_path = args.finalist_receipt.resolve()
    finalist_receipt = load_json(finalist_path)
    if not isinstance(finalist_receipt, dict):
        raise FastPathError("cycle finalist receipt is invalid")
    unsigned_finalist = {
        key: value for key, value in finalist_receipt.items() if key != "receipt_sha256"
    }
    finalist = finalist_receipt.get("finalist")
    if (
        finalist_receipt.get("receipt_sha256")
        != sha256_bytes(canonical_json_bytes(unsigned_finalist))
        or finalist_receipt.get("cycle_definition_sha256") != _sha(cycle_path)
        or finalist_receipt.get("test300_allowed") is not True
        or not isinstance(finalist, dict)
    ):
        raise FastPathError("one frozen cycle finalist is required before test300")
    bank_path = Path(str(finalist["bank_path"]))
    if _sha(bank_path) != finalist.get("bank_file_sha256"):
        raise FastPathError("finalist Bank file SHA-256 drifted")
    bank = StaticBankArtifact.model_validate_json(bank_path.read_bytes(), strict=True)
    if bank.bank_sha256 != finalist.get("bank_sha256"):
        raise FastPathError("finalist internal Bank SHA-256 drifted")
    test_root = args.output_root.resolve()
    _require_remaining_budget(
        cycle,
        cycle_root=cycle_path.parent,
        remaining_assistant_outer=600,
    )
    lease_path = cycle_path.parent / "test300-consumption.json"
    lease = {
        "schema_version": 1,
        "kind": "core-fast-s1-finalist-test300-consumption",
        "cycle_id": cycle["cycle_id"],
        "finalist_receipt_sha256": _sha(finalist_path),
        "test_root": str(test_root),
    }
    _write_or_verify(lease_path, lease, label="test300 consumption lease")
    finalization_rounds = cycle.get("finalization_round_ids", ["r31", "r32"])
    if not isinstance(finalization_rounds, list) or not finalization_rounds:
        raise FastPathError("cycle has no frozen engine spec for test300")
    engine = _load_engine(cycle, str(finalization_rounds[0]), test_root)
    engine.validate(require_runtime=True)
    parent = engine.s1_parent_bank()
    test_queries = tuple(
        query for query in engine.queries() if query.split == "test_frozen"
    )
    parent_rows = engine._assistant_many(  # noqa: SLF001
        split="s1-finalist-test300",
        config="s1-parent",
        queries=test_queries,
        bank=parent,
    )
    candidate_rows = engine._assistant_many(  # noqa: SLF001
        split="s1-finalist-test300",
        config="s1-finalist",
        queries=test_queries,
        bank=bank,
    )
    passed, reasons, metrics = engine._s1_gate(  # noqa: SLF001
        parent_rows,
        candidate_rows,
        test_queries,
        phase="test300",
        treated_capabilities=frozenset(
            capability
            for capability in CAPABILITIES
            if next(
                item for item in parent.skills if item.capability_id == capability
            ).skill_sha256
            != next(
                item for item in bank.skills if item.capability_id == capability
            ).skill_sha256
        ),
    )
    selected = bank if passed else parent
    selected_path = test_root / "banks" / "cycle-selected.json"
    _write_or_verify(
        selected_path,
        selected.model_dump(mode="json"),
        label="cycle selected Bank",
    )
    receipt_unsigned = {
        "schema_version": 1,
        "kind": "core-fast-s1-finalist-test300",
        "cycle_id": cycle["cycle_id"],
        "finalist_receipt_sha256": _sha(finalist_path),
        "parent_bank_sha256": parent.bank_sha256,
        "finalist_bank_sha256": bank.bank_sha256,
        "passed": passed,
        "reasons": list(reasons),
        "gate": metrics,
        "selected_bank_sha256": selected.bank_sha256,
        "test300_consumed": True,
        "outer_assistant_query_count": 600,
        "judge_calls": 0,
        "s2_calls": 0,
        "s3_calls": 0,
        "five_config_matrix_run": False,
        "disclosure": (
            "test300 is now consumed by this paired R12-vs-finalist selection and is no "
            "longer an untouched five-configuration test"
        ),
    }
    receipt = {
        **receipt_unsigned,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_unsigned)),
    }
    result_path = test_root / "s1-finalist-test300.json"
    _write_or_verify(result_path, receipt, label="S1 finalist test300 receipt")
    print(
        json.dumps(
            {
                "test_receipt": str(result_path),
                "passed": passed,
                "selected_bank": selected.bank_sha256,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize and test the R12 counterfactual S1 cycle"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-adaptive")
    freeze.add_argument("--cycle-root", type=Path, required=True)
    freeze.add_argument("--cycle-id", required=True)
    freeze.add_argument("--round-id", required=True)
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--run-root", type=Path, required=True)
    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--cycle-definition", type=Path, required=True)
    test = sub.add_parser("s1-finalist-test")
    test.add_argument("--cycle-definition", type=Path, required=True)
    test.add_argument("--finalist-receipt", type=Path, required=True)
    test.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "freeze-adaptive":
            return freeze_adaptive(args)
        return finalize(args) if args.command == "finalize" else finalist_test(args)
    except (FastPathError, OSError, ValueError) as error:
        print(f"S1 counterfactual cycle error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
