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

from skillchain.evaluation.core_fast.models import (  # noqa: E402
    CAPABILITIES,
    CoreFastSpec,
    StageDecision,
    load_core_fast_spec,
)
from skillchain import config  # noqa: E402
from skillchain.evaluation.core_fast.store import atomic_write_json  # noqa: E402
from skillchain.tools.serialization import sha256_bytes  # noqa: E402


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _create_only(path: Path, payload: object) -> None:
    if path.exists():
        raise ValueError(f"destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _source_binding(
    *,
    source_spec_path: Path,
    source_root: Path,
    source_stage: str,
    source_round_id: str,
    route_results_path: Path,
    preparatory_binding_path: Path | None,
) -> tuple[CoreFastSpec, dict[str, object]]:
    source_spec = load_core_fast_spec(source_spec_path)
    manifest_path = source_root / "manifest.json"
    decision_path = source_root / "decisions" / f"{source_stage}.json"
    bank_path = source_root / "banks" / f"{source_stage}-selected.json"
    if not all(path.is_file() for path in (manifest_path, decision_path, bank_path)):
        raise ValueError("source root lacks its manifest, decision, or selected Bank")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = StageDecision.model_validate_json(
        decision_path.read_bytes(), strict=True
    )
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    settings = manifest.get(f"{source_stage}_settings")
    if (
        manifest.get("input_sha256", {}).get("spec") != _sha(source_spec_path)
        or decision.stage != source_stage
        or not decision.accepted
        or decision.alias_of is not None
        or decision.metrics.get("round_id") != source_round_id
        or not isinstance(settings, dict)
        or settings.get("round_id") != source_round_id
        or decision.selected_bank != bank.get("bank_sha256")
    ):
        raise ValueError("source stage is not an accepted, SHA-bound round")
    if source_stage == "s1":
        if preparatory_binding_path is None:
            raise ValueError("initial S2 round requires --preparatory-binding")
        preparatory_binding_path = preparatory_binding_path.resolve()
        authorization = json.loads(preparatory_binding_path.read_text(encoding="utf-8"))
        source_s1 = authorization.get("source_s1", {})
        if (
            authorization.get("kind") != "core-fast-s2-preparatory-branch"
            or authorization.get("status") != "authorized-pre-s2-runtime-ready"
            or source_s1.get("round_id") != source_round_id
            or source_s1.get("bank_sha256") != decision.selected_bank
            or source_s1.get("bank_file_sha256") != _sha(bank_path)
            or source_s1.get("decision_file_sha256") != _sha(decision_path)
            or source_s1.get("manifest_file_sha256") != _sha(manifest_path)
        ):
            raise ValueError("S1 source is not the authorized S2 preparatory branch")
        preparatory = {
            "preparatory_binding_path": str(preparatory_binding_path),
            "preparatory_binding_file_sha256": _sha(preparatory_binding_path),
            "preparatory_round_id": source_round_id,
            "preparatory_bank_sha256": decision.selected_bank,
        }
    else:
        inherited = source_spec.s2_parent
        if inherited is None:
            raise ValueError("accepted S2 source lacks its preparatory authorization")
        preparatory = {
            "preparatory_binding_path": inherited.preparatory_binding_path,
            "preparatory_binding_file_sha256": (
                inherited.preparatory_binding_file_sha256
            ),
            "preparatory_round_id": inherited.preparatory_round_id,
            "preparatory_bank_sha256": inherited.preparatory_bank_sha256,
        }
    return source_spec, {
        "source_stage": source_stage,
        "source_round_id": source_round_id,
        "bank_path": str(bank_path.resolve()),
        "bank_file_sha256": _sha(bank_path),
        "bank_sha256": decision.selected_bank,
        "decision_path": str(decision_path.resolve()),
        "decision_file_sha256": _sha(decision_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": _sha(manifest_path),
        **preparatory,
        "route_results_path": str(route_results_path.resolve()),
        "route_results_sha256": None,
    }


def _bootstrap_spec(args: argparse.Namespace) -> dict[str, object]:
    source_spec_path = args.source_spec.resolve()
    source_root = args.source_root.resolve()
    source_spec, binding = _source_binding(
        source_spec_path=source_spec_path,
        source_root=source_root,
        source_stage=args.source_stage,
        source_round_id=args.source_round_id,
        route_results_path=args.route_results_path,
        preparatory_binding_path=args.preparatory_binding,
    )
    memory = tuple(json.loads(item) for item in args.memory_json)
    if any(not isinstance(item, dict) for item in memory):
        raise ValueError("every --memory-json value must encode one object")
    payload = source_spec.model_dump(mode="json")
    payload["experiment_id"] = args.experiment_id
    if args.route_model_qualification is not None:
        payload["models"]["route_only"]["requested_model"] = (
            args.route_model_qualification
        )
        payload["models"]["route_only"]["moving_alias"] = False
        payload["concurrency"]["assistant"] = (
            config.QWEN35_ROUTE_QUALIFICATION_CONCURRENCY
        )
        payload["concurrency"]["assistant_requests_per_second"] = (
            config.QWEN35_ROUTE_QUALIFICATION_REQUESTS_PER_SECOND
        )
    payload["s2_parent"] = binding
    payload["s2_settings"] = {
        "round_id": args.round_id,
        "cycle_id": args.cycle_id,
        "target_capability": args.target_capability,
        "target_predicted_capability": args.target_predicted_capability,
        "proposal_mode": args.proposal_mode,
        "failure_example_count": 3,
        "parent_success_example_count": 3,
        "historical_regression_example_count": 3,
        "historical_regression_query_ids": sorted(
            set(args.historical_regression_query_id)
        ),
        "prior_experiment_memory": list(memory),
    }
    payload["disclosures"] = [
        *payload["disclosures"],
        "Adaptive S2 is Description-only, one capability per independent round.",
        "The parent route800 must be frozen before Creator or candidate calls.",
        "Adaptive S2 specs may run only through s2; S3/Judge/test remain sealed.",
    ]
    if args.route_model_qualification is not None:
        payload["disclosures"].append(
            "Qualification-only route model differs from the full Assistant; "
            "this spec may create parent route800 evidence but cannot run S2."
        )
    return CoreFastSpec.model_validate_json(
        json.dumps(payload), strict=True
    ).model_dump(mode="json")


def _freeze_spec(args: argparse.Namespace) -> dict[str, object]:
    bootstrap_spec = load_core_fast_spec(args.bootstrap_spec.resolve())
    if bootstrap_spec.s2_parent is None or bootstrap_spec.s2_settings is None:
        raise ValueError("bootstrap spec lacks adaptive S2 bindings")
    receipt_path = args.route_bootstrap.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    route_path = Path(bootstrap_spec.s2_parent.route_results_path).resolve()
    if (
        receipt.get("kind") != "core-fast-s2-parent-route800-bootstrap"
        or receipt.get("row_count") != 800
        or receipt.get("source_stage") != bootstrap_spec.s2_parent.source_stage
        or receipt.get("source_round_id") != bootstrap_spec.s2_parent.source_round_id
        or receipt.get("parent_bank_sha256") != bootstrap_spec.s2_parent.bank_sha256
        or Path(str(receipt.get("route_results_path"))).resolve() != route_path
        or not route_path.is_file()
        or receipt.get("route_results_sha256") != _sha(route_path)
    ):
        raise ValueError("route800 bootstrap does not bind the S2 parent spec")
    payload = bootstrap_spec.model_dump(mode="json")
    payload["s2_parent"]["route_results_sha256"] = receipt["route_results_sha256"]
    return CoreFastSpec.model_validate_json(
        json.dumps(payload), strict=True
    ).model_dump(mode="json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare create-only adaptive S2 round specs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser(
        "bootstrap-spec", help="bind an accepted S1/S2 parent before route800"
    )
    bootstrap.add_argument("--source-spec", type=Path, required=True)
    bootstrap.add_argument("--source-root", type=Path, required=True)
    bootstrap.add_argument("--source-stage", choices=("s1", "s2"), required=True)
    bootstrap.add_argument("--source-round-id", required=True)
    bootstrap.add_argument("--preparatory-binding", type=Path)
    bootstrap.add_argument("--route-results-path", type=Path, required=True)
    bootstrap.add_argument("--output-spec", type=Path, required=True)
    bootstrap.add_argument("--experiment-id", required=True)
    bootstrap.add_argument("--cycle-id", required=True)
    bootstrap.add_argument("--round-id", required=True)
    bootstrap.add_argument("--target-capability", choices=CAPABILITIES, required=True)
    bootstrap.add_argument(
        "--route-model-qualification",
        choices=(config.QWEN35_ROUTE_QUALIFICATION_MODEL,),
        help=(
            "freeze an isolated route-only model profile; S2 remains blocked "
            "until the full Assistant uses the same model"
        ),
    )
    bootstrap.add_argument("--target-predicted-capability", choices=CAPABILITIES)
    bootstrap.add_argument(
        "--proposal-mode",
        choices=(
            "single-description-counterfactual-v1",
            "contrastive-description-ir-v2",
        ),
        default="single-description-counterfactual-v1",
    )
    bootstrap.add_argument(
        "--historical-regression-query-id", action="append", default=[]
    )
    bootstrap.add_argument("--memory-json", action="append", default=[])
    freeze = commands.add_parser(
        "freeze-spec", help="bind the completed parent route800 SHA"
    )
    freeze.add_argument("--bootstrap-spec", type=Path, required=True)
    freeze.add_argument("--route-bootstrap", type=Path, required=True)
    freeze.add_argument("--output-spec", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = (
            _bootstrap_spec(args)
            if args.command == "bootstrap-spec"
            else _freeze_spec(args)
        )
        _create_only(args.output_spec.resolve(), payload)
        print(
            json.dumps(
                {
                    "status": "created",
                    "command": args.command,
                    "output_spec": str(args.output_spec.resolve()),
                    "round_id": payload["s2_settings"]["round_id"],
                    "source_round_id": payload["s2_parent"]["source_round_id"],
                    "route_results_sha256": payload["s2_parent"][
                        "route_results_sha256"
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"S2 adaptive preparation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
