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

from skillchain.evaluation.core_fast.models import (  # noqa: E402
    AssistantObservation,
    CoreFastSpec,
)
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes  # noqa: E402


CYCLE_ID = "s1-counterfactual-v1"
R12_ROOT = Path(
    r"D:\athena\experiment-runs\portfolio-core-dual-policy-campaign-20260814-v2"
    r"\runs\r12-counterfactual-rule"
)
R12_BANK_SHA = "e70ed907825833a0bbb97ccde068fd38d4a6342824cd9dcdfb543723feb096cd"
R12_BANK_FILE_SHA = "edf83d288fcbb67b78b53a6390b713f845238836b7fc4f3378aca34c23bf9489"
R12_DECISION_FILE_SHA = (
    "1edc7fba7f05ee3deffc8a41907931a76e660fba5058262e3251156957923bc4"
)
R12_MANIFEST_FILE_SHA = (
    "7c24fd2192784ff2b33d2975557d34120c91e39a36aea2a0fa5cca09370afdc8"
)
R12_STYLE_SHA = "a5c6c53fa2778d1237bc5fe7ba652205bacdf1616090e91e6ac44f85af88b5ef"
R12_PROTECTION_IDS = tuple(
    sorted(
        f"r2-core-{suffix}"
        for suffix in ("0261", "0263", "0265", "0361", "0436", "0440", "1012", "1161")
    )
)


ROUND_DEFINITIONS = {
    "r31": {
        "capability": "utility.recipe_guidance",
        "surface": "action-policy",
        "gain_seeds": tuple(
            sorted(
                f"r2-core-{suffix}"
                for suffix in ("0799", "0273", "0272", "0448", "0674")
            )
        ),
        "regressions": tuple(
            sorted(f"r2-core-{suffix}" for suffix in ("0447", "0975", "0374"))
        ),
        "memory": (
            {
                "capability": "utility.recipe_guidance",
                "hypothesis": "counterfactual action policy for the stable Recipe gain cluster",
                "historical_signal": "R8 gained seven parent failures without a parent-success regression",
                "forbidden_carry_forward": "no rejected R17/R26/R29 Skill text or Bank",
            },
        ),
    },
    "r32": {
        "capability": "product.multi_search",
        "surface": "response-policy",
        "gain_seeds": tuple(sorted(f"r2-core-{suffix}" for suffix in ("0358", "0258"))),
        "regressions": tuple(
            sorted(f"r2-core-{suffix}" for suffix in ("0758", "0784", "0682"))
        ),
        "memory": (
            {
                "capability": "product.multi_search",
                "hypothesis": "counterfactual evidence-to-item association response policy",
                "historical_signal": "Multi response gains repeated but broad rewrites regressed protected states",
                "forbidden_carry_forward": "no rejected R17/R26/R29 Skill text or Bank",
            },
        ),
    },
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_create_only(path: Path, payload: object) -> None:
    content = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"create-only artifact differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, dict):
        raise ValueError(f"JSON object required: {path}")
    return raw


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"JSONL object required: {path}")
            rows.append(raw)
    return rows


def _resolve_spec_path(base_spec_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_spec_path.parent / path).resolve()


def _verify_parent_opt800(
    *,
    opt_path: Path,
    base: dict[str, object],
    queries: list[dict[str, object]],
) -> None:
    rows = _read_jsonl(opt_path)
    observations = [
        AssistantObservation.model_validate(row, strict=True) for row in rows
    ]
    expected_ids = {
        str(row["query_id"]) for row in queries if row.get("split") == "opt_pool"
    }
    if (
        len(observations) != 800
        or len({row.query_id for row in observations}) != 800
        or {row.query_id for row in observations} != expected_ids
    ):
        raise ValueError("R12 parent opt800 does not cover the fixed opt800 exactly")
    models = base.get("models")
    runtime = base.get("runtime")
    if not isinstance(models, dict) or not isinstance(runtime, dict):
        raise ValueError("base spec model/runtime identity is missing")
    assistant = models.get("assistant")
    if not isinstance(assistant, dict):
        raise ValueError("base spec Assistant identity is missing")
    expected_model = assistant.get("requested_model")
    expected_contract = runtime.get("assistant_contract")
    request_ids: list[str] = []
    for observation in observations:
        context = observation.replay_context
        result = context.get("assistant_result")
        response = context.get("response")
        receipt = context.get("receipt")
        if not all(isinstance(item, dict) for item in (result, response, receipt)):
            raise ValueError(
                f"R12 parent opt800 evidence is missing: {observation.query_id}"
            )
        assert (
            isinstance(result, dict)
            and isinstance(response, dict)
            and isinstance(receipt, dict)
        )
        if (
            result.get("bank_sha256") != R12_BANK_SHA
            or result.get("backbone_model") != expected_model
            or response.get("backbone_model") != expected_model
            or observation.assistant_contract != expected_contract
        ):
            raise ValueError(
                f"R12 parent opt800 identity differs: {observation.query_id}"
            )
        calls = receipt.get("model_calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(
                f"R12 parent opt800 call receipt is missing: {observation.query_id}"
            )
        for call in calls:
            if (
                not isinstance(call, dict)
                or call.get("requested_model") != expected_model
                or call.get("response_model") != expected_model
                or not isinstance(call.get("provider_request_id"), str)
                or not call["provider_request_id"]
            ):
                raise ValueError(
                    f"R12 parent opt800 model call differs: {observation.query_id}"
                )
            request_ids.append(str(call["provider_request_id"]))
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("R12 parent opt800 provider request IDs are not unique")


def _verify_r12() -> dict[str, object]:
    bank_path = R12_ROOT / "banks" / "s1-selected.json"
    decision_path = R12_ROOT / "decisions" / "s1.json"
    manifest_path = R12_ROOT / "manifest.json"
    expected = (
        (bank_path, R12_BANK_FILE_SHA),
        (decision_path, R12_DECISION_FILE_SHA),
        (manifest_path, R12_MANIFEST_FILE_SHA),
    )
    for path, digest in expected:
        if _sha(path) != digest:
            raise ValueError(f"R12 source artifact SHA drifted: {path}")
    bank = StaticBankArtifact.model_validate_json(bank_path.read_bytes(), strict=True)
    if bank.bank_sha256 != R12_BANK_SHA:
        raise ValueError("R12 internal Bank SHA drifted")
    style = next(
        item
        for item in bank.skills
        if item.capability_id == "product.style_recommendation"
    )
    if style.skill_sha256 != R12_STYLE_SHA:
        raise ValueError("R12 protected Style Skill SHA drifted")
    decision = _load_json(decision_path)
    metrics = decision.get("metrics")
    if (
        decision.get("accepted") is not True
        or decision.get("alias_of") is not None
        or decision.get("selected_bank") != R12_BANK_SHA
        or not isinstance(metrics, dict)
        or metrics.get("round_id") != "r12"
    ):
        raise ValueError("R12 source decision is not the accepted selected parent")
    return {
        "source_round_id": "r12",
        "bank_path": str(bank_path),
        "bank_file_sha256": R12_BANK_FILE_SHA,
        "bank_sha256": R12_BANK_SHA,
        "decision_path": str(decision_path),
        "decision_file_sha256": R12_DECISION_FILE_SHA,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": R12_MANIFEST_FILE_SHA,
        "parent_protection_query_ids": list(R12_PROTECTION_IDS),
        "protected_skill_sha256": {"product.style_recommendation": R12_STYLE_SHA},
    }


def _round_spec(
    base: dict[str, object],
    *,
    round_id: str,
    parent_binding: dict[str, object],
    fixed_samples: dict[str, object] | None = None,
) -> dict[str, object]:
    definition = ROUND_DEFINITIONS[round_id]
    capability = str(definition["capability"])
    payload = json.loads(json.dumps(base))
    payload["experiment_id"] = f"{CYCLE_ID}-{round_id}-{capability.replace('.', '-')}"
    payload["s1_parent"] = parent_binding
    if fixed_samples is not None:
        payload["fixed_samples"] = fixed_samples
    payload["s1_settings"] = {
        "round_id": round_id,
        "feedback_mode": "fresh-per-round",
        "feedback_total_count": 9,
        "feedback_canary_count": 3,
        "feedback_format_retry_limit": 1,
        "feedback_selection_policy": "parent-counterfactual-v6",
        "feedback_allocation": "target-focused",
        "target_capabilities": [capability],
        "proposal_mode": "single-surface-counterfactual-fanout-v4",
        "max_patched_capabilities": 1,
        "protected_capabilities": sorted(
            item for item in payload["capabilities"] if item != capability
        ),
        "creator_directives": [],
        "required_patch_phrases": {},
        "bounded_edit_surface": "author-fields",
        "prior_experiment_memory": {capability: list(definition["memory"])},
        "cycle_id": CYCLE_ID,
        "target_surface": definition["surface"],
        "counterfactual_gain_seed_query_ids": list(definition["gain_seeds"]),
        "counterfactual_regression_query_ids": list(definition["regressions"]),
        "parent_protection_query_ids": list(R12_PROTECTION_IDS),
    }
    payload["limits"] = {
        **payload["limits"],
        "max_feedback_calls": 18,
        "max_creator_calls": 3,
    }
    payload["disclosures"] = [
        *payload["disclosures"],
        "This forward-only cycle uses accepted R12 as its only legal parent; rejected R17/R26/R29 Banks and Skills are excluded.",
        "R31 and R32 are frozen together before provider calls and each patches one capability and one policy surface.",
        "body75 is a fixed previously observed gate; mechanism design cannot change after either round, and test300 is consumed once by one frozen finalist.",
    ]
    return CoreFastSpec.model_validate_json(
        canonical_json_bytes(payload), strict=True
    ).model_dump(mode="json")


def bootstrap(args: argparse.Namespace) -> int:
    parent = _verify_r12()
    cycle_root = args.cycle_root.resolve()
    parent_opt_path = cycle_root / "artifacts" / "r12-parent-opt800.jsonl"
    parent_binding = {
        **parent,
        "opt_results_path": str(parent_opt_path),
        "opt_results_sha256": None,
    }
    base = _load_json(args.base_spec.resolve())
    spec = _round_spec(base, round_id="r31", parent_binding=parent_binding)
    output = cycle_root / "specs" / "s1-parent-opt800-bootstrap.json"
    _write_create_only(output, spec)
    print(
        json.dumps(
            {
                "spec": str(output),
                "run_root": str(cycle_root / "runs" / "parent-opt800"),
            },
            indent=2,
        )
    )
    return 0


def freeze_cycle(args: argparse.Namespace) -> int:
    parent = _verify_r12()
    cycle_root = args.cycle_root.resolve()
    bootstrap_receipt = _load_json(args.bootstrap_receipt.resolve())
    if (
        bootstrap_receipt.get("kind") != "core-fast-s1-parent-opt800-bootstrap"
        or bootstrap_receipt.get("source_round_id") != "r12"
        or bootstrap_receipt.get("parent_bank_sha256") != R12_BANK_SHA
        or bootstrap_receipt.get("row_count") != 800
    ):
        raise ValueError("S1 parent opt800 bootstrap identity differs")
    opt_path = Path(str(bootstrap_receipt["opt_results_path"]))
    opt_sha = str(bootstrap_receipt["opt_results_sha256"])
    if _sha(opt_path) != opt_sha:
        raise ValueError("S1 parent opt800 file differs from bootstrap")
    fixed_samples = bootstrap_receipt.get("fixed_samples")
    if not isinstance(fixed_samples, dict):
        raise ValueError("S1 parent opt800 bootstrap lacks fixed samples")
    parent_binding = {
        **parent,
        "opt_results_path": str(opt_path),
        "opt_results_sha256": opt_sha,
    }
    base_spec_path = args.base_spec.resolve()
    base = _load_json(base_spec_path)
    paths = base.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("base spec lacks paths")
    queries = _read_jsonl(_resolve_spec_path(base_spec_path, str(paths["queries"])))
    _verify_parent_opt800(opt_path=opt_path, base=base, queries=queries)
    fold_rows = _read_jsonl(
        _resolve_spec_path(base_spec_path, str(paths["opt_fold_mapping"]))
    )
    replay_ids = {
        str(row["query_id"]) for row in fold_rows if row.get("role") == "replay"
    }
    replay_target_counts = {
        round_id: sum(
            row.get("query_id") in replay_ids
            and row.get("canonical_capability") == definition["capability"]
            for row in queries
        )
        for round_id, definition in ROUND_DEFINITIONS.items()
    }
    specs: dict[str, dict[str, object]] = {}
    spec_paths: dict[str, Path] = {}
    for round_id in ("r31", "r32"):
        spec = _round_spec(
            base,
            round_id=round_id,
            parent_binding=parent_binding,
            fixed_samples=fixed_samples,
        )
        path = cycle_root / "specs" / f"{round_id}.json"
        specs[round_id] = spec
        spec_paths[round_id] = path
    for round_id in ("r31", "r32"):
        _write_create_only(spec_paths[round_id], specs[round_id])
    assistant_outer_remaining = (
        sum(
            2 * (replay_target_counts[round_id] + 3) + 400 + 150
            for round_id in ("r31", "r32")
        )
        + 550
        + 600
    )
    parent_observed = float(bootstrap_receipt.get("observed_dashscope_cost_cny", 0.0))
    budget_estimate = parent_observed + assistant_outer_remaining * 0.0025 + 36 * 0.007
    if budget_estimate > 10.0:
        raise ValueError(
            f"counterfactual cycle worst-case DashScope estimate exceeds CNY 10: {budget_estimate:.6f}"
        )
    receipt_unsigned = {
        "schema_version": 1,
        "kind": "core-fast-s1-counterfactual-cycle-definition",
        "cycle_id": CYCLE_ID,
        "parent_round_id": "r12",
        "parent_bank_sha256": R12_BANK_SHA,
        "parent_opt_sha256": opt_sha,
        "round_order": ["r31", "r32", "r33"],
        "round_specs": {
            round_id: {
                "path": str(spec_paths[round_id]),
                "sha256": sha256_bytes(canonical_json_bytes(specs[round_id])),
                "run_root": str(cycle_root / "runs" / round_id),
                "target_capability": ROUND_DEFINITIONS[round_id]["capability"],
                "target_surface": ROUND_DEFINITIONS[round_id]["surface"],
            }
            for round_id in ("r31", "r32")
        },
        "test_policy": "one-frozen-finalist-paired-test300-v1",
        "maximum_creator_sessions": 2,
        "dashscope_stage_budget_cny": 10.0,
        "budget_estimate": {
            "basis": "observed_parent_plus_buffered_empirical_outer_call_ceiling",
            "parent_opt800_observed_cny": parent_observed,
            "replay_target_upper_counts": replay_target_counts,
            "remaining_assistant_outer_call_ceiling": assistant_outer_remaining,
            "assistant_cny_per_outer_ceiling": 0.0025,
            "feedback_attempt_ceiling": 36,
            "feedback_cny_per_attempt_ceiling": 0.007,
            "projected_total_dashscope_cny": budget_estimate,
            "within_cny10": True,
        },
    }
    receipt = {
        **receipt_unsigned,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_unsigned)),
    }
    receipt_path = cycle_root / "cycle-definition.json"
    _write_create_only(receipt_path, receipt)
    print(
        json.dumps(
            {
                "cycle_definition": str(receipt_path),
                "round_specs": {key: str(value) for key, value in spec_paths.items()},
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the R12 counterfactual S1 cycle"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("bootstrap", "freeze-cycle"):
        item = sub.add_parser(command)
        item.add_argument(
            "--base-spec",
            type=Path,
            default=REPOSITORY_ROOT / "specs" / "core-experiment-fast-v1.json",
        )
        item.add_argument("--cycle-root", type=Path, required=True)
        if command == "freeze-cycle":
            item.add_argument("--bootstrap-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return bootstrap(args) if args.command == "bootstrap" else freeze_cycle(args)


if __name__ == "__main__":
    raise SystemExit(main())
