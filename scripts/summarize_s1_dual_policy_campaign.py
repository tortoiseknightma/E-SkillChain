"""Build a canonical summary from completed dual-policy S1 rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillchain.evaluation.core_fast.models import CallResult, StageDecision
from skillchain.tools.serialization import canonical_json_bytes


def _cost_and_calls(root: Path) -> tuple[int, float]:
    calls = 0
    cost = 0.0
    for path in (root / "calls").glob("*/*.result.json"):
        result = CallResult.model_validate_json(path.read_bytes(), strict=True)
        calls += 1
        cost += result.cost_cny
    return calls, cost


def summarize(campaign_root: Path) -> dict[str, object]:
    rounds: list[dict[str, object]] = []
    for decision_path in sorted(
        campaign_root.glob("runs/*/decisions/s1.json"),
        key=lambda path: int(path.parents[1].name.split("-", 1)[0][1:]),
    ):
        decision = StageDecision.model_validate_json(
            decision_path.read_bytes(), strict=True
        )
        root = decision_path.parents[1]
        calls, cost = _cost_and_calls(root)
        branches = []
        for branch in decision.metrics.get("fanout_branches", []):
            if not isinstance(branch, dict):
                continue
            surfaces = []
            for item in branch.get("surface_branches", []):
                if not isinstance(item, dict) or not isinstance(
                    item.get("screen"), dict
                ):
                    continue
                screen = item["screen"]
                surfaces.append(
                    {
                        "surface": item["surface"],
                        "decision": screen["decision"],
                        "gains": screen["gain_count"],
                        "regressions": screen["regression_count"],
                        "net_gain": screen["net_gain"],
                    }
                )
            if surfaces:
                branches.append(
                    {"capability": branch["capability"], "surfaces": surfaces}
                )
        replay = decision.metrics.get("replay_gate")
        body = decision.metrics.get("gate")
        rounds.append(
            {
                "round_id": decision.metrics.get("round_id"),
                "run_name": root.name,
                "accepted": decision.accepted,
                "reasons": list(decision.reasons),
                "candidate_bank": decision.candidate_bank,
                "selected_bank": decision.selected_bank,
                "retained_capabilities": decision.metrics.get(
                    "retained_patch_capabilities", []
                ),
                "feedback_provider_calls": decision.metrics.get(
                    "feedback_provider_calls_this_round", 0
                ),
                "creator_calls": decision.metrics.get("fanout_creator_call_count", 0),
                "call_results": calls,
                "observed_cost_cny": cost,
                "branches": branches,
                "replay_macro_delta_pp": (
                    replay.get("macro_delta_pp") if isinstance(replay, dict) else None
                ),
                "replay_capability_delta_pp": (
                    replay.get("capability_delta_pp")
                    if isinstance(replay, dict)
                    else None
                ),
                "body_macro_delta_pp": (
                    body.get("macro_delta_pp") if isinstance(body, dict) else None
                ),
                "body_capability_delta_pp": (
                    body.get("capability_delta_pp") if isinstance(body, dict) else None
                ),
            }
        )
    return {
        "schema_version": 1,
        "kind": "s1-dual-policy-campaign-summary",
        "completed_rounds": len(rounds),
        "total_observed_cost_cny": sum(
            float(item["observed_cost_cny"]) for item in rounds
        ),
        "rounds": rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.campaign_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
