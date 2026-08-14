"""Build a coverage-aware top-10 from completed dual-policy S1 rounds.

The shortlist is descriptive, not an acceptance override.  It ranks only
rounds that reached the frozen body gate, keeps every accepted round first,
then orders by observed body effect and coverage.  A deterministic diversity
repair swaps in the strongest body-evaluated candidate for any capability
otherwise absent from the shortlist.  Rejected candidates remain rejected.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from skillchain.tools.serialization import canonical_json_bytes


ALL_CAPABILITIES = (
    "knowledge.visual_encyclopedia",
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
)


def _round_number(round_id: str) -> int:
    return int(round_id.removeprefix("r"))


def _body_evaluated(decision_path: Path) -> dict[str, Any] | None:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    metrics = decision.get("metrics", {})
    body = metrics.get("gate")
    replay = metrics.get("replay_gate")
    retained = tuple(metrics.get("retained_patch_capabilities", ()))
    if not isinstance(body, dict) or not isinstance(replay, dict) or not retained:
        return None
    round_id = str(metrics["round_id"])
    hard_delta = float(body["hard_error_delta_pp"])
    oracle_complete = bool(body["oracle_coverage_complete"])
    accepted = bool(decision["accepted"])
    return {
        "round_id": round_id,
        "round_number": _round_number(round_id),
        "accepted": accepted,
        "deployable": accepted,
        "candidate_bank": decision.get("candidate_bank"),
        "selected_bank": decision["selected_bank"],
        "retained_capabilities": list(retained),
        "capability_coverage_count": len(retained),
        "replay_macro_delta_pp": float(replay["macro_delta_pp"]),
        "body_macro_delta_pp": float(body["macro_delta_pp"]),
        "body_capability_delta_pp": body["capability_delta_pp"],
        "body_hard_error_delta_pp": hard_delta,
        "body_oracle_coverage_complete": oracle_complete,
        "gate_reasons": list(decision["reasons"]),
        "decision_path": str(decision_path),
        "selection_basis": "body-effect",
    }


def _primary_key(row: dict[str, Any]) -> tuple[object, ...]:
    return (
        -int(row["accepted"]),
        -float(row["body_macro_delta_pp"]),
        -int(row["capability_coverage_count"]),
        -float(row["replay_macro_delta_pp"]),
        int(row["round_number"]),
    )


def _safe_to_remove(row: dict[str, Any], coverage_counts: Counter[str]) -> bool:
    return all(
        coverage_counts[capability] > 1 for capability in row["retained_capabilities"]
    )


def rank(*, campaign_roots: list[Path], limit: int = 10) -> dict[str, Any]:
    decisions: list[Path] = []
    for root in campaign_roots:
        decisions.extend(root.glob("runs/*/decisions/s1.json"))
    decisions.sort(
        key=lambda path: _round_number(path.parent.parent.name.split("-", 1)[0])
    )
    if len(decisions) != 30:
        raise RuntimeError(f"expected 30 terminal decisions, observed {len(decisions)}")

    body_evaluated = [
        row for path in decisions if (row := _body_evaluated(path)) is not None
    ]
    excluded = [
        {
            "round_id": row["round_id"],
            "reason": "body oracle coverage incomplete",
        }
        for row in body_evaluated
        if not row["body_oracle_coverage_complete"]
    ]
    candidates = [row for row in body_evaluated if row["body_oracle_coverage_complete"]]
    candidates.sort(key=_primary_key)
    if len(candidates) < limit:
        raise RuntimeError(f"only {len(candidates)} rounds reached body gate")

    selected = candidates[:limit]
    selected_ids = {row["round_id"] for row in selected}
    coverage_counts: Counter[str] = Counter(
        capability for row in selected for capability in row["retained_capabilities"]
    )
    body_evaluated_union = {
        capability for row in candidates for capability in row["retained_capabilities"]
    }

    diversity_swaps: list[dict[str, str]] = []
    for missing in sorted(body_evaluated_union - set(coverage_counts)):
        replacement = next(
            row
            for row in candidates
            if row["round_id"] not in selected_ids
            and missing in row["retained_capabilities"]
        )
        removable = next(
            row
            for row in reversed(selected)
            if not row["accepted"] and _safe_to_remove(row, coverage_counts)
        )
        selected.remove(removable)
        selected_ids.remove(removable["round_id"])
        for capability in removable["retained_capabilities"]:
            coverage_counts[capability] -= 1
        replacement = dict(replacement)
        replacement["selection_basis"] = f"coverage-repair:{missing}"
        selected.append(replacement)
        selected_ids.add(replacement["round_id"])
        for capability in replacement["retained_capabilities"]:
            coverage_counts[capability] += 1
        diversity_swaps.append(
            {
                "missing_capability": missing,
                "removed_round": removable["round_id"],
                "added_round": replacement["round_id"],
            }
        )

    selected.sort(key=_primary_key)
    for rank_index, row in enumerate(selected, start=1):
        row["rank"] = rank_index

    selected_union = sorted(
        {capability for row in selected for capability in row["retained_capabilities"]}
    )
    return {
        "schema_version": 1,
        "kind": "s1-dual-policy-campaign-top10",
        "policy_version": "s1-campaign-top10-balanced-v1",
        "round_count": len(decisions),
        "body_gate_round_count": len(body_evaluated),
        "eligible_body_round_count": len(candidates),
        "excluded_from_ranking": excluded,
        "selection_limit": limit,
        "ranking_rule": (
            "accepted first; then descending body macro delta, capability coverage, "
            "replay macro delta, and ascending round id; deterministic diversity "
            "repair covers every capability represented by a body-evaluated candidate"
        ),
        "acceptance_override": False,
        "deployable_round_ids": [
            row["round_id"] for row in selected if row["accepted"]
        ],
        "selected_capability_union": selected_union,
        "uncovered_capabilities": sorted(set(ALL_CAPABILITIES) - set(selected_union)),
        "body_evaluated_capability_union": sorted(body_evaluated_union),
        "diversity_swaps": diversity_swaps,
        "top10": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = rank(campaign_roots=args.campaign_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
