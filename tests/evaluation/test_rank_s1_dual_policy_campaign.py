from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rank_s1_dual_policy_campaign import rank


CAPABILITIES = (
    "knowledge.visual_encyclopedia",
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
)


def _write_decision(
    root: Path,
    *,
    number: int,
    accepted: bool = False,
    capabilities: tuple[str, ...] = (),
    replay: float | None = None,
    body: float | None = None,
) -> None:
    round_id = f"r{number}"
    path = root / "runs" / f"{round_id}-fixture" / "decisions" / "s1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "round_id": round_id,
        "retained_patch_capabilities": list(capabilities),
    }
    if replay is not None:
        metrics["replay_gate"] = {"macro_delta_pp": replay}
    if body is not None:
        metrics["gate"] = {
            "macro_delta_pp": body,
            "hard_error_delta_pp": 0.0,
            "oracle_coverage_complete": True,
            "capability_delta_pp": {capability: 0.0 for capability in CAPABILITIES},
        }
    path.write_text(
        json.dumps(
            {
                "accepted": accepted,
                "candidate_bank": f"candidate-{number}",
                "selected_bank": f"selected-{number}",
                "reasons": [] if accepted else ["fixture rejection"],
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )


def test_rank_prefers_accepted_and_repairs_body_evaluated_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    for number in range(1, 31):
        _write_decision(root, number=number)

    _write_decision(
        root,
        number=1,
        accepted=True,
        capabilities=("product.style_recommendation",),
        replay=0.2,
        body=2.1,
    )
    for number in range(2, 13):
        _write_decision(
            root,
            number=number,
            capabilities=("product.multi_search",),
            replay=float(number),
            body=float(20 - number),
        )
    _write_decision(
        root,
        number=13,
        capabilities=("knowledge.visual_encyclopedia",),
        replay=-1.0,
        body=-1.0,
    )

    payload = rank(campaign_roots=[root])

    assert payload["round_count"] == 30
    assert payload["body_gate_round_count"] == 13
    assert payload["eligible_body_round_count"] == 13
    assert payload["deployable_round_ids"] == ["r1"]
    assert payload["top10"][0]["round_id"] == "r1"
    assert len(payload["top10"]) == 10
    assert "knowledge.visual_encyclopedia" in payload["selected_capability_union"]
    assert payload["diversity_swaps"] == [
        {
            "missing_capability": "knowledge.visual_encyclopedia",
            "removed_round": "r10",
            "added_round": "r13",
        }
    ]
    assert (
        next(row for row in payload["top10"] if row["round_id"] == "r13")["deployable"]
        is False
    )


def test_rank_requires_all_thirty_terminal_decisions(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    for number in range(1, 30):
        _write_decision(root, number=number)

    with pytest.raises(RuntimeError, match="expected 30 terminal decisions"):
        rank(campaign_roots=[root])
