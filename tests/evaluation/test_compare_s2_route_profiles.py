from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_s2_route_profiles import compare_profiles, main
from skillchain.evaluation.core_fast.models import CAPABILITIES, S2RouteObservation
from skillchain.evaluation.core_fast.store import atomic_write_jsonl


def _rows(*, model: str, selected: dict[int, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(800):
        expected = CAPABILITIES[index % len(CAPABILITIES)]
        row = S2RouteObservation(
            query_id=f"q-{index:04d}",
            expected_capability=expected,
            acceptable_capabilities=(expected,),
            selected_capability=selected.get(index, expected),
            bank_sha256="a" * 64,
            requested_model=model,
            source_call_id=f"call-{index:04d}",
        )
        rows.append(row.model_dump(mode="json"))
    return rows


def test_compare_route_profiles_reports_paired_headroom(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "comparison.json"
    atomic_write_jsonl(
        baseline,
        _rows(model="qwen3.7-flash-2026-07-15", selected={0: CAPABILITIES[1]}),
    )
    atomic_write_jsonl(
        candidate,
        _rows(
            model="qwen3.5-flash-2026-02-23",
            selected={0: CAPABILITIES[1], 1: CAPABILITIES[2], 2: CAPABILITIES[3]},
        ),
    )

    comparison = compare_profiles(baseline_path=baseline, candidate_path=candidate)

    assert comparison["baseline"]["correct_count"] == 799
    assert comparison["candidate"]["correct_count"] == 797
    assert comparison["paired"]["corrected_count"] == 0
    assert comparison["paired"]["broken_query_ids"] == ["q-0001", "q-0002"]
    assert comparison["candidate_error_headroom_delta"] == 2
    assert comparison["candidate_is_strictly_weaker"] is True
    assert (
        main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == comparison
    with pytest.raises(FileExistsError):
        main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--output",
                str(output),
            ]
        )

    reordered = tmp_path / "reordered.jsonl"
    atomic_write_jsonl(
        reordered,
        list(reversed(_rows(model="qwen3.5-flash-2026-02-23", selected={}))),
    )
    with pytest.raises(ValueError, match="query order differs"):
        compare_profiles(baseline_path=baseline, candidate_path=reordered)
