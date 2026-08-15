#!/usr/bin/env python3
"""Compare two immutable S2 route-only opt800 profiles."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.evaluation.core_fast.models import (  # noqa: E402
    CAPABILITIES,
    S2RouteObservation,
)
from skillchain.evaluation.core_fast.store import atomic_write_json  # noqa: E402
from skillchain.tools.serialization import (  # noqa: E402
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)


def _load(
    path: Path, *, label: str
) -> tuple[bytes, dict[str, S2RouteObservation], tuple[str, ...]]:
    content = read_stable_regular_file(path.resolve(), label=label)
    raw_rows = parse_canonical_jsonl(content, label=label)
    rows = [S2RouteObservation.model_validate(item, strict=True) for item in raw_rows]
    if len(rows) != 800 or len({row.query_id for row in rows}) != 800:
        raise ValueError(f"{label} must contain 800 unique route observations")
    if len({row.requested_model for row in rows}) != 1:
        raise ValueError(f"{label} must use one requested model")
    if len({row.bank_sha256 for row in rows}) != 1:
        raise ValueError(f"{label} must use one Bank")
    if len({row.source_call_id for row in rows}) != 800:
        raise ValueError(f"{label} must contain 800 unique source calls")
    return (
        content,
        {row.query_id: row for row in rows},
        tuple(row.query_id for row in rows),
    )


def _is_correct(row: S2RouteObservation) -> bool:
    return row.selected_capability in set(row.acceptable_capabilities)


def _macro_f1(rows: dict[str, S2RouteObservation]) -> float:
    scores: list[float] = []
    for capability in CAPABILITIES:
        tp = fp = fn = 0
        for row in rows.values():
            expected = row.expected_capability
            predicted = row.selected_capability
            tp += expected == capability and predicted == capability
            fp += expected != capability and predicted == capability
            fn += expected == capability and predicted != capability
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return sum(scores) / len(scores)


def _summary(rows: dict[str, S2RouteObservation]) -> dict[str, object]:
    correct = sum(_is_correct(row) for row in rows.values())
    per_capability: dict[str, dict[str, int]] = {}
    for capability in CAPABILITIES:
        selected = [
            row for row in rows.values() if row.expected_capability == capability
        ]
        cap_correct = sum(_is_correct(row) for row in selected)
        per_capability[capability] = {
            "query_count": len(selected),
            "correct_count": cap_correct,
            "error_count": len(selected) - cap_correct,
        }
    confusions = Counter(
        (row.expected_capability, row.selected_capability)
        for row in rows.values()
        if not _is_correct(row)
    )
    return {
        "model": next(iter(rows.values())).requested_model,
        "query_count": len(rows),
        "correct_count": correct,
        "error_count": len(rows) - correct,
        "accuracy": correct / len(rows),
        "macro_f1": _macro_f1(rows),
        "per_capability": per_capability,
        "confusions": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(
                confusions.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def compare_profiles(*, baseline_path: Path, candidate_path: Path) -> dict[str, object]:
    baseline_bytes, baseline, baseline_order = _load(
        baseline_path, label="baseline route profile"
    )
    candidate_bytes, candidate, candidate_order = _load(
        candidate_path, label="candidate route profile"
    )
    if set(baseline) != set(candidate):
        raise ValueError("route profile query sets differ")
    if baseline_order != candidate_order:
        raise ValueError("route profile query order differs")
    for query_id in baseline:
        before = baseline[query_id]
        after = candidate[query_id]
        if (
            before.query_id != after.query_id
            or before.expected_capability != after.expected_capability
            or before.acceptable_capabilities != after.acceptable_capabilities
            or before.bank_sha256 != after.bank_sha256
        ):
            raise ValueError(f"route profile identity differs: {query_id}")
    baseline_summary = _summary(baseline)
    candidate_summary = _summary(candidate)
    corrected = sorted(
        query_id
        for query_id in baseline
        if not _is_correct(baseline[query_id]) and _is_correct(candidate[query_id])
    )
    broken = sorted(
        query_id
        for query_id in baseline
        if _is_correct(baseline[query_id]) and not _is_correct(candidate[query_id])
    )
    both_correct = sum(
        _is_correct(baseline[query_id]) and _is_correct(candidate[query_id])
        for query_id in baseline
    )
    both_incorrect = len(baseline) - both_correct - len(corrected) - len(broken)
    headroom_delta = int(candidate_summary["error_count"]) - int(
        baseline_summary["error_count"]
    )
    return {
        "schema_version": 1,
        "kind": "core-fast-s2-route-model-comparison",
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": sha256_bytes(baseline_bytes),
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "bank_sha256": next(iter(baseline.values())).bank_sha256,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "paired": {
            "corrected_count": len(corrected),
            "corrected_query_ids": corrected,
            "broken_count": len(broken),
            "broken_query_ids": broken,
            "both_correct_count": both_correct,
            "both_incorrect_count": both_incorrect,
        },
        "candidate_error_headroom_delta": headroom_delta,
        "candidate_is_strictly_weaker": (
            int(candidate_summary["correct_count"])
            < int(baseline_summary["correct_count"])
            and float(candidate_summary["macro_f1"])
            < float(baseline_summary["macro_f1"])
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = compare_profiles(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
    )
    atomic_write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
