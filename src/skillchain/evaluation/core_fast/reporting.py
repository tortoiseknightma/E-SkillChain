from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

from .models import CAPABILITIES, CONFIGS, CallResult, StageDecision
from .store import atomic_write_json, atomic_write_jsonl, load_json

if TYPE_CHECKING:
    from .engine import CoreFastEngine


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise RuntimeError(f"report input is missing: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"report row is not an object: {path}")
            rows.append(value)
    return rows


def _route_macro_f1(rows: list[dict[str, object]]) -> float:
    scores: list[float] = []
    for capability in CAPABILITIES:
        tp = fp = fn = 0
        for row in rows:
            query = row["query"]
            assistant = row["assistant"]
            assert isinstance(query, dict) and isinstance(assistant, dict)
            expected = query["canonical_capability"]
            predicted = assistant["selected_capability"]
            tp += expected == capability and predicted == capability
            fp += expected != capability and predicted == capability
            fn += expected == capability and predicted != capability
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return mean(scores)


def _confusion(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    matrix = {
        expected: {predicted: 0 for predicted in (*CAPABILITIES, "<none>")}
        for expected in CAPABILITIES
    }
    for row in rows:
        query = row["query"]
        assistant = row["assistant"]
        assert isinstance(query, dict) and isinstance(assistant, dict)
        expected = query["canonical_capability"]
        predicted = assistant["selected_capability"] or "<none>"
        matrix[str(expected)][str(predicted)] += 1
    return matrix


def _config_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    by_capability: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        query = row["query"]
        assert isinstance(query, dict)
        by_capability[str(query["canonical_capability"])].append(row)
    capability_gcs = {}
    for capability in CAPABILITIES:
        values = by_capability[capability]
        capability_gcs[capability] = mean(
            float(row["assistant"]["gcs_score"])
            for row in values  # type: ignore[index]
        )
    judges = [row["judge"] for row in rows if isinstance(row.get("judge"), dict)]
    return {
        "query_count": len(rows),
        "capability_macro_gcs": mean(capability_gcs.values()),
        "query_micro_gcs": mean(
            float(row["assistant"]["gcs_score"])
            for row in rows  # type: ignore[index]
        ),
        "mean_j_project": (
            mean(float(item["j_project"]) for item in judges) if judges else None
        ),
        "route_macro_f1": _route_macro_f1(rows),
        "hard_errors": sum(bool(row["assistant"]["hard_error"]) for row in rows),  # type: ignore[index]
        "tool_compliance": mean(
            not bool(row["assistant"]["tool_violation"])
            for row in rows  # type: ignore[index]
        ),
        "card_compliance": mean(
            not bool(row["assistant"]["card_violation"])
            for row in rows  # type: ignore[index]
        ),
        "evidence_compliance": mean(
            not bool(row["assistant"]["evidence_violation"])
            for row in rows  # type: ignore[index]
        ),
        "capability_gcs": capability_gcs,
        "confusion": _confusion(rows),
    }


def _slice_metrics(
    rows: list[dict[str, object]],
    key: str,
    extractor: Callable[[dict[str, object]], str],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[extractor(row)].append(row)
    return {
        key: {
            value: {
                "count": len(items),
                "query_micro_gcs": mean(
                    float(item["assistant"]["gcs_score"])
                    for item in items  # type: ignore[index]
                ),
                "mean_j_project": mean(
                    float(item["judge"]["j_project"])
                    for item in items  # type: ignore[index]
                ),
            }
            for value, items in sorted(grouped.items())
        }
    }


def _bootstrap_delta(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    def keyed(rows: Iterable[dict[str, object]]) -> dict[str, dict[str, object]]:
        return {str(row["query"]["query_id"]): row for row in rows}  # type: ignore[index]

    before = keyed(left)
    after = keyed(right)
    if set(before) != set(after):
        raise RuntimeError("paired bootstrap configs have different query sets")
    groups: dict[str, list[str]] = defaultdict(list)
    for query_id, row in before.items():
        groups[str(row["query"]["leakage_group_id"])].append(query_id)  # type: ignore[index]
    group_ids = sorted(groups)

    def delta_for(sampled: Iterable[str]) -> float:
        differences: list[float] = []
        for group in sampled:
            for query_id in groups[group]:
                differences.append(
                    float(after[query_id]["assistant"]["gcs_score"])  # type: ignore[index]
                    - float(before[query_id]["assistant"]["gcs_score"])  # type: ignore[index]
                )
        return mean(differences)

    observed = delta_for(group_ids)
    rng = random.Random(seed)
    samples = sorted(
        delta_for(rng.choices(group_ids, k=len(group_ids))) for _ in range(replicates)
    )
    low = samples[int(0.025 * (replicates - 1))]
    high = samples[int(0.975 * (replicates - 1))]
    return {
        "group_count": len(group_ids),
        "replicates": replicates,
        "delta": observed,
        "ci95_low": low,
        "ci95_high": high,
    }


def _usage(root: Path) -> dict[str, object]:
    by_role: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_cny": 0.0,
            "latency_ms": 0,
        }
    )
    statuses: Counter[str] = Counter()
    cost_bases: Counter[str] = Counter()
    for path in (root / "calls").glob("*/*.result.json"):
        result = CallResult.model_validate(load_json(path), strict=True)
        item = by_role[result.role]
        item["calls"] += 1
        item["input_tokens"] += result.input_tokens
        item["output_tokens"] += result.output_tokens
        item["cost_cny"] += result.cost_cny
        item["latency_ms"] += result.latency_ms
        statuses[result.status] += 1
        cost_bases[result.cost_basis] += 1
    total = {
        key: sum(float(item[key]) for item in by_role.values())
        for key in ("calls", "input_tokens", "output_tokens", "cost_cny", "latency_ms")
    }
    total["calls"] = int(total["calls"])
    total["input_tokens"] = int(total["input_tokens"])
    total["output_tokens"] = int(total["output_tokens"])
    total["latency_ms"] = int(total["latency_ms"])
    return {
        "total": total,
        "by_role": dict(by_role),
        "statuses": dict(statuses),
        "cost_bases": dict(cost_bases),
    }


def _representative_cases(
    by_config: Mapping[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    static = {str(row["query"]["query_id"]): row for row in by_config["llm_static"]}  # type: ignore[index]
    full = {str(row["query"]["query_id"]): row for row in by_config["full"]}  # type: ignore[index]
    cases: list[dict[str, object]] = []
    for query_id in static:
        delta = (
            float(full[query_id]["assistant"]["gcs_score"])  # type: ignore[index]
            - float(static[query_id]["assistant"]["gcs_score"])  # type: ignore[index]
        )
        cases.append(
            {
                "query_id": query_id,
                "capability": static[query_id]["query"]["canonical_capability"],  # type: ignore[index]
                "delta_gcs": delta,
                "static": static[query_id]["assistant"],
                "full": full[query_id]["assistant"],
            }
        )
    improved = sorted(
        cases, key=lambda item: (-float(item["delta_gcs"]), str(item["query_id"]))
    )[:3]
    regressed = sorted(
        cases, key=lambda item: (float(item["delta_gcs"]), str(item["query_id"]))
    )[:3]
    return [{**item, "case_type": "improvement"} for item in improved] + [
        {**item, "case_type": "failure_or_regression"} for item in regressed
    ]


def build_report(engine: "CoreFastEngine") -> dict[str, object]:
    report_root = engine.output_root / "reports"
    val = _read_jsonl(report_root / "val-results.jsonl")
    test = _read_jsonl(report_root / "test-results.jsonl")
    if len(val) != 1000 or len(test) != 1500:
        raise RuntimeError("logical result geometry must be val1000/test1500")
    by_config = {
        config: [row for row in test if row["config"] == config] for config in CONFIGS
    }
    val_by_config = {
        config: [row for row in val if row["config"] == config] for config in CONFIGS
    }
    metrics = {config: _config_metrics(rows) for config, rows in by_config.items()}
    val_metrics = {
        config: _config_metrics(rows) for config, rows in val_by_config.items()
    }
    comparisons = (
        ("noskill_to_static", "noskill", "llm_static"),
        ("static_to_s1", "llm_static", "s1"),
        ("s1_to_s1s2", "s1", "s1s2"),
        ("s1s2_to_full", "s1s2", "full"),
    )
    deltas = {}
    bootstrap = {}
    for name, before, after in comparisons:
        before_j = metrics[before]["mean_j_project"]
        after_j = metrics[after]["mean_j_project"]
        deltas[name] = {
            "capability_macro_gcs": (
                metrics[after]["capability_macro_gcs"]
                - metrics[before]["capability_macro_gcs"]
            ),
            "mean_j_project": (
                None if before_j is None or after_j is None else after_j - before_j
            ),
        }
        bootstrap[name] = _bootstrap_delta(
            by_config[before],
            by_config[after],
            replicates=engine.spec.bootstrap_replicates,
            seed=engine.spec.bootstrap_seed,
        )
    slices: dict[str, object] = {}
    for config, rows in by_config.items():
        config_slices: dict[str, object] = {}
        extractors = {
            "capability": lambda row: str(row["query"]["canonical_capability"]),  # type: ignore[index]
            "boundary": lambda row: (
                str(row["query"]["boundary_strategy"])  # type: ignore[index]
                if row["query"]["is_boundary"]  # type: ignore[index]
                else "non_boundary"
            ),
            "source": lambda row: str(row["assistant"].get("source") or "unknown"),  # type: ignore[union-attr]
            "repair": lambda row: str(row["assistant"].get("repair") or "none"),  # type: ignore[union-attr]
            "style_submode": lambda row: str(
                row["assistant"].get("style_submode") or "none"
            ),  # type: ignore[union-attr]
            "cross_intent": lambda row: (
                "cross_intent"
                if str(row["query"]["boundary_strategy"]).startswith("cross_intent")  # type: ignore[index]
                else "other"
            ),
        }
        for key, extractor in extractors.items():
            config_slices.update(_slice_metrics(rows, key, extractor))
        slices[config] = config_slices
    decisions = {
        stage: StageDecision.model_validate_json(
            (engine.output_root / "decisions" / f"{stage}.json").read_bytes(),
            strict=True,
        ).model_dump(mode="json")
        for stage in ("s1", "s2", "full")
    }
    cases = _representative_cases(by_config)
    atomic_write_jsonl(report_root / "cases.jsonl", cases, overwrite=True)
    report = {
        "schema_version": 1,
        "kind": "core-experiment-fast-report",
        "experiment_id": engine.spec.experiment_id,
        "geometry": {"val_logical_rows": len(val), "test_logical_rows": len(test)},
        "decisions": decisions,
        "val_metrics": val_metrics,
        "test_metrics": metrics,
        "stage_deltas": deltas,
        "paired_leakage_group_bootstrap": bootstrap,
        "slices": slices,
        "usage": _usage(engine.output_root),
        "route_all1500": (
            load_json(report_root / "route-all1500-summary.json")
            if (report_root / "route-all1500-summary.json").is_file()
            else None
        ),
        "disclosures": list(engine.spec.disclosures),
    }
    atomic_write_json(report_root / "summary.json", report, overwrite=True)
    return report


__all__ = ["build_report"]
