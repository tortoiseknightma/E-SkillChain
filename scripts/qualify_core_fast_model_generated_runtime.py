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

from skillchain.evaluation.core_fast import (  # noqa: E402
    CoreFastEngine,
    FastPathError,
    load_core_fast_spec,
)
from skillchain.evaluation.core_fast.adapters import load_adapter  # noqa: E402
from skillchain.evaluation.core_fast.models import CAPABILITIES  # noqa: E402
from skillchain.evaluation.core_fast.store import atomic_write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify the model-generated Core Fast answer replay with symmetric "
            "parent/no-op arms over the frozen opt replay200."
        )
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec_path = args.spec.resolve()
    spec = load_core_fast_spec(spec_path)
    adapter = load_adapter(spec, cwd=REPOSITORY_ROOT, spec_path=spec_path)
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=args.output_root.resolve(),
        adapter=adapter,
    )
    engine.initialize()
    opt = engine.opt_static()
    folds = engine.opt_fold_roles()
    replay_queries = tuple(
        query
        for query in engine.queries()
        if query.split == "opt_pool" and folds[query.query_id] == "replay"
    )
    static = {query.query_id: opt[query.query_id] for query in replay_queries}
    parent = engine.static_bank()
    eligible_queries = tuple(
        query
        for query in replay_queries
        if static[query.query_id].selected_capability == query.canonical_capability
        and engine._body_replay_parent_eligible(static[query.query_id])
    )
    eligible_static = {
        query.query_id: static[query.query_id] for query in eligible_queries
    }
    arm_a = engine._assistant_many(
        split="runtime-noop-replay200",
        config="s1-branch-qualification-a",
        queries=eligible_queries,
        bank=parent,
        reuse_parent=eligible_static,
    )
    arm_b = engine._assistant_many(
        split="runtime-noop-replay200",
        config="s1-branch-qualification-b",
        queries=eligible_queries,
        bank=parent,
        reuse_parent=eligible_static,
    )
    if engine._common_trace_violations(arm_a, arm_b):
        raise FastPathError("no-op qualification route/tool evidence drifted")

    by_capability: dict[str, dict[str, object]] = {}
    total_gains = total_regressions = 0
    for capability in CAPABILITIES:
        selected = tuple(
            query
            for query in eligible_queries
            if query.canonical_capability == capability
        )
        screen = engine._s1_capability_screen(
            capability=capability,
            queries=selected,
            baseline={query.query_id: arm_a[query.query_id] for query in selected},
            candidate={query.query_id: arm_b[query.query_id] for query in selected},
        )
        total_gains += int(screen["gain_count"])
        total_regressions += int(screen["static_success_to_candidate_failure_count"])
        by_capability[capability] = screen

    qualified = total_regressions == 0 and total_gains <= 2
    payload = {
        "schema_version": 1,
        "kind": "core-fast-model-generated-runtime-noop-qualification",
        "assistant_contract": spec.runtime.assistant_contract,
        "static_bank_sha256": parent.bank_sha256,
        "opt_static_results_sha256": spec.opt_static_results_sha256,
        "replay_population_count": len(replay_queries),
        "treatment_reached_count": len(eligible_queries),
        "aliased_parent_query_ids": [
            query.query_id for query in replay_queries if query not in eligible_queries
        ],
        "arm_a": engine._summary(arm_a, eligible_queries),
        "arm_b": engine._summary(arm_b, eligible_queries),
        "total_gains": total_gains,
        "total_regressions": total_regressions,
        "qualified": qualified,
        "qualification_thresholds": {
            "maximum_noop_regressions": 0,
            "maximum_noop_gains": 2,
        },
        "capabilities": by_capability,
        "observed_cost_cny": engine.calls.observed_cost(),
        "interpretation": (
            "This estimates stochastic answer-generation variation with identical "
            "Bank, route, tool trace, public evidence, and replay primitive."
        ),
    }
    artifact = engine.output_root / "runtime-noop-qualification.json"
    if artifact.exists():
        existing = json.loads(artifact.read_text(encoding="utf-8"))
        if existing != payload:
            raise FastPathError("runtime no-op qualification changed on resume")
    else:
        atomic_write_json(artifact, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if qualified else 3


if __name__ == "__main__":
    raise SystemExit(main())
