from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from skillchain.evaluation.core_fast.engine import CoreFastEngine, FastPathError
from skillchain.evaluation.core_fast.fake_provider import FakeCoreFastAdapter
from skillchain.evaluation.core_fast.feedback_selection import (
    build_discovery_feedback_population,
    select_feedback_samples,
)
from skillchain.evaluation.core_fast.models import CAPABILITIES, S1Settings
from skillchain.tools.serialization import canonical_json_bytes

_fixture_spec = importlib.util.spec_from_file_location(
    "core_fast_fixture_module", Path(__file__).with_name("test_core_fast.py")
)
assert _fixture_spec is not None and _fixture_spec.loader is not None
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
fast_fixture = _fixture_module.fast_fixture


def _settings(*, count: int, allocation: str = "balanced-six-capability") -> S1Settings:
    payload: dict[str, object] = {
        "feedback_mode": "fresh-per-round",
        "feedback_total_count": count,
        "feedback_canary_count": min(6, count),
        "feedback_selection_policy": "discovery-stratified-v1",
        "feedback_allocation": allocation,
    }
    if allocation == "target-focused":
        payload.update(
            {
                "target_capabilities": ("product.multi_search",),
                "max_patched_capabilities": 1,
                "protected_capabilities": tuple(
                    item for item in CAPABILITIES if item != "product.multi_search"
                ),
            }
        )
    return S1Settings.model_validate(payload, strict=True)


def _population(fast_fixture):
    spec, spec_path, queries = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=spec_path.parent / "population",
        adapter=FakeCoreFastAdapter(),
    )
    return build_discovery_feedback_population(
        queries=queries,
        observations=engine.opt_static(),
        fold_roles=engine.opt_fold_roles(),
    )


def test_discovery_population_covers_600_and_excludes_replay(fast_fixture) -> None:
    population = _population(fast_fixture)
    query_ids = {row["query_id"] for row in population}
    replay_ids = {
        query_id
        for query_id, role in CoreFastEngine(
            spec=fast_fixture[0],
            spec_path=fast_fixture[1],
            output_root=fast_fixture[1].parent / "roles",
            adapter=FakeCoreFastAdapter(),
        )
        .opt_fold_roles()
        .items()
        if role == "replay"
    }
    assert len(population) == len(query_ids) == 600
    assert query_ids.isdisjoint(replay_ids)


@pytest.mark.parametrize("count", (12, 24, 48, 60))
def test_selection_counts_are_deterministic_and_unique(
    fast_fixture, count: int
) -> None:
    population = _population(fast_fixture)
    first = select_feedback_samples(population, _settings(count=count))
    second = select_feedback_samples(population, _settings(count=count))
    assert canonical_json_bytes(list(first)) == canonical_json_bytes(list(second))
    assert len(first) == count
    assert len({row["query_id"] for row in first}) == count
    assert len({row["asset_id"] for row in first}) == count
    assert len({row["leakage_group_id"] for row in first}) == count


def test_balanced_and_target_focused_allocations(fast_fixture) -> None:
    population = _population(fast_fixture)
    balanced = select_feedback_samples(population, _settings(count=48))
    assert {
        capability: sum(row["capability"] == capability for row in balanced)
        for capability in CAPABILITIES
    } == {capability: 8 for capability in CAPABILITIES}
    focused = select_feedback_samples(
        population, _settings(count=48, allocation="target-focused")
    )
    assert all(row["capability"] == "product.multi_search" for row in focused)
    classes = [row["selection_class"] for row in focused]
    assert classes.count("body_fixable_failure") == sum(
        row["capability"] == "product.multi_search"
        and row["selection_class"] == "body_fixable_failure"
        for row in population
    )
    assert classes.count("success_anchor") >= 1


def test_prepare_is_byte_exact_and_manifest_drift_fails_closed(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _ = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=tmp_path / "prepare",
        adapter=FakeCoreFastAdapter(),
    )
    first = engine.prepare_feedback_selection()
    manifest = engine.output_root / "inputs" / "feedback-selection-manifest.json"
    before = manifest.read_bytes()
    second = engine.prepare_feedback_selection()
    assert first == second
    assert manifest.read_bytes() == before
    value = json.loads(before)
    value["effective_count"] -= 1
    manifest.write_bytes(canonical_json_bytes(value))
    with pytest.raises(FastPathError, match="changed on resume"):
        engine.prepare_feedback_selection()


def test_same_round_resume_uses_exact_feedback_results(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _ = fast_fixture
    adapter = FakeCoreFastAdapter(reject_stages=frozenset({"s1"}))
    root = tmp_path / "resume"
    first = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=root,
        adapter=adapter,
    )
    first.run_s1()
    assert adapter.calls["feedback"] == spec.s1_settings.feedback_total_count
    decision = root / "decisions" / "s1.json"
    decision.unlink()
    second_adapter = FakeCoreFastAdapter(reject_stages=frozenset({"s1"}))
    second = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=root,
        adapter=second_adapter,
    )
    resumed = second.run_s1()
    assert second_adapter.calls["feedback"] == 0
    assert resumed.metrics["feedback_resume_cache_hits"] == (
        spec.s1_settings.feedback_total_count
    )
    assert resumed.metrics["feedback_provider_calls_this_round"] == 0


def test_creator_payload_contains_summary_but_no_hidden_population(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _ = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=tmp_path / "privacy",
        adapter=FakeCoreFastAdapter(reject_stages=frozenset({"s1"})),
    )
    engine.run_s1()
    intent = json.loads(
        (
            engine.output_root / "calls" / "creator" / "s1-creator-once.intent.json"
        ).read_text(encoding="utf-8")
    )
    encoded = canonical_json_bytes(intent["payload"])
    assert intent["payload"]["discovery_summary"]["population_count"] == 600
    assert intent["payload"]["selection_manifest_sha256"]
    for forbidden in (
        b'"split"',
        b"replay_context",
        b"runtime_paths",
        b"scorer_calls",
        b"test_frozen",
        b"body_gate",
    ):
        assert forbidden not in encoded


def test_feedback_count_changes_spec_identity(fast_fixture) -> None:
    spec = fast_fixture[0]
    count12 = spec.model_copy(
        update={
            "s1_settings": _settings(count=12),
            "limits": spec.limits.model_copy(update={"max_feedback_calls": 12}),
        }
    )
    count24 = spec.model_copy(
        update={
            "s1_settings": _settings(count=24),
            "limits": spec.limits.model_copy(update={"max_feedback_calls": 24}),
        }
    )
    assert canonical_json_bytes(
        count12.model_dump(mode="json")
    ) != canonical_json_bytes(count24.model_dump(mode="json"))
