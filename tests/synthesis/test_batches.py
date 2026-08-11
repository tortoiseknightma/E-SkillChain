import json
from types import SimpleNamespace
from pathlib import Path

import pytest

import skillchain.synthesis.batches as batches
from skillchain.synthesis.batches import (
    accept_generated_batch,
    compute_generation_input_sha256,
    corpus_status,
    rebuild_accepted_queries,
    reject_generated_batch,
    stage_generated_batch,
    verify_accepted_corpus,
)
from skillchain.synthesis.models import CorpusPlan
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from conftest import make_query_v2


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _accept_valid_batch(corpus_fixture):
    staged = corpus_fixture.stage_valid_batch()
    return accept_generated_batch(
        corpus_fixture.root,
        staged.name,
        confirmation="ACCEPT",
        plan_path=corpus_fixture.plan_path,
        allow_provisional_asset_groups=True,
    )


def _stage_kwargs_for_batch(corpus_fixture, base_batch_id: str) -> dict:
    plan = CorpusPlan.model_validate_json(corpus_fixture.plan_path.read_bytes())
    plan_items = [item for item in plan.queries if item.batch_id == base_batch_id]
    draft_path = corpus_fixture.draft_path.with_name(f"{base_batch_id}.jsonl")
    draft_manifest_path = draft_path.with_suffix(".manifest.json")
    draft_bytes = canonical_jsonl_bytes(
        [
            {
                "plan_id": item.plan_id,
                "turns": [
                    {
                        "role": "user",
                        "content": f"机械占位-{item.plan_id}",
                    }
                ],
            }
            for item in plan_items
        ]
    )
    draft_path.write_bytes(draft_bytes)
    plan_manifest_path = corpus_fixture.plan_path.with_name(
        f"{corpus_fixture.plan_path.stem}.manifest.json"
    )
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    seed_manifest = json.loads(
        (corpus_fixture.root / "seeds/accepted/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    draft_manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 2,
                "base_batch_id": base_batch_id,
                "data_origin": "synthetic_derived",
                "provider": "codex",
                "model_display_name": "5.6 Sol Ultra",
                "model_claim_source": "user_confirmation",
                "generated_at": "2026-07-11T00:00:00Z",
                "plan_sha256": plan_manifest["plan_sha256"],
                "asset_catalog_sha256": plan_manifest["asset_catalog_sha256"],
                "leakage_policy_version": plan_manifest["leakage_policy_version"],
                "seed_set_sha256": seed_manifest["seed_set_sha256"],
                "draft_sha256": sha256_bytes(draft_bytes),
                "generation_input_sha256": compute_generation_input_sha256(
                    base_batch_id=base_batch_id,
                    plan_items=plan_items,
                    plan_sha256=plan_manifest["plan_sha256"],
                    asset_catalog_sha256=plan_manifest["asset_catalog_sha256"],
                    leakage_policy_version=plan_manifest["leakage_policy_version"],
                    seed_set_sha256=seed_manifest["seed_set_sha256"],
                ),
            }
        )
    )
    return {
        "draft_path": draft_path,
        "draft_manifest_path": draft_manifest_path,
        "plan_path": corpus_fixture.plan_path,
        "queries_root": corpus_fixture.root,
        "base_batch_id": base_batch_id,
        "allow_provisional_asset_groups": True,
    }


def _bind_fixture_plan_to_fake_catalog(corpus_fixture, catalog_sha: str = "c" * 64):
    plan_path = corpus_fixture.plan_path
    manifest_path = plan_path.with_name(f"{plan_path.stem}.manifest.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["asset_catalog_sha256"] = catalog_sha
    plan["capability_assignments_sha256"] = "a" * 64
    plan["leakage_policy_version"] = "dataset-asset-components-v1"
    for item in plan["queries"]:
        item["capability_assignment_id"] = f"fixture.{item['plan_id']}"
        item["capability_assignment_source_sha256"] = "b" * 64
    plan_bytes = canonical_json_bytes(plan)
    plan_path.write_bytes(plan_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plan_sha256"] = sha256_bytes(plan_bytes)
    manifest["asset_catalog_sha256"] = catalog_sha
    manifest["capability_assignments_sha256"] = "a" * 64
    manifest["leakage_policy_version"] = "dataset-asset-components-v1"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    references = {
        item["asset_id"]: (item["image_path"], item["leakage_group_id"])
        for item in plan["queries"]
    }

    class FakeCatalog:
        def __init__(self):
            self.manifest = SimpleNamespace(
                catalog_sha256=catalog_sha,
                leakage_policy_version="dataset-asset-components-v1",
            )

        def verify_reference(self, asset_id, image_path, leakage_group_id=None):
            expected_path, expected_group = references[asset_id]
            if image_path != expected_path or leakage_group_id != expected_group:
                raise ValueError("catalog reference mismatch")
            return SimpleNamespace(leakage_group_id=expected_group)

        def require_verified_files(self):
            return None

        def verify_asset_ids(self, asset_ids):
            return None

    return FakeCatalog()


def test_stage_requires_exactly_the_planned_25_rows(corpus_fixture):
    corpus_fixture.write_draft(corpus_fixture.draft_rows()[:-1])

    with pytest.raises(ValueError, match="恰好 25 条"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_enriches_queries_in_plan_order_without_trusting_draft_metadata(
    corpus_fixture,
):
    corpus_fixture.write_draft(list(reversed(corpus_fixture.draft_rows())))

    staged = stage_generated_batch(**corpus_fixture.stage_kwargs)
    queries = _read_jsonl(staged / "results.jsonl")

    assert [row["query_id"] for row in queries] == [
        item.plan_id for item in corpus_fixture.batch_items
    ]
    assert queries[0]["image_path"] == corpus_fixture.batch_items[0].image_path
    assert (
        queries[0]["canonical_intent"] == corpus_fixture.batch_items[0].canonical_intent
    )
    assert "gt_intent" not in queries[0]
    assert (
        queries[0]["leakage_group_id"] == corpus_fixture.batch_items[0].leakage_group_id
    )
    assert (
        queries[0]["generator_batch_id"]
        == corpus_fixture.batch_items[0].generator_batch_id
    )
    assert queries[0]["split"] == corpus_fixture.batch_items[0].provisional_split
    assert queries[0]["synth_provider"] == "codex"
    assert queries[0]["synth_model"] == "5.6 Sol Ultra"
    assert queries[0]["synthesis_batch_id"] == "dev-mini-001-r1"

    manifest = json.loads((staged / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_origin"] == "synthetic_derived"

    report = json.loads((staged / "quality_report.json").read_text(encoding="utf-8"))
    assert report["data_origin"] == "synthetic_derived"
    assert report["count"] == 25
    assert report["boundary_count"] == 5
    assert report["duplicate_count"] == 0
    assert report["single_turn_count"] == 25


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "missing", "extra_field"])
def test_stage_rejects_result_set_that_does_not_match_plan(corpus_fixture, fault):
    rows = corpus_fixture.draft_rows()
    if fault == "duplicate":
        rows[-1]["plan_id"] = rows[0]["plan_id"]
    elif fault == "unknown":
        rows[-1]["plan_id"] = "unknown-plan-id"
    elif fault == "missing":
        rows.pop()
    else:
        rows[0]["gt_intent"] = "utility"
    corpus_fixture.write_draft(rows)

    with pytest.raises(ValueError, match="plan_id|恰好 25 条|extra"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_rejects_noncanonical_turns(corpus_fixture):
    rows = corpus_fixture.draft_rows()
    rows[0]["turns"] = [
        {"role": "user", "content": "机械占位-00"},
        {"role": "user", "content": "机械占位-01"},
    ]
    corpus_fixture.write_draft(rows)

    with pytest.raises(ValueError, match="turns"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_rejects_normalized_text_duplicates(corpus_fixture):
    rows = corpus_fixture.draft_rows()
    rows[0]["turns"][0]["content"] = "机械 重复！"
    rows[1]["turns"][0]["content"] = "机械重复"
    corpus_fixture.write_draft(rows)

    with pytest.raises(ValueError, match="重复"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_rejects_text_already_in_accepted_queries(corpus_fixture):
    duplicate = make_query_v2(
        query_id="accepted-001",
        image_path="query_images/exact_match/accepted.jpg",
        text="机械占位-dm-001",
        intent="exact_match",
        split="dev_mini",
        generator_batch_id="accepted-001",
    )
    (corpus_fixture.root / "queries.jsonl").write_bytes(
        canonical_jsonl_bytes([duplicate])
    )
    corpus_fixture.write_draft()

    with pytest.raises(ValueError, match="accepted.*重复"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_rejects_tampered_plan_or_seed(corpus_fixture):
    corpus_fixture.write_draft()
    corpus_fixture.plan_path.write_bytes(b'{"tampered":true}\n')

    with pytest.raises(ValueError, match="plan.*哈希"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_rejects_tampered_accepted_seed(corpus_fixture):
    corpus_fixture.write_draft()
    (corpus_fixture.root / "seeds/accepted/seed_examples.json").write_bytes(
        b'{"tampered":true}\n'
    )

    with pytest.raises(ValueError, match="seed_set_sha256.*哈希"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "plan_sha256",
        "seed_set_sha256",
        "draft_sha256",
        "generation_input_sha256",
    ],
)
def test_stage_rejects_stale_or_forged_draft_provenance(corpus_fixture, field):
    corpus_fixture.write_draft(manifest_overrides={field: "f" * 64})

    with pytest.raises(ValueError, match=field):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_rejects_draft_changed_after_manifest_was_written(corpus_fixture):
    corpus_fixture.write_draft()
    rows = corpus_fixture.draft_rows()
    rows[0]["turns"][-1]["content"] += "-tampered"
    corpus_fixture.draft_path.write_bytes(canonical_jsonl_bytes(rows))

    with pytest.raises(ValueError, match="draft_sha256"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_rejects_legacy_batch_draft_manifest(corpus_fixture):
    corpus_fixture.write_draft(manifest_overrides={"schema_version": 1})

    with pytest.raises(ValueError, match="schema_version"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_rejects_noncanonical_batch_draft_manifest(corpus_fixture):
    corpus_fixture.write_draft()
    corpus_fixture.draft_manifest_path.write_bytes(
        b" " + corpus_fixture.draft_manifest_path.read_bytes()
    )

    with pytest.raises(ValueError, match="canonical"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_generation_input_hash_binds_all_25_planned_items(corpus_fixture):
    corpus_fixture.write_draft()
    manifest = json.loads(
        corpus_fixture.draft_manifest_path.read_text(encoding="utf-8")
    )
    kwargs = {
        "base_batch_id": "dev-mini-001",
        "plan_items": list(reversed(corpus_fixture.batch_items)),
        "plan_sha256": manifest["plan_sha256"],
        "asset_catalog_sha256": manifest["asset_catalog_sha256"],
        "leakage_policy_version": manifest["leakage_policy_version"],
        "seed_set_sha256": manifest["seed_set_sha256"],
    }

    assert (
        compute_generation_input_sha256(**kwargs) == manifest["generation_input_sha256"]
    )
    changed_items = list(corpus_fixture.batch_items)
    changed_items[0] = changed_items[0].model_copy(
        update={"template_family": "changed-template-family"}
    )
    kwargs["plan_items"] = changed_items
    assert (
        compute_generation_input_sha256(**kwargs) != manifest["generation_input_sha256"]
    )
    kwargs["plan_items"] = corpus_fixture.batch_items[:-1]
    with pytest.raises(ValueError, match="完整 25"):
        compute_generation_input_sha256(**kwargs)


def test_stage_rejects_wrong_draft_model_claim(corpus_fixture):
    corpus_fixture.write_draft(manifest_overrides={"model_display_name": "GPT-5"})

    with pytest.raises(ValueError, match="5.6 Sol Ultra"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_refuses_second_active_revision(corpus_fixture):
    corpus_fixture.stage_valid_batch()
    corpus_fixture.write_draft()

    with pytest.raises(ValueError, match="active staging"):
        stage_generated_batch(**corpus_fixture.stage_kwargs)


def test_stage_rejects_a_batch_that_is_not_the_next_plan_prefix(corpus_fixture):
    kwargs = _stage_kwargs_for_batch(corpus_fixture, "dev-mini-002")

    with pytest.raises(ValueError, match="active plan order.*dev-mini-001"):
        stage_generated_batch(**kwargs)

    assert not (corpus_fixture.root / "staging").exists()


def test_provisional_batch_operations_require_explicit_opt_in(corpus_fixture):
    corpus_fixture.write_draft()
    stage_kwargs = dict(corpus_fixture.stage_kwargs)
    stage_kwargs.pop("allow_provisional_asset_groups")
    with pytest.raises(ValueError, match="provisional"):
        stage_generated_batch(**stage_kwargs)

    staged = stage_generated_batch(**corpus_fixture.stage_kwargs)
    with pytest.raises(ValueError, match="provisional"):
        accept_generated_batch(
            corpus_fixture.root,
            staged.name,
            confirmation="ACCEPT",
            plan_path=corpus_fixture.plan_path,
        )


def test_catalog_bound_batch_operation_requires_matching_catalog(corpus_fixture):
    catalog = _bind_fixture_plan_to_fake_catalog(corpus_fixture)
    corpus_fixture.write_draft()
    kwargs = dict(corpus_fixture.stage_kwargs)

    with pytest.raises(ValueError, match="asset catalog"):
        stage_generated_batch(**kwargs)

    kwargs["asset_catalog"] = catalog
    staged = stage_generated_batch(**kwargs)
    assert staged.is_dir()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asset_catalog_sha256", "d" * 64),
        ("leakage_policy_version", "different-formal-policy-v1"),
    ],
)
def test_catalog_bound_stage_rejects_wrong_draft_catalog_binding(
    corpus_fixture, field, value
):
    catalog = _bind_fixture_plan_to_fake_catalog(corpus_fixture)
    corpus_fixture.write_draft(manifest_overrides={field: value})
    kwargs = dict(corpus_fixture.stage_kwargs, asset_catalog=catalog)

    with pytest.raises(ValueError, match=field):
        stage_generated_batch(**kwargs)


def test_accept_requires_literal_confirmation_and_is_idempotent(corpus_fixture):
    staged = corpus_fixture.stage_valid_batch()

    with pytest.raises(ValueError, match="ACCEPT"):
        accept_generated_batch(
            corpus_fixture.root,
            staged.name,
            confirmation="yes",
            plan_path=corpus_fixture.plan_path,
            allow_provisional_asset_groups=True,
        )

    accepted = accept_generated_batch(
        corpus_fixture.root,
        staged.name,
        confirmation="ACCEPT",
        plan_path=corpus_fixture.plan_path,
        allow_provisional_asset_groups=True,
    )
    ledger_before = (corpus_fixture.root / "accepted-ledger.jsonl").read_bytes()

    assert accepted == corpus_fixture.root / "accepted" / staged.name
    assert not staged.exists()
    assert len(_read_jsonl(corpus_fixture.root / "queries.jsonl")) == 25
    assert (
        accept_generated_batch(
            corpus_fixture.root,
            staged.name,
            confirmation="ACCEPT",
            plan_path=corpus_fixture.plan_path,
            allow_provisional_asset_groups=True,
        )
        == accepted
    )
    assert (corpus_fixture.root / "accepted-ledger.jsonl").read_bytes() == ledger_before


def test_accept_rejects_legacy_out_of_order_staging(corpus_fixture):
    _accept_valid_batch(corpus_fixture)
    staged = stage_generated_batch(
        **_stage_kwargs_for_batch(corpus_fixture, "dev-mini-002")
    )
    (corpus_fixture.root / "accepted-ledger.jsonl").unlink()

    with pytest.raises(ValueError, match="active plan order.*dev-mini-001"):
        accept_generated_batch(
            corpus_fixture.root,
            staged.name,
            confirmation="ACCEPT",
            plan_path=corpus_fixture.plan_path,
            allow_provisional_asset_groups=True,
        )

    assert staged.is_dir()


def test_reject_preserves_content_reason_and_allocates_revision_two(corpus_fixture):
    staged = corpus_fixture.stage_valid_batch()
    results_before = (staged / "results.jsonl").read_bytes()

    rejected = reject_generated_batch(
        corpus_fixture.root, staged.name, reason="机械拒绝原因"
    )

    assert rejected == corpus_fixture.root / "rejected" / staged.name
    assert (rejected / "results.jsonl").read_bytes() == results_before
    assert json.loads((rejected / "reason.json").read_text(encoding="utf-8")) == {
        "reason": "机械拒绝原因"
    }
    corpus_fixture.write_draft()
    second = stage_generated_batch(**corpus_fixture.stage_kwargs)
    assert second.name == "dev-mini-001-r2"


def test_reject_resumes_after_reason_was_written_before_publish_failure(
    corpus_fixture, monkeypatch
):
    staged = corpus_fixture.stage_valid_batch()
    original_publish = batches.atomic_publish_new_directory

    def fail_publish_once(source: Path, destination: Path) -> Path:
        raise PermissionError(13, "simulated sharing lock")

    monkeypatch.setattr(
        batches, "atomic_publish_new_directory", fail_publish_once
    )
    with pytest.raises(PermissionError, match="simulated sharing lock"):
        reject_generated_batch(
            corpus_fixture.root, staged.name, reason="表达过度具体"
        )

    reason_path = staged / "reason.json"
    assert json.loads(reason_path.read_text(encoding="utf-8")) == {
        "reason": "表达过度具体"
    }

    monkeypatch.setattr(
        batches, "atomic_publish_new_directory", original_publish
    )
    rejected = reject_generated_batch(
        corpus_fixture.root, staged.name, reason="表达过度具体"
    )

    assert rejected == corpus_fixture.root / "rejected" / staged.name
    assert not staged.exists()


def test_reject_refuses_different_reason_after_partial_publish_failure(
    corpus_fixture, monkeypatch
):
    staged = corpus_fixture.stage_valid_batch()

    def fail_publish_once(source: Path, destination: Path) -> Path:
        raise PermissionError(13, "simulated sharing lock")

    monkeypatch.setattr(
        batches, "atomic_publish_new_directory", fail_publish_once
    )
    with pytest.raises(PermissionError, match="simulated sharing lock"):
        reject_generated_batch(
            corpus_fixture.root, staged.name, reason="原始拒绝原因"
        )

    with pytest.raises(FileExistsError):
        reject_generated_batch(
            corpus_fixture.root, staged.name, reason="改写后的拒绝原因"
        )

    assert json.loads((staged / "reason.json").read_text(encoding="utf-8")) == {
        "reason": "原始拒绝原因"
    }


@pytest.mark.parametrize("target", ["results", "plan", "seeds"])
def test_accept_revalidates_every_content_hash(corpus_fixture, target):
    staged = corpus_fixture.stage_valid_batch()
    if target == "results":
        (staged / "results.jsonl").write_bytes(b'{"tampered":true}\n')
    elif target == "plan":
        corpus_fixture.plan_path.write_bytes(b'{"tampered":true}\n')
    else:
        (corpus_fixture.root / "seeds/accepted/seed_examples.json").write_bytes(
            b'{"tampered":true}\n'
        )

    with pytest.raises(ValueError, match="哈希"):
        accept_generated_batch(
            corpus_fixture.root,
            staged.name,
            confirmation="ACCEPT",
            plan_path=corpus_fixture.plan_path,
            allow_provisional_asset_groups=True,
        )


def test_accept_failure_restores_staging_and_ledger(corpus_fixture, monkeypatch):
    staged = corpus_fixture.stage_valid_batch()

    def fail_publish(*args, **kwargs):
        raise OSError("mechanical publish failure")

    monkeypatch.setattr(
        "skillchain.synthesis.batches.publish_staged_directory_and_file",
        fail_publish,
    )
    with pytest.raises(OSError, match="mechanical publish failure"):
        accept_generated_batch(
            corpus_fixture.root,
            staged.name,
            confirmation="ACCEPT",
            plan_path=corpus_fixture.plan_path,
            allow_provisional_asset_groups=True,
        )

    assert staged.is_dir()
    assert not (corpus_fixture.root / "accepted-ledger.jsonl").exists()


def test_rebuild_repairs_only_derived_queries_file(corpus_fixture):
    staged = corpus_fixture.stage_valid_batch()
    accept_generated_batch(
        corpus_fixture.root,
        staged.name,
        confirmation="ACCEPT",
        plan_path=corpus_fixture.plan_path,
        allow_provisional_asset_groups=True,
    )
    accepted_results = (
        corpus_fixture.root / "accepted" / staged.name / "results.jsonl"
    ).read_bytes()
    (corpus_fixture.root / "queries.jsonl").write_bytes(b"tampered\n")

    rebuilt = rebuild_accepted_queries(corpus_fixture.root)

    assert rebuilt.read_bytes() == accepted_results
    assert (
        corpus_fixture.root / "accepted" / staged.name / "results.jsonl"
    ).read_bytes() == accepted_results


def test_verify_accepted_corpus_returns_the_verified_derived_queries(corpus_fixture):
    accepted = _accept_valid_batch(corpus_fixture)

    queries = verify_accepted_corpus(corpus_fixture.root)

    assert len(queries) == 25
    assert [query.query_id for query in queries] == [
        row["query_id"] for row in _read_jsonl(accepted / "results.jsonl")
    ]


@pytest.mark.parametrize("artifact", ["ledger", "manifest", "results"])
def test_verify_accepted_corpus_rejects_noncanonical_authoritative_files(
    corpus_fixture, artifact
):
    accepted = _accept_valid_batch(corpus_fixture)
    paths = {
        "ledger": corpus_fixture.root / "accepted-ledger.jsonl",
        "manifest": accepted / "manifest.json",
        "results": accepted / "results.jsonl",
    }
    path = paths[artifact]
    path.write_bytes(b" " + path.read_bytes())

    with pytest.raises(ValueError, match="canonical"):
        verify_accepted_corpus(corpus_fixture.root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_id", "other-r1"),
        ("base_batch_id", "other"),
        ("revision", 2),
        ("count", 24),
        ("results_sha256", "f" * 64),
        ("plan_sha256", "e" * 64),
        ("seed_set_sha256", "d" * 64),
    ],
)
def test_verify_accepted_corpus_rejects_any_ledger_batch_field_mismatch(
    corpus_fixture, field, value
):
    _accept_valid_batch(corpus_fixture)
    ledger_path = corpus_fixture.root / "accepted-ledger.jsonl"
    entries = _read_jsonl(ledger_path)
    entries[0][field] = value
    ledger_path.write_bytes(canonical_jsonl_bytes(entries))

    with pytest.raises(ValueError, match="ledger"):
        verify_accepted_corpus(corpus_fixture.root)


@pytest.mark.parametrize("fault", ["missing", "unexpected", "non_directory"])
def test_verify_accepted_corpus_rejects_accepted_directory_set_drift(
    corpus_fixture, fault
):
    accepted = _accept_valid_batch(corpus_fixture)
    accepted_root = corpus_fixture.root / "accepted"
    if fault == "missing":
        accepted.rename(corpus_fixture.root / "detached-batch")
    elif fault == "unexpected":
        (accepted_root / "untracked-r1").mkdir()
    else:
        (accepted_root / "untracked-file").write_text("drift", encoding="utf-8")

    with pytest.raises(ValueError, match="accepted 目录"):
        verify_accepted_corpus(corpus_fixture.root)


def test_verify_accepted_corpus_checks_results_sha_and_count(corpus_fixture):
    accepted = _accept_valid_batch(corpus_fixture)
    results_path = accepted / "results.jsonl"
    rows = _read_jsonl(results_path)
    rows[0]["text"] = "通过 canonical 编码但未被 manifest 授权的篡改"
    rows[0]["turns"][-1]["content"] = rows[0]["text"]
    results_path.write_bytes(canonical_jsonl_bytes(rows))

    with pytest.raises(ValueError, match="SHA256"):
        verify_accepted_corpus(corpus_fixture.root)

    rows.pop()
    results_bytes = canonical_jsonl_bytes(rows)
    results_sha256 = sha256_bytes(results_bytes)
    results_path.write_bytes(results_bytes)
    manifest_path = accepted / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results_sha256"] = results_sha256
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    ledger_path = corpus_fixture.root / "accepted-ledger.jsonl"
    entries = _read_jsonl(ledger_path)
    entries[0]["results_sha256"] = results_sha256
    ledger_path.write_bytes(canonical_jsonl_bytes(entries))
    (corpus_fixture.root / "queries.jsonl").write_bytes(results_bytes)

    with pytest.raises(ValueError, match="count"):
        verify_accepted_corpus(corpus_fixture.root)


def test_verify_accepted_corpus_proves_derived_queries_follow_ledger_order(
    corpus_fixture,
):
    first = _accept_valid_batch(corpus_fixture)
    first_results = (first / "results.jsonl").read_bytes()
    second_rows = _read_jsonl(first / "results.jsonl")
    for row in second_rows:
        row["text"] = "第二批-" + row["text"]
        row["turns"][-1]["content"] = row["text"]
        row["synthesis_batch_id"] = "dev-mini-002-r1"
    second_results = canonical_jsonl_bytes(second_rows)
    second_sha256 = sha256_bytes(second_results)
    second = corpus_fixture.root / "accepted/dev-mini-002-r1"
    second.mkdir()
    (second / "results.jsonl").write_bytes(second_results)
    second_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest.update(
        {
            "base_batch_id": "dev-mini-002",
            "revision": 1,
            "results_sha256": second_sha256,
        }
    )
    (second / "manifest.json").write_bytes(canonical_json_bytes(second_manifest))
    ledger_path = corpus_fixture.root / "accepted-ledger.jsonl"
    entries = _read_jsonl(ledger_path)
    second_entry = dict(entries[0])
    second_entry.update(
        {
            "batch_id": "dev-mini-002-r1",
            "base_batch_id": "dev-mini-002",
            "revision": 1,
            "results_sha256": second_sha256,
        }
    )
    ledger_path.write_bytes(canonical_jsonl_bytes([entries[0], second_entry]))
    derived_path = corpus_fixture.root / "queries.jsonl"
    derived_path.write_bytes(first_results + second_results)
    assert len(verify_accepted_corpus(corpus_fixture.root)) == 50

    tampered_derived = second_results + first_results
    derived_path.write_bytes(tampered_derived)
    with pytest.raises(ValueError, match="ledger 顺序"):
        verify_accepted_corpus(corpus_fixture.root)
    assert derived_path.read_bytes() == tampered_derived


@pytest.mark.parametrize("artifact", ["ledger", "manifest", "results", "derived"])
def test_verify_accepted_corpus_fails_closed_when_an_artifact_is_missing(
    corpus_fixture, artifact
):
    accepted = _accept_valid_batch(corpus_fixture)
    paths = {
        "ledger": corpus_fixture.root / "accepted-ledger.jsonl",
        "manifest": accepted / "manifest.json",
        "results": accepted / "results.jsonl",
        "derived": corpus_fixture.root / "queries.jsonl",
    }
    paths[artifact].unlink()

    with pytest.raises(ValueError, match="不存在|普通文件"):
        verify_accepted_corpus(corpus_fixture.root)


def test_status_reports_next_unaccepted_base_batch(corpus_fixture):
    staged = corpus_fixture.stage_valid_batch()
    accept_generated_batch(
        corpus_fixture.root,
        staged.name,
        confirmation="ACCEPT",
        plan_path=corpus_fixture.plan_path,
        allow_provisional_asset_groups=True,
    )

    status = corpus_status(corpus_fixture.root, plan_path=corpus_fixture.plan_path)

    assert status["accepted_batches"] == 1
    assert status["accepted_queries"] == 25
    assert status["next_batch_id"] == "dev-mini-002"
    assert status["next_revision"] == 1
    assert status["seed_status"] == "accepted"
