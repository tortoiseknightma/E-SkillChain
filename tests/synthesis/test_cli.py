import json

from skillchain.schemas import Query
from skillchain.synthesis.cli import main
from skillchain.synthesis.labeling import run_cross_review
from skillchain.synthesis.planning import activate_dev_plan
from conftest import make_query_v2


def test_status_json_is_machine_readable_for_empty_root(tmp_path, capsys):
    root = tmp_path / "queries"

    assert main(["--queries-root", str(root), "status", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["accepted_queries"] == 0
    assert output["seed_status"] == "missing"
    assert output["staged_seed_batch_ids"] == []
    assert output["next_batch_id"] is None


def test_guard_model_without_confirmation_is_write_free(tmp_path, capsys):
    root = tmp_path / "queries"

    assert main(["--queries-root", str(root), "guard-model"]) == 2

    assert "5.6 Sol Ultra" in capsys.readouterr().err
    assert not root.exists()


def test_guard_model_accepts_exact_display_name_without_writing(tmp_path):
    root = tmp_path / "queries"

    assert (
        main(
            [
                "--queries-root",
                str(root),
                "guard-model",
                "--confirmed-model",
                "5.6 Sol Ultra",
            ]
        )
        == 0
    )
    assert not root.exists()


def test_plan_dev_mini_activates_plan_and_show_next(fake_image_root, tmp_path, capsys):
    root = tmp_path / "queries"

    assert (
        main(
            [
                "--queries-root",
                str(root),
                "plan-dev-mini",
                "--data-root",
                str(fake_image_root),
                "--allow-provisional-asset-groups",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (root / "plans/dev_mini.json").is_file()
    assert (root / "plans/active.json").is_file()

    assert main(["--queries-root", str(root), "show-next", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["batch_id"] == "dev-mini-001"
    assert shown["revision"] == 1
    assert len(shown["items"]) == 25
    assert shown["plan_sha256"]


def test_cli_stages_accepts_and_reports_stats(corpus_fixture, capsys):
    activate_dev_plan(
        corpus_fixture.plan_path,
        corpus_fixture.root,
        allow_provisional_asset_groups=True,
    )
    corpus_fixture.write_draft()

    assert (
        main(
            [
                "--queries-root",
                str(corpus_fixture.root),
                "stage-batch",
                "--draft",
                str(corpus_fixture.draft_path),
                "--draft-manifest",
                str(corpus_fixture.draft_manifest_path),
                "--base-batch-id",
                "dev-mini-001",
                "--allow-provisional-asset-groups",
            ]
        )
        == 0
    )
    staged = json.loads(capsys.readouterr().out)
    assert staged["batch_id"] == "dev-mini-001-r1"

    assert (
        main(
            [
                "--queries-root",
                str(corpus_fixture.root),
                "accept-batch",
                "--batch-id",
                "dev-mini-001-r1",
                "--confirmation",
                "ACCEPT",
                "--allow-provisional-asset-groups",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["--queries-root", str(corpus_fixture.root), "stats", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["total"] == 25
    assert stats["label_status"] == {"auto": 25}
    assert sum(row["count"] for row in stats["split_intent_boundary"]) == 25


def test_arbitrate_cli_prints_absolute_image_context_and_records_choice(
    tmp_path, capsys, monkeypatch
):
    root = tmp_path / "queries"
    accepted = root / "queries.jsonl"
    accepted.parent.mkdir(parents=True)
    query = make_query_v2(
        query_id="dm-001",
        image_path="query_images/mechanical.jpg",
        text="机械占位-001",
        intent="exact_match",
        split="dev_mini",
        generator_batch_id="dev-mini-001",
    )
    accepted.write_text(query.model_dump_json() + "\n", encoding="utf-8")
    run_cross_review(
        accepted,
        output_dir=root / "labels",
        chat_fn=lambda *args, **kwargs: '{"intent":"exact_match","reason":"机械理由"}',
        allow_provisional_asset_groups=True,
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "multi_product")

    assert (
        main(
            [
                "--queries-root",
                str(root),
                "arbitrate",
                "--image-root",
                str(tmp_path),
                "--allow-provisional-asset-groups",
            ]
        )
        == 0
    )

    output_lines = capsys.readouterr().out.splitlines()
    context = json.loads(output_lines[0])
    assert context["image_path"] == str((tmp_path / query.image_path).resolve())
    labeled = Query.model_validate_json(
        (root / "labels/labeled_queries.jsonl").read_text(encoding="utf-8")
    )
    assert labeled.gt_intent == "multi_product"
    assert labeled.label_status == "arbitrated"


def test_arbitrate_cli_requires_explicit_utility_capability(
    tmp_path, capsys, monkeypatch
):
    root = tmp_path / "queries"
    accepted = root / "queries.jsonl"
    accepted.parent.mkdir(parents=True)
    query = make_query_v2(
        query_id="dm-001",
        image_path="query_images/utility.jpg",
        text="read this utility image",
        intent="utility",
        split="dev_mini",
        generator_batch_id="dev-mini-001",
    )
    accepted.write_text(query.model_dump_json() + "\n", encoding="utf-8")
    run_cross_review(
        accepted,
        output_dir=root / "labels",
        chat_fn=lambda *args, **kwargs: '{"intent":"exact_match","reason":"disagree"}',
        allow_provisional_asset_groups=True,
    )
    answers = iter(
        [
            "utility",
            "utility.recipe_guidance",
            "utility.document_reading,utility.recipe_guidance",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert (
        main(
            [
                "--queries-root",
                str(root),
                "arbitrate",
                "--image-root",
                str(tmp_path),
                "--allow-provisional-asset-groups",
            ]
        )
        == 0
    )

    capsys.readouterr()
    labeled = Query.model_validate_json(
        (root / "labels/labeled_queries.jsonl").read_text(encoding="utf-8")
    )
    assert labeled.canonical_capability == "utility.recipe_guidance"
    assert labeled.acceptable_capabilities == [
        "utility.document_reading",
        "utility.recipe_guidance",
    ]


def test_plan_full_cli_writes_plan_without_activating_it(
    fake_image_root, tmp_path, capsys
):
    for intent in (
        "exact_match",
        "multi_product",
        "divergent_rec",
        "encyclopedia",
        "utility",
    ):
        # exact_match needs 359 non-dev asset groups after the 50 locked-dev
        # assets are excluded; keep comfortable headroom for every intent.
        for index in range(61, 451):
            (fake_image_root / intent / f"image-{index:03d}.jpg").write_bytes(
                b"mechanical-image"
            )
    root = tmp_path / "queries"
    assert (
        main(
            [
                "--queries-root",
                str(root),
                "plan-dev-mini",
                "--data-root",
                str(fake_image_root),
                "--allow-provisional-asset-groups",
            ]
        )
        == 0
    )
    capsys.readouterr()
    active_before = (root / "plans/active.json").read_bytes()

    assert (
        main(
            [
                "--queries-root",
                str(root),
                "plan-full",
                "--data-root",
                str(fake_image_root),
                "--allow-provisional-asset-groups",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["count"] == 4500
    assert result["batches"] == 180
    assert (root / "plans/full.json").is_file()
    assert (root / "plans/active.json").read_bytes() == active_before


def test_activate_plan_full_cli_resumes_after_locked_dev_batches(
    full_activation_fixture, capsys
):
    assert (
        main(
            [
                "--queries-root",
                str(full_activation_fixture.queries_root),
                "activate-plan",
                "full",
                "--plan",
                str(full_activation_fixture.full_plan_path),
                "--allow-provisional-asset-groups",
            ]
        )
        == 0
    )

    activated = json.loads(capsys.readouterr().out)
    assert activated["scope"] == "full"
    assert (
        main(
            [
                "--queries-root",
                str(full_activation_fixture.queries_root),
                "status",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["next_batch_id"] == "full-009"


def test_split_full_cli_persists_assignments_and_freezes_test(
    labeled_split_fixture, capsys
):
    root = labeled_split_fixture.queries_root

    assert (
        main(
            [
                "--queries-root",
                str(root),
                "split-full",
                "--plan",
                str(labeled_split_fixture.full_plan_path),
                "--seed",
                "20260711",
                "--allow-provisional-asset-groups",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["split_counts"] == {
        "dev_mini": 200,
        "opt_pool": 2800,
        "test_frozen": 1000,
        "val": 500,
    }
    assert (root / "test_frozen.jsonl").is_file()
    assigned = [
        Query.model_validate_json(line)
        for line in (root / "labels/labeled_queries.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    assert sum(query.split == "test_frozen" for query in assigned) == 1000


def test_write_command_stops_before_changes_when_frozen_hash_is_tampered(
    fake_image_root, tmp_path, capsys
):
    root = tmp_path / "queries"
    root.mkdir()
    (root / "test_frozen.jsonl").write_bytes(b"tampered\n")
    (root / "test_frozen.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "query_schema_version": 2,
                "profile": "full",
                "split_sizes": {
                    "dev_mini": 200,
                    "opt_pool": 2800,
                    "val": 500,
                    "test_frozen": 1000,
                },
                "count": 1000,
                "sha256": "0" * 64,
                "assignment_sha256": "1" * 64,
                "seed": 20260711,
                "taxonomy_version": "ecommerce-coarse-v1",
                "full_plan_sha256": "2" * 64,
                "asset_catalog_sha256": None,
                "leakage_policy_version": "relative-path-plus-plan-groups-v1",
                "grouping_policy_version": "query-connected-components-v1",
                "group_fields": [
                    "leakage_group_id",
                    "boundary_group_id",
                    "template_family",
                    "generator_batch_id",
                ],
                "component_count": 1,
                "leakage_violations": 0,
                "intent_counts": {"exact_match": 1000},
                "capability_counts": {"product.exact_match": 1000},
                "boundary_count": 0,
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--queries-root",
                str(root),
                "plan-dev-mini",
                "--data-root",
                str(fake_image_root),
                "--dry-run",
                "--allow-provisional-asset-groups",
            ]
        )
        == 2
    )

    error = capsys.readouterr().err
    assert any(token in error for token in ("哈希", "canonical", "完整性"))
    assert not (root / "plans").exists()
