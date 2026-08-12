from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.prepare_core_fast_qwen37_lineage as lineage
from skillchain.evaluation.core_fast.models import (
    CallIntent,
    CallResult,
    S1Settings,
    load_core_fast_spec,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_SPEC = REPOSITORY_ROOT / "specs" / "core-experiment-fast-v1.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _fixed_samples() -> dict[str, list[dict[str, str]]]:
    source = json.loads(BASE_SPEC.read_text(encoding="utf-8"))
    return source["fixed_samples"]


def test_prepare_bootstrap_and_freeze_r1_spec(tmp_path: Path) -> None:
    bootstrap_spec_path = tmp_path / "specs" / "bootstrap.json"
    opt_path = tmp_path / "artifacts" / "static-opt800.jsonl"
    bootstrap = lineage.prepare_static_bootstrap_spec(
        base_spec_path=BASE_SPEC,
        output_spec_path=bootstrap_spec_path,
        experiment_id="qwen37-static-test",
        opt_destination=opt_path,
    )

    assert bootstrap.paths.opt_static_results == str(opt_path.resolve())
    assert bootstrap.s1_settings.round_id == "r1"
    assert bootstrap.s1_settings.feedback_mode == "fresh-per-round"
    assert bootstrap.s1_settings.feedback_total_count == 48
    assert bootstrap.s1_settings.creator_directives == ()
    assert bootstrap.s1_settings.required_patch_phrases == {}
    assert bootstrap.s1_settings.protected_capabilities == (
        "knowledge.visual_encyclopedia",
    )
    assert Path(bootstrap.paths.queries).is_absolute()

    rows = [{"query_id": f"opt-{index:04d}"} for index in range(800)]
    opt_path.parent.mkdir(parents=True)
    opt_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    bootstrap_result_path = tmp_path / "static-opt800-bootstrap.json"
    _write_json(
        bootstrap_result_path,
        {
            "schema_version": 1,
            "kind": "core-fast-static-opt800-bootstrap",
            "assistant_model": lineage.QWEN37_ASSISTANT_MODEL,
            "opt_static_results": str(opt_path.resolve()),
            "opt_static_results_sha256": sha256_bytes(opt_path.read_bytes()),
            "row_count": 800,
            "gcs_success_count": 200,
            "hard_error_count": 0,
            "fixed_samples": _fixed_samples(),
        },
    )
    r1_path = tmp_path / "specs" / "r1.json"
    r1 = lineage.freeze_r1_spec(
        bootstrap_spec_path=bootstrap_spec_path,
        bootstrap_result_path=bootstrap_result_path,
        output_spec_path=r1_path,
        experiment_id="qwen37-s1-r1-test",
        target_capability="product.exact_match",
        creator_directives=("Patch only evidence-grounded exact-match closure.",),
        required_patch_phrases=("supported candidate",),
    )

    assert load_core_fast_spec(r1_path) == r1
    assert r1.opt_static_results_sha256 == sha256_bytes(opt_path.read_bytes())
    assert r1.fixed_samples.model_dump(mode="json") == _fixed_samples()
    assert r1.s1_settings.feedback_mode == "fresh-per-round"
    assert r1.s1_settings.target_capabilities == ("product.exact_match",)
    assert r1.s1_settings.max_patched_capabilities == 1
    assert r1.s1_settings.protected_capabilities == tuple(
        sorted(set(lineage.CAPABILITIES) - {"product.exact_match"})
    )
    assert r1.s1_settings.required_patch_phrases == {
        "product.exact_match": ("supported candidate",)
    }
    with pytest.raises(FileExistsError):
        lineage.freeze_r1_spec(
            bootstrap_spec_path=bootstrap_spec_path,
            bootstrap_result_path=bootstrap_result_path,
            output_spec_path=r1_path,
            experiment_id="qwen37-s1-r1-test",
            target_capability="product.exact_match",
            creator_directives=("Patch only evidence-grounded exact-match closure.",),
        )


def _fresh_r1_root(root: Path, *, opt_sha256: str = "a" * 64) -> None:
    fixed = _fixed_samples()
    feedback_model = "qwen3.8-max"
    _write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "kind": "core-experiment-fast-run",
            "experiment_id": "qwen37-s1-r1-test",
            "input_sha256": {"opt_static_results": opt_sha256},
            "models": {"feedback": {"requested_model": feedback_model}},
            "fixed_samples": fixed,
            "s1_settings": {"round_id": "r1", "feedback_mode": "fresh"},
        },
    )
    for sample in fixed["canary12"]:
        query_id = sample["query_id"]
        call_id = lineage._safe_call_id(query_id)
        _write_json(
            root / "calls" / "feedback" / f"{call_id}.intent.json",
            CallIntent(
                call_id=call_id,
                role="feedback",
                purpose=f"S1 fixed canary Feedback {query_id}",
                requested_model=feedback_model,
                payload={
                    "operation": "strict_visual_feedback",
                    "query": {"query_id": query_id},
                    "baseline": {"query_id": query_id},
                    "sample_role": sample["role"],
                    "runtime_paths": {"opt_static_results": "source.jsonl"},
                    "no_replacement": True,
                },
            ).model_dump(mode="json"),
        )
        _write_json(
            root / "calls" / "feedback" / f"{call_id}.result.json",
            CallResult(
                call_id=call_id,
                role="feedback",
                status="success",
                requested_model=feedback_model,
                returned_model=feedback_model,
                schema_valid=True,
                output={"feedback": {"summary": query_id}},
            ).model_dump(mode="json"),
        )


def test_feedback_bundle_is_byte_exact_create_only_and_tamper_evident(
    tmp_path: Path,
) -> None:
    source = tmp_path / "r1"
    bundle = tmp_path / "bundle"
    target = tmp_path / "r2"
    _fresh_r1_root(source)

    digest = lineage.export_feedback_bundle(source_root=source, bundle_dir=bundle)
    manifest_bytes = (bundle / lineage.FEEDBACK_MANIFEST_NAME).read_bytes()
    assert digest == sha256_bytes(manifest_bytes)

    imported_digest = lineage.import_feedback_bundle(
        bundle_dir=bundle, target_root=target
    )
    assert imported_digest == digest
    assert (
        lineage.verify_feedback_import(bundle_dir=bundle, target_root=target) == digest
    )
    assert (
        target / "inputs" / lineage.FEEDBACK_MANIFEST_NAME
    ).read_bytes() == manifest_bytes
    for source_file in (bundle / "calls" / "feedback").glob("*.json"):
        target_file = target / "calls" / "feedback" / source_file.name
        assert target_file.read_bytes() == source_file.read_bytes()

    with pytest.raises(FileExistsError):
        lineage.import_feedback_bundle(bundle_dir=bundle, target_root=target)

    changed = next((target / "calls" / "feedback").glob("*.result.json"))
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(lineage.LineagePreparationError):
        lineage.verify_feedback_import(bundle_dir=bundle, target_root=target)


def test_freeze_reuse_round_binds_feedback_manifest(tmp_path: Path) -> None:
    with pytest.raises(
        lineage.LineagePreparationError, match="cross-round Feedback reuse is forbidden"
    ):
        lineage.freeze_reuse_round_spec(
            r1_spec_path=tmp_path / "r1.json",
            bundle_dir=tmp_path / "bundle",
            output_spec_path=tmp_path / "r2.json",
            experiment_id="fresh-r2",
            round_id="r2",
            target_capability="product.exact_match",
            creator_directives=("Use fresh Feedback.",),
        )


def test_feedback_export_rejects_incomplete_source(tmp_path: Path) -> None:
    source = tmp_path / "r1"
    _fresh_r1_root(source)
    next((source / "calls" / "feedback").glob("*.result.json")).unlink()

    with pytest.raises((FileNotFoundError, lineage.ArtifactFormatError)):
        lineage.export_feedback_bundle(
            source_root=source, bundle_dir=tmp_path / "bundle"
        )


def test_s1_round_identity_supports_forward_rounds() -> None:
    base = load_core_fast_spec(BASE_SPEC)
    payload = base.s1_settings.model_dump()
    assert (
        S1Settings.model_validate({**payload, "round_id": "r10"}, strict=True).round_id
        == "r10"
    )
    assert (
        S1Settings.model_validate({**payload, "round_id": "r11"}, strict=True).round_id
        == "r11"
    )

    parser = lineage.build_parser()
    command = [
        "freeze-reuse-round",
        "--r1-spec",
        "r1.json",
        "--bundle-dir",
        "feedback",
        "--output-spec",
        "r10.json",
        "--experiment-id",
        "qwen37-s1-r10-test",
        "--target-capability",
        "product.multi_search",
        "--creator-directive",
        "Copy one public mapping payload exactly.",
        "--round-id",
    ]
    assert parser.parse_args([*command, "r10"]).round_id == "r10"
    with pytest.raises(SystemExit):
        parser.parse_args([*command, "r11"])
