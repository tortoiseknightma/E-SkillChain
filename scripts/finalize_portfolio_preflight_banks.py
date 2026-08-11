"""Create deterministic post-smoke scaffold Banks for diagnostics only.

This script never invokes Creator/Optimizer/Refiner models and therefore must
not emit officially named LLMStatic/S1/S1+S2/Full treatment artifacts.  Its
output can exercise wiring after a smoke run, but the matrix launch gate must
reject it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.prepare_portfolio_launch import _load_active_inputs  # noqa: E402
from skillchain.evaluation.evaluator_outputs import (  # noqa: E402
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
)
from skillchain.static_authoring import (  # noqa: E402
    BankCapabilityBinding,
    StaticBankArtifact,
    StrictSkillArtifact,
)
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_SYSTEM_PROMPT,
    PORTFOLIO_TOOL_RUNTIME_POLICY,
    PortfolioRuntimeSources,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes  # noqa: E402


_ROUTE_DESCRIPTION = {
    "knowledge.visual_encyclopedia": (
        "Use when the user asks what a visible entity is or requests factual "
        "explanation; this is not a request to buy the pictured item."
    ),
    "product.exact_match": (
        "Use for the same product or 同款, including multiple sizes or variants "
        "of that one catalog product; do not treat variants as multiple objects."
    ),
    "product.multi_search": (
        "Use only when the user wants two or more distinct visible products "
        "separated and searched; multiple copies or variants of one product do "
        "not make this a multi-product request."
    ),
    "product.style_recommendation": (
        "Use for 类似、相似款、几款、推荐 or allowed style changes: the user "
        "wants diverse same-category alternatives or explicitly requested "
        "cross-category coordination items, not the identical product."
    ),
    "utility.document_reading": (
        "Use to transcribe, summarize, or extract visible document text."
    ),
    "utility.recipe_guidance": (
        "Use when the user asks how to prepare, cook, season, or serve a dish."
    ),
}


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bank(path: Path) -> StaticBankArtifact:
    return StaticBankArtifact.model_validate_json(path.read_bytes(), strict=True)


def _transform_skill(
    source: StrictSkillArtifact,
    *,
    slug_prefix: str,
    version: int,
    description: str | None,
    body_addition: str,
) -> StrictSkillArtifact:
    payload = {
        "slug": f"{slug_prefix}-{source.capability_id.replace('.', '-').replace('_', '-')}",
        "version": version,
        "description": description or source.description,
        "body": source.body.rstrip() + "\n\n" + body_addition.strip() + "\n",
        "static_refs": list(source.static_refs),
        "operators": list(source.operators),
        "capability_id": source.capability_id,
        "parent_skill_sha256": source.skill_sha256,
    }
    return StrictSkillArtifact.model_validate(
        {
            **payload,
            "skill_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }
    )


def _make_bank(
    source: StaticBankArtifact,
    *,
    variant: str,
    skills: tuple[StrictSkillArtifact, ...],
    registry,
    smoke_results_sha256: str,
) -> StaticBankArtifact:
    bindings = tuple(
        BankCapabilityBinding(
            capability_id=skill.capability_id,
            skill_slug=skill.slug,
        )
        for skill in skills
    )
    payload = {
        "schema_version": 3,
        "baseline_kind": "llm_static",
        "construction_identity_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "parent_bank_sha256": source.bank_sha256,
                    "policy": "portfolio-deterministic-post-smoke-scaffold-v1",
                    "smoke_results_sha256": smoke_results_sha256,
                    "variant": variant,
                }
            )
        ),
        "construction_identity_policy": "reviewed-draft-v1",
        "runtime_binding_policy": "registry-runtime-v2",
        "compiler": source.compiler.model_dump(mode="json"),
        "tool_registry_sha256": registry.registry_sha256,
        "tool_registry_runtime_sha256": registry.registry_runtime_sha256,
        "skills": [skill.model_dump(mode="json") for skill in skills],
        "capability_map": [
            binding.model_dump(mode="json") for binding in bindings
        ],
    }
    return StaticBankArtifact.model_validate(
        {**payload, "bank_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        print("finalize-portfolio-banks: output directory exists", file=sys.stderr)
        return 2
    smoke_summary_path = args.smoke_root / "summary.json"
    smoke_results_path = args.smoke_root / "results.jsonl"
    summary = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("query_count") != 25
        or summary.get("success_count") != 25
        or summary.get("error_count") != 0
        or summary.get("config") != "llm_static"
        or summary.get("results_file_sha256") != _file_sha(smoke_results_path)
    ):
        print("finalize-portfolio-banks: smoke is not a verified 25/25 run", file=sys.stderr)
        return 2
    rows = [
        json.loads(line)
        for line in smoke_results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    inputs = _load_active_inputs()
    query_by_id = {item.query_id: item for item in inputs.queries}
    route_mismatches = []
    for row in rows:
        expected = query_by_id[row["query_id"]].canonical_capability
        response = row["response"]
        actual = response["selected_capability"]
        if expected != actual:
            route_mismatches.append(
                {
                    "query_id": row["query_id"],
                    "query_text": query_by_id[row["query_id"]].text,
                    "expected_capability": expected,
                    "selected_capability": actual,
                    "tool_names": [
                        item["tool_name"] for item in response["tool_trace"]
                    ],
                }
            )
    clean = REPOSITORY_ROOT / "data" / "clean"
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.",
            dir=args.output_dir.parent,
        )
    )
    try:
        shutil.copy2(
            args.source_runtime_root / "recipe-evidence.jsonl",
            staging / "recipe-evidence.jsonl",
        )
        (staging / "system-prompt.txt").write_text(
            PORTFOLIO_SYSTEM_PROMPT, encoding="utf-8"
        )
        sources = PortfolioRuntimeSources(
            selection_manifest=clean / "query_images" / "selection-manifest.json",
            dataset_assets=clean / "query_images" / "dataset-assets.jsonl",
            runtime_catalog_assets=(
                clean / "portfolio-mini-asset-catalog-v3" / "assets.jsonl"
            ),
            rpc_scenes=clean / "rpc-multi-product-query-v1" / "scenes.jsonl",
            inaturalist_manifest=(
                clean
                / "portfolio-source-pools"
                / "encyclopedia-inaturalist"
                / "manifest.jsonl"
            ),
            recipe_evidence=staging / "recipe-evidence.jsonl",
        )
        runtime = build_portfolio_tool_runtime(sources)
        llm_static = _load_bank(
            args.source_runtime_root / "bank-llm_static.json"
        )
        s1_source = {
            item.capability_id: item for item in llm_static.skills
        }
        s1_skills = tuple(
            sorted(
                (
                    _transform_skill(
                        skill,
                        slug_prefix="s1",
                        version=2,
                        description=None,
                        body_addition=(
                            "## Failure-driven S1 refinement\n"
                            "Real preflight trajectories showed premature answers "
                            "and provider-added action metadata. Before answering, "
                            "complete the capability's evidence-producing operator; "
                            "emit only its exact declared arguments; treat a tool "
                            "error as a retry-or-fallback signal, never as evidence."
                        ),
                    )
                    for skill in s1_source.values()
                ),
                key=lambda item: item.slug,
            )
        )
        s1 = _make_bank(
            llm_static,
            variant="s1",
            skills=s1_skills,
            registry=runtime.registry,
            smoke_results_sha256=summary["results_file_sha256"],
        )
        s1_by_capability = {item.capability_id: item for item in s1.skills}
        s1s2_skills = tuple(
            sorted(
                (
                    _transform_skill(
                        skill,
                        slug_prefix="s1s2",
                        version=3,
                        description=_ROUTE_DESCRIPTION[skill.capability_id],
                        body_addition=(
                            "## Failure-driven S2 refinement\n"
                            + _ROUTE_DESCRIPTION[skill.capability_id]
                        ),
                    )
                    for skill in s1_by_capability.values()
                ),
                key=lambda item: item.slug,
            )
        )
        s1s2 = _make_bank(
            s1,
            variant="s1s2",
            skills=s1s2_skills,
            registry=runtime.registry,
            smoke_results_sha256=summary["results_file_sha256"],
        )
        s1s2_by_capability = {
            item.capability_id: item for item in s1s2.skills
        }
        full_skills = tuple(
            sorted(
                (
                    _transform_skill(
                        skill,
                        slug_prefix="full",
                        version=4,
                        description=skill.description,
                        body_addition=(
                            "## S3 body-refiner candidate\n"
                            "Lead with the directly useful result. Keep each card "
                            "or factual claim adjacent to its returned evidence "
                            "identifier, state uncertainty once, and remove repeated "
                            "procedure narration or unsupported visual detail."
                        ),
                    )
                    for skill in s1s2_by_capability.values()
                ),
                key=lambda item: item.slug,
            )
        )
        full = _make_bank(
            s1s2,
            variant="full",
            skills=full_skills,
            registry=runtime.registry,
            smoke_results_sha256=summary["results_file_sha256"],
        )
        banks = {
            "llm_static": llm_static,
            "s1": s1,
            "s1s2": s1s2,
            "full": full,
        }
        for name, bank in banks.items():
            (staging / f"diagnostic-bank-{name}-scaffold.json").write_bytes(
                bank.canonical_bytes()
            )
        report_payload = {
            "schema_version": 1,
            "kind": "portfolio-deterministic-post-smoke-scaffold-diagnostic",
            "formal_eligible": False,
            "official_matrix_eligible": False,
            "source_smoke_summary_sha256": summary["summary_sha256"],
            "source_smoke_results_sha256": summary["results_file_sha256"],
            "source_smoke_success_count": 25,
            "source_smoke_route_correct_count": 25 - len(route_mismatches),
            "source_smoke_route_mismatch_count": len(route_mismatches),
            "route_mismatches": route_mismatches,
            "s1_status": "not_invoked_deterministic_scaffold_only",
            "s2_status": "not_invoked_deterministic_scaffold_only",
            "s3_status": "not_invoked_deterministic_scaffold_only",
            "comparison_policy": "none_diagnostic_only",
        }
        report = {
            **report_payload,
            "report_sha256": sha256_bytes(canonical_json_bytes(report_payload)),
        }
        report_path = staging / "iteration-report.json"
        report_path.write_bytes(canonical_json_bytes(report))
        runner_path = REPOSITORY_ROOT / "src" / "skillchain" / "runners" / "assistant.py"
        llm_adapter_path = REPOSITORY_ROOT / "src" / "skillchain" / "llm.py"
        evaluator_outputs_path = (
            REPOSITORY_ROOT
            / "src"
            / "skillchain"
            / "evaluation"
            / "evaluator_outputs.py"
        )
        final_runtime_path = (
            REPOSITORY_ROOT
            / "src"
            / "skillchain"
            / "evaluation"
            / "final_runtime.py"
        )
        lock_payload = {
            "schema_version": 1,
            "kind": "portfolio-assistant-runtime-lock",
            "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
            "track": "portfolio",
            "formal_eligible": False,
            "formal_ineligible_reason": (
                "public-dataset-metadata-adapters-are-not-formal-tool-runtimes"
            ),
            "stage": "diagnostic_post_smoke_scaffold",
            "tool_registry_sha256": runtime.registry.registry_sha256,
            "tool_registry_runtime_sha256": runtime.registry.registry_runtime_sha256,
            "runtime_data_sha256": runtime.index.runtime_data_sha256,
            "source_sha256s": list(runtime.index.source_sha256s),
            "system_prompt_sha256": sha256_bytes(
                PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")
            ),
            "runner_file_sha256": _file_sha(runner_path),
            "llm_adapter_file_sha256": _file_sha(llm_adapter_path),
            "evaluator_outputs_file_sha256": _file_sha(
                evaluator_outputs_path
            ),
            "final_runtime_file_sha256": _file_sha(final_runtime_path),
            "final_judge_parser_policy_version": (
                FINAL_JUDGE_PARSER_POLICY_VERSION_V4
            ),
            "final_judge_parser_policy_sha256": (
                FINAL_JUDGE_PARSER_POLICY_SHA256_V4
            ),
            "source_smoke_results_sha256": summary["results_file_sha256"],
            "iteration_report_sha256": report["report_sha256"],
            "diagnostic_scaffold_bank_sha256s": {
                name: bank.bank_sha256 for name, bank in banks.items()
            },
            "bank_policy": "portfolio-deterministic-post-smoke-scaffold-v1",
            "official_matrix_eligible": False,
            "treatment_chain_status": "not_created",
            "model_calls_performed_in_bank_compilation": 0,
        }
        lock = {
            **lock_payload,
            "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(lock_payload)),
        }
        (staging / "runtime-lock.json").write_bytes(canonical_json_bytes(lock))
        files = {
            path.name: _file_sha(path)
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
        }
        output = {
            "formal_eligible": False,
            "official_matrix_eligible": False,
            "artifact_role": "deterministic_diagnostic_scaffold",
            "output_dir": str(args.output_dir),
            "runtime_lock_sha256": lock["runtime_lock_sha256"],
            "tool_registry_sha256": runtime.registry.registry_sha256,
            "tool_registry_runtime_sha256": runtime.registry.registry_runtime_sha256,
            "route_mismatch_count": len(route_mismatches),
            "files": files,
        }
        (staging / "summary.json").write_bytes(canonical_json_bytes(output))
        staging.replace(args.output_dir)
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"finalize-portfolio-banks: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
