"""Prepare a diagnostic Portfolio runtime with deterministic scaffold Banks.

The Banks emitted here are useful only for wiring/tool-runtime smoke tests.
They are deliberately named as scaffolds and the resulting runtime lock is
ineligible for the five-configuration treatment matrix.  Real LLMStatic/S1/
S2/S3 treatment Banks are built by the model-evolution pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.static_authoring import (  # noqa: E402
    BankCapabilityBinding,
    CompilerIdentity,
    StaticBankArtifact,
    StrictSkillArtifact,
)
from skillchain.evaluation.evaluator_outputs import (  # noqa: E402
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
)
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_SYSTEM_PROMPT,
    PORTFOLIO_TOOL_RUNTIME_POLICY,
    PortfolioRuntimeSources,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes  # noqa: E402


_CATEGORY_TERMS = {
    "Ayam_bakar": ("\u70e4\u9e21",),
    "Bibimbap": ("\u62cc\u996d",),
    "Biryani": ("\u5370\u5ea6", "\u9999\u996d"),
    "Chicken_curry": ("\u5496\u55b1\u9e21",),
    "Doufunao": ("\u8c46\u8150\u8111",),
    "Lebkuchen": ("\u59dc\u997c",),
    "Potato_salad": ("\u571f\u8c46\u6c99\u62c9",),
}


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _extract_recipe_evidence(
    archive: Path, source_manifest: Path
) -> tuple[bytes, int]:
    source_rows = _jsonl(source_manifest)
    category_rows: dict[str, list[dict]] = {}
    for row in source_rows:
        category_rows.setdefault(row["category"], []).append(row)
    missing_categories = set(_CATEGORY_TERMS) - set(category_rows)
    if missing_categories:
        raise ValueError(
            "ISIA source pool lacks recipe categories: "
            + ", ".join(sorted(missing_categories))
        )
    found: dict[str, dict] = {}
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.namelist()
        if len(members) != 1:
            raise ValueError("recipe archive must contain exactly one corpus file")
        with bundle.open(members[0]) as stream:
            for line_number, content in enumerate(stream, 1):
                for category, terms in _CATEGORY_TERMS.items():
                    if category in found:
                        continue
                    if all(term.encode("utf-8") in content for term in terms):
                        row = json.loads(content)
                        if not row.get("recipeIngredient") or not row.get(
                            "recipeInstructions"
                        ):
                            continue
                        found[category] = {
                            "category": category,
                            "name": str(row["name"]).strip(),
                            "dish": str(row.get("dish") or "").strip(),
                            "recipeIngredient": [
                                str(item).strip()
                                for item in row["recipeIngredient"]
                                if str(item).strip()
                            ],
                            "recipeInstructions": [
                                str(item).strip()
                                for item in row["recipeInstructions"]
                                if str(item).strip()
                            ],
                            "author": str(row.get("author") or "").strip() or None,
                            "source_record_id": f"recipe_corpus_full.json:{line_number}",
                            "source_uri": "https://www.xiachufang.com/",
                            "source_archive_sha256": _file_sha(archive),
                            "license_id": "LicenseRef-Xiachufang-Research-Corpus",
                        }
                if len(found) == len(_CATEGORY_TERMS):
                    break
    if set(found) != set(_CATEGORY_TERMS):
        raise ValueError(
            "recipe corpus lacks evidence for: "
            + ", ".join(sorted(set(_CATEGORY_TERMS) - set(found)))
        )
    rows = [found[category] for category in sorted(found)]
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows), len(rows)


def _skill_body(draft: dict, variant: str) -> str:
    lines = [
        f"# {draft['objective']}",
        "",
        "## Procedure",
    ]
    for step in draft["steps"]:
        lines.append(
            f"- Use `{step['tool_name']}`: {step['instruction']}"
        )
    lines.extend(
        [
            "",
            "## Fallback",
            draft["fallback_instruction"],
        ]
    )
    if variant in {"s1", "s1s2", "full"}:
        lines.extend(
            [
                "",
                "## Diagnostic S1-shaped scaffold",
                (
                    "Treat short or vague user wording as normal: infer only the "
                    "minimum likely intent supported by the image and turns, avoid "
                    "inventing extra constraints, obtain tool evidence before the "
                    "final answer, and prefer an explicit partial result over an "
                    "unsupported completion."
                ),
            ]
        )
    if variant in {"s1s2", "full"}:
        lines.extend(
            [
                "",
                "## Diagnostic S2-shaped scaffold",
                (
                    "Route by the requested outcome, not isolated keywords; keep "
                    "same-product lookup, multi-item decomposition, diverse style "
                    "recommendation, visual encyclopedia explanation, document "
                    "reading, and recipe guidance distinct."
                ),
            ]
        )
    if variant == "full":
        lines.extend(
            [
                "",
                "## Diagnostic S3-shaped scaffold",
                (
                    "Lead with the directly useful result, preserve evidence "
                    "identifiers beside claims or cards, state uncertainty once "
                    "without boilerplate, and do not repeat the procedure."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _bank(drafts: list[dict], variant: str, registry) -> StaticBankArtifact:
    skills: list[StrictSkillArtifact] = []
    bindings: list[BankCapabilityBinding] = []
    route_cues = {
        "knowledge.visual_encyclopedia": "Visual entity explanation with cited facts.",
        "product.exact_match": "Find the same catalog product.",
        "product.multi_search": "Separate and search multiple visible products.",
        "product.style_recommendation": (
            "Recommend evidence-backed same-category alternatives or requested "
            "cross-category coordination items."
        ),
        "utility.document_reading": "Read, transcribe, summarize, or extract document text.",
        "utility.recipe_guidance": "Identify a dish tentatively and provide cited cooking guidance.",
    }
    for draft in sorted(drafts, key=lambda item: item["capability_id"]):
        slug = (
            f"{variant.replace('_', '-')}-"
            + draft["capability_id"].replace(".", "-").replace("_", "-")
        )
        description = draft["objective"]
        if variant in {"s1s2", "full"}:
            description = route_cues[draft["capability_id"]]
        payload = {
            "slug": slug,
            "version": 1,
            "description": description,
            "body": _skill_body(draft, variant),
            "static_refs": [],
            "operators": sorted({step["tool_name"] for step in draft["steps"]}),
            "capability_id": draft["capability_id"],
            "parent_skill_sha256": None,
        }
        skill = StrictSkillArtifact.model_validate(
            {
                **payload,
                "skill_sha256": sha256_bytes(canonical_json_bytes(payload)),
            }
        )
        skills.append(skill)
        bindings.append(
            BankCapabilityBinding(
                capability_id=draft["capability_id"], skill_slug=slug
            )
        )
    construction = sha256_bytes(
        canonical_json_bytes(
            {
                "accepted_draft_bundle_sha256": (
                    "41fd1e636c060db028a298189b0db98847c06d5f0bd651c1074e8a0d3f4078b3"
                ),
                "policy": "portfolio-deterministic-diagnostic-scaffold-v1",
                "variant": variant,
            }
        )
    )
    payload = {
        "schema_version": 3,
        "baseline_kind": "llm_static",
        "construction_identity_sha256": construction,
        "construction_identity_policy": "reviewed-draft-v1",
        "runtime_binding_policy": "registry-runtime-v2",
        "compiler": CompilerIdentity(compiler_version="4.0.0").model_dump(mode="json"),
        "tool_registry_sha256": registry.registry_sha256,
        "tool_registry_runtime_sha256": registry.registry_runtime_sha256,
        "skills": [item.model_dump(mode="json") for item in skills],
        "capability_map": [item.model_dump(mode="json") for item in bindings],
    }
    return StaticBankArtifact.model_validate(
        {**payload, "bank_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        print("prepare-portfolio-runtime: output directory exists", file=sys.stderr)
        return 2
    clean = REPOSITORY_ROOT / "data" / "clean"
    raw = REPOSITORY_ROOT / "data" / "raw"
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.",
            dir=args.output_dir.parent,
        )
    )
    try:
        recipe_bytes, recipe_rows = _extract_recipe_evidence(
            raw / "recipes" / "xiachufang_recipe_corpus_full.zip",
            clean / "portfolio-source-pools" / "recipe-isia" / "manifest.jsonl",
        )
        recipe_path = staging / "recipe-evidence.jsonl"
        recipe_path.write_bytes(recipe_bytes)
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
            recipe_evidence=recipe_path,
        )
        runtime = build_portfolio_tool_runtime(sources)
        draft_bundle = json.loads(
            (
                REPOSITORY_ROOT
                / "runs"
                / "formal-authoring"
                / "llm-static-codex-primary-20260724-high-v5"
                / "pre-review-draft.json"
            ).read_text(encoding="utf-8")
        )
        scaffold_banks = {
            variant: _bank(draft_bundle["drafts"], variant, runtime.registry)
            for variant in ("llm_static", "s1", "s1s2", "full")
        }
        for variant, bank in scaffold_banks.items():
            (staging / f"diagnostic-bank-{variant}-scaffold.json").write_bytes(
                bank.canonical_bytes()
            )
        system_prompt_path = staging / "system-prompt.txt"
        system_prompt_path.write_text(PORTFOLIO_SYSTEM_PROMPT, encoding="utf-8")
        tool_bindings = [
            {
                "tool_name": spec.name,
                "tool_spec_sha256": spec.spec_sha256,
                "runtime_binding_sha256": runtime.registry.runtime_binding_sha256(
                    spec.name
                ),
            }
            for spec in runtime.registry.specs()
        ]
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
            "stage": "diagnostic_scaffold",
            "tool_registry_sha256": runtime.registry.registry_sha256,
            "tool_registry_runtime_sha256": (
                runtime.registry.registry_runtime_sha256
            ),
            "runtime_data_sha256": runtime.index.runtime_data_sha256,
            "source_sha256s": list(runtime.index.source_sha256s),
            "tool_bindings": tool_bindings,
            "recipe_evidence_rows": recipe_rows,
            "diagnostic_scaffold_bank_sha256s": {
                variant: bank.bank_sha256
                for variant, bank in scaffold_banks.items()
            },
            "bank_policy": "portfolio-deterministic-diagnostic-scaffold-v1",
            "official_matrix_eligible": False,
            "treatment_chain_status": "not_created",
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
            "model_calls_performed": 0,
        }
        lock = {
            **lock_payload,
            "runtime_lock_sha256": sha256_bytes(
                canonical_json_bytes(lock_payload)
            ),
        }
        lock_path = staging / "runtime-lock.json"
        lock_path.write_bytes(canonical_json_bytes(lock))
        files = {
            path.name: _file_sha(path)
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
        }
        summary = {
            "formal_eligible": False,
            "official_matrix_eligible": False,
            "artifact_role": "deterministic_diagnostic_scaffold",
            "model_calls_performed": 0,
            "output_dir": str(args.output_dir),
            "runtime_lock_sha256": lock["runtime_lock_sha256"],
            "tool_registry_sha256": runtime.registry.registry_sha256,
            "tool_registry_runtime_sha256": (
                runtime.registry.registry_runtime_sha256
            ),
            "files": files,
        }
        (staging / "summary.json").write_bytes(canonical_json_bytes(summary))
        staging.replace(args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"prepare-portfolio-runtime: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
