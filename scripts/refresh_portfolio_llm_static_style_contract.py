"""Refresh only the Style Skill after a frozen ToolSpec contract change.

The operation is deterministic, create-only, and performs no provider calls.
It preserves the accepted LLMStatic draft for five capabilities, derives the
Style draft from the current frozen TaskSpec/ToolSpec, compiles all six through
the trusted compiler, and proves that the five unrelated Skill artifacts are
byte-identical to the source Bank.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from skillchain.evaluation.portfolio_static_opt_runtime import (  # noqa: E402
    STATIC_CONTRACT_REFRESH_POLICY_VERSION,
)
from skillchain.evaluation.portfolio_treatments import (  # noqa: E402
    compile_portfolio_contract_refreshed_llm_static_bank,
)
from skillchain.static_authoring import (  # noqa: E402
    AuthoringDraftBundle,
    AuthoringInput,
    CanonicalSpecification,
    StaticBankArtifact,
    build_authoring_draft_bundle,
    build_spec_draft_bundle,
)
from skillchain.synthesis.store import (  # noqa: E402
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.task_spec import (  # noqa: E402
    load_mvp_task_specification_v1,
)
from skillchain.tools.registry import (  # noqa: E402
    build_mvp_registry_spec,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STYLE_CAPABILITY = "product.style_recommendation"
_OUTPUT_SEMANTIC = "semantic-authoring-input.json"
_OUTPUT_DRAFT = "contract-refresh-draft.json"
_OUTPUT_BANK = "bank-llm_static.json"
_OUTPUT_RECEIPT = "static-contract-refresh-receipt.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--source-bank-file-sha256", required=True)
    parser.add_argument("--source-semantic-authoring-input", type=Path, required=True)
    parser.add_argument("--source-semantic-authoring-input-file-sha256", required=True)
    parser.add_argument("--source-draft", type=Path, required=True)
    parser.add_argument("--source-draft-file-sha256", required=True)
    parser.add_argument("--tool-registry-runtime-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _read_canonical_model(path: Path, digest: str, model_type, *, label: str):
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} external SHA-256 is invalid")
    content = read_stable_regular_file(path, label=label)
    if sha256_bytes(content) != digest:
        raise ValueError(f"{label} external digest mismatch")
    raw = parse_canonical_json(content, label=label)
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise ValueError(f"{label} is not a canonical object")
    return model_type.model_validate(raw, strict=True), content


def _canonical_specification(
    *,
    kind: str,
    version: str,
    identity_sha256: str,
    value,
) -> CanonicalSpecification:
    content = canonical_json_bytes(value.model_dump(mode="json"))
    return CanonicalSpecification.model_validate(
        {
            "specification_kind": kind,
            "version": version,
            "identity_sha256": identity_sha256,
            "canonical_json": content.decode("utf-8"),
            "bytes_sha256": sha256_bytes(content),
        },
        strict=True,
    )


def build_contract_refresh(
    *,
    old_semantic: AuthoringInput,
    old_draft: AuthoringDraftBundle,
    old_bank: StaticBankArtifact,
    old_semantic_file_sha256: str,
    old_draft_file_sha256: str,
    old_bank_file_sha256: str,
    tool_registry_runtime_sha256: str,
) -> tuple[AuthoringInput, AuthoringDraftBundle, StaticBankArtifact, dict]:
    """Build and audit the deterministic one-capability contract refresh."""

    if _SHA256.fullmatch(tool_registry_runtime_sha256) is None:
        raise ValueError("tool registry runtime SHA-256 is invalid")
    rebound_source_draft = build_authoring_draft_bundle(
        authoring_input_sha256=old_semantic.input_sha256,
        drafts=old_draft.drafts,
    )
    if (
        old_bank.construction_identity_sha256 != rebound_source_draft.bundle_sha256
        or old_bank.tool_registry_sha256 != old_semantic.tool_registry.identity_sha256
        or old_bank.baseline_kind != "llm_static"
    ):
        raise ValueError("source LLMStatic lineage is inconsistent")

    task_specification = load_mvp_task_specification_v1()
    registry = build_mvp_registry_spec(include_multi_product=True)
    semantic_payload = old_semantic.model_dump(mode="json")
    semantic_payload["task_specification"] = _canonical_specification(
        kind="task_specification",
        version=task_specification.task_spec_version,
        identity_sha256=task_specification.task_spec_sha256,
        value=task_specification,
    ).model_dump(mode="json")
    semantic_payload["tool_registry"] = _canonical_specification(
        kind="tool_registry",
        version=f"mvp-tool-registry-v{registry.manifest.schema_version}",
        identity_sha256=registry.registry_sha256,
        value=registry.manifest,
    ).model_dump(mode="json")
    semantic_payload.pop("input_sha256", None)
    semantic = AuthoringInput.model_validate(
        {
            **semantic_payload,
            "input_sha256": sha256_bytes(canonical_json_bytes(semantic_payload)),
        },
        strict=True,
    )

    spec_bundle = build_spec_draft_bundle(semantic)
    source_drafts = {item.capability_id: item for item in old_draft.drafts}
    current_style = next(
        item for item in spec_bundle.drafts if item.capability_id == _STYLE_CAPABILITY
    )
    if set(source_drafts) != {item.capability_id for item in spec_bundle.drafts}:
        raise ValueError("source and current TaskSpec capability sets differ")
    source_drafts[_STYLE_CAPABILITY] = current_style
    draft = build_authoring_draft_bundle(
        authoring_input_sha256=semantic.input_sha256,
        drafts=tuple(source_drafts[key] for key in sorted(source_drafts)),
    )
    bank = compile_portfolio_contract_refreshed_llm_static_bank(
        semantic,
        draft,
        tool_registry_runtime_sha256=tool_registry_runtime_sha256,
    )

    old_skills = {item.capability_id: item for item in old_bank.skills}
    new_skills = {item.capability_id: item for item in bank.skills}
    if set(old_skills) != set(new_skills) or len(new_skills) != 6:
        raise ValueError("contract refresh must retain exactly six capabilities")
    unchanged: dict[str, str] = {}
    unchanged_rows: list[dict[str, str]] = []
    for capability_id in sorted(old_skills):
        if capability_id == _STYLE_CAPABILITY:
            continue
        old_bytes = canonical_json_bytes(
            old_skills[capability_id].model_dump(mode="json")
        )
        new_bytes = canonical_json_bytes(
            new_skills[capability_id].model_dump(mode="json")
        )
        if old_bytes != new_bytes:
            raise ValueError(f"unrelated Skill changed: {capability_id}")
        slug = new_skills[capability_id].slug
        unchanged[slug] = new_skills[capability_id].skill_sha256
        unchanged_rows.append(
            {
                "capability_id": capability_id,
                "slug": slug,
                "old_skill_sha256": old_skills[capability_id].skill_sha256,
                "new_skill_sha256": new_skills[capability_id].skill_sha256,
                "canonical_bytes_sha256": sha256_bytes(new_bytes),
            }
        )
    old_style = old_skills[_STYLE_CAPABILITY]
    new_style = new_skills[_STYLE_CAPABILITY]
    if old_style.skill_sha256 == new_style.skill_sha256 or canonical_json_bytes(
        old_style.model_dump(mode="json")
    ) == canonical_json_bytes(new_style.model_dump(mode="json")):
        raise ValueError("Style Skill did not change with ToolSpec 2.3")

    semantic_bytes = semantic.canonical_bytes()
    draft_bytes = draft.canonical_bytes()
    bank_bytes = bank.canonical_bytes()
    receipt_payload = {
        "schema_version": 1,
        "kind": "portfolio-static-contract-refresh-receipt",
        "policy_version": STATIC_CONTRACT_REFRESH_POLICY_VERSION,
        "status": "completed_zero_call",
        "refresh_scope": "style_skill_only",
        "algorithm_gain_eligible": False,
        "provider_calls": 0,
        "old_semantic_authoring_input_file_sha256": old_semantic_file_sha256,
        "old_semantic_authoring_input_sha256": old_semantic.input_sha256,
        "new_semantic_authoring_input_file": _OUTPUT_SEMANTIC,
        "new_semantic_authoring_input_file_sha256": sha256_bytes(semantic_bytes),
        "new_semantic_authoring_input_sha256": semantic.input_sha256,
        "old_draft_file_sha256": old_draft_file_sha256,
        "old_draft_sha256": old_draft.bundle_sha256,
        "old_rebound_draft_sha256": rebound_source_draft.bundle_sha256,
        "new_draft_file": _OUTPUT_DRAFT,
        "new_draft_file_sha256": sha256_bytes(draft_bytes),
        "new_draft_sha256": draft.bundle_sha256,
        "old_bank_file_sha256": old_bank_file_sha256,
        "old_bank_sha256": old_bank.bank_sha256,
        "new_bank_file": _OUTPUT_BANK,
        "new_bank_file_sha256": sha256_bytes(bank_bytes),
        "new_bank_sha256": bank.bank_sha256,
        "old_style_skill_sha256": old_style.skill_sha256,
        "new_style_skill_sha256": new_style.skill_sha256,
        "style_tool_version": "2.3.0",
        "style_tool_spec_sha256": next(
            item.spec_sha256
            for item in registry.specs()
            if item.name == "style_similar_search"
        ),
        "task_spec_sha256": task_specification.task_spec_sha256,
        "tool_registry_sha256": registry.registry_sha256,
        "tool_registry_runtime_sha256": tool_registry_runtime_sha256,
        "unchanged_skill_sha256s": dict(sorted(unchanged.items())),
        "unchanged_skills": unchanged_rows,
    }
    receipt = {
        **receipt_payload,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_payload)),
    }
    return semantic, draft, bank, receipt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        print("refresh-static-style: output directory exists", file=sys.stderr)
        return 2
    staging: Path | None = None
    try:
        old_bank, _ = _read_canonical_model(
            args.source_bank,
            args.source_bank_file_sha256,
            StaticBankArtifact,
            label="source LLMStatic Bank",
        )
        old_semantic, _ = _read_canonical_model(
            args.source_semantic_authoring_input,
            args.source_semantic_authoring_input_file_sha256,
            AuthoringInput,
            label="source semantic authoring input",
        )
        old_draft, _ = _read_canonical_model(
            args.source_draft,
            args.source_draft_file_sha256,
            AuthoringDraftBundle,
            label="source LLMStatic draft",
        )
        semantic, draft, bank, receipt = build_contract_refresh(
            old_semantic=old_semantic,
            old_draft=old_draft,
            old_bank=old_bank,
            old_semantic_file_sha256=args.source_semantic_authoring_input_file_sha256,
            old_draft_file_sha256=args.source_draft_file_sha256,
            old_bank_file_sha256=args.source_bank_file_sha256,
            tool_registry_runtime_sha256=args.tool_registry_runtime_sha256,
        )
        outputs = {
            _OUTPUT_SEMANTIC: semantic.canonical_bytes(),
            _OUTPUT_DRAFT: draft.canonical_bytes(),
            _OUTPUT_BANK: bank.canonical_bytes(),
            _OUTPUT_RECEIPT: canonical_json_bytes(receipt),
        }
        staging = new_staging_directory(args.output_dir)
        for name, content in outputs.items():
            (staging / name).write_bytes(content)
        atomic_publish_new_directory(staging, args.output_dir)
        staging = None
        print(
            canonical_json_bytes(
                {
                    "output_dir": str(args.output_dir.resolve()),
                    "semantic_authoring_input_sha256": semantic.input_sha256,
                    "bank_sha256": bank.bank_sha256,
                    "bank_file_sha256": sha256_bytes(outputs[_OUTPUT_BANK]),
                    "receipt_sha256": receipt["receipt_sha256"],
                    "receipt_file_sha256": sha256_bytes(outputs[_OUTPUT_RECEIPT]),
                    "style_skill_sha256": receipt["new_style_skill_sha256"],
                    "unchanged_skill_count": len(receipt["unchanged_skills"]),
                    "provider_calls": 0,
                }
            ).decode("utf-8")
        )
        return 0
    except Exception as error:
        print(f"refresh-static-style: {error}", file=sys.stderr)
        return 2
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
