"""Freeze a 30-round Core Fast S1 dual-policy campaign.

The generated specs all share one already-qualified Static parent and differ
only in their pre-registered Feedback/Creator mechanism.  Each spec is a
create-only experiment identity; no run artifact or previous candidate is
used as a parent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from skillchain.evaluation.core_fast.models import CoreFastSpec
from skillchain.tools.serialization import canonical_json_bytes


@dataclass(frozen=True)
class RoundDesign:
    slug: str
    selection_policy: str
    hypothesis: str
    directives: tuple[str, ...]


_COMMON = (
    "Return two independently attributable surfaces. Action policy may change only "
    "tool choice, public arguments, continuation, retry, and stopping. Response "
    "policy may change only visible-evidence claim selection, cards, answer sections, "
    "uncertainty, and fallback.",
    "Keep the routing Description and all five non-target Skills byte-exact. Either "
    "surface must inherit when its evidence does not support one bounded change. Do "
    "not mention evaluation labels, held-out data, judges, or private runtime paths.",
)


def _designs() -> tuple[RoundDesign, ...]:
    rows = (
        (
            "dual-baseline",
            "discovery-dual-policy-v5",
            "Baseline dual attribution",
            "Choose the highest-support causal failure cluster for each surface.",
        ),
        (
            "attributed-packet",
            "discovery-attributed-v4",
            "Explicit stage attribution",
            "Patch only when the first causal divergence belongs to the named surface.",
        ),
        (
            "supported-clusters",
            "discovery-supported-clusters-v3",
            "High-support clusters",
            "Prefer a recurring cluster; inherit for singleton or mixed-stage evidence.",
        ),
        (
            "contrastive-packet",
            "discovery-contrastive-v2",
            "Failure/success contrast",
            "Derive one state-conditional difference between failures and protected successes.",
        ),
        (
            "stratified-control",
            "discovery-stratified-v1",
            "Stratified control",
            "Use one minimal edit supported by multiple independent examples.",
        ),
        (
            "protected-first",
            "discovery-dual-policy-v5",
            "Protected-success-first",
            "State the protected branch first, then add the narrowest non-overlapping repair.",
        ),
        (
            "minimal-replace",
            "discovery-dual-policy-v5",
            "One-clause replacement",
            "Replace one conflicting clause; do not append a broad checklist or template.",
        ),
        (
            "risk-prediction",
            "discovery-dual-policy-v5",
            "Predicted risk constraint",
            "Predict gains and regressions from packet examples; inherit unless predicted gains are at least four times regressions.",
        ),
        (
            "action-first",
            "discovery-dual-policy-v5",
            "Action-policy emphasis",
            "Patch action only for repeated tool/argument/stop failures; keep response inherited unless it has independent evidence.",
        ),
        (
            "response-first",
            "discovery-dual-policy-v5",
            "Response-policy emphasis",
            "Patch response only after successful fixed tool evidence; keep action inherited unless it has independent evidence.",
        ),
        (
            "first-divergence",
            "discovery-attributed-v4",
            "First-divergence repair",
            "Name the earliest action or response divergence and repair only that decision.",
        ),
        (
            "counterfactual-rule",
            "discovery-contrastive-v2",
            "Counterfactual branch rule",
            "Write one if-and-only-if condition separating a failed trace from the nearest protected success.",
        ),
        (
            "failed-edit-memory",
            "discovery-dual-policy-v5",
            "Rejected-edit avoidance",
            "Avoid broad tool-first, copier, and fallback mandates that previously caused regressions; prefer a narrower exception.",
        ),
        (
            "regression-anchor",
            "discovery-dual-policy-v5",
            "Regression-anchor protection",
            "Treat every protected success and prior regression pattern as a hard semantic boundary for the edit.",
        ),
        (
            "positive-negative-joint",
            "discovery-supported-clusters-v3",
            "Joint success/failure induction",
            "Induce the smallest rule supported by both recurring failures and successes; inherit on conflict.",
        ),
        (
            "source-entailment",
            "discovery-dual-policy-v5",
            "Source entailment",
            "For response policy, select only claims directly entailed by visible evidence and abstain on unsupported user premises.",
        ),
        (
            "calibrated-abstention",
            "discovery-dual-policy-v5",
            "Calibrated abstention",
            "Add abstention only for a precisely observed no-evidence state; preserve useful supported answers.",
        ),
        (
            "argument-construction",
            "discovery-attributed-v4",
            "Typed public arguments",
            "For action policy, repair only recurring public argument construction failures without changing tool order.",
        ),
        (
            "continuation-stop",
            "discovery-attributed-v4",
            "Continuation and stopping",
            "For action policy, continue only when terminal evidence is absent and stop once sufficient public evidence exists.",
        ),
        (
            "fallback-boundary",
            "discovery-contrastive-v2",
            "Fallback boundary",
            "For response policy, separate successful-empty evidence from tool errors and nonempty evidence; preserve parent fallback wording.",
        ),
        (
            "card-selection",
            "discovery-dual-policy-v5",
            "Evidence-bounded cards",
            "Select cards only when their public evidence supports the answer; do not turn serializer rules into action rules.",
        ),
        (
            "knowledge-source",
            "discovery-supported-clusters-v3",
            "Knowledge/recipe source claims",
            "For source-backed capabilities, prefer literal supported claims and calibrated uncertainty; do not alter lookup actions without action evidence.",
        ),
        (
            "ocr-span-plan",
            "discovery-supported-clusters-v3",
            "OCR field-to-span planning",
            "For document response, map requested fields to one or more literal visible spans without normalization or inference.",
        ),
        (
            "product-association",
            "discovery-supported-clusters-v3",
            "Product association semantics",
            "For product response, preserve item-to-candidate and evidence-to-card associations while avoiding unsupported attributes.",
        ),
        (
            "smallest-stable-edit",
            "discovery-dual-policy-v5",
            "Stability-first edit",
            "Among supported repairs choose the shortest edit that leaves the greatest number of protected states unchanged.",
        ),
        (
            "capability-best-cluster",
            "discovery-dual-policy-v5",
            "Per-capability best cluster",
            "Independently choose each capability's strongest causal cluster; do not force the same surface across capabilities.",
        ),
        (
            "single-surface-conservative",
            "discovery-dual-policy-v5",
            "Single-surface conservative",
            "Patch at most one of action or response for this capability; inherit both if attribution is ambiguous.",
        ),
        (
            "complementary-surfaces",
            "discovery-dual-policy-v5",
            "Complementary dual surfaces",
            "Patch both surfaces only when evidence shows two independent failures and the edits do not duplicate responsibilities.",
        ),
        (
            "coverage-finalist",
            "discovery-attributed-v4",
            "Coverage-aware finalist",
            "Prefer independently supported edits across capabilities, while every local decision remains bounded by protected successes.",
        ),
        (
            "meta-finalist",
            "discovery-dual-policy-v5",
            "Campaign synthesis",
            "Use only stable mechanisms supported by prior campaign diagnostics; never copy a rejected candidate or relax the gates.",
        ),
    )
    return tuple(
        RoundDesign(
            slug=slug,
            selection_policy=policy,
            hypothesis=hypothesis,
            directives=(_COMMON[0], directive, _COMMON[1]),
        )
        for slug, policy, hypothesis, directive in rows
    )


def _load_spec(path: Path) -> CoreFastSpec:
    return CoreFastSpec.model_validate_json(path.read_bytes(), strict=True)


def freeze_campaign(*, base_spec: Path, output_dir: Path) -> dict[str, object]:
    base = _load_spec(base_spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, design in enumerate(_designs(), start=1):
        round_id = f"r{index}"
        experiment_id = f"portfolio-core-dual-policy-campaign-{round_id}-{design.slug}"
        settings = base.s1_settings.model_copy(
            update={
                "round_id": round_id,
                "feedback_mode": "fresh-per-round",
                "feedback_total_count": 48,
                "feedback_canary_count": 6,
                "feedback_format_retry_limit": 1,
                "feedback_selection_policy": design.selection_policy,
                "feedback_allocation": "balanced-six-capability",
                "proposal_mode": "six-capability-dual-policy-fanout-fanin-v3",
                "max_patched_capabilities": 6,
                "protected_capabilities": (),
                "creator_directives": design.directives,
                "prior_experiment_memory": {},
            }
        )
        limits = base.limits.model_copy(
            update={"max_feedback_calls": max(base.limits.max_feedback_calls, 96)}
        )
        spec = base.model_copy(
            update={
                "experiment_id": experiment_id,
                "s1_settings": settings,
                "limits": limits,
                "disclosures": (
                    *base.disclosures,
                    f"Campaign {round_id} hypothesis: {design.hypothesis}.",
                    "This round starts from the same frozen Static parent; no rejected candidate is a parent.",
                ),
            }
        )
        path = output_dir / f"{round_id}-{design.slug}.json"
        if path.exists():
            if path.read_bytes() != canonical_json_bytes(spec.model_dump(mode="json")):
                raise RuntimeError(f"campaign spec already differs: {path}")
        else:
            path.write_bytes(canonical_json_bytes(spec.model_dump(mode="json")))
        records.append(
            {
                "round_id": round_id,
                "slug": design.slug,
                "hypothesis": design.hypothesis,
                "feedback_selection_policy": design.selection_policy,
                "experiment_id": experiment_id,
                "spec": str(path),
                "output_root": str(
                    output_dir.parent / "runs" / f"{round_id}-{design.slug}"
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "s1-dual-policy-30-round-campaign",
        "base_spec": str(base_spec.resolve()),
        "static_results_sha256": base.opt_static_results_sha256,
        "static_bank_sha256": base.static_bank_file_sha256,
        "rounds": records,
    }
    manifest_path = output_dir.parent / "campaign-manifest.json"
    payload = canonical_json_bytes(manifest)
    if manifest_path.exists() and manifest_path.read_bytes() != payload:
        raise RuntimeError(f"campaign manifest already differs: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_bytes(payload)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_campaign(base_spec=args.base_spec, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
