#!/usr/bin/env python3
"""Prepare one immutable active-model Core Fast S1 lineage.

This helper performs filesystem preparation only.  It never loads a runtime
adapter and therefore cannot make provider calls.  The historical filename is
retained so existing launch instructions remain valid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.evaluation.core_fast.models import (  # noqa: E402
    CAPABILITIES,
    S1_BODY_PATCH_TARGETS,
    CallIntent,
    CallResult,
    CoreFastSpec,
    load_core_fast_spec,
)
from skillchain import config  # noqa: E402
from skillchain.evaluation.core_fast.store import atomic_write_json  # noqa: E402
from skillchain.tools.serialization import (  # noqa: E402
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    parse_strict_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)


DEFAULT_SPEC = REPOSITORY_ROOT / "specs" / "core-experiment-fast-v1.json"
ACTIVE_ASSISTANT_MODEL = config.ASSISTANT_MODEL
FEEDBACK_MANIFEST_NAME = "feedback-source-manifest.json"
FEEDBACK_MANIFEST_KIND = "core-fast-feedback-reuse-source-v1"
STALE_DISCLOSURE_FRAGMENTS = (
    "Static opt800 has no successful",
    "S1 R2/R3 reuse the exact 12 Feedback",
    "bootstrap-only spec:",
    "Qwen3.7 Static v2 is the selected baseline input",
    "The recorded S1 R1-R10 optimization allowance is exhausted",
    "The model-generated action-response runtime requires a fresh symmetric Static",
    "This default spec is retained for input validation",
)


class LineagePreparationError(RuntimeError):
    """Raised when a lineage input cannot be safely frozen or reused."""


def _load_canonical_object(path: Path, *, label: str) -> dict[str, Any]:
    content = read_stable_regular_file(path, label=label)
    value = parse_canonical_json(content, label=label)
    if not isinstance(value, dict):
        raise LineagePreparationError(f"{label} must be a JSON object")
    return value


def _load_tracked_spec_object(path: Path) -> dict[str, Any]:
    """Load the human-formatted tracked spec without weakening artifacts."""

    content = read_stable_regular_file(path, label="base Core Fast spec")
    value = parse_strict_json(content, label="base Core Fast spec")
    if not isinstance(value, dict):
        raise LineagePreparationError("base Core Fast spec must be a JSON object")
    return value


def _validated_spec(payload: object) -> CoreFastSpec:
    return CoreFastSpec.model_validate_json(canonical_json_bytes(payload), strict=True)


def _absolute_path(value: str, *, base_dir: Path) -> str:
    expanded = Path(os.path.expandvars(value))
    return str(expanded if expanded.is_absolute() else (base_dir / expanded).resolve())


def _portable_paths(spec: CoreFastSpec, *, base_dir: Path) -> dict[str, object]:
    """Preserve path meaning when a generated spec is written elsewhere."""

    payload = spec.paths.model_dump(mode="json")
    for name, value in tuple(payload.items()):
        if isinstance(value, str) and value:
            payload[name] = _absolute_path(value, base_dir=base_dir)
        elif isinstance(value, list):
            payload[name] = [_absolute_path(item, base_dir=base_dir) for item in value]
    return payload


def _assert_active_model(spec: CoreFastSpec) -> None:
    assistant = spec.models["assistant"].requested_model
    route_only = spec.models["route_only"].requested_model
    if assistant != ACTIVE_ASSISTANT_MODEL or route_only != assistant:
        raise LineagePreparationError(
            "lineage preparation requires the pinned active Assistant and "
            "matching route-only model"
        )


def _clean_disclosures(values: Sequence[str]) -> list[str]:
    return [
        item
        for item in values
        if not any(fragment in item for fragment in STALE_DISCLOSURE_FRAGMENTS)
    ]


def prepare_static_bootstrap_spec(
    *,
    base_spec_path: Path,
    output_spec_path: Path,
    experiment_id: str,
    opt_destination: Path,
) -> CoreFastSpec:
    """Create a bootstrap-only spec for a fresh, create-only Static opt800."""

    base_spec_path = base_spec_path.resolve()
    # A bootstrap is the migration boundary between runtime contracts.  The
    # tracked default may intentionally still point at the last completed
    # lineage, whose Literal no longer validates after a contract version
    # bump.  Upgrade only the runtime identity before validating; all other
    # fields still come from the canonical tracked spec.
    base_payload = _load_tracked_spec_object(base_spec_path)
    runtime = base_payload.get("runtime")
    if not isinstance(runtime, dict):
        raise LineagePreparationError("base Core Fast spec lacks runtime settings")
    runtime["assistant_contract"] = "core-fast-model-generated-action-response-v1"
    gates = base_payload.get("gates")
    if not isinstance(gates, dict):
        raise LineagePreparationError("base Core Fast spec lacks gate settings")
    gates["s1_max_capability_drop_pp"] = 5.0
    base = _validated_spec(base_payload)
    _assert_active_model(base)
    destination = opt_destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)

    payload = base.model_dump(mode="json")
    payload["experiment_id"] = experiment_id
    payload["paths"] = _portable_paths(base, base_dir=base_spec_path.parent)
    payload["paths"]["opt_static_results"] = str(destination)
    # opt_static_results_sha256 intentionally remains the old lineage's valid
    # placeholder. CoreFastEngine.initialize_static_opt800 does not read it;
    # freeze_r1_spec replaces it with the observed fresh artifact SHA.
    payload["s1_settings"] = {
        "round_id": "r1",
        "feedback_mode": "fresh-per-round",
        "feedback_total_count": 48,
        "feedback_canary_count": 6,
        "feedback_format_retry_limit": 0,
        "feedback_selection_policy": "discovery-stratified-v1",
        "feedback_allocation": "balanced-six-capability",
        "target_capabilities": list(S1_BODY_PATCH_TARGETS),
        "proposal_mode": "six-capability-fanout-fanin-v2",
        "max_patched_capabilities": len(S1_BODY_PATCH_TARGETS),
        "protected_capabilities": [],
        "creator_directives": [],
        "required_patch_phrases": {},
    }
    payload["limits"]["max_feedback_calls"] = 48
    payload["limits"]["max_creator_calls"] = 8
    disclosures = _clean_disclosures(base.disclosures)
    disclosures.append(
        "bootstrap-only spec: opt_static_results_sha256 is a placeholder and "
        "must be replaced from static-opt800-bootstrap.json before S1."
    )
    payload["disclosures"] = disclosures

    frozen = _validated_spec(payload)
    atomic_write_json(output_spec_path.resolve(), frozen.model_dump(mode="json"))
    return frozen


def freeze_active_default_spec(
    *,
    base_spec_path: Path,
    bootstrap_result_path: Path,
    output_spec_path: Path,
    experiment_id: str,
) -> CoreFastSpec:
    """Bind completed active-model Static evidence without changing S1 policy."""

    base = load_core_fast_spec(base_spec_path.resolve())
    _assert_active_model(base)
    bootstrap = _load_canonical_object(
        bootstrap_result_path.resolve(), label="Static opt800 bootstrap"
    )
    if (
        bootstrap.get("kind") != "core-fast-static-opt800-bootstrap"
        or bootstrap.get("assistant_model") != ACTIVE_ASSISTANT_MODEL
        or bootstrap.get("assistant_contract") != base.runtime.assistant_contract
        or bootstrap.get("row_count") != 800
    ):
        raise LineagePreparationError("Static bootstrap identity differs")
    opt_path = Path(str(bootstrap.get("opt_static_results", ""))).resolve()
    if not opt_path.is_file():
        raise LineagePreparationError("fresh Static opt800 is missing")
    opt_sha = sha256_bytes(
        read_stable_regular_file(opt_path, label="fresh Static opt800")
    )
    if bootstrap.get("opt_static_results_sha256") != opt_sha:
        raise LineagePreparationError("fresh Static opt800 SHA differs")
    fixed_samples = bootstrap.get("fixed_samples")
    if not isinstance(fixed_samples, dict):
        raise LineagePreparationError("Static bootstrap lacks fixed samples")

    payload = base.model_dump(mode="json")
    payload["experiment_id"] = experiment_id
    payload["paths"]["opt_static_results"] = str(opt_path)
    payload["opt_static_results_sha256"] = opt_sha
    payload["fixed_samples"] = fixed_samples
    disclosures = _clean_disclosures(base.disclosures)
    disclosures.append(
        f"Active Static opt800 was freshly generated with {ACTIVE_ASSISTANT_MODEL}; "
        "the tracked spec binds its SHA and deterministic fixed samples."
    )
    payload["disclosures"] = list(dict.fromkeys(disclosures))
    frozen = _validated_spec(payload)
    atomic_write_json(output_spec_path.resolve(), frozen.model_dump(mode="json"))
    return frozen


def freeze_r1_spec(
    *,
    bootstrap_spec_path: Path,
    bootstrap_result_path: Path,
    output_spec_path: Path,
    experiment_id: str,
    target_capability: str | None,
    creator_directives: Sequence[str],
    required_patch_phrases: Sequence[str] = (),
    fanout: bool = False,
    round_id: str = "r1",
    feedback_total_count: int = 48,
    feedback_selection_policy: str = "discovery-stratified-v1",
    bounded_edit_surface: str = "author-fields",
    prior_experiment_roots: Sequence[Path] = (),
) -> CoreFastSpec:
    """Bind fresh Static to either one capability or six isolated branches."""

    bootstrap_spec_path = bootstrap_spec_path.resolve()
    source = load_core_fast_spec(bootstrap_spec_path)
    _assert_active_model(source)
    if fanout and (target_capability is not None or required_patch_phrases):
        raise LineagePreparationError(
            "fan-out R1 uses all Body capabilities and no shared required phrases"
        )
    if not fanout and target_capability not in S1_BODY_PATCH_TARGETS:
        raise LineagePreparationError(
            "R1 target must expose a model-generated Body surface"
        )
    if not creator_directives:
        raise LineagePreparationError("R1 requires at least one Creator directive")

    bootstrap = _load_canonical_object(
        bootstrap_result_path.resolve(), label="Static opt800 bootstrap"
    )
    if bootstrap.get("kind") != "core-fast-static-opt800-bootstrap":
        raise LineagePreparationError("unexpected Static opt800 bootstrap kind")
    if bootstrap.get("assistant_model") != ACTIVE_ASSISTANT_MODEL:
        raise LineagePreparationError("Static bootstrap Assistant model differs")
    if bootstrap.get("assistant_contract") != source.runtime.assistant_contract:
        raise LineagePreparationError("Static bootstrap Assistant contract differs")
    if bootstrap.get("row_count") != 800:
        raise LineagePreparationError("Static bootstrap must bind exactly 800 rows")

    configured_opt = Path(
        _absolute_path(
            source.paths.opt_static_results, base_dir=bootstrap_spec_path.parent
        )
    ).resolve()
    reported_opt = Path(str(bootstrap.get("opt_static_results", ""))).resolve()
    if reported_opt != configured_opt:
        raise LineagePreparationError(
            "Static bootstrap destination differs from the bootstrap spec"
        )
    opt_content = read_stable_regular_file(configured_opt, label="fresh Static opt800")
    opt_rows = parse_canonical_jsonl(opt_content, label="fresh Static opt800")
    if len(opt_rows) != 800:
        raise LineagePreparationError("fresh Static opt800 must contain 800 rows")
    observed_sha256 = sha256_bytes(opt_content)
    if bootstrap.get("opt_static_results_sha256") != observed_sha256:
        raise LineagePreparationError("fresh Static opt800 SHA differs from bootstrap")

    fixed_samples = bootstrap.get("fixed_samples")
    if not isinstance(fixed_samples, dict):
        raise LineagePreparationError("Static bootstrap lacks fixed samples")

    payload = source.model_dump(mode="json")
    payload["experiment_id"] = experiment_id
    payload["opt_static_results_sha256"] = observed_sha256
    payload["fixed_samples"] = fixed_samples
    phrases = sorted(set(required_patch_phrases))
    prior_memory: dict[str, list[dict[str, object]]] = {}
    for prior_root in prior_experiment_roots:
        resolved = prior_root.resolve()
        decision = _load_canonical_object(
            resolved / "decisions" / "s1.json", label="prior S1 decision"
        )
        metrics = decision.get("metrics")
        if not isinstance(metrics, dict) or not isinstance(
            metrics.get("fanout_branches"), list
        ):
            raise LineagePreparationError("prior S1 decision lacks fan-out evidence")
        if metrics.get("body_accessed") and decision.get("accepted") is True:
            raise LineagePreparationError(
                "accepted prior candidate is not failure memory"
            )
        for branch in metrics["fanout_branches"]:
            if (
                not isinstance(branch, dict)
                or branch.get("capability") not in CAPABILITIES
            ):
                raise LineagePreparationError("prior S1 branch identity is invalid")
            capability = str(branch["capability"])
            candidate_path = (
                resolved / "banks" / f"s1-branch-{capability}-candidate.json"
            )
            candidate_body = None
            if candidate_path.exists():
                candidate = _load_canonical_object(
                    candidate_path, label="prior S1 branch candidate"
                )
                skills = candidate.get("skills")
                if not isinstance(skills, list):
                    raise LineagePreparationError("prior S1 candidate lacks Skills")
                skill = next(
                    (
                        item
                        for item in skills
                        if isinstance(item, dict)
                        and item.get("capability_id") == capability
                    ),
                    None,
                )
                if not isinstance(skill, dict) or not isinstance(
                    skill.get("body"), str
                ):
                    raise LineagePreparationError("prior S1 candidate Skill is invalid")
                candidate_body = skill["body"]
            prior_memory.setdefault(capability, []).append(
                {
                    "capability": capability,
                    "source_experiment_id": decision.get("metrics", {}).get(
                        "experiment_id", resolved.name
                    ),
                    "source_round_id": decision.get("metrics", {}).get("round_id"),
                    "branch_status": branch.get("status"),
                    "branch_rejection_reason": branch.get("reason"),
                    "screen": branch.get("screen"),
                    "candidate_body": candidate_body,
                }
            )
    s1_settings = {
        "round_id": round_id,
        "feedback_mode": "fresh-per-round",
        "feedback_total_count": feedback_total_count,
        "feedback_canary_count": 6,
        "feedback_selection_policy": feedback_selection_policy,
        "creator_directives": list(dict.fromkeys(creator_directives)),
        "bounded_edit_surface": bounded_edit_surface,
        "prior_experiment_memory": prior_memory,
        "required_patch_phrases": (
            {target_capability: phrases}
            if not fanout and target_capability is not None and phrases
            else {}
        ),
    }
    if fanout:
        s1_settings.update(
            {
                "feedback_allocation": "balanced-six-capability",
                "target_capabilities": list(S1_BODY_PATCH_TARGETS),
                "proposal_mode": "six-capability-fanout-fanin-v2",
                "max_patched_capabilities": len(S1_BODY_PATCH_TARGETS),
                "protected_capabilities": [],
            }
        )
        payload["limits"]["max_creator_calls"] = 8
    else:
        assert target_capability is not None
        s1_settings.update(
            {
                "feedback_allocation": "target-focused",
                "target_capabilities": [target_capability],
                "proposal_mode": "sparse-parent-patch-v1",
                "max_patched_capabilities": 1,
                "protected_capabilities": sorted(
                    set(CAPABILITIES) - {target_capability}
                ),
            }
        )
    payload["s1_settings"] = s1_settings
    payload["limits"]["max_feedback_calls"] = feedback_total_count
    disclosures = _clean_disclosures(source.disclosures)
    disclosures.extend(
        (
            f"Static opt800 was freshly generated with {ACTIVE_ASSISTANT_MODEL}; "
            "this spec binds its SHA and reproducible fixed sample roles.",
            "Historical S1 R1-R10 remain rejected and read-only; this is a "
            "separately authorized model-generated Body lineage.",
            "S1 R1 uses a spec-frozen discovery600 selection and 48 fresh Feedback "
            "calls with canary6 plus a full-batch terminal gate before Creator.",
            "S1 may change only the selected capability Body authoring fields; "
            "Description and the other five Skills remain byte-exact. The action "
            "model freely chooses tools and generates the final response format.",
            "Running S1 creates new Feedback, Creator, and Assistant calls and has "
            "not been started by lineage preparation.",
        )
    )
    payload["disclosures"] = list(dict.fromkeys(disclosures))

    frozen = _validated_spec(payload)
    atomic_write_json(output_spec_path.resolve(), frozen.model_dump(mode="json"))
    return frozen


def _safe_call_id(query_id: str) -> str:
    value = f"canary-{query_id}"
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in value)


def _source_feedback_pairs(
    source_root: Path,
) -> tuple[dict[str, Any], list[dict[str, object]], dict[str, bytes]]:
    run_manifest_path = source_root / "manifest.json"
    run_manifest_bytes = read_stable_regular_file(
        run_manifest_path, label="R1 run manifest"
    )
    manifest_value = parse_canonical_json(run_manifest_bytes, label="R1 run manifest")
    if not isinstance(manifest_value, dict):
        raise LineagePreparationError("R1 run manifest must be an object")
    if manifest_value.get("kind") != "core-experiment-fast-run":
        raise LineagePreparationError("source is not a Core Fast run")
    s1_settings = manifest_value.get("s1_settings")
    if not isinstance(s1_settings, dict) or (
        s1_settings.get("round_id"),
        s1_settings.get("feedback_mode"),
    ) != ("r1", "fresh"):
        raise LineagePreparationError("Feedback source must be a fresh R1 run")
    models = manifest_value.get("models")
    feedback_model = (
        models.get("feedback", {}).get("requested_model")
        if isinstance(models, dict) and isinstance(models.get("feedback"), dict)
        else None
    )
    if not isinstance(feedback_model, str) or not feedback_model:
        raise LineagePreparationError("R1 manifest lacks the Feedback model")
    fixed_samples = manifest_value.get("fixed_samples")
    canary = fixed_samples.get("canary12") if isinstance(fixed_samples, dict) else None
    if not isinstance(canary, list) or len(canary) != 12:
        raise LineagePreparationError("R1 manifest must bind canary12")

    expected_ids: list[str] = []
    call_entries: list[dict[str, object]] = []
    content_by_relative_path: dict[str, bytes] = {}
    for sample in canary:
        if not isinstance(sample, dict):
            raise LineagePreparationError("invalid canary12 entry")
        query_id = sample.get("query_id")
        sample_role = sample.get("role")
        if not isinstance(query_id, str) or sample_role not in {"failure", "anchor"}:
            raise LineagePreparationError("invalid canary12 identity or role")
        call_id = _safe_call_id(query_id)
        if call_id in expected_ids:
            raise LineagePreparationError("duplicate canary Feedback call ID")
        expected_ids.append(call_id)
        pair: dict[str, object] = {"call_id": call_id}
        parsed: dict[str, CallIntent | CallResult] = {}
        for suffix, model_type in (
            ("intent", CallIntent),
            ("result", CallResult),
        ):
            relative = f"calls/feedback/{call_id}.{suffix}.json"
            path = source_root / Path(relative)
            content = read_stable_regular_file(
                path, label=f"R1 Feedback {call_id} {suffix}"
            )
            raw = parse_canonical_json(content, label=f"R1 Feedback {call_id} {suffix}")
            parsed[suffix] = model_type.model_validate(raw, strict=True)
            pair[f"{suffix}_path"] = relative
            pair[f"{suffix}_sha256"] = sha256_bytes(content)
            pair[f"{suffix}_bytes"] = len(content)
            content_by_relative_path[relative] = content
        intent = parsed["intent"]
        result = parsed["result"]
        assert isinstance(intent, CallIntent)
        assert isinstance(result, CallResult)
        if (
            intent.call_id != call_id
            or intent.role != "feedback"
            or intent.requested_model != feedback_model
            or intent.purpose != f"S1 fixed canary Feedback {query_id}"
            or result.call_id != call_id
            or result.role != "feedback"
            or result.requested_model != feedback_model
        ):
            raise LineagePreparationError("R1 Feedback call identity differs")
        query = intent.payload.get("query")
        baseline = intent.payload.get("baseline")
        if (
            intent.payload.get("operation") != "strict_visual_feedback"
            or not isinstance(query, dict)
            or query.get("query_id") != query_id
            or not isinstance(baseline, dict)
            or baseline.get("query_id") != query_id
            or intent.payload.get("sample_role") != sample_role
            or intent.payload.get("no_replacement") is not True
        ):
            raise LineagePreparationError("R1 Feedback payload differs from canary12")
        call_entries.append(pair)

    feedback_dir = source_root / "calls" / "feedback"
    observed_intents = {path.name for path in feedback_dir.glob("*.intent.json")}
    observed_results = {path.name for path in feedback_dir.glob("*.result.json")}
    if observed_intents != {f"{item}.intent.json" for item in expected_ids} or (
        observed_results != {f"{item}.result.json" for item in expected_ids}
    ):
        raise LineagePreparationError(
            "R1 Feedback directory is not exactly the frozen 12 call pairs"
        )

    input_sha = manifest_value.get("input_sha256")
    opt_sha = (
        input_sha.get("opt_static_results") if isinstance(input_sha, dict) else None
    )
    if not isinstance(opt_sha, str) or len(opt_sha) != 64:
        raise LineagePreparationError("R1 manifest lacks the Static opt800 SHA")
    reuse_manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": FEEDBACK_MANIFEST_KIND,
        "source_experiment_id": manifest_value.get("experiment_id"),
        "source_run_manifest_sha256": sha256_bytes(run_manifest_bytes),
        "opt_static_results_sha256": opt_sha,
        "feedback_model": feedback_model,
        "canary_query_ids": [str(item["query_id"]) for item in canary],
        "call_count": 12,
        "call_pairs": call_entries,
    }
    return reuse_manifest, call_entries, content_by_relative_path


def export_feedback_bundle(*, source_root: Path, bundle_dir: Path) -> str:
    """Export the exact 12 R1 Feedback pairs into a create-only bundle."""

    source_root = source_root.resolve()
    bundle_dir = bundle_dir.resolve()
    if bundle_dir.exists():
        raise FileExistsError(bundle_dir)
    manifest, _pairs, contents = _source_feedback_pairs(source_root)
    bundle_dir.mkdir(parents=True)
    for relative, content in contents.items():
        destination = bundle_dir / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(content)
    manifest_path = bundle_dir / FEEDBACK_MANIFEST_NAME
    atomic_write_json(manifest_path, manifest)
    return sha256_bytes(manifest_path.read_bytes())


def _validated_bundle(bundle_dir: Path) -> tuple[bytes, dict[str, bytes]]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / FEEDBACK_MANIFEST_NAME
    manifest_bytes = read_stable_regular_file(
        manifest_path, label="Feedback reuse source manifest"
    )
    value = parse_canonical_json(manifest_bytes, label="Feedback reuse source manifest")
    if not isinstance(value, dict) or value.get("kind") != FEEDBACK_MANIFEST_KIND:
        raise LineagePreparationError("unexpected Feedback reuse manifest")
    pairs = value.get("call_pairs")
    if value.get("call_count") != 12 or not isinstance(pairs, list) or len(pairs) != 12:
        raise LineagePreparationError("Feedback reuse bundle must contain 12 pairs")
    feedback_model = value.get("feedback_model")
    expected_paths: set[str] = set()
    contents: dict[str, bytes] = {}
    call_ids: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get("call_id"), str):
            raise LineagePreparationError("invalid Feedback reuse pair entry")
        call_id = pair["call_id"]
        if call_id in call_ids:
            raise LineagePreparationError("duplicate Feedback reuse call ID")
        call_ids.add(call_id)
        for suffix, model_type in (
            ("intent", CallIntent),
            ("result", CallResult),
        ):
            relative = f"calls/feedback/{call_id}.{suffix}.json"
            if pair.get(f"{suffix}_path") != relative:
                raise LineagePreparationError("unsafe Feedback reuse artifact path")
            content = read_stable_regular_file(
                bundle_dir / Path(relative), label=f"Feedback bundle {call_id} {suffix}"
            )
            if pair.get(f"{suffix}_sha256") != sha256_bytes(content) or pair.get(
                f"{suffix}_bytes"
            ) != len(content):
                raise LineagePreparationError("Feedback reuse artifact digest differs")
            raw = parse_canonical_json(
                content, label=f"Feedback bundle {call_id} {suffix}"
            )
            model = model_type.model_validate(raw, strict=True)
            if (
                model.call_id != call_id
                or model.role != "feedback"
                or model.requested_model != feedback_model
            ):
                raise LineagePreparationError(
                    "Feedback reuse artifact identity differs"
                )
            expected_paths.add(relative)
            contents[relative] = content
    directory = bundle_dir / "calls" / "feedback"
    observed_paths = {
        path.relative_to(bundle_dir).as_posix() for path in directory.glob("*.json")
    }
    if observed_paths != expected_paths:
        raise LineagePreparationError(
            "Feedback reuse bundle has missing or extra calls"
        )
    return manifest_bytes, contents


def freeze_reuse_round_spec(
    *,
    r1_spec_path: Path,
    bundle_dir: Path,
    output_spec_path: Path,
    experiment_id: str,
    round_id: str,
    target_capability: str,
    creator_directives: Sequence[str],
    required_patch_phrases: Sequence[str] = (),
    expected_feedback_manifest_sha256: str | None = None,
) -> CoreFastSpec:
    """Reject the superseded cross-round Feedback reuse workflow."""

    del (
        r1_spec_path,
        bundle_dir,
        output_spec_path,
        experiment_id,
        round_id,
        target_capability,
        creator_directives,
        required_patch_phrases,
        expected_feedback_manifest_sha256,
    )
    raise LineagePreparationError(
        "cross-round Feedback reuse is forbidden; freeze a fresh-per-round spec"
    )


def _legacy_freeze_reuse_round_spec_removed(
    *,
    r1_spec_path: Path,
    bundle_dir: Path,
    output_spec_path: Path,
    experiment_id: str,
    round_id: str,
    target_capability: str,
    creator_directives: Sequence[str],
    required_patch_phrases: Sequence[str] = (),
    expected_feedback_manifest_sha256: str | None = None,
) -> CoreFastSpec:
    """Historical implementation retained only as unread migration context."""

    if round_id not in {f"r{index}" for index in range(2, 11)}:
        raise LineagePreparationError("Feedback reuse round must be r2-r10")
    r1_spec_path = r1_spec_path.resolve()
    source = load_core_fast_spec(r1_spec_path)
    _assert_active_model(source)
    if (
        source.s1_settings.round_id != "r1"
        or source.s1_settings.feedback_mode != "fresh"
    ):
        raise LineagePreparationError("reuse rounds must derive from the fresh R1 spec")
    if target_capability not in CAPABILITIES:
        raise LineagePreparationError(f"unknown target capability: {target_capability}")
    if target_capability == "knowledge.visual_encyclopedia":
        raise LineagePreparationError(
            "Core Fast S1 protects knowledge.visual_encyclopedia after R0"
        )
    if not creator_directives:
        raise LineagePreparationError(
            "Feedback reuse round requires at least one Creator directive"
        )

    manifest_bytes, _contents = _validated_bundle(bundle_dir)
    observed_manifest_sha256 = sha256_bytes(manifest_bytes)
    if (
        expected_feedback_manifest_sha256 is not None
        and expected_feedback_manifest_sha256 != observed_manifest_sha256
    ):
        raise LineagePreparationError(
            "Feedback source manifest differs from the requested SHA"
        )
    manifest = parse_canonical_json(
        manifest_bytes, label="Feedback reuse source manifest"
    )
    assert isinstance(manifest, dict)
    if manifest.get("opt_static_results_sha256") != source.opt_static_results_sha256:
        raise LineagePreparationError(
            "Feedback bundle Static opt800 differs from the R1 spec"
        )
    if manifest.get("feedback_model") != source.models["feedback"].requested_model:
        raise LineagePreparationError("Feedback bundle model differs from the R1 spec")
    expected_canary = [item.query_id for item in source.fixed_samples.canary12]
    if manifest.get("canary_query_ids") != expected_canary:
        raise LineagePreparationError(
            "Feedback bundle canary12 differs from the R1 spec"
        )

    payload = source.model_dump(mode="json")
    payload["experiment_id"] = experiment_id
    payload["paths"] = _portable_paths(source, base_dir=r1_spec_path.parent)
    phrases = sorted(set(required_patch_phrases))
    payload["s1_settings"] = {
        "round_id": round_id,
        "feedback_mode": "reuse-exact-call-results",
        "feedback_reuse_manifest_sha256": observed_manifest_sha256,
        "proposal_mode": "sparse-parent-patch-v1",
        "max_patched_capabilities": 1,
        "protected_capabilities": sorted(set(CAPABILITIES) - {target_capability}),
        "creator_directives": list(dict.fromkeys(creator_directives)),
        "required_patch_phrases": ({target_capability: phrases} if phrases else {}),
    }
    disclosures = _clean_disclosures(source.disclosures)
    disclosures.append(
        f"S1 {round_id.upper()} reuses the exact 12 Feedback call pairs "
        "from the SHA-bound fresh R1 manifest and makes no replacement Feedback calls."
    )
    payload["disclosures"] = list(dict.fromkeys(disclosures))
    frozen = _validated_spec(payload)
    atomic_write_json(output_spec_path.resolve(), frozen.model_dump(mode="json"))
    return frozen


def import_feedback_bundle(*, bundle_dir: Path, target_root: Path) -> str:
    """Copy a validated bundle into a new run root without replacing any byte."""

    manifest_bytes, contents = _validated_bundle(bundle_dir)
    target_root = target_root.resolve()
    manifest_destination = target_root / "inputs" / FEEDBACK_MANIFEST_NAME
    feedback_destination = target_root / "calls" / "feedback"
    if manifest_destination.exists() or feedback_destination.exists():
        raise FileExistsError(
            "Feedback reuse destination already exists; replacement is forbidden"
        )
    for relative, content in contents.items():
        destination = target_root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(content)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    with manifest_destination.open("xb") as handle:
        handle.write(manifest_bytes)
    verify_feedback_import(bundle_dir=bundle_dir, target_root=target_root)
    return sha256_bytes(manifest_bytes)


def verify_feedback_import(*, bundle_dir: Path, target_root: Path) -> str:
    """Verify every imported byte and reject missing or extra Feedback calls."""

    manifest_bytes, contents = _validated_bundle(bundle_dir)
    target_root = target_root.resolve()
    imported_manifest = read_stable_regular_file(
        target_root / "inputs" / FEEDBACK_MANIFEST_NAME,
        label="imported Feedback source manifest",
    )
    if imported_manifest != manifest_bytes:
        raise LineagePreparationError("imported Feedback source manifest differs")
    for relative, expected in contents.items():
        observed = read_stable_regular_file(
            target_root / Path(relative), label=f"imported {relative}"
        )
        if observed != expected:
            raise LineagePreparationError("imported Feedback call bytes differ")
    feedback_dir = target_root / "calls" / "feedback"
    observed_paths = {
        path.relative_to(target_root).as_posix() for path in feedback_dir.glob("*.json")
    }
    if observed_paths != set(contents):
        raise LineagePreparationError("imported Feedback cache has extra calls")
    return sha256_bytes(manifest_bytes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a create-only active-model Core Fast S1 lineage."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    bootstrap = commands.add_parser("bootstrap-spec")
    bootstrap.add_argument("--base-spec", type=Path, default=DEFAULT_SPEC)
    bootstrap.add_argument("--output-spec", type=Path, required=True)
    bootstrap.add_argument("--experiment-id", required=True)
    bootstrap.add_argument("--opt-destination", type=Path, required=True)

    freeze_default = commands.add_parser("freeze-default")
    freeze_default.add_argument("--base-spec", type=Path, default=DEFAULT_SPEC)
    freeze_default.add_argument("--bootstrap-result", type=Path, required=True)
    freeze_default.add_argument("--output-spec", type=Path, required=True)
    freeze_default.add_argument("--experiment-id", required=True)

    r1 = commands.add_parser("freeze-r1")
    r1.add_argument("--bootstrap-spec", type=Path, required=True)
    r1.add_argument("--bootstrap-result", type=Path, required=True)
    r1.add_argument("--output-spec", type=Path, required=True)
    r1.add_argument("--experiment-id", required=True)
    r1.add_argument("--target-capability", choices=S1_BODY_PATCH_TARGETS, required=True)
    r1.add_argument("--creator-directive", action="append", required=True)
    r1.add_argument("--required-patch-phrase", action="append", default=[])

    fanout_r1 = commands.add_parser("freeze-fanout-r1")
    fanout_r1.add_argument("--bootstrap-spec", type=Path, required=True)
    fanout_r1.add_argument("--bootstrap-result", type=Path, required=True)
    fanout_r1.add_argument("--output-spec", type=Path, required=True)
    fanout_r1.add_argument("--experiment-id", required=True)
    fanout_r1.add_argument("--round-id", default="r1")
    fanout_r1.add_argument(
        "--feedback-total-count", type=int, choices=(48, 60), default=48
    )
    fanout_r1.add_argument(
        "--prior-experiment-root", type=Path, action="append", default=[]
    )
    fanout_r1.add_argument(
        "--feedback-selection-policy",
        choices=(
            "discovery-stratified-v1",
            "discovery-contrastive-v2",
            "discovery-supported-clusters-v3",
            "discovery-attributed-v4",
        ),
        default="discovery-stratified-v1",
    )
    fanout_r1.add_argument("--creator-directive", action="append", required=True)
    fanout_r1.add_argument(
        "--bounded-edit-surface",
        choices=("author-fields", "single-step-replace"),
        default="author-fields",
    )

    export = commands.add_parser("export-feedback")
    export.add_argument("--source-root", type=Path, required=True)
    export.add_argument("--bundle-dir", type=Path, required=True)

    import_command = commands.add_parser("import-feedback")
    import_command.add_argument("--bundle-dir", type=Path, required=True)
    import_command.add_argument("--target-root", type=Path, required=True)

    verify = commands.add_parser("verify-feedback")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    verify.add_argument("--target-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "bootstrap-spec":
            spec = prepare_static_bootstrap_spec(
                base_spec_path=args.base_spec,
                output_spec_path=args.output_spec,
                experiment_id=args.experiment_id,
                opt_destination=args.opt_destination,
            )
            result = {
                "status": "created",
                "spec": str(args.output_spec),
                "experiment_id": spec.experiment_id,
            }
        elif args.command == "freeze-default":
            spec = freeze_active_default_spec(
                base_spec_path=args.base_spec,
                bootstrap_result_path=args.bootstrap_result,
                output_spec_path=args.output_spec,
                experiment_id=args.experiment_id,
            )
            result = {
                "status": "created",
                "spec": str(args.output_spec),
                "experiment_id": spec.experiment_id,
            }
        elif args.command == "freeze-r1":
            spec = freeze_r1_spec(
                bootstrap_spec_path=args.bootstrap_spec,
                bootstrap_result_path=args.bootstrap_result,
                output_spec_path=args.output_spec,
                experiment_id=args.experiment_id,
                target_capability=args.target_capability,
                creator_directives=args.creator_directive,
                required_patch_phrases=args.required_patch_phrase,
            )
            result = {
                "status": "created",
                "spec": str(args.output_spec),
                "experiment_id": spec.experiment_id,
            }
        elif args.command == "freeze-fanout-r1":
            spec = freeze_r1_spec(
                bootstrap_spec_path=args.bootstrap_spec,
                bootstrap_result_path=args.bootstrap_result,
                output_spec_path=args.output_spec,
                experiment_id=args.experiment_id,
                target_capability=None,
                creator_directives=args.creator_directive,
                fanout=True,
                round_id=args.round_id,
                feedback_total_count=args.feedback_total_count,
                feedback_selection_policy=args.feedback_selection_policy,
                bounded_edit_surface=args.bounded_edit_surface,
                prior_experiment_roots=args.prior_experiment_root,
            )
            result = {
                "status": "created",
                "spec": str(args.output_spec),
                "experiment_id": spec.experiment_id,
            }
        elif args.command == "export-feedback":
            digest = export_feedback_bundle(
                source_root=args.source_root, bundle_dir=args.bundle_dir
            )
            result = {
                "status": "exported",
                "manifest_sha256": digest,
                "bundle_dir": str(args.bundle_dir),
            }
        elif args.command == "import-feedback":
            digest = import_feedback_bundle(
                bundle_dir=args.bundle_dir, target_root=args.target_root
            )
            result = {
                "status": "imported",
                "manifest_sha256": digest,
                "target_root": str(args.target_root),
            }
        else:
            digest = verify_feedback_import(
                bundle_dir=args.bundle_dir, target_root=args.target_root
            )
            result = {
                "status": "verified",
                "manifest_sha256": digest,
                "target_root": str(args.target_root),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (
        ArtifactFormatError,
        FileExistsError,
        LineagePreparationError,
        OSError,
        ValidationError,
        ValueError,
    ) as error:
        print(f"Core Fast lineage preparation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
