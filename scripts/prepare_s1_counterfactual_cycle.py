#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.evaluation.core_fast.models import (  # noqa: E402
    AssistantObservation,
    CoreFastSpec,
)
from skillchain.evaluation.core_fast.feedback_selection import (  # noqa: E402
    build_parent_counterfactual_manifest,
    build_parent_counterfactual_population,
    select_parent_counterfactual_samples,
)
from skillchain.evolution.s1_sparse_patch import (  # noqa: E402
    S1_CAPABILITY_ACTION_TOOLS,
    S1_RESPONSE_OPERATION_BY_FAILURE_FAMILY,
    compile_counterfactual_semantic_policy_branch,
    compile_counterfactual_typed_policy_branch,
    parse_counterfactual_semantic_policy_patch,
    parse_counterfactual_capability_response_policy_patch,
    parse_counterfactual_surface_closed_policy_patch,
    parse_counterfactual_typed_policy_patch,
    response_operation_for_capability_failure_family,
)
from skillchain.schemas import Query  # noqa: E402
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes  # noqa: E402


CYCLE_ID = "s1-counterfactual-v2"
R12_ROOT = Path(
    r"D:\athena\experiment-runs\portfolio-core-dual-policy-campaign-20260814-v2"
    r"\runs\r12-counterfactual-rule"
)
R12_BANK_SHA = "e70ed907825833a0bbb97ccde068fd38d4a6342824cd9dcdfb543723feb096cd"
R12_BANK_FILE_SHA = "edf83d288fcbb67b78b53a6390b713f845238836b7fc4f3378aca34c23bf9489"
R12_DECISION_FILE_SHA = (
    "1edc7fba7f05ee3deffc8a41907931a76e660fba5058262e3251156957923bc4"
)
R12_MANIFEST_FILE_SHA = (
    "7c24fd2192784ff2b33d2975557d34120c91e39a36aea2a0fa5cca09370afdc8"
)
R12_STYLE_SHA = "a5c6c53fa2778d1237bc5fe7ba652205bacdf1616090e91e6ac44f85af88b5ef"
R12_PROTECTION_IDS = tuple(
    sorted(
        f"r2-core-{suffix}"
        for suffix in ("0261", "0263", "0265", "0361", "0436", "0440", "1012", "1161")
    )
)


ROUND_DEFINITIONS = {
    "r31": {
        "capability": "utility.recipe_guidance",
        "surface": "action-policy",
        "gain_seeds": tuple(
            sorted(
                f"r2-core-{suffix}"
                for suffix in ("0799", "0273", "0272", "0448", "0674")
            )
        ),
        "regressions": tuple(
            sorted(f"r2-core-{suffix}" for suffix in ("0447", "0975", "0374"))
        ),
        "memory": (
            {
                "capability": "utility.recipe_guidance",
                "hypothesis": "counterfactual action policy for the stable Recipe gain cluster",
                "historical_signal": "R8 gained seven parent failures without a parent-success regression",
                "forbidden_carry_forward": "no rejected R17/R26/R29 Skill text or Bank",
            },
        ),
    },
    "r32": {
        "capability": "product.multi_search",
        "surface": "response-policy",
        "gain_seeds": tuple(sorted(f"r2-core-{suffix}" for suffix in ("0358", "0258"))),
        "regressions": tuple(
            sorted(f"r2-core-{suffix}" for suffix in ("0758", "0784", "0682"))
        ),
        "memory": (
            {
                "capability": "product.multi_search",
                "hypothesis": "counterfactual evidence-to-item association response policy",
                "historical_signal": "Multi response gains repeated but broad rewrites regressed protected states",
                "forbidden_carry_forward": "no rejected R17/R26/R29 Skill text or Bank",
            },
        ),
    },
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_create_only(path: Path, payload: object) -> None:
    content = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"create-only artifact differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, dict):
        raise ValueError(f"JSON object required: {path}")
    return raw


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"JSONL object required: {path}")
            rows.append(raw)
    return rows


def _resolve_spec_path(base_spec_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_spec_path.parent / path).resolve()


def _verify_parent_opt800(
    *,
    opt_path: Path,
    base: dict[str, object],
    queries: list[dict[str, object]],
) -> None:
    rows = _read_jsonl(opt_path)
    observations = [
        AssistantObservation.model_validate(row, strict=True) for row in rows
    ]
    expected_ids = {
        str(row["query_id"]) for row in queries if row.get("split") == "opt_pool"
    }
    if (
        len(observations) != 800
        or len({row.query_id for row in observations}) != 800
        or {row.query_id for row in observations} != expected_ids
    ):
        raise ValueError("R12 parent opt800 does not cover the fixed opt800 exactly")
    models = base.get("models")
    runtime = base.get("runtime")
    if not isinstance(models, dict) or not isinstance(runtime, dict):
        raise ValueError("base spec model/runtime identity is missing")
    assistant = models.get("assistant")
    if not isinstance(assistant, dict):
        raise ValueError("base spec Assistant identity is missing")
    expected_model = assistant.get("requested_model")
    expected_contract = runtime.get("assistant_contract")
    request_ids: list[str] = []
    for observation in observations:
        context = observation.replay_context
        result = context.get("assistant_result")
        response = context.get("response")
        receipt = context.get("receipt")
        if not all(isinstance(item, dict) for item in (result, response, receipt)):
            raise ValueError(
                f"R12 parent opt800 evidence is missing: {observation.query_id}"
            )
        assert (
            isinstance(result, dict)
            and isinstance(response, dict)
            and isinstance(receipt, dict)
        )
        if (
            result.get("bank_sha256") != R12_BANK_SHA
            or result.get("backbone_model") != expected_model
            or response.get("backbone_model") != expected_model
            or observation.assistant_contract != expected_contract
        ):
            raise ValueError(
                f"R12 parent opt800 identity differs: {observation.query_id}"
            )
        calls = receipt.get("model_calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(
                f"R12 parent opt800 call receipt is missing: {observation.query_id}"
            )
        for call in calls:
            if (
                not isinstance(call, dict)
                or call.get("requested_model") != expected_model
                or call.get("response_model") != expected_model
                or not isinstance(call.get("provider_request_id"), str)
                or not call["provider_request_id"]
            ):
                raise ValueError(
                    f"R12 parent opt800 model call differs: {observation.query_id}"
                )
            request_ids.append(str(call["provider_request_id"]))
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("R12 parent opt800 provider request IDs are not unique")


def _verify_r12() -> dict[str, object]:
    bank_path = R12_ROOT / "banks" / "s1-selected.json"
    decision_path = R12_ROOT / "decisions" / "s1.json"
    manifest_path = R12_ROOT / "manifest.json"
    expected = (
        (bank_path, R12_BANK_FILE_SHA),
        (decision_path, R12_DECISION_FILE_SHA),
        (manifest_path, R12_MANIFEST_FILE_SHA),
    )
    for path, digest in expected:
        if _sha(path) != digest:
            raise ValueError(f"R12 source artifact SHA drifted: {path}")
    bank = StaticBankArtifact.model_validate_json(bank_path.read_bytes(), strict=True)
    if bank.bank_sha256 != R12_BANK_SHA:
        raise ValueError("R12 internal Bank SHA drifted")
    style = next(
        item
        for item in bank.skills
        if item.capability_id == "product.style_recommendation"
    )
    if style.skill_sha256 != R12_STYLE_SHA:
        raise ValueError("R12 protected Style Skill SHA drifted")
    decision = _load_json(decision_path)
    metrics = decision.get("metrics")
    if (
        decision.get("accepted") is not True
        or decision.get("alias_of") is not None
        or decision.get("selected_bank") != R12_BANK_SHA
        or not isinstance(metrics, dict)
        or metrics.get("round_id") != "r12"
    ):
        raise ValueError("R12 source decision is not the accepted selected parent")
    return {
        "source_round_id": "r12",
        "bank_path": str(bank_path),
        "bank_file_sha256": R12_BANK_FILE_SHA,
        "bank_sha256": R12_BANK_SHA,
        "decision_path": str(decision_path),
        "decision_file_sha256": R12_DECISION_FILE_SHA,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": R12_MANIFEST_FILE_SHA,
        "parent_protection_query_ids": list(R12_PROTECTION_IDS),
        "protected_skill_sha256": {"product.style_recommendation": R12_STYLE_SHA},
    }


def _round_spec(
    base: dict[str, object],
    *,
    round_id: str,
    parent_binding: dict[str, object],
    definition: dict[str, object] | None = None,
    cycle_id: str = CYCLE_ID,
    fixed_samples: dict[str, object] | None = None,
    typed_contract: bool = False,
    preflight_path: Path | None = None,
    preflight_sha256: str | None = None,
) -> dict[str, object]:
    definition = ROUND_DEFINITIONS[round_id] if definition is None else definition
    capability = str(definition["capability"])
    selection_policy = str(
        definition.get(
            "selection_policy",
            "parent-counterfactual-v7"
            if typed_contract
            else "parent-counterfactual-v6",
        )
    )
    payload = json.loads(json.dumps(base))
    payload["experiment_id"] = f"{cycle_id}-{round_id}-{capability.replace('.', '-')}"
    payload["s1_parent"] = parent_binding
    if fixed_samples is not None:
        payload["fixed_samples"] = fixed_samples
    payload["s1_settings"] = {
        "round_id": round_id,
        "feedback_mode": "fresh-per-round",
        "feedback_total_count": 9,
        "feedback_canary_count": 3,
        "feedback_format_retry_limit": 1,
        "feedback_selection_policy": selection_policy,
        "feedback_allocation": "target-focused",
        "target_capabilities": [capability],
        "proposal_mode": (
            "single-surface-counterfactual-fanout-v9"
            if typed_contract and selection_policy == "parent-counterfactual-v12"
            else "single-surface-counterfactual-fanout-v8"
            if typed_contract and selection_policy == "parent-counterfactual-v11"
            else "single-surface-counterfactual-fanout-v7"
            if typed_contract and selection_policy == "parent-counterfactual-v10"
            else "single-surface-counterfactual-fanout-v6"
            if typed_contract and selection_policy == "parent-counterfactual-v9"
            else "single-surface-counterfactual-fanout-v5"
            if typed_contract
            else "single-surface-counterfactual-fanout-v4"
        ),
        "max_patched_capabilities": 1,
        "protected_capabilities": sorted(
            item for item in payload["capabilities"] if item != capability
        ),
        "creator_directives": [],
        "required_patch_phrases": {},
        "bounded_edit_surface": "author-fields",
        "prior_experiment_memory": {capability: list(definition["memory"])},
        "cycle_id": cycle_id,
        "target_surface": definition["surface"],
        "counterfactual_gain_seed_query_ids": list(definition["gain_seeds"]),
        "counterfactual_regression_query_ids": list(definition["regressions"]),
        "counterfactual_parent_success_exclude_query_ids": list(
            definition.get("parent_success_exclusions", ())
        ),
        "parent_protection_query_ids": list(R12_PROTECTION_IDS),
    }
    if typed_contract:
        if preflight_path is None or preflight_sha256 is None:
            raise ValueError("typed cycle spec requires its all-round preflight")
        payload["s1_settings"].update(
            {
                "cycle_preflight_path": str(preflight_path),
                "cycle_preflight_sha256": preflight_sha256,
            }
        )
    payload["limits"] = {
        **payload["limits"],
        "max_feedback_calls": 18,
        "max_creator_calls": 3,
    }
    cycle_disclosures = (
        [
            "R31 and R32 are frozen together before provider calls and each patches one capability and one policy surface.",
            "body75 is a fixed previously observed gate; mechanism design cannot change after either round, and test300 is consumed once by one frozen finalist.",
        ]
        if cycle_id == CYCLE_ID
        else [
            "The batch freezes evidence feasibility and treatment sensitivity before provider calls; each round freezes its exact adaptive memory create-only after the preceding retrospective.",
            "body75 remains a fixed previously observed gate and cannot be used to rewrite the current batch evidence or risk thresholds.",
        ]
    )
    payload["disclosures"] = [
        *payload["disclosures"],
        "This forward-only cycle uses accepted R12 as its only legal parent; rejected R17/R26/R29 Banks and Skills are excluded.",
        *cycle_disclosures,
    ]
    return CoreFastSpec.model_validate_json(
        canonical_json_bytes(payload), strict=True
    ).model_dump(mode="json")


def bootstrap(args: argparse.Namespace) -> int:
    parent = _verify_r12()
    cycle_root = args.cycle_root.resolve()
    parent_opt_path = cycle_root / "artifacts" / "r12-parent-opt800.jsonl"
    parent_binding = {
        **parent,
        "opt_results_path": str(parent_opt_path),
        "opt_results_sha256": None,
    }
    base = _load_json(args.base_spec.resolve())
    spec = _round_spec(base, round_id="r31", parent_binding=parent_binding)
    output = cycle_root / "specs" / "s1-parent-opt800-bootstrap.json"
    _write_create_only(output, spec)
    print(
        json.dumps(
            {
                "spec": str(output),
                "run_root": str(cycle_root / "runs" / "parent-opt800"),
            },
            indent=2,
        )
    )
    return 0


def _response_probe_clauses(
    signature: dict[str, object], capability: str
) -> tuple[str, str]:
    family = signature.get("response_failure_family")
    clauses = {
        "item-association": (
            "the fixed successful multi product tool output contains item associations",
            "state only item associations supported by that public evidence",
        ),
        "card-closure": (
            "the fixed successful multi product tool output contains matched candidates",
            "emit cards only for candidates referenced by the supported item associations",
        ),
        "unsupported-claim": (
            "a proposed material claim is not supported by the fixed public evidence",
            "omit that unsupported material claim from the answer",
        ),
        "citation-closure": (
            "a material answer claim is supported by one fixed public source",
            "attach that exact visible source handle to the same claim",
        ),
        "fallback-branch": (
            "the fixed successful tool output contains no supported evidence",
            "use only the parent fallback branch for that empty public evidence",
        ),
        "style-evidence": (
            "the fixed public style evidence contains a literal facet and value",
            "state only that literal facet and value with its visible evidence handle",
        ),
        "output-structure": (
            "the supported answer is ready for the parent response sections",
            "place it in the parent response sections without adding a material claim",
        ),
        "grounding-other": (
            "the fixed public evidence does not support a proposed material claim",
            "omit that unsupported material claim",
        ),
        "output-other": (
            "the supported answer is ready for the parent output contract",
            "preserve the parent output contract around that supported answer",
        ),
    }
    if family not in clauses:
        raise ValueError(
            f"response treatment family is not representable for {capability}: {family}"
        )
    return clauses[str(family)]


def _treatment_probe(
    *,
    parent: StaticBankArtifact,
    spec: CoreFastSpec,
    selected: tuple[dict[str, object], ...],
) -> dict[str, object]:
    settings = spec.s1_settings
    capability = settings.target_capabilities[0]
    surface = settings.target_surface
    assert surface is not None
    binding = spec.s1_parent
    if binding is not None and capability in binding.protected_skill_sha256:
        raise ValueError(
            "target capability is byte-exact protected by the S1 parent binding: "
            f"{capability}"
        )
    parent_skill = next(
        item for item in parent.skills if item.capability_id == capability
    )
    success_ids = tuple(
        sorted(
            str(row["query_id"])
            for row in selected
            if row["counterfactual_role"] == "parent_success"
        )
    )
    semantic_contract = getattr(
        settings, "proposal_mode", "single-surface-counterfactual-fanout-v5"
    ) in {
        "single-surface-counterfactual-fanout-v6",
        "single-surface-counterfactual-fanout-v7",
        "single-surface-counterfactual-fanout-v8",
        "single-surface-counterfactual-fanout-v9",
    }
    surface_closed_contract = (
        getattr(settings, "proposal_mode", "single-surface-counterfactual-fanout-v5")
        == "single-surface-counterfactual-fanout-v7"
    )
    capability_response_contract = getattr(
        settings, "proposal_mode", "single-surface-counterfactual-fanout-v5"
    ) in {
        "single-surface-counterfactual-fanout-v8",
        "single-surface-counterfactual-fanout-v9",
    }
    payload: dict[str, object] = {
        "schema_version": 5
        if capability_response_contract
        else 4
        if surface_closed_contract
        else 3
        if semantic_contract
        else 2,
        "capability_id": capability,
        "parent_skill_sha256": parent_skill.skill_sha256,
        "target_surface": surface,
        "non_target_surface_action": "inherit",
        "must_preserve": [
            (
                {"query_id": query_id}
                if surface_closed_contract or capability_response_contract
                else {
                    "query_id": query_id,
                    "provider_visible_state": (
                        f"the provider visible success state {query_id} remains unchanged"
                    ),
                }
            )
            for query_id in success_ids
        ],
    }
    if surface == "action-policy":
        operators = S1_CAPABILITY_ACTION_TOOLS[capability]
        if not operators:
            raise ValueError(f"action treatment has no operator: {capability}")
        selected_failure = next(
            (
                row
                for row in selected
                if row["counterfactual_role"] == "cluster_failure"
            ),
            None,
        )
        action_signature = (
            selected_failure.get("action_treatment_signature")
            if isinstance(selected_failure, dict)
            else None
        )
        if getattr(
            settings, "feedback_selection_policy", "parent-counterfactual-v7"
        ) in {
            "parent-counterfactual-v8",
            "parent-counterfactual-v9",
            "parent-counterfactual-v10",
            "parent-counterfactual-v11",
            "parent-counterfactual-v12",
        }:
            if not isinstance(action_signature, dict):
                raise ValueError(
                    "action treatment has no provider-visible failure state"
                )
            prior = action_signature.get("prior_tool_name")
            prior_status = action_signature.get("prior_tool_status")
            if (
                action_signature.get("phase") != "after-tool"
                or prior not in operators
                or prior_status not in {"success", "invalid-arguments", "error"}
                or action_signature.get("public_evidence") != "unknown"
            ):
                raise ValueError("action treatment failure state is not representable")
            action_directive = selected_failure.get("action_treatment_directive")
            if not isinstance(action_directive, dict):
                if settings.feedback_selection_policy == "parent-counterfactual-v12":
                    raise ValueError("action treatment directive is not frozen")
                action_directive = {
                    "operation": (
                        "stop-action-loop"
                        if prior_status == "success"
                        else "retry-tool-once"
                    ),
                    "tool_name": None if prior_status == "success" else prior,
                    "arguments_from": (
                        "none"
                        if prior_status == "success"
                        else "current-user-request"
                        if prior_status == "invalid-arguments"
                        else "last-valid-arguments"
                    ),
                }
            payload.update(
                {
                    "action_when": dict(action_signature),
                    "action_then": dict(action_directive),
                }
            )
        else:
            prior = operators[0] if len(operators) > 1 else None
            target = operators[1] if len(operators) > 1 else operators[0]
            payload.update(
                {
                    "action_when": (
                        {
                            "phase": "after-tool",
                            "prior_tool_name": prior,
                            "prior_tool_status": "success",
                            "public_evidence": "nonempty",
                        }
                        if prior is not None
                        else {
                            "phase": "before-first-tool",
                            "prior_tool_name": None,
                            "prior_tool_status": "not-called",
                            "public_evidence": "unknown",
                        }
                    ),
                    "action_then": {
                        "operation": "invoke-tool-once",
                        "tool_name": target,
                        "arguments_from": (
                            "last-visible-tool-output"
                            if prior is not None
                            else "current-user-request"
                        ),
                    },
                }
            )
    else:
        selected_failure = next(
            (
                row
                for row in selected
                if row["counterfactual_role"] == "cluster_failure"
            ),
            None,
        )
        response_signature = (
            selected_failure.get("response_treatment_signature")
            if isinstance(selected_failure, dict)
            else None
        )
        if settings.feedback_selection_policy in {
            "parent-counterfactual-v8",
            "parent-counterfactual-v9",
            "parent-counterfactual-v10",
            "parent-counterfactual-v11",
            "parent-counterfactual-v12",
        }:
            if not isinstance(response_signature, dict):
                raise ValueError(
                    "response treatment has no provider-visible failure family"
                )
            if semantic_contract:
                evidence = response_signature.get("terminal_evidence_class")
                family = response_signature.get("response_failure_family")
                operation = (
                    response_operation_for_capability_failure_family(
                        capability_id=capability,
                        failure_family=str(family),
                        predicted_reason_code=str(
                            response_signature.get("predicted_reason_code")
                        ),
                    )
                    if capability_response_contract
                    else S1_RESPONSE_OPERATION_BY_FAILURE_FAMILY.get(str(family))
                )
                if not isinstance(evidence, dict) or operation is None:
                    raise ValueError("response treatment has no typed behavior")
                payload.update(
                    {
                        "response_when": {
                            "terminal_tool_names": evidence.get("tool_names"),
                            "terminal_evidence_outcome": evidence.get("outcome"),
                            "evidence_kinds": evidence.get("evidence_kinds"),
                        },
                        "response_then": {"operation": operation},
                    }
                )
            else:
                when, then = _response_probe_clauses(response_signature, capability)
                payload.update({"when": when, "then": then})
        else:
            when = "the fixed public tool evidence contains an item association"
            then = (
                "preserve only the item association supported by that public evidence"
            )
            payload.update({"when": when, "then": then})
    if semantic_contract:
        parse_semantic = (
            parse_counterfactual_capability_response_policy_patch
            if capability_response_contract
            else parse_counterfactual_surface_closed_policy_patch
            if surface_closed_contract
            else parse_counterfactual_semantic_policy_patch
        )
        proposal = parse_semantic(
            canonical_json_bytes(payload),
            capability_id=capability,
            parent_skill_sha256=parent_skill.skill_sha256,
            target_surface=surface,
            parent_success_query_ids=success_ids,
            expected_action_condition=(
                payload.get("action_when") if surface == "action-policy" else None
            ),
            expected_action_directive=(
                payload.get("action_then") if surface == "action-policy" else None
            ),
            expected_response_signature=(
                response_signature if surface == "response-policy" else None
            ),
        )
        compiled = compile_counterfactual_semantic_policy_branch(
            parent_bank=parent, proposal=proposal
        )
    else:
        proposal = parse_counterfactual_typed_policy_patch(
            canonical_json_bytes(payload),
            capability_id=capability,
            parent_skill_sha256=parent_skill.skill_sha256,
            target_surface=surface,
            parent_success_query_ids=success_ids,
        )
        compiled = compile_counterfactual_typed_policy_branch(
            parent_bank=parent, proposal=proposal
        )
    candidate_skill = next(
        item for item in compiled.bank.skills if item.capability_id == capability
    )
    expected_heading = (
        "## S1 action policy overlay"
        if surface == "action-policy"
        else "## S1 response policy overlay"
    )
    forbidden_heading = (
        "## S1 response policy overlay"
        if surface == "action-policy"
        else "## S1 action policy overlay"
    )
    protected_exact = all(
        left == right
        for left, right in zip(parent.skills, compiled.bank.skills, strict=True)
        if left.capability_id != capability
    )
    structural_sensitive = bool(
        candidate_skill.skill_sha256 != parent_skill.skill_sha256
        and candidate_skill.description == parent_skill.description
        and candidate_skill.operators == parent_skill.operators
        and expected_heading in candidate_skill.body
        and forbidden_heading not in candidate_skill.body
        and protected_exact
    )
    behavior_sensitive = True
    behavior_probe: dict[str, object] = {}
    if semantic_contract and surface == "response-policy":
        assert isinstance(response_signature, dict)
        predicted_reason = response_signature.get("predicted_reason_code")
        predicted_component = response_signature.get("predicted_metric_component")
        terminal_evidence = response_signature.get("terminal_evidence_class")
        failure_rows = tuple(
            row
            for row in selected
            if row.get("counterfactual_role") == "cluster_failure"
        )
        success_rows = tuple(
            row
            for row in selected
            if row.get("counterfactual_role") == "parent_success"
        )
        behavior_sensitive = bool(
            isinstance(predicted_reason, str)
            and isinstance(predicted_component, str)
            and isinstance(terminal_evidence, dict)
            and len(failure_rows) == 3
            and len(success_rows) == 3
            and all(
                isinstance(row.get("state"), dict)
                and predicted_reason in row["state"].get("response_reason_codes", ())
                and predicted_component
                in row["state"].get("failed_response_components", ())
                and row["state"].get("terminal_response_evidence_class")
                == terminal_evidence
                for row in failure_rows
            )
            and all(
                isinstance(row.get("state"), dict)
                and predicted_component
                not in row["state"].get("failed_response_components", ())
                and row["state"].get("terminal_response_evidence_class")
                == terminal_evidence
                for row in success_rows
            )
        )
        behavior_probe = {
            "probe_response_predicted_reason_code": predicted_reason,
            "probe_response_predicted_metric_component": predicted_component,
            "probe_response_failure_count": len(failure_rows),
            "probe_response_protected_success_count": len(success_rows),
            "probe_response_behavior_sensitive": behavior_sensitive,
        }
    sensitive = structural_sensitive and behavior_sensitive
    return {
        "treatment_sensitive": sensitive,
        "structural_treatment_sensitive": structural_sensitive,
        "behavior_treatment_sensitive": behavior_sensitive,
        "probe_candidate_bank_sha256": compiled.bank.bank_sha256,
        "probe_candidate_skill_sha256": candidate_skill.skill_sha256,
        "probe_policy_text_sha256": sha256_bytes(compiled.policy_text.encode("utf-8")),
        "target_skill_changed": candidate_skill.skill_sha256
        != parent_skill.skill_sha256,
        "target_description_unchanged": candidate_skill.description
        == parent_skill.description,
        "target_operators_unchanged": candidate_skill.operators
        == parent_skill.operators,
        "non_target_surface_absent": forbidden_heading not in candidate_skill.body,
        "protected_skills_byte_exact": protected_exact,
        **(
            {
                "probe_action_prior_tool_name": payload["action_when"][
                    "prior_tool_name"
                ],
                "probe_action_prior_tool_status": payload["action_when"][
                    "prior_tool_status"
                ],
                "probe_action_tool_name": payload["action_then"]["tool_name"],
            }
            if surface == "action-policy"
            else {
                "probe_response_failure_family": (
                    response_signature.get("response_failure_family")
                    if isinstance(response_signature, dict)
                    else None
                )
            }
        ),
        **behavior_probe,
    }


def _build_cycle_preflight(
    *,
    preliminary_specs: dict[str, dict[str, object]],
    parent: StaticBankArtifact,
    parent_opt: dict[str, AssistantObservation],
    queries: tuple[Query, ...],
    parent_opt_sha256: str,
    round_ids: tuple[str, ...] = ("r31", "r32"),
    cycle_id: str = CYCLE_ID,
) -> dict[str, object]:
    rounds: dict[str, object] = {}
    for round_id in round_ids:
        spec = CoreFastSpec.model_validate_json(
            canonical_json_bytes(preliminary_specs[round_id]), strict=True
        )
        identity = {
            "target_capability": spec.s1_settings.target_capabilities[0],
            "target_surface": spec.s1_settings.target_surface,
        }
        try:
            population = build_parent_counterfactual_population(
                queries=queries,
                observations=parent_opt,
                settings=spec.s1_settings,
            )
            selected = select_parent_counterfactual_samples(
                population, spec.s1_settings
            )
            manifest = build_parent_counterfactual_manifest(
                population=population,
                selected=selected,
                settings=spec.s1_settings,
                parent_bank_sha256=parent.bank_sha256,
                parent_opt_sha256=parent_opt_sha256,
            )
            probe = _treatment_probe(
                parent=parent,
                spec=spec,
                selected=selected,
            )
            roles = Counter(str(row["counterfactual_role"]) for row in selected)
            evidence_feasible = roles == Counter(
                {
                    "cluster_failure": 3,
                    "parent_success": 3,
                    "historical_regression": 3,
                }
            )
            treatment_separable = True
            if spec.s1_settings.feedback_selection_policy in {
                "parent-counterfactual-v8",
                "parent-counterfactual-v9",
                "parent-counterfactual-v10",
                "parent-counterfactual-v11",
                "parent-counterfactual-v12",
            }:
                if spec.s1_settings.target_surface == "action-policy":
                    failure_signatures = {
                        canonical_json_bytes(row.get("action_treatment_signature"))
                        for row in selected
                        if row["counterfactual_role"] == "cluster_failure"
                    }
                    protected_signatures = {
                        canonical_json_bytes(row.get("action_treatment_signature"))
                        for row in selected
                        if row["counterfactual_role"] == "parent_success"
                    }
                    treatment_separable = (
                        len(failure_signatures) == 1
                        and canonical_json_bytes(None) not in failure_signatures
                        and failure_signatures.isdisjoint(protected_signatures)
                    )
                else:
                    failure_signatures = {
                        canonical_json_bytes(row.get("response_treatment_signature"))
                        for row in selected
                        if row["counterfactual_role"] == "cluster_failure"
                    }
                    protected_signatures = {
                        canonical_json_bytes(row.get("response_treatment_signature"))
                        for row in selected
                        if row["counterfactual_role"] == "parent_success"
                    }
                    treatment_separable = (
                        len(failure_signatures) == 1
                        and canonical_json_bytes(None) not in failure_signatures
                        and failure_signatures.isdisjoint(protected_signatures)
                    )
            rounds[round_id] = {
                **identity,
                "evidence_feasible": evidence_feasible,
                "treatment_separable": treatment_separable,
                "parent_success_exclusion_count": len(
                    spec.s1_settings.counterfactual_parent_success_exclude_query_ids
                ),
                "parent_success_exclusions_sha256": sha256_bytes(
                    canonical_json_bytes(
                        list(
                            spec.s1_settings.counterfactual_parent_success_exclude_query_ids
                        )
                    )
                ),
                "selection_manifest_sha256": manifest["manifest_sha256"],
                "selected_query_ids": list(manifest["selected_query_ids"]),
                **probe,
            }
        except (KeyError, TypeError, ValueError) as error:
            rounds[round_id] = {
                **identity,
                "evidence_feasible": False,
                "treatment_sensitive": False,
                "treatment_separable": False,
                "failure_reason": str(error),
            }
    passed = all(
        isinstance(item, dict)
        and item.get("evidence_feasible") is True
        and item.get("treatment_sensitive") is True
        and item.get("treatment_separable") is True
        for item in rounds.values()
    )
    return {
        "schema_version": 1,
        "kind": "core-fast-s1-counterfactual-cycle-preflight",
        "cycle_id": cycle_id,
        "parent_round_id": "r12",
        "parent_bank_sha256": parent.bank_sha256,
        "parent_opt_sha256": parent_opt_sha256,
        "round_order": list(round_ids),
        "rounds": rounds,
        "provider_calls": 0,
        "passed": passed,
    }


def freeze_cycle(args: argparse.Namespace) -> int:
    parent = _verify_r12()
    cycle_root = args.cycle_root.resolve()
    bootstrap_receipt = _load_json(args.bootstrap_receipt.resolve())
    if (
        bootstrap_receipt.get("kind") != "core-fast-s1-parent-opt800-bootstrap"
        or bootstrap_receipt.get("source_round_id") != "r12"
        or bootstrap_receipt.get("parent_bank_sha256") != R12_BANK_SHA
        or bootstrap_receipt.get("row_count") != 800
    ):
        raise ValueError("S1 parent opt800 bootstrap identity differs")
    opt_path = Path(str(bootstrap_receipt["opt_results_path"]))
    opt_sha = str(bootstrap_receipt["opt_results_sha256"])
    if _sha(opt_path) != opt_sha:
        raise ValueError("S1 parent opt800 file differs from bootstrap")
    fixed_samples = bootstrap_receipt.get("fixed_samples")
    if not isinstance(fixed_samples, dict):
        raise ValueError("S1 parent opt800 bootstrap lacks fixed samples")
    parent_binding = {
        **parent,
        "opt_results_path": str(opt_path),
        "opt_results_sha256": opt_sha,
    }
    base_spec_path = args.base_spec.resolve()
    base = _load_json(base_spec_path)
    paths = base.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("base spec lacks paths")
    queries = _read_jsonl(_resolve_spec_path(base_spec_path, str(paths["queries"])))
    _verify_parent_opt800(opt_path=opt_path, base=base, queries=queries)
    fold_rows = _read_jsonl(
        _resolve_spec_path(base_spec_path, str(paths["opt_fold_mapping"]))
    )
    replay_ids = {
        str(row["query_id"]) for row in fold_rows if row.get("role") == "replay"
    }
    replay_target_counts = {
        round_id: sum(
            row.get("query_id") in replay_ids
            and row.get("canonical_capability") == definition["capability"]
            for row in queries
        )
        for round_id, definition in ROUND_DEFINITIONS.items()
    }
    parent_bank = StaticBankArtifact.model_validate_json(
        Path(str(parent_binding["bank_path"])).read_bytes(), strict=True
    )
    parent_opt = {
        observation.query_id: observation
        for observation in (
            AssistantObservation.model_validate(row, strict=True)
            for row in _read_jsonl(opt_path)
        )
    }
    typed_preflight_path = cycle_root / "cycle-preflight.json"
    preliminary_specs = {
        round_id: _round_spec(
            base,
            round_id=round_id,
            parent_binding=parent_binding,
            fixed_samples=fixed_samples,
            typed_contract=True,
            preflight_path=typed_preflight_path,
            preflight_sha256="0" * 64,
        )
        for round_id in ("r31", "r32")
    }
    preflight = _build_cycle_preflight(
        preliminary_specs=preliminary_specs,
        parent=parent_bank,
        parent_opt=parent_opt,
        queries=tuple(Query.model_validate(row, strict=True) for row in queries),
        parent_opt_sha256=opt_sha,
    )
    _write_create_only(typed_preflight_path, preflight)
    if preflight["passed"] is not True:
        raise ValueError(
            "counterfactual cycle preflight failed; no runnable specs were frozen"
        )
    preflight_sha = _sha(typed_preflight_path)
    specs: dict[str, dict[str, object]] = {}
    spec_paths: dict[str, Path] = {}
    for round_id in ("r31", "r32"):
        spec = _round_spec(
            base,
            round_id=round_id,
            parent_binding=parent_binding,
            fixed_samples=fixed_samples,
            typed_contract=True,
            preflight_path=typed_preflight_path,
            preflight_sha256=preflight_sha,
        )
        path = cycle_root / "specs" / f"{round_id}.json"
        specs[round_id] = spec
        spec_paths[round_id] = path
    for round_id in ("r31", "r32"):
        _write_create_only(spec_paths[round_id], specs[round_id])
    assistant_outer_remaining = (
        sum(
            2 * (replay_target_counts[round_id] + 3) + 400 + 150
            for round_id in ("r31", "r32")
        )
        + 550
        + 600
    )
    parent_observed = float(bootstrap_receipt.get("observed_dashscope_cost_cny", 0.0))
    budget_estimate = parent_observed + assistant_outer_remaining * 0.0025 + 36 * 0.007
    if budget_estimate > 10.0:
        raise ValueError(
            f"counterfactual cycle worst-case DashScope estimate exceeds CNY 10: {budget_estimate:.6f}"
        )
    receipt_unsigned = {
        "schema_version": 1,
        "kind": "core-fast-s1-counterfactual-cycle-definition",
        "cycle_id": CYCLE_ID,
        "parent_round_id": "r12",
        "parent_bank_sha256": R12_BANK_SHA,
        "parent_opt_sha256": opt_sha,
        "cycle_preflight_path": str(typed_preflight_path),
        "cycle_preflight_sha256": preflight_sha,
        "round_order": ["r31", "r32", "r33"],
        "round_specs": {
            round_id: {
                "path": str(spec_paths[round_id]),
                "sha256": sha256_bytes(canonical_json_bytes(specs[round_id])),
                "run_root": str(cycle_root / "runs" / round_id),
                "target_capability": ROUND_DEFINITIONS[round_id]["capability"],
                "target_surface": ROUND_DEFINITIONS[round_id]["surface"],
            }
            for round_id in ("r31", "r32")
        },
        "test_policy": "one-frozen-finalist-paired-test300-v1",
        "maximum_creator_sessions": 2,
        "dashscope_stage_budget_cny": 10.0,
        "budget_estimate": {
            "basis": "observed_parent_plus_buffered_empirical_outer_call_ceiling",
            "parent_opt800_observed_cny": parent_observed,
            "replay_target_upper_counts": replay_target_counts,
            "remaining_assistant_outer_call_ceiling": assistant_outer_remaining,
            "assistant_cny_per_outer_ceiling": 0.0025,
            "feedback_attempt_ceiling": 36,
            "feedback_cny_per_attempt_ceiling": 0.007,
            "projected_total_dashscope_cny": budget_estimate,
            "within_cny10": True,
        },
    }
    receipt = {
        **receipt_unsigned,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_unsigned)),
    }
    receipt_path = cycle_root / "cycle-definition.json"
    _write_create_only(receipt_path, receipt)
    print(
        json.dumps(
            {
                "cycle_definition": str(receipt_path),
                "round_specs": {key: str(value) for key, value in spec_paths.items()},
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the R12 counterfactual S1 cycle"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("bootstrap", "freeze-cycle"):
        item = sub.add_parser(command)
        item.add_argument(
            "--base-spec",
            type=Path,
            default=REPOSITORY_ROOT / "specs" / "core-experiment-fast-v1.json",
        )
        item.add_argument("--cycle-root", type=Path, required=True)
        if command == "freeze-cycle":
            item.add_argument("--bootstrap-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return bootstrap(args) if args.command == "bootstrap" else freeze_cycle(args)


if __name__ == "__main__":
    raise SystemExit(main())
