from __future__ import annotations

import hashlib
from collections import Counter

from skillchain.evaluation.evaluator_outputs import VisualFeedbackOutput
from skillchain.evolution.s1_sparse_patch import (
    S1_RESPONSE_OPERATION_BY_FAILURE_FAMILY,
)

from .models import CAPABILITIES, CallIntent, CallResult


def _bucket(query_id: str) -> int:
    return hashlib.sha256(query_id.encode("utf-8")).digest()[0] % 4


class FakeCoreFastAdapter:
    """Deterministic no-network provider used only for focused/E2E tests."""

    def __init__(
        self,
        *,
        reject_stages: frozenset[str] = frozenset(),
        feedback_fail_ids: frozenset[str] = frozenset(),
        feedback_schema_fail_ids: frozenset[str] = frozenset(),
        judge_first_status: str | None = None,
        assistant_fail_ids: frozenset[str] = frozenset(),
        break_s3_common_trace: bool = False,
    ) -> None:
        self.reject_stages = reject_stages
        self.feedback_fail_ids = feedback_fail_ids
        self.feedback_schema_fail_ids = feedback_schema_fail_ids
        self.judge_first_status = judge_first_status
        self.assistant_fail_ids = assistant_fail_ids
        self.break_s3_common_trace = break_s3_common_trace
        self.calls: Counter[str] = Counter()

    def _result(
        self,
        intent: CallIntent,
        output: dict[str, object],
        *,
        returned_model: str | None = None,
    ) -> CallResult:
        return CallResult(
            call_id=intent.call_id,
            role=intent.role,
            status="success",
            requested_model=intent.requested_model,
            returned_model=returned_model or intent.requested_model,
            raw_output="{}",
            schema_valid=True,
            output=output,
            input_tokens=10,
            output_tokens=5,
            cost_cny=0.001,
            cost_basis="planning_estimate",
            latency_ms=1,
        )

    def invoke(self, intent: CallIntent) -> CallResult:
        self.calls[intent.role] += 1
        if intent.role == "feedback":
            query = intent.payload["query"]
            assert isinstance(query, dict)
            query_id = str(query["query_id"])
            if query_id in self.feedback_fail_ids:
                return CallResult(
                    call_id=intent.call_id,
                    role=intent.role,
                    status="provider_error",
                    requested_model=intent.requested_model,
                    returned_model="qwen3.8-max-202608",
                    failure_reason="injected Feedback failure",
                )
            if query_id in self.feedback_schema_fail_ids:
                return CallResult(
                    call_id=intent.call_id,
                    role=intent.role,
                    status="schema_error",
                    requested_model=intent.requested_model,
                    returned_model="qwen3.8-max-202608",
                    raw_output="{}",
                    schema_valid=False,
                    failure_reason="injected Feedback schema failure",
                )
            feedback = VisualFeedbackOutput(
                schema_version=1,
                summary=f"fixed feedback for {query_id}",
                rule_violations=(),
                ideal_response_gaps=(),
                skill_suggestions=(
                    (
                        "[policy_compatible] [action-policy] make tool-first and "
                        "stopping conditions explicit"
                        if intent.payload.get("attribution_policy")
                        == "dual-policy-attribution-v1"
                        else (
                            "[policy_compatible] "
                            f"[{intent.payload['target_surface']}] make only the "
                            "frozen target surface conditional"
                        )
                        if intent.payload.get("attribution_policy")
                        in {
                            "single-surface-counterfactual-v4",
                            "single-surface-counterfactual-v5",
                            "single-surface-counterfactual-v6",
                        }
                        else "[policy_compatible] make the existing contract explicit"
                    ),
                ),
            )
            return self._result(
                intent,
                {"feedback": feedback.model_dump(mode="json")},
                returned_model="qwen3.8-max-202608",
            )
        if intent.role == "creator":
            operation = str(intent.payload["operation"])
            parent = intent.payload["parent_bank"]
            assert isinstance(parent, dict)
            skills = parent["skills"]
            assert isinstance(skills, list)
            if operation == "s1_single_surface_counterfactual_creator":
                if "s1" in self.reject_stages:
                    return self._result(intent, {"invalid": True})
                capability = str(intent.payload["target_capability"])
                surface = str(intent.payload["target_surface"])
                requirements = intent.payload["requirements"]
                assert isinstance(requirements, dict)
                success_ids = requirements["must_preserve_exactly"]
                assert isinstance(success_ids, list)
                skill = next(
                    item
                    for item in skills
                    if isinstance(item, dict)
                    and item.get("capability_id") == capability
                )
                payload: dict[str, object] = {
                    "schema_version": (
                        3
                        if intent.payload.get("proposal_mode")
                        == "single-surface-counterfactual-fanout-v6"
                        else 2
                        if intent.payload.get("proposal_mode")
                        == "single-surface-counterfactual-fanout-v5"
                        else 1
                    ),
                    "capability_id": capability,
                    "parent_skill_sha256": skill["skill_sha256"],
                    "target_surface": surface,
                    "non_target_surface_action": "inherit",
                    "must_preserve": [
                        {
                            "query_id": query_id,
                            "provider_visible_state": (
                                f"the provider-visible success state for {query_id} remains unchanged"
                            ),
                        }
                        for query_id in success_ids
                    ],
                }
                if payload["schema_version"] in {2, 3} and surface == "action-policy":
                    expected = requirements.get(
                        "action_condition_is_bound_to_selected_failure_state"
                    )
                    payload.update(
                        {
                            "action_when": expected
                            if isinstance(expected, dict)
                            else {
                                "phase": "before-first-tool",
                                "prior_tool_name": None,
                                "prior_tool_status": "not-called",
                                "public_evidence": "unknown",
                            },
                            "action_then": {
                                "operation": "retry-tool-once"
                                if isinstance(expected, dict)
                                and expected.get("phase") == "after-tool"
                                else "invoke-tool-once",
                                "tool_name": expected.get("prior_tool_name")
                                if isinstance(expected, dict)
                                and expected.get("phase") == "after-tool"
                                else skill["operators"][0],
                                "arguments_from": (
                                    "last-valid-arguments"
                                    if isinstance(expected, dict)
                                    and expected.get("prior_tool_status") == "error"
                                    else "current-user-request"
                                ),
                            },
                        }
                    )
                elif payload["schema_version"] == 3:
                    signature = requirements.get(
                        "response_behavior_is_bound_to_selected_failure"
                    )
                    assert isinstance(signature, dict)
                    evidence = signature["terminal_evidence_class"]
                    assert isinstance(evidence, dict)
                    family = str(signature["response_failure_family"])
                    payload.update(
                        {
                            "response_when": {
                                "terminal_tool_names": evidence["tool_names"],
                                "terminal_evidence_outcome": evidence["outcome"],
                                "evidence_kinds": evidence["evidence_kinds"],
                            },
                            "response_then": {
                                "operation": S1_RESPONSE_OPERATION_BY_FAILURE_FAMILY[
                                    family
                                ]
                            },
                        }
                    )
                else:
                    payload.update(
                        {
                            "when": "the provider-visible target state matches the frozen failure cluster",
                            "then": "apply only the requested policy-surface correction",
                        }
                    )
                return self._result(intent, payload)
            if operation in {"s1_creator", "s1_dual_policy_creator"}:
                if "s1" in self.reject_stages:
                    return self._result(intent, {"invalid": True})
                templates = intent.payload["parent_authoring_content"]
                assert isinstance(templates, list)
                template_by_capability = {
                    str(item["capability_id"]): item
                    for item in templates
                    if isinstance(item, dict)
                }
                skill_by_capability = {
                    str(item["capability_id"]): item
                    for item in skills
                    if isinstance(item, dict)
                }
                requirements = intent.payload["requirements"]
                assert isinstance(requirements, dict)
                targets = requirements["target_capabilities"]
                assert isinstance(targets, list) and targets
                dual_target = requirements.get("dual_policy_capability")
                if isinstance(dual_target, str):
                    skill = skill_by_capability[dual_target]
                    return self._result(
                        intent,
                        {
                            "schema_version": 1,
                            "capability_id": dual_target,
                            "parent_skill_sha256": skill["skill_sha256"],
                            "action_policy": {
                                "action": "patch",
                                "policy_text": (
                                    "Call the declared tool before answering, copy "
                                    "only query-bound arguments, and stop after the "
                                    "first successful terminal evidence call."
                                ),
                            },
                            "response_policy": {
                                "action": "patch",
                                "policy_text": (
                                    "Ground every answer claim in visible evidence, "
                                    "copy exact public card handles, and use the "
                                    "existing fallback only when evidence is empty."
                                ),
                            },
                        },
                    )
                fanout_targets = requirements.get(
                    "fanout_capabilities_may_patch_or_inherit"
                )
                branch_target = requirements.get("branch_capability_must_patch")
                selected_targets = (
                    {str(item) for item in fanout_targets}
                    if isinstance(fanout_targets, list)
                    else {
                        str(branch_target)
                        if branch_target in targets
                        else next(
                            capability
                            for capability in (
                                "knowledge.visual_encyclopedia",
                                "utility.recipe_guidance",
                                "product.style_recommendation",
                                "utility.document_reading",
                            )
                            if capability in targets
                        )
                    }
                )
                generated = []
                for capability in sorted(skill_by_capability):
                    skill = skill_by_capability[capability]
                    entry = {
                        "capability_id": capability,
                        "action": "inherit",
                        "parent_skill_sha256": skill["skill_sha256"],
                    }
                    if capability in selected_targets:
                        template = template_by_capability[capability]
                        steps = [dict(item) for item in template["steps"]]
                        addition = " Model-generated Body treatment."
                        steps[0]["instruction"] = (
                            str(steps[0]["instruction"]) + addition
                        )
                        fallback = str(template["fallback_instruction"])
                        if (
                            capability == "knowledge.visual_encyclopedia"
                            and "not enough evidence" not in fallback.casefold()
                        ):
                            fallback += " State not enough evidence."
                        entry = {
                            **entry,
                            "action": "patch",
                            "patch": {
                                "objective": template["objective"],
                                "steps": steps,
                                "fallback_instruction": fallback,
                                "citation_source_ids": template["citation_source_ids"],
                            },
                        }
                    generated.append(entry)
                return self._result(
                    intent,
                    {"schema_version": 1, "skills": generated},
                )
            if operation == "s2_route_optimizer":
                if "s2" in self.reject_stages:
                    return self._result(intent, {"invalid": True})
                skill = skills[0]
                return self._result(
                    intent,
                    {
                        "edits": [
                            {
                                "capability_id": skill["capability_id"],
                                "description": str(skill["description"])
                                + " Boundary routing cues clarified.",
                            }
                        ]
                    },
                )
            if "full" in self.reject_stages:
                return self._result(intent, {"invalid": True})
            skill = skills[0]
            return self._result(
                intent,
                {
                    "edits": [
                        {
                            "capability_id": skill["capability_id"],
                            "body": str(skill["body"]) + "\nS3 evidence rule.\n",
                        }
                    ]
                },
            )
        if intent.role == "assistant":
            query = intent.payload["query"]
            assert isinstance(query, dict)
            query_id = str(query["query_id"])
            if query_id in self.assistant_fail_ids:
                return CallResult(
                    call_id=intent.call_id,
                    role=intent.role,
                    status="provider_error",
                    requested_model=intent.requested_model,
                    failure_reason="injected Assistant failure",
                )
            expected = str(query["canonical_capability"])
            config = str(intent.payload["config"])
            split = str(intent.payload["split"])
            bucket = _bucket(query_id)
            if config == "llm_static" and split == "static-opt800":
                # Bootstrap fixtures need both route-correct failures and
                # anchors so deterministic canary/body selection is covered.
                # Each capability's first eight rows are six body failures
                # followed by two anchors; the rest exercise route failures.
                position = int(query_id.rsplit("-", maxsplit=1)[1]) // len(CAPABILITIES)
                route_ok = position < 8
                evidence = True
                output_ok = position >= 6
            elif config in {"noskill", "llm_static"}:
                route_ok = config == "noskill" or bucket >= 2
                evidence = True
                output_ok = True
            elif "s1" in config and "s2" not in config:
                route_ok = bucket >= 1
                evidence = True
                output_ok = True
            elif "full" in config:
                route_ok = True
                evidence = True
                output_ok = True
            else:
                route_ok = True
                evidence = True
                output_ok = bucket >= 1
            reuse = intent.payload.get("reuse_parent_route_and_tool")
            action_reuse = intent.payload.get("reuse_parent_route_only")
            if isinstance(action_reuse, dict):
                selected = action_reuse.get("selected_capability")
                route_key = action_reuse.get("route_trace_key")
                tool_key = f"tool:{query_id}:{selected}:action-replay"
                trace = []
                replay = action_reuse.get("replay_context", {})
            elif isinstance(reuse, dict):
                selected = reuse.get("selected_capability")
                route_key = reuse.get("route_trace_key")
                tool_key = str(reuse["tool_trace_key"])
                if self.break_s3_common_trace:
                    tool_key += ":changed"
                trace = reuse.get("tool_trace", [])
                replay = reuse.get("replay_context", {})
            else:
                selected = (
                    expected
                    if route_ok or config == "noskill"
                    else CAPABILITIES[
                        (CAPABILITIES.index(expected) + 1) % len(CAPABILITIES)
                    ]
                )
                selected = None if config == "noskill" else selected
                route_key = None if config == "noskill" else f"route:{selected}"
                tool_key = f"tool:{query_id}:{selected}"
                trace = []
                bank = intent.payload.get("bank")
                bank_sha256 = (
                    bank.get("bank_sha256") if isinstance(bank, dict) else None
                )
                replay = {
                    "evidence": f"fake evidence for {query_id}",
                    "response": {"backbone_model": intent.requested_model},
                    "receipt": {
                        "model_calls": [
                            {
                                "requested_model": intent.requested_model,
                                "response_model": intent.requested_model,
                                "provider_request_id": f"fake-{intent.call_id}",
                            }
                        ]
                    },
                    "assistant_result": {
                        "bank_sha256": bank_sha256,
                        "backbone_model": intent.requested_model,
                    },
                }
            components = {
                "route_acceptable": route_ok,
                "no_hard_error": True,
                "tool_contract_pass": True,
                "evidence_grounded": evidence,
                "output_contract_pass": output_ok,
            }
            observation = {
                "schema_version": 1,
                "query_id": query_id,
                "response_text": f"fake response {config} {query_id}",
                "selected_capability": selected,
                "route_trace_key": route_key,
                "tool_trace_key": tool_key,
                "tool_trace": trace,
                "replay_context": replay,
                "answer_mode": "supported"
                if all(components.values())
                else "unresolved",
                "oracle_available": True,
                "gcs_components": components,
                "gcs_score": float(all(components.values())),
                "gcs_reason_codes": [],
                "hard_error": False,
                "card_violation": False,
                "evidence_violation": not evidence,
                "tool_violation": False,
                "source": str(query.get("image_path", "unknown")).split("/")[1],
                "repair": "none",
                "style_submode": "fake"
                if expected == "product.style_recommendation"
                else None,
                "assistant_contract": "core-fast-model-generated-action-response-v1",
            }
            return self._result(intent, {"observation": observation})
        if intent.role == "judge":
            query = intent.payload["query"]
            assistant = intent.payload["assistant_observation"]
            assert isinstance(query, dict) and isinstance(assistant, dict)
            if self.judge_first_status in {
                "empty_response",
                "schema_error",
                "provider_error",
            } and intent.call_id.endswith("attempt1"):
                return CallResult(
                    call_id=intent.call_id,
                    role=intent.role,
                    status=self.judge_first_status,  # type: ignore[arg-type]
                    requested_model=intent.requested_model,
                    failure_reason="injected first Judge failure",
                )
            if self.judge_first_status == "invalid_output" and intent.call_id.endswith(
                "attempt1"
            ):
                return self._result(intent, {"judge": {"invalid": True}})
            score = min(100.0, 40.0 + float(assistant["gcs_score"]) * 60.0)
            return self._result(
                intent,
                {
                    "judge": {
                        "schema_version": 1,
                        "query_id": query["query_id"],
                        "j_project": score,
                        "dimensions": {"project": score},
                    }
                },
            )
        query = intent.payload["query"]
        assert isinstance(query, dict)
        return self._result(
            intent, {"selected_capability": query["canonical_capability"]}
        )


def create_adapter(*, spec, cwd, base_dir):
    del spec, cwd, base_dir
    return FakeCoreFastAdapter()


def create_adapter_for_bridge():
    return FakeCoreFastAdapter()


__all__ = [
    "FakeCoreFastAdapter",
    "create_adapter",
    "create_adapter_for_bridge",
]
