from __future__ import annotations

import hashlib
from collections import Counter

from skillchain.evaluation.evaluator_outputs import VisualFeedbackOutput

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
                    "[policy_compatible] make the existing contract explicit",
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
            if operation == "s1_creator":
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
                target = next(
                    capability
                    for capability in (
                        "utility.recipe_guidance",
                        "product.style_recommendation",
                        "utility.document_reading",
                    )
                    if capability in targets
                )
                generated = []
                for capability in sorted(skill_by_capability):
                    skill = skill_by_capability[capability]
                    entry = {
                        "capability_id": capability,
                        "action": "inherit",
                        "parent_skill_sha256": skill["skill_sha256"],
                    }
                    if capability == target:
                        template = template_by_capability[capability]
                        entry = {
                            **entry,
                            "action": "patch",
                            "patch": {
                                "objective": template["objective"],
                                "steps": template["steps"],
                                "fallback_instruction": template[
                                    "fallback_instruction"
                                ],
                                "citation_source_ids": template["citation_source_ids"],
                                "semantic_policy": {
                                    "schema_version": 1,
                                    "policy_version": "core-fast-semantic-policy-v2",
                                    "evidence_terms": ["ingredient"],
                                    "require_all_terms": False,
                                    "abstain_when_no_evidence": True,
                                    "ocr_extraction_plan": "all-lines",
                                },
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
            if isinstance(reuse, dict):
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
