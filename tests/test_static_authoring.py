from __future__ import annotations

import base64
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from skillchain import config
from skillchain.llm import LLMResponse, LLMToolCall, LLMUsage
from skillchain.schemas import Skill
from skillchain.static_authoring import (
    AUTHORING_CONTENT_SCHEMA_NAME,
    AUTHORING_SUBMISSION_TOOL_NAME,
    FORCED_SUBMISSION_RESPONSE_FORMAT,
    PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    AuthoringBudgets,
    AuthoringContractError,
    CanonicalAuthoringRequest,
    ControlledAuthoringGateway,
    DraftToolStep,
    FixedDecoding,
    FormalContainerAuthoringGateway,
    ModelIdentity,
    OfflineAuthoringReplayGateway,
    OutputCoverage,
    PriceSchedule,
    PromptIdentity,
    ReviewChecklist,
    RuleCoverage,
    UnifiedChatTransport,
    authoring_content_json_schema,
    authoring_submission_tool_definition,
    build_authoring_content_output_contract,
    build_canonical_authoring_request,
    build_authoring_draft_bundle,
    build_authoring_packet,
    build_capability_draft,
    build_human_review,
    build_public_source_lock,
    build_public_source_material,
    build_spec_draft_bundle,
    finalize_llm_static,
    invoke_skill_operator,
    invoke_llm_static,
    load_authoring_packet,
    load_skill_markdown,
    load_static_bank,
    load_verified_authoring_input,
    load_verified_authoring_invocation_input,
    load_verified_authoring_sandbox_profile,
    load_verified_authoring_run,
    load_verified_price_schedule,
    load_verified_public_source_material,
    parse_skill_markdown,
    render_skill_markdown,
    run_spec_baseline,
    runtime_skill_for_capability,
)
from skillchain.task_spec import (
    load_default_task_specification,
    load_mvp_task_specification_v1,
)
from skillchain.taxonomy import load_default_taxonomy_registry
from skillchain.tools.contracts import RetrievalArtifactBinding
from skillchain.tools.kb_lookup import (
    KBCitation,
    KBHit,
    KBRetrievalArtifactBinding,
)
from skillchain.tools.model_artifacts import ModelRuntimeBinding
from skillchain.tools.registry import (
    MVPToolServices,
    RegistryError,
    ToolExecutionContext,
    ToolRegistry,
    build_mvp_registry,
    build_mvp_registry_spec,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)


@pytest.fixture
def contracts():
    return (
        load_default_taxonomy_registry(),
        load_default_task_specification(),
        build_mvp_registry_spec(),
    )


@pytest.fixture
def v5_contracts():
    return (
        load_default_taxonomy_registry(),
        load_mvp_task_specification_v1(),
        build_mvp_registry_spec(include_multi_product=True),
    )


class _Artifact:
    def __init__(self, kind: str):
        self.runtime_binding = ModelRuntimeBinding(
            artifact_kind=kind,
            model_id=f"static-fixture-{kind}",
            backend_name="fixture",
            backend_version="1",
            manifest_sha256=("a" if kind == "object_detector" else "b") * 64,
            artifacts=(),
        )


class _RuntimeService:
    def __init__(self, name: str, *, artifact_kind: str | None = None):
        self.name = name
        if artifact_kind is not None:
            self.artifact = _Artifact(artifact_kind)

    @property
    def formal_runtime_binding_sha256(self):
        return sha256_bytes(f"static-formal:{self.name}".encode())

    def __getattr__(self, _name):
        def unavailable(*_args, **_kwargs):
            raise RuntimeError("unused static authoring fixture service")

        return unavailable


class _ProductService(_RuntimeService):
    def __init__(self):
        super().__init__("product")
        self.artifact_binding = RetrievalArtifactBinding(
            mode="verified",
            index_integrity_sha256="1" * 64,
            eligibility_sha256="2" * 64,
            query_artifact_sha256="3" * 64,
            gallery_artifact_sha256="4" * 64,
            asset_catalog_sha256="5" * 64,
            products_parquet_sha256="6" * 64,
            leakage_policy_version="fixture-leakage-v1",
        )


class _KBService(_RuntimeService):
    def __init__(self):
        super().__init__("kb")
        self.binding = KBRetrievalArtifactBinding(
            mode="verified",
            kind="encyclopedia",
            bundle_sha256="1" * 64,
            index_integrity_sha256="2" * 64,
            catalog_sha256="3" * 64,
            catalog_entries_sha256="4" * 64,
            indexed_entries_sha256="5" * 64,
            tokenizer_policy_version="fixture-tokenizer-v1",
            ranking_policy_version="fixture-ranking-v1",
        )

    def artifact_binding_for(self, kind):
        if kind == "encyclopedia":
            return self.binding
        return self.binding.model_copy(update={"kind": "recipe"})

    def encyclopedia_lookup(self, _entity):
        return [
            KBHit(
                rank=1,
                score=1.0,
                title="fixture",
                text="fixture text",
                kind="encyclopedia",
                origin="dump",
                verification_status="source_verified",
                citation=KBCitation(
                    entry_id="entry-q-1",
                    source_dataset="fixture",
                    source_revision="v1",
                    source_record_id="q-1",
                    source_uri="https://example.invalid/source",
                    license_id="CC-BY-4.0",
                    char_start=0,
                    char_end=12,
                    excerpt_sha256=sha256_bytes(b"fixture text"),
                ),
                artifact_binding=self.binding,
            )
        ]


class _SafetyResolver:
    @property
    def formal_runtime_binding_sha256(self):
        return sha256_bytes(b"static-formal:safety")

    def __call__(self, _query_id, _asset_id):
        return None


def _formal_registry(
    *,
    product_service: _ProductService | None = None,
) -> ToolRegistry:
    return build_mvp_registry(
        MVPToolServices(
            product_search=product_service or _ProductService(),
            kb_lookup=_KBService(),
            object_detection=_RuntimeService(
                "detector", artifact_kind="object_detector"
            ),
            document_ocr=_RuntimeService("ocr", artifact_kind="document_ocr"),
            safety_approval_for=_SafetyResolver(),
        )
    )


@pytest.fixture
def formal_contracts(canonical_registry_factory):
    return (
        load_default_taxonomy_registry(),
        load_default_task_specification(),
        canonical_registry_factory(name="static-formal-registry").registry,
    )


def _prompt(
    template: str = "Return canonical structured Skills from these bytes only.",
):
    return PromptIdentity(
        prompt_id="static-author-v3",
        prompt_version="3.0.0",
        template=template,
        prompt_sha256=sha256_bytes(template.encode()),
    )


def _source(
    content: bytes = b"public authoring guidance v1",
    *,
    source_id: str = "public-guide",
    revision: str = "rev-1",
):
    return build_public_source_material(
        source_id=source_id,
        url=f"https://example.org/guides/{revision}",
        revision=revision,
        media_type="text/plain",
        content=content,
        license_id="CC-BY-4.0",
        license_evidence_url="https://example.org/licenses/cc-by-4.0",
        license_evidence=b"CC-BY-4.0 evidence bytes",
    )


def _input(
    contracts,
    *,
    model: str = "qwen-static-fixture",
    source_content: bytes = b"public authoring guidance v1",
    max_cost: int = 1000,
    public_sources=None,
    response_format: str = "canonical_authoring_payload_v1",
):
    taxonomy, tasks, registry = contracts
    return build_authoring_packet(
        taxonomy=taxonomy,
        task_specification=tasks,
        tool_registry=registry,
        prompt=_prompt(),
        model=ModelIdentity(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            model=model,
            revision=model.rsplit("-", 1)[-1],
        ),
        decoding=FixedDecoding(
            seed=20260721,
            max_output_tokens=200,
            response_format=response_format,
        ),
        price_schedule=PriceSchedule(
            schedule_id=f"fixture-{model}-2026-07",
            provider="qwen",
            model=model,
            input_microusd_per_million_tokens=1_000_000,
            output_microusd_per_million_tokens=1_000_000,
            source_input_microunits_per_million_tokens=1_000_000,
            source_output_microunits_per_million_tokens=1_000_000,
            conversion_microusd_per_source_unit=1_000_000,
            tier_max_input_tokens=32_000,
            source_url="https://example.org/pricing/rev-1",
            source_revision="rev-1",
        ),
        budgets=AuthoringBudgets(
            max_input_tokens=100,
            max_output_tokens=200,
            max_total_tokens=250,
            max_cost_microusd=max_cost,
            max_human_review_minutes=15,
        ),
        public_sources=(
            tuple(public_sources)
            if public_sources is not None
            else (_source(source_content),)
        ),
    )


def _verified_source(tmp_path: Path, content: bytes = b"public authoring guidance v1"):
    source_path = tmp_path / "public-guide.txt"
    acquisition_path = tmp_path / "acquisition.json"
    evidence_path = tmp_path / "license.txt"
    lock_path = tmp_path / "public-source-lock.json"
    acquisition = b'{"method":"https","revision":"rev-1"}\n'
    evidence = b"CC-BY-4.0 evidence bytes"
    source_path.write_bytes(content)
    acquisition_path.write_bytes(acquisition)
    evidence_path.write_bytes(evidence)
    lock = build_public_source_lock(
        source_id="public-guide",
        url="https://example.org/guides/rev-1",
        revision="rev-1",
        media_type="text/plain",
        content_sha256=sha256_bytes(content),
        acquisition_record_sha256=sha256_bytes(acquisition),
        license_id="CC-BY-4.0",
        license_evidence_url="https://example.org/licenses/cc-by-4.0",
        license_evidence_sha256=sha256_bytes(evidence),
    )
    lock_path.write_bytes(canonical_json_bytes(lock.model_dump(mode="json")))
    return load_verified_public_source_material(
        source_lock_path=lock_path,
        expected_source_lock_file_sha256=sha256_bytes(lock_path.read_bytes()),
        content_path=source_path,
        acquisition_record_path=acquisition_path,
        license_evidence_path=evidence_path,
    )


def _verified_input(
    contracts,
    tmp_path: Path,
    *,
    model="qwen-static-fixture",
    defer_tool_registry_runtime: bool = False,
    response_format: str = "canonical_authoring_payload_v1",
):
    taxonomy, tasks, registry = contracts
    source = _verified_source(tmp_path)
    schedule = PriceSchedule(
        schedule_id=f"fixture-{model}-2026-07",
        provider="qwen",
        model=model,
        input_microusd_per_million_tokens=1_000_000,
        output_microusd_per_million_tokens=1_000_000,
        source_input_microunits_per_million_tokens=1_000_000,
        source_output_microunits_per_million_tokens=1_000_000,
        conversion_microusd_per_source_unit=1_000_000,
        tier_max_input_tokens=32_000,
        source_url="https://example.org/pricing/rev-1",
        source_revision="rev-1",
    )
    schedule_path = tmp_path / "price-schedule.json"
    schedule_path.write_bytes(canonical_json_bytes(schedule.model_dump(mode="json")))
    verified_schedule = load_verified_price_schedule(
        schedule_path,
        expected_file_sha256=sha256_bytes(schedule_path.read_bytes()),
    )
    value = build_authoring_packet(
        taxonomy=taxonomy,
        task_specification=tasks,
        tool_registry=registry,
        prompt=_prompt(),
        model=ModelIdentity(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            model=model,
            revision=model.rsplit("-", 1)[-1],
        ),
        decoding=FixedDecoding(
            seed=20260721,
            max_output_tokens=200,
            response_format=response_format,
        ),
        price_schedule=verified_schedule,
        budgets=AuthoringBudgets(
            max_input_tokens=100,
            max_output_tokens=200,
            max_total_tokens=250,
            max_cost_microusd=1000,
            max_human_review_minutes=15,
        ),
        public_sources=(source,),
        defer_tool_registry_runtime=defer_tool_registry_runtime,
    )
    input_path = tmp_path / "verified-authoring-input.json"
    input_path.write_bytes(value.canonical_bytes())
    verified = load_verified_authoring_input(
        input_path,
        expected_file_sha256=sha256_bytes(input_path.read_bytes()),
        registry=registry,
        expected_taxonomy_sha256=taxonomy.taxonomy_sha256,
        expected_task_specification_sha256=tasks.task_spec_sha256,
        expected_tool_registry_sha256=registry.registry_sha256,
        expected_tool_registry_runtime_sha256=registry.registry_runtime_sha256,
        public_sources=(source,),
        price_schedule=verified_schedule,
    )
    return verified, source, verified_schedule


class RequestOnlyFakeTransport:
    """Has no contracts closure; derives everything from canonical request bytes."""

    def __init__(
        self,
        *,
        finish_reason="stop",
        input_tokens=40,
        output_tokens=80,
        provider=None,
        model=None,
        mutate_bundle=None,
    ):
        self.calls = []
        self.finish_reason = finish_reason
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.provider = provider
        self.model = model
        self.mutate_bundle = mutate_bundle

    def complete(self, canonical_request: bytes) -> LLMResponse:
        self.calls.append(canonical_request)
        raw = parse_canonical_json(canonical_request, label="fake request")
        request = CanonicalAuthoringRequest.model_validate(raw, strict=True)
        spec_bundle = build_spec_draft_bundle(request.authoring_input)
        source_id = request.authoring_input.public_sources[0].source_id
        bundle = build_authoring_draft_bundle(
            authoring_input_sha256=spec_bundle.authoring_input_sha256,
            drafts=tuple(
                build_capability_draft(
                    capability_id=draft.capability_id,
                    objective=draft.objective,
                    steps=draft.steps,
                    fallback_instruction=draft.fallback_instruction,
                    fallback_may_request_clarification=(
                        draft.fallback_may_request_clarification
                    ),
                    fallback_must_state_uncertainty=(
                        draft.fallback_must_state_uncertainty
                    ),
                    citation_source_ids=(source_id,),
                    rule_coverage=draft.rule_coverage,
                    output_coverage=draft.output_coverage,
                )
                for draft in spec_bundle.drafts
            ),
        )
        if self.mutate_bundle is not None:
            bundle = self.mutate_bundle(bundle)
        provider = self.provider or request.authoring_input.model.provider
        model = self.model or request.authoring_input.model.model
        payload = {
            "schema_version": 1,
            "drafts": [
                {
                    key: value
                    for key, value in draft.model_dump(mode="json").items()
                    if key != "draft_sha256"
                }
                for draft in bundle.drafts
            ],
        }
        return LLMResponse(
            provider=provider,
            endpoint=config.PROVIDER_ENDPOINTS[provider],
            requested_model=model,
            response_model=model,
            request_id="provider-request-1",
            text=canonical_json_bytes(payload).decode(),
            usage=LLMUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            ),
            finish_reason=self.finish_reason,
            latency_ms=17,
        )


class SubmissionFakeTransport(RequestOnlyFakeTransport):
    """Wrap the valid fixture payload in the v4 non-executable submission call."""

    def __init__(
        self,
        *,
        assistant_text="",
        submission_name=AUTHORING_SUBMISSION_TOOL_NAME,
        arguments_json=None,
        extra_submission=False,
        response_finish_reason="tool_calls",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.assistant_text = assistant_text
        self.submission_name = submission_name
        self.arguments_json = arguments_json
        self.extra_submission = extra_submission
        self.response_finish_reason = response_finish_reason

    def complete(self, canonical_request: bytes) -> LLMResponse:
        response = super().complete(canonical_request)
        arguments = self.arguments_json
        if arguments is None:
            # Deliberately provider-formatted rather than repository-canonical.
            arguments = json.dumps(
                json.loads(response.text),
                ensure_ascii=False,
                indent=2,
            )
        calls = [
            LLMToolCall(
                call_id="submission-1",
                name=self.submission_name,
                arguments_json=arguments,
            )
        ]
        if self.extra_submission:
            calls.append(
                LLMToolCall(
                    call_id="submission-2",
                    name=self.submission_name,
                    arguments_json=arguments,
                )
            )
        return response.model_copy(
            update={
                "text": self.assistant_text,
                "tool_calls": tuple(calls),
                "finish_reason": self.response_finish_reason,
            }
        )


class ContentSubmissionFakeTransport:
    """Emit v2 author content while deliberately varying representation order."""

    def __init__(self, mutate_payload=None):
        self.calls = []
        self.mutate_payload = mutate_payload

    def complete(self, canonical_request: bytes) -> LLMResponse:
        self.calls.append(canonical_request)
        raw = parse_canonical_json(canonical_request, label="content request")
        request = CanonicalAuthoringRequest.model_validate(raw, strict=True)
        source_ids = [item.source_id for item in request.authoring_input.public_sources]
        bundle = build_spec_draft_bundle(request.authoring_input)
        payload = {
            "schema_version": 2,
            "drafts": [
                {
                    "capability_id": draft.capability_id,
                    "objective": draft.objective,
                    "steps": [
                        {
                            "instruction": step.instruction,
                            "tool_name": step.tool_name,
                            "success_rule_ids": list(reversed(step.success_rule_ids)),
                        }
                        for step in draft.steps
                    ],
                    "fallback_instruction": draft.fallback_instruction,
                    "citation_source_ids": list(reversed(source_ids)),
                }
                for draft in reversed(bundle.drafts)
            ],
        }
        if self.mutate_payload is not None:
            payload = self.mutate_payload(payload)
        return LLMResponse(
            provider=request.authoring_input.model.provider,
            endpoint=request.authoring_input.model.endpoint,
            requested_model=request.authoring_input.model.model,
            response_model=request.authoring_input.model.model,
            request_id="provider-content-submission-1",
            text=json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            usage=LLMUsage(input_tokens=40, output_tokens=80),
            finish_reason="stop",
            latency_ms=17,
        )


def _invoke_diagnostic(value, tmp_path, transport=None):
    transport = transport or RequestOnlyFakeTransport()
    if value.decoding.response_format == PROVIDER_JSON_CONTENT_RESPONSE_FORMAT:
        request = build_canonical_authoring_request(value)
        response = transport.complete(request.canonical_bytes())
        gateway = OfflineAuthoringReplayGateway(value, response)
    else:
        gateway = ControlledAuthoringGateway(transport)
    invocation = invoke_llm_static(
        value,
        gateway,
        tmp_path,
        require_formal_eligibility=False,
    )
    return invocation, transport


def test_input_is_self_contained_content_addressed_and_formally_loadable(
    contracts, tmp_path
):
    value = _input(contracts)
    raw = parse_canonical_json(value.canonical_bytes(), label="input")

    assert raw["taxonomy"]["canonical_json"].endswith("\n")
    assert raw["task_specification"]["canonical_json"].endswith("\n")
    assert raw["tool_registry"]["canonical_json"].endswith("\n")
    assert base64.b64decode(raw["public_sources"][0]["content_base64"]) == (
        b"public authoring guidance v1"
    )
    assert raw["public_sources"][0]["content_text"] == ("public authoring guidance v1")
    assert raw["public_sources"][0]["text_transform"] == "utf8-identity-v1"
    assert (
        base64.b64decode(raw["public_sources"][0]["license_evidence_base64"])
        == b"CC-BY-4.0 evidence bytes"
    )
    assert value.tool_registry_runtime_sha256 == contracts[2].registry_runtime_sha256

    path = tmp_path / "input.json"
    path.write_bytes(value.canonical_bytes())
    loaded = load_authoring_packet(
        path,
        expected_file_sha256=sha256_bytes(path.read_bytes()),
        expected_registry=contracts[2],
    )
    assert loaded == value
    with pytest.raises(AuthoringContractError, match="digest mismatch"):
        load_authoring_packet(
            path,
            expected_file_sha256="0" * 64,
            expected_registry=contracts[2],
        )


def test_public_bytes_and_license_evidence_are_hash_verified():
    source = _source()
    raw = source.model_dump(mode="python")
    raw["content_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="content_sha256"):
        type(source).model_validate(raw, strict=True)
    raw = source.model_dump(mode="python")
    raw["license_evidence_base64"] = base64.b64encode(b"changed").decode()
    with pytest.raises(ValidationError, match="license evidence"):
        type(source).model_validate(raw, strict=True)
    raw = source.model_dump(mode="python")
    raw["license_id"] = "unknown"
    with pytest.raises(ValidationError, match="license_id"):
        type(source).model_validate(raw, strict=True)


def test_actual_public_content_is_scanned_for_evaluation_pollution():
    with pytest.raises(ValidationError, match="forbidden"):
        _source(b"hidden evaluation corpus labels")
    source = _source()
    raw = source.model_dump(mode="python")
    polluted = b"gold judge rubric"
    raw["license_evidence_text"] = polluted.decode("utf-8")
    raw["license_evidence_base64"] = base64.b64encode(polluted).decode()
    raw["license_evidence_sha256"] = sha256_bytes(polluted)
    raw["license_evidence_text"] = polluted.decode("utf-8")
    with pytest.raises(ValidationError, match="forbidden"):
        type(source).model_validate(raw, strict=True)


def test_public_authoring_material_requires_exact_readable_utf8_text():
    source = _source()
    raw = source.model_dump(mode="python")
    raw["content_text"] = "different readable text"
    with pytest.raises(ValidationError, match="readable text differs"):
        type(source).model_validate(raw, strict=True)

    with pytest.raises(AuthoringContractError, match="UTF-8"):
        build_public_source_material(
            source_id="invalid-utf8",
            url="https://example.test/material/rev-1",
            revision="rev-1",
            media_type="text/plain",
            content=b"\xff",
            license_id="CC-BY-4.0",
            license_evidence_url="https://example.test/license/rev-1",
            license_evidence=b"license evidence",
        )


def test_model_requires_seed_compatible_provider():
    with pytest.raises(ValidationError, match="seed support"):
        ModelIdentity(
            provider="claude",
            endpoint=config.PROVIDER_ENDPOINTS["claude"],
            model="claude-static-fixture",
            revision="fixture",
        )


def test_model_revision_and_price_conversion_are_fail_closed():
    with pytest.raises(ValidationError, match="revision must be immutable"):
        ModelIdentity(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            model="qwen3-vl-flash-latest",
            revision="latest",
        )
    with pytest.raises(ValidationError, match="end with"):
        ModelIdentity(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            model="qwen3-vl-flash",
            revision="2026-01-22",
        )
    schedule = PriceSchedule(
        schedule_id="conversion-fixture-v1",
        provider="qwen",
        model="qwen3-vl-flash-2026-01-22",
        input_microusd_per_million_tokens=22_500,
        output_microusd_per_million_tokens=225_000,
        source_input_microunits_per_million_tokens=150_000,
        source_output_microunits_per_million_tokens=1_500_000,
        conversion_microusd_per_source_unit=150_000,
        tier_max_input_tokens=32_000,
        source_url="https://example.org/pricing/rev-1",
        source_revision="rev-1",
    )
    raw = schedule.model_dump(mode="python")
    raw["input_microusd_per_million_tokens"] += 1
    with pytest.raises(ValidationError, match="frozen conversion"):
        PriceSchedule.model_validate(raw, strict=True)


def test_formal_gateway_fails_before_unisolated_python_transport_call(
    contracts, tmp_path
):
    transport = RequestOnlyFakeTransport()
    gateway = ControlledAuthoringGateway(transport)
    with pytest.raises(AuthoringContractError, match="not mechanically proven"):
        invoke_llm_static(_input(contracts), gateway, tmp_path)
    assert transport.calls == []
    assert gateway.attestation.formal_eligible is False
    assert gateway.attestation.process_isolated is False


def test_diagnostic_gateway_receives_only_canonical_request_and_audits_full_call(
    contracts, tmp_path
):
    value = _input(contracts)
    invocation, transport = _invoke_diagnostic(value, tmp_path)

    assert transport.calls == [invocation.request.canonical_bytes()]
    assert RequestOnlyFakeTransport.complete.__closure__ is None
    assert invocation.request.authoring_input == value
    assert invocation.request.decoding.seed == 20260721
    assert invocation.request.decoding.temperature_milli == 0
    assert invocation.request.decoding.top_p_milli == 1000
    assert invocation.request.decoding.max_output_tokens == 200
    assert invocation.cost_microusd == 120
    assert invocation.response.finish_reason == "stop"
    assert invocation.request_file_sha256 == sha256_bytes(
        (tmp_path / "authoring-request.json").read_bytes()
    )
    assert invocation.response_file_sha256 == sha256_bytes(
        (tmp_path / "authoring-response.json").read_bytes()
    )
    saved_response = parse_canonical_json(
        (tmp_path / "authoring-response.json").read_bytes(), label="saved response"
    )
    assert saved_response["request_id"] == "provider-request-1"
    assert saved_response["usage"] == {"input_tokens": 40, "output_tokens": 80}


def test_gateway_is_one_use(contracts, tmp_path):
    gateway = ControlledAuthoringGateway(RequestOnlyFakeTransport())
    value = _input(contracts)
    invoke_llm_static(
        value, gateway, tmp_path / "first", require_formal_eligibility=False
    )
    with pytest.raises(AuthoringContractError, match="only one call"):
        invoke_llm_static(
            value, gateway, tmp_path / "second", require_formal_eligibility=False
        )


def test_unified_transport_enforces_frozen_decode_and_single_attempt(
    contracts, tmp_path, monkeypatch
):
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        request_bytes = messages[1]["content"].encode("utf-8")
        return RequestOnlyFakeTransport().complete(request_bytes)

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    value = _input(contracts)
    invocation, _ = _invoke_diagnostic(value, tmp_path, UnifiedChatTransport())

    assert invocation.response.finish_reason == "stop"
    assert len(calls) == 1
    provider, messages, kwargs = calls[0]
    assert provider == "qwen"
    assert messages[0] == {"role": "system", "content": value.prompt.template}
    assert messages[1]["content"].encode() == invocation.request.canonical_bytes()
    assert kwargs == {
        "model": "qwen-static-fixture",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260721,
        "max_tokens": 200,
        "json_mode": True,
        "thinking": False,
        "max_attempts": 1,
        "record_usage": False,
    }


def test_unified_transport_forces_one_non_executable_submission(
    contracts, tmp_path, monkeypatch
):
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        request_bytes = messages[1]["content"].encode("utf-8")
        return SubmissionFakeTransport().complete(request_bytes)

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    value = _input(
        contracts,
        response_format=FORCED_SUBMISSION_RESPONSE_FORMAT,
    )
    invocation, _ = _invoke_diagnostic(value, tmp_path, UnifiedChatTransport())

    assert invocation.response.finish_reason == "tool_calls"
    assert invocation.response.text == ""
    assert len(invocation.response.tool_calls) == 1
    assert invocation.response.tool_calls[0].name == AUTHORING_SUBMISSION_TOOL_NAME
    assert invocation.pre_review_draft.drafts
    assert invocation.request.schema_version == 2
    assert invocation.request.output_contract is not None
    assert invocation.request.output_contract.tool_definition_sha256 == sha256_bytes(
        canonical_json_bytes(authoring_submission_tool_definition())
    )
    assert len(calls) == 1
    _, _, kwargs = calls[0]
    tool = authoring_submission_tool_definition()
    parameters = tool["function"]["parameters"]
    assert "$defs" not in parameters
    assert parameters["required"] == ["schema_version", "drafts"]
    assert kwargs == {
        "model": "qwen-static-fixture",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260721,
        "max_tokens": 200,
        "thinking": False,
        "max_attempts": 1,
        "record_usage": False,
        "json_mode": False,
        "tools": [tool],
        "tool_choice": {
            "type": "function",
            "function": {"name": AUTHORING_SUBMISSION_TOOL_NAME},
        },
        "parallel_tool_calls": False,
    }


def test_unified_transport_uses_provider_json_mode_for_v5_content(
    v5_contracts,
    monkeypatch,
):
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        return ContentSubmissionFakeTransport().complete(
            messages[1]["content"].encode("utf-8")
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    value = _input(
        v5_contracts,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )
    request = build_canonical_authoring_request(value)

    response = UnifiedChatTransport().complete(request.canonical_bytes())

    assert response.finish_reason == "stop"
    assert response.tool_calls == ()
    assert request.schema_version == 3
    assert request.output_contract is not None
    assert request.output_contract.provider_guarantee == "json_syntax_only"
    assert len(calls) == 1
    assert calls[0][2] == {
        "model": "qwen-static-fixture",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260721,
        "max_tokens": 200,
        "thinking": False,
        "max_attempts": 1,
        "record_usage": False,
        "json_mode": True,
    }


def test_v5_diagnostic_rejects_wrapped_provider_transport_before_complete(
    v5_contracts,
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_chat(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("provider transport must not be reached")

    class WrappedProviderTransport:
        def __init__(self):
            self.inner = UnifiedChatTransport()
            self.complete_calls = 0

        def complete(self, canonical_request):
            self.complete_calls += 1
            return self.inner.complete(canonical_request)

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    value = _input(
        v5_contracts,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )
    transport = WrappedProviderTransport()

    with pytest.raises(
        AuthoringContractError,
        match="sealed offline response replay",
    ):
        invoke_llm_static(
            value,
            ControlledAuthoringGateway(transport),
            tmp_path,
            require_formal_eligibility=False,
        )

    assert transport.complete_calls == 0
    assert calls == []

    direct_transport = WrappedProviderTransport()
    with pytest.raises(
        AuthoringContractError,
        match="cannot execute through a diagnostic transport",
    ):
        ControlledAuthoringGateway(direct_transport).complete_once(
            build_canonical_authoring_request(value).canonical_bytes()
        )
    assert direct_transport.complete_calls == 0
    assert calls == []


def test_content_submission_schema_is_packet_bound_and_compiler_owned(
    v5_contracts,
):
    value = _input(
        v5_contracts,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )
    schema = authoring_content_json_schema(value)
    contract = build_authoring_content_output_contract(value)
    variants = schema["properties"]["drafts"]["items"]["oneOf"]
    tasks = load_mvp_task_specification_v1().capabilities_by_id

    assert value.schema_version == 4
    assert value.compiler.compiler_version == "4.0.0"
    assert schema["title"] == AUTHORING_CONTENT_SCHEMA_NAME
    assert contract.authoring_input_sha256 == value.input_sha256
    assert contract.provider_response_format == "json_object"
    assert contract.provider_guarantee == "json_syntax_only"
    assert contract.json_schema_enforcement == "prompt_and_audit_only"
    assert contract.runner_packet_contract_validation is True
    assert contract.runner_validation_engine == "pydantic_and_trusted_compiler"
    assert contract.tools_supplied is False
    assert contract.json_schema_sha256 == sha256_bytes(canonical_json_bytes(schema))
    assert schema["properties"]["drafts"]["minItems"] == len(tasks)
    assert schema["properties"]["drafts"]["maxItems"] == len(tasks)
    assert len(variants) == len(tasks)
    for variant in variants:
        properties = variant["properties"]
        capability_id = properties["capability_id"]["const"]
        task = tasks[capability_id]
        assert set(properties) == {
            "capability_id",
            "objective",
            "steps",
            "fallback_instruction",
            "citation_source_ids",
        }
        assert not {
            "rule_coverage",
            "output_coverage",
            "fallback_may_request_clarification",
            "fallback_must_state_uncertainty",
            "step_id",
        } & set(properties)
        step_properties = properties["steps"]["items"]["properties"]
        assert step_properties["tool_name"]["enum"] == list(task.allowed_tools)
        assert step_properties["success_rule_ids"]["uniqueItems"] is True
        assert set(step_properties["success_rule_ids"]["items"]["enum"]) == {
            rule.rule_id for rule in task.success_criteria
        }


def test_authoring_contract_generations_must_advance_together(
    contracts,
    v5_contracts,
):
    with pytest.raises(AuthoringContractError, match="must be selected together"):
        _input(
            contracts,
            response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
        )

    taxonomy, legacy_tasks, _legacy_registry = contracts
    _taxonomy, _v1_tasks, v2_registry = v5_contracts
    with pytest.raises(AuthoringContractError, match="must be selected together"):
        _input(
            (taxonomy, legacy_tasks, v2_registry),
            response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
        )


def test_authoring_rejects_task_spec_tool_version_not_in_embedded_registry(
    v5_contracts,
):
    value = _input(
        v5_contracts,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )
    raw = value.model_dump(mode="json")
    registry = json.loads(value.tool_registry.canonical_json)
    tool = next(
        item for item in registry["tools"] if item["name"] == "multi_product_search"
    )
    tool["tool_version"] = "9.9.9"
    tool_unsigned = dict(tool)
    tool_unsigned.pop("spec_sha256")
    tool["spec_sha256"] = sha256_bytes(canonical_json_bytes(tool_unsigned))
    registry_unsigned = dict(registry)
    registry_unsigned.pop("registry_sha256")
    registry["registry_sha256"] = sha256_bytes(canonical_json_bytes(registry_unsigned))
    registry_bytes = canonical_json_bytes(registry)
    raw["tool_registry"]["identity_sha256"] = registry["registry_sha256"]
    raw["tool_registry"]["canonical_json"] = registry_bytes.decode("utf-8")
    raw["tool_registry"]["bytes_sha256"] = sha256_bytes(registry_bytes)
    raw.pop("input_sha256")
    raw["input_sha256"] = sha256_bytes(canonical_json_bytes(raw))
    coordinated = type(value).model_validate(raw, strict=True)

    with pytest.raises(AuthoringContractError, match="not executable"):
        authoring_content_json_schema(coordinated)


def test_v5_spec_bank_binds_registry_v2_generation(v5_contracts, tmp_path):
    value = _input(
        v5_contracts,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )

    result = run_spec_baseline(
        value,
        tmp_path,
        require_formal_eligibility=False,
    )

    assert result.bank.schema_version == 3
    assert result.bank.compiler.compiler_version == "4.0.0"
    assert result.bank.runtime_binding_policy == "registry-runtime-v2"
    assert result.manifest.schema_version == 3
    assert result.manifest.compiler_version == "4.0.0"
    assert result.manifest.runtime_binding_policy == "registry-runtime-v2"
    multi_skill = runtime_skill_for_capability(
        result.bank,
        "product.multi_search",
        v5_contracts[2],
    )
    assert multi_skill.operators == ["multi_product_search"]


def test_content_submission_injects_task_spec_fields_and_canonical_order(
    v5_contracts, tmp_path
):
    value = _input(
        v5_contracts,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )
    invocation, transport = _invoke_diagnostic(
        value,
        tmp_path,
        ContentSubmissionFakeTransport(),
    )
    tasks = load_mvp_task_specification_v1().capabilities_by_id

    assert len(transport.calls) == 1
    assert invocation.request.schema_version == 3
    assert invocation.request.output_contract is not None
    assert invocation.request.output_contract.provider_response_format == "json_object"
    assert tuple(
        draft.capability_id for draft in invocation.pre_review_draft.drafts
    ) == (tuple(sorted(tasks)))
    for draft in invocation.pre_review_draft.drafts:
        task = tasks[draft.capability_id]
        assert draft.fallback_may_request_clarification == (
            task.fallback.may_request_clarification
        )
        assert draft.fallback_must_state_uncertainty is True
        assert draft.output_coverage.response_kind == task.output_contract.response_kind
        assert draft.rule_coverage.success_rule_ids == tuple(
            sorted(rule.rule_id for rule in task.success_criteria)
        )
        assert tuple(step.step_id for step in draft.steps) == tuple(
            f"{index:02d}.{step.tool_name}"
            for index, step in enumerate(draft.steps, start=1)
        )
        assert all(
            step.success_rule_ids == tuple(sorted(step.success_rule_ids))
            for step in draft.steps
        )


@pytest.mark.parametrize(
    ("mutate_payload", "message"),
    [
        (
            lambda payload: {"drafts": payload["drafts"]},
            "author content JSON is not valid",
        ),
        (
            lambda payload: {
                **payload,
                "drafts": [
                    {
                        **payload["drafts"][0],
                        "rule_coverage": {},
                    },
                    *payload["drafts"][1:],
                ],
            },
            "author content JSON is not valid",
        ),
        (
            lambda payload: {
                **payload,
                "drafts": [
                    {
                        **payload["drafts"][0],
                        "steps": [
                            {
                                **payload["drafts"][0]["steps"][0],
                                "success_rule_ids": [
                                    payload["drafts"][0]["steps"][0][
                                        "success_rule_ids"
                                    ][0],
                                    payload["drafts"][0]["steps"][0][
                                        "success_rule_ids"
                                    ][0],
                                ],
                            },
                            *payload["drafts"][0]["steps"][1:],
                        ],
                    },
                    *payload["drafts"][1:],
                ],
            },
            "author content JSON is not valid",
        ),
        (
            lambda payload: {
                **payload,
                "drafts": [
                    {
                        **payload["drafts"][0],
                        "steps": [
                            {
                                **step,
                                "success_rule_ids": [
                                    payload["drafts"][0]["steps"][0][
                                        "success_rule_ids"
                                    ][0]
                                ],
                            }
                            for step in payload["drafts"][0]["steps"]
                        ],
                    },
                    *payload["drafts"][1:],
                ],
            },
            "do not cover every success rule",
        ),
    ],
)
def test_content_submission_fails_closed(
    v5_contracts, tmp_path, mutate_payload, message
):
    value = _input(
        v5_contracts,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )
    with pytest.raises(AuthoringContractError, match=message):
        _invoke_diagnostic(
            value,
            tmp_path,
            ContentSubmissionFakeTransport(mutate_payload),
        )
    assert (tmp_path / "authoring-response.json").is_file()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"finish_reason": "length"}, "truncated"),
        (
            {
                "tool_calls": (
                    LLMToolCall(
                        call_id="unexpected-call",
                        name="submit_authoring_content",
                        arguments_json="{}",
                    ),
                )
            },
            "must not contain tool calls",
        ),
        ({"text": '{"schema_version":2,"schema_version":2,"drafts":[]}'}, "not valid"),
    ],
)
def test_json_content_envelope_fails_closed(
    v5_contracts,
    tmp_path,
    updates,
    message,
):
    class MutatingTransport(ContentSubmissionFakeTransport):
        def complete(self, canonical_request):
            return super().complete(canonical_request).model_copy(update=updates)

    value = _input(
        v5_contracts,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )
    with pytest.raises(AuthoringContractError, match=message):
        _invoke_diagnostic(value, tmp_path, MutatingTransport())
    assert (tmp_path / "authoring-response.json").is_file()


def test_request_v3_rejects_coordinated_json_schema_rehash(v5_contracts, tmp_path):
    value = _input(
        v5_contracts,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )
    invocation, _ = _invoke_diagnostic(
        value,
        tmp_path,
        ContentSubmissionFakeTransport(),
    )
    raw = parse_canonical_json(
        invocation.request.canonical_bytes(),
        label="JSON content request",
    )
    output_contract = raw["output_contract"]
    schema = json.loads(output_contract["json_schema_canonical_json"])
    schema["properties"]["drafts"]["maxItems"] += 1
    schema_bytes = canonical_json_bytes(schema)
    output_contract["json_schema_canonical_json"] = schema_bytes.decode("utf-8")
    output_contract["json_schema_sha256"] = sha256_bytes(schema_bytes)
    output_contract.pop("contract_sha256")
    output_contract["contract_sha256"] = sha256_bytes(
        canonical_json_bytes(output_contract)
    )
    raw.pop("request_sha256")
    raw["request_sha256"] = sha256_bytes(canonical_json_bytes(raw))

    with pytest.raises(ValidationError, match="request-v3 output contract drifted"):
        CanonicalAuthoringRequest.model_validate(raw, strict=True)


def test_request_v2_rejects_coordinated_output_contract_rehash(contracts, tmp_path):
    value = _input(
        contracts,
        response_format=FORCED_SUBMISSION_RESPONSE_FORMAT,
    )
    invocation, _ = _invoke_diagnostic(
        value,
        tmp_path,
        SubmissionFakeTransport(),
    )
    raw = parse_canonical_json(
        invocation.request.canonical_bytes(),
        label="structured request",
    )
    output_contract = raw["output_contract"]
    output_contract["parallel_tool_calls"] = True
    output_contract.pop("contract_sha256")
    output_contract["contract_sha256"] = sha256_bytes(
        canonical_json_bytes(output_contract)
    )
    raw.pop("request_sha256")
    raw["request_sha256"] = sha256_bytes(canonical_json_bytes(raw))

    with pytest.raises(ValidationError, match="parallel_tool_calls"):
        CanonicalAuthoringRequest.model_validate(raw, strict=True)


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (
            SubmissionFakeTransport(assistant_text="also returning prose"),
            "must not include assistant text",
        ),
        (
            SubmissionFakeTransport(submission_name="unexpected"),
            "unexpected submission name",
        ),
        (
            SubmissionFakeTransport(extra_submission=True),
            "exactly one submission",
        ),
        (
            SubmissionFakeTransport(
                response_finish_reason="stop",
            ),
            "did not finish with tool_calls",
        ),
        (
            SubmissionFakeTransport(
                arguments_json=('{"schema_version":1,"schema_version":1,"drafts":[]}'),
            ),
            "arguments are not valid",
        ),
    ],
)
def test_forced_submission_envelope_fails_closed(
    contracts, tmp_path, transport, message
):
    value = _input(
        contracts,
        response_format=FORCED_SUBMISSION_RESPONSE_FORMAT,
    )
    with pytest.raises(AuthoringContractError, match=message):
        _invoke_diagnostic(value, tmp_path, transport)
    assert (tmp_path / "authoring-response.json").is_file()


@pytest.mark.parametrize(
    "transport, message",
    [
        (RequestOnlyFakeTransport(finish_reason="length"), "truncated"),
        (RequestOnlyFakeTransport(input_tokens=0), "positive"),
        (RequestOnlyFakeTransport(output_tokens=0), "positive"),
        (RequestOnlyFakeTransport(input_tokens=101), "budget"),
        (RequestOnlyFakeTransport(output_tokens=201), "budget"),
        (
            RequestOnlyFakeTransport(provider="claude", model="claude-static-fixture"),
            "identity drift",
        ),
    ],
)
def test_call_fails_closed_but_preserves_full_response_audit(
    contracts, tmp_path, transport, message
):
    with pytest.raises(AuthoringContractError, match=message):
        _invoke_diagnostic(_input(contracts), tmp_path, transport)
    assert len(transport.calls) == 1
    assert (tmp_path / "authoring-request.json").is_file()
    assert (tmp_path / "authoring-response.json").is_file()


def test_formal_script_recovers_provider_facts_from_rejected_draft(contracts, tmp_path):
    from scripts.run_formal_authoring import _persisted_response_facts

    value = _input(contracts)
    response = LLMResponse(
        provider=value.model.provider,
        endpoint=value.model.endpoint,
        requested_model=value.model.model,
        response_model=value.model.model,
        request_id="provider-rejected-draft-1",
        text='{"drafts":[],"schema_version":1}',
        usage=LLMUsage(input_tokens=40, output_tokens=80),
        finish_reason="stop",
        latency_ms=123,
    )
    (tmp_path / "authoring-response.json").write_bytes(
        canonical_json_bytes(response.model_dump(mode="json"))
    )

    facts = _persisted_response_facts(
        tmp_path,
        SimpleNamespace(value=value),
    )

    assert facts["formal_provider_call_eligible"] is True
    assert facts["cost_microusd"] == 120
    assert facts["input_tokens"] == 40
    assert facts["output_tokens"] == 80
    assert facts["endpoint"] == value.model.endpoint
    assert facts["requested_model"] == value.model.model
    assert facts["response_text_sha256"] == sha256_bytes(response.text.encode())


def test_formal_script_recovers_forced_submission_provider_facts(contracts, tmp_path):
    from scripts.run_formal_authoring import _persisted_response_facts

    value = _input(
        contracts,
        response_format=FORCED_SUBMISSION_RESPONSE_FORMAT,
    )
    arguments = '{"schema_version":1,"drafts":[]}'
    response = LLMResponse(
        provider=value.model.provider,
        endpoint=value.model.endpoint,
        requested_model=value.model.model,
        response_model=value.model.model,
        request_id="provider-rejected-submission-1",
        text="",
        tool_calls=(
            LLMToolCall(
                call_id="submission-1",
                name=AUTHORING_SUBMISSION_TOOL_NAME,
                arguments_json=arguments,
            ),
        ),
        usage=LLMUsage(input_tokens=40, output_tokens=80),
        finish_reason="tool_calls",
        latency_ms=123,
    )
    (tmp_path / "authoring-response.json").write_bytes(
        canonical_json_bytes(response.model_dump(mode="json"))
    )

    facts = _persisted_response_facts(
        tmp_path,
        SimpleNamespace(value=value),
    )

    assert facts["formal_provider_call_eligible"] is True
    assert facts["response_format"] == FORCED_SUBMISSION_RESPONSE_FORMAT
    assert facts["tool_call_count"] == 1
    assert facts["submission_arguments_sha256"] == sha256_bytes(arguments.encode())


def test_formal_script_recovers_json_content_provider_facts(v5_contracts, tmp_path):
    from scripts.run_formal_authoring import _persisted_response_facts

    value = _input(
        v5_contracts,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )
    content = '{"schema_version":2,"drafts":[]}'
    response = LLMResponse(
        provider=value.model.provider,
        endpoint=value.model.endpoint,
        requested_model=value.model.model,
        response_model=value.model.model,
        request_id="provider-rejected-json-content-1",
        text=content,
        usage=LLMUsage(input_tokens=40, output_tokens=80),
        finish_reason="stop",
        latency_ms=123,
    )
    (tmp_path / "authoring-response.json").write_bytes(
        canonical_json_bytes(response.model_dump(mode="json"))
    )

    facts = _persisted_response_facts(
        tmp_path,
        SimpleNamespace(value=value),
    )

    assert facts["formal_provider_call_eligible"] is True
    assert facts["response_format"] == PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
    assert facts["tool_call_count"] == 0
    assert facts["submission_arguments_sha256"] is None
    assert facts["response_text_sha256"] == sha256_bytes(content.encode())


def test_price_schedule_budget_is_runner_calculated(contracts, tmp_path):
    with pytest.raises(AuthoringContractError, match="budget"):
        _invoke_diagnostic(_input(contracts, max_cost=119), tmp_path)


def test_noncanonical_draft_text_is_rejected(contracts, tmp_path):
    class NonCanonicalTransport(RequestOnlyFakeTransport):
        def complete(self, canonical_request):
            response = super().complete(canonical_request)
            return response.model_copy(update={"text": response.text.rstrip("\n")})

    with pytest.raises(AuthoringContractError, match="not canonical"):
        _invoke_diagnostic(_input(contracts), tmp_path, NonCanonicalTransport())


def test_model_returns_hash_free_payload_and_runner_adds_identities(
    contracts, tmp_path
):
    invocation, _ = _invoke_diagnostic(_input(contracts), tmp_path)
    raw = parse_canonical_json(
        invocation.response.text.encode("utf-8"), label="hash-free author payload"
    )
    assert set(raw) == {"schema_version", "drafts"}
    assert all("draft_sha256" not in item for item in raw["drafts"])
    assert invocation.pre_review_draft.authoring_input_sha256 == (
        invocation.authoring_input_sha256
    )
    assert all(item.draft_sha256 for item in invocation.pre_review_draft.drafts)


def _replace_first_draft(bundle, **updates):
    first = bundle.drafts[0]
    values = {
        "capability_id": first.capability_id,
        "objective": first.objective,
        "steps": first.steps,
        "fallback_instruction": first.fallback_instruction,
        "fallback_may_request_clarification": (
            first.fallback_may_request_clarification
        ),
        "fallback_must_state_uncertainty": first.fallback_must_state_uncertainty,
        "citation_source_ids": first.citation_source_ids,
        "rule_coverage": first.rule_coverage,
        "output_coverage": first.output_coverage,
    }
    values.update(updates)
    replacement = build_capability_draft(**values)
    return build_authoring_draft_bundle(
        authoring_input_sha256=bundle.authoring_input_sha256,
        drafts=(replacement, *bundle.drafts[1:]),
    )


@pytest.mark.parametrize(
    "coverage_field",
    [
        "precondition_rule_ids",
        "success_rule_ids",
        "failure_rule_ids",
        "acceptable_answer_rule_ids",
        "safety_rule_ids",
        "fallback_trigger_rule_ids",
        "fallback_response_rule_ids",
        "indeterminate_rule_ids",
    ],
)
def test_compiler_requires_exact_coverage_of_every_rule_family(
    contracts, tmp_path, coverage_field
):
    def mutate(bundle):
        first = bundle.drafts[0]
        coverage = first.rule_coverage.model_dump(mode="python")
        coverage[coverage_field] = ("invented.rule",)
        return _replace_first_draft(
            bundle, rule_coverage=RuleCoverage.model_validate(coverage)
        )

    transport = RequestOnlyFakeTransport(mutate_bundle=mutate)
    with pytest.raises(AuthoringContractError, match="every frozen rule"):
        _invoke_diagnostic(_input(contracts), tmp_path, transport)


def test_compiler_requires_complete_output_contract(contracts, tmp_path):
    def mutate(bundle):
        first = bundle.drafts[0]
        output = first.output_coverage.model_dump(mode="python")
        output["required_sections"] = ("invented_section",)
        return _replace_first_draft(
            bundle, output_coverage=OutputCoverage.model_validate(output)
        )

    with pytest.raises(AuthoringContractError, match="complete output contract"):
        _invoke_diagnostic(
            _input(contracts),
            tmp_path,
            RequestOnlyFakeTransport(mutate_bundle=mutate),
        )


def test_compiler_rejects_tool_outside_capability_allowlist(contracts, tmp_path):
    all_tools = {item.name for item in contracts[2].specs()}

    def mutate(bundle):
        first = bundle.drafts[0]
        allowed = {item.tool_name for item in first.steps}
        tool = sorted(all_tools - allowed)[0]
        old = first.steps[0]
        step = DraftToolStep(
            step_id=f"00.{tool}",
            instruction=f"Invoke {tool}.",
            tool_name=tool,
            success_rule_ids=old.success_rule_ids,
        )
        return _replace_first_draft(bundle, steps=(step, *first.steps[1:]))

    with pytest.raises(AuthoringContractError, match="outside capability allowlist"):
        _invoke_diagnostic(
            _input(contracts),
            tmp_path,
            RequestOnlyFakeTransport(mutate_bundle=mutate),
        )


def test_compiler_requires_fallback_flags_step_union_and_verified_citations(
    contracts, tmp_path
):
    def wrong_fallback(bundle):
        first = bundle.drafts[0]
        return _replace_first_draft(
            bundle,
            fallback_may_request_clarification=(
                not first.fallback_may_request_clarification
            ),
        )

    with pytest.raises(AuthoringContractError, match="fallback flags"):
        _invoke_diagnostic(
            _input(contracts),
            tmp_path / "fallback",
            RequestOnlyFakeTransport(mutate_bundle=wrong_fallback),
        )

    def incomplete_steps(bundle):
        first = bundle.drafts[0]
        one_rule = (first.rule_coverage.success_rule_ids[0],)
        steps = tuple(
            step.model_copy(update={"success_rule_ids": one_rule})
            for step in first.steps
        )
        return _replace_first_draft(bundle, steps=steps)

    with pytest.raises(AuthoringContractError, match="collectively cover"):
        _invoke_diagnostic(
            _input(contracts),
            tmp_path / "steps",
            RequestOnlyFakeTransport(mutate_bundle=incomplete_steps),
        )

    def missing_citation(bundle):
        return _replace_first_draft(bundle, citation_source_ids=())

    with pytest.raises(AuthoringContractError, match="at least one verified"):
        _invoke_diagnostic(
            _input(contracts),
            tmp_path / "missing-citation",
            RequestOnlyFakeTransport(mutate_bundle=missing_citation),
        )

    def unknown_citation(bundle):
        return _replace_first_draft(bundle, citation_source_ids=("invented-source",))

    with pytest.raises(AuthoringContractError, match="unverified public source"):
        _invoke_diagnostic(
            _input(contracts),
            tmp_path / "citation",
            RequestOnlyFakeTransport(mutate_bundle=unknown_citation),
        )


def test_human_review_compiles_post_review_and_records_verifiable_diff(
    contracts, tmp_path
):
    value = _input(contracts)
    invocation, _ = _invoke_diagnostic(value, tmp_path)
    first = invocation.pre_review_draft.drafts[0]
    changed = _replace_first_draft(
        invocation.pre_review_draft,
        objective=first.objective + " Preserve explicit uncertainty.",
    )
    review = build_human_review(
        pre_review=invocation.pre_review_draft,
        post_review=changed,
        reviewer_id="human-reviewer-1",
        review_minutes=4,
        change_reason="Clarified uncertainty without adding new facts.",
    )
    result = finalize_llm_static(value, invocation, review, tmp_path)

    changed_skill = next(
        item for item in result.bank.skills if item.capability_id == first.capability_id
    )
    assert changed_skill.description.endswith("Preserve explicit uncertainty.")
    assert "Preserve explicit uncertainty." in changed_skill.body
    assert review.changed is True
    assert result.manifest.human_review_sha256 == review.review_sha256
    assert result.manifest.formal_eligible is False
    assert result.manifest.formal_ineligibility_reason == (
        "python_transport_is_not_mechanically_isolated"
    )


def test_llm_static_is_structurally_nonformal_and_full_audit_reloads(
    formal_contracts, tmp_path
):
    contracts = formal_contracts
    trust_root = tmp_path / "trust"
    trust_root.mkdir()
    verified, source, schedule = _verified_input(contracts, trust_root)
    run_root = tmp_path / "llm-run"
    invocation, _ = _invoke_diagnostic(verified.value, run_root)
    review = build_human_review(
        pre_review=invocation.pre_review_draft,
        post_review=invocation.pre_review_draft,
        reviewer_id="human-1",
        review_minutes=1,
        change_reason="Verified the unchanged structured draft.",
    )
    result = finalize_llm_static(verified.value, invocation, review, run_root)
    manifest_path = run_root / "authoring-manifest.json"
    loaded = load_verified_authoring_run(
        run_root,
        expected_manifest_file_sha256=sha256_bytes(manifest_path.read_bytes()),
        registry=contracts[2],
        expected_taxonomy_sha256=contracts[0].taxonomy_sha256,
        expected_task_specification_sha256=contracts[1].task_spec_sha256,
        expected_tool_registry_sha256=contracts[2].registry_sha256,
        expected_tool_registry_runtime_sha256=(contracts[2].registry_runtime_sha256),
        public_sources=(source,),
        price_schedule=schedule,
    )
    assert loaded.result.bank == result.bank
    assert loaded.result.manifest.formal_eligible is False

    raw = result.manifest.model_dump(mode="json")
    raw["formal_eligible"] = True
    raw["formal_ineligibility_reason"] = None
    raw.pop("manifest_sha256")
    raw["manifest_sha256"] = sha256_bytes(canonical_json_bytes(raw))
    forged_bytes = canonical_json_bytes(raw)
    manifest_path.write_bytes(forged_bytes)
    with pytest.raises(AuthoringContractError, match="manifest audit values"):
        load_verified_authoring_run(
            run_root,
            expected_manifest_file_sha256=sha256_bytes(forged_bytes),
            registry=contracts[2],
            expected_taxonomy_sha256=contracts[0].taxonomy_sha256,
            expected_task_specification_sha256=contracts[1].task_spec_sha256,
            expected_tool_registry_sha256=contracts[2].registry_sha256,
            expected_tool_registry_runtime_sha256=(
                contracts[2].registry_runtime_sha256
            ),
            public_sources=(source,),
            price_schedule=schedule,
        )


def test_forced_submission_full_audit_finalizes_and_reloads(formal_contracts, tmp_path):
    contracts = formal_contracts
    trust_root = tmp_path / "structured-trust"
    trust_root.mkdir()
    verified, source, schedule = _verified_input(
        contracts,
        trust_root,
        response_format=FORCED_SUBMISSION_RESPONSE_FORMAT,
    )
    run_root = tmp_path / "structured-llm-run"
    invocation, _ = _invoke_diagnostic(
        verified.value,
        run_root,
        SubmissionFakeTransport(),
    )
    review = build_human_review(
        pre_review=invocation.pre_review_draft,
        post_review=invocation.pre_review_draft,
        reviewer_id="human-structured-1",
        review_minutes=1,
        change_reason="Verified the unchanged forced-submission draft.",
    )
    result = finalize_llm_static(verified.value, invocation, review, run_root)
    arguments = invocation.response.tool_calls[0].arguments_json.encode("utf-8")
    assert result.manifest.draft_text_sha256 == sha256_bytes(arguments)

    manifest_path = run_root / "authoring-manifest.json"
    loaded = load_verified_authoring_run(
        run_root,
        expected_manifest_file_sha256=sha256_bytes(manifest_path.read_bytes()),
        registry=contracts[2],
        expected_taxonomy_sha256=contracts[0].taxonomy_sha256,
        expected_task_specification_sha256=contracts[1].task_spec_sha256,
        expected_tool_registry_sha256=contracts[2].registry_sha256,
        expected_tool_registry_runtime_sha256=contracts[2].registry_runtime_sha256,
        public_sources=(source,),
        price_schedule=schedule,
    )
    assert loaded.result.bank == result.bank
    assert loaded.result.manifest.finish_reason == "tool_calls"


def test_formal_container_runner_binds_profile_receipt_and_reload(
    formal_contracts, tmp_path, monkeypatch
):
    contracts = formal_contracts
    trust_root = tmp_path / "trust-formal-runner"
    trust_root.mkdir()
    verified, source, schedule = _verified_input(
        contracts,
        trust_root,
        defer_tool_registry_runtime=True,
    )
    invocation_input = load_verified_authoring_invocation_input(
        verified.path,
        expected_file_sha256=verified.file_sha256,
        expected_taxonomy_sha256=contracts[0].taxonomy_sha256,
        expected_task_specification_sha256=contracts[1].task_spec_sha256,
        expected_tool_registry_sha256=contracts[2].registry_sha256,
        public_sources=(source,),
        price_schedule=schedule,
    )
    engine = tmp_path / "docker.exe"
    engine.write_bytes(Path(sys.executable).read_bytes())
    image_digest = "1" * 64
    profile_payload = {
        "schema_version": 2,
        "engine": "docker",
        "engine_path": str(engine.resolve()),
        "engine_binary_sha256": sha256_bytes(engine.read_bytes()),
        "image_reference": f"example/skillchain-author@sha256:{image_digest}",
        "image_digest": image_digest,
        "network_name": "provider-egress-qwen-v1",
        "network_id": "3" * 64,
        "network_policy_sha256": "2" * 64,
        "proxy_url": "http://skillchain-egress-proxy:3128",
        "proxy_container_name": "skillchain-egress-proxy",
        "proxy_image_digest": image_digest,
        "proxy_external_network_name": "provider-egress-external-v1",
        "allowed_provider_endpoint": config.PROVIDER_ENDPOINTS["qwen"],
        "credential_env_name": config.PROVIDER_API_KEY_ENV["qwen"],
        "memory_megabytes": 512,
        "cpu_count_milli": 1000,
        "pids_limit": 64,
        "timeout_seconds": 60,
    }
    profile_payload["profile_sha256"] = sha256_bytes(
        canonical_json_bytes(profile_payload)
    )
    profile_path = tmp_path / "sandbox-profile.json"
    profile_bytes = canonical_json_bytes(profile_payload)
    profile_path.write_bytes(profile_bytes)
    profile = load_verified_authoring_sandbox_profile(
        profile_path, expected_file_sha256=sha256_bytes(profile_bytes)
    )
    monkeypatch.setenv(config.PROVIDER_API_KEY_ENV["qwen"], "temporary-test-key")
    commands = []
    worker_failure = {"enabled": False}

    def fake_container_run(command, **kwargs):
        if command[1:3] == ["network", "inspect"]:
            payload = [
                {
                    "Id": "3" * 64,
                    "Internal": True,
                    "Containers": {"proxy-id": {}},
                }
            ]
            return subprocess.CompletedProcess(
                command, 0, json.dumps(payload).encode(), b""
            )
        if command[1:3] == ["container", "inspect"]:
            payload = [
                {
                    "Id": "proxy-id",
                    "Image": f"sha256:{image_digest}",
                    "State": {"Running": True},
                    "Config": {
                        "Cmd": [
                            "python",
                            "-I",
                            "-m",
                            "skillchain.runners.egress_proxy",
                        ],
                        "Env": [
                            "SKILLCHAIN_ALLOWED_HOST=dashscope.aliyuncs.com",
                            "SKILLCHAIN_ALLOWED_PORT=443",
                        ],
                    },
                    "HostConfig": {
                        "ReadonlyRootfs": True,
                        "CapDrop": ["ALL"],
                        "SecurityOpt": ["no-new-privileges"],
                    },
                    "NetworkSettings": {
                        "Networks": {
                            "provider-egress-qwen-v1": {},
                            "provider-egress-external-v1": {},
                        }
                    },
                }
            ]
            return subprocess.CompletedProcess(
                command, 0, json.dumps(payload).encode(), b""
            )
        commands.append((command, kwargs))
        if worker_failure["enabled"]:
            return subprocess.CompletedProcess(
                command,
                17,
                b"captured-worker-stdout",
                b"captured-worker-stderr",
            )
        mount_values = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--mount"
        ]
        input_mount, output_mount = mount_values
        output_root = Path(output_mount.split(",dst=/output", 1)[0][14:])
        input_root = Path(input_mount.split(",dst=/input", 1)[0][14:])
        request_bytes = (input_root / "authoring-request.json").read_bytes()
        response = RequestOnlyFakeTransport().complete(request_bytes)
        (output_root / "authoring-response.json").write_bytes(
            canonical_json_bytes(response.model_dump(mode="json"))
        )
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(
        "skillchain.static_authoring.subprocess.run", fake_container_run
    )
    run_root = tmp_path / "formal-llm-static"
    invocation = invoke_llm_static(
        invocation_input,
        FormalContainerAuthoringGateway(profile),
        run_root,
    )
    review = build_human_review(
        pre_review=invocation.pre_review_draft,
        post_review=invocation.pre_review_draft,
        reviewer_id="human-formal-1",
        review_minutes=1,
        change_reason="Verified the isolated structured draft.",
    )
    result = finalize_llm_static(verified, invocation, review, run_root)
    assert result.manifest.formal_eligible is True
    assert result.manifest.formal_ineligibility_reason is None
    assert invocation.isolation_attestation.execution_mode == "container_job"
    assert commands[0][0][1:6] == [
        "run",
        "--rm",
        "--pull=never",
        "--read-only",
        "--cap-drop=ALL",
    ]
    assert commands[0][1]["shell"] is False

    manifest_path = run_root / "authoring-manifest.json"
    loaded = load_verified_authoring_run(
        run_root,
        expected_manifest_file_sha256=sha256_bytes(manifest_path.read_bytes()),
        registry=contracts[2],
        expected_taxonomy_sha256=contracts[0].taxonomy_sha256,
        expected_task_specification_sha256=contracts[1].task_spec_sha256,
        expected_tool_registry_sha256=contracts[2].registry_sha256,
        expected_tool_registry_runtime_sha256=contracts[2].registry_runtime_sha256,
        public_sources=(source,),
        price_schedule=schedule,
        sandbox_profile=profile,
    )
    assert loaded.result.manifest.formal_eligible is True

    worker_failure["enabled"] = True
    failed_root = tmp_path / "formal-llm-static-failed"
    with pytest.raises(AuthoringContractError, match="isolated authoring job"):
        invoke_llm_static(
            invocation_input,
            FormalContainerAuthoringGateway(profile),
            failed_root,
        )
    assert (failed_root / "authoring-input.json").is_file()
    assert (failed_root / "authoring-request.json").is_file()
    assert (failed_root / "container-stdout.bin").read_bytes() == (
        b"captured-worker-stdout"
    )
    assert (failed_root / "container-stderr.bin").read_bytes() == (
        b"captured-worker-stderr"
    )
    failure = parse_canonical_json(
        (failed_root / "job-failure.json").read_bytes(),
        label="job failure",
    )
    assert failure["exit_code"] == 17


def test_human_review_rejects_bad_checklist_diff_and_pollution(contracts, tmp_path):
    value = _input(contracts)
    invocation, _ = _invoke_diagnostic(value, tmp_path)
    with pytest.raises(ValidationError, match="ReviewChecklist"):
        build_human_review(
            pre_review=invocation.pre_review_draft,
            post_review=invocation.pre_review_draft,
            reviewer_id="human-1",
            review_minutes=1,
            change_reason="No changes.",
            checklist=("schema_valid",),
        )
    with pytest.raises((AuthoringContractError, ValidationError), match="forbidden"):
        build_human_review(
            pre_review=invocation.pre_review_draft,
            post_review=invocation.pre_review_draft,
            reviewer_id="human-1",
            review_minutes=1,
            change_reason="Read hidden_corpus.json before approval.",
        )
    with pytest.raises(ValidationError, match="greater than 0"):
        build_human_review(
            pre_review=invocation.pre_review_draft,
            post_review=invocation.pre_review_draft,
            reviewer_id="human-1",
            review_minutes=0,
            change_reason="No changes.",
        )
    with pytest.raises(ValidationError, match="source_citations_checked"):
        build_human_review(
            pre_review=invocation.pre_review_draft,
            post_review=invocation.pre_review_draft,
            reviewer_id="human-1",
            review_minutes=1,
            change_reason="No changes.",
            checklist=ReviewChecklist.model_construct(source_citations_checked=False),
        )


def test_human_review_cannot_change_structural_permissions(contracts, tmp_path):
    value = _input(contracts)
    invocation, _ = _invoke_diagnostic(value, tmp_path)
    first = invocation.pre_review_draft.drafts[0]
    one_rule = (first.rule_coverage.success_rule_ids[0],)
    changed_step = first.steps[0].model_copy(update={"success_rule_ids": one_rule})
    changed = _replace_first_draft(
        invocation.pre_review_draft,
        steps=(changed_step, *first.steps[1:]),
    )
    with pytest.raises(ValidationError, match="allowed edit scope"):
        build_human_review(
            pre_review=invocation.pre_review_draft,
            post_review=changed,
            reviewer_id="human-1",
            review_minutes=1,
            change_reason="Attempted a structural rewrite.",
        )


def test_human_review_budget_fails_closed(contracts, tmp_path):
    value = _input(contracts)
    invocation, _ = _invoke_diagnostic(value, tmp_path)
    review = build_human_review(
        pre_review=invocation.pre_review_draft,
        post_review=invocation.pre_review_draft,
        reviewer_id="human-1",
        review_minutes=16,
        change_reason="No changes were needed.",
    )
    with pytest.raises(AuthoringContractError, match="review budget"):
        finalize_llm_static(value, invocation, review, tmp_path)


def test_finalize_rejects_coordinated_in_memory_or_saved_audit_tamper(
    contracts, tmp_path
):
    value = _input(contracts)
    invocation, _ = _invoke_diagnostic(value, tmp_path)
    review = build_human_review(
        pre_review=invocation.pre_review_draft,
        post_review=invocation.pre_review_draft,
        reviewer_id="human-1",
        review_minutes=1,
        change_reason="No changes were needed.",
    )
    response_path = tmp_path / "authoring-response.json"
    raw = parse_canonical_json(response_path.read_bytes(), label="response")
    raw["latency_ms"] += 1
    response_path.write_bytes(canonical_json_bytes(raw))
    with pytest.raises(AuthoringContractError, match="does not match invocation"):
        finalize_llm_static(value, invocation, review, tmp_path)

    # Restore persisted bytes, then forge only the review flag in memory.
    response_path.write_bytes(
        canonical_json_bytes(invocation.response.model_dump(mode="json"))
    )
    forged_review = review.model_copy(update={"changed": True})
    with pytest.raises(AuthoringContractError, match="review artifact"):
        finalize_llm_static(value, invocation, forged_review, tmp_path)


def test_spec_baseline_identity_and_bank_ignore_all_public_source_variants(
    contracts, tmp_path
):
    first_input = _input(contracts)
    second_input = _input(
        contracts,
        model="qwen-different-fixture",
        max_cost=9999,
        public_sources=(
            _source(
                b"different public bytes",
                source_id="alternate-guide-a",
                revision="rev-a",
            ),
            _source(
                b"another public source",
                source_id="alternate-guide-b",
                revision="rev-b",
            ),
        ),
    )
    first = run_spec_baseline(
        first_input, tmp_path / "first", require_formal_eligibility=False
    )
    second = run_spec_baseline(
        second_input, tmp_path / "second", require_formal_eligibility=False
    )

    assert first_input.input_sha256 != second_input.input_sha256
    assert first.bank.construction_identity_sha256 == (
        second.bank.construction_identity_sha256
    )
    assert first.bank.canonical_bytes() == second.bank.canonical_bytes()
    assert all(
        not item.citation_source_ids
        for item in build_spec_draft_bundle(first_input).drafts
    )
    assert "source: " in first.bank.skills[0].body
    assert "alternate-guide" not in second.bank.skills[0].body
    assert first.manifest.authoring_input_sha256 != (
        second.manifest.authoring_input_sha256
    )
    assert first.manifest.formal_eligible is False
    assert first.manifest.formal_ineligibility_reason == (
        "unverified_in_memory_authoring_input"
    )


def test_diagnostic_registry_is_provisional_only(contracts, tmp_path):
    registry = contracts[2]
    assert registry.formal_runtime_ready is False

    provisional = run_spec_baseline(
        _input(contracts),
        tmp_path / "provisional",
        require_formal_eligibility=False,
    )
    assert provisional.manifest.formal_eligible is False

    trust_root = tmp_path / "diagnostic-trust"
    trust_root.mkdir()
    with pytest.raises(AuthoringContractError, match="explicit runtime bindings"):
        _verified_input(contracts, trust_root)


def test_public_builder_does_not_trust_self_reported_service_hashes():
    class DivergentProductService(_ProductService):
        def trace_text_product_search(self, _query):
            raise RuntimeError("different behavior hidden behind the same self-hash")

    first = _formal_registry()
    second = _formal_registry(
        product_service=DivergentProductService(),
    )

    assert first.registry_runtime_sha256 == second.registry_runtime_sha256
    assert first.formal_runtime_ready is False
    assert second.formal_runtime_ready is False
    with pytest.raises(RegistryError, match="reviewed concrete services"):
        second.require_formal_runtime()


def test_formal_spec_requires_verified_handle_and_full_run_reloads(
    formal_contracts, tmp_path
):
    contracts = formal_contracts
    with pytest.raises(AuthoringContractError, match="VerifiedAuthoringInput"):
        run_spec_baseline(_input(contracts), tmp_path / "refused")

    trust_root = tmp_path / "trust"
    trust_root.mkdir()
    verified, source, schedule = _verified_input(contracts, trust_root)
    run_root = tmp_path / "formal-run"
    result = run_spec_baseline(verified, run_root)
    manifest_path = run_root / "authoring-manifest.json"
    manifest_digest = sha256_bytes(manifest_path.read_bytes())
    loaded = load_verified_authoring_run(
        run_root,
        expected_manifest_file_sha256=manifest_digest,
        registry=contracts[2],
        expected_taxonomy_sha256=contracts[0].taxonomy_sha256,
        expected_task_specification_sha256=contracts[1].task_spec_sha256,
        expected_tool_registry_sha256=contracts[2].registry_sha256,
        expected_tool_registry_runtime_sha256=(contracts[2].registry_runtime_sha256),
        public_sources=(source,),
        price_schedule=schedule,
    )

    assert result.manifest.formal_eligible is True
    assert result.manifest.formal_ineligibility_reason is None
    assert result.manifest.authoring_input_verification == "externally_verified"
    assert result.bank.construction_identity_policy == "spec-content-v1"
    assert result.bank.runtime_binding_policy == "registry-runtime-v1"
    assert loaded.result.bank == result.bank
    assert loaded.manifest_file_sha256 == manifest_digest
    assert contracts[2].formal_runtime_ready is True

    with pytest.raises(AuthoringContractError, match="protocol locks"):
        load_verified_authoring_input(
            verified.path,
            expected_file_sha256=verified.file_sha256,
            registry=contracts[2],
            expected_taxonomy_sha256="0" * 64,
            expected_task_specification_sha256=contracts[1].task_spec_sha256,
            expected_tool_registry_sha256=contracts[2].registry_sha256,
            expected_tool_registry_runtime_sha256=(
                contracts[2].registry_runtime_sha256
            ),
            public_sources=(source,),
            price_schedule=schedule,
        )


def test_frozen_primary_packet_has_no_public_material_and_defers_runtime(
    formal_contracts, tmp_path
):
    root = Path(__file__).resolve().parents[1]
    lock = json.loads(
        (root / "specs/authoring/authoring-freeze-lock-v3.json").read_text(
            encoding="utf-8"
        )
    )
    variant = lock["variants"]["primary"]
    packet_path = root / variant["packet_file"]
    price_path = root / variant["price_schedule_file"]
    schedule = load_verified_price_schedule(
        price_path,
        expected_file_sha256=variant["price_schedule_file_sha256"],
    )
    taxonomy, tasks, registry = formal_contracts
    verified = load_verified_authoring_input(
        packet_path,
        expected_file_sha256=variant["packet_file_sha256"],
        registry=registry,
        expected_taxonomy_sha256=taxonomy.taxonomy_sha256,
        expected_task_specification_sha256=tasks.task_spec_sha256,
        expected_tool_registry_sha256=registry.registry_sha256,
        expected_tool_registry_runtime_sha256=registry.registry_runtime_sha256,
        public_sources=(),
        price_schedule=schedule,
    )
    assert verified.value.schema_version == 3
    assert verified.value.external_verification == "externally_verified"
    assert verified.value.model.model == config.LEGACY_AUTHOR_MODEL
    assert verified.value.model.revision == config.LEGACY_AUTHOR_MODEL_REVISION
    assert verified.value.public_sources == ()
    assert verified.value.reference_skill_bundle is None
    assert verified.value.tool_registry_runtime_binding == (
        "deferred_until_bank_compile"
    )
    assert verified.value.tool_registry_runtime_sha256 is None
    result = run_spec_baseline(verified, tmp_path / "frozen-primary-spec")
    assert result.manifest.formal_eligible is True
    assert result.bank.tool_registry_runtime_sha256 == registry.registry_runtime_sha256


def test_verified_handle_rereads_external_source_and_price_files(
    formal_contracts, tmp_path
):
    contracts = formal_contracts
    source_root = tmp_path / "source-tamper"
    source_root.mkdir()
    verified, source, _ = _verified_input(contracts, source_root)
    source.content_path.write_bytes(b"changed public guidance")
    with pytest.raises(AuthoringContractError, match="content digest mismatch"):
        run_spec_baseline(verified, tmp_path / "source-run")

    price_root = tmp_path / "price-tamper"
    price_root.mkdir()
    verified, _, schedule = _verified_input(contracts, price_root)
    schedule.path.write_bytes(b"{}\n")
    with pytest.raises(AuthoringContractError, match="file digest mismatch"):
        run_spec_baseline(verified, tmp_path / "price-run")


def test_external_manifest_digest_rejects_coordinated_self_rehash(
    formal_contracts, tmp_path
):
    contracts = formal_contracts
    trust_root = tmp_path / "trust"
    trust_root.mkdir()
    verified, source, schedule = _verified_input(contracts, trust_root)
    run_root = tmp_path / "run"
    run_spec_baseline(verified, run_root)
    manifest_path = run_root / "authoring-manifest.json"
    original_digest = sha256_bytes(manifest_path.read_bytes())
    raw = parse_canonical_json(manifest_path.read_bytes(), label="manifest")
    raw["prompt_sha256"] = "0" * 64
    raw.pop("manifest_sha256")
    raw["manifest_sha256"] = sha256_bytes(canonical_json_bytes(raw))
    manifest_path.write_bytes(canonical_json_bytes(raw))
    with pytest.raises(AuthoringContractError, match="file digest mismatch"):
        load_verified_authoring_run(
            run_root,
            expected_manifest_file_sha256=original_digest,
            registry=contracts[2],
            expected_taxonomy_sha256=contracts[0].taxonomy_sha256,
            expected_task_specification_sha256=contracts[1].task_spec_sha256,
            expected_tool_registry_sha256=contracts[2].registry_sha256,
            expected_tool_registry_runtime_sha256=(
                contracts[2].registry_runtime_sha256
            ),
            public_sources=(source,),
            price_schedule=schedule,
        )


def test_real_skill_bank_roundtrip_and_runtime_consumption(contracts, tmp_path):
    result = run_spec_baseline(
        _input(contracts), tmp_path, require_formal_eligibility=False
    )
    bank_path = tmp_path / "static-bank.json"
    loaded = load_static_bank(
        bank_path,
        expected_file_sha256=sha256_bytes(bank_path.read_bytes()),
        registry=contracts[2],
    )

    assert loaded.canonical_bytes() == result.bank.canonical_bytes()
    assert len(loaded.runtime_skills()) == len(contracts[1].capabilities)
    capability = contracts[1].capabilities[0].capability_id
    runtime = runtime_skill_for_capability(loaded, capability, contracts[2])
    artifact = next(item for item in loaded.skills if item.capability_id == capability)
    assert isinstance(runtime, Skill)
    assert runtime.model_dump() == {
        "slug": artifact.slug,
        "version": artifact.version,
        "description": artifact.description,
        "body": artifact.body,
        "static_refs": list(artifact.static_refs),
        "operators": list(artifact.operators),
    }
    assert artifact.description in artifact.body
    assert set(runtime.operators) <= set(contracts[1].capabilities[0].allowed_tools)
    rendered = render_skill_markdown(artifact)
    assert parse_skill_markdown(rendered) == artifact
    skill_path = tmp_path / "skills" / artifact.slug / "SKILL.md"
    assert (
        load_skill_markdown(
            skill_path,
            expected_file_sha256=sha256_bytes(skill_path.read_bytes()),
        )
        == artifact
    )


def test_runtime_skill_dispatches_declared_operator_through_live_registry(
    formal_contracts, tmp_path
):
    contracts = formal_contracts
    result = run_spec_baseline(
        _input(contracts), tmp_path, require_formal_eligibility=False
    )
    capability = "knowledge.visual_encyclopedia"
    runtime = runtime_skill_for_capability(result.bank, capability, contracts[2])
    context = ToolExecutionContext(
        query_id="q-1",
        query_asset_id="asset-1",
        query_text="dress",
        resolve_asset=lambda _asset_id: tmp_path / "query.png",
    )
    invocation = invoke_skill_operator(
        runtime,
        "encyclopedia_lookup",
        {"entity": "dress"},
        context,
        contracts[2],
    )
    assert len(invocation.output) == 1
    assert invocation.output[0]["citation"]["entry_id"] == "entry-q-1"
    assert invocation.tool_name == "encyclopedia_lookup"
    with pytest.raises(AuthoringContractError, match="not declared"):
        invoke_skill_operator(
            runtime,
            "recipe_lookup",
            {"dish": "soup"},
            context,
            contracts[2],
        )


def test_bank_loader_binds_live_registry_runtime_hash(contracts, tmp_path):
    result = run_spec_baseline(
        _input(contracts),
        tmp_path / "original",
        require_formal_eligibility=False,
    )
    raw = result.bank.model_dump(mode="json")
    raw["tool_registry_runtime_sha256"] = "0" * 64
    raw.pop("bank_sha256")
    raw["bank_sha256"] = sha256_bytes(canonical_json_bytes(raw))
    forged = type(result.bank).model_validate(raw)
    path = tmp_path / "coordinated-bank.json"
    path.write_bytes(forged.canonical_bytes())
    with pytest.raises(AuthoringContractError, match="live registry binding"):
        load_static_bank(
            path,
            expected_file_sha256=sha256_bytes(path.read_bytes()),
            registry=contracts[2],
        )


def test_forged_in_process_input_is_revalidated(contracts, tmp_path):
    value = _input(contracts)
    forged = value.model_copy(update={"input_sha256": "0" * 64})
    with pytest.raises(AuthoringContractError, match="frozen schema"):
        run_spec_baseline(forged, tmp_path)


def test_runner_surfaces_have_no_private_data_parameters():
    forbidden = {
        "corpus",
        "evaluation",
        "bank",
        "trajectory",
        "labels",
        "response",
        "rubric",
        "gold",
        "judge",
        "gate",
        "test",
        "result",
    }
    for function in (invoke_llm_static, finalize_llm_static, run_spec_baseline):
        assert not set(inspect.signature(function).parameters) & forbidden
