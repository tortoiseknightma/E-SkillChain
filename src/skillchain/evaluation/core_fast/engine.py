from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import ValidationError

from skillchain import config as project_config
from skillchain.evolution.s1_gcs_gate import (
    S1GCSGateError,
    S1_CONTRACT_REGRESSION_REASON_CODES,
    screen_s1_development_patches,
)
from skillchain.evolution.s1_sparse_patch import (
    CompiledPolicySurfaceBranch,
    PolicySurfaceCompositionReceiptV1,
    SparseCompilationReceiptV1,
    S1SparsePatchError,
    bind_sparse_patch_draft,
    compile_counterfactual_semantic_policy_branch,
    compile_counterfactual_policy_branch,
    compile_counterfactual_typed_policy_branch,
    compile_sparse_s1_candidate,
    compile_policy_surface_branch,
    compose_policy_surface_branches,
    compose_screened_sparse_bank,
    counterfactual_semantic_policy_patch_output_json_schema,
    counterfactual_policy_patch_output_json_schema,
    counterfactual_typed_policy_patch_output_json_schema,
    decode_sparse_parent_content,
    dual_policy_patch_output_json_schema,
    load_sparse_compilation_receipt,
    parse_dual_policy_patch,
    parse_counterfactual_semantic_policy_patch,
    parse_counterfactual_policy_patch,
    parse_counterfactual_typed_policy_patch,
    sparse_author_content_lexical_guard,
    sparse_patch_output_json_schema,
)
from skillchain.evaluation.evaluator_outputs import VisualFeedbackOutput
from skillchain.evaluation.portfolio_gcs import (
    GCS_CAPABILITY_ORDER,
    GCS_BOOTSTRAP_POLICY_VERSION,
    GCS_BOOTSTRAP_REPLICATES,
    GCS_BOOTSTRAP_ROOT_SEED,
    GCSQueryScoreV2,
    build_gcs_population_v2,
)
from skillchain.schemas import Query
from skillchain.static_authoring import (
    AuthoringInput,
    StaticBankArtifact,
    render_skill_markdown,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

from .adapters import CoreFastAdapter
from .feedback_selection import (
    CounterfactualEvidenceError,
    build_discovery_failure_summary,
    build_discovery_feedback_population,
    build_feedback_selection_manifest,
    build_parent_counterfactual_manifest,
    build_parent_counterfactual_population,
    project_feedback_observation,
    project_feedback_observation_for_surface,
    project_feedback_query,
    select_feedback_samples,
    select_parent_counterfactual_samples,
)
from .models import (
    AssistantObservation,
    CAPABILITIES,
    CONFIGS,
    CallIntent,
    CallResult,
    CallRole,
    CoreFastSpec,
    FixedSample,
    FixedSamples,
    JudgeObservation,
    SPLIT_COUNTS,
    StageDecision,
)
from .pacing import StartPacer
from .store import (
    CallStore,
    FastStoreError,
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
)


class FastPathError(RuntimeError):
    pass


_S1_BOUNDED_RISK_POLICY_VERSION = "s1-capability-bounded-risk-v1"
_S1_CAPABILITY_MIN_GAINS = 1
_S1_CAPABILITY_MIN_NET_GAIN = 1
_S1_CAPABILITY_MAX_REGRESSIONS = 2
_S1_CAPABILITY_MIN_GAIN_REGRESSION_RATIO = 4.0
_S1_CAPABILITY_MAX_FAILURE_SEVERITY_ESCALATIONS = 0
_COUNTERFACTUAL_PROPOSAL_MODES = frozenset(
    {
        "single-surface-counterfactual-fanout-v4",
        "single-surface-counterfactual-fanout-v5",
        "single-surface-counterfactual-fanout-v6",
    }
)
_COUNTERFACTUAL_SELECTION_POLICIES = frozenset(
    {
        "parent-counterfactual-v6",
        "parent-counterfactual-v7",
        "parent-counterfactual-v8",
        "parent-counterfactual-v9",
    }
)


class _S1CandidateRejected(ValueError):
    """Stable, non-sensitive reason for rejecting a sparse Creator proposal."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ValidationSummary:
    query_count: int
    split_counts: dict[str, int]
    capability_count: int
    static_bank_sha256: str
    observed_cost_cny: float
    projected_worst_case_cost_cny: float
    runtime_ready: bool
    discovery_count: int = 600
    replay_count: int = 200
    feedback_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "query_count": self.query_count,
            "split_counts": self.split_counts,
            "capability_count": self.capability_count,
            "static_bank_sha256": self.static_bank_sha256,
            "observed_cost_cny": self.observed_cost_cny,
            "projected_worst_case_cost_cny": self.projected_worst_case_cost_cny,
            "runtime_ready": self.runtime_ready,
            "discovery_count": self.discovery_count,
            "replay_count": self.replay_count,
            "feedback_count": self.feedback_count,
        }


def _read_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise FastPathError(f"blank JSONL row at {path}:{line_number}")
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise FastPathError(
                        f"invalid JSONL row at {path}:{line_number}"
                    ) from error
    except OSError as error:
        raise FastPathError(f"cannot read required artifact: {path}") from error
    return rows


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise FastPathError(f"cannot hash required artifact: {path}") from error
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in value)


def _bank_by_capability(bank: StaticBankArtifact) -> dict[str, object]:
    return {skill.capability_id: skill for skill in bank.skills}


def _observation_from_result(result: CallResult, query_id: str) -> AssistantObservation:
    if result.status != "success" or not result.schema_valid or result.output is None:
        return AssistantObservation(
            query_id=query_id,
            response_text="",
            selected_capability=None,
            route_trace_key=None,
            tool_trace_key=f"failed:{result.call_id}",
            answer_mode="unresolved",
            oracle_available=False,
            gcs_components={
                "route_acceptable": False,
                "no_hard_error": False,
                "tool_contract_pass": False,
                "evidence_grounded": False,
                "output_contract_pass": False,
            },
            gcs_score=0.0,
            gcs_reason_codes=(),
            hard_error=True,
            card_violation=True,
            evidence_violation=True,
            tool_violation=True,
        )
    payload = result.output.get("observation", result.output)
    try:
        return AssistantObservation.model_validate(payload, strict=True)
    except ValidationError:
        # A captured but malformed model/adapter output is an experiment
        # failure, not an orchestration crash.  It stays in the fixed
        # denominator as the same conservative hard-error row used above.
        return AssistantObservation(
            query_id=query_id,
            response_text="",
            selected_capability=None,
            route_trace_key=None,
            tool_trace_key=f"invalid:{result.call_id}",
            answer_mode="unresolved",
            oracle_available=False,
            gcs_components={
                "route_acceptable": False,
                "no_hard_error": False,
                "tool_contract_pass": False,
                "evidence_grounded": False,
                "output_contract_pass": False,
            },
            gcs_score=0.0,
            gcs_reason_codes=(),
            hard_error=True,
            card_violation=True,
            evidence_violation=True,
            tool_violation=True,
        )


def _judge_from_result(result: CallResult, query_id: str) -> JudgeObservation | None:
    if result.status != "success" or not result.schema_valid or result.output is None:
        return None
    payload = result.output.get("judge", result.output)
    try:
        return JudgeObservation.model_validate(payload, strict=True)
    except ValidationError:
        return None


class CoreFastEngine:
    def __init__(
        self,
        *,
        spec: CoreFastSpec,
        spec_path: Path,
        output_root: Path,
        adapter: CoreFastAdapter,
    ) -> None:
        self.spec = spec
        self.spec_path = spec_path.resolve()
        self.base_dir = self.spec_path.parent
        self.output_root = output_root.resolve()
        self.adapter = adapter
        self.calls = CallStore(self.output_root, spec)
        self._feedback_start_pacer = StartPacer(
            spec.concurrency.feedback_requests_per_second
        )
        self._queries: tuple[Query, ...] | None = None
        self._query_by_id: dict[str, Query] | None = None
        self._static_bank: StaticBankArtifact | None = None
        self._opt_static: dict[str, AssistantObservation] | None = None
        self._s1_parent_bank: StaticBankArtifact | None = None
        self._s1_parent_opt: dict[str, AssistantObservation] | None = None
        self._opt_attribution: dict[str, dict[str, object]] | None = None
        self._opt_fold_roles: dict[str, str] | None = None
        self._val_gate_roles: dict[str, str] | None = None
        self._s1_authoring_input: AuthoringInput | None = None

    # ------------------------------- loading/validation --------------------

    def _path(self, field_name: str) -> Path:
        return self.spec.resolved_path(field_name, base_dir=self.base_dir)

    def queries(self) -> tuple[Query, ...]:
        if self._queries is not None:
            return self._queries
        values: list[Query] = []
        for index, raw in enumerate(_read_jsonl(self._path("queries")), start=1):
            try:
                values.append(Query.model_validate(raw, strict=True))
            except ValidationError as error:
                raise FastPathError(f"invalid Core query at row {index}") from error
        ids = tuple(item.query_id for item in values)
        if len(ids) != len(set(ids)):
            raise FastPathError("Core query IDs are not unique")
        self._queries = tuple(values)
        self._query_by_id = {item.query_id: item for item in values}
        return self._queries

    def query_by_id(self) -> dict[str, Query]:
        self.queries()
        assert self._query_by_id is not None
        return self._query_by_id

    def static_bank(self) -> StaticBankArtifact:
        if self._static_bank is not None:
            return self._static_bank
        path = self._path("static_bank")
        if (
            self.spec.static_bank_file_sha256 is not None
            and _file_sha(path) != self.spec.static_bank_file_sha256
        ):
            raise FastPathError("Static Bank file SHA-256 differs from the frozen spec")
        try:
            self._static_bank = StaticBankArtifact.model_validate_json(
                path.read_bytes(), strict=True
            )
        except (OSError, ValidationError) as error:
            raise FastPathError(f"invalid Static Bank: {path}") from error
        self._require_six_capabilities(self._static_bank)
        return self._static_bank

    def _bound_path(self, value: str) -> Path:
        expanded = Path(os.path.expandvars(value))
        return (
            expanded if expanded.is_absolute() else (self.base_dir / expanded).resolve()
        )

    def s1_parent_bank(self) -> StaticBankArtifact:
        """Load and prove the accepted forward parent without rewriting Static."""

        binding = self.spec.s1_parent
        if binding is None:
            return self.static_bank()
        if self._s1_parent_bank is not None:
            return self._s1_parent_bank
        bank_path = self._bound_path(binding.bank_path)
        if _file_sha(bank_path) != binding.bank_file_sha256:
            raise FastPathError("S1 parent Bank file SHA-256 drifted")
        try:
            bank = StaticBankArtifact.model_validate_json(
                bank_path.read_bytes(), strict=True
            )
        except (OSError, ValidationError) as error:
            raise FastPathError("S1 parent Bank is invalid") from error
        self._require_six_capabilities(bank)
        if bank.bank_sha256 != binding.bank_sha256:
            raise FastPathError("S1 parent internal Bank SHA-256 drifted")

        decision_path = self._bound_path(binding.decision_path)
        if _file_sha(decision_path) != binding.decision_file_sha256:
            raise FastPathError("S1 parent decision file SHA-256 drifted")
        try:
            decision = StageDecision.model_validate_json(
                decision_path.read_bytes(), strict=True
            )
        except (OSError, ValidationError) as error:
            raise FastPathError("S1 parent decision is invalid") from error
        if (
            decision.stage != "s1"
            or not decision.accepted
            or decision.alias_of is not None
            or decision.selected_bank != binding.bank_sha256
            or decision.metrics.get("round_id") != binding.source_round_id
        ):
            raise FastPathError(
                "S1 parent must be the accepted selected Bank of its bound round"
            )

        manifest_path = self._bound_path(binding.manifest_path)
        if _file_sha(manifest_path) != binding.manifest_file_sha256:
            raise FastPathError("S1 parent manifest file SHA-256 drifted")
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise FastPathError("S1 parent manifest is invalid")
        source_settings = manifest.get("s1_settings")
        if (
            not isinstance(source_settings, dict)
            or source_settings.get("round_id") != binding.source_round_id
        ):
            raise FastPathError("S1 parent manifest round identity drifted")

        skills = _bank_by_capability(bank)
        for capability, expected_sha in binding.protected_skill_sha256.items():
            skill = skills.get(capability)
            if skill is None or getattr(skill, "skill_sha256", None) != expected_sha:
                raise FastPathError(
                    f"S1 parent protected Skill SHA-256 drifted: {capability}"
                )
        self._s1_parent_bank = bank
        return bank

    def s1_parent_opt(self) -> dict[str, AssistantObservation]:
        """Load opt800 observations produced fresh by the accepted parent."""

        binding = self.spec.s1_parent
        if binding is None:
            return self.opt_static()
        if self._s1_parent_opt is not None:
            return self._s1_parent_opt
        if binding.opt_results_sha256 is None:
            raise FastPathError("S1 parent opt800 SHA-256 is not frozen")
        path = self._bound_path(binding.opt_results_path)
        if _file_sha(path) != binding.opt_results_sha256:
            raise FastPathError("S1 parent opt800 observations SHA-256 drifted")
        observations: dict[str, AssistantObservation] = {}
        for index, raw in enumerate(_read_jsonl(path), start=1):
            if not isinstance(raw, dict):
                raise FastPathError(f"invalid S1 parent opt800 row {index}")
            payload = raw.get("observation", raw)
            try:
                item = AssistantObservation.model_validate(payload, strict=True)
            except ValidationError as error:
                raise FastPathError(f"invalid S1 parent opt800 row {index}") from error
            if item.query_id in observations:
                raise FastPathError(
                    f"duplicate S1 parent opt800 query: {item.query_id}"
                )
            observations[item.query_id] = item
        expected = {
            item.query_id for item in self.queries() if item.split == "opt_pool"
        }
        if set(observations) != expected:
            raise FastPathError("S1 parent opt800 must cover the fixed opt800 exactly")
        self._validate_opt_static_model_identity(observations)
        parent_sha = self.s1_parent_bank().bank_sha256
        for query_id, observation in observations.items():
            result = observation.replay_context.get("assistant_result")
            if not isinstance(result, dict) or result.get("bank_sha256") != parent_sha:
                raise FastPathError(
                    f"S1 parent opt800 Bank identity differs: {query_id}"
                )
            if observation.assistant_contract != self.spec.runtime.assistant_contract:
                raise FastPathError(
                    f"S1 parent opt800 runtime contract differs: {query_id}"
                )
        self._s1_parent_opt = observations
        return observations

    def _s1_parent_alias(self) -> Literal["llm_static", "s1"]:
        return "s1" if self.spec.s1_parent is not None else "llm_static"

    def s1_authoring_input(self) -> AuthoringInput:
        if self._s1_authoring_input is not None:
            return self._s1_authoring_input
        path = self._path("s1_authoring_input")
        if _file_sha(path) != self.spec.s1_authoring_input_file_sha256:
            raise FastPathError(
                "S1 AuthoringInput SHA-256 differs from the frozen spec"
            )
        try:
            value = AuthoringInput.model_validate_json(path.read_bytes(), strict=True)
            decode_sparse_parent_content(self.static_bank(), value)
        except (OSError, ValidationError, S1SparsePatchError) as error:
            raise FastPathError("invalid S1 sparse AuthoringInput") from error
        self._s1_authoring_input = value
        return value

    def opt_static(self) -> dict[str, AssistantObservation]:
        if self._opt_static is not None:
            return self._opt_static
        path = self._path("opt_static_results")
        if _file_sha(path) != self.spec.opt_static_results_sha256:
            raise FastPathError(
                "Static opt800 observations SHA-256 differs from the frozen spec"
            )
        observations: dict[str, AssistantObservation] = {}
        for index, raw in enumerate(_read_jsonl(path), start=1):
            if not isinstance(raw, dict):
                raise FastPathError(f"invalid opt Static row {index}")
            payload = raw.get("observation", raw)
            try:
                item = AssistantObservation.model_validate(payload, strict=True)
            except ValidationError as error:
                raise FastPathError(f"invalid opt Static row {index}") from error
            if item.query_id in observations:
                raise FastPathError(f"duplicate opt Static query: {item.query_id}")
            observations[item.query_id] = item
        expected = {
            item.query_id for item in self.queries() if item.split == "opt_pool"
        }
        if set(observations) != expected:
            raise FastPathError(
                "opt Static results must cover the fixed opt800 exactly"
            )
        self._validate_opt_static_model_identity(observations)
        self._opt_static = observations
        return observations

    def _validate_opt_static_model_identity(
        self, observations: Mapping[str, AssistantObservation]
    ) -> None:
        expected_model = self.spec.models["assistant"].requested_model
        provider_request_ids: list[str] = []
        for query_id, observation in observations.items():
            context = observation.replay_context
            response = context.get("response")
            receipt = context.get("receipt")
            assistant_result = context.get("assistant_result")
            if not all(
                isinstance(item, dict) for item in (response, receipt, assistant_result)
            ):
                raise FastPathError(
                    f"Static opt800 model identity evidence is missing: {query_id}"
                )
            assert isinstance(response, dict)
            assert isinstance(receipt, dict)
            assert isinstance(assistant_result, dict)
            if (
                response.get("backbone_model") != expected_model
                or assistant_result.get("backbone_model") != expected_model
            ):
                raise FastPathError(
                    f"Static opt800 Assistant model differs from spec: {query_id}"
                )
            model_calls = receipt.get("model_calls")
            if not isinstance(model_calls, list) or not model_calls:
                raise FastPathError(
                    f"Static opt800 model-call receipt is missing: {query_id}"
                )
            for call in model_calls:
                if (
                    not isinstance(call, dict)
                    or call.get("requested_model") != expected_model
                    or call.get("response_model") != expected_model
                    or not isinstance(call.get("provider_request_id"), str)
                    or not call["provider_request_id"]
                ):
                    raise FastPathError(
                        f"Static opt800 model-call identity differs: {query_id}"
                    )
                provider_request_ids.append(str(call["provider_request_id"]))
        if len(provider_request_ids) != len(set(provider_request_ids)):
            raise FastPathError("Static opt800 provider request IDs are not unique")

    def fixed_samples_from_static(
        self, observations: Mapping[str, AssistantObservation]
    ) -> FixedSamples:
        """Select new-model S1 samples deterministically from discovery600."""

        roles = self.opt_fold_roles()
        selected_canary: list[FixedSample] = []
        selected_body: list[FixedSample] = []
        for capability in CAPABILITIES:
            population = [
                query
                for query in self.queries()
                if query.split == "opt_pool"
                and query.canonical_capability == capability
                and roles[query.query_id] == "discovery"
            ]
            failures = [
                query
                for query in population
                if observations[query.query_id].hard_error
                or observations[query.query_id].gcs_score < 1.0
            ]
            anchors = [
                query
                for query in population
                if not observations[query.query_id].hard_error
                and observations[query.query_id].gcs_score == 1.0
            ]
            canary_queries = []
            if failures:
                canary_queries.append(failures[0])
            if anchors:
                canary_queries.append(anchors[0])
            for query in (*failures, *anchors):
                if query not in canary_queries:
                    canary_queries.append(query)
                if len(canary_queries) == 2:
                    break
            if len(canary_queries) != 2:
                raise FastPathError(
                    f"new Static baseline cannot supply canary2 for {capability}"
                )
            selected_canary.extend(
                FixedSample(
                    query_id=query.query_id,
                    capability=capability,
                    role=(
                        "anchor"
                        if observations[query.query_id].gcs_score == 1.0
                        and not observations[query.query_id].hard_error
                        else "failure"
                    ),
                )
                for query in canary_queries
            )

            route_correct = [
                query
                for query in population
                if observations[query.query_id].gcs_components["route_acceptable"]
            ]
            body_failures = [
                query
                for query in route_correct
                if observations[query.query_id].hard_error
                or observations[query.query_id].gcs_score < 1.0
            ]
            body_anchors = [
                query
                for query in route_correct
                if not observations[query.query_id].hard_error
                and observations[query.query_id].gcs_score == 1.0
            ]
            body_queries = [*body_failures[:6], *body_anchors[:2]]
            for query in (*body_failures[6:], *body_anchors[2:]):
                if len(body_queries) == 8:
                    break
                if query not in body_queries:
                    body_queries.append(query)
            if len(body_queries) != 8:
                raise FastPathError(
                    f"new Static baseline cannot supply body8 for {capability}"
                )
            selected_body.extend(
                FixedSample(
                    query_id=query.query_id,
                    capability=capability,
                    role=(
                        "body_anchor"
                        if observations[query.query_id].gcs_score == 1.0
                        and not observations[query.query_id].hard_error
                        else "body_failure"
                    ),
                )
                for query in body_queries
            )
        return FixedSamples(
            canary12=tuple(selected_canary),
            dev_smoke24=self.spec.fixed_samples.dev_smoke24,
            body48=tuple(selected_body),
        )

    def opt_attribution(self) -> dict[str, dict[str, object]]:
        if self._opt_attribution is not None:
            return self._opt_attribution
        rows: dict[str, dict[str, object]] = {}
        for index, raw in enumerate(
            _read_jsonl(self._path("opt_route_attribution")), start=1
        ):
            if not isinstance(raw, dict) or not isinstance(raw.get("query_id"), str):
                raise FastPathError(f"invalid route attribution row {index}")
            query_id = raw["query_id"]
            if query_id in rows:
                raise FastPathError(f"duplicate route attribution: {query_id}")
            rows[query_id] = raw
        expected = {
            item.query_id for item in self.queries() if item.split == "opt_pool"
        }
        if set(rows) != expected:
            raise FastPathError("route attribution must cover fixed opt800 exactly")
        self._opt_attribution = rows
        return rows

    def opt_fold_roles(self) -> dict[str, str]:
        if self._opt_fold_roles is not None:
            return self._opt_fold_roles
        path = self._path("opt_fold_mapping")
        if _file_sha(path) != self.spec.opt_fold_mapping_sha256:
            raise FastPathError("opt fold mapping SHA-256 differs from the frozen spec")
        roles: dict[str, str] = {}
        for index, raw in enumerate(_read_jsonl(path), start=1):
            if not isinstance(raw, dict):
                raise FastPathError(f"invalid opt fold row {index}")
            query_id = raw.get("query_id")
            role = raw.get("role")
            if (
                not isinstance(query_id, str)
                or role not in {"discovery", "replay"}
                or query_id in roles
            ):
                raise FastPathError(f"invalid opt fold binding at row {index}")
            roles[query_id] = str(role)
        expected = {
            item.query_id for item in self.queries() if item.split == "opt_pool"
        }
        counts = Counter(roles.values())
        if set(roles) != expected or counts != Counter(
            {"discovery": 600, "replay": 200}
        ):
            raise FastPathError("opt fold mapping must bind discovery600/replay200")
        self._opt_fold_roles = roles
        return roles

    def val_gate_roles(self) -> dict[str, str]:
        if self._val_gate_roles is not None:
            return self._val_gate_roles
        path = self._path("val_gate_assignments")
        if _file_sha(path) != self.spec.val_gate_assignments_sha256:
            raise FastPathError(
                "validation gate assignment SHA-256 differs from the frozen spec"
            )
        raw = load_json(path)
        if not isinstance(raw, dict) or not isinstance(raw.get("audit"), dict):
            raise FastPathError("invalid validation gate assignment artifact")
        mapping = raw["audit"].get("query_id_to_gate")
        if not isinstance(mapping, dict):
            raise FastPathError("validation gate artifact lacks query_id_to_gate")
        roles = {
            str(query_id): str(gate)
            for query_id, gate in mapping.items()
            if isinstance(query_id, str) and isinstance(gate, str)
        }
        expected = {item.query_id for item in self.queries() if item.split == "val"}
        counts = Counter(roles.values())
        if set(roles) != expected or counts != Counter(
            {"body_gate": 75, "route_gate": 75, "shadow_val": 50}
        ):
            raise FastPathError(
                "validation gates must bind body75/route75/shadow50 exactly"
            )
        self._val_gate_roles = roles
        return roles

    def _queries_for_val_gate(self, gate: str) -> list[Query]:
        roles = self.val_gate_roles()
        return [
            query
            for query in self.queries()
            if query.split == "val" and roles[query.query_id] == gate
        ]

    def validate(self, *, require_runtime: bool) -> ValidationSummary:
        queries = self.queries()
        counts = Counter(item.split for item in queries)
        if dict(counts) != SPLIT_COUNTS:
            raise FastPathError(f"Core split counts differ: {dict(counts)}")
        if len(queries) != 1500:
            raise FastPathError("Core dataset must contain exactly 1,500 rows")
        if {item.canonical_capability for item in queries} != set(CAPABILITIES):
            raise FastPathError("Core dataset must cover the fixed six capabilities")
        bank = self.static_bank()
        self.s1_authoring_input()
        opt_static = self.opt_static()
        s1_parent = self.s1_parent_bank()
        s1_parent_opt = self.s1_parent_opt()
        self._require_counterfactual_cycle_preflight()
        self.opt_attribution()
        fold_roles = self.opt_fold_roles()
        self.val_gate_roles()
        by_id = self.query_by_id()
        for query_id, observation in opt_static.items():
            result = observation.replay_context.get("assistant_result")
            if (
                not isinstance(result, dict)
                or result.get("bank_sha256") != bank.bank_sha256
            ):
                raise FastPathError(
                    "Static opt800 observation Bank differs from the frozen parent: "
                    f"{query_id}"
                )
            if observation.assistant_contract != self.spec.runtime.assistant_contract:
                raise FastPathError(
                    "Static opt800 observation Assistant contract differs from the "
                    f"active runtime: {query_id}"
                )
        sample_observations = (
            s1_parent_opt if self.spec.s1_parent is not None else opt_static
        )
        sample_sets = (
            (self.spec.fixed_samples.canary12, "opt_pool"),
            (self.spec.fixed_samples.dev_smoke24, "dev_mini"),
            (self.spec.fixed_samples.body48, "opt_pool"),
        )
        for samples, split in sample_sets:
            for sample in samples:
                query = by_id.get(sample.query_id)
                if query is None:
                    raise FastPathError(
                        f"fixed query does not exist: {sample.query_id}"
                    )
                if (
                    query.split != split
                    or query.canonical_capability != sample.capability
                ):
                    raise FastPathError(
                        f"fixed query binding differs: {sample.query_id}"
                    )
        for sample in self.spec.fixed_samples.canary12:
            if fold_roles[sample.query_id] != "discovery":
                raise FastPathError(
                    f"Feedback canary is outside discovery600: {sample.query_id}"
                )
            observation = sample_observations[sample.query_id]
            if sample.role == "failure" and not (
                observation.hard_error or observation.gcs_score < 1.0
            ):
                raise FastPathError(
                    f"canary failure is not a recorded failure: {sample.query_id}"
                )
            if sample.role == "anchor" and (
                observation.hard_error or observation.gcs_score < 1.0
            ):
                raise FastPathError(
                    f"canary anchor is not a recorded success: {sample.query_id}"
                )
        for sample in self.spec.fixed_samples.body48:
            observation = sample_observations[sample.query_id]
            if not observation.gcs_components["route_acceptable"]:
                raise FastPathError(f"body48 must be route-correct: {sample.query_id}")
            if sample.role == "body_failure" and not (
                observation.hard_error
                or observation.card_violation
                or observation.evidence_violation
                or observation.tool_violation
                or not observation.gcs_components["output_contract_pass"]
            ):
                raise FastPathError(
                    f"body48 failure lacks a Body/contract failure: {sample.query_id}"
                )
            if sample.role == "body_anchor" and (
                observation.hard_error or observation.gcs_score < 1.0
            ):
                raise FastPathError(
                    f"body48 anchor is not a recorded success: {sample.query_id}"
                )
        if self.spec.s1_parent is not None:
            binding = self.spec.s1_parent
            for query_id in binding.parent_protection_query_ids:
                query = by_id.get(query_id)
                if query is None or query.canonical_capability != (
                    "product.style_recommendation"
                ):
                    raise FastPathError(
                        f"S1 parent protection binding differs: {query_id}"
                    )
            if s1_parent.bank_sha256 == bank.bank_sha256:
                raise FastPathError("accepted S1 parent unexpectedly aliases Static")
        runtime_ready = True
        # Deliberately coarse call-count ceiling, not a per-call reservation
        # proof.  It includes every physical candidate, all 1,500 post-freeze
        # route-only gaps, and one format retry for every Judge call.
        max_calls = {
            "feedback": self.spec.s1_settings.feedback_total_count,
            "creator": self.spec.limits.max_creator_calls,
            "assistant": 624 + 224 + 296 + 200 + 1500,
            "judge": 2 * (48 + 1500),
            "route_only": 4500,
        }
        projected_cost = sum(
            max_calls[role] * self.spec.models[role].estimated_call_cost_cny
            for role in max_calls
        )
        if projected_cost > self.spec.limits.external_cost_cny:
            raise FastPathError(
                "configured worst-case estimate exceeds the CNY 250 hard cap: "
                f"{projected_cost:.6f}"
            )
        if require_runtime:
            for role, model in self.spec.models.items():
                if model.credential_env and not os.environ.get(model.credential_env):
                    raise FastPathError(
                        f"missing credential {model.credential_env} for {role}"
                    )
            if self.spec.runtime.adapter == "command":
                for role in ("feedback", "creator", "assistant", "judge", "route_only"):
                    command = self.spec.runtime.commands.for_role(role)
                    if not command:
                        raise FastPathError(f"missing runtime command for {role}")
                    executable = shutil.which(command[0])
                    if executable is None and not Path(command[0]).is_file():
                        raise FastPathError(
                            f"command executable not found: {command[0]}"
                        )
                    if any(
                        Path(item).name == "core_fast_call.py" for item in command
                    ) and not os.environ.get("CORE_FAST_CALL_FACTORY"):
                        raise FastPathError(
                            "CORE_FAST_CALL_FACTORY is required by core_fast_call.py"
                        )
            runtime_validator = getattr(self.adapter, "validate_runtime", None)
            if runtime_validator is not None:
                try:
                    runtime_validator()
                except (OSError, RuntimeError, ValueError) as error:
                    raise FastPathError(
                        f"runtime adapter is not ready: {error}"
                    ) from error
        else:
            runtime_ready = False
        return ValidationSummary(
            query_count=len(queries),
            split_counts=dict(counts),
            capability_count=len(CAPABILITIES),
            static_bank_sha256=bank.bank_sha256,
            observed_cost_cny=self.calls.observed_cost(),
            projected_worst_case_cost_cny=projected_cost,
            runtime_ready=runtime_ready,
            feedback_count=self.spec.s1_settings.feedback_total_count,
        )

    def initialize(self) -> None:
        summary = self.validate(require_runtime=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_root / "manifest.json"
        payload = {
            "schema_version": 1,
            "kind": "core-experiment-fast-run",
            "experiment_id": self.spec.experiment_id,
            "spec_path": str(self.spec_path),
            "input_sha256": {
                "spec": _file_sha(self.spec_path),
                "asset_catalog": self.spec.asset_catalog_sha256,
                "static_bank_file": self.spec.static_bank_file_sha256,
                "queries": _file_sha(self._path("queries")),
                "static_bank": _file_sha(self._path("static_bank")),
                "opt_static_results": _file_sha(self._path("opt_static_results")),
                "opt_route_attribution": _file_sha(self._path("opt_route_attribution")),
                "opt_fold_mapping": _file_sha(self._path("opt_fold_mapping")),
                "val_gate_assignments": _file_sha(self._path("val_gate_assignments")),
                "s1_authoring_input": _file_sha(self._path("s1_authoring_input")),
                "s1_parent_bank_file": (
                    None
                    if self.spec.s1_parent is None
                    else self.spec.s1_parent.bank_file_sha256
                ),
                "s1_parent_decision_file": (
                    None
                    if self.spec.s1_parent is None
                    else self.spec.s1_parent.decision_file_sha256
                ),
                "s1_parent_manifest_file": (
                    None
                    if self.spec.s1_parent is None
                    else self.spec.s1_parent.manifest_file_sha256
                ),
                "s1_parent_opt_results": (
                    None
                    if self.spec.s1_parent is None
                    else self.spec.s1_parent.opt_results_sha256
                ),
                "feedback_schema": sha256_bytes(
                    canonical_json_bytes(VisualFeedbackOutput.model_json_schema())
                ),
                "assistant_result_schema": sha256_bytes(
                    canonical_json_bytes(AssistantObservation.model_json_schema())
                ),
                "assistant_execution_contract": sha256_bytes(
                    self.spec.runtime.assistant_contract.encode("utf-8")
                ),
                "judge_result_schema": sha256_bytes(
                    canonical_json_bytes(JudgeObservation.model_json_schema())
                ),
            },
            "split_counts": self.spec.split_counts,
            "capabilities": list(self.spec.capabilities),
            "configs": list(self.spec.configs),
            "fixed_samples": self.spec.fixed_samples.model_dump(mode="json"),
            "models": {
                key: value.model_dump(mode="json")
                for key, value in self.spec.models.items()
            },
            "gates": self.spec.gates.model_dump(mode="json"),
            "s1_settings": self.spec.s1_settings.model_dump(mode="json"),
            "s1_parent": (
                None
                if self.spec.s1_parent is None
                else self.spec.s1_parent.model_dump(mode="json")
            ),
            "concurrency": self.spec.concurrency.model_dump(mode="json"),
            "limits": self.spec.limits.model_dump(mode="json"),
            "runtime": self.spec.runtime.model_dump(mode="json"),
            "bootstrap": {
                "replicates": self.spec.bootstrap_replicates,
                "seed": self.spec.bootstrap_seed,
            },
            "summary_at_start": summary.as_dict(),
            "disclosures": list(self.spec.disclosures),
        }
        if manifest_path.exists():
            existing = load_json(manifest_path)
            keys = (
                "experiment_id",
                "input_sha256",
                "fixed_samples",
                "models",
                "gates",
                "s1_settings",
                "s1_parent",
                "concurrency",
                "limits",
                "runtime",
                "bootstrap",
            )
            if not isinstance(existing, dict) or any(
                existing.get(key) != payload[key] for key in keys
            ):
                raise FastPathError("output root belongs to a different Fast Path run")
        else:
            atomic_write_json(manifest_path, payload)
        bank_path = self.output_root / "banks" / "llm_static.json"
        if not bank_path.exists():
            atomic_write_json(bank_path, self.static_bank().model_dump(mode="json"))
        if self.spec.s1_parent is not None:
            parent_path = self.output_root / "banks" / "s1-parent.json"
            parent_payload = self.s1_parent_bank().model_dump(mode="json")
            if parent_path.exists():
                if load_json(parent_path) != parent_payload:
                    raise FastPathError("S1 parent Bank differs on resume")
            else:
                atomic_write_json(parent_path, parent_payload)

    def initialize_static_opt800(self) -> None:
        """Initialize only inputs needed to produce a new Static baseline."""

        destination = self._path("opt_static_results")
        manifest_path = self.output_root / "static-opt800-manifest.json"
        if destination.exists() and not manifest_path.exists():
            raise FastPathError(
                "Static opt800 destination predates this run manifest; use a new lineage"
            )
        queries = self.queries()
        if Counter(item.split for item in queries) != Counter(SPLIT_COUNTS):
            raise FastPathError("Core split geometry differs before Static opt800")
        bank = self.static_bank()
        self.opt_fold_roles()
        if (
            self.spec.models["assistant"].requested_model
            != self.spec.models["route_only"].requested_model
        ):
            raise FastPathError(
                "Static Assistant and route-only model identities differ"
            )
        if self.spec.runtime.adapter != "python":
            raise FastPathError(
                "fresh Static opt800 requires the reviewed Python adapter"
            )
        for role in ("assistant",):
            model = self.spec.models[role]
            if model.credential_env and not os.environ.get(model.credential_env):
                raise FastPathError(
                    f"missing credential {model.credential_env} for {role}"
                )
        runtime_validator = getattr(self.adapter, "validate_runtime", None)
        if runtime_validator is not None:
            try:
                runtime_validator()
            except (OSError, RuntimeError, ValueError) as error:
                raise FastPathError(f"runtime adapter is not ready: {error}") from error
        self.output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "kind": "core-fast-static-opt800-run",
            "experiment_id": self.spec.experiment_id,
            "assistant_model": self.spec.models["assistant"].model_dump(mode="json"),
            "assistant_contract": self.spec.runtime.assistant_contract,
            "concurrency": self.spec.concurrency.model_dump(mode="json"),
            "queries_sha256": _file_sha(self._path("queries")),
            "static_bank_file_sha256": _file_sha(self._path("static_bank")),
            "static_bank_sha256": bank.bank_sha256,
            "destination": str(destination),
        }
        if manifest_path.exists():
            if load_json(manifest_path) != payload:
                raise FastPathError("Static opt800 output root belongs to another run")
        else:
            atomic_write_json(manifest_path, payload)

    def initialize_s1_parent_opt800(self) -> None:
        """Initialize a create-only opt800 run for the accepted S1 parent."""

        binding = self.spec.s1_parent
        if binding is None:
            raise FastPathError("s1-parent-opt800 requires an S1 parent binding")
        destination = self._bound_path(binding.opt_results_path)
        manifest_path = self.output_root / "s1-parent-opt800-manifest.json"
        if destination.exists() and not manifest_path.exists():
            raise FastPathError(
                "S1 parent opt800 destination predates this run manifest; use a new lineage"
            )
        queries = self.queries()
        if Counter(item.split for item in queries) != Counter(SPLIT_COUNTS):
            raise FastPathError("Core split geometry differs before S1 parent opt800")
        parent = self.s1_parent_bank()
        self.opt_fold_roles()
        if self.spec.runtime.adapter != "python":
            raise FastPathError(
                "fresh S1 parent opt800 requires the reviewed Python adapter"
            )
        model = self.spec.models["assistant"]
        if model.credential_env and not os.environ.get(model.credential_env):
            raise FastPathError(
                f"missing credential {model.credential_env} for assistant"
            )
        runtime_validator = getattr(self.adapter, "validate_runtime", None)
        if runtime_validator is not None:
            try:
                runtime_validator()
            except (OSError, RuntimeError, ValueError) as error:
                raise FastPathError(f"runtime adapter is not ready: {error}") from error
        self.output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "kind": "core-fast-s1-parent-opt800-run",
            "experiment_id": self.spec.experiment_id,
            "source_round_id": binding.source_round_id,
            "assistant_model": model.model_dump(mode="json"),
            "assistant_contract": self.spec.runtime.assistant_contract,
            "concurrency": self.spec.concurrency.model_dump(mode="json"),
            "queries_sha256": _file_sha(self._path("queries")),
            "parent_bank_file_sha256": binding.bank_file_sha256,
            "parent_bank_sha256": parent.bank_sha256,
            "source_decision_file_sha256": binding.decision_file_sha256,
            "source_manifest_file_sha256": binding.manifest_file_sha256,
            "destination": str(destination),
        }
        if manifest_path.exists():
            if load_json(manifest_path) != payload:
                raise FastPathError(
                    "S1 parent opt800 output root belongs to another run"
                )
        else:
            atomic_write_json(manifest_path, payload)

    # ------------------------------- calls ---------------------------------

    def _call(
        self,
        *,
        role: CallRole,
        call_id: str,
        purpose: str,
        payload: dict[str, object],
    ) -> CallResult:
        existing = self.calls.get(role, call_id)
        if role == "feedback" and existing is None:
            if self.calls.role_count(role) >= self.spec.limits.max_feedback_calls:
                raise FastPathError("Feedback call limit reached")
            self._feedback_start_pacer.wait()
        if role == "creator" and existing is None:
            if self.calls.role_count(role) >= self.spec.limits.max_creator_calls:
                raise FastPathError("Creator/optimizer call limit reached")
        intent = CallIntent(
            call_id=call_id,
            role=role,
            purpose=purpose,
            requested_model=self.spec.models[role].requested_model,
            payload=payload,
        )
        try:
            return self.calls.invoke(intent, self.adapter.invoke)
        except FastStoreError as error:
            raise FastPathError(str(error)) from error

    def _assistant_call_id(self, split: str, config: str, query_id: str) -> str:
        return _safe_id(f"{split}-{config}-{query_id}")

    def _assistant_one(
        self,
        *,
        split: str,
        config: str,
        query: Query,
        bank: StaticBankArtifact | None,
        reuse_parent: AssistantObservation | None = None,
        replay_surface: Literal["action-policy", "response-policy"] | None = None,
    ) -> AssistantObservation:
        payload: dict[str, object] = {
            "operation": "assistant",
            "split": split,
            "config": config,
            "query": query.model_dump(mode="json"),
            "bank": None if bank is None else bank.model_dump(mode="json"),
            "runtime_paths": self.spec.paths.model_dump(mode="json"),
            "score_with": "grounded-contract-success-v2",
        }
        if reuse_parent is not None:
            if replay_surface is None:
                replay_surface = "response-policy"
            replay_key = (
                "reuse_parent_route_only"
                if replay_surface == "action-policy"
                else "reuse_parent_route_and_tool"
            )
            payload[replay_key] = {
                "selected_capability": reuse_parent.selected_capability,
                "route_trace_key": reuse_parent.route_trace_key,
                "tool_trace_key": reuse_parent.tool_trace_key,
                "tool_trace": [
                    item.model_dump(mode="json") for item in reuse_parent.tool_trace
                ],
                "replay_context": reuse_parent.replay_context,
            }
        result = self._call(
            role="assistant",
            call_id=self._assistant_call_id(split, config, query.query_id),
            purpose=f"{split} {config} Assistant+tools+GCS",
            payload=payload,
        )
        observation = _observation_from_result(result, query.query_id)
        return observation

    @staticmethod
    def _common_trace_violations(
        parent: Mapping[str, AssistantObservation],
        candidate: Mapping[str, AssistantObservation],
    ) -> tuple[str, ...]:
        return tuple(
            query_id
            for query_id in parent
            if (
                candidate[query_id].selected_capability
                != parent[query_id].selected_capability
                or candidate[query_id].route_trace_key
                != parent[query_id].route_trace_key
                or candidate[query_id].tool_trace_key != parent[query_id].tool_trace_key
                or candidate[query_id].tool_trace != parent[query_id].tool_trace
            )
        )

    @staticmethod
    def _body_replay_parent_eligible(observation: AssistantObservation) -> bool:
        """Return whether one frozen observation can support answer-only replay."""

        context = observation.replay_context
        response = context.get("response")
        receipt = context.get("receipt")
        if not isinstance(response, dict) or not isinstance(receipt, dict):
            return False
        route_attempt = receipt.get("route_attempt")
        return bool(
            isinstance(response.get("selected_capability"), str)
            and response.get("selected_capability")
            and isinstance(response.get("skill_slug"), str)
            and response.get("skill_slug")
            and isinstance(response.get("route_trace_sha256"), str)
            and response.get("route_trace_sha256")
            and isinstance(route_attempt, dict)
            and route_attempt.get("status") == "selected"
        )

    @staticmethod
    def _action_replay_parent_eligible(observation: AssistantObservation) -> bool:
        """Return whether an observation carries a reusable selected route."""

        context = observation.replay_context
        response = context.get("response")
        receipt = context.get("receipt")
        if not isinstance(response, dict) or not isinstance(receipt, dict):
            return False
        route_attempt = receipt.get("route_attempt")
        return bool(
            isinstance(response.get("selected_capability"), str)
            and response.get("selected_capability")
            and isinstance(response.get("skill_slug"), str)
            and response.get("skill_slug")
            and isinstance(response.get("route_trace_sha256"), str)
            and response.get("route_trace_sha256")
            and isinstance(route_attempt, dict)
            and route_attempt.get("status") == "selected"
        )

    def _assistant_many(
        self,
        *,
        split: str,
        config: str,
        queries: Sequence[Query],
        bank: StaticBankArtifact | None,
        reuse_parent: Mapping[str, AssistantObservation] | None = None,
        replay_surface: Literal["action-policy", "response-policy"] | None = None,
    ) -> dict[str, AssistantObservation]:
        if split == "test_frozen" and any(
            self._existing_decision(stage) is None for stage in ("s1", "s2", "full")
        ):
            raise FastPathError(
                "test300 responses are sealed until all three Bank decisions are frozen"
            )
        pending = [
            query
            for query in queries
            if self.calls.get(
                "assistant", self._assistant_call_id(split, config, query.query_id)
            )
            is None
        ]
        if pending:
            try:
                self.calls.ensure_budget("assistant", len(pending))
            except FastStoreError as error:
                raise FastPathError(str(error)) from error
        results: dict[str, AssistantObservation] = {}
        with ThreadPoolExecutor(max_workers=self.spec.concurrency.assistant) as pool:
            futures = {
                pool.submit(
                    self._assistant_one,
                    split=split,
                    config=config,
                    query=query,
                    bank=bank,
                    reuse_parent=(
                        None if reuse_parent is None else reuse_parent[query.query_id]
                    ),
                    replay_surface=replay_surface,
                ): query.query_id
                for query in queries
            }
            for future in as_completed(futures):
                query_id = futures[future]
                try:
                    results[query_id] = future.result()
                except FastPathError:
                    raise
                except Exception as error:
                    raise FastPathError(
                        f"Assistant execution failed: {query_id}"
                    ) from error
        return {query.query_id: results[query.query_id] for query in queries}

    # ------------------------------- banks ---------------------------------

    @staticmethod
    def _require_six_capabilities(bank: StaticBankArtifact) -> None:
        if {skill.capability_id for skill in bank.skills} != set(CAPABILITIES):
            raise FastPathError("candidate Bank does not cover all six capabilities")

    @staticmethod
    def _rehash(payload: dict[str, object], field: str) -> str:
        unsigned = dict(payload)
        unsigned.pop(field, None)
        return sha256_bytes(canonical_json_bytes(unsigned))

    def _compile_candidate_payload(
        self,
        payload: dict[str, object],
        *,
        stage: str,
        parent: StaticBankArtifact,
    ) -> StaticBankArtifact | None:
        """Compile the small model-facing edit shape into the strict Bank schema."""

        parent_skills = {
            skill.capability_id: skill.model_dump(mode="json")
            for skill in parent.skills
        }
        updates: dict[str, dict[str, object]] = {}
        if stage == "s1" and isinstance(payload.get("skills"), list):
            for raw in payload["skills"]:
                if not isinstance(raw, dict) or not isinstance(
                    raw.get("capability_id"), str
                ):
                    return None
                capability = raw["capability_id"]
                if capability in updates:
                    return None
                allowed = {
                    "capability_id",
                    "description",
                    "body",
                    "static_refs",
                    "operators",
                }
                if set(raw) - allowed:
                    return None
                updates[capability] = dict(raw)
            if set(updates) != set(CAPABILITIES):
                return None
        elif stage in {"s2", "full"} and isinstance(payload.get("edits"), list):
            field = "description" if stage == "s2" else "body"
            for raw in payload["edits"]:
                if not isinstance(raw, dict) or set(raw) != {"capability_id", field}:
                    return None
                capability = raw.get("capability_id")
                if not isinstance(capability, str) or capability in updates:
                    return None
                updates[capability] = dict(raw)
        else:
            return None
        if not updates or set(updates) - set(CAPABILITIES):
            return None

        compiled: list[dict[str, object]] = []
        for capability in CAPABILITIES:
            source = dict(parent_skills[capability])
            change = updates.get(capability)
            if change is not None:
                for key, value in change.items():
                    if key != "capability_id":
                        source[key] = value
                source["version"] = int(source["version"]) + 1
                source["parent_skill_sha256"] = parent_skills[capability][
                    "skill_sha256"
                ]
                source["skill_sha256"] = self._rehash(source, "skill_sha256")
            compiled.append(source)
        bank_payload = parent.model_dump(mode="json")
        bank_payload["skills"] = sorted(compiled, key=lambda item: str(item["slug"]))
        bank_payload["bank_sha256"] = self._rehash(bank_payload, "bank_sha256")
        try:
            bank = StaticBankArtifact.model_validate_json(
                canonical_json_bytes(bank_payload), strict=True
            )
            self._require_six_capabilities(bank)
            # Rendering is the canonical Markdown compiler check and validates
            # every operator/static-reference shape through StrictSkillArtifact.
            for skill in bank.skills:
                render_skill_markdown(skill)
            return bank
        except (ValidationError, ValueError):
            return None

    def _candidate_bank(
        self,
        result: CallResult,
        *,
        stage: str,
        parent: StaticBankArtifact,
        feedback_bundle_sha256: str | None = None,
        feedback_patchable_capabilities: frozenset[str] = frozenset(),
        s1_expected_patch_capabilities: frozenset[str] | None = None,
        s1_artifact_prefix: str = "s1",
    ) -> StaticBankArtifact | None:
        if (
            result.status != "success"
            or not result.schema_valid
            or result.output is None
        ):
            return None
        model_payload = result.output
        full_payload = result.output.get("bank")
        if stage == "s1":
            if full_payload is not None or feedback_bundle_sha256 is None:
                raise _S1CandidateRejected("invalid_sparse_output_envelope")
            bank = self._compile_sparse_s1_payload(
                model_payload,
                parent=parent,
                feedback_bundle_sha256=feedback_bundle_sha256,
                feedback_patchable_capabilities=feedback_patchable_capabilities,
                expected_patch_capabilities=s1_expected_patch_capabilities,
                artifact_prefix=s1_artifact_prefix,
            )
            if bank is None:
                raise _S1CandidateRejected("invalid_sparse_compilation")
            path = self.output_root / "banks" / f"{s1_artifact_prefix}-candidate.json"
            self._write_canonical_resume_artifact(
                path,
                bank.model_dump(mode="json"),
                label=f"{s1_artifact_prefix} candidate Bank",
            )
            return bank
        if isinstance(full_payload, dict):
            try:
                supplied = StaticBankArtifact.model_validate_json(
                    canonical_json_bytes(full_payload), strict=True
                )
                self._require_six_capabilities(supplied)
            except (ValidationError, FastPathError):
                return None
            supplied_by_capability = _bank_by_capability(supplied)
            parent_by_capability = _bank_by_capability(parent)
            field = "description" if stage == "s2" else "body"
            forbidden = self._boundary_changes(parent, supplied, stage)
            if forbidden:
                return None
            model_payload = {
                "edits": [
                    {
                        "capability_id": capability,
                        field: getattr(supplied_by_capability[capability], field),
                    }
                    for capability in CAPABILITIES
                    if getattr(supplied_by_capability[capability], field)
                    != getattr(parent_by_capability[capability], field)
                ]
            }
        bank = self._compile_candidate_payload(
            model_payload, stage=stage, parent=parent
        )
        if bank is None:
            return None
        path = self.output_root / "banks" / f"{stage}-candidate.json"
        if not path.exists():
            atomic_write_json(path, bank.model_dump(mode="json"))
        return bank

    def _compile_sparse_s1_payload(
        self,
        payload: dict[str, object],
        *,
        parent: StaticBankArtifact,
        feedback_bundle_sha256: str,
        feedback_patchable_capabilities: frozenset[str],
        expected_patch_capabilities: frozenset[str] | None = None,
        artifact_prefix: str = "s1",
    ) -> StaticBankArtifact | None:
        try:
            authoring_input = self.s1_authoring_input()
            draft = bind_sparse_patch_draft(
                canonical_json_bytes(payload),
                parent_bank=parent,
                authoring_input=authoring_input,
                feedback_bundle_sha256=feedback_bundle_sha256,
            )
            patched = frozenset(
                item.capability_id for item in draft.skills if item.action == "patch"
            )
            if (
                not patched
                or len(patched) > self.spec.s1_settings.max_patched_capabilities
                or patched & set(self.spec.s1_settings.protected_capabilities)
                or not patched <= feedback_patchable_capabilities
            ):
                raise _S1CandidateRejected("sparse_patch_scope_rejected")
            if (
                expected_patch_capabilities is not None
                and patched != expected_patch_capabilities
            ):
                raise _S1CandidateRejected("sparse_patch_branch_scope_rejected")
            draft_by_capability = {item.capability_id: item for item in draft.skills}
            if self.spec.s1_settings.bounded_edit_surface == "single-step-replace":
                templates = {
                    item.capability_id: item
                    for item in decode_sparse_parent_content(
                        parent, self.s1_authoring_input()
                    )
                }
                for capability in patched:
                    item = draft_by_capability[capability]
                    assert item.patch is not None
                    source = templates[capability]
                    if len(item.patch.steps) != len(source.steps):
                        raise _S1CandidateRejected(
                            "bounded_single_step_replace_rejected"
                        )
                    changed_steps = sum(
                        left != right
                        for left, right in zip(
                            item.patch.steps, source.steps, strict=True
                        )
                    )
                    if (
                        item.patch.objective != source.objective
                        or item.patch.fallback_instruction
                        != source.fallback_instruction
                        or item.patch.citation_source_ids != source.citation_source_ids
                        or changed_steps != 1
                    ):
                        raise _S1CandidateRejected(
                            "bounded_single_step_replace_rejected"
                        )
            for (
                capability,
                phrases,
            ) in self.spec.s1_settings.required_patch_phrases.items():
                if (
                    expected_patch_capabilities is not None
                    and capability not in expected_patch_capabilities
                ):
                    continue
                item = draft_by_capability.get(capability)
                if item is None or item.action != "patch" or item.patch is None:
                    raise _S1CandidateRejected("required_patch_target_missing")
                authored = "\n".join(
                    (
                        item.patch.objective,
                        *(step.instruction for step in item.patch.steps),
                        item.patch.fallback_instruction,
                    )
                ).casefold()
                if any(phrase.casefold() not in authored for phrase in phrases):
                    raise _S1CandidateRejected("required_patch_phrase_missing")
            compiled = compile_sparse_s1_candidate(
                parent_bank=parent,
                authoring_input=authoring_input,
                sparse_draft=draft,
                tool_registry_runtime_sha256=parent.tool_registry_runtime_sha256,
            )
            parent_by_capability = _bank_by_capability(parent)
            candidate_by_capability = _bank_by_capability(compiled.bank)
            if any(
                parent_by_capability[capability].description
                != candidate_by_capability[capability].description
                for capability in CAPABILITIES
            ):
                raise _S1CandidateRejected("frozen_description_changed")
        except _S1CandidateRejected:
            raise
        except S1SparsePatchError as error:
            raise _S1CandidateRejected("sparse_contract_rejected") from error
        except (ValidationError, ValueError) as error:
            raise _S1CandidateRejected("sparse_compilation_rejected") from error
        artifacts = {
            f"{artifact_prefix}-sparse-draft.json": draft.model_dump(mode="json"),
            f"{artifact_prefix}-sparse-compilation-receipt.json": compiled.receipt.model_dump(
                mode="json"
            ),
        }
        for name, artifact in artifacts.items():
            path = self.output_root / "banks" / name
            if path.exists():
                if load_json(path) != artifact:
                    raise FastPathError(f"S1 sparse artifact changed on resume: {name}")
            else:
                atomic_write_json(path, artifact)
        return compiled.bank

    @staticmethod
    def _write_canonical_resume_artifact(
        path: Path, payload: object, *, label: str
    ) -> None:
        expected = canonical_json_bytes(payload)
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as error:
                raise FastPathError(f"cannot read {label} on resume") from error
            if existing != expected:
                raise FastPathError(f"{label} changed on resume")
            return
        atomic_write_json(path, payload)

    @staticmethod
    def _boundary_changes(
        parent: StaticBankArtifact, candidate: StaticBankArtifact, stage: str
    ) -> tuple[str, ...]:
        before = _bank_by_capability(parent)
        after = _bank_by_capability(candidate)
        if set(before) != set(after):
            return ("capability set changed",)
        allowed = {"s2": {"description"}, "full": {"body"}}[stage]
        compared = {"description", "body", "static_refs", "operators"}
        violations: list[str] = []
        for capability in CAPABILITIES:
            source = before[capability]
            target = after[capability]
            changed = {
                field
                for field in compared
                if getattr(source, field) != getattr(target, field)
            }
            forbidden = changed - allowed
            if forbidden:
                violations.append(
                    f"{capability} changed forbidden fields: {','.join(sorted(forbidden))}"
                )
        return tuple(violations)

    def _write_selected_bank(self, stage: str, bank: StaticBankArtifact) -> Path:
        path = self.output_root / "banks" / f"{stage}-selected.json"
        if path.exists():
            existing = StaticBankArtifact.model_validate_json(
                path.read_bytes(), strict=True
            )
            if existing != bank:
                raise FastPathError(f"selected {stage} Bank changed on resume")
        else:
            atomic_write_json(path, bank.model_dump(mode="json"))
        return path

    def _load_selected_bank(self, stage: str) -> StaticBankArtifact:
        path = self.output_root / "banks" / f"{stage}-selected.json"
        try:
            return StaticBankArtifact.model_validate_json(
                path.read_bytes(), strict=True
            )
        except (OSError, ValueError, ValidationError) as error:
            raise FastPathError(f"missing or invalid selected {stage} Bank") from error

    def _decision_path(self, stage: str) -> Path:
        return self.output_root / "decisions" / f"{stage}.json"

    def _existing_decision(self, stage: str) -> StageDecision | None:
        path = self._decision_path(stage)
        if not path.exists():
            return None
        try:
            return StageDecision.model_validate_json(path.read_bytes(), strict=True)
        except (ValueError, ValidationError) as error:
            raise FastPathError(f"invalid {stage} decision") from error

    def _save_decision(self, decision: StageDecision) -> StageDecision:
        path = self._decision_path(decision.stage)
        if path.exists():
            existing = StageDecision.model_validate_json(path.read_bytes(), strict=True)
            if existing != decision:
                raise FastPathError(f"{decision.stage} decision changed on resume")
            return existing
        atomic_write_json(path, decision.model_dump(mode="json"))
        return decision

    # ------------------------------- metrics/gates -------------------------

    def _summary(
        self,
        rows: Mapping[str, AssistantObservation],
        queries: Sequence[Query],
    ) -> dict[str, object]:
        query_by_id = {item.query_id: item for item in queries}
        by_capability: dict[str, list[AssistantObservation]] = defaultdict(list)
        for query_id, row in rows.items():
            capability = query_by_id[query_id].canonical_capability
            assert capability is not None
            by_capability[capability].append(row)
        capability_gcs = {
            capability: (
                sum(item.gcs_score for item in by_capability[capability])
                / len(by_capability[capability])
            )
            for capability in CAPABILITIES
        }
        return {
            "query_count": len(rows),
            "capability_macro_gcs": sum(capability_gcs.values()) / len(CAPABILITIES),
            "query_micro_gcs": sum(item.gcs_score for item in rows.values())
            / len(rows),
            "capability_gcs": capability_gcs,
            "hard_errors": sum(item.hard_error for item in rows.values()),
            "card_violations": sum(item.card_violation for item in rows.values()),
            "evidence_violations": sum(
                item.evidence_violation for item in rows.values()
            ),
            "tool_violations": sum(item.tool_violation for item in rows.values()),
        }

    @staticmethod
    def _route_macro_f1(
        rows: Mapping[str, AssistantObservation], queries: Sequence[Query]
    ) -> float:
        by_id = {query.query_id: query for query in queries}
        scores: list[float] = []
        for capability in CAPABILITIES:
            tp = fp = fn = 0
            for query_id, row in rows.items():
                expected = by_id[query_id].canonical_capability
                predicted = row.selected_capability
                tp += expected == capability and predicted == capability
                fp += expected != capability and predicted == capability
                fn += expected == capability and predicted != capability
            denominator = 2 * tp + fp + fn
            scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
        return sum(scores) / len(scores)

    @staticmethod
    def _smoke_ok(rows: Mapping[str, AssistantObservation]) -> bool:
        return len(rows) == 24 and not any(row.hard_error for row in rows.values())

    @staticmethod
    def _s1_smoke_ok(rows: Mapping[str, AssistantObservation]) -> bool:
        # dev smoke is an operational preflight, not a candidate-selection
        # population.  A captured response-contract lapse remains a valid
        # observation and is screened only on paired replay200 evidence.
        return len(rows) == 24 and all(row.oracle_available for row in rows.values())

    @staticmethod
    def _s1_development_scores(
        rows: Mapping[str, AssistantObservation],
        queries: Sequence[Query],
        *,
        config: Literal["llm_static", "s1"],
    ) -> tuple[GCSQueryScoreV2, ...]:
        query_ids = {query.query_id for query in queries}
        if len(query_ids) != len(queries) or set(rows) != query_ids:
            raise FastPathError("S1 development screen population is not rectangular")
        population = build_gcs_population_v2(tuple(queries))
        component_by_query = {
            item.query_id: item.component_id for item in population.bindings
        }
        scores: list[GCSQueryScoreV2] = []
        try:
            for query in queries:
                capability = query.canonical_capability
                assert capability is not None
                observation = rows[query.query_id]
                components = observation.gcs_components
                scores.append(
                    GCSQueryScoreV2(
                        query_id=query.query_id,
                        config=config,
                        canonical_capability=capability,
                        component_id=component_by_query[query.query_id],
                        route_disposition=(
                            "pass" if components["route_acceptable"] else "fail"
                        ),
                        answer_mode=observation.answer_mode,
                        oracle_available=observation.oracle_available,
                        semantic_claim_support_resolved=observation.oracle_available,
                        route_acceptable=int(components["route_acceptable"]),
                        no_hard_error=int(components["no_hard_error"]),
                        tool_contract_pass=int(components["tool_contract_pass"]),
                        evidence_grounded=int(components["evidence_grounded"]),
                        output_contract_pass=int(components["output_contract_pass"]),
                        hard_error=int(observation.hard_error),
                        gcs=int(observation.gcs_score),
                        reason_codes=observation.gcs_reason_codes,
                        evaluated_capability=capability,
                        style_support_status=(
                            "unresolved"
                            if capability == "product.style_recommendation"
                            else None
                        ),
                    )
                )
        except (KeyError, ValidationError, ValueError) as error:
            raise FastPathError("invalid S1 development screen score") from error
        return tuple(scores)

    def _s1_capability_screen(
        self,
        *,
        capability: str,
        queries: Sequence[Query],
        baseline: Mapping[str, AssistantObservation],
        candidate: Mapping[str, AssistantObservation],
    ) -> dict[str, object]:
        """Screen one fan-out branch on its own common route/tool treatment.

        Each branch reuses the frozen Static route and tool trace, so the only
        changed execution surface is the branch's model-generated answer under
        the candidate Skill Body.  The
        forward-only risk policy admits a bounded number of parent-success
        regressions only when gains dominate them by at least four to one.
        Failure-reason migration remains diagnostic; an ordinary failure that
        escalates into a hard/runtime failure is still rejected.
        """

        selected = tuple(
            query for query in queries if query.canonical_capability == capability
        )
        query_ids = tuple(query.query_id for query in selected)
        if (
            not query_ids
            or set(baseline) != set(query_ids)
            or set(candidate) != set(query_ids)
        ):
            raise FastPathError("S1 fan-out branch population is not rectangular")
        baseline_success = sum(int(baseline[item].gcs_score) for item in query_ids)
        candidate_success = sum(int(candidate[item].gcs_score) for item in query_ids)
        gain_query_ids = tuple(
            item
            for item in query_ids
            if baseline[item].gcs_score == 0 and candidate[item].gcs_score == 1
        )
        regression_query_ids = tuple(
            item
            for item in query_ids
            if baseline[item].gcs_score == 1 and candidate[item].gcs_score == 0
        )
        severity_escalation_query_ids = tuple(
            item
            for item in query_ids
            if baseline[item].gcs_score == 0
            and not baseline[item].hard_error
            and candidate[item].gcs_score == 0
            and candidate[item].hard_error
        )
        gains = len(gain_query_ids)
        paired_regressions = len(regression_query_ids)
        net_gain = gains - paired_regressions
        failure_reason_migrations = []
        for item in query_ids:
            if baseline[item].gcs_score != 0 or candidate[item].gcs_score != 0:
                continue
            baseline_reasons = set(baseline[item].gcs_reason_codes)
            candidate_reasons = set(candidate[item].gcs_reason_codes)
            if baseline_reasons == candidate_reasons and (
                item not in severity_escalation_query_ids
            ):
                continue
            failure_reason_migrations.append(
                {
                    "query_id": item,
                    "baseline_reason_codes": sorted(baseline_reasons),
                    "candidate_reason_codes": sorted(candidate_reasons),
                    "introduced_reason_codes": sorted(
                        candidate_reasons - baseline_reasons
                    ),
                    "resolved_reason_codes": sorted(
                        baseline_reasons - candidate_reasons
                    ),
                    "severity_escalated": item in severity_escalation_query_ids,
                }
            )
        contract_reasons = []
        for reason in S1_CONTRACT_REGRESSION_REASON_CODES:
            baseline_count = sum(
                reason in baseline[item].gcs_reason_codes for item in query_ids
            )
            candidate_count = sum(
                reason in candidate[item].gcs_reason_codes for item in query_ids
            )
            contract_reasons.append(
                {
                    "reason_code": reason,
                    "baseline_count": baseline_count,
                    "candidate_count": candidate_count,
                    "new_occurrence_count": sum(
                        reason not in baseline[item].gcs_reason_codes
                        and reason in candidate[item].gcs_reason_codes
                        for item in query_ids
                    ),
                }
            )
        trace_mismatches = sorted(
            item
            for item in query_ids
            if (
                candidate[item].selected_capability
                != baseline[item].selected_capability
                or candidate[item].route_trace_key != baseline[item].route_trace_key
                or candidate[item].tool_trace_key != baseline[item].tool_trace_key
                or candidate[item].tool_trace != baseline[item].tool_trace
            )
        )
        reason_codes: set[str] = set()
        if gains < _S1_CAPABILITY_MIN_GAINS:
            reason_codes.add("gains_below_minimum")
        if net_gain < _S1_CAPABILITY_MIN_NET_GAIN:
            reason_codes.add("net_gain_below_minimum")
        if paired_regressions > _S1_CAPABILITY_MAX_REGRESSIONS:
            reason_codes.add("regressions_exceed_limit")
        if paired_regressions and gains < (
            _S1_CAPABILITY_MIN_GAIN_REGRESSION_RATIO * paired_regressions
        ):
            reason_codes.add("gain_regression_ratio_below_four")
        if len(severity_escalation_query_ids) > (
            _S1_CAPABILITY_MAX_FAILURE_SEVERITY_ESCALATIONS
        ):
            reason_codes.add("failure_severity_escalated")
        if trace_mismatches:
            reason_codes.add("common_route_or_tool_trace_changed")
        return {
            "capability_id": capability,
            "decision": "inherit_parent" if reason_codes else "retain_patch",
            "baseline_success_count": baseline_success,
            "candidate_success_count": candidate_success,
            "gain_count": gains,
            "gain_query_ids": list(gain_query_ids),
            "static_success_to_candidate_failure_count": paired_regressions,
            "regression_query_ids": list(regression_query_ids),
            "net_gain": net_gain,
            "gain_to_regression_ratio": (
                None if paired_regressions == 0 else gains / paired_regressions
            ),
            "bounded_risk_policy_version": _S1_BOUNDED_RISK_POLICY_VERSION,
            "bounded_risk_thresholds": {
                "minimum_gains": _S1_CAPABILITY_MIN_GAINS,
                "minimum_net_gain": _S1_CAPABILITY_MIN_NET_GAIN,
                "maximum_regressions": _S1_CAPABILITY_MAX_REGRESSIONS,
                "minimum_gain_to_regression_ratio": (
                    _S1_CAPABILITY_MIN_GAIN_REGRESSION_RATIO
                ),
                "maximum_failure_severity_escalations": (
                    _S1_CAPABILITY_MAX_FAILURE_SEVERITY_ESCALATIONS
                ),
            },
            "failure_reason_migration_count": len(failure_reason_migrations),
            "failure_reason_migrations": failure_reason_migrations,
            "failure_severity_escalation_count": len(severity_escalation_query_ids),
            "failure_severity_escalation_query_ids": list(
                severity_escalation_query_ids
            ),
            "contract_reasons": contract_reasons,
            "common_trace_mismatch_query_ids": trace_mismatches,
            "reason_codes": sorted(reason_codes),
        }

    @staticmethod
    def _policy_success(
        observation: AssistantObservation,
        surface: Literal["action-policy", "response-policy"],
    ) -> bool:
        components = observation.gcs_components
        if surface == "action-policy":
            return bool(
                components["route_acceptable"] and components["tool_contract_pass"]
            )
        return bool(
            components["no_hard_error"]
            and components["evidence_grounded"]
            and components["output_contract_pass"]
        )

    def _s1_policy_screen(
        self,
        *,
        capability: str,
        surface: Literal["action-policy", "response-policy"],
        queries: Sequence[Query],
        baseline: Mapping[str, AssistantObservation],
        candidate: Mapping[str, AssistantObservation],
    ) -> dict[str, object]:
        """Apply bounded risk to one policy surface and its own success set."""

        query_ids = tuple(query.query_id for query in queries)
        if (
            not query_ids
            or set(baseline) != set(query_ids)
            or set(candidate) != set(query_ids)
        ):
            raise FastPathError("S1 policy screen population is not rectangular")
        parent_ok = {
            query_id: self._policy_success(baseline[query_id], surface)
            for query_id in query_ids
        }
        candidate_ok = {
            query_id: self._policy_success(candidate[query_id], surface)
            for query_id in query_ids
        }
        gains = tuple(
            query_id
            for query_id in query_ids
            if not parent_ok[query_id] and candidate_ok[query_id]
        )
        regressions = tuple(
            query_id
            for query_id in query_ids
            if parent_ok[query_id] and not candidate_ok[query_id]
        )

        def severe(row: AssistantObservation) -> bool:
            if surface == "response-policy":
                return row.hard_error
            response = row.replay_context.get("response")
            error_code = (
                response.get("error_code") if isinstance(response, dict) else None
            )
            return bool(
                row.tool_violation
                or any(item.status == "error" for item in row.tool_trace)
                or error_code in {"runtime_error", "timeout"}
            )

        escalations = tuple(
            query_id
            for query_id in query_ids
            if not parent_ok[query_id]
            and not severe(baseline[query_id])
            and not candidate_ok[query_id]
            and severe(candidate[query_id])
        )
        reason_codes: set[str] = set()
        net_gain = len(gains) - len(regressions)
        if len(gains) < _S1_CAPABILITY_MIN_GAINS:
            reason_codes.add("gains_below_minimum")
        if net_gain < _S1_CAPABILITY_MIN_NET_GAIN:
            reason_codes.add("net_gain_below_minimum")
        if len(regressions) > _S1_CAPABILITY_MAX_REGRESSIONS:
            reason_codes.add("regressions_exceed_limit")
        if regressions and len(gains) < (
            _S1_CAPABILITY_MIN_GAIN_REGRESSION_RATIO * len(regressions)
        ):
            reason_codes.add("gain_regression_ratio_below_four")
        if escalations:
            reason_codes.add("failure_severity_escalated")
        if surface == "response-policy":
            trace_mismatches = self._common_trace_violations(baseline, candidate)
            if trace_mismatches:
                reason_codes.add("fixed_route_or_tool_trace_changed")
        else:
            trace_mismatches = tuple(
                query_id
                for query_id in query_ids
                if (
                    candidate[query_id].selected_capability
                    != baseline[query_id].selected_capability
                    or candidate[query_id].route_trace_key
                    != baseline[query_id].route_trace_key
                )
            )
            if trace_mismatches:
                reason_codes.add("fixed_route_changed")
        migrations = [
            {
                "query_id": query_id,
                "baseline_reason_codes": list(baseline[query_id].gcs_reason_codes),
                "candidate_reason_codes": list(candidate[query_id].gcs_reason_codes),
                "severity_escalated": query_id in escalations,
            }
            for query_id in query_ids
            if not parent_ok[query_id]
            and not candidate_ok[query_id]
            and baseline[query_id].gcs_reason_codes
            != candidate[query_id].gcs_reason_codes
        ]
        return {
            "policy_version": "s1-attributable-policy-screen-v1",
            "capability_id": capability,
            "surface": surface,
            "metric_components": (
                ["route_acceptable", "tool_contract_pass"]
                if surface == "action-policy"
                else ["no_hard_error", "evidence_grounded", "output_contract_pass"]
            ),
            "protected_success_query_ids": [
                query_id for query_id in query_ids if parent_ok[query_id]
            ],
            "baseline_success_count": sum(parent_ok.values()),
            "candidate_success_count": sum(candidate_ok.values()),
            "gain_count": len(gains),
            "gain_query_ids": list(gains),
            "regression_count": len(regressions),
            "regression_query_ids": list(regressions),
            "net_gain": net_gain,
            "gain_to_regression_ratio": (
                None if not regressions else len(gains) / len(regressions)
            ),
            "failure_severity_escalation_count": len(escalations),
            "failure_severity_escalation_query_ids": list(escalations),
            "failure_reason_migrations": migrations,
            "trace_mismatch_query_ids": list(trace_mismatches),
            "decision": "inherit_parent" if reason_codes else "retain_patch",
            "reason_codes": sorted(reason_codes),
        }

    def _paired_component_bootstrap(
        self,
        parent: Mapping[str, AssistantObservation],
        candidate: Mapping[str, AssistantObservation],
        queries: Sequence[Query],
        *,
        scope: str,
    ) -> dict[str, object]:
        """Bootstrap capability-macro GCS over query-connected components."""

        population = build_gcs_population_v2(tuple(queries))
        mutable_members: dict[str, list[str]] = defaultdict(list)
        for binding in population.bindings:
            mutable_members[binding.component_id].append(binding.query_id)
        members = {
            component: tuple(sorted(query_ids))
            for component, query_ids in mutable_members.items()
        }
        component_ids = tuple(
            component for component, _size in population.component_sizes
        )
        by_id = {query.query_id: query for query in queries}
        seed_material = canonical_json_bytes(
            {
                "policy_version": GCS_BOOTSTRAP_POLICY_VERSION,
                "root_seed": GCS_BOOTSTRAP_ROOT_SEED,
                "scope": scope,
                "population_mapping_sha256": population.population_mapping_sha256,
            }
        )
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:16], "big")
        rng = random.Random(seed)
        draw_hasher = hashlib.sha256()
        samples: list[float] = []
        for _replicate in range(GCS_BOOTSTRAP_REPLICATES):
            draw = tuple(
                component_ids[rng.randrange(len(component_ids))] for _ in component_ids
            )
            draw_hasher.update(canonical_json_bytes(list(draw)))
            parent_values: dict[str, list[float]] = defaultdict(list)
            candidate_values: dict[str, list[float]] = defaultdict(list)
            for component in draw:
                for query_id in members[component]:
                    capability = by_id[query_id].canonical_capability
                    assert capability is not None
                    parent_values[capability].append(parent[query_id].gcs_score)
                    candidate_values[capability].append(candidate[query_id].gcs_score)
            if set(parent_values) != set(GCS_CAPABILITY_ORDER) or set(
                candidate_values
            ) != set(GCS_CAPABILITY_ORDER):
                continue
            parent_macro = sum(
                sum(parent_values[capability]) / len(parent_values[capability])
                for capability in GCS_CAPABILITY_ORDER
            ) / len(GCS_CAPABILITY_ORDER)
            candidate_macro = sum(
                sum(candidate_values[capability]) / len(candidate_values[capability])
                for capability in GCS_CAPABILITY_ORDER
            ) / len(GCS_CAPABILITY_ORDER)
            samples.append(100.0 * (candidate_macro - parent_macro))
        samples.sort()
        complete = len(samples) == GCS_BOOTSTRAP_REPLICATES

        def percentile(quantile: float) -> float | None:
            if not complete:
                return None
            position = (len(samples) - 1) * quantile
            lower = int(position)
            upper = min(lower + 1, len(samples) - 1)
            fraction = position - lower
            return samples[lower] * (1.0 - fraction) + samples[upper] * fraction

        return {
            "component_count": len(component_ids),
            "replicates_requested": GCS_BOOTSTRAP_REPLICATES,
            "replicates_available": len(samples),
            "ci95_low_pp": percentile(0.025),
            "ci95_high_pp": percentile(0.975),
            "derived_seed": seed,
            "draw_stream_sha256": draw_hasher.hexdigest(),
        }

    def _s1_gate(
        self,
        parent: Mapping[str, AssistantObservation],
        candidate: Mapping[str, AssistantObservation],
        queries: Sequence[Query],
        *,
        phase: str,
        treated_capabilities: frozenset[str] | None = None,
    ) -> tuple[bool, tuple[str, ...], dict[str, object]]:
        if phase not in {"replay200", "body_gate75", "test300"}:
            raise ValueError(f"unknown S1 gate phase: {phase}")
        query_by_id = {query.query_id: query for query in queries}
        evaluated_candidate = candidate
        if treated_capabilities is not None:
            if not treated_capabilities or not treated_capabilities <= set(
                CAPABILITIES
            ):
                raise ValueError("S1 treated capabilities are invalid")
            evaluated_candidate = {}
            for query_id in parent:
                capability = query_by_id[query_id].canonical_capability
                parent_row = parent[query_id]
                candidate_row = candidate[query_id]
                treatment_reached = (
                    capability in treated_capabilities
                    and parent_row.selected_capability == capability
                    and candidate_row.selected_capability == capability
                )
                # Description/routing is frozen in S1.  A target row where the
                # two independent route samples disagree did not receive a
                # comparable Body treatment, so it is conservatively aliased
                # to parent just like every byte-inherited capability.
                evaluated_candidate[query_id] = (
                    candidate_row if treatment_reached else parent_row
                )
        before = self._summary(parent, queries)
        after = self._summary(evaluated_candidate, queries)
        reasons: list[str] = []
        coverage_failures = tuple(
            sorted(
                {
                    query_id
                    for rows in (parent, evaluated_candidate)
                    for query_id, observation in rows.items()
                    if not observation.oracle_available
                }
            )
        )
        if coverage_failures:
            reasons.append("S1 oracle coverage is incomplete")
        macro_delta_pp = 100.0 * (
            float(after["capability_macro_gcs"]) - float(before["capability_macro_gcs"])
        )
        hard_error_delta_pp = 100.0 * (
            (int(after["hard_errors"]) - int(before["hard_errors"])) / len(queries)
        )
        minimum_macro = (
            self.spec.gates.s1_replay_macro_delta_pp_min
            if phase == "replay200"
            else self.spec.gates.s1_system_macro_delta_pp_min
        )
        if macro_delta_pp < minimum_macro - 1e-12:
            reasons.append(f"capability-macro GCS delta is below {minimum_macro:g}pp")
        if hard_error_delta_pp > self.spec.gates.s1_hard_error_delta_pp_max + 1e-12:
            reasons.append("hard-error delta exceeds +1pp")
        floor_pp = self.spec.gates.s1_max_capability_drop_pp
        capability_delta_pp: dict[str, float] = {}
        for capability in CAPABILITIES:
            delta_pp = 100.0 * (
                float(after["capability_gcs"][capability])
                - float(before["capability_gcs"][capability])
            )
            capability_delta_pp[capability] = delta_pp
            if delta_pp < -floor_pp - 1e-12:
                reasons.append(f"{capability} declined by more than {floor_pp:g}pp")
        bootstrap: dict[str, object] | None = None
        if phase in {"body_gate75", "test300"}:
            bootstrap = self._paired_component_bootstrap(
                parent,
                evaluated_candidate,
                queries,
                scope=("s1-body-gate75" if phase == "body_gate75" else "s1-test300"),
            )
            low = bootstrap["ci95_low_pp"]
            if low is None:
                reasons.append("S1 component bootstrap interval is incomplete")
            elif float(low) < self.spec.gates.s1_bootstrap_ci95_lower_pp_min - 1e-12:
                reasons.append("S1 component-bootstrap CI95 lower bound is below 0pp")
        metrics: dict[str, object] = {
            "phase": phase,
            "parent": before,
            "candidate": after,
            "macro_delta_pp": macro_delta_pp,
            "hard_error_delta_pp": hard_error_delta_pp,
            "capability_delta_pp": capability_delta_pp,
            "oracle_coverage_complete": not coverage_failures,
            "oracle_coverage_failure_query_ids": list(coverage_failures),
            "treated_capabilities": (
                None if treated_capabilities is None else sorted(treated_capabilities)
            ),
            "untreated_rows_aliased_to_parent": (
                0
                if treated_capabilities is None
                else sum(
                    query.canonical_capability not in treated_capabilities
                    for query in queries
                )
            ),
            "treated_route_mismatch_rows_aliased_to_parent": (
                0
                if treated_capabilities is None
                else sum(
                    query.canonical_capability in treated_capabilities
                    and (
                        parent[query.query_id].selected_capability
                        != query.canonical_capability
                        or candidate[query.query_id].selected_capability
                        != query.canonical_capability
                    )
                    for query in queries
                )
            ),
        }
        if bootstrap is not None:
            metrics["paired_component_bootstrap"] = bootstrap
        return not reasons, tuple(reasons), metrics

    def _s2_gate(
        self,
        parent: Mapping[str, AssistantObservation],
        candidate: Mapping[str, AssistantObservation],
        queries: Sequence[Query],
    ) -> tuple[bool, tuple[str, ...], dict[str, object]]:
        before = self._summary(parent, queries)
        after = self._summary(candidate, queries)
        before_f1 = self._route_macro_f1(parent, queries)
        after_f1 = self._route_macro_f1(candidate, queries)
        by_id = {query.query_id: query for query in queries}
        corrected = broken = 0
        for query_id in parent:
            acceptable = set(by_id[query_id].acceptable_capabilities)
            old_ok = parent[query_id].selected_capability in acceptable
            new_ok = candidate[query_id].selected_capability in acceptable
            corrected += not old_ok and new_ok
            broken += old_ok and not new_ok
        utility = corrected - self.spec.gates.s2_broken_penalty * broken
        reasons: list[str] = []
        if after_f1 <= before_f1:
            reasons.append("route macro-F1 did not strictly improve")
        if utility <= 0:
            reasons.append("corrected - 2.5*broken is not positive")
        if after["capability_macro_gcs"] < before["capability_macro_gcs"]:
            reasons.append("capability-macro GCS declined")
        if after["hard_errors"] > before["hard_errors"]:
            reasons.append("hard errors increased")
        metrics = {
            "parent": before,
            "candidate": after,
            "parent_route_macro_f1": before_f1,
            "candidate_route_macro_f1": after_f1,
            "corrected": corrected,
            "broken": broken,
            "corrected_minus_penalized_broken": utility,
        }
        return not reasons, tuple(reasons), metrics

    def _creator_schema(
        self,
        stage: str,
        *,
        parent: StaticBankArtifact | None = None,
        s1_patch_capabilities: tuple[str, ...] | None = None,
        dual_policy_capability: str | None = None,
        counterfactual_surface: Literal["action-policy", "response-policy"]
        | None = None,
        counterfactual_parent_success_ids: tuple[str, ...] = (),
        counterfactual_action_condition: Mapping[str, object] | None = None,
        counterfactual_response_signature: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        capability_enum = list(CAPABILITIES)
        if stage == "s1":
            if parent is None:
                raise ValueError("S1 sparse schema requires the parent Bank")
            by_capability = _bank_by_capability(parent)
            if counterfactual_surface is not None:
                targets = self.spec.s1_settings.target_capabilities
                if len(targets) != 1:
                    raise ValueError("counterfactual schema requires one target")
                capability = targets[0]
                if self.spec.s1_settings.proposal_mode == (
                    "single-surface-counterfactual-fanout-v6"
                ):
                    return counterfactual_semantic_policy_patch_output_json_schema(
                        capability_id=capability,
                        parent_skill_sha256=by_capability[capability].skill_sha256,
                        target_surface=counterfactual_surface,
                        parent_success_query_ids=counterfactual_parent_success_ids,
                        expected_action_condition=counterfactual_action_condition,
                        expected_response_signature=counterfactual_response_signature,
                    )
                if self.spec.s1_settings.proposal_mode == (
                    "single-surface-counterfactual-fanout-v5"
                ):
                    return counterfactual_typed_policy_patch_output_json_schema(
                        capability_id=capability,
                        parent_skill_sha256=by_capability[capability].skill_sha256,
                        target_surface=counterfactual_surface,
                        parent_success_query_ids=counterfactual_parent_success_ids,
                        expected_action_condition=counterfactual_action_condition,
                    )
                return counterfactual_policy_patch_output_json_schema(
                    capability_id=capability,
                    parent_skill_sha256=by_capability[capability].skill_sha256,
                    target_surface=counterfactual_surface,
                    parent_success_query_ids=counterfactual_parent_success_ids,
                )
            if dual_policy_capability is not None:
                if dual_policy_capability not in by_capability:
                    raise ValueError("dual-policy capability is absent from parent")
                return dual_policy_patch_output_json_schema(
                    capability_id=dual_policy_capability,
                    parent_skill_sha256=by_capability[
                        dual_policy_capability
                    ].skill_sha256,
                )
            templates = {
                item.capability_id: item
                for item in decode_sparse_parent_content(
                    parent, self.s1_authoring_input()
                )
            }
            return sparse_patch_output_json_schema(
                parent_skill_sha256_by_capability={
                    capability: by_capability[capability].skill_sha256
                    for capability in CAPABILITIES
                },
                frozen_objective_by_capability={
                    capability: templates[capability].objective
                    for capability in CAPABILITIES
                },
                patch_capabilities=(
                    self.spec.s1_settings.target_capabilities
                    if s1_patch_capabilities is None
                    else s1_patch_capabilities
                ),
            )
        field = "description" if stage == "s2" else "body"
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["edits"],
            "properties": {
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 6 if stage == "s2" else 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["capability_id", field],
                        "properties": {
                            "capability_id": {
                                "type": "string",
                                "enum": capability_enum,
                            },
                            field: {"type": "string", "minLength": 1},
                        },
                    },
                }
            },
        }

    def _route_attribution_confusion(
        self, attribution: Mapping[str, Mapping[str, object]]
    ) -> dict[str, dict[str, int]]:
        by_id = self.query_by_id()
        matrix = {
            expected: {predicted: 0 for predicted in (*CAPABILITIES, "<none>")}
            for expected in CAPABILITIES
        }
        for query_id, row in attribution.items():
            expected = str(
                row.get("expected_capability") or by_id[query_id].canonical_capability
            )
            predicted_value = (
                row.get("selected_capability")
                or row.get("predicted_capability")
                or row.get("local_capability")
            )
            predicted = (
                str(predicted_value) if predicted_value in CAPABILITIES else "<none>"
            )
            matrix[expected][predicted] += 1
        return matrix

    # ------------------------------- S1 ------------------------------------

    def run_static_opt800(self) -> dict[str, object]:
        """Execute a fresh Static baseline and publish one create-only JSONL.

        The configured ``paths.opt_static_results`` is the destination.  It
        must not already exist, which keeps historical model lineages
        immutable and prevents a new Assistant identity from being projected
        over an old baseline.
        """

        destination = self._path("opt_static_results")
        queries = [query for query in self.queries() if query.split == "opt_pool"]
        rows = self._assistant_many(
            split="static-opt800",
            config="llm_static",
            queries=queries,
            bank=self.static_bank(),
        )
        self._validate_opt_static_model_identity(rows)
        fixed_samples = self.fixed_samples_from_static(rows)
        payloads = [rows[query.query_id].model_dump(mode="json") for query in queries]
        expected_bytes = b"".join(canonical_json_bytes(row) for row in payloads)
        if destination.exists():
            if destination.read_bytes() != expected_bytes:
                raise FastPathError(
                    "Static opt800 destination differs from the completed call journal"
                )
        else:
            atomic_write_jsonl(destination, payloads)
        observed_sha256 = _file_sha(destination)
        bootstrap = {
            "schema_version": 1,
            "kind": "core-fast-static-opt800-bootstrap",
            "assistant_model": self.spec.models["assistant"].requested_model,
            "assistant_contract": self.spec.runtime.assistant_contract,
            "opt_static_results": str(destination),
            "opt_static_results_sha256": observed_sha256,
            "row_count": len(rows),
            "gcs_success_count": sum(int(item.gcs_score) for item in rows.values()),
            "hard_error_count": sum(int(item.hard_error) for item in rows.values()),
            "observed_dashscope_cost_cny": self.calls.observed_cost(),
            "fixed_samples": fixed_samples.model_dump(mode="json"),
        }
        bootstrap_path = self.output_root / "static-opt800-bootstrap.json"
        if bootstrap_path.exists():
            if load_json(bootstrap_path) != bootstrap:
                raise FastPathError("Static opt800 bootstrap differs on resume")
        else:
            atomic_write_json(bootstrap_path, bootstrap)
        return bootstrap

    def run_s1_parent_opt800(self) -> dict[str, object]:
        """Execute and freeze opt800 under the accepted S1 parent."""

        binding = self.spec.s1_parent
        if binding is None:
            raise FastPathError("s1-parent-opt800 requires an S1 parent binding")
        destination = self._bound_path(binding.opt_results_path)
        queries = [query for query in self.queries() if query.split == "opt_pool"]
        parent = self.s1_parent_bank()
        rows = self._assistant_many(
            split=f"s1-parent-opt800-{binding.source_round_id}",
            # The split/call ID carries the accepted-parent lineage identity;
            # the live adapter only accepts the five canonical matrix configs.
            # Executing the accepted Bank is still the S1 treatment.
            config="s1",
            queries=queries,
            bank=parent,
        )
        self._validate_opt_static_model_identity(rows)
        for query_id, observation in rows.items():
            result = observation.replay_context.get("assistant_result")
            if (
                not isinstance(result, dict)
                or result.get("bank_sha256") != parent.bank_sha256
                or observation.assistant_contract
                != self.spec.runtime.assistant_contract
            ):
                raise FastPathError(
                    f"fresh S1 parent opt800 identity differs: {query_id}"
                )
        fixed_samples = self.fixed_samples_from_static(rows)
        payloads = [rows[query.query_id].model_dump(mode="json") for query in queries]
        expected_bytes = b"".join(canonical_json_bytes(row) for row in payloads)
        if destination.exists():
            if destination.read_bytes() != expected_bytes:
                raise FastPathError(
                    "S1 parent opt800 destination differs from the completed journal"
                )
        else:
            atomic_write_jsonl(destination, payloads)
        observed_sha256 = _file_sha(destination)
        bootstrap = {
            "schema_version": 1,
            "kind": "core-fast-s1-parent-opt800-bootstrap",
            "source_round_id": binding.source_round_id,
            "parent_bank_sha256": parent.bank_sha256,
            "parent_bank_file_sha256": binding.bank_file_sha256,
            "source_decision_file_sha256": binding.decision_file_sha256,
            "source_manifest_file_sha256": binding.manifest_file_sha256,
            "assistant_model": self.spec.models["assistant"].requested_model,
            "assistant_contract": self.spec.runtime.assistant_contract,
            "opt_results_path": str(destination),
            "opt_results_sha256": observed_sha256,
            "row_count": len(rows),
            "gcs_success_count": sum(int(item.gcs_score) for item in rows.values()),
            "hard_error_count": sum(int(item.hard_error) for item in rows.values()),
            "observed_dashscope_cost_cny": self.calls.observed_cost(),
            "fixed_samples": fixed_samples.model_dump(mode="json"),
        }
        bootstrap_path = self.output_root / "s1-parent-opt800-bootstrap.json"
        if bootstrap_path.exists():
            if load_json(bootstrap_path) != bootstrap:
                raise FastPathError("S1 parent opt800 bootstrap differs on resume")
        else:
            atomic_write_json(bootstrap_path, bootstrap)
        return bootstrap

    @staticmethod
    def _write_or_verify_json(path: Path, payload: object, *, label: str) -> None:
        if path.exists():
            if load_json(path) != payload:
                raise FastPathError(f"{label} changed on resume")
            return
        atomic_write_json(path, payload)

    def _require_counterfactual_cycle_preflight(
        self, prepared: Mapping[str, object] | None = None
    ) -> dict[str, object] | None:
        """Fail before provider calls unless every pre-registered round is viable."""

        settings = self.spec.s1_settings
        if settings.proposal_mode not in {
            "single-surface-counterfactual-fanout-v5",
            "single-surface-counterfactual-fanout-v6",
        }:
            return None
        if (
            settings.cycle_preflight_path is None
            or settings.cycle_preflight_sha256 is None
        ):
            raise FastPathError("typed counterfactual cycle preflight is not bound")
        path = Path(settings.cycle_preflight_path)
        if not path.is_absolute():
            path = (self.spec_path.parent / path).resolve()
        if not path.is_file() or _file_sha(path) != settings.cycle_preflight_sha256:
            raise FastPathError("typed counterfactual cycle preflight SHA differs")
        receipt = load_json(path)
        rounds = receipt.get("rounds")
        binding = self.spec.s1_parent
        if (
            receipt.get("kind") != "core-fast-s1-counterfactual-cycle-preflight"
            or receipt.get("passed") is not True
            or receipt.get("cycle_id") != settings.cycle_id
            or binding is None
            or receipt.get("parent_bank_sha256") != binding.bank_sha256
            or receipt.get("parent_opt_sha256") != binding.opt_results_sha256
            or not isinstance(rounds, dict)
            or len(rounds) < 2
            or any(
                not isinstance(item, dict)
                or item.get("evidence_feasible") is not True
                or item.get("treatment_sensitive") is not True
                or (
                    settings.feedback_selection_policy
                    in {"parent-counterfactual-v8", "parent-counterfactual-v9"}
                    and item.get("treatment_separable") is not True
                )
                for item in rounds.values()
            )
        ):
            raise FastPathError("typed counterfactual all-round preflight did not pass")
        current = rounds.get(settings.round_id)
        if (
            not isinstance(current, dict)
            or current.get("target_capability") != settings.target_capabilities[0]
            or current.get("target_surface") != settings.target_surface
        ):
            raise FastPathError("typed counterfactual round preflight identity differs")
        if prepared is not None:
            manifest = prepared.get("selection_manifest")
            if not isinstance(manifest, dict) or current.get(
                "selection_manifest_sha256"
            ) != manifest.get("manifest_sha256"):
                raise FastPathError(
                    "typed counterfactual evidence selection differs from preflight"
                )
        return receipt

    def prepare_feedback_selection(self) -> dict[str, object]:
        """Freeze discovery600 summary and selection without provider calls."""

        self.output_root.mkdir(parents=True, exist_ok=True)
        identity_path = self.output_root / "feedback-round-identity.json"
        identity = {
            "schema_version": 1,
            "kind": "core-fast-feedback-round-identity",
            "experiment_id": self.spec.experiment_id,
            "round_id": self.spec.s1_settings.round_id,
            "spec_sha256": _file_sha(self.spec_path),
            "feedback_mode": self.spec.s1_settings.feedback_mode,
            "feedback_total_count": self.spec.s1_settings.feedback_total_count,
            "feedback_canary_count": self.spec.s1_settings.feedback_canary_count,
            "feedback_selection_policy": (
                self.spec.s1_settings.feedback_selection_policy
            ),
            "feedback_allocation": self.spec.s1_settings.feedback_allocation,
            "target_capabilities": list(self.spec.s1_settings.target_capabilities),
        }
        self._write_or_verify_json(
            identity_path, identity, label="Feedback round identity"
        )
        counterfactual = (
            self.spec.s1_settings.feedback_selection_policy
            in _COUNTERFACTUAL_SELECTION_POLICIES
        )
        opt = self.s1_parent_opt() if counterfactual else self.opt_static()
        roles = self.opt_fold_roles()
        try:
            if counterfactual:
                population = build_parent_counterfactual_population(
                    queries=self.queries(),
                    observations=opt,
                    settings=self.spec.s1_settings,
                )
                selected = select_parent_counterfactual_samples(
                    population, self.spec.s1_settings
                )
                binding = self.spec.s1_parent
                assert binding is not None and binding.opt_results_sha256 is not None
                manifest = build_parent_counterfactual_manifest(
                    population=population,
                    selected=selected,
                    settings=self.spec.s1_settings,
                    parent_bank_sha256=binding.bank_sha256,
                    parent_opt_sha256=binding.opt_results_sha256,
                )
                summary_unsigned = {
                    "schema_version": 1,
                    "kind": "core-fast-parent-counterfactual-summary",
                    "cycle_id": self.spec.s1_settings.cycle_id,
                    "round_id": self.spec.s1_settings.round_id,
                    "population_count": len(population),
                    "population_sha256": sha256_bytes(
                        canonical_json_bytes(list(population))
                    ),
                    "selected_cluster_sha256": next(
                        str(row["cluster_sha256"])
                        for row in selected
                        if row["counterfactual_role"] == "cluster_failure"
                    ),
                    "role_counts": dict(
                        sorted(
                            Counter(
                                str(row["counterfactual_role"]) for row in selected
                            ).items()
                        )
                    ),
                }
                summary = {
                    **summary_unsigned,
                    "summary_sha256": sha256_bytes(
                        canonical_json_bytes(summary_unsigned)
                    ),
                }
            else:
                population = build_discovery_feedback_population(
                    queries=self.queries(), observations=opt, fold_roles=roles
                )
                summary = build_discovery_failure_summary(population)
                selected = select_feedback_samples(population, self.spec.s1_settings)
                manifest = build_feedback_selection_manifest(
                    population=population,
                    selected=selected,
                    settings=self.spec.s1_settings,
                    opt_static_sha256=self.spec.opt_static_results_sha256,
                    opt_fold_mapping_sha256=self.spec.opt_fold_mapping_sha256,
                )
        except (CounterfactualEvidenceError, KeyError, TypeError, ValueError) as error:
            raise FastPathError(
                f"cannot prepare S1 Feedback selection: {error}"
            ) from error
        inputs = self.output_root / "inputs"
        self._write_or_verify_json(
            inputs / "discovery-failure-summary.json",
            summary,
            label="discovery failure summary",
        )
        self._write_or_verify_json(
            inputs / "feedback-selection-manifest.json",
            manifest,
            label="Feedback selection manifest",
        )
        receipt_unsigned = {
            "schema_version": 1,
            "kind": "core-fast-feedback-selection-receipt",
            "round_id": self.spec.s1_settings.round_id,
            "discovery_summary_sha256": summary["summary_sha256"],
            "selection_manifest_sha256": manifest["manifest_sha256"],
            "provider_calls": 0,
        }
        receipt = {
            **receipt_unsigned,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_unsigned)),
        }
        self._write_or_verify_json(
            inputs / "feedback-selection-receipt.json",
            receipt,
            label="Feedback selection receipt",
        )
        return {
            "discovery_summary": summary,
            "selection_manifest": manifest,
            "selection_receipt": receipt,
        }

    def _feedback_payload(
        self,
        *,
        query: Query,
        baseline: AssistantObservation,
        sample: Mapping[str, object],
        selection_manifest_sha256: str,
        round_id: str,
    ) -> dict[str, object]:
        return {
            "operation": "strict_visual_feedback",
            "round_id": round_id,
            "selection_manifest_sha256": selection_manifest_sha256,
            "selection_ordinal": sample["selection_ordinal"],
            "query": project_feedback_query(query),
            "baseline": (
                project_feedback_observation_for_surface(
                    baseline, self.spec.s1_settings.target_surface
                )
                if self.spec.s1_settings.proposal_mode in _COUNTERFACTUAL_PROPOSAL_MODES
                and self.spec.s1_settings.target_surface is not None
                else project_feedback_observation(baseline)
            ),
            "sample_role": sample["role"],
            "selection_class": sample["selection_class"],
            "failure_cluster": sample["failure_cluster"],
            "target_surface": self.spec.s1_settings.target_surface,
            "attribution_policy": (
                "single-surface-counterfactual-v6"
                if self.spec.s1_settings.proposal_mode
                == "single-surface-counterfactual-fanout-v6"
                else "single-surface-counterfactual-v5"
                if self.spec.s1_settings.proposal_mode
                == "single-surface-counterfactual-fanout-v5"
                else "single-surface-counterfactual-v4"
                if self.spec.s1_settings.proposal_mode
                == "single-surface-counterfactual-fanout-v4"
                else "dual-policy-attribution-v1"
                if self.spec.s1_settings.proposal_mode
                == "six-capability-dual-policy-fanout-fanin-v3"
                else "answer-stage-body-only-v1"
                if self.spec.s1_settings.feedback_selection_policy
                == "discovery-attributed-v4"
                else "unrestricted-development-v1"
            ),
            "failure_cluster_sha256": sample["cluster_sha256"],
            "response_schema": VisualFeedbackOutput.model_json_schema(),
            "no_replacement": True,
        }

    @staticmethod
    def _parse_feedback_result(result: CallResult) -> dict[str, object] | None:
        if result.status != "success" or not result.schema_valid or not result.output:
            return None
        raw = result.output.get("feedback", result.output)
        try:
            model = VisualFeedbackOutput.model_validate_json(
                canonical_json_bytes(raw), strict=True
            )
        except ValidationError:
            return None
        compatible = tuple(
            suggestion
            for suggestion in model.skill_suggestions
            if suggestion.startswith("[policy_compatible] ")
        )
        parsed = model.model_dump(mode="json")
        parsed["skill_suggestions"] = list(compatible)
        return parsed

    def _feedback_batch(
        self,
        *,
        samples: Sequence[Mapping[str, object]],
        query_by_id: Mapping[str, Query],
        opt: Mapping[str, AssistantObservation],
        selection_manifest_sha256: str,
    ) -> tuple[dict[str, dict[str, object]], int, int]:
        feedback: dict[str, dict[str, object]] = {}
        resume_hits = 0
        provider_calls = 0
        workers = min(self.spec.concurrency.feedback, len(samples))
        if workers == 0:
            return feedback, resume_hits, provider_calls
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for sample in samples:
                query_id = str(sample["query_id"])
                call_id = _safe_id(
                    f"{self.spec.s1_settings.round_id}-feedback-{int(sample['selection_ordinal']):03d}-{query_id}"
                )
                if self.calls.get("feedback", call_id) is not None:
                    resume_hits += 1
                else:
                    provider_calls += 1
                query = query_by_id[query_id]
                payload = self._feedback_payload(
                    query=query,
                    baseline=opt[query_id],
                    sample=sample,
                    selection_manifest_sha256=selection_manifest_sha256,
                    round_id=self.spec.s1_settings.round_id,
                )
                futures[
                    pool.submit(
                        self._call,
                        role="feedback",
                        call_id=call_id,
                        purpose=(
                            f"S1 {self.spec.s1_settings.round_id} frozen discovery Feedback "
                            f"{int(sample['selection_ordinal']):03d} {query_id}"
                        ),
                        payload=payload,
                    )
                ] = sample
            for future in as_completed(futures):
                sample = futures[future]
                result = future.result()
                query_id = str(sample["query_id"])
                feedback[query_id] = {
                    "query_id": query_id,
                    "selection_ordinal": sample["selection_ordinal"],
                    "capability": sample["capability"],
                    "sample_role": sample["role"],
                    "selection_class": sample["selection_class"],
                    "failure_cluster": sample["failure_cluster"],
                    "cluster_sha256": sample["cluster_sha256"],
                    **(
                        {
                            "action_treatment_signature": sample[
                                "action_treatment_signature"
                            ]
                        }
                        if "action_treatment_signature" in sample
                        else {}
                    ),
                    **(
                        {
                            "response_treatment_signature": sample[
                                "response_treatment_signature"
                            ]
                        }
                        if "response_treatment_signature" in sample
                        else {}
                    ),
                    "baseline_observation": (
                        project_feedback_observation_for_surface(
                            opt[query_id], self.spec.s1_settings.target_surface
                        )
                        if self.spec.s1_settings.proposal_mode
                        in _COUNTERFACTUAL_PROPOSAL_MODES
                        and self.spec.s1_settings.target_surface is not None
                        else project_feedback_observation(opt[query_id])
                    ),
                    "status": result.status,
                    "feedback": self._parse_feedback_result(result),
                    "failure_reason": result.failure_reason,
                    "attempt_count": 1,
                    "prior_attempt_failures": [],
                }
        retryable = tuple(
            row
            for row in feedback.values()
            if row["feedback"] is None
            and row["status"] in {"empty_response", "schema_error"}
        )
        for attempt in range(1, self.spec.s1_settings.feedback_format_retry_limit + 1):
            if not retryable:
                break
            remaining_budget = (
                self.spec.limits.max_feedback_calls - self.calls.role_count("feedback")
            )
            if remaining_budget <= 0:
                break
            retryable = retryable[:remaining_budget]
            with ThreadPoolExecutor(max_workers=min(workers, len(retryable))) as pool:
                retry_futures = {}
                for row in retryable:
                    query_id = str(row["query_id"])
                    sample = next(
                        item for item in samples if str(item["query_id"]) == query_id
                    )
                    call_id = _safe_id(
                        f"{self.spec.s1_settings.round_id}-feedback-"
                        f"{int(sample['selection_ordinal']):03d}-{query_id}-"
                        f"format-retry-{attempt}"
                    )
                    if self.calls.get("feedback", call_id) is not None:
                        resume_hits += 1
                    else:
                        provider_calls += 1
                    payload = self._feedback_payload(
                        query=query_by_id[query_id],
                        baseline=opt[query_id],
                        sample=sample,
                        selection_manifest_sha256=selection_manifest_sha256,
                        round_id=self.spec.s1_settings.round_id,
                    )
                    retry_futures[
                        pool.submit(
                            self._call,
                            role="feedback",
                            call_id=call_id,
                            purpose=(
                                f"S1 {self.spec.s1_settings.round_id} Feedback "
                                f"format retry {attempt} {query_id}"
                            ),
                            payload=payload,
                        )
                    ] = row
                for future in as_completed(retry_futures):
                    row = retry_futures[future]
                    result = future.result()
                    prior = list(row["prior_attempt_failures"])
                    prior.append(
                        {
                            "status": row["status"],
                            "failure_reason": row["failure_reason"],
                        }
                    )
                    row.update(
                        {
                            "status": result.status,
                            "feedback": self._parse_feedback_result(result),
                            "failure_reason": result.failure_reason,
                            "attempt_count": int(row["attempt_count"]) + 1,
                            "prior_attempt_failures": prior,
                        }
                    )
            retryable = tuple(
                row
                for row in feedback.values()
                if row["feedback"] is None
                and row["status"] in {"empty_response", "schema_error"}
            )
        return feedback, resume_hits, provider_calls

    @staticmethod
    def _feedback_batch_gate(
        feedback: Mapping[str, Mapping[str, object]],
        *,
        expected_count: int,
    ) -> dict[str, object]:
        observed_count = len(feedback)
        missing_count = max(expected_count - observed_count, 0)
        parse_failures = missing_count + sum(
            row["feedback"] is None for row in feedback.values()
        )
        service_failures = sum(
            row["status"] in {"provider_error", "interrupted_unknown"}
            for row in feedback.values()
        )
        parse_error_rate = parse_failures / expected_count
        service_error_rate = service_failures / expected_count
        passed = (
            observed_count == expected_count
            and parse_error_rate <= project_config.FEEDBACK_JUDGE_ACCEPTABLE_ERROR_RATE
            and service_error_rate <= project_config.FEEDBACK_JUDGE_SERVICE_ERROR_RATE
        )
        return {
            "passed": passed,
            "expected_count": expected_count,
            "observed_count": observed_count,
            "missing_count": missing_count,
            "parse_failure_count": parse_failures,
            "parse_error_rate": parse_error_rate,
            "service_failure_count": service_failures,
            "service_error_rate": service_error_rate,
        }

    def _feedback_evidence_bundle(
        self,
        *,
        parent: StaticBankArtifact,
        prepared: Mapping[str, object],
        feedback: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        summary = prepared["discovery_summary"]
        manifest = prepared["selection_manifest"]
        assert isinstance(summary, dict) and isinstance(manifest, dict)
        grouped = {
            capability: [
                feedback[query_id]
                for query_id in manifest["selected_query_ids"]
                if feedback[query_id]["capability"] == capability
            ]
            for capability in CAPABILITIES
        }
        suggestions: dict[tuple[str, str], dict[str, object]] = {}
        protected_suggestions: dict[tuple[str, str], dict[str, object]] = {}
        rejected_counts: Counter[str] = Counter()
        for query_id in manifest["selected_query_ids"]:
            row = feedback[query_id]
            parsed = row["feedback"]
            if not isinstance(parsed, dict):
                rejected_counts["missing_or_invalid_feedback"] += 1
                continue
            for suggestion in parsed.get("skill_suggestions", []):
                if not isinstance(suggestion, str) or not suggestion.startswith(
                    "[policy_compatible] "
                ):
                    rejected_counts["non_policy_compatible"] += 1
                    continue
                normalized = " ".join(suggestion.split())
                surface: str | None = None
                surface_labeled = (
                    self.spec.s1_settings.proposal_mode
                    == "six-capability-dual-policy-fanout-fanin-v3"
                    or self.spec.s1_settings.proposal_mode
                    in _COUNTERFACTUAL_PROPOSAL_MODES
                )
                if surface_labeled:
                    body = normalized.removeprefix("[policy_compatible] ")
                    surface_matches = tuple(
                        candidate
                        for candidate in ("action-policy", "response-policy")
                        if body.startswith(f"[{candidate}] ")
                    )
                    if len(surface_matches) != 1:
                        rejected_counts["missing_or_ambiguous_policy_surface"] += 1
                        continue
                    surface = surface_matches[0]
                    target_surface = self.spec.s1_settings.target_surface
                    if (
                        self.spec.s1_settings.proposal_mode
                        in _COUNTERFACTUAL_PROPOSAL_MODES
                        and surface != target_surface
                    ):
                        rejected_counts["cross_surface_suggestion"] += 1
                        continue
                key = (str(row["capability"]), normalized.casefold())
                counterfactual_failure = (
                    self.spec.s1_settings.proposal_mode
                    in _COUNTERFACTUAL_PROPOSAL_MODES
                    and row.get("sample_role") == "cluster_failure"
                )
                actionable_failure = counterfactual_failure or (
                    row.get("sample_role") == "failure"
                    and (
                        not surface_labeled
                        and self.spec.s1_settings.feedback_selection_policy
                        != "discovery-attributed-v4"
                        or row.get("selection_class")
                        in {"body_fixable_failure", "boundary_failure"}
                    )
                )
                if (
                    row.get("sample_role") in {"failure", "cluster_failure"}
                    and not actionable_failure
                ):
                    rejected_counts["non_body_fixable_failure"] += 1
                    continue
                destination = (
                    suggestions if actionable_failure else protected_suggestions
                )
                existing = destination.setdefault(
                    key,
                    {
                        "capability": row["capability"],
                        "suggestion": normalized,
                        **({"surface": surface} if surface is not None else {}),
                        "support_query_ids": [],
                    },
                )
                existing["support_query_ids"].append(query_id)  # type: ignore[union-attr]

        def aggregate(
            items: Mapping[tuple[str, str], dict[str, object]],
        ) -> list[dict[str, object]]:
            aggregated = []
            for key in sorted(items):
                item = items[key]
                query_ids = sorted(
                    set(item["support_query_ids"])  # type: ignore[arg-type]
                )
                aggregated.append(
                    {
                        **item,
                        "support_query_ids": query_ids,
                        "support_count": len(query_ids),
                    }
                )
            return aggregated

        actionable = aggregate(suggestions)
        protected = aggregate(protected_suggestions)
        unsigned = {
            "schema_version": 2,
            "round_id": self.spec.s1_settings.round_id,
            "parent_bank_sha256": parent.bank_sha256,
            "opt_static_sha256": (
                self.spec.opt_static_results_sha256
                if self.spec.s1_parent is None
                else self.spec.s1_parent.opt_results_sha256
            ),
            "discovery_population_sha256": manifest.get(
                "discovery_population_sha256", manifest.get("population_sha256")
            ),
            "selection_manifest_sha256": manifest["manifest_sha256"],
            "selection_policy": self.spec.s1_settings.feedback_selection_policy,
            "feedback_total_count": self.spec.s1_settings.feedback_total_count,
            "feedback_success_count": sum(
                row["feedback"] is not None for row in feedback.values()
            ),
            "target_capabilities": list(self.spec.s1_settings.target_capabilities),
            "discovery_summary": summary,
            "feedback_by_capability": grouped,
            "policy_compatible_suggestions": actionable,
            "protected_success_suggestions": protected,
            "rejected_suggestion_counts": dict(sorted(rejected_counts.items())),
        }
        return {
            **unsigned,
            "bundle_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }

    @staticmethod
    def _fanout_creator_payload(
        *,
        capabilities: Sequence[str],
        parent: StaticBankArtifact,
        templates: Sequence[object],
        feedback_bundle: Mapping[str, object],
        feedback_bundle_sha256: str,
        requirements: Mapping[str, object],
        output_schema: Mapping[str, object],
        prior_experiment_memory: Sequence[Mapping[str, object]] = (),
    ) -> dict[str, object]:
        """Build one structured Creator request containing six isolated branches."""

        grouped = feedback_bundle["feedback_by_capability"]
        assert isinstance(grouped, dict)
        suggestions = feedback_bundle["policy_compatible_suggestions"]
        protected_suggestions = feedback_bundle["protected_success_suggestions"]
        assert isinstance(suggestions, list) and isinstance(protected_suggestions, list)
        selected = tuple(capabilities)
        summary = feedback_bundle["discovery_summary"]
        assert isinstance(summary, dict)
        capability_summaries = summary.get("capabilities")
        assert isinstance(capability_summaries, dict)
        selected_rows = {
            capability: list(grouped.get(capability, [])) for capability in selected
        }
        return {
            "operation": "s1_creator",
            "fanout_policy": "six-capability-fanout-fanin-v2",
            "fanout_capabilities": list(selected),
            "parent_bank": parent.model_dump(mode="json"),
            "parent_authoring_content": [
                item.model_dump(mode="json") for item in templates
            ],
            "feedback_evidence_bundle": {
                "bundle_sha256": feedback_bundle_sha256,
                "failure_examples_by_capability": {
                    capability: [
                        row
                        for row in selected_rows[capability]
                        if isinstance(row, dict)
                        and row.get("sample_role") == "failure"
                        and (
                            feedback_bundle["selection_policy"]
                            not in {
                                "discovery-attributed-v4",
                                "discovery-dual-policy-v5",
                            }
                            or row.get("selection_class") == "body_fixable_failure"
                        )
                    ]
                    for capability in selected
                },
                "diagnostic_non_body_failures_by_capability": {
                    capability: [
                        row
                        for row in selected_rows[capability]
                        if isinstance(row, dict)
                        and row.get("sample_role") == "failure"
                        and row.get("selection_class") != "body_fixable_failure"
                    ]
                    for capability in selected
                },
                "protected_success_examples_by_capability": {
                    capability: [
                        row
                        for row in selected_rows[capability]
                        if isinstance(row, dict) and row.get("sample_role") == "anchor"
                    ]
                    for capability in selected
                },
            },
            "capability_discovery_summary": {
                capability: capability_summaries[capability] for capability in selected
            },
            "selection_manifest_sha256": feedback_bundle["selection_manifest_sha256"],
            "failure_repair_suggestions_by_capability": {
                capability: [
                    item
                    for item in suggestions
                    if isinstance(item, dict) and item.get("capability") == capability
                ]
                for capability in selected
            },
            "protected_success_suggestions_by_capability": {
                capability: [
                    item
                    for item in protected_suggestions
                    if isinstance(item, dict) and item.get("capability") == capability
                ]
                for capability in selected
            },
            "feedback_bundle_sha256": feedback_bundle_sha256,
            "prior_experiment_memory": [dict(item) for item in prior_experiment_memory],
            "parent_body_content": {
                item.capability_id: {
                    "objective": item.objective,
                    "steps": [step.model_dump(mode="json") for step in item.steps],
                    "fallback_instruction": item.fallback_instruction,
                    "citation_source_ids": list(item.citation_source_ids),
                }
                for item in templates
            },
            "requirements": dict(requirements),
            "output_schema": dict(output_schema),
        }

    @classmethod
    def _dual_policy_creator_payload(
        cls,
        *,
        capability: str,
        parent: StaticBankArtifact,
        templates: Sequence[object],
        feedback_bundle: Mapping[str, object],
        feedback_bundle_sha256: str,
        requirements: Mapping[str, object],
        output_schema: Mapping[str, object],
        prior_experiment_memory: Sequence[Mapping[str, object]] = (),
    ) -> dict[str, object]:
        payload = cls._fanout_creator_payload(
            capabilities=(capability,),
            parent=parent,
            templates=templates,
            feedback_bundle=feedback_bundle,
            feedback_bundle_sha256=feedback_bundle_sha256,
            requirements=requirements,
            output_schema=output_schema,
            prior_experiment_memory=prior_experiment_memory,
        )
        grouped = feedback_bundle["feedback_by_capability"]
        assert isinstance(grouped, dict)
        rows = [item for item in grouped.get(capability, []) if isinstance(item, dict)]
        suggestions = feedback_bundle["policy_compatible_suggestions"]
        protected_suggestions = feedback_bundle["protected_success_suggestions"]
        assert isinstance(suggestions, list) and isinstance(protected_suggestions, list)

        def components(row: Mapping[str, object]) -> Mapping[str, object]:
            observation = row.get("baseline_observation")
            if isinstance(observation, dict) and isinstance(
                observation.get("gcs_components"), dict
            ):
                return observation["gcs_components"]
            projected = row.get("gcs_components")
            return projected if isinstance(projected, dict) else {}

        action_failures = []
        response_failures = []
        action_protected = []
        response_protected = []
        for row in rows:
            item_components = components(row)
            route_ok = bool(item_components.get("route_acceptable"))
            no_hard = bool(item_components.get("no_hard_error"))
            tool_ok = bool(item_components.get("tool_contract_pass"))
            evidence_ok = bool(item_components.get("evidence_grounded"))
            output_ok = bool(item_components.get("output_contract_pass"))
            # Attribute only the action/tool loop here. Response-format
            # failures belong to the response-policy screen.
            action_ok = route_ok and tool_ok
            response_ok = no_hard and evidence_ok and output_ok
            if row.get("sample_role") == "failure" and route_ok and not action_ok:
                action_failures.append(row)
            if (
                row.get("sample_role") == "failure"
                and route_ok
                and tool_ok
                and not response_ok
            ):
                response_failures.append(row)
            if action_ok:
                action_protected.append(row)
            if route_ok and tool_ok and response_ok:
                response_protected.append(row)
        payload["operation"] = "s1_dual_policy_creator"
        payload["fanout_policy"] = "six-capability-dual-policy-fanout-fanin-v3"
        payload["policy_surface_evidence"] = {
            "action-policy": {
                "failure_examples": action_failures,
                "protected_success_examples": action_protected,
                "allowed_change": (
                    "tool-first choice, public argument construction, continuation, "
                    "retry, and stopping conditions only"
                ),
                "forbidden_change": (
                    "evidence wording, product cards, answer sections, and fallback text"
                ),
                "failure_repair_suggestions": [
                    item
                    for item in suggestions
                    if isinstance(item, dict)
                    and item.get("capability") == capability
                    and item.get("surface") == "action-policy"
                ],
                "protected_success_suggestions": [
                    item
                    for item in protected_suggestions
                    if isinstance(item, dict)
                    and item.get("capability") == capability
                    and item.get("surface") == "action-policy"
                ],
            },
            "response-policy": {
                "failure_examples": response_failures,
                "protected_success_examples": response_protected,
                "allowed_change": (
                    "visible-evidence claim selection, public cards, answer sections, "
                    "uncertainty, and fallback branch only"
                ),
                "forbidden_change": (
                    "tool choice, tool arguments, tool order, retry, and stopping"
                ),
                "failure_repair_suggestions": [
                    item
                    for item in suggestions
                    if isinstance(item, dict)
                    and item.get("capability") == capability
                    and item.get("surface") == "response-policy"
                ],
                "protected_success_suggestions": [
                    item
                    for item in protected_suggestions
                    if isinstance(item, dict)
                    and item.get("capability") == capability
                    and item.get("surface") == "response-policy"
                ],
            },
        }
        return payload

    def _run_s1_fanout_fanin(
        self,
        *,
        parent: StaticBankArtifact,
        opt: Mapping[str, AssistantObservation],
        query_by_id: Mapping[str, Query],
        feedback_bundle: Mapping[str, object],
        feedback_bundle_sha256: str,
        patchable_capabilities: frozenset[str],
        metrics: dict[str, object],
    ) -> StageDecision:
        """Run isolated capability branches, then combine only passed patches."""

        settings = self.spec.s1_settings
        fold_roles = self.opt_fold_roles()
        replay_queries = tuple(
            query
            for query in self.queries()
            if query.split == "opt_pool" and fold_roles[query.query_id] == "replay"
        )
        static_replay = {
            query.query_id: opt[query.query_id] for query in replay_queries
        }
        templates = decode_sparse_parent_content(parent, self.s1_authoring_input())
        base_requirements = {
            "round_id": settings.round_id,
            "model_generated_body_only": True,
            "model_owns_tool_choice_and_final_response": True,
            "response_compiler": "disabled",
            "complete_six_capability_actions": True,
            "default_action": "inherit",
            "freeze_all_descriptions": True,
            "max_patched_capabilities": 1,
            "creator_directives": list(settings.creator_directives),
            "author_content_lexical_guard": sparse_author_content_lexical_guard(),
            "single_candidate": True,
            "bounded_edit_surface": settings.bounded_edit_surface,
            "prior_experiment_memory_is_diagnostic_not_parent": bool(
                settings.prior_experiment_memory
            ),
        }
        branch_records: list[dict[str, object]] = []
        passed: list[
            tuple[str, StaticBankArtifact, SparseCompilationReceiptV1, str]
        ] = []
        fanin_parent_evidence = dict(static_replay)
        fanin_candidate_evidence = dict(static_replay)
        fanin_treatment_query_ids: set[str] = set()
        metrics["fanout_policy"] = "six-capability-fanout-fanin-v2"
        metrics["replay_accessed"] = False

        creator_targets = tuple(
            capability
            for capability in CAPABILITIES
            if capability in settings.target_capabilities
            and capability in patchable_capabilities
        )
        creator_call_count = 0
        metrics["fanout_creator_targets"] = list(creator_targets)

        for capability in CAPABILITIES:
            branch_id = _safe_id(capability)
            prefix = f"s1-branch-{branch_id}"
            record: dict[str, object] = {
                "capability": capability,
                "status": "not_patchable",
                "creator_called": False,
                "smoke_accessed": False,
                "replay_accessed": False,
                "retained": False,
            }
            if capability not in settings.target_capabilities:
                record["reason"] = "capability_not_selected_for_body_patch"
                branch_records.append(record)
                self._write_canonical_resume_artifact(
                    self.output_root / "banks" / f"{prefix}-decision.json",
                    record,
                    label=f"S1 fan-out {capability} decision",
                )
                continue
            if capability not in patchable_capabilities:
                record["status"] = "no_policy_compatible_feedback"
                record["reason"] = "no_policy_compatible_feedback"
                branch_records.append(record)
                self._write_canonical_resume_artifact(
                    self.output_root / "banks" / f"{prefix}-decision.json",
                    record,
                    label=f"S1 fan-out {capability} decision",
                )
                continue

            requirements = {
                **base_requirements,
                "target_capabilities": [capability],
                "branch_capability_must_patch": capability,
                "each_capability_is_an_independent_creator_session": True,
                "cross_capability_tradeoffs_are_forbidden": True,
                "required_patch_phrases": {
                    capability: list(
                        settings.required_patch_phrases.get(capability, ())
                    )
                },
            }
            creator = self._call(
                role="creator",
                call_id=f"s1-creator-{branch_id}",
                purpose=f"S1 isolated model-generated Body Creator for {capability}",
                payload=self._fanout_creator_payload(
                    capabilities=(capability,),
                    parent=parent,
                    templates=templates,
                    feedback_bundle=feedback_bundle,
                    feedback_bundle_sha256=feedback_bundle_sha256,
                    requirements=requirements,
                    output_schema=self._creator_schema(
                        "s1",
                        parent=parent,
                        s1_patch_capabilities=(capability,),
                    ),
                    prior_experiment_memory=settings.prior_experiment_memory.get(
                        capability, ()
                    ),
                ),
            )
            creator_call_count += 1
            record["creator_called"] = True
            record["creator_call_id"] = creator.call_id
            record["creator_input_tokens"] = creator.input_tokens
            try:
                branch = self._candidate_bank(
                    creator,
                    stage="s1",
                    parent=parent,
                    feedback_bundle_sha256=feedback_bundle_sha256,
                    feedback_patchable_capabilities=frozenset({capability}),
                    s1_expected_patch_capabilities=frozenset({capability}),
                    s1_artifact_prefix=prefix,
                )
            except _S1CandidateRejected as error:
                branch = None
                record["reason"] = error.reason_code
            if branch is None:
                record["status"] = "candidate_rejected"
                record.setdefault("reason", "provider_or_schema_invalid")
                branch_records.append(record)
                self._write_canonical_resume_artifact(
                    self.output_root / "banks" / f"{prefix}-decision.json",
                    record,
                    label=f"S1 fan-out {capability} decision",
                )
                continue

            # Each branch sees only its four capability-local smoke rows. The
            # other five Skills are byte-exact inherited and cannot change in
            # this branch, so rerunning their rows would add stochastic noise
            # and provider cost without testing the candidate treatment.
            smoke_queries = tuple(
                query_by_id[item.query_id]
                for item in self.spec.fixed_samples.dev_smoke24
                if item.capability == capability
            )
            smoke = self._assistant_many(
                split=f"dev-smoke24-{branch_id}",
                config=f"s1-branch-{branch_id}",
                queries=smoke_queries,
                bank=branch,
            )
            record["smoke_accessed"] = True
            oracle_failures = sorted(
                query_id for query_id, row in smoke.items() if not row.oracle_available
            )
            record["smoke_oracle_coverage_failure_query_ids"] = oracle_failures
            if oracle_failures:
                record["status"] = "smoke_failed"
                record["reason"] = "smoke_oracle_coverage_incomplete"
                branch_records.append(record)
                self._write_canonical_resume_artifact(
                    self.output_root / "banks" / f"{prefix}-decision.json",
                    record,
                    label=f"S1 fan-out {capability} decision",
                )
                continue

            target_population = tuple(
                query
                for query in replay_queries
                if query.canonical_capability == capability
            )
            target_queries = tuple(
                query
                for query in target_population
                if static_replay[query.query_id].selected_capability == capability
                and self._body_replay_parent_eligible(static_replay[query.query_id])
            )
            record["replay_population_count"] = len(target_population)
            record["treatment_reached_count"] = len(target_queries)
            record["aliased_parent_query_ids"] = [
                query.query_id
                for query in target_population
                if query not in target_queries
            ]
            if not target_queries:
                record["status"] = "screen_rejected"
                record["reason"] = "no_treatment_reached_replay_rows"
                branch_records.append(record)
                self._write_canonical_resume_artifact(
                    self.output_root / "banks" / f"{prefix}-decision.json",
                    record,
                    label=f"S1 fan-out {capability} decision",
                )
                continue
            target_static = {
                query.query_id: static_replay[query.query_id]
                for query in target_queries
            }
            target_parent = self._assistant_many(
                split=f"opt-replay-{branch_id}",
                config=f"s1-branch-{branch_id}-parent-control-a",
                queries=target_queries,
                bank=parent,
                reuse_parent=target_static,
            )
            target_candidate = self._assistant_many(
                split=f"opt-replay-{branch_id}",
                config=f"s1-branch-{branch_id}-candidate-a",
                queries=target_queries,
                bank=branch,
                reuse_parent=target_static,
            )
            confirmation_queries = tuple(
                query
                for query in target_queries
                if (
                    target_parent[query.query_id].gcs_score
                    != target_candidate[query.query_id].gcs_score
                    or target_parent[query.query_id].hard_error
                    != target_candidate[query.query_id].hard_error
                )
            )
            confirmation_static = {
                query.query_id: target_static[query.query_id]
                for query in confirmation_queries
            }
            target_parent_b = self._assistant_many(
                split=f"opt-replay-{branch_id}",
                config=f"s1-branch-{branch_id}-parent-control-b",
                queries=confirmation_queries,
                bank=parent,
                reuse_parent=confirmation_static,
            )
            target_candidate_b = self._assistant_many(
                split=f"opt-replay-{branch_id}",
                config=f"s1-branch-{branch_id}-candidate-b",
                queries=confirmation_queries,
                bank=branch,
                reuse_parent=confirmation_static,
            )
            confirmation_query_ids = frozenset(
                query.query_id for query in confirmation_queries
            )
            stable_query_ids = frozenset(
                query.query_id
                for query in target_queries
                if query.query_id not in confirmation_query_ids
                or (
                    target_parent[query.query_id].gcs_score
                    == target_parent_b[query.query_id].gcs_score
                    and target_parent[query.query_id].hard_error
                    == target_parent_b[query.query_id].hard_error
                    and target_candidate[query.query_id].gcs_score
                    == target_candidate_b[query.query_id].gcs_score
                    and target_candidate[query.query_id].hard_error
                    == target_candidate_b[query.query_id].hard_error
                )
            )
            record["adaptive_confirmation_count"] = len(confirmation_queries)
            stochastic_query_ids = sorted(
                query.query_id
                for query in target_queries
                if query.query_id not in stable_query_ids
            )
            record["rollout_stochastic_query_ids"] = stochastic_query_ids
            record["stable_treatment_count"] = len(stable_query_ids)
            stable_queries = tuple(
                query for query in target_queries if query.query_id in stable_query_ids
            )
            stable_parent = {
                query.query_id: target_parent[query.query_id]
                for query in stable_queries
            }
            stable_candidate = {
                query.query_id: target_candidate[query.query_id]
                for query in stable_queries
            }
            metrics["replay_accessed"] = True
            record["replay_accessed"] = True
            oracle_failures = sorted(
                query_id
                for rows in (
                    target_parent,
                    target_parent_b,
                    target_candidate,
                    target_candidate_b,
                )
                for query_id, row in rows.items()
                if not row.oracle_available
            )
            if oracle_failures:
                record["status"] = "replay_failed"
                record["reason"] = "replay_oracle_coverage_incomplete"
                record["replay_oracle_coverage_failure_query_ids"] = oracle_failures
                branch_records.append(record)
                self._write_canonical_resume_artifact(
                    self.output_root / "banks" / f"{prefix}-decision.json",
                    record,
                    label=f"S1 fan-out {capability} decision",
                )
                continue
            screen = self._s1_capability_screen(
                capability=capability,
                queries=stable_queries,
                baseline=stable_parent,
                candidate=stable_candidate,
            )
            record["screen"] = screen
            record["status"] = (
                "passed" if screen["decision"] == "retain_patch" else "screen_rejected"
            )
            record["retained"] = screen["decision"] == "retain_patch"
            if record["retained"]:
                receipt_path = (
                    self.output_root
                    / "banks"
                    / f"{prefix}-sparse-compilation-receipt.json"
                )
                try:
                    receipt = load_sparse_compilation_receipt(
                        receipt_path,
                        expected_file_sha256=_file_sha(receipt_path),
                    )
                except (OSError, S1SparsePatchError) as error:
                    raise FastPathError(
                        f"S1 fan-out {capability} compilation receipt is invalid"
                    ) from error
                screen_sha = sha256_bytes(canonical_json_bytes(screen))
                passed.append((capability, branch, receipt, screen_sha))
                fanin_parent_evidence.update(stable_parent)
                fanin_candidate_evidence.update(stable_candidate)
                fanin_treatment_query_ids.update(stable_query_ids)
            branch_records.append(record)
            self._write_canonical_resume_artifact(
                self.output_root / "banks" / f"{prefix}-decision.json",
                record,
                label=f"S1 fan-out {capability} decision",
            )

        metrics["fanout_creator_call_count"] = creator_call_count
        metrics["fanout_branches"] = branch_records
        retained = tuple(sorted(item[0] for item in passed))
        metrics["retained_patch_capabilities"] = list(retained)
        metrics["reverted_patch_capabilities"] = [
            item for item in settings.target_capabilities if item not in set(retained)
        ]
        if not passed:
            reasons = ("fan-out screening retained no capability branch",)
            self._write_selected_bank("s1", parent)
            return self._save_decision(
                StageDecision(
                    stage="s1",
                    accepted=False,
                    alias_of="llm_static",
                    parent_bank=parent.bank_sha256,
                    candidate_bank=None,
                    selected_bank=parent.bank_sha256,
                    reasons=reasons,
                    metrics=metrics,
                )
            )

        combined, fanin_steps = self._compose_fanout_banks(parent=parent, passed=passed)
        self._write_canonical_resume_artifact(
            self.output_root / "banks" / "s1-fanin-candidate.json",
            combined.model_dump(mode="json"),
            label="S1 fan-in candidate Bank",
        )
        self._write_canonical_resume_artifact(
            self.output_root / "banks" / "s1-fanin-receipt.json",
            {
                "schema_version": 1,
                "policy_version": "six-capability-fanout-fanin-v2",
                "parent_bank_sha256": parent.bank_sha256,
                "retained_capabilities": list(retained),
                "steps": fanin_steps,
                "combined_bank_sha256": combined.bank_sha256,
            },
            label="S1 fan-in receipt",
        )
        metrics["fanin_steps"] = fanin_steps
        metrics["screened_candidate_bank"] = combined.bank_sha256

        metrics["fanin_replay_population_count"] = len(replay_queries)
        metrics["fanin_treatment_reached_count"] = len(fanin_treatment_query_ids)
        metrics["fanin_aliased_parent_query_ids"] = [
            query.query_id
            for query in replay_queries
            if query.query_id not in fanin_treatment_query_ids
        ]
        replay_ok, replay_reasons, replay_metrics = self._s1_gate(
            fanin_parent_evidence,
            fanin_candidate_evidence,
            replay_queries,
            phase="replay200",
            treated_capabilities=frozenset(retained),
        )
        metrics["replay_gate"] = replay_metrics
        reasons = [f"replay200: {item}" for item in replay_reasons]
        accepted = False
        if replay_ok:
            gate_queries = self._queries_for_val_gate("body_gate")
            metrics["body_accessed"] = True
            static_rows = self._assistant_many(
                split="body-gate75",
                config="llm_static",
                queries=gate_queries,
                bank=parent,
            )
            treated_gate_queries = tuple(
                query
                for query in gate_queries
                if query.canonical_capability in set(retained)
                and static_rows[query.query_id].selected_capability
                == query.canonical_capability
                and self._body_replay_parent_eligible(static_rows[query.query_id])
            )
            treated_static_rows = {
                query.query_id: static_rows[query.query_id]
                for query in treated_gate_queries
            }
            parent_rows_a = self._assistant_many(
                split="body-gate75",
                config="s1-branch-body-parent-control-a",
                queries=treated_gate_queries,
                bank=parent,
                reuse_parent=treated_static_rows,
            )
            candidate_rows_a = self._assistant_many(
                split="body-gate75",
                config="s1-branch-body-candidate-a",
                queries=treated_gate_queries,
                bank=combined,
                reuse_parent=treated_static_rows,
            )
            confirmation_gate_queries = tuple(
                query
                for query in treated_gate_queries
                if (
                    parent_rows_a[query.query_id].gcs_score
                    != candidate_rows_a[query.query_id].gcs_score
                    or parent_rows_a[query.query_id].hard_error
                    != candidate_rows_a[query.query_id].hard_error
                )
            )
            confirmation_gate_static = {
                query.query_id: treated_static_rows[query.query_id]
                for query in confirmation_gate_queries
            }
            parent_rows_b = self._assistant_many(
                split="body-gate75",
                config="s1-branch-body-parent-control-b",
                queries=confirmation_gate_queries,
                bank=parent,
                reuse_parent=confirmation_gate_static,
            )
            candidate_rows_b = self._assistant_many(
                split="body-gate75",
                config="s1-branch-body-candidate-b",
                queries=confirmation_gate_queries,
                bank=combined,
                reuse_parent=confirmation_gate_static,
            )
            confirmation_gate_query_ids = frozenset(
                query.query_id for query in confirmation_gate_queries
            )
            stable_gate_query_ids = frozenset(
                query.query_id
                for query in treated_gate_queries
                if query.query_id not in confirmation_gate_query_ids
                or (
                    parent_rows_a[query.query_id].gcs_score
                    == parent_rows_b[query.query_id].gcs_score
                    and parent_rows_a[query.query_id].hard_error
                    == parent_rows_b[query.query_id].hard_error
                    and candidate_rows_a[query.query_id].gcs_score
                    == candidate_rows_b[query.query_id].gcs_score
                    and candidate_rows_a[query.query_id].hard_error
                    == candidate_rows_b[query.query_id].hard_error
                )
            )
            metrics["body_adaptive_confirmation_count"] = len(confirmation_gate_queries)
            evaluated_parent_gate = dict(static_rows)
            evaluated_candidate_gate = dict(static_rows)
            evaluated_parent_gate.update(
                {
                    query_id: parent_rows_a[query_id]
                    for query_id in stable_gate_query_ids
                }
            )
            evaluated_candidate_gate.update(
                {
                    query_id: candidate_rows_a[query_id]
                    for query_id in stable_gate_query_ids
                }
            )
            metrics["body_treatment_reached_count"] = len(stable_gate_query_ids)
            metrics["body_rollout_stochastic_query_ids"] = sorted(
                query.query_id
                for query in treated_gate_queries
                if query.query_id not in stable_gate_query_ids
            )
            accepted, gate_reasons, gate_metrics = self._s1_gate(
                evaluated_parent_gate,
                evaluated_candidate_gate,
                gate_queries,
                phase="body_gate75",
                treated_capabilities=frozenset(retained),
            )
            reasons.extend(gate_reasons)
            metrics["gate"] = gate_metrics
        selected = combined if accepted else parent
        self._write_selected_bank("s1", selected)
        return self._save_decision(
            StageDecision(
                stage="s1",
                accepted=accepted,
                alias_of=None if accepted else "llm_static",
                parent_bank=parent.bank_sha256,
                candidate_bank=combined.bank_sha256,
                selected_bank=selected.bank_sha256,
                reasons=tuple(reasons),
                metrics=metrics,
            )
        )

    def _run_s1_dual_policy_fanout_fanin(
        self,
        *,
        parent: StaticBankArtifact,
        opt: Mapping[str, AssistantObservation],
        query_by_id: Mapping[str, Query],
        feedback_bundle: Mapping[str, object],
        feedback_bundle_sha256: str,
        patchable_capabilities: frozenset[str],
        metrics: dict[str, object],
    ) -> StageDecision:
        """Screen action and response treatments independently, then compose."""

        settings = self.spec.s1_settings
        fold_roles = self.opt_fold_roles()
        replay_queries = tuple(
            query
            for query in self.queries()
            if query.split == "opt_pool" and fold_roles[query.query_id] == "replay"
        )
        static_replay = {
            query.query_id: opt[query.query_id] for query in replay_queries
        }
        templates = decode_sparse_parent_content(parent, self.s1_authoring_input())
        parent_by_capability = _bank_by_capability(parent)
        branch_records: list[dict[str, object]] = []
        composed_capability_branches: list[
            tuple[str, StaticBankArtifact, PolicySurfaceCompositionReceiptV1, str]
        ] = []
        creator_call_count = 0
        metrics.update(
            {
                "fanout_policy": "six-capability-dual-policy-fanout-fanin-v3",
                "policy_surfaces": ["action-policy", "response-policy"],
                "replay_accessed": False,
            }
        )

        for capability in CAPABILITIES:
            capability_record: dict[str, object] = {
                "capability": capability,
                "creator_called": False,
                "surface_branches": [],
                "retained_surfaces": [],
            }
            if capability not in settings.target_capabilities or capability not in (
                patchable_capabilities
            ):
                capability_record["status"] = "not_patchable"
                branch_records.append(capability_record)
                continue
            branch_id = _safe_id(capability)
            prefix = f"s1-dual-{branch_id}"
            requirements = {
                "round_id": settings.round_id,
                "target_capabilities": list(settings.target_capabilities),
                "dual_policy_capability": capability,
                "one_creator_two_independent_surfaces": True,
                "freeze_all_descriptions": True,
                "action_policy_scope": (
                    "tool-first choice, public arguments, continuation, and stop"
                ),
                "response_policy_scope": (
                    "visible evidence, cards, answer, uncertainty, and fallback"
                ),
                "surface_default_action": "inherit",
                "surface_patches_must_not_cross_scope": True,
                "creator_directives": list(settings.creator_directives),
                "author_content_lexical_guard": sparse_author_content_lexical_guard(),
            }
            creator = self._call(
                role="creator",
                call_id=f"{prefix}-creator-once",
                purpose=f"S1 dual-policy Creator for {capability}",
                payload=self._dual_policy_creator_payload(
                    capability=capability,
                    parent=parent,
                    templates=templates,
                    feedback_bundle=feedback_bundle,
                    feedback_bundle_sha256=feedback_bundle_sha256,
                    requirements=requirements,
                    output_schema=self._creator_schema(
                        "s1",
                        parent=parent,
                        dual_policy_capability=capability,
                    ),
                    prior_experiment_memory=settings.prior_experiment_memory.get(
                        capability, ()
                    ),
                ),
            )
            creator_call_count += 1
            capability_record["creator_called"] = True
            capability_record["creator_call_id"] = creator.call_id
            try:
                if (
                    creator.status != "success"
                    or not creator.schema_valid
                    or creator.output is None
                ):
                    raise S1SparsePatchError("dual-policy Creator result is invalid")
                proposal = parse_dual_policy_patch(
                    canonical_json_bytes(creator.output),
                    capability_id=capability,
                    parent_skill_sha256=parent_by_capability[capability].skill_sha256,
                )
            except S1SparsePatchError:
                capability_record["status"] = "candidate_rejected"
                capability_record["reason"] = "dual_policy_contract_rejected"
                branch_records.append(capability_record)
                continue

            retained: list[CompiledPolicySurfaceBranch] = []
            for surface in ("action-policy", "response-policy"):
                compiled = compile_policy_surface_branch(
                    parent_bank=parent,
                    proposal=proposal,
                    surface=surface,
                )
                surface_record: dict[str, object] = {
                    "surface": surface,
                    "status": "inherited",
                    "retained": False,
                }
                if compiled is None:
                    capability_record["surface_branches"].append(surface_record)  # type: ignore[union-attr]
                    continue
                artifact_name = f"{prefix}-{surface}"
                self._write_canonical_resume_artifact(
                    self.output_root / "banks" / f"{artifact_name}-candidate.json",
                    compiled.bank.model_dump(mode="json"),
                    label=f"S1 {capability} {surface} candidate",
                )
                self._write_canonical_resume_artifact(
                    self.output_root / "banks" / f"{artifact_name}-receipt.json",
                    compiled.receipt.model_dump(mode="json"),
                    label=f"S1 {capability} {surface} receipt",
                )
                eligible = (
                    self._action_replay_parent_eligible
                    if surface == "action-policy"
                    else self._body_replay_parent_eligible
                )
                target_queries = tuple(
                    query
                    for query in replay_queries
                    if query.canonical_capability == capability
                    and static_replay[query.query_id].selected_capability == capability
                    and eligible(static_replay[query.query_id])
                )
                surface_record["population_count"] = len(target_queries)
                if not target_queries:
                    surface_record.update(
                        {"status": "screen_rejected", "reason": "no_eligible_rows"}
                    )
                    capability_record["surface_branches"].append(surface_record)  # type: ignore[union-attr]
                    continue
                frozen = {
                    query.query_id: static_replay[query.query_id]
                    for query in target_queries
                }
                parent_rows = self._assistant_many(
                    split=f"opt-replay-{artifact_name}",
                    config=f"s1-branch-{branch_id}-{surface}-parent",
                    queries=target_queries,
                    bank=parent,
                    reuse_parent=frozen,
                    replay_surface=surface,
                )
                candidate_rows = self._assistant_many(
                    split=f"opt-replay-{artifact_name}",
                    config=f"s1-branch-{branch_id}-{surface}-candidate",
                    queries=target_queries,
                    bank=compiled.bank,
                    reuse_parent=frozen,
                    replay_surface=surface,
                )
                metrics["replay_accessed"] = True
                oracle_failures = sorted(
                    query_id
                    for rows in (parent_rows, candidate_rows)
                    for query_id, row in rows.items()
                    if not row.oracle_available
                )
                if oracle_failures:
                    surface_record.update(
                        {
                            "status": "replay_failed",
                            "reason": "oracle_coverage_incomplete",
                            "oracle_failure_query_ids": oracle_failures,
                        }
                    )
                else:
                    screen = self._s1_policy_screen(
                        capability=capability,
                        surface=surface,
                        queries=target_queries,
                        baseline=parent_rows,
                        candidate=candidate_rows,
                    )
                    surface_record["screen"] = screen
                    surface_record["retained"] = screen["decision"] == "retain_patch"
                    surface_record["status"] = (
                        "passed" if surface_record["retained"] else "screen_rejected"
                    )
                    if surface_record["retained"]:
                        retained.append(compiled)
                capability_record["surface_branches"].append(surface_record)  # type: ignore[union-attr]

            capability_record["retained_surfaces"] = [
                item.receipt.surface for item in retained
            ]
            if retained:
                composed = compose_policy_surface_branches(
                    parent_bank=parent,
                    capability_id=capability,
                    branches=retained,
                )
                self._write_canonical_resume_artifact(
                    self.output_root
                    / "banks"
                    / f"{prefix}-dual-policy-composition-receipt.json",
                    composed.receipt.model_dump(mode="json"),
                    label=f"S1 {capability} dual-policy composition receipt",
                )
                screen_sha = sha256_bytes(
                    canonical_json_bytes(capability_record["surface_branches"])
                )
                composed_capability_branches.append(
                    (capability, composed.bank, composed.receipt, screen_sha)
                )
                capability_record["status"] = "passed"
                capability_record["combined_bank_sha256"] = composed.bank.bank_sha256
                capability_record["composition_receipt_sha256"] = (
                    composed.receipt.receipt_sha256
                )
            else:
                capability_record["status"] = "screen_rejected"
            branch_records.append(capability_record)

        metrics["fanout_creator_call_count"] = creator_call_count
        metrics["fanout_branches"] = branch_records
        retained_caps = tuple(item[0] for item in composed_capability_branches)
        metrics["retained_patch_capabilities"] = list(retained_caps)
        if not composed_capability_branches:
            self._write_selected_bank("s1", parent)
            return self._save_decision(
                StageDecision(
                    stage="s1",
                    accepted=False,
                    alias_of="llm_static",
                    parent_bank=parent.bank_sha256,
                    candidate_bank=None,
                    selected_bank=parent.bank_sha256,
                    reasons=("dual-policy screening retained no branch",),
                    metrics=metrics,
                )
            )
        combined, fanin_steps = self._compose_fanout_banks(
            parent=parent,
            passed=composed_capability_branches,
        )
        self._write_canonical_resume_artifact(
            self.output_root / "banks" / "s1-dual-policy-fanin-candidate.json",
            combined.model_dump(mode="json"),
            label="S1 dual-policy fan-in candidate Bank",
        )
        self._write_canonical_resume_artifact(
            self.output_root / "banks" / "s1-dual-policy-fanin-receipt.json",
            {
                "policy_version": "six-capability-dual-policy-fanout-fanin-v3",
                "parent_bank_sha256": parent.bank_sha256,
                "candidate_bank_sha256": combined.bank_sha256,
                "retained_capabilities": list(retained_caps),
                "steps": fanin_steps,
            },
            label="S1 dual-policy fan-in receipt",
        )
        metrics["fanin_steps"] = fanin_steps
        metrics["screened_candidate_bank"] = combined.bank_sha256

        # The two local screens establish attribution.  The combined candidate
        # must then survive the ordinary full action/tool replay and body gate;
        # no local evidence is reused as the global acceptance result.
        parent_replay = self._assistant_many(
            split="opt-replay-dual-policy-fanin",
            config="s1-branch-dual-fanin-parent",
            queries=replay_queries,
            bank=parent,
        )
        candidate_replay = self._assistant_many(
            split="opt-replay-dual-policy-fanin",
            config="s1-branch-dual-fanin-candidate",
            queries=replay_queries,
            bank=combined,
        )
        replay_ok, replay_reasons, replay_metrics = self._s1_gate(
            parent_replay,
            candidate_replay,
            replay_queries,
            phase="replay200",
            treated_capabilities=frozenset(retained_caps),
        )
        metrics["replay_gate"] = replay_metrics
        reasons = [f"replay200: {item}" for item in replay_reasons]
        accepted = False
        if replay_ok:
            metrics["body_accessed"] = True
            gate_queries = self._queries_for_val_gate("body_gate")
            parent_gate = self._assistant_many(
                split="body-gate75-dual-policy",
                config="s1-branch-dual-body-parent",
                queries=gate_queries,
                bank=parent,
            )
            candidate_gate = self._assistant_many(
                split="body-gate75-dual-policy",
                config="s1-branch-dual-body-candidate",
                queries=gate_queries,
                bank=combined,
            )
            accepted, gate_reasons, gate_metrics = self._s1_gate(
                parent_gate,
                candidate_gate,
                gate_queries,
                phase="body_gate75",
                treated_capabilities=frozenset(retained_caps),
            )
            reasons.extend(gate_reasons)
            metrics["gate"] = gate_metrics
        selected = combined if accepted else parent
        self._write_selected_bank("s1", selected)
        return self._save_decision(
            StageDecision(
                stage="s1",
                accepted=accepted,
                alias_of=None if accepted else "llm_static",
                parent_bank=parent.bank_sha256,
                candidate_bank=combined.bank_sha256,
                selected_bank=selected.bank_sha256,
                reasons=tuple(reasons),
                metrics=metrics,
            )
        )

    @staticmethod
    def _compose_fanout_banks(
        *,
        parent: StaticBankArtifact,
        passed: Sequence[
            tuple[
                str,
                StaticBankArtifact,
                SparseCompilationReceiptV1 | PolicySurfaceCompositionReceiptV1,
                str,
            ]
        ],
    ) -> tuple[StaticBankArtifact, list[dict[str, object]]]:
        """Compose independently parent-bound branches into one Bank."""

        parent_by_capability = _bank_by_capability(parent)
        selected = dict(parent_by_capability)
        steps: list[dict[str, object]] = []
        immutable_parent = parent.model_dump(
            mode="json",
            exclude={"construction_identity_sha256", "skills", "bank_sha256"},
        )
        for capability, branch, receipt, screen_sha in sorted(
            passed, key=lambda item: item[0]
        ):
            branch_by_capability = _bank_by_capability(branch)
            immutable_branch = branch.model_dump(
                mode="json",
                exclude={"construction_identity_sha256", "skills", "bank_sha256"},
            )
            receipt_lineage_ok = (
                receipt.parent_bank_sha256 == parent.bank_sha256
                and receipt.candidate_bank_sha256 == branch.bank_sha256
            )
            if isinstance(receipt, SparseCompilationReceiptV1):
                binding_by_capability = {
                    item.capability_id: item for item in receipt.bindings
                }
                receipt_lineage_ok = receipt_lineage_ok and (
                    set(binding_by_capability) == set(parent_by_capability)
                    and binding_by_capability[capability].action == "patch"
                )
            else:
                receipt_lineage_ok = receipt_lineage_ok and (
                    receipt.capability_id == capability
                )
            if (
                immutable_branch != immutable_parent
                or not receipt_lineage_ok
                or set(branch_by_capability) != set(parent_by_capability)
            ):
                raise FastPathError(
                    f"S1 fan-out branch lineage differs at {capability}"
                )
            for other in CAPABILITIES:
                parent_bytes = canonical_json_bytes(
                    parent_by_capability[other].model_dump(mode="json")
                )
                branch_bytes = canonical_json_bytes(
                    branch_by_capability[other].model_dump(mode="json")
                )
                if other != capability and branch_bytes != parent_bytes:
                    raise FastPathError(
                        f"S1 fan-out branch changed another capability: {capability}"
                    )
            selected[capability] = branch_by_capability[capability]
            steps.append(
                {
                    "capability": capability,
                    "branch_bank_sha256": branch.bank_sha256,
                    "compilation_receipt_sha256": receipt.receipt_sha256,
                    "screen_sha256": screen_sha,
                    "selected_skill_sha256": branch_by_capability[
                        capability
                    ].skill_sha256,
                }
            )
        policy_version = (
            "six-capability-dual-policy-fanout-fanin-v3"
            if any(
                isinstance(item[2], PolicySurfaceCompositionReceiptV1)
                for item in passed
            )
            else "six-capability-fanout-fanin-v2"
        )
        construction_identity = sha256_bytes(
            canonical_json_bytes(
                {
                    "policy_version": policy_version,
                    "parent_bank_sha256": parent.bank_sha256,
                    "steps": steps,
                }
            )
        )
        payload = parent.model_dump(mode="json")
        payload["construction_identity_sha256"] = construction_identity
        payload["skills"] = [
            selected[item.capability_id].model_dump(mode="json")
            for item in parent.skills
        ]
        payload.pop("bank_sha256", None)
        payload["bank_sha256"] = sha256_bytes(canonical_json_bytes(payload))
        try:
            bank = StaticBankArtifact.model_validate(payload, strict=True)
        except ValidationError as error:
            raise FastPathError("S1 fan-in Bank is invalid") from error
        return bank, steps

    def _run_s1_counterfactual_branch(
        self,
        *,
        parent: StaticBankArtifact,
        opt: Mapping[str, AssistantObservation],
        query_by_id: Mapping[str, Query],
        feedback_bundle: Mapping[str, object],
        feedback_bundle_sha256: str,
        patchable_capabilities: frozenset[str],
        metrics: dict[str, object],
    ) -> StageDecision:
        settings = self.spec.s1_settings
        binding = self.spec.s1_parent
        assert binding is not None and settings.target_surface is not None
        target = settings.target_capabilities[0]
        parent_by_capability = _bank_by_capability(parent)
        grouped = feedback_bundle["feedback_by_capability"]
        assert isinstance(grouped, dict)
        evidence_rows = grouped[target]
        assert isinstance(evidence_rows, list)
        parent_success_ids = tuple(
            sorted(
                str(row["query_id"])
                for row in evidence_rows
                if row.get("sample_role") == "parent_success"
            )
        )
        expected_action_condition: Mapping[str, object] | None = None
        expected_response_signature: Mapping[str, object] | None = None
        if (
            settings.feedback_selection_policy
            in {"parent-counterfactual-v8", "parent-counterfactual-v9"}
            and settings.target_surface == "action-policy"
        ):
            action_conditions = {
                canonical_json_bytes(row.get("action_treatment_signature"))
                for row in evidence_rows
                if row.get("sample_role") == "cluster_failure"
            }
            if (
                len(action_conditions) != 1
                or canonical_json_bytes(None) in action_conditions
            ):
                raise FastPathError(
                    "counterfactual action evidence lacks one bound treatment state"
                )
            raw_condition = next(
                row.get("action_treatment_signature")
                for row in evidence_rows
                if row.get("sample_role") == "cluster_failure"
            )
            if not isinstance(raw_condition, dict):
                raise FastPathError(
                    "counterfactual action treatment state is not an object"
                )
            expected_action_condition = raw_condition
        if (
            settings.feedback_selection_policy == "parent-counterfactual-v9"
            and settings.target_surface == "response-policy"
        ):
            response_signatures = {
                canonical_json_bytes(row.get("response_treatment_signature"))
                for row in evidence_rows
                if row.get("sample_role") == "cluster_failure"
            }
            if (
                len(response_signatures) != 1
                or canonical_json_bytes(None) in response_signatures
            ):
                raise FastPathError(
                    "counterfactual response evidence lacks one bound scored behavior"
                )
            raw_signature = next(
                row.get("response_treatment_signature")
                for row in evidence_rows
                if row.get("sample_role") == "cluster_failure"
            )
            if not isinstance(raw_signature, dict):
                raise FastPathError(
                    "counterfactual response treatment signature is not an object"
                )
            expected_response_signature = raw_signature
        branch_records: list[dict[str, object]] = [
            {
                "capability": capability,
                "status": (
                    "target_pending" if capability == target else "protected_inherit"
                ),
                "surface": settings.target_surface if capability == target else None,
                "creator_called": False,
                "parent_skill_sha256": parent_by_capability[capability].skill_sha256,
                "selected_skill_sha256": parent_by_capability[capability].skill_sha256,
                "inherited_bytes_exact": capability != target,
            }
            for capability in CAPABILITIES
        ]
        metrics.update(
            {
                "cycle_id": settings.cycle_id,
                "parent_round_id": binding.source_round_id,
                "parent_bank_file_sha256": binding.bank_file_sha256,
                "parent_decision_file_sha256": binding.decision_file_sha256,
                "parent_manifest_file_sha256": binding.manifest_file_sha256,
                "fanout_policy": settings.proposal_mode,
                "target_capability": target,
                "target_surface": settings.target_surface,
                "explicit_parent_success_query_ids": list(parent_success_ids),
                "cycle_parent_protection_query_ids": list(
                    binding.parent_protection_query_ids
                ),
                "cycle_parent_protection_disposition": (
                    "byte_exact_protected_skill_and_parent_aliased_rows"
                ),
                "fanout_branches": branch_records,
            }
        )
        if len(parent_success_ids) != 3 or target not in patchable_capabilities:
            branch_records[CAPABILITIES.index(target)].update(
                {"status": "candidate_rejected", "reason": "feedback_not_actionable"}
            )
            self._write_selected_bank("s1", parent)
            return self._save_decision(
                StageDecision(
                    stage="s1",
                    accepted=False,
                    alias_of="s1",
                    parent_bank=parent.bank_sha256,
                    candidate_bank=None,
                    selected_bank=parent.bank_sha256,
                    reasons=(
                        "counterfactual Feedback did not authorize one target rule",
                    ),
                    metrics=metrics,
                )
            )

        creator = self._call(
            role="creator",
            call_id=f"{settings.round_id}-counterfactual-creator-once",
            purpose=(
                f"S1 {settings.round_id} one-rule {target} {settings.target_surface} Creator"
            ),
            payload={
                "operation": "s1_single_surface_counterfactual_creator",
                "cycle_id": settings.cycle_id,
                "round_id": settings.round_id,
                "proposal_mode": settings.proposal_mode,
                "parent_bank_sha256": parent.bank_sha256,
                "parent_bank": parent.model_dump(mode="json"),
                "parent_skill": parent_by_capability[target].model_dump(mode="json"),
                "target_capability": target,
                "target_surface": settings.target_surface,
                "counterfactual_evidence": evidence_rows,
                "policy_compatible_suggestions": feedback_bundle[
                    "policy_compatible_suggestions"
                ],
                "historical_experiment_memory": list(
                    settings.prior_experiment_memory.get(target, ())
                ),
                "requirements": {
                    "one_conditional_rule_only": True,
                    "when_provider_visible_state_only": True,
                    "then_target_surface_only": settings.target_surface,
                    "action_output_is_typed_ir_without_response_text": (
                        settings.proposal_mode
                        in {
                            "single-surface-counterfactual-fanout-v5",
                            "single-surface-counterfactual-fanout-v6",
                        }
                        and settings.target_surface == "action-policy"
                    ),
                    "action_condition_is_bound_to_selected_failure_state": (
                        expected_action_condition
                    ),
                    "action_transition_obeys_capability_tool_order": (
                        settings.proposal_mode
                        in {
                            "single-surface-counterfactual-fanout-v5",
                            "single-surface-counterfactual-fanout-v6",
                        }
                        and settings.target_surface == "action-policy"
                    ),
                    "response_output_is_typed_ir_without_action_or_answer_text": (
                        settings.proposal_mode
                        == "single-surface-counterfactual-fanout-v6"
                        and settings.target_surface == "response-policy"
                    ),
                    "response_behavior_is_bound_to_selected_failure": (
                        expected_response_signature
                    ),
                    "non_target_surface": "inherit",
                    "must_preserve_exactly": list(parent_success_ids),
                    "canonical_compilation": (
                        "If and only if <when>, <then>. Otherwise preserve the parent "
                        "behavior, including <must_preserve>."
                    ),
                    "forbid_checklists_templates_cross_surface_and_evaluation_labels": True,
                },
                "output_schema": self._creator_schema(
                    "s1",
                    parent=parent,
                    counterfactual_surface=settings.target_surface,
                    counterfactual_parent_success_ids=parent_success_ids,
                    counterfactual_action_condition=expected_action_condition,
                    counterfactual_response_signature=expected_response_signature,
                ),
            },
        )
        target_record = branch_records[CAPABILITIES.index(target)]
        target_record["creator_called"] = True
        target_record["creator_call_id"] = creator.call_id
        try:
            if (
                creator.status != "success"
                or not creator.schema_valid
                or creator.output is None
            ):
                raise S1SparsePatchError("counterfactual Creator result is invalid")
            if settings.proposal_mode == "single-surface-counterfactual-fanout-v6":
                semantic_proposal = parse_counterfactual_semantic_policy_patch(
                    canonical_json_bytes(creator.output),
                    capability_id=target,
                    parent_skill_sha256=parent_by_capability[target].skill_sha256,
                    target_surface=settings.target_surface,
                    parent_success_query_ids=parent_success_ids,
                    expected_action_condition=expected_action_condition,
                    expected_response_signature=expected_response_signature,
                )
                compiled = compile_counterfactual_semantic_policy_branch(
                    parent_bank=parent, proposal=semantic_proposal
                )
            elif settings.proposal_mode == "single-surface-counterfactual-fanout-v5":
                typed_proposal = parse_counterfactual_typed_policy_patch(
                    canonical_json_bytes(creator.output),
                    capability_id=target,
                    parent_skill_sha256=parent_by_capability[target].skill_sha256,
                    target_surface=settings.target_surface,
                    parent_success_query_ids=parent_success_ids,
                    expected_action_condition=expected_action_condition,
                )
                compiled = compile_counterfactual_typed_policy_branch(
                    parent_bank=parent, proposal=typed_proposal
                )
            else:
                proposal = parse_counterfactual_policy_patch(
                    canonical_json_bytes(creator.output),
                    capability_id=target,
                    parent_skill_sha256=parent_by_capability[target].skill_sha256,
                    target_surface=settings.target_surface,
                    parent_success_query_ids=parent_success_ids,
                )
                compiled = compile_counterfactual_policy_branch(
                    parent_bank=parent, proposal=proposal
                )
        except S1SparsePatchError:
            target_record.update(
                {
                    "status": "candidate_rejected",
                    "reason": "counterfactual_contract_rejected",
                }
            )
            self._write_selected_bank("s1", parent)
            return self._save_decision(
                StageDecision(
                    stage="s1",
                    accepted=False,
                    alias_of="s1",
                    parent_bank=parent.bank_sha256,
                    candidate_bank=None,
                    selected_bank=parent.bank_sha256,
                    reasons=(
                        "Creator returned an invalid conditional single-surface rule",
                    ),
                    metrics=metrics,
                )
            )

        candidate = compiled.bank
        candidate_by_capability = _bank_by_capability(candidate)
        style_capability = "product.style_recommendation"
        expected_style_sha = binding.protected_skill_sha256.get(style_capability)
        if (
            expected_style_sha is None
            or candidate_by_capability[style_capability].skill_sha256
            != expected_style_sha
        ):
            raise FastPathError("counterfactual candidate changed protected R12 Style")
        for capability in CAPABILITIES:
            if capability == target:
                continue
            if canonical_json_bytes(
                candidate_by_capability[capability].model_dump(mode="json")
            ) != canonical_json_bytes(
                parent_by_capability[capability].model_dump(mode="json")
            ):
                raise FastPathError(
                    f"counterfactual candidate changed protected capability: {capability}"
                )
        self._write_canonical_resume_artifact(
            self.output_root / "banks" / "s1-counterfactual-candidate.json",
            candidate.model_dump(mode="json"),
            label="S1 counterfactual candidate Bank",
        )
        self._write_canonical_resume_artifact(
            self.output_root / "banks" / "s1-counterfactual-compilation-receipt.json",
            compiled.receipt.model_dump(mode="json"),
            label="S1 counterfactual compilation receipt",
        )
        target_record.update(
            {
                "status": "screen_pending",
                "candidate_skill_sha256": candidate_by_capability[target].skill_sha256,
                "candidate_bank_sha256": candidate.bank_sha256,
                "compilation_receipt_sha256": compiled.receipt.receipt_sha256,
            }
        )

        roles = self.opt_fold_roles()
        eligible = (
            self._action_replay_parent_eligible
            if settings.target_surface == "action-policy"
            else self._body_replay_parent_eligible
        )
        replay_queries = tuple(
            query
            for query in self.queries()
            if query.split == "opt_pool" and roles[query.query_id] == "replay"
        )
        local_ids = {
            query.query_id
            for query in replay_queries
            if query.canonical_capability == target
            and opt[query.query_id].selected_capability == target
            and eligible(opt[query.query_id])
        } | set(parent_success_ids)
        local_queries = tuple(query_by_id[query_id] for query_id in sorted(local_ids))
        frozen = {query.query_id: opt[query.query_id] for query in local_queries}
        parent_local = self._assistant_many(
            split=f"{settings.round_id}-local-{settings.target_surface}",
            config="s1-parent-control",
            queries=local_queries,
            bank=parent,
            reuse_parent=frozen,
            replay_surface=settings.target_surface,
        )
        candidate_local = self._assistant_many(
            split=f"{settings.round_id}-local-{settings.target_surface}",
            config="s1-candidate-treatment",
            queries=local_queries,
            bank=candidate,
            reuse_parent=frozen,
            replay_surface=settings.target_surface,
        )
        screen = self._s1_policy_screen(
            capability=target,
            surface=settings.target_surface,
            queries=local_queries,
            baseline=parent_local,
            candidate=candidate_local,
        )
        protected_regressions = sorted(
            query_id
            for query_id in parent_success_ids
            if query_id in screen["regression_query_ids"]
        )
        screen["explicit_parent_success_regression_query_ids"] = protected_regressions
        if protected_regressions:
            screen["decision"] = "inherit_parent"
            screen["reason_codes"] = sorted(
                set(screen["reason_codes"]) | {"explicit_parent_success_regressed"}
            )
        metrics["local_screen"] = screen
        metrics["replay_accessed"] = True
        target_record["local_screen"] = screen
        if screen["decision"] != "retain_patch":
            target_record["status"] = "screen_rejected"
            self._write_selected_bank("s1", parent)
            return self._save_decision(
                StageDecision(
                    stage="s1",
                    accepted=False,
                    alias_of="s1",
                    parent_bank=parent.bank_sha256,
                    candidate_bank=candidate.bank_sha256,
                    selected_bank=parent.bank_sha256,
                    reasons=("bounded-risk local screen rejected the target branch",),
                    metrics=metrics,
                )
            )

        target_record["status"] = "local_screen_passed"
        parent_replay = self._assistant_many(
            split=f"{settings.round_id}-replay200",
            config="s1-parent",
            queries=replay_queries,
            bank=parent,
        )
        candidate_replay = self._assistant_many(
            split=f"{settings.round_id}-replay200",
            config="s1-candidate",
            queries=replay_queries,
            bank=candidate,
        )
        replay_ok, replay_reasons, replay_metrics = self._s1_gate(
            parent_replay,
            candidate_replay,
            replay_queries,
            phase="replay200",
            treated_capabilities=frozenset({target}),
        )
        metrics["replay_gate"] = replay_metrics
        reasons = [f"replay200: {reason}" for reason in replay_reasons]
        accepted = False
        if replay_ok:
            metrics["body_accessed"] = True
            body_queries = self._queries_for_val_gate("body_gate")
            parent_body = self._assistant_many(
                split=f"{settings.round_id}-body-gate75",
                config="s1-parent",
                queries=body_queries,
                bank=parent,
            )
            candidate_body = self._assistant_many(
                split=f"{settings.round_id}-body-gate75",
                config="s1-candidate",
                queries=body_queries,
                bank=candidate,
            )
            accepted, body_reasons, body_metrics = self._s1_gate(
                parent_body,
                candidate_body,
                body_queries,
                phase="body_gate75",
                treated_capabilities=frozenset({target}),
            )
            reasons.extend(f"body_gate75: {reason}" for reason in body_reasons)
            metrics["gate"] = body_metrics
        selected = candidate if accepted else parent
        target_record["status"] = "accepted" if accepted else "formal_gate_rejected"
        target_record["selected_skill_sha256"] = (
            candidate_by_capability[target].skill_sha256
            if accepted
            else parent_by_capability[target].skill_sha256
        )
        target_record["inherited_bytes_exact"] = not accepted
        if accepted:
            accepted_unsigned = {
                "schema_version": 1,
                "kind": "core-fast-s1-accepted-branch",
                "cycle_id": settings.cycle_id,
                "round_id": settings.round_id,
                "parent_round_id": binding.source_round_id,
                "parent_bank_sha256": parent.bank_sha256,
                "capability": target,
                "surface": settings.target_surface,
                "candidate_bank_sha256": candidate.bank_sha256,
                "candidate_skill_sha256": candidate_by_capability[target].skill_sha256,
                "compilation_receipt_sha256": compiled.receipt.receipt_sha256,
                "local_screen_sha256": sha256_bytes(canonical_json_bytes(screen)),
                "replay_gate": replay_metrics,
                "body_gate": metrics["gate"],
            }
            accepted_artifact = {
                **accepted_unsigned,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(accepted_unsigned)),
            }
            self._write_canonical_resume_artifact(
                self.output_root / "accepted-branch.json",
                accepted_artifact,
                label="S1 accepted branch",
            )
            metrics["accepted_branch_receipt_sha256"] = accepted_artifact[
                "receipt_sha256"
            ]
        self._write_selected_bank("s1", selected)
        return self._save_decision(
            StageDecision(
                stage="s1",
                accepted=accepted,
                alias_of=None if accepted else "s1",
                parent_bank=parent.bank_sha256,
                candidate_bank=candidate.bank_sha256,
                selected_bank=selected.bank_sha256,
                reasons=tuple(reasons),
                metrics=metrics,
            )
        )

    def run_s1(self) -> StageDecision:
        existing = self._existing_decision("s1")
        if existing is not None:
            return existing
        counterfactual = (
            self.spec.s1_settings.proposal_mode in _COUNTERFACTUAL_PROPOSAL_MODES
        )
        parent = self.s1_parent_bank() if counterfactual else self.static_bank()
        opt = self.s1_parent_opt() if counterfactual else self.opt_static()
        query_by_id = self.query_by_id()

        try:
            prepared = self.prepare_feedback_selection()
        except FastPathError as error:
            if counterfactual and "insufficient_counterfactual_evidence" in str(error):
                target = self.spec.s1_settings.target_capabilities[0]
                branches = [
                    {
                        "capability": capability,
                        "status": (
                            "insufficient_counterfactual_evidence"
                            if capability == target
                            else "protected_inherit"
                        ),
                        "creator_called": False,
                    }
                    for capability in CAPABILITIES
                ]
                self._write_selected_bank("s1", parent)
                return self._save_decision(
                    StageDecision(
                        stage="s1",
                        accepted=False,
                        alias_of="s1",
                        parent_bank=parent.bank_sha256,
                        candidate_bank=None,
                        selected_bank=parent.bank_sha256,
                        reasons=("insufficient_counterfactual_evidence",),
                        metrics={
                            "round_id": self.spec.s1_settings.round_id,
                            "cycle_id": self.spec.s1_settings.cycle_id,
                            "parent_round_id": self.spec.s1_parent.source_round_id
                            if self.spec.s1_parent
                            else None,
                            "fanout_branches": branches,
                            "creator_called": False,
                            "replay_accessed": False,
                            "body_accessed": False,
                        },
                    )
                )
            raise
        manifest = prepared["selection_manifest"]
        assert isinstance(manifest, dict)
        self._require_counterfactual_cycle_preflight(prepared)
        samples = manifest["selected_samples"]
        assert isinstance(samples, list)
        canary_count = self.spec.s1_settings.feedback_canary_count
        canary_feedback, canary_resume_hits, canary_provider_calls = (
            self._feedback_batch(
                samples=samples[:canary_count],
                query_by_id=query_by_id,
                opt=opt,
                selection_manifest_sha256=str(manifest["manifest_sha256"]),
            )
        )
        feedback: dict[str, dict[str, object]] = dict(canary_feedback)
        resume_hits = canary_resume_hits
        provider_calls = canary_provider_calls
        canary_gate = self._feedback_batch_gate(
            canary_feedback, expected_count=canary_count
        )
        if canary_gate["passed"]:
            remaining_feedback, remaining_hits, remaining_calls = self._feedback_batch(
                samples=samples[canary_count:],
                query_by_id=query_by_id,
                opt=opt,
                selection_manifest_sha256=str(manifest["manifest_sha256"]),
            )
            feedback.update(remaining_feedback)
            resume_hits += remaining_hits
            provider_calls += remaining_calls
        else:
            reasons = ("Feedback canary failed the frozen operational error limits",)
            metrics = {
                "round_id": self.spec.s1_settings.round_id,
                "feedback_mode": self.spec.s1_settings.feedback_mode,
                "feedback_requested_count": self.spec.s1_settings.feedback_total_count,
                "feedback_canary_count": canary_count,
                "feedback_success_count": sum(
                    row["feedback"] is not None for row in feedback.values()
                ),
                "feedback_failure_count": sum(
                    row["feedback"] is None for row in feedback.values()
                ),
                "feedback_provider_calls_this_round": provider_calls,
                "feedback_resume_cache_hits": resume_hits,
                "feedback_remaining_batch_called": False,
                "feedback_canary_gate": canary_gate,
                "feedback_full_batch_gate": None,
                "selection_manifest_sha256": manifest["manifest_sha256"],
                "replay_accessed": False,
                "body_accessed": False,
            }
            if counterfactual:
                target = self.spec.s1_settings.target_capabilities[0]
                metrics.update(
                    {
                        "cycle_id": self.spec.s1_settings.cycle_id,
                        "parent_round_id": self.spec.s1_parent.source_round_id
                        if self.spec.s1_parent
                        else None,
                        "fanout_branches": [
                            {
                                "capability": capability,
                                "status": (
                                    "feedback_gate_rejected"
                                    if capability == target
                                    else "protected_inherit"
                                ),
                                "creator_called": False,
                            }
                            for capability in CAPABILITIES
                        ],
                    }
                )
            self._write_selected_bank("s1", parent)
            return self._save_decision(
                StageDecision(
                    stage="s1",
                    accepted=False,
                    alias_of=self._s1_parent_alias(),
                    parent_bank=parent.bank_sha256,
                    candidate_bank=None,
                    selected_bank=parent.bank_sha256,
                    reasons=reasons,
                    metrics=metrics,
                )
            )
        full_gate = self._feedback_batch_gate(
            feedback,
            expected_count=self.spec.s1_settings.feedback_total_count,
        )
        if not full_gate["passed"]:
            reasons = (
                "Full Feedback batch failed the frozen completeness or error limits",
            )
            metrics = {
                "round_id": self.spec.s1_settings.round_id,
                "feedback_mode": self.spec.s1_settings.feedback_mode,
                "feedback_requested_count": (
                    self.spec.s1_settings.feedback_total_count
                ),
                "feedback_canary_count": canary_count,
                "feedback_success_count": sum(
                    row["feedback"] is not None for row in feedback.values()
                ),
                "feedback_failure_count": sum(
                    row["feedback"] is None for row in feedback.values()
                ),
                "feedback_provider_calls_this_round": provider_calls,
                "feedback_resume_cache_hits": resume_hits,
                "feedback_remaining_batch_called": True,
                "feedback_canary_gate": canary_gate,
                "feedback_full_batch_gate": full_gate,
                "selection_manifest_sha256": manifest["manifest_sha256"],
                "creator_called": False,
                "replay_accessed": False,
                "body_accessed": False,
            }
            if counterfactual:
                target = self.spec.s1_settings.target_capabilities[0]
                metrics.update(
                    {
                        "cycle_id": self.spec.s1_settings.cycle_id,
                        "parent_round_id": self.spec.s1_parent.source_round_id
                        if self.spec.s1_parent
                        else None,
                        "fanout_branches": [
                            {
                                "capability": capability,
                                "status": (
                                    "feedback_gate_rejected"
                                    if capability == target
                                    else "protected_inherit"
                                ),
                                "creator_called": False,
                            }
                            for capability in CAPABILITIES
                        ],
                    }
                )
            self._write_selected_bank("s1", parent)
            return self._save_decision(
                StageDecision(
                    stage="s1",
                    accepted=False,
                    alias_of=self._s1_parent_alias(),
                    parent_bank=parent.bank_sha256,
                    candidate_bank=None,
                    selected_bank=parent.bank_sha256,
                    reasons=reasons,
                    metrics=metrics,
                )
            )
        feedback_bundle = self._feedback_evidence_bundle(
            parent=parent, prepared=prepared, feedback=feedback
        )
        feedback_bundle_sha256 = str(feedback_bundle["bundle_sha256"])
        self._write_or_verify_json(
            self.output_root / "inputs" / "feedback-evidence-bundle.json",
            feedback_bundle,
            label="Feedback evidence bundle",
        )
        grouped = feedback_bundle["feedback_by_capability"]
        assert isinstance(grouped, dict)
        patchable_capabilities = frozenset(
            capability
            for capability, rows in grouped.items()
            if any(
                isinstance(row.get("feedback"), dict)
                and bool(row["feedback"].get("skill_suggestions"))
                for row in rows
            )
        )
        base_metrics: dict[str, object] = {
            "feedback_success_count": sum(
                item["feedback"] is not None for item in feedback.values()
            ),
            "round_id": self.spec.s1_settings.round_id,
            "feedback_mode": self.spec.s1_settings.feedback_mode,
            "feedback_requested_count": self.spec.s1_settings.feedback_total_count,
            "feedback_canary_count": canary_count,
            "feedback_failure_count": sum(
                item["feedback"] is None for item in feedback.values()
            ),
            "feedback_provider_calls_this_round": provider_calls,
            "feedback_resume_cache_hits": resume_hits,
            "feedback_remaining_batch_called": True,
            "feedback_canary_gate": canary_gate,
            "feedback_full_batch_gate": full_gate,
            "feedback_selection_policy": (
                self.spec.s1_settings.feedback_selection_policy
            ),
            "feedback_allocation": self.spec.s1_settings.feedback_allocation,
            "selection_manifest_sha256": manifest["manifest_sha256"],
            "discovery_summary_sha256": prepared["discovery_summary"]["summary_sha256"],
            "feedback_bundle_sha256": feedback_bundle_sha256,
            "feedback_bundle_byte_size": len(canonical_json_bytes(feedback_bundle)),
            "feedback_cluster_coverage": manifest["cluster_coverage"],
            "feedback_capability_coverage": manifest["capability_quotas"],
            "feedback_truncated_field_count": 0,
            "replay_accessed": False,
            "body_accessed": False,
        }
        if self.spec.s1_settings.proposal_mode == "six-capability-fanout-fanin-v2":
            return self._run_s1_fanout_fanin(
                parent=parent,
                opt=opt,
                query_by_id=query_by_id,
                feedback_bundle=feedback_bundle,
                feedback_bundle_sha256=feedback_bundle_sha256,
                patchable_capabilities=patchable_capabilities,
                metrics=base_metrics,
            )
        if counterfactual:
            return self._run_s1_counterfactual_branch(
                parent=parent,
                opt=opt,
                query_by_id=query_by_id,
                feedback_bundle=feedback_bundle,
                feedback_bundle_sha256=feedback_bundle_sha256,
                patchable_capabilities=patchable_capabilities,
                metrics=base_metrics,
            )
        if (
            self.spec.s1_settings.proposal_mode
            == "six-capability-dual-policy-fanout-fanin-v3"
        ):
            return self._run_s1_dual_policy_fanout_fanin(
                parent=parent,
                opt=opt,
                query_by_id=query_by_id,
                feedback_bundle=feedback_bundle,
                feedback_bundle_sha256=feedback_bundle_sha256,
                patchable_capabilities=patchable_capabilities,
                metrics=base_metrics,
            )
        templates = decode_sparse_parent_content(parent, self.s1_authoring_input())
        creator = self._call(
            role="creator",
            call_id="s1-creator-once",
            purpose="S1 parent-bound sparse capability Creator",
            payload={
                "operation": "s1_creator",
                "parent_bank": parent.model_dump(mode="json"),
                "parent_authoring_content": [
                    item.model_dump(mode="json") for item in templates
                ],
                "feedback_evidence_bundle": feedback_bundle,
                "discovery_summary": feedback_bundle["discovery_summary"],
                "selection_manifest_sha256": feedback_bundle[
                    "selection_manifest_sha256"
                ],
                "policy_compatible_suggestions": feedback_bundle[
                    "policy_compatible_suggestions"
                ],
                "feedback_bundle_sha256": feedback_bundle_sha256,
                "parent_body_content": {
                    item.capability_id: {
                        "objective": item.objective,
                        "steps": [step.model_dump(mode="json") for step in item.steps],
                        "fallback_instruction": item.fallback_instruction,
                        "citation_source_ids": list(item.citation_source_ids),
                    }
                    for item in decode_sparse_parent_content(
                        parent, self.s1_authoring_input()
                    )
                },
                "requirements": {
                    "round_id": self.spec.s1_settings.round_id,
                    "target_capabilities": list(
                        self.spec.s1_settings.target_capabilities
                    ),
                    "model_generated_body_only": True,
                    "model_owns_tool_choice_and_final_response": True,
                    "response_compiler": "disabled",
                    "complete_six_capability_actions": True,
                    "default_action": "inherit",
                    "patch_only_with_policy_compatible_suggestion": True,
                    "freeze_all_descriptions": True,
                    "protected_capabilities_must_inherit": list(
                        self.spec.s1_settings.protected_capabilities
                    ),
                    "max_patched_capabilities": (
                        self.spec.s1_settings.max_patched_capabilities
                    ),
                    "creator_directives": list(
                        self.spec.s1_settings.creator_directives
                    ),
                    "required_patch_phrases": {
                        capability: list(phrases)
                        for capability, phrases in (
                            self.spec.s1_settings.required_patch_phrases.items()
                        )
                    },
                    "author_content_lexical_guard": (
                        sparse_author_content_lexical_guard()
                    ),
                    "single_candidate": True,
                },
                "output_schema": self._creator_schema("s1", parent=parent),
            },
        )
        candidate_rejection_reason: str | None = None
        try:
            candidate = self._candidate_bank(
                creator,
                stage="s1",
                parent=parent,
                feedback_bundle_sha256=feedback_bundle_sha256,
                feedback_patchable_capabilities=patchable_capabilities,
            )
        except _S1CandidateRejected as error:
            candidate = None
            candidate_rejection_reason = error.reason_code
        reasons: list[str] = []
        metrics: dict[str, object] = {
            **base_metrics,
            "creator_input_tokens": creator.input_tokens,
        }
        accepted = False
        if candidate is None:
            reasons.append("Creator failed or returned an invalid sparse S1 proposal")
            metrics["creator_candidate_rejection_reason"] = (
                candidate_rejection_reason or "provider_or_schema_invalid"
            )
        else:
            smoke_queries = [
                query_by_id[item.query_id]
                for item in self.spec.fixed_samples.dev_smoke24
            ]
            smoke = self._assistant_many(
                split="dev-smoke24",
                config="s1-candidate",
                queries=smoke_queries,
                bank=candidate,
            )
            metrics["smoke_hard_errors"] = sum(row.hard_error for row in smoke.values())
            smoke_oracle_failures = sorted(
                query_id for query_id, row in smoke.items() if not row.oracle_available
            )
            metrics["smoke_oracle_coverage_complete"] = not smoke_oracle_failures
            metrics["smoke_oracle_coverage_failure_query_ids"] = smoke_oracle_failures
            if not self._s1_smoke_ok(smoke):
                reasons.append(
                    "candidate failed fixed dev smoke24 operational/oracle coverage"
                )
            else:
                fold_roles = self.opt_fold_roles()
                metrics["replay_accessed"] = True
                replay_queries = [
                    query
                    for query in self.queries()
                    if query.split == "opt_pool"
                    and fold_roles[query.query_id] == "replay"
                ]
                static_replay = {
                    query.query_id: opt[query.query_id] for query in replay_queries
                }
                raw_candidate = candidate
                raw_candidate_replay = self._assistant_many(
                    split="opt-replay200",
                    config="s1-candidate",
                    queries=replay_queries,
                    bank=raw_candidate,
                )
                raw_replay_oracle_failures = sorted(
                    {
                        query_id
                        for rows in (static_replay, raw_candidate_replay)
                        for query_id, row in rows.items()
                        if not row.oracle_available
                    }
                )
                metrics["raw_candidate_bank"] = raw_candidate.bank_sha256
                metrics[
                    "raw_replay_oracle_coverage_complete"
                ] = not raw_replay_oracle_failures
                metrics["raw_replay_oracle_coverage_failure_query_ids"] = (
                    raw_replay_oracle_failures
                )
                if raw_replay_oracle_failures:
                    reasons.append("raw replay200 oracle coverage is incomplete")
                    self._write_selected_bank("s1", parent)
                    return self._save_decision(
                        StageDecision(
                            stage="s1",
                            accepted=False,
                            alias_of="llm_static",
                            parent_bank=parent.bank_sha256,
                            candidate_bank=raw_candidate.bank_sha256,
                            selected_bank=parent.bank_sha256,
                            reasons=tuple(reasons),
                            metrics=metrics,
                        )
                    )
                baseline_scores = self._s1_development_scores(
                    static_replay, replay_queries, config="llm_static"
                )
                raw_candidate_scores = self._s1_development_scores(
                    raw_candidate_replay, replay_queries, config="s1"
                )
                try:
                    development_screen = screen_s1_development_patches(
                        queries=replay_queries,
                        baseline_scores=baseline_scores,
                        candidate_scores=raw_candidate_scores,
                    )
                except S1GCSGateError as error:
                    raise FastPathError(
                        "S1 replay200 development screen failed"
                    ) from error
                screen_payload = development_screen.model_dump(mode="json")
                screen_bytes = canonical_json_bytes(screen_payload)
                screen_path = self.output_root / "banks" / "s1-development-screen.json"
                self._write_canonical_resume_artifact(
                    screen_path,
                    screen_payload,
                    label="S1 development screen",
                )

                compilation_receipt_path = (
                    self.output_root / "banks" / "s1-sparse-compilation-receipt.json"
                )
                try:
                    compilation_receipt = load_sparse_compilation_receipt(
                        compilation_receipt_path,
                        expected_file_sha256=_file_sha(compilation_receipt_path),
                    )
                except (OSError, S1SparsePatchError) as error:
                    raise FastPathError(
                        "S1 sparse compilation receipt is invalid"
                    ) from error
                patched_capabilities = frozenset(
                    item.capability_id
                    for item in compilation_receipt.bindings
                    if item.action == "patch"
                )
                retained_capabilities = tuple(
                    sorted(
                        item.capability_id
                        for item in development_screen.decisions
                        if item.capability_id in patched_capabilities
                        and item.decision == "retain_patch"
                    )
                )
                reverted_capabilities = tuple(
                    sorted(patched_capabilities - set(retained_capabilities))
                )
                try:
                    screened = compose_screened_sparse_bank(
                        parent_bank=parent,
                        creator_candidate_bank=raw_candidate,
                        creator_compilation_receipt=compilation_receipt,
                        development_screen_sha256=sha256_bytes(screen_bytes),
                        retained_capability_ids=retained_capabilities,
                    )
                except (S1SparsePatchError, ValidationError, ValueError) as error:
                    raise FastPathError("S1 screened sparse Bank is invalid") from error
                screened_artifacts = {
                    "s1-screened-candidate.json": screened.bank.model_dump(mode="json"),
                    "s1-screened-bank-receipt.json": screened.receipt.model_dump(
                        mode="json"
                    ),
                }
                for name, artifact in screened_artifacts.items():
                    self._write_canonical_resume_artifact(
                        self.output_root / "banks" / name,
                        artifact,
                        label=f"S1 {name}",
                    )
                candidate = screened.bank
                metrics["development_screen_sha256"] = sha256_bytes(screen_bytes)
                metrics["development_screen"] = screen_payload
                metrics["retained_patch_capabilities"] = list(retained_capabilities)
                metrics["reverted_patch_capabilities"] = list(reverted_capabilities)
                metrics["screened_candidate_bank"] = candidate.bank_sha256

                if not retained_capabilities:
                    reasons.append("replay200 screen reverted all sparse patches")
                else:
                    replay_rerun = candidate.bank_sha256 != raw_candidate.bank_sha256
                    metrics["composite_replay_rerun"] = replay_rerun
                    candidate_replay = (
                        self._assistant_many(
                            split="opt-replay200-composite",
                            config="s1-candidate",
                            queries=replay_queries,
                            bank=candidate,
                        )
                        if replay_rerun
                        else raw_candidate_replay
                    )
                    replay_ok, replay_reasons, replay_metrics = self._s1_gate(
                        static_replay,
                        candidate_replay,
                        replay_queries,
                        phase="replay200",
                        treated_capabilities=frozenset(retained_capabilities),
                    )
                    metrics["replay_gate"] = replay_metrics
                    if not replay_ok:
                        reasons.extend(f"replay200: {item}" for item in replay_reasons)
                    else:
                        gate_queries = self._queries_for_val_gate("body_gate")
                        metrics["body_accessed"] = True
                        static_rows = self._assistant_many(
                            split="body-gate75",
                            config="llm_static",
                            queries=gate_queries,
                            bank=parent,
                        )
                        candidate_rows = self._assistant_many(
                            split="body-gate75",
                            config="s1-candidate",
                            queries=gate_queries,
                            bank=candidate,
                        )
                        accepted, gate_reasons, gate_metrics = self._s1_gate(
                            static_rows,
                            candidate_rows,
                            gate_queries,
                            phase="body_gate75",
                            treated_capabilities=frozenset(retained_capabilities),
                        )
                        reasons.extend(gate_reasons)
                        metrics["gate"] = gate_metrics
        selected = candidate if accepted and candidate is not None else parent
        self._write_selected_bank("s1", selected)
        decision = StageDecision(
            stage="s1",
            accepted=accepted,
            alias_of=None if accepted else "llm_static",
            parent_bank=parent.bank_sha256,
            candidate_bank=None if candidate is None else candidate.bank_sha256,
            selected_bank=selected.bank_sha256,
            reasons=tuple(reasons),
            metrics=metrics,
        )
        return self._save_decision(decision)

    # ------------------------------- S2 ------------------------------------

    def run_s2(self) -> StageDecision:
        existing = self._existing_decision("s2")
        if existing is not None:
            return existing
        self.run_s1()
        parent = self._load_selected_bank("s1")
        attribution = self.opt_attribution()
        query_by_id = self.query_by_id()
        boundary_queries = [
            query
            for capability in CAPABILITIES
            for query in [
                item
                for item in self.queries()
                if item.split == "opt_pool"
                and item.is_boundary
                and item.canonical_capability == capability
            ][:8]
        ]
        boundaries = [
            {
                "query": query.model_dump(mode="json"),
                "attribution": attribution[query.query_id],
            }
            for query in boundary_queries
        ]
        creator = self._call(
            role="creator",
            call_id="s2-route-optimizer-once",
            purpose="S2 single Description/route-surface optimizer",
            payload={
                "operation": "s2_route_optimizer",
                "parent_bank": parent.model_dump(mode="json"),
                "opt800_route_attribution": list(attribution.values()),
                "opt800_confusion_matrix": self._route_attribution_confusion(
                    attribution
                ),
                "boundary_examples": boundaries,
                "requirements": {
                    "single_candidate": True,
                    "description_only": True,
                    "preserve_body_operators_static_refs": True,
                },
                "output_schema": self._creator_schema("s2"),
            },
        )
        candidate = self._candidate_bank(creator, stage="s2", parent=parent)
        reasons: list[str] = []
        metrics: dict[str, object] = {}
        accepted = False
        if candidate is None:
            reasons.append("optimizer failed or returned an invalid Bank")
        else:
            violations = self._boundary_changes(parent, candidate, "s2")
            if violations:
                reasons.extend(violations)
            else:
                smoke_queries = [
                    query_by_id[item.query_id]
                    for item in self.spec.fixed_samples.dev_smoke24
                ]
                smoke = self._assistant_many(
                    split="dev-smoke24",
                    config="s2-candidate",
                    queries=smoke_queries,
                    bank=candidate,
                )
                metrics["smoke_hard_errors"] = sum(
                    row.hard_error for row in smoke.values()
                )
                if not self._smoke_ok(smoke):
                    reasons.append("candidate failed fixed dev smoke24")
                else:
                    val_queries = [
                        query for query in self.queries() if query.split == "val"
                    ]
                    parent_config = (
                        "s1-candidate" if self.run_s1().accepted else "llm_static"
                    )
                    parent_rows = self._assistant_many(
                        split="val",
                        config=parent_config,
                        queries=val_queries,
                        bank=parent,
                    )
                    candidate_rows = self._assistant_many(
                        split="val",
                        config="s2-candidate",
                        queries=val_queries,
                        bank=candidate,
                    )
                    accepted, gate_reasons, gate_metrics = self._s2_gate(
                        parent_rows, candidate_rows, val_queries
                    )
                    reasons.extend(gate_reasons)
                    metrics["gate"] = gate_metrics
        selected = candidate if accepted and candidate is not None else parent
        self._write_selected_bank("s2", selected)
        decision = StageDecision(
            stage="s2",
            accepted=accepted,
            alias_of=None if accepted else "s1",
            parent_bank=parent.bank_sha256,
            candidate_bank=None if candidate is None else candidate.bank_sha256,
            selected_bank=selected.bank_sha256,
            reasons=tuple(reasons),
            metrics=metrics,
        )
        return self._save_decision(decision)

    # ------------------------------- S3 ------------------------------------

    def _judge_one(
        self,
        *,
        split: str,
        config: str,
        query: Query,
        observation: AssistantObservation,
    ) -> JudgeObservation:
        base_id = _safe_id(f"{split}-{config}-{query.query_id}")
        payload = {
            "operation": "final_judge",
            "query": query.model_dump(mode="json"),
            "assistant_observation": observation.model_dump(mode="json"),
            "runtime_paths": self.spec.paths.model_dump(mode="json"),
        }
        first = self._call(
            role="judge",
            call_id=f"{base_id}-attempt1",
            purpose=f"{split} paired Final Judge {config} {query.query_id}",
            payload=payload,
        )
        parsed = _judge_from_result(first, query.query_id)
        if parsed is not None:
            return parsed
        retryable_format_failure = (
            first.status
            in {
                "empty_response",
                "schema_error",
            }
            or first.status == "success"
        )
        if not retryable_format_failure:
            return JudgeObservation(
                query_id=query.query_id, j_project=0.0, dimensions={}
            )
        second = self._call(
            role="judge",
            call_id=f"{base_id}-attempt2",
            purpose=f"{split} format-only Final Judge retry {config} {query.query_id}",
            payload={**payload, "equivalent_retry_of": f"{base_id}-attempt1"},
        )
        return _judge_from_result(second, query.query_id) or JudgeObservation(
            query_id=query.query_id, j_project=0.0, dimensions={}
        )

    def _judge_many(
        self,
        *,
        split: str,
        config: str,
        queries: Sequence[Query],
        observations: Mapping[str, AssistantObservation],
    ) -> dict[str, JudgeObservation]:
        results: dict[str, JudgeObservation] = {}
        with ThreadPoolExecutor(max_workers=self.spec.concurrency.final_judge) as pool:
            futures = {
                pool.submit(
                    self._judge_one,
                    split=split,
                    config=config,
                    query=query,
                    observation=observations[query.query_id],
                ): query.query_id
                for query in queries
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return {query.query_id: results[query.query_id] for query in queries}

    def run_full(self) -> StageDecision:
        existing = self._existing_decision("full")
        if existing is not None:
            return existing
        self.run_s2()
        parent = self._load_selected_bank("s2")
        query_by_id = self.query_by_id()
        body_queries = [
            query_by_id[item.query_id] for item in self.spec.fixed_samples.body48
        ]
        parent_config = (
            "s2-candidate"
            if self.run_s2().accepted
            else ("s1-candidate" if self.run_s1().accepted else "llm_static")
        )
        body_parent = self._assistant_many(
            split="opt-body48",
            config=parent_config,
            queries=body_queries,
            bank=parent,
        )
        creator = self._call(
            role="creator",
            call_id="s3-body-refiner-once",
            purpose="S3 single Body refiner with at most three rule edits",
            payload={
                "operation": "s3_body_refiner",
                "parent_bank": parent.model_dump(mode="json"),
                "body48_trajectories": [
                    {
                        "query": query.model_dump(mode="json"),
                        "parent": body_parent[query.query_id].model_dump(mode="json"),
                    }
                    for query in body_queries
                ],
                "requirements": {
                    "body_only": True,
                    "max_rule_level_edits": 3,
                    "single_candidate": True,
                },
                "output_schema": self._creator_schema("full"),
            },
        )
        candidate = self._candidate_bank(creator, stage="full", parent=parent)
        reasons: list[str] = []
        metrics: dict[str, object] = {}
        accepted = False
        affected: tuple[str, ...] = ()
        if candidate is None:
            reasons.append("refiner failed or returned an invalid Bank")
        else:
            violations = self._boundary_changes(parent, candidate, "full")
            before = _bank_by_capability(parent)
            after = _bank_by_capability(candidate)
            affected = tuple(
                capability
                for capability in CAPABILITIES
                if getattr(before[capability], "body")
                != getattr(after[capability], "body")
            )
            edits = creator.output.get("edits") if creator.output else None
            if not isinstance(edits, list) or len(edits) > 3:
                violations += ("S3 output must declare at most three rule-level edits",)
            if not affected:
                violations += ("S3 candidate did not change any Body",)
            if violations:
                reasons.extend(violations)
            else:
                smoke_queries = [
                    query_by_id[item.query_id]
                    for item in self.spec.fixed_samples.dev_smoke24
                ]
                parent_smoke = self._assistant_many(
                    split="dev-smoke24-parent-s3",
                    config=parent_config,
                    queries=smoke_queries,
                    bank=parent,
                )
                candidate_smoke = self._assistant_many(
                    split="dev-smoke24",
                    config="full-candidate",
                    queries=smoke_queries,
                    bank=candidate,
                    reuse_parent=parent_smoke,
                )
                metrics["smoke_hard_errors"] = sum(
                    row.hard_error for row in candidate_smoke.values()
                )
                smoke_trace_violations = self._common_trace_violations(
                    parent_smoke, candidate_smoke
                )
                metrics["smoke_common_trace_violations"] = len(smoke_trace_violations)
                if smoke_trace_violations:
                    reasons.append("candidate changed route/tool trace in smoke24")
                elif not self._smoke_ok(candidate_smoke):
                    reasons.append("candidate failed fixed common-trace smoke24")
                else:
                    val_queries = [
                        query for query in self.queries() if query.split == "val"
                    ]
                    parent_rows = self._assistant_many(
                        split="val",
                        config=parent_config,
                        queries=val_queries,
                        bank=parent,
                    )
                    candidate_rows = self._assistant_many(
                        split="val",
                        config="full-candidate",
                        queries=val_queries,
                        bank=candidate,
                        reuse_parent=parent_rows,
                    )
                    before_summary = self._summary(parent_rows, val_queries)
                    after_summary = self._summary(candidate_rows, val_queries)
                    gate_reasons: list[str] = []
                    trace_violations = self._common_trace_violations(
                        parent_rows, candidate_rows
                    )
                    if trace_violations:
                        gate_reasons.append("route/tool trace changed on val200")
                    if (
                        after_summary["capability_macro_gcs"]
                        <= before_summary["capability_macro_gcs"]
                    ):
                        gate_reasons.append(
                            "capability-macro GCS did not strictly improve"
                        )
                    for key in (
                        "hard_errors",
                        "card_violations",
                        "evidence_violations",
                    ):
                        if after_summary[key] > before_summary[key]:
                            gate_reasons.append(f"{key} increased")
                    subset: list[Query] = []
                    for capability in affected:
                        subset.extend(
                            query
                            for query in val_queries
                            if query.canonical_capability == capability
                        )
                        subset = subset[: self.spec.gates.s3_judge_subset_max]
                    # Rebuild deterministically: first four within each affected capability.
                    subset = [
                        query
                        for capability in affected
                        for query in [
                            item
                            for item in val_queries
                            if item.canonical_capability == capability
                        ][: self.spec.gates.s3_judge_subset_per_affected_capability]
                    ][: self.spec.gates.s3_judge_subset_max]
                    if trace_violations:
                        delta_j = None
                    else:
                        parent_judge = self._judge_many(
                            split="val-s3-gate",
                            config="parent",
                            queries=subset,
                            observations=parent_rows,
                        )
                        candidate_judge = self._judge_many(
                            split="val-s3-gate",
                            config="candidate",
                            queries=subset,
                            observations=candidate_rows,
                        )
                        delta_j = (
                            sum(
                                candidate_judge[q.query_id].j_project
                                - parent_judge[q.query_id].j_project
                                for q in subset
                            )
                            / len(subset)
                            if subset
                            else 0.0
                        )
                    if delta_j is not None and delta_j < 0:
                        gate_reasons.append("paired mean delta J is negative")
                    reasons.extend(gate_reasons)
                    accepted = not gate_reasons
                    metrics["gate"] = {
                        "parent": before_summary,
                        "candidate": after_summary,
                        "affected_capabilities": list(affected),
                        "judge_query_ids": [query.query_id for query in subset],
                        "paired_mean_delta_j": delta_j,
                        "common_trace_violation_query_ids": list(trace_violations),
                    }
        selected = candidate if accepted and candidate is not None else parent
        self._write_selected_bank("full", selected)
        decision = StageDecision(
            stage="full",
            accepted=accepted,
            alias_of=None if accepted else "s1s2",
            parent_bank=parent.bank_sha256,
            candidate_bank=None if candidate is None else candidate.bank_sha256,
            selected_bank=selected.bank_sha256,
            reasons=tuple(reasons),
            metrics=metrics,
        )
        return self._save_decision(decision)

    # ------------------------------- final ---------------------------------

    def _physical_config(self, config: str) -> str:
        decisions = {
            stage: self._existing_decision(stage) for stage in ("s1", "s2", "full")
        }
        if any(value is None for value in decisions.values()):
            raise FastPathError(
                "all S1/S2/S3 decisions must exist before logical result assembly"
            )
        s1 = decisions["s1"]
        s2 = decisions["s2"]
        full = decisions["full"]
        assert s1 is not None and s2 is not None and full is not None
        if config == "s1" and not s1.accepted:
            return "llm_static"
        if config == "s1s2" and not s2.accepted:
            return self._physical_config("s1")
        if config == "full" and not full.accepted:
            return self._physical_config("s1s2")
        return config

    def _bank_for_config(self, config: str) -> StaticBankArtifact | None:
        physical = self._physical_config(config)
        if physical == "noskill":
            return None
        if physical == "llm_static":
            return self.static_bank()
        return self._load_selected_bank(
            {"s1": "s1", "s1s2": "s2", "full": "full"}[physical]
        )

    @staticmethod
    def _val_run_label(physical: str) -> str:
        return {
            "noskill": "noskill",
            "llm_static": "llm_static",
            "s1": "s1-candidate",
            "s1s2": "s2-candidate",
            "full": "full-candidate",
        }[physical]

    def _load_completed_assistant_rows(
        self, *, split: str, physical: str, queries: Sequence[Query]
    ) -> dict[str, AssistantObservation]:
        label = self._val_run_label(physical) if split == "val" else physical
        rows: dict[str, AssistantObservation] = {}
        for query in queries:
            result = self.calls.get(
                "assistant", self._assistant_call_id(split, label, query.query_id)
            )
            if result is None:
                raise FastPathError(
                    f"completed Assistant result missing: {split}/{physical}/{query.query_id}"
                )
            rows[query.query_id] = _observation_from_result(result, query.query_id)
        return rows

    def _load_completed_judge_rows(
        self, *, split: str, physical: str, queries: Sequence[Query]
    ) -> dict[str, JudgeObservation]:
        rows: dict[str, JudgeObservation] = {}
        for query in queries:
            base_id = _safe_id(f"{split}-{physical}-{query.query_id}")
            first = self.calls.get("judge", f"{base_id}-attempt1")
            if first is None:
                raise FastPathError(
                    f"completed Judge result missing: {split}/{physical}/{query.query_id}"
                )
            parsed = _judge_from_result(first, query.query_id)
            retryable = first.status in {
                "empty_response",
                "schema_error",
            } or (first.status == "success" and parsed is None)
            if parsed is None and retryable:
                second = self.calls.get("judge", f"{base_id}-attempt2")
                if second is None:
                    raise FastPathError(
                        "format-failed Judge call lacks its one allowed retry: "
                        f"{query.query_id}"
                    )
                parsed = _judge_from_result(second, query.query_id)
            rows[query.query_id] = parsed or JudgeObservation(
                query_id=query.query_id, j_project=0.0, dimensions={}
            )
        return rows

    def rebuild_logical_results(self) -> None:
        """Reassemble val/test logical matrices without invoking any model."""

        val_queries = [query for query in self.queries() if query.split == "val"]
        test_queries = [
            query for query in self.queries() if query.split == "test_frozen"
        ]
        physicals = {self._physical_config(config) for config in CONFIGS}
        val_rows = {
            physical: self._load_completed_assistant_rows(
                split="val", physical=physical, queries=val_queries
            )
            for physical in physicals
        }
        test_rows = {
            physical: self._load_completed_assistant_rows(
                split="test_frozen", physical=physical, queries=test_queries
            )
            for physical in physicals
        }
        test_judges = {
            physical: self._load_completed_judge_rows(
                split="test_frozen", physical=physical, queries=test_queries
            )
            for physical in physicals
        }
        logical_val = self._logical_rows(
            split="val", assistant_by_physical=val_rows, judge_by_physical=None
        )
        logical_test = self._logical_rows(
            split="test_frozen",
            assistant_by_physical=test_rows,
            judge_by_physical=test_judges,
        )
        if len(logical_val) != 1000 or len(logical_test) != 1500:
            raise FastPathError("logical result geometry must be val1000/test1500")
        atomic_write_jsonl(
            self.output_root / "reports" / "val-results.jsonl",
            logical_val,
            overwrite=True,
        )
        atomic_write_jsonl(
            self.output_root / "reports" / "test-results.jsonl",
            logical_test,
            overwrite=True,
        )

    def _finalize_val_results(self) -> None:
        """Complete and materialize the five-config val1000 matrix."""

        val_queries = [query for query in self.queries() if query.split == "val"]
        physicals = {self._physical_config(config) for config in CONFIGS}
        for physical in physicals:
            self._assistant_many(
                split="val",
                config=self._val_run_label(physical),
                queries=val_queries,
                bank=self._bank_for_config(physical),
            )
        val_rows = {
            physical: self._load_completed_assistant_rows(
                split="val", physical=physical, queries=val_queries
            )
            for physical in physicals
        }
        logical_val = self._logical_rows(
            split="val", assistant_by_physical=val_rows, judge_by_physical=None
        )
        if len(logical_val) != 1000:
            raise FastPathError("logical val result geometry must be exactly 1,000")
        atomic_write_jsonl(
            self.output_root / "reports" / "val-results.jsonl",
            logical_val,
            overwrite=True,
        )

    def _logical_rows(
        self,
        *,
        split: str,
        assistant_by_physical: Mapping[str, Mapping[str, AssistantObservation]],
        judge_by_physical: Mapping[str, Mapping[str, JudgeObservation]] | None,
    ) -> list[dict[str, object]]:
        queries = [query for query in self.queries() if query.split == split]
        rows: list[dict[str, object]] = []
        for config in CONFIGS:
            physical = self._physical_config(config)
            for query in queries:
                assistant = assistant_by_physical[physical][query.query_id]
                judge = (
                    None
                    if judge_by_physical is None
                    else judge_by_physical[physical][query.query_id]
                )
                rows.append(
                    {
                        "schema_version": 1,
                        "split": split,
                        "config": config,
                        "physical_config": physical,
                        "alias_of": None if physical == config else physical,
                        "query": {
                            "query_id": query.query_id,
                            "canonical_capability": query.canonical_capability,
                            "canonical_intent": query.canonical_intent,
                            "acceptable_capabilities": query.acceptable_capabilities,
                            "leakage_group_id": query.leakage_group_id,
                            "is_boundary": query.is_boundary,
                            "boundary_strategy": query.boundary_strategy,
                            "template_family": query.template_family,
                        },
                        "assistant": assistant.model_dump(mode="json"),
                        "judge": None
                        if judge is None
                        else judge.model_dump(mode="json"),
                    }
                )
        return rows

    def _run_all_route_report(self) -> None:
        output = self.output_root / "reports" / "route-all1500.jsonl"
        if output.exists():
            raw_logical = _read_jsonl(output)
            if len(raw_logical) != 4500 or not all(
                isinstance(row, dict) for row in raw_logical
            ):
                raise FastPathError("route-all1500 must contain 4,500 logical rows")
            logical = [row for row in raw_logical if isinstance(row, dict)]
        else:
            queries = list(self.queries())
            logical = []
            physical_cache: dict[tuple[str, str], str | None] = {}
            for config in ("llm_static", "s1", "s1s2"):
                physical = self._physical_config(config)
                bank = self._bank_for_config(config)
                for query in queries:
                    key = (physical, query.query_id)
                    if key not in physical_cache:
                        execution_label = (
                            self._val_run_label(physical)
                            if query.split == "val"
                            else physical
                        )
                        assistant_result = self.calls.get(
                            "assistant",
                            self._assistant_call_id(
                                query.split, execution_label, query.query_id
                            ),
                        )
                        if assistant_result is not None:
                            route = _observation_from_result(
                                assistant_result, query.query_id
                            ).selected_capability
                        elif physical == "llm_static" and query.split == "opt_pool":
                            # The normalized opt800 Static input is already a real
                            # end-to-end result and must not be paid for twice.
                            route = self.opt_static()[
                                query.query_id
                            ].selected_capability
                        else:
                            result = self._call(
                                role="route_only",
                                call_id=_safe_id(
                                    f"all1500-{physical}-{query.query_id}"
                                ),
                                purpose="post-freeze descriptive route-only evaluation",
                                payload={
                                    "operation": "route_only",
                                    "config": physical,
                                    "query": query.model_dump(mode="json"),
                                    "bank": None
                                    if bank is None
                                    else bank.model_dump(mode="json"),
                                    "runtime_paths": self.spec.paths.model_dump(
                                        mode="json"
                                    ),
                                    "bank_frozen": True,
                                },
                            )
                            route = (
                                result.output.get("selected_capability")
                                if result.status == "success" and result.output
                                else None
                            )
                        physical_cache[key] = route if isinstance(route, str) else None
                    logical.append(
                        {
                            "query_id": query.query_id,
                            "split": query.split,
                            "config": config,
                            "physical_config": physical,
                            "selected_capability": physical_cache[key],
                            "expected_capability": query.canonical_capability,
                            "is_boundary": query.is_boundary,
                        }
                    )
            atomic_write_jsonl(output, logical)
        by_config = {
            config: [row for row in logical if row["config"] == config]
            for config in ("llm_static", "s1", "s1s2")
        }
        route_summary: dict[str, object] = {}
        for config, rows in by_config.items():
            f1_scores: list[float] = []
            confusion = {
                expected: {predicted: 0 for predicted in (*CAPABILITIES, "<none>")}
                for expected in CAPABILITIES
            }
            for row in rows:
                expected = str(row["expected_capability"])
                predicted = row["selected_capability"]
                confusion[expected][
                    str(predicted) if predicted is not None else "<none>"
                ] += 1
            for capability in CAPABILITIES:
                tp = fp = fn = 0
                for row in rows:
                    expected = row["expected_capability"]
                    predicted = row["selected_capability"]
                    tp += expected == capability and predicted == capability
                    fp += expected != capability and predicted == capability
                    fn += expected == capability and predicted != capability
                denominator = 2 * tp + fp + fn
                f1_scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
            route_summary[config] = {
                "query_count": len(rows),
                "route_macro_f1": sum(f1_scores) / len(f1_scores),
                "confusion": confusion,
            }
        atomic_write_json(
            self.output_root / "reports" / "route-all1500-summary.json",
            route_summary,
        )

    def run_test(self) -> None:
        self.run_full()
        self._finalize_val_results()
        test_queries = [
            query for query in self.queries() if query.split == "test_frozen"
        ]
        test_physical = {self._physical_config(config) for config in CONFIGS}
        test_rows: dict[str, dict[str, AssistantObservation]] = {}
        test_judges: dict[str, dict[str, JudgeObservation]] = {}
        for physical in test_physical:
            observations = self._assistant_many(
                split="test_frozen",
                config=physical,
                queries=test_queries,
                bank=self._bank_for_config(physical),
            )
            test_rows[physical] = observations
            test_judges[physical] = self._judge_many(
                split="test_frozen",
                config=physical,
                queries=test_queries,
                observations=observations,
            )
        # Assemble from the durable call records so a later `report` command
        # follows exactly the same no-resampling path.
        self.rebuild_logical_results()
        self._run_all_route_report()
        self.report()

    # ------------------------------- public API ----------------------------

    def run(self, *, through: str) -> None:
        if through not in {"s1", "s2", "full", "test"}:
            raise ValueError(f"unknown through stage: {through}")
        self.initialize()
        self.run_s1()
        if through == "s1":
            return
        self.run_s2()
        if through == "s2":
            return
        self.run_full()
        if through == "full":
            self._finalize_val_results()
            return
        self.run_test()

    def report(self) -> dict[str, object]:
        from .reporting import build_report

        self.rebuild_logical_results()
        return build_report(self)


__all__ = ["CoreFastEngine", "FastPathError", "ValidationSummary"]
