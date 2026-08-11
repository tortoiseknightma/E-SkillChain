from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import ValidationError

from skillchain.evaluation.evaluator_outputs import VisualFeedbackOutput
from skillchain.schemas import Query
from skillchain.static_authoring import StaticBankArtifact, render_skill_markdown
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

from .adapters import CoreFastAdapter
from .models import (
    AssistantObservation,
    CAPABILITIES,
    CONFIGS,
    CallIntent,
    CallResult,
    CallRole,
    CoreFastSpec,
    JudgeObservation,
    SPLIT_COUNTS,
    StageDecision,
)
from .store import (
    CallStore,
    FastStoreError,
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
)


class FastPathError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidationSummary:
    query_count: int
    split_counts: dict[str, int]
    capability_count: int
    static_bank_sha256: str
    observed_cost_cny: float
    projected_worst_case_cost_cny: float
    runtime_ready: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "query_count": self.query_count,
            "split_counts": self.split_counts,
            "capability_count": self.capability_count,
            "static_bank_sha256": self.static_bank_sha256,
            "observed_cost_cny": self.observed_cost_cny,
            "projected_worst_case_cost_cny": self.projected_worst_case_cost_cny,
            "runtime_ready": self.runtime_ready,
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
            gcs_components={
                "route_acceptable": False,
                "no_hard_error": False,
                "tool_contract_pass": False,
                "evidence_grounded": False,
                "output_contract_pass": False,
            },
            gcs_score=0.0,
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
            gcs_components={
                "route_acceptable": False,
                "no_hard_error": False,
                "tool_contract_pass": False,
                "evidence_grounded": False,
                "output_contract_pass": False,
            },
            gcs_score=0.0,
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
        self._queries: tuple[Query, ...] | None = None
        self._query_by_id: dict[str, Query] | None = None
        self._static_bank: StaticBankArtifact | None = None
        self._opt_static: dict[str, AssistantObservation] | None = None
        self._opt_attribution: dict[str, dict[str, object]] | None = None

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
        try:
            self._static_bank = StaticBankArtifact.model_validate_json(
                path.read_bytes(), strict=True
            )
        except (OSError, ValidationError) as error:
            raise FastPathError(f"invalid Static Bank: {path}") from error
        self._require_six_capabilities(self._static_bank)
        return self._static_bank

    def opt_static(self) -> dict[str, AssistantObservation]:
        if self._opt_static is not None:
            return self._opt_static
        path = self._path("opt_static_results")
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
        self._opt_static = observations
        return observations

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
        opt_static = self.opt_static()
        self.opt_attribution()
        by_id = self.query_by_id()
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
            observation = opt_static[sample.query_id]
            if sample.role == "failure" and not (
                observation.hard_error or observation.gcs_score < 1.0
            ):
                raise FastPathError(
                    f"canary failure is not a recorded failure: {sample.query_id}"
                )
            if sample.role == "anchor" and observation.hard_error:
                raise FastPathError(
                    f"canary anchor has a hard error: {sample.query_id}"
                )
        for sample in self.spec.fixed_samples.body48:
            observation = opt_static[sample.query_id]
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
        runtime_ready = True
        # Deliberately coarse call-count ceiling, not a per-call reservation
        # proof.  It includes every physical candidate, all 1,500 post-freeze
        # route-only gaps, and one format retry for every Judge call.
        max_calls = {
            "feedback": 12,
            "creator": 3,
            "assistant": 424 + 224 + 296 + 200 + 1500,
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
                "queries": _file_sha(self._path("queries")),
                "static_bank": _file_sha(self._path("static_bank")),
                "opt_static_results": _file_sha(self._path("opt_static_results")),
                "opt_route_attribution": _file_sha(self._path("opt_route_attribution")),
                "feedback_schema": sha256_bytes(
                    canonical_json_bytes(VisualFeedbackOutput.model_json_schema())
                ),
                "assistant_result_schema": sha256_bytes(
                    canonical_json_bytes(AssistantObservation.model_json_schema())
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

    # ------------------------------- calls ---------------------------------

    def _call(
        self,
        *,
        role: CallRole,
        call_id: str,
        purpose: str,
        payload: dict[str, object],
    ) -> CallResult:
        if role == "feedback" and self.calls.get(role, call_id) is None:
            if self.calls.role_count(role) >= self.spec.limits.max_feedback_calls:
                raise FastPathError("Feedback call limit reached")
        if role == "creator" and self.calls.get(role, call_id) is None:
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
            payload["reuse_parent_route_and_tool"] = {
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

    def _assistant_many(
        self,
        *,
        split: str,
        config: str,
        queries: Sequence[Query],
        bank: StaticBankArtifact | None,
        reuse_parent: Mapping[str, AssistantObservation] | None = None,
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
    ) -> StaticBankArtifact | None:
        if (
            result.status != "success"
            or not result.schema_valid
            or result.output is None
        ):
            return None
        model_payload = result.output
        full_payload = result.output.get("bank")
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
            if stage == "s1":
                model_payload = {
                    "skills": [
                        {
                            "capability_id": capability,
                            "description": supplied_by_capability[
                                capability
                            ].description,
                            "body": supplied_by_capability[capability].body,
                            "static_refs": supplied_by_capability[
                                capability
                            ].static_refs,
                            "operators": supplied_by_capability[capability].operators,
                        }
                        for capability in CAPABILITIES
                    ]
                }
            else:
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

    def _s1_gate(
        self,
        parent: Mapping[str, AssistantObservation],
        candidate: Mapping[str, AssistantObservation],
        queries: Sequence[Query],
    ) -> tuple[bool, tuple[str, ...], dict[str, object]]:
        before = self._summary(parent, queries)
        after = self._summary(candidate, queries)
        reasons: list[str] = []
        if after["capability_macro_gcs"] <= before["capability_macro_gcs"]:
            reasons.append("capability-macro GCS did not strictly improve")
        if after["hard_errors"] > before["hard_errors"]:
            reasons.append("hard errors increased")
        floor = self.spec.gates.s1_max_capability_drop_pp / 100
        for capability in CAPABILITIES:
            delta = (
                after["capability_gcs"][capability]
                - before["capability_gcs"][capability]
            )
            if delta < -floor - 1e-12:
                reasons.append(f"{capability} declined by more than 3pp")
        return not reasons, tuple(reasons), {"parent": before, "candidate": after}

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

    @staticmethod
    def _creator_schema(stage: str) -> dict[str, object]:
        capability_enum = list(CAPABILITIES)
        if stage == "s1":
            item = {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "capability_id",
                    "description",
                    "body",
                    "static_refs",
                    "operators",
                ],
                "properties": {
                    "capability_id": {"type": "string", "enum": capability_enum},
                    "description": {"type": "string", "minLength": 1},
                    "body": {"type": "string", "minLength": 1},
                    "static_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "operators": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            }
            return {
                "type": "object",
                "additionalProperties": False,
                "required": ["skills"],
                "properties": {
                    "skills": {
                        "type": "array",
                        "minItems": 6,
                        "maxItems": 6,
                        "items": item,
                    }
                },
            }
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

    def run_s1(self) -> StageDecision:
        existing = self._existing_decision("s1")
        if existing is not None:
            return existing
        parent = self.static_bank()
        opt = self.opt_static()
        query_by_id = self.query_by_id()

        feedback: dict[str, object] = {}
        with ThreadPoolExecutor(max_workers=self.spec.concurrency.feedback) as pool:
            futures = {}
            for sample in self.spec.fixed_samples.canary12:
                query = query_by_id[sample.query_id]
                call_id = _safe_id(f"canary-{sample.query_id}")
                payload = {
                    "operation": "strict_visual_feedback",
                    "query": query.model_dump(mode="json"),
                    "baseline": opt[sample.query_id].model_dump(mode="json"),
                    "sample_role": sample.role,
                    "response_schema": VisualFeedbackOutput.model_json_schema(),
                    "runtime_paths": self.spec.paths.model_dump(mode="json"),
                    "no_replacement": True,
                }
                futures[
                    pool.submit(
                        self._call,
                        role="feedback",
                        call_id=call_id,
                        purpose=f"S1 fixed canary Feedback {sample.query_id}",
                        payload=payload,
                    )
                ] = sample
            for future in as_completed(futures):
                sample = futures[future]
                result = future.result()
                parsed: object | None = None
                if result.status == "success" and result.schema_valid and result.output:
                    raw = result.output.get("feedback", result.output)
                    try:
                        parsed = VisualFeedbackOutput.model_validate_json(
                            canonical_json_bytes(raw), strict=True
                        ).model_dump(mode="json")
                    except ValidationError:
                        parsed = None
                feedback[sample.query_id] = {
                    "capability": sample.capability,
                    "sample_role": sample.role,
                    "status": result.status,
                    "feedback": parsed,
                    "failure_reason": result.failure_reason,
                }

        grouped = {
            capability: [
                feedback[sample.query_id]
                for sample in self.spec.fixed_samples.canary12
                if sample.capability == capability
            ]
            for capability in CAPABILITIES
        }
        creator = self._call(
            role="creator",
            call_id="s1-creator-once",
            purpose="S1 failure-driven six-capability Creator",
            payload={
                "operation": "s1_creator",
                "parent_bank": parent.model_dump(mode="json"),
                "feedback_by_capability": grouped,
                "feedback_success_count": sum(
                    item["feedback"] is not None for item in feedback.values()
                ),
                "feedback_missing_capabilities": [
                    capability
                    for capability, rows in grouped.items()
                    if not any(row["feedback"] is not None for row in rows)
                ],
                "requirements": {
                    "complete_six_capability_bank": True,
                    "single_candidate": True,
                },
                "output_schema": self._creator_schema("s1"),
            },
        )
        candidate = self._candidate_bank(creator, stage="s1", parent=parent)
        reasons: list[str] = []
        metrics: dict[str, object] = {
            "feedback_success_count": sum(
                item["feedback"] is not None for item in feedback.values()
            )
        }
        accepted = False
        if candidate is None:
            reasons.append("Creator failed or returned an invalid six-capability Bank")
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
            if not self._smoke_ok(smoke):
                reasons.append("candidate failed fixed dev smoke24")
            else:
                val_queries = [
                    query for query in self.queries() if query.split == "val"
                ]
                static_rows = self._assistant_many(
                    split="val",
                    config="llm_static",
                    queries=val_queries,
                    bank=parent,
                )
                candidate_rows = self._assistant_many(
                    split="val",
                    config="s1-candidate",
                    queries=val_queries,
                    bank=candidate,
                )
                accepted, gate_reasons, gate_metrics = self._s1_gate(
                    static_rows, candidate_rows, val_queries
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
