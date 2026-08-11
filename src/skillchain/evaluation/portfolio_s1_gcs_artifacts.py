"""Export deeply verified GCS-v2 inputs for the offline S1 gates.

Historical replay populations reuse completed Static opt800 checkpoints.  A
sparse Round 2 development replay instead scores both physical Banks in the
same fresh execution, just like the body gate.  This module turns those
execution shapes into the same byte-bound files consumed by
:mod:`skillchain.evolution.s1_gcs_gate`.  It performs no provider calls and
publishes into a new directory only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from types import SimpleNamespace
from types import MappingProxyType
from typing import Annotated, Literal, Mapping, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from scripts import run_portfolio_shard as established
from skillchain.evaluation.assistant_runs import AssistantRequestSnapshot
from skillchain.evaluation.portfolio_gcs import (
    GCS_CAPABILITY_ORDER,
    GCS_V2_POLICY_SHA256,
    GCSQueryScoreV2,
    build_gcs_population_v2,
    portfolio_gcs_oracles_v2,
    score_portfolio_gcs_v2,
)
from skillchain.evaluation.portfolio_launch import (
    load_portfolio_launch_package,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_s1_experiment_runtime import (
    S1_BODY_GATE_SCOPE,
    S1_EXPERIMENT_CONTROL_KIND,
    S1_OPT_REPLAY_SCOPE,
    VerifiedPortfolioS1ExperimentLaunch,
    VerifiedPortfolioS1ExperimentRuntime,
    load_verified_portfolio_s1_experiment_launch,
    load_verified_portfolio_s1_experiment_runtime_evidence,
    validate_portfolio_s1_experiment_evidence_control,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import (
    VerifiedStaticGCSCorpus,
    load_verified_static_gcs_corpus,
)
from skillchain.evolution.s1_gcs_gate import (
    S1_GCS_GATE_POLICY_SHA256,
    S1_GCS_GATE_POLICY_VERSION,
    S1GCSEvidenceBinding,
    S1Round2ResponseContractDiagnostic,
    S1Round2ResponseContractDiagnostics,
    make_s1_gcs_evidence_binding,
    make_s1_round2_response_contract_diagnostic,
    make_s1_round2_response_contract_diagnostics,
)
from skillchain.schemas import Query
from skillchain.static_authoring import StaticBankArtifact
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.task_spec import (
    MVP_TASK_SPEC_V1_PATH,
    TaskSpecification,
    load_mvp_task_specification_v1,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)


S1_GCS_ARTIFACT_POLICY_VERSION = "portfolio-s1-gcs-gate-inputs-v1"
S1GCSArtifactPhase = Literal["replay", "body_gate"]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_ANALYZER_CLI_PATH = _REPOSITORY_ROOT / "scripts/analyze_portfolio_s1_population.py"
_GATE_CLI_PATH = _REPOSITORY_ROOT / "scripts/run_portfolio_s1_gcs_gate.py"
_GATE_MODULE_PATH = _REPOSITORY_ROOT / "src/skillchain/evolution/s1_gcs_gate.py"
_EXPORT_MANIFEST_NAME = "export-manifest.json"
_EXPORT_FILE_NAMES = frozenset(
    {
        "queries.jsonl",
        "baseline-scores.jsonl",
        "candidate-scores.jsonl",
        "population-binding.json",
        "evidence-binding.json",
        "parent-bank.json",
        "candidate-bank.json",
    }
)
_EXPECTED_GEOMETRY = {
    "replay": (S1_OPT_REPLAY_SCOPE, "opt_pool", 200, ("s1",)),
    "body_gate": (
        S1_BODY_GATE_SCOPE,
        "val",
        75,
        ("llm_static", "s1"),
    ),
}


def _expected_config_order(
    phase: S1GCSArtifactPhase, launch_plan: Mapping[str, object]
) -> tuple[str, ...]:
    """Return the exact physical execution geometry for this launch lineage."""

    if phase == "replay" and launch_plan.get("sparse_development_paired") is True:
        return ("llm_static", "s1")
    return _EXPECTED_GEOMETRY[phase][3]


def _same_execution_baseline(
    scored: Mapping[str, tuple[GCSQueryScoreV2, ...]],
    *,
    source_kind: str,
    control_sha256: object,
) -> tuple[tuple[GCSQueryScoreV2, ...], dict[str, object]]:
    """Bind the baseline to the physical Static rows from this execution."""

    baseline = scored.get("llm_static", ())
    return baseline, {
        "source_kind": source_kind,
        "source_execution_control_sha256": control_sha256,
        "provider_model_call_count": 0,
    }


def _fresh_execution_baseline(
    scored: Mapping[str, tuple[GCSQueryScoreV2, ...]],
    *,
    phase: S1GCSArtifactPhase,
    sparse_development_paired: bool,
    control_sha256: object,
    static_execution_root: str | Path | None,
    static_expected_control_file_sha256: str | None,
) -> tuple[tuple[GCSQueryScoreV2, ...], dict[str, object]] | None:
    """Choose a fresh physical Static baseline when the launch requires one."""

    if phase == "replay":
        if not sparse_development_paired:
            return None
        if (
            static_execution_root is not None
            or static_expected_control_file_sha256 is not None
        ):
            raise PortfolioS1GCSArtifactError(
                "paired sparse replay forbids a historical Static baseline"
            )
        source_kind = "same-development-replay-execution-physical-llm-static"
    else:
        source_kind = "same-body-gate-execution-physical-llm-static"
    return _same_execution_baseline(
        scored,
        source_kind=source_kind,
        control_sha256=control_sha256,
    )


class PortfolioS1GCSArtifactError(ValueError):
    """The S1 execution, Static baseline, or GCS output was not trustworthy."""


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PortfolioS1GCSExportFileV1(_StrictFrozenModel):
    file_sha256: Sha256
    size_bytes: int = Field(ge=0)


class PortfolioS1GCSExportManifestV1(_StrictFrozenModel):
    """External-rooted exact inventory for a gate input publication."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-gcs-gate-export-manifest"] = (
        "portfolio-s1-gcs-gate-export-manifest"
    )
    policy_version: Literal[S1_GCS_ARTIFACT_POLICY_VERSION] = (
        S1_GCS_ARTIFACT_POLICY_VERSION
    )
    gate_policy_version: Literal[S1_GCS_GATE_POLICY_VERSION] = (
        S1_GCS_GATE_POLICY_VERSION
    )
    phase: S1GCSArtifactPhase
    execution_scope: str
    source_split: Literal["opt_pool", "val"]
    query_count: int = Field(gt=0)
    execution_control_file_sha256: Sha256
    execution_control_sha256: Sha256
    launch_plan_file_sha256: Sha256
    launch_plan_sha256: Sha256
    runtime_lock_file_sha256: Sha256
    runtime_lock_sha256: Sha256
    exporter_file_sha256: Sha256
    analyzer_cli_file_sha256: Sha256
    gate_cli_file_sha256: Sha256
    gate_module_file_sha256: Sha256
    gate_policy_sha256: Sha256
    files: dict[str, PortfolioS1GCSExportFileV1]
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if set(self.files) != _EXPORT_FILE_NAMES:
            raise ValueError("S1 GCS export manifest inventory is not exact")
        unsigned = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != sha256_bytes(canonical_json_bytes(unsigned)):
            raise ValueError("S1 GCS export manifest self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class PublishedS1GCSGateInputs:
    root: Path
    phase: S1GCSArtifactPhase
    query_count: int
    population_binding_sha256: str
    evidence_binding_sha256: str
    baseline_scores_file_sha256: str
    candidate_scores_file_sha256: str
    export_manifest_file_sha256: str


@dataclass(frozen=True)
class VerifiedS1GCSGateExport:
    """A stable byte snapshot of one externally-rooted gate export."""

    root: Path
    manifest: PortfolioS1GCSExportManifestV1
    manifest_file_sha256: str
    _files: Mapping[str, bytes]

    def content(self, name: str) -> bytes:
        try:
            return self._files[name]
        except KeyError as error:
            raise PortfolioS1GCSArtifactError(
                f"S1 GCS export has no verified {name}"
            ) from error


def _verified_source_digest(path: Path, *, label: str) -> str:
    return sha256_bytes(read_stable_regular_file(path, label=label, max_bytes=8 << 20))


def _build_export_manifest(
    *,
    phase: S1GCSArtifactPhase,
    execution_scope: str,
    source_split: Literal["opt_pool", "val"],
    query_count: int,
    control: Mapping[str, object],
    control_file_sha256: str,
    launch: VerifiedPortfolioS1ExperimentLaunch,
    runtime: VerifiedPortfolioS1ExperimentRuntime,
    files: Mapping[str, bytes],
) -> PortfolioS1GCSExportManifestV1:
    if set(files) != _EXPORT_FILE_NAMES:
        raise PortfolioS1GCSArtifactError(
            "S1 GCS export manifest requires the exact gate input inventory"
        )
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-gcs-gate-export-manifest",
        "policy_version": S1_GCS_ARTIFACT_POLICY_VERSION,
        "gate_policy_version": S1_GCS_GATE_POLICY_VERSION,
        "phase": phase,
        "execution_scope": execution_scope,
        "source_split": source_split,
        "query_count": query_count,
        "execution_control_file_sha256": control_file_sha256,
        "execution_control_sha256": control["control_sha256"],
        "launch_plan_file_sha256": launch.plan_file_sha256,
        "launch_plan_sha256": launch.plan["launch_plan_sha256"],
        "runtime_lock_file_sha256": runtime.runtime_lock_file_sha256,
        "runtime_lock_sha256": runtime.runtime_lock["runtime_lock_sha256"],
        "exporter_file_sha256": _verified_source_digest(
            Path(__file__).resolve(), label="active S1 GCS artifact exporter"
        ),
        "analyzer_cli_file_sha256": _verified_source_digest(
            _ANALYZER_CLI_PATH, label="active S1 GCS analyzer CLI"
        ),
        "gate_cli_file_sha256": _verified_source_digest(
            _GATE_CLI_PATH, label="active S1 GCS gate CLI"
        ),
        "gate_module_file_sha256": _verified_source_digest(
            _GATE_MODULE_PATH, label="active S1 GCS gate module"
        ),
        "gate_policy_sha256": S1_GCS_GATE_POLICY_SHA256,
        "files": {
            name: {
                "file_sha256": sha256_bytes(content),
                "size_bytes": len(content),
            }
            for name, content in sorted(files.items())
        },
    }
    try:
        return PortfolioS1GCSExportManifestV1.model_validate(
            {
                **payload,
                "manifest_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioS1GCSArtifactError(
            "S1 GCS export manifest could not be constructed"
        ) from error


def _validate_population_binding(
    content: bytes,
    *,
    manifest: PortfolioS1GCSExportManifestV1,
) -> dict[str, object]:
    try:
        raw = parse_canonical_json(content, label="S1 GCS population binding")
    except ArtifactFormatError as error:
        raise PortfolioS1GCSArtifactError(
            "S1 GCS population binding is invalid"
        ) from error
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise PortfolioS1GCSArtifactError("S1 GCS population binding is not canonical")
    supplied = raw.get("binding_sha256")
    unsigned = dict(raw)
    unsigned.pop("binding_sha256", None)
    expected = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-gcs-population-binding",
        "policy_version": S1_GCS_ARTIFACT_POLICY_VERSION,
        "phase": manifest.phase,
        "execution_scope": manifest.execution_scope,
        "source_split": manifest.source_split,
        "query_count": manifest.query_count,
        "execution_control_file_sha256": manifest.execution_control_file_sha256,
        "execution_control_sha256": manifest.execution_control_sha256,
        "launch_plan_file_sha256": manifest.launch_plan_file_sha256,
        "launch_plan_sha256": manifest.launch_plan_sha256,
        "runtime_lock_file_sha256": manifest.runtime_lock_file_sha256,
        "runtime_lock_sha256": manifest.runtime_lock_sha256,
        "s1_gcs_artifacts_file_sha256": manifest.exporter_file_sha256,
        "analyzer_cli_file_sha256": manifest.analyzer_cli_file_sha256,
        "gate_cli_file_sha256": manifest.gate_cli_file_sha256,
        "gate_module_file_sha256": manifest.gate_module_file_sha256,
        "gate_policy_sha256": manifest.gate_policy_sha256,
    }
    if any(
        raw.get(key) != value for key, value in expected.items()
    ) or supplied != sha256_bytes(canonical_json_bytes(unsigned)):
        raise PortfolioS1GCSArtifactError(
            "S1 GCS population binding differs from its export manifest"
        )
    query_ids = raw.get("query_ids")
    if (
        not isinstance(query_ids, list)
        or len(query_ids) != manifest.query_count
        or len(set(query_ids)) != manifest.query_count
        or not all(isinstance(value, str) and value for value in query_ids)
    ):
        raise PortfolioS1GCSArtifactError(
            "S1 GCS population binding query inventory is invalid"
        )
    return raw


def _validate_export_semantics(
    contents: Mapping[str, bytes],
    *,
    manifest: PortfolioS1GCSExportManifestV1,
    population_binding: Mapping[str, object],
) -> None:
    try:
        query_rows = parse_canonical_jsonl(
            contents["queries.jsonl"], label="S1 GCS exported queries"
        )
        queries = tuple(Query.model_validate(row, strict=True) for row in query_rows)
        baseline_rows = parse_canonical_jsonl(
            contents["baseline-scores.jsonl"],
            label="S1 GCS exported baseline scores",
        )
        baseline = tuple(
            GCSQueryScoreV2.model_validate(row, strict=True) for row in baseline_rows
        )
        candidate_rows = parse_canonical_jsonl(
            contents["candidate-scores.jsonl"],
            label="S1 GCS exported candidate scores",
        )
        candidate = tuple(
            GCSQueryScoreV2.model_validate(row, strict=True) for row in candidate_rows
        )
        parent = StaticBankArtifact.model_validate_json(
            contents["parent-bank.json"], strict=True
        )
        candidate_bank = StaticBankArtifact.model_validate_json(
            contents["candidate-bank.json"], strict=True
        )
    except (ArtifactFormatError, ValidationError) as error:
        raise PortfolioS1GCSArtifactError(
            "S1 GCS export contains an invalid typed artifact"
        ) from error
    query_ids = [query.query_id for query in queries]
    population = build_gcs_population_v2(queries)
    if (
        canonical_jsonl_bytes(tuple(query.model_dump(mode="json") for query in queries))
        != contents["queries.jsonl"]
        or canonical_jsonl_bytes(tuple(row.model_dump(mode="json") for row in baseline))
        != contents["baseline-scores.jsonl"]
        or canonical_jsonl_bytes(
            tuple(row.model_dump(mode="json") for row in candidate)
        )
        != contents["candidate-scores.jsonl"]
        or parent.canonical_bytes() != contents["parent-bank.json"]
        or candidate_bank.canonical_bytes() != contents["candidate-bank.json"]
        or len(queries) != manifest.query_count
        or population_binding.get("query_ids") != query_ids
        or population_binding.get("population") != population.model_dump(mode="json")
        or population_binding.get("parent_bank_sha256") != parent.bank_sha256
        or population_binding.get("candidate_bank_sha256") != candidate_bank.bank_sha256
        or [row.query_id for row in baseline] != query_ids
        or [row.query_id for row in candidate] != query_ids
        or any(row.config != "llm_static" for row in baseline)
        or any(row.config != "s1" for row in candidate)
    ):
        raise PortfolioS1GCSArtifactError(
            "S1 GCS export population does not close over queries, scores, and Banks"
        )
    raw_diagnostics = population_binding.get(
        "round2_response_contract_diagnostics"
    )
    if raw_diagnostics is not None:
        try:
            diagnostics = S1Round2ResponseContractDiagnostics.model_validate(
                raw_diagnostics, strict=True
            )
        except ValidationError as error:
            raise PortfolioS1GCSArtifactError(
                "S1 GCS Round 2 contract diagnostics are invalid"
            ) from error
        if diagnostics.query_ids != tuple(query_ids):
            raise PortfolioS1GCSArtifactError(
                "S1 GCS Round 2 contract diagnostics differ from queries"
            )


def load_verified_portfolio_s1_gcs_gate_export(
    root: str | Path,
    *,
    expected_manifest_file_sha256: str,
    expected_phase: S1GCSArtifactPhase | None = None,
) -> VerifiedS1GCSGateExport:
    """Load an exact gate publication rooted by an out-of-band manifest SHA."""

    supplied_root = Path(root).absolute()
    if supplied_root.is_symlink():
        raise PortfolioS1GCSArtifactError("S1 GCS export root must not be a symlink")
    try:
        verified_root = supplied_root.resolve(strict=True)
    except OSError as error:
        raise PortfolioS1GCSArtifactError(
            "S1 GCS export root is unavailable"
        ) from error
    if not verified_root.is_dir():
        raise PortfolioS1GCSArtifactError("S1 GCS export root is not a directory")
    expected_inventory = _EXPORT_FILE_NAMES | {_EXPORT_MANIFEST_NAME}
    entries = tuple(verified_root.iterdir())
    if {path.name for path in entries} != expected_inventory or any(
        not path.is_file() or path.is_symlink() for path in entries
    ):
        raise PortfolioS1GCSArtifactError("S1 GCS export inventory drifted")
    manifest_bytes = read_stable_regular_file(
        verified_root / _EXPORT_MANIFEST_NAME,
        label="S1 GCS export manifest",
        max_bytes=4 << 20,
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_file_sha256:
        raise PortfolioS1GCSArtifactError("S1 GCS export manifest file SHA-256 drifted")
    try:
        manifest = PortfolioS1GCSExportManifestV1.model_validate_json(
            manifest_bytes, strict=True
        )
    except ValidationError as error:
        raise PortfolioS1GCSArtifactError(
            "S1 GCS export manifest is invalid"
        ) from error
    if manifest.canonical_bytes() != manifest_bytes:
        raise PortfolioS1GCSArtifactError("S1 GCS export manifest is not canonical")
    if expected_phase is not None and manifest.phase != expected_phase:
        raise PortfolioS1GCSArtifactError("S1 GCS export phase is not authorized")
    expected_scope, expected_split, expected_count, _ = _EXPECTED_GEOMETRY[
        manifest.phase
    ]
    if (
        manifest.execution_scope != expected_scope
        or manifest.source_split != expected_split
        or manifest.query_count != expected_count
    ):
        raise PortfolioS1GCSArtifactError("S1 GCS export geometry drifted")
    active_sources = {
        "exporter_file_sha256": _verified_source_digest(
            Path(__file__).resolve(), label="active S1 GCS artifact exporter"
        ),
        "analyzer_cli_file_sha256": _verified_source_digest(
            _ANALYZER_CLI_PATH, label="active S1 GCS analyzer CLI"
        ),
        "gate_cli_file_sha256": _verified_source_digest(
            _GATE_CLI_PATH, label="active S1 GCS gate CLI"
        ),
        "gate_module_file_sha256": _verified_source_digest(
            _GATE_MODULE_PATH, label="active S1 GCS gate module"
        ),
        "gate_policy_sha256": S1_GCS_GATE_POLICY_SHA256,
    }
    if any(getattr(manifest, key) != value for key, value in active_sources.items()):
        raise PortfolioS1GCSArtifactError("S1 GCS export source identity drifted")
    contents: dict[str, bytes] = {}
    for name in sorted(_EXPORT_FILE_NAMES):
        content = read_stable_regular_file(
            verified_root / name,
            label=f"S1 GCS export {name}",
            max_bytes=256 << 20,
        )
        record = manifest.files[name]
        if (
            len(content) != record.size_bytes
            or sha256_bytes(content) != record.file_sha256
        ):
            raise PortfolioS1GCSArtifactError(f"S1 GCS export {name} drifted")
        contents[name] = content
    if {path.name for path in verified_root.iterdir()} != expected_inventory:
        raise PortfolioS1GCSArtifactError(
            "S1 GCS export inventory changed while loading"
        )
    population_binding = _validate_population_binding(
        contents["population-binding.json"], manifest=manifest
    )
    _validate_export_semantics(
        contents,
        manifest=manifest,
        population_binding=population_binding,
    )
    try:
        evidence = S1GCSEvidenceBinding.model_validate_json(
            contents["evidence-binding.json"], strict=True
        )
    except ValidationError as error:
        raise PortfolioS1GCSArtifactError(
            "S1 GCS export evidence binding is invalid"
        ) from error
    if (
        evidence.canonical_bytes() != contents["evidence-binding.json"]
        or evidence.phase != manifest.phase
        or evidence.population_binding_file_sha256
        != manifest.files["population-binding.json"].file_sha256
        or evidence.queries_file_sha256 != manifest.files["queries.jsonl"].file_sha256
        or evidence.baseline_scores_file_sha256
        != manifest.files["baseline-scores.jsonl"].file_sha256
        or evidence.candidate_scores_file_sha256
        != manifest.files["candidate-scores.jsonl"].file_sha256
        or evidence.parent_bank_file_sha256
        != manifest.files["parent-bank.json"].file_sha256
        or evidence.candidate_bank_file_sha256
        != manifest.files["candidate-bank.json"].file_sha256
    ):
        raise PortfolioS1GCSArtifactError(
            "S1 GCS evidence binding differs from the verified files"
        )
    return VerifiedS1GCSGateExport(
        root=verified_root,
        manifest=manifest,
        manifest_file_sha256=expected_manifest_file_sha256,
        _files=MappingProxyType(contents),
    )


def _verify_publication(output: Path, files: Mapping[str, bytes]) -> None:
    if (
        not output.is_dir()
        or {path.name for path in output.iterdir() if path.is_file()} != set(files)
        or any(path.is_dir() or path.is_symlink() for path in output.iterdir())
    ):
        raise PortfolioS1GCSArtifactError("S1 GCS publication inventory drifted")
    for name, expected in files.items():
        observed = read_stable_regular_file(
            output / name,
            label=f"published S1 GCS {name}",
            max_bytes=256 << 20,
        )
        if observed != expected or sha256_bytes(observed) != sha256_bytes(expected):
            raise PortfolioS1GCSArtifactError(
                f"published S1 GCS {name} failed byte verification"
            )


def _canonical_control(
    execution_root: Path, *, expected_file_sha256: str
) -> tuple[bytes, dict[str, object]]:
    path = execution_root / "execution-control.json"
    try:
        content = read_stable_regular_file(
            path, label="S1 execution control", max_bytes=4 << 20
        )
        raw = parse_canonical_json(content, label="S1 execution control")
    except (OSError, ArtifactFormatError) as error:
        raise PortfolioS1GCSArtifactError(
            "S1 execution control cannot be read"
        ) from error
    if (
        sha256_bytes(content) != expected_file_sha256
        or not isinstance(raw, dict)
        or canonical_json_bytes(raw) != content
    ):
        raise PortfolioS1GCSArtifactError("S1 execution control bytes drifted")
    supplied = raw.get("control_sha256")
    unsigned = dict(raw)
    unsigned.pop("control_sha256", None)
    if (
        raw.get("schema_version") != 4
        or raw.get("kind") != S1_EXPERIMENT_CONTROL_KIND
        or supplied != sha256_bytes(canonical_json_bytes(unsigned))
    ):
        raise PortfolioS1GCSArtifactError("S1 execution control identity drifted")
    return content, raw


def _launch_facade(launch: VerifiedPortfolioS1ExperimentLaunch) -> SimpleNamespace:
    return SimpleNamespace(plan=SimpleNamespace(**dict(launch.plan)))


def _load_core_inputs(control: Mapping[str, object], plan: Mapping[str, object]):
    root = control.get("core_input_binding_launch_root")
    digest = control.get("core_input_binding_launch_plan_file_sha256")
    if not isinstance(root, str) or not isinstance(digest, str):
        raise PortfolioS1GCSArtifactError("S1 control lacks its Core input binding")
    source = load_portfolio_launch_package(root, expected_plan_file_sha256=digest)
    inputs = reconstruct_verified_portfolio_core_inputs(source.plan)
    if (
        inputs.expected_plan_sha256 != plan.get("portfolio_plan_sha256")
        or inputs.expected_query_artifact_sha256 != plan.get("query_artifact_sha256")
        or inputs.expected_capability_assignments_sha256
        != plan.get("capability_assignments_sha256")
        or inputs.expected_output_catalog_sha256 != plan.get("runtime_catalog_sha256")
    ):
        raise PortfolioS1GCSArtifactError(
            "S1 launch differs from the reconstructed Core inputs"
        )
    return inputs


def _task_spec(runtime: VerifiedPortfolioS1ExperimentRuntime) -> TaskSpecification:
    active = load_mvp_task_specification_v1()
    semantic = runtime.semantic_authoring_input
    try:
        raw = parse_canonical_json(
            semantic.task_specification.content,
            label="S1 embedded TaskSpec",
        )
        embedded = TaskSpecification.model_validate(raw, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise PortfolioS1GCSArtifactError("S1 embedded TaskSpec is invalid") from error
    lock = runtime.runtime_lock
    active_bytes = read_stable_regular_file(
        MVP_TASK_SPEC_V1_PATH, label="active S1 GCS TaskSpec", max_bytes=4 << 20
    )
    if (
        embedded != active
        or semantic.task_specification.identity_sha256 != embedded.task_spec_sha256
        or semantic.task_specification.version != embedded.task_spec_version
        or semantic.task_specification.content
        != canonical_json_bytes(embedded.model_dump(mode="json"))
        or lock.get("task_spec_version") != embedded.task_spec_version
        or lock.get("task_spec_sha256") != embedded.task_spec_sha256
        or lock.get("task_spec_file_sha256") != sha256_bytes(active_bytes)
        or lock.get("gcs_policy_sha256") != GCS_V2_POLICY_SHA256
    ):
        raise PortfolioS1GCSArtifactError("S1 TaskSpec/GCS binding drifted")
    return embedded


def _checkpoint_request(value: object) -> AssistantRequestSnapshot:
    return AssistantRequestSnapshot.model_validate_json(
        canonical_json_bytes(value), strict=True
    )


def _checkpoint_inventory_sha256(
    inventory: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> str:
    return sha256_bytes(canonical_json_bytes(list(inventory)))


def _load_candidate_scores(
    *,
    execution_root: Path,
    launch: VerifiedPortfolioS1ExperimentLaunch,
    runtime: VerifiedPortfolioS1ExperimentRuntime,
    query_by_id: Mapping[str, Query],
    assistant_query_by_id: Mapping[str, object],
    population,
    task_spec: TaskSpecification,
) -> tuple[
    dict[str, tuple[GCSQueryScoreV2, ...]],
    tuple[dict[str, object], ...],
    tuple[S1Round2ResponseContractDiagnostic, ...],
]:
    facade = _launch_facade(launch)
    oracles = portfolio_gcs_oracles_v2()
    rows: dict[str, list[GCSQueryScoreV2]] = {
        config: [] for config in launch.plan["config_order"]
    }
    inventory: list[dict[str, object]] = []
    contract_diagnostics: list[S1Round2ResponseContractDiagnostic] = []
    members = sorted(
        launch.instances,
        key=lambda item: (item.config, item.query_ordinal, item.query_id),
    )
    for member in members:
        query = query_by_id.get(member.query_id)
        public_query = assistant_query_by_id.get(member.query_id)
        if query is None or public_query is None:
            raise PortfolioS1GCSArtifactError("S1 launch references an unknown query")
        path = (execution_root / member.assistant_output_relpath).resolve(strict=True)
        if execution_root not in path.parents:
            raise PortfolioS1GCSArtifactError("S1 checkpoint escapes execution root")
        content = read_stable_regular_file(
            path, label=f"S1 checkpoint {member.query_id}", max_bytes=16 << 20
        )
        try:
            raw = parse_canonical_json(content, label="S1 checkpoint")
            request = _checkpoint_request(
                raw.get("request") if isinstance(raw, dict) else None
            )
        except (ArtifactFormatError, ValidationError) as error:
            raise PortfolioS1GCSArtifactError(
                "S1 checkpoint request is invalid"
            ) from error
        if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
            raise PortfolioS1GCSArtifactError("S1 checkpoint is not canonical")
        response, receipt, sidecar = established._assistant_row(
            path, require_schema_v2=True, include_scorer_evidence=True
        )
        if sidecar is None:
            raise AssertionError("schema-v2 checkpoint lost its scorer sidecar")
        result = established._assistant_result(facade, request, response)
        established._verify_scorer_sidecar_binding(
            sidecar,
            launch=facade,
            member=member,
            request=request,
            result=result,
            receipt=receipt,
        )
        expected_bank = runtime.banks[member.config].bank_sha256
        if (
            raw.get("schema_version") != 2
            or raw.get("kind") != "portfolio-assistant-checkpoint"
            or raw.get("instance_sha256") != member.instance_sha256
            or raw.get("query_ordinal") != member.query_ordinal
            or request.matrix_run_id != launch.plan["matrix_run_id"]
            or request.config != member.config
            or request.query_ordinal != member.query_ordinal
            or request.query != public_query
            or request.query.query_id != member.query_id
            or request.treatment.bank_sha256 != expected_bank
            or receipt.request_sha256 != request.request_sha256
            or sidecar.config != member.config
            or (execution_root / member.final_output_relpath).exists()
        ):
            raise PortfolioS1GCSArtifactError(
                "S1 checkpoint identity differs from its launch/runtime"
            )
        score = score_portfolio_gcs_v2(
            query,
            result,
            receipt,
            sidecar,
            task_spec,
            oracles,
            population=population,
        )
        rows[member.config].append(score)
        checkpoint_file_sha256 = sha256_bytes(content)
        contract_diagnostics.append(
            make_s1_round2_response_contract_diagnostic(
                query_id=member.query_id,
                config=member.config,
                checkpoint_file_sha256=checkpoint_file_sha256,
                checkpoint_row_sha256=raw["row_sha256"],
                response=response,
                receipt=receipt,
            )
        )
        inventory.append(
            {
                "query_id": member.query_id,
                "config": member.config,
                "instance_sha256": member.instance_sha256,
                "checkpoint_file_sha256": checkpoint_file_sha256,
                "checkpoint_row_sha256": raw["row_sha256"],
                "scorer_evidence_sha256": sidecar.evidence_sha256,
            }
        )
        if (
            read_stable_regular_file(
                path,
                label=f"S1 checkpoint recheck {member.query_id}",
                max_bytes=16 << 20,
            )
            != content
        ):
            raise PortfolioS1GCSArtifactError("S1 checkpoint changed while scoring")
    return (
        {
            config: tuple(sorted(values, key=lambda item: item.query_id))
            for config, values in rows.items()
        },
        tuple(inventory),
        tuple(contract_diagnostics),
    )


def _rescore_static_subset(
    corpus: VerifiedStaticGCSCorpus,
    *,
    queries: tuple[Query, ...],
    population,
    expected_parent_bank_sha256: str,
    task_spec: TaskSpecification,
) -> tuple[tuple[GCSQueryScoreV2, ...], dict[str, object]]:
    if (
        corpus.runtime.bank.bank_sha256 != expected_parent_bank_sha256
        or corpus.task_spec != task_spec
    ):
        raise PortfolioS1GCSArtifactError(
            "Static opt800 corpus differs from the S1 parent runtime"
        )
    source = corpus.row_by_query_id()
    oracles = portfolio_gcs_oracles_v2()
    scores: list[GCSQueryScoreV2] = []
    inventory: list[dict[str, object]] = []
    for query in queries:
        row = source.get(query.query_id)
        if row is None or row.query != query or row.request.config != "llm_static":
            raise PortfolioS1GCSArtifactError(
                "Static opt800 corpus lacks an exact replay query"
            )
        scores.append(
            score_portfolio_gcs_v2(
                query,
                row.result,
                row.receipt,
                row.sidecar,
                task_spec,
                oracles,
                population=population,
            )
        )
        inventory.append(
            {
                "query_id": query.query_id,
                "checkpoint_file_sha256": row.checkpoint_file_sha256,
                "checkpoint_row_sha256": row.checkpoint_row_sha256,
                "scorer_evidence_sha256": row.sidecar.evidence_sha256,
            }
        )
    return tuple(scores), {
        "source_kind": "verified-static-opt800-execution-v6-exact-query-reuse",
        "source_corpus_sha256": corpus.corpus_sha256,
        "source_control_file_sha256": corpus.control_file_sha256,
        "source_checkpoint_set_sha256": corpus.checkpoint_set_sha256,
        "source_sidecar_set_sha256": corpus.sidecar_set_sha256,
        "subset_checkpoint_inventory_sha256": _checkpoint_inventory_sha256(inventory),
        "provider_model_call_count": 0,
    }


def export_portfolio_s1_gcs_gate_inputs(
    execution_root: str | Path,
    *,
    expected_control_file_sha256: str,
    artifact_repository_root: str | Path,
    output_dir: str | Path,
    static_execution_root: str | Path | None = None,
    static_expected_control_file_sha256: str | None = None,
) -> PublishedS1GCSGateInputs:
    """Score one complete S1 population and publish direct gate inputs."""

    execution = Path(execution_root).absolute().resolve(strict=True)
    repository = Path(artifact_repository_root).absolute().resolve(strict=True)
    output = Path(output_dir).absolute()
    if execution != repository and repository not in execution.parents:
        raise PortfolioS1GCSArtifactError("S1 execution escapes artifact repository")
    control_bytes, control = _canonical_control(
        execution, expected_file_sha256=expected_control_file_sha256
    )
    launch = load_verified_portfolio_s1_experiment_launch(
        control["launch_root"],
        expected_plan_file_sha256=control["launch_plan_file_sha256"],
    )
    runtime = load_verified_portfolio_s1_experiment_runtime_evidence(
        control["runtime_root"],
        expected_runtime_lock_file_sha256=control["runtime_lock_file_sha256"],
    )
    validate_portfolio_s1_experiment_evidence_control(control, launch, runtime)
    scope = launch.plan["execution_mode"]
    phase: S1GCSArtifactPhase = (
        "replay" if scope == S1_OPT_REPLAY_SCOPE else "body_gate"
    )
    expected_scope, split, count, _legacy_configs = _EXPECTED_GEOMETRY[phase]
    configs = _expected_config_order(phase, launch.plan)
    if (
        scope != expected_scope
        or tuple(launch.plan["config_order"]) != configs
        or launch.plan["selected_split"] != split
        or control.get("pairwise_judge_enabled") is not False
        or control.get("legacy_final_judge_enabled") is not False
        or control.get("analyzer_provider_call_count") != 0
    ):
        raise PortfolioS1GCSArtifactError("S1 GCS execution scope drifted")
    inputs = _load_core_inputs(control, launch.plan)
    query_ids = {member.query_id for member in launch.instances}
    queries = tuple(
        sorted(
            (
                item
                for item in inputs.queries
                if item.query_id in query_ids and item.split == split
            ),
            key=lambda item: item.query_id,
        )
    )
    if (
        len(queries) != count
        or {item.query_id for item in queries} != query_ids
        or {item.canonical_capability for item in queries} != set(GCS_CAPABILITY_ORDER)
        or any(item.split == "test_frozen" for item in queries)
    ):
        raise PortfolioS1GCSArtifactError("S1 GCS query population is incomplete")
    population = build_gcs_population_v2(queries)
    task_spec = _task_spec(runtime)
    scored, candidate_inventory, diagnostic_rows = _load_candidate_scores(
        execution_root=execution,
        launch=launch,
        runtime=runtime,
        query_by_id={item.query_id: item for item in queries},
        assistant_query_by_id={
            item.query_id: item
            for item in inputs.assistant_queries
            if item.query_id in query_ids
        },
        population=population,
        task_spec=task_spec,
    )
    candidate = scored.get("s1", ())
    if len(candidate) != count:
        raise PortfolioS1GCSArtifactError("S1 candidate score population is incomplete")
    paired_sparse_replay = (
        phase == "replay" and launch.plan.get("sparse_development_paired") is True
    )
    fresh_baseline = _fresh_execution_baseline(
        scored,
        phase=phase,
        sparse_development_paired=paired_sparse_replay,
        control_sha256=control["control_sha256"],
        static_execution_root=static_execution_root,
        static_expected_control_file_sha256=(static_expected_control_file_sha256),
    )
    if fresh_baseline is not None:
        baseline, baseline_source = fresh_baseline
    elif phase == "replay":
        if static_execution_root is None or static_expected_control_file_sha256 is None:
            raise PortfolioS1GCSArtifactError(
                "replay export requires the completed Static opt800 execution"
            )
        corpus = load_verified_static_gcs_corpus(
            static_execution_root,
            expected_control_file_sha256=static_expected_control_file_sha256,
            artifact_repository_root=repository,
        )
        baseline, baseline_source = _rescore_static_subset(
            corpus,
            queries=queries,
            population=population,
            expected_parent_bank_sha256=runtime.banks["llm_static"].bank_sha256,
            task_spec=task_spec,
        )
    else:  # pragma: no cover - every body gate returns a fresh baseline above
        raise AssertionError("body gate lost its physical Static baseline")
    if (
        len(baseline) != count
        or {item.query_id for item in baseline} != query_ids
        or {item.query_id for item in candidate} != query_ids
        or any(item.config != "llm_static" for item in baseline)
        or any(item.config != "s1" for item in candidate)
    ):
        raise PortfolioS1GCSArtifactError("S1 paired GCS rows are not rectangular")

    response_contract_diagnostics: S1Round2ResponseContractDiagnostics | None = None
    if configs == ("llm_static", "s1"):
        response_contract_diagnostics = make_s1_round2_response_contract_diagnostics(
            query_ids=tuple(item.query_id for item in queries),
            rows=diagnostic_rows,
        )

    queries_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in queries)
    )
    baseline_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in baseline)
    )
    candidate_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in candidate)
    )
    parent_bank_bytes = runtime.banks["llm_static"].canonical_bytes()
    candidate_bank_bytes = runtime.banks["s1"].canonical_bytes()
    binding_payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-gcs-population-binding",
        "policy_version": S1_GCS_ARTIFACT_POLICY_VERSION,
        "phase": phase,
        "execution_scope": scope,
        "source_split": split,
        "query_count": count,
        "query_ids": [item.query_id for item in queries],
        "population": population.model_dump(mode="json"),
        "execution_control_file_sha256": sha256_bytes(control_bytes),
        "execution_control_sha256": control["control_sha256"],
        "launch_plan_file_sha256": launch.plan_file_sha256,
        "launch_plan_sha256": launch.plan["launch_plan_sha256"],
        "runtime_lock_file_sha256": runtime.runtime_lock_file_sha256,
        "runtime_lock_sha256": runtime.runtime_lock["runtime_lock_sha256"],
        "s1_gcs_artifacts_file_sha256": _verified_source_digest(
            Path(__file__).resolve(), label="active S1 GCS artifact exporter"
        ),
        "analyzer_cli_file_sha256": _verified_source_digest(
            _ANALYZER_CLI_PATH, label="active S1 GCS analyzer CLI"
        ),
        "gate_cli_file_sha256": _verified_source_digest(
            _GATE_CLI_PATH, label="active S1 GCS gate CLI"
        ),
        "gate_module_file_sha256": _verified_source_digest(
            _GATE_MODULE_PATH, label="active S1 GCS gate module"
        ),
        "gate_policy_sha256": S1_GCS_GATE_POLICY_SHA256,
        "parent_bank_sha256": runtime.banks["llm_static"].bank_sha256,
        "candidate_bank_sha256": runtime.banks["s1"].bank_sha256,
        "candidate_checkpoint_inventory_sha256": _checkpoint_inventory_sha256(
            candidate_inventory
        ),
        **(
            {}
            if response_contract_diagnostics is None
            else {
                "round2_response_contract_diagnostics": (
                    response_contract_diagnostics.model_dump(mode="json")
                )
            }
        ),
        "baseline_source": baseline_source,
        "pairwise_judge_call_count": 0,
        "legacy_final_judge_call_count": 0,
        "analyzer_provider_model_call_count": 0,
    }
    binding = {
        **binding_payload,
        "binding_sha256": sha256_bytes(canonical_json_bytes(binding_payload)),
    }
    binding_bytes = canonical_json_bytes(binding)
    evidence = make_s1_gcs_evidence_binding(
        phase=phase,
        population_binding_file_sha256=sha256_bytes(binding_bytes),
        queries_file_sha256=sha256_bytes(queries_bytes),
        baseline_scores_file_sha256=sha256_bytes(baseline_bytes),
        candidate_scores_file_sha256=sha256_bytes(candidate_bytes),
        parent_bank_file_sha256=sha256_bytes(parent_bank_bytes),
        candidate_bank_file_sha256=sha256_bytes(candidate_bank_bytes),
    )
    gate_files = {
        "queries.jsonl": queries_bytes,
        "baseline-scores.jsonl": baseline_bytes,
        "candidate-scores.jsonl": candidate_bytes,
        "population-binding.json": binding_bytes,
        "evidence-binding.json": evidence.canonical_bytes(),
        "parent-bank.json": parent_bank_bytes,
        "candidate-bank.json": candidate_bank_bytes,
    }
    manifest = _build_export_manifest(
        phase=phase,
        execution_scope=scope,
        source_split=split,
        query_count=count,
        control=control,
        control_file_sha256=sha256_bytes(control_bytes),
        launch=launch,
        runtime=runtime,
        files=gate_files,
    )
    manifest_bytes = manifest.canonical_bytes()
    files = {**gate_files, _EXPORT_MANIFEST_NAME: manifest_bytes}
    if output.exists():
        raise PortfolioS1GCSArtifactError("S1 GCS output directory already exists")
    staging = new_staging_directory(output)
    try:
        for name, content in files.items():
            atomic_create_file(staging / name, content)
        atomic_publish_new_directory(staging, output)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    _verify_publication(output, files)
    return PublishedS1GCSGateInputs(
        root=output,
        phase=phase,
        query_count=count,
        population_binding_sha256=sha256_bytes(binding_bytes),
        evidence_binding_sha256=evidence.binding_sha256,
        baseline_scores_file_sha256=sha256_bytes(baseline_bytes),
        candidate_scores_file_sha256=sha256_bytes(candidate_bytes),
        export_manifest_file_sha256=sha256_bytes(manifest_bytes),
    )


__all__ = [
    "PortfolioS1GCSArtifactError",
    "PortfolioS1GCSExportFileV1",
    "PortfolioS1GCSExportManifestV1",
    "PublishedS1GCSGateInputs",
    "S1_GCS_ARTIFACT_POLICY_VERSION",
    "VerifiedS1GCSGateExport",
    "export_portfolio_s1_gcs_gate_inputs",
    "load_verified_portfolio_s1_gcs_gate_export",
]
