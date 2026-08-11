"""Disk-level verification for matrix-ready Portfolio treatment runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Mapping

from pydantic import BaseModel, ValidationError

from skillchain.codex_authoring import CODEX_COMMAND_SHAPE
from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION,
    PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
    PortfolioModelInvocationReceipt,
    PortfolioRuntimeCompatibilityRebind,
    PortfolioStageGateReport,
    PortfolioStageGateResultSet,
    PortfolioStageMutation,
    PortfolioTreatmentChainManifest,
    PortfolioTreatmentConfig,
    PortfolioTreatmentError,
    VerifiedPortfolioTreatmentChain,
    apply_portfolio_stage_mutation,
    build_portfolio_execution_artifact_aliases,
    require_verified_portfolio_treatment_chain,
    verify_portfolio_runtime_compatibility_rebind_chain,
    verify_portfolio_treatment_chain,
    verify_portfolio_stage_gate_evidence,
)
from skillchain.evaluation.portfolio_attribution import (
    PortfolioAttributionError,
    PortfolioParentAttributionPacket,
    verify_portfolio_parent_attribution_gate_binding,
)
from skillchain.evaluation.portfolio_s3_textopt import (
    parse_portfolio_s3_rejected_edit_buffer,
    parse_portfolio_s3_text_patch_artifact,
)
from skillchain.static_authoring import StaticBankArtifact
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_ORDER: tuple[PortfolioTreatmentConfig, ...] = (
    "llm_static",
    "s1",
    "s1s2",
    "full",
)
_VERIFIED_RUNTIME_MARKER = object()
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_EVENT_BYTES = 16 * 1024 * 1024
_S3_LEGACY_RUNTIME_RECEIPT_CONTRACT = (
    "1.4.0",
    "portfolio-evolution-gate-bound-ephemeral-turn-v4",
)
_S3_TEXTOPT_RUNTIME_RECEIPT_CONTRACT = (
    "1.5.0",
    "portfolio-evolution-textopt-patch-turn-v5",
)
_SUPPORTED_S3_RUNTIME_RECEIPT_CONTRACTS = frozenset(
    {
        _S3_LEGACY_RUNTIME_RECEIPT_CONTRACT,
        _S3_TEXTOPT_RUNTIME_RECEIPT_CONTRACT,
    }
)
_S3_BASE_INPUT_ROLES = (
    "parent_bank",
    "prior_stage_gate",
    "stage_input",
)
_S3_TEXTOPT_BUFFER_INPUT_ROLES = (
    "parent_bank",
    "prior_stage_gate",
    "rejected_edit_buffer",
    "stage_input",
)
_S3_TEXTOPT_COMPILER_SNAPSHOT_FILE = "textopt-compiler-source.py"


@dataclass(frozen=True)
class VerifiedPortfolioTreatmentRuntime:
    """Opaque handle binding a runtime lock to a fully replayed treatment chain."""

    root: Path
    runtime_lock: Mapping[str, object]
    runtime_lock_file_sha256: str
    manifest_file_sha256: str
    chain: VerifiedPortfolioTreatmentChain
    _marker: object = field(repr=False)


def _safe_path(root: Path, relative: str, *, label: str) -> Path:
    candidate = root / Path(relative)
    resolved_root = root.resolve(strict=True)
    resolved_parent = candidate.parent.resolve(strict=True)
    if (
        resolved_parent != resolved_root
        and resolved_root not in resolved_parent.parents
    ):
        raise PortfolioTreatmentError(f"{label} escapes the treatment runtime root")
    return candidate


def _read_bound_file(
    root: Path,
    relative: str,
    expected_file_sha256: str,
    *,
    label: str,
    max_bytes: int = _MAX_JSON_BYTES,
) -> bytes:
    if not _SHA256_RE.fullmatch(expected_file_sha256):
        raise PortfolioTreatmentError(f"{label} expected SHA-256 is invalid")
    path = _safe_path(root, relative, label=label)
    try:
        content = read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    except (ArtifactFormatError, OSError) as error:
        raise PortfolioTreatmentError(f"{label} cannot be read safely") from error
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioTreatmentError(f"{label} file digest mismatch")
    return content


def _canonical_model(
    content: bytes,
    model_type: type[BaseModel],
    *,
    label: str,
) -> BaseModel:
    try:
        raw = parse_canonical_json(content, label=label)
        if not isinstance(raw, dict):
            raise PortfolioTreatmentError(f"{label} must contain an object")
        value = model_type.model_validate(raw, strict=True)
    except PortfolioTreatmentError:
        raise
    except (ArtifactFormatError, ValidationError) as error:
        raise PortfolioTreatmentError(f"{label} is not canonical and valid") from error
    canonical_method = getattr(value, "canonical_bytes", None)
    if canonical_method is None or canonical_method() != content:
        raise PortfolioTreatmentError(f"{label} bytes are not canonical")
    return value


def _canonical_object(content: bytes, *, label: str) -> dict:
    try:
        raw = parse_canonical_json(content, label=label)
    except ArtifactFormatError as error:
        raise PortfolioTreatmentError(f"{label} is not canonical JSON") from error
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise PortfolioTreatmentError(f"{label} must be a canonical object")
    return raw


def _verify_input_artifacts(
    root: Path,
    manifest: PortfolioTreatmentChainManifest,
    *,
    gates: Mapping[str, PortfolioStageGateReport],
    gate_results: Mapping[
        str,
        tuple[PortfolioStageGateResultSet, PortfolioStageGateResultSet],
    ],
) -> None:
    for record in manifest.records:
        for binding in record.input_artifacts:
            content = _read_bound_file(
                root,
                binding.artifact_file,
                binding.artifact_file_sha256,
                label=f"{record.config} input {binding.artifact_kind}",
            )
            raw = _canonical_object(
                content,
                label=f"{record.config} input {binding.artifact_kind}",
            )
            # Content identity is either the canonical file digest itself or a
            # top-level self/content hash in the bound artifact.
            if (
                binding.artifact_content_sha256 != sha256_bytes(content)
                and binding.artifact_content_sha256 not in raw.values()
            ):
                raise PortfolioTreatmentError(
                    f"{record.config} input content identity mismatch: "
                    f"{binding.artifact_kind}"
                )
            if binding.artifact_kind in {"route_examples", "body_attribution"}:
                packet = _canonical_model(
                    content,
                    PortfolioParentAttributionPacket,
                    label=f"{record.config} current-parent attribution",
                )
                assert isinstance(packet, PortfolioParentAttributionPacket)
                expected = {
                    "s1s2": ("s2_route_optimizer", "s1"),
                    "full": ("s3_body_refiner", "s1s2"),
                }.get(record.config)
                gate = gates.get(record.config)
                result_pair = gate_results.get(record.config)
                if (
                    expected is None
                    or gate is None
                    or result_pair is None
                    or packet.stage != expected[0]
                    or packet.source_config != expected[1]
                    or packet.source_bank_sha256 != record.parent_bank_sha256
                    or packet.query_ids != gate.evaluation_query_ids
                    or packet.optimization_query_ids_sha256
                    != sha256_bytes(
                        canonical_json_bytes(list(manifest.optimization_query_ids))
                    )
                ):
                    raise PortfolioTreatmentError(
                        f"{record.config} attribution is not current-parent/gate bound"
                    )
                try:
                    verify_portfolio_parent_attribution_gate_binding(
                        packet,
                        result_pair[0],
                    )
                except PortfolioAttributionError as error:
                    raise PortfolioTreatmentError(
                        f"{record.config} attribution is not gate-parent-smoke bound"
                    ) from error


def _verify_evolution_session_lineage(
    root: Path,
    manifest: PortfolioTreatmentChainManifest,
    *,
    invocations: Mapping[
        PortfolioTreatmentConfig,
        PortfolioModelInvocationReceipt,
    ],
    output_banks: Mapping[PortfolioTreatmentConfig, StaticBankArtifact] | None = None,
    candidate_banks: Mapping[
        PortfolioTreatmentConfig,
        StaticBankArtifact,
    ]
    | None = None,
    mutations: Mapping[PortfolioTreatmentConfig, PortfolioStageMutation] | None = None,
) -> None:
    records = {item.config: item for item in manifest.records}
    details: dict[str, tuple[dict, bytes]] = {}
    for config in ("s1", "s1s2", "full"):
        record = records[config]
        matches = tuple(
            item
            for item in record.input_artifacts
            if item.artifact_kind == "detailed_invocation_receipt"
        )
        if len(matches) != 1:
            raise PortfolioTreatmentError(
                f"{config} lacks one bound detailed invocation receipt"
            )
        binding = matches[0]
        content = _read_bound_file(
            root,
            binding.artifact_file,
            binding.artifact_file_sha256,
            label=f"{config} detailed invocation receipt",
        )
        raw = _canonical_object(
            content,
            label=f"{config} detailed invocation receipt",
        )
        unsigned = dict(raw)
        supplied = unsigned.pop("receipt_sha256", None)
        if (
            not isinstance(supplied, str)
            or supplied != sha256_bytes(canonical_json_bytes(unsigned))
            or raw.get("status") != "completed"
            or raw.get("stage") != record.stage
            or raw.get("thread_id") != invocations[config].thread_id
            or raw.get("candidate_bank_sha256") != record.candidate_bank_sha256
            or raw.get("raw_model_output_sha256") != record.raw_model_output_file_sha256
            or raw.get("implementation_id") != record.implementation_id
            or raw.get("implementation_version") != record.implementation_version
            or (
                config == "s1s2"
                and (
                    raw.get("implementation_version") != "1.3.0"
                    or raw.get("policy_version")
                    != "portfolio-evolution-actionable-scope-clean-turn-v3"
                )
            )
            or (
                config == "full"
                and (
                    raw.get("implementation_version"),
                    raw.get("policy_version"),
                )
                not in _SUPPORTED_S3_RUNTIME_RECEIPT_CONTRACTS
            )
            or raw.get("implementation_file_sha256")
            != record.implementation_file_sha256
        ):
            raise PortfolioTreatmentError(
                f"{config} detailed invocation evidence differs from the chain"
            )
        normalized_command = raw.get("normalized_command")
        if not isinstance(normalized_command, list) or raw.get(
            "command_sha256"
        ) != sha256_bytes(canonical_json_bytes(normalized_command)):
            raise PortfolioTreatmentError(
                f"{config} detailed invocation command identity is invalid"
            )
        implementation_relative = (
            Path(binding.artifact_file).parent / "implementation-source.py"
        ).as_posix()
        _read_bound_file(
            root,
            implementation_relative,
            record.implementation_file_sha256,
            label=f"{config} implementation source snapshot",
        )
        details[config] = (raw, content)

    s2, _ = details["s1s2"]
    s3, _ = details["full"]
    s3_record = records["full"]
    prior_receipts = tuple(
        item
        for item in s3_record.input_artifacts
        if item.artifact_kind == "prior_invocation_receipt"
    )
    prior_gates = tuple(
        item
        for item in s3_record.input_artifacts
        if item.artifact_kind == "prior_stage_gate"
    )
    if prior_receipts or len(prior_gates) != 1:
        raise PortfolioTreatmentError(
            "Full must bind only its selected-parent S2 gate lineage"
        )
    prior_gate = prior_gates[0]
    s2_gate = records["s1s2"]
    s2_command = s2.get("normalized_command")
    s3_command = s3.get("normalized_command")
    s3_inputs = s3.get("input_files")
    s3_input_roles = (
        tuple(item.get("role") for item in s3_inputs)
        if isinstance(s3_inputs, list)
        and all(isinstance(item, dict) for item in s3_inputs)
        else ()
    )
    s3_output_files = s3.get("output_files")
    s3_patch_outputs = (
        tuple(
            item
            for item in s3_output_files
            if item.get("file") == "s3-text-patch.json"
        )
        if isinstance(s3_output_files, list)
        and all(isinstance(item, dict) for item in s3_output_files)
        else ()
    )
    bound_gate_inputs = (
        tuple(item for item in s3_inputs if item.get("role") == "prior_stage_gate")
        if isinstance(s3_inputs, list)
        and all(isinstance(item, dict) for item in s3_inputs)
        else ()
    )
    expected_s2_command = [
        item for item in CODEX_COMMAND_SHAPE if item != "--ephemeral"
    ]
    s3_contract = (
        s3.get("implementation_version"),
        s3.get("policy_version"),
    )
    expected_s3_input_roles = (
        {_S3_BASE_INPUT_ROLES}
        if s3_contract == _S3_LEGACY_RUNTIME_RECEIPT_CONTRACT
        else {_S3_BASE_INPUT_ROLES, _S3_TEXTOPT_BUFFER_INPUT_ROLES}
    )
    if (
        s2.get("implementation_version") != "1.3.0"
        or s2.get("policy_version")
        != "portfolio-evolution-actionable-scope-clean-turn-v3"
        or s3_contract not in _SUPPORTED_S3_RUNTIME_RECEIPT_CONTRACTS
        or s2.get("session_mode") != "new_persistent"
        or s3.get("session_mode") != "ephemeral"
        or s2.get("ephemeral") is not False
        or s3.get("ephemeral") is not True
        or s2.get("new_session_count") != 1
        or s3.get("new_session_count") != 1
        or s2.get("resume_count") != 0
        or s3.get("resume_count") != 0
        or s2.get("session_turn_index") != 1
        or s3.get("session_turn_index") != 1
        or s2.get("followup_count") != 0
        or s3.get("followup_count") != 0
        or not isinstance(s2.get("thread_id"), str)
        or not isinstance(s3.get("thread_id"), str)
        or s2.get("thread_id") == s3.get("thread_id")
        or s3.get("resume_thread_id") is not None
        or not isinstance(s2.get("session_scratch_path"), str)
        or s2.get("session_scratch_device") is None
        or s2.get("session_scratch_inode") is None
        or s3.get("session_scratch_path") is not None
        or s3.get("session_scratch_device") is not None
        or s3.get("session_scratch_inode") is not None
        or s2.get("requested_model") != s3.get("requested_model")
        or s2.get("reasoning_effort") != s3.get("reasoning_effort")
        or s2.get("implementation_id") != s3.get("implementation_id")
        or s2.get("codex_executable_sha256") != s3.get("codex_executable_sha256")
        or s2_command != expected_s2_command
        or s3_command != list(CODEX_COMMAND_SHAPE)
        or s3_input_roles not in expected_s3_input_roles
        or len(bound_gate_inputs) != 1
        or bound_gate_inputs[0].get("file_sha256") != prior_gate.artifact_file_sha256
        or bound_gate_inputs[0].get("content_sha256") != prior_gate.artifact_file_sha256
        or s3.get("prior_invocation_receipt_file_sha256") is not None
        or prior_gate.artifact_file != s2_gate.gate_report_file
        or prior_gate.artifact_file_sha256 != s2_gate.gate_report_file_sha256
        or s3.get("prior_gate_report_file_sha256") != prior_gate.artifact_file_sha256
    ):
        raise PortfolioTreatmentError(
            "S2 archived-session/S3 clean-turn gate lineage is invalid"
        )

    rejected_buffer_inputs = tuple(
        item
        for item in s3_record.input_artifacts
        if item.artifact_kind == "rejected_edit_buffer"
    )
    detailed_buffer_inputs = (
        tuple(
            item
            for item in s3_inputs
            if item.get("role") == "rejected_edit_buffer"
        )
        if isinstance(s3_inputs, list)
        and all(isinstance(item, dict) for item in s3_inputs)
        else ()
    )
    if s3_contract == _S3_LEGACY_RUNTIME_RECEIPT_CONTRACT:
        compiler_relative = (
            Path(
                next(
                    item.artifact_file
                    for item in s3_record.input_artifacts
                    if item.artifact_kind == "detailed_invocation_receipt"
                )
            ).parent
            / _S3_TEXTOPT_COMPILER_SNAPSHOT_FILE
        )
        if (
            s3_patch_outputs
            or rejected_buffer_inputs
            or detailed_buffer_inputs
            or s3.get("s3_textopt_compiler_file_sha256") is not None
            or _safe_path(
                root,
                compiler_relative.as_posix(),
                label="legacy S3 TextOpt compiler snapshot",
            ).exists()
        ):
            raise PortfolioTreatmentError(
                "legacy S3 runtime unexpectedly contains TextOpt artifacts"
            )
        return

    compiler_file_sha256 = s3.get("s3_textopt_compiler_file_sha256")
    if not isinstance(compiler_file_sha256, str):
        raise PortfolioTreatmentError(
            "S3 TextOpt runtime lacks its compiler source binding"
        )
    compiler_relative = (
        Path(
            next(
                item.artifact_file
                for item in s3_record.input_artifacts
                if item.artifact_kind == "detailed_invocation_receipt"
            )
        ).parent
        / _S3_TEXTOPT_COMPILER_SNAPSHOT_FILE
    )
    compiler_bytes = _read_bound_file(
        root,
        compiler_relative.as_posix(),
        compiler_file_sha256,
        label="S3 TextOpt compiler snapshot",
    )
    runner_relative = compiler_relative.parent / "implementation-source.py"
    runner_bytes = _read_bound_file(
        root,
        runner_relative.as_posix(),
        s3_record.implementation_file_sha256,
        label="S3 evolution runner snapshot",
    )
    from skillchain.evaluation import portfolio_s3_textopt as textopt_module
    from scripts import run_portfolio_evolution_model as evolution_runner

    live_compiler_path = Path(textopt_module.__file__).resolve(strict=True)
    try:
        live_compiler_bytes = read_stable_regular_file(
            live_compiler_path,
            label="live S3 TextOpt compiler",
            max_bytes=_MAX_JSON_BYTES,
        )
    except (ArtifactFormatError, OSError) as error:
        raise PortfolioTreatmentError(
            "live S3 TextOpt compiler cannot be read"
        ) from error
    if live_compiler_bytes != compiler_bytes:
        raise PortfolioTreatmentError(
            "live S3 TextOpt compiler differs from the runtime snapshot"
        )
    live_runner_path = Path(evolution_runner.__file__).resolve(strict=True)
    try:
        live_runner_bytes = read_stable_regular_file(
            live_runner_path,
            label="live S3 evolution runner",
            max_bytes=_MAX_JSON_BYTES,
        )
    except (ArtifactFormatError, OSError) as error:
        raise PortfolioTreatmentError(
            "live S3 evolution runner cannot be read"
        ) from error
    if live_runner_bytes != runner_bytes:
        raise PortfolioTreatmentError(
            "live S3 evolution runner differs from the runtime snapshot"
        )

    if len(s3_patch_outputs) != 1:
        raise PortfolioTreatmentError(
            "S3 TextOpt runtime lacks one normalized patch artifact"
        )
    patch_binding = s3_patch_outputs[0]
    patch_file_sha256 = patch_binding.get("file_sha256")
    if not isinstance(patch_file_sha256, str):
        raise PortfolioTreatmentError("S3 TextOpt patch digest is invalid")
    patch_relative = Path(
        next(
            item.artifact_file
            for item in s3_record.input_artifacts
            if item.artifact_kind == "detailed_invocation_receipt"
        )
    ).parent / "s3-text-patch.json"
    patch_bytes = _read_bound_file(
        root,
        patch_relative.as_posix(),
        patch_file_sha256,
        label="S3 normalized TextOpt patch",
    )
    try:
        normalized_patch = parse_portfolio_s3_text_patch_artifact(patch_bytes)
    except PortfolioTreatmentError as error:
        raise PortfolioTreatmentError(
            "S3 normalized TextOpt patch is invalid"
        ) from error

    if len(detailed_buffer_inputs) != len(rejected_buffer_inputs):
        raise PortfolioTreatmentError(
            "S3 rejected-edit buffer lineage is incomplete"
        )
    rejected_buffer = None
    if detailed_buffer_inputs:
        detailed_buffer = detailed_buffer_inputs[0]
        packaged_buffer = rejected_buffer_inputs[0]
        if (
            detailed_buffer.get("file_sha256")
            != packaged_buffer.artifact_file_sha256
            or detailed_buffer.get("content_sha256")
            != packaged_buffer.artifact_content_sha256
        ):
            raise PortfolioTreatmentError(
                "S3 rejected-edit buffer lineage digest differs"
            )
        buffer_bytes = _read_bound_file(
            root,
            packaged_buffer.artifact_file,
            packaged_buffer.artifact_file_sha256,
            label="S3 rejected-edit buffer",
        )
        try:
            rejected_buffer = parse_portfolio_s3_rejected_edit_buffer(buffer_bytes)
        except PortfolioTreatmentError as error:
            raise PortfolioTreatmentError(
                "S3 rejected-edit buffer is invalid"
            ) from error

    if output_banks is None or candidate_banks is None or mutations is None:
        raise PortfolioTreatmentError("S3 TextOpt replay inputs are unavailable")
    body_bindings = tuple(
        item
        for item in s3_record.input_artifacts
        if item.artifact_kind == "body_attribution"
    )
    if len(body_bindings) != 1:
        raise PortfolioTreatmentError(
            "S3 TextOpt runtime lacks one body-attribution binding"
        )
    body_binding = body_bindings[0]
    body_bytes = _read_bound_file(
        root,
        body_binding.artifact_file,
        body_binding.artifact_file_sha256,
        label="S3 body attribution",
    )
    body_packet = _canonical_model(
        body_bytes,
        PortfolioParentAttributionPacket,
        label="S3 body attribution",
    )
    assert isinstance(body_packet, PortfolioParentAttributionPacket)
    parent_bank = output_banks.get("s1s2")
    candidate_bank = candidate_banks.get("full")
    mutation = mutations.get("full")
    if parent_bank is None or candidate_bank is None or mutation is None:
        raise PortfolioTreatmentError("S3 TextOpt replay chain is incomplete")
    raw_bytes = _read_bound_file(
        root,
        s3_record.raw_model_output_file,
        s3_record.raw_model_output_file_sha256,
        label="S3 raw TextOpt proposal",
    )
    try:
        actionable_capability_ids, optimization_scope = (
            evolution_runner._actionable_mutation_scope(
                "s3_body_refiner",
                body_packet,
                parent_bank=parent_bank,
            )
        )
        replayed_patch, replayed_change = (
            evolution_runner._parse_and_compile_s3_text_patch(
            raw_bytes,
            parent_bank=parent_bank,
            actionable_capability_ids=actionable_capability_ids,
            optimization_scope=optimization_scope,
            rejected_edit_buffer=rejected_buffer,
            )
        )
    except Exception as error:
        raise PortfolioTreatmentError(
            "S3 raw proposal cannot replay through TextOpt"
        ) from error
    if (
        replayed_patch != normalized_patch
        or tuple(mutation.changes) != (replayed_change,)
        or apply_portfolio_stage_mutation(parent_bank, mutation) != candidate_bank
    ):
        raise PortfolioTreatmentError(
            "S3 patch, mutation, or candidate differs from raw TextOpt replay"
        )


def _verify_invocation_sidecars(
    root: Path,
    *,
    receipt_relative: str,
    raw_output_relative: str,
    receipt: PortfolioModelInvocationReceipt,
) -> None:
    receipt_path = _safe_path(root, receipt_relative, label="invocation receipt")
    stage_root = receipt_path.parent
    expected_raw = _safe_path(root, raw_output_relative, label="raw model output")
    if expected_raw.parent.resolve(strict=True) != stage_root.resolve(strict=True):
        raise PortfolioTreatmentError(
            "raw model output and invocation receipt must share one stage directory"
        )
    sidecars = (
        (("prompt.txt",), receipt.prompt_sha256, _MAX_JSON_BYTES),
        (
            ("events.jsonl", "codex-events.jsonl"),
            receipt.event_log_sha256,
            _MAX_EVENT_BYTES,
        ),
        (
            ("stderr.bin", "codex-stderr.bin"),
            receipt.stderr_sha256,
            _MAX_JSON_BYTES,
        ),
    )
    for filenames, expected_sha256, max_bytes in sidecars:
        present = tuple(
            stage_root / filename
            for filename in filenames
            if (stage_root / filename).is_file()
        )
        if len(present) != 1:
            raise PortfolioTreatmentError(
                "model invocation sidecar is missing or ambiguous: "
                + "/".join(filenames)
            )
        path = present[0]
        try:
            content = read_stable_regular_file(
                path,
                label=f"model invocation {path.name}",
                max_bytes=max_bytes,
            )
        except (ArtifactFormatError, OSError) as error:
            raise PortfolioTreatmentError(
                f"model invocation sidecar is missing or unsafe: {path.name}"
            ) from error
        if sha256_bytes(content) != expected_sha256:
            raise PortfolioTreatmentError(
                f"model invocation sidecar digest mismatch: {path.name}"
            )


def _load_runtime_compatibility_rebind(
    root: Path,
    *,
    lock: dict,
    manifest: PortfolioTreatmentChainManifest,
    manifest_file_sha256: str,
) -> VerifiedPortfolioTreatmentChain | None:
    """Load a byte-preserving accepted-chain compatibility projection."""

    relative = lock.get("runtime_compatibility_rebind_file")
    expected_file_sha256 = lock.get("runtime_compatibility_rebind_file_sha256")
    if relative is None and expected_file_sha256 is None:
        return None
    if relative != "compatibility-rebind.json" or not isinstance(
        expected_file_sha256, str
    ):
        raise PortfolioTreatmentError(
            "runtime lock has an incomplete compatibility-rebind binding"
        )
    content = _read_bound_file(
        root,
        relative,
        expected_file_sha256,
        label="Portfolio runtime compatibility rebind",
    )
    rebind = _canonical_model(
        content,
        PortfolioRuntimeCompatibilityRebind,
        label="Portfolio runtime compatibility rebind",
    )
    assert isinstance(rebind, PortfolioRuntimeCompatibilityRebind)
    if (
        lock.get("runtime_compatibility_rebind_sha256") != rebind.rebind_sha256
        or lock.get("core_runtime_sources_dir") != rebind.core_runtime_sources_dir
        or lock.get("core_runtime_sources_receipt_file_sha256")
        != rebind.core_runtime_sources_receipt_file_sha256
        or lock.get("tool_registry_sha256") != rebind.target_tool_registry_sha256
        or lock.get("tool_registry_runtime_sha256")
        != rebind.target_tool_registry_runtime_sha256
        or lock.get("runtime_data_sha256") != rebind.target_runtime_data_sha256
        or lock.get("source_sha256s") != list(rebind.target_source_sha256s)
        or lock.get("model_calls_performed") != 0
    ):
        raise PortfolioTreatmentError(
            "runtime lock differs from its compatibility-rebind target"
        )

    parent_root = _safe_path(
        root,
        rebind.source_runtime_dir,
        label="accepted parent runtime",
    )
    if not parent_root.is_dir() or parent_root.is_symlink():
        raise PortfolioTreatmentError(
            "accepted parent runtime must be a real directory"
        )
    observed_files: list[tuple[str, str]] = []
    for path in sorted(parent_root.rglob("*")):
        if path.is_symlink():
            raise PortfolioTreatmentError("accepted parent runtime contains a symlink")
        if not path.is_file():
            continue
        relative_path = path.relative_to(parent_root).as_posix()
        try:
            file_content = read_stable_regular_file(
                path,
                label=f"accepted parent runtime {relative_path}",
                max_bytes=_MAX_JSON_BYTES,
            )
        except (ArtifactFormatError, OSError) as error:
            raise PortfolioTreatmentError(
                f"accepted parent runtime file cannot be read: {relative_path}"
            ) from error
        observed_files.append((relative_path, sha256_bytes(file_content)))
    declared_files = [
        (item.relative_path, item.file_sha256) for item in rebind.source_runtime_files
    ]
    if observed_files != declared_files:
        raise PortfolioTreatmentError(
            "accepted parent runtime file set or bytes differ from rebind lineage"
        )
    parent = load_verified_portfolio_treatment_runtime(
        parent_root,
        expected_runtime_lock_file_sha256=(rebind.source_runtime_lock_file_sha256),
        _allow_legacy_missing_execution_aliases=True,
    )
    if (
        parent.runtime_lock.get("runtime_lock_sha256")
        != rebind.source_runtime_lock_sha256
        or parent.manifest_file_sha256 != rebind.source_treatment_manifest_file_sha256
        or parent.chain.manifest.chain_sha256 != rebind.source_treatment_chain_sha256
        or manifest_file_sha256 != parent.manifest_file_sha256
        or manifest != parent.chain.manifest
        or parent.chain.compatibility_rebind is not None
    ):
        raise PortfolioTreatmentError(
            "compatibility rebind does not preserve its accepted parent chain"
        )

    core_receipt_content = _read_bound_file(
        root,
        f"{rebind.core_runtime_sources_dir}/receipt.json",
        rebind.core_runtime_sources_receipt_file_sha256,
        label="Core runtime-source receipt",
    )
    core_receipt = _canonical_object(
        core_receipt_content,
        label="Core runtime-source receipt",
    )
    unsigned_core_receipt = dict(core_receipt)
    core_receipt_sha256 = unsigned_core_receipt.pop("receipt_sha256", None)
    if (
        core_receipt.get("kind") != "portfolio-core-runtime-sources-receipt"
        or core_receipt.get("formal_eligible") is not False
        or core_receipt.get("provider_call_count") != 0
        or core_receipt_sha256 != rebind.core_runtime_sources_receipt_sha256
        or core_receipt_sha256
        != sha256_bytes(canonical_json_bytes(unsigned_core_receipt))
        or core_receipt.get("runtime_data_sha256") != rebind.target_runtime_data_sha256
        or core_receipt.get("runtime_source_sha256s")
        != list(rebind.target_source_sha256s)
    ):
        raise PortfolioTreatmentError(
            "Core runtime-source receipt differs from compatibility rebind"
        )

    outputs: dict[PortfolioTreatmentConfig, StaticBankArtifact] = {}
    candidates: dict[PortfolioTreatmentConfig, StaticBankArtifact] = {}
    for binding in rebind.bank_bindings:
        rebound_content = _read_bound_file(
            root,
            binding.rebound_bank_file,
            binding.rebound_bank_file_sha256,
            label=f"{binding.config} {binding.role} rebound Bank",
        )
        rebound = _canonical_model(
            rebound_content,
            StaticBankArtifact,
            label=f"{binding.config} {binding.role} rebound Bank",
        )
        assert isinstance(rebound, StaticBankArtifact)
        if rebound.bank_sha256 != binding.rebound_bank_sha256:
            raise PortfolioTreatmentError(
                f"{binding.config} {binding.role} rebound Bank content mismatch"
            )
        target = outputs if binding.role == "output" else candidates
        target[binding.config] = rebound
    return verify_portfolio_runtime_compatibility_rebind_chain(
        rebind=rebind,
        source_chain=parent.chain,
        output_banks=outputs,
        candidate_banks=candidates,
    )


def load_verified_portfolio_treatment_runtime(
    runtime_root: str | Path,
    *,
    expected_runtime_lock_file_sha256: str,
    _allow_legacy_missing_execution_aliases: bool = False,
) -> VerifiedPortfolioTreatmentRuntime:
    """Load and replay every artifact required by an official matrix runtime."""

    if not _SHA256_RE.fullmatch(expected_runtime_lock_file_sha256):
        raise PortfolioTreatmentError("runtime-lock expected SHA-256 is invalid")
    root = Path(runtime_root).absolute()
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise PortfolioTreatmentError(
            "treatment runtime root does not exist"
        ) from error
    if not root.is_dir():
        raise PortfolioTreatmentError("treatment runtime root is not a directory")

    lock_bytes = _read_bound_file(
        root,
        "runtime-lock.json",
        expected_runtime_lock_file_sha256,
        label="Portfolio treatment runtime lock",
    )
    lock = _canonical_object(lock_bytes, label="Portfolio treatment runtime lock")
    supplied_lock_sha256 = lock.get("runtime_lock_sha256")
    unsigned_lock = dict(lock)
    unsigned_lock.pop("runtime_lock_sha256", None)
    if (
        not isinstance(supplied_lock_sha256, str)
        or supplied_lock_sha256 != sha256_bytes(canonical_json_bytes(unsigned_lock))
        or lock.get("kind") != "portfolio-assistant-runtime-lock"
        or lock.get("track") != "portfolio"
        or lock.get("formal_eligible") is not False
        or lock.get("bank_policy") != PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION
        or lock.get("official_matrix_eligible") is not True
        or lock.get("treatment_chain_status") != "ready_for_matrix"
    ):
        raise PortfolioTreatmentError(
            "runtime lock is not a matrix-ready real treatment runtime"
        )

    manifest_relative = lock.get("treatment_chain_manifest_file")
    manifest_file_sha256 = lock.get("treatment_chain_manifest_file_sha256")
    if manifest_relative != "treatment-chain-manifest.json" or not isinstance(
        manifest_file_sha256, str
    ):
        raise PortfolioTreatmentError("runtime lock lacks its treatment manifest")
    manifest_bytes = _read_bound_file(
        root,
        manifest_relative,
        manifest_file_sha256,
        label="Portfolio treatment chain manifest",
    )
    manifest = _canonical_model(
        manifest_bytes,
        PortfolioTreatmentChainManifest,
        label="Portfolio treatment chain manifest",
    )
    assert isinstance(manifest, PortfolioTreatmentChainManifest)
    if (
        lock.get("treatment_chain_sha256") != manifest.chain_sha256
        or manifest.status != "ready_for_matrix"
    ):
        raise PortfolioTreatmentError(
            "runtime lock and treatment manifest identity differ"
        )

    rebound_chain = _load_runtime_compatibility_rebind(
        root,
        lock=lock,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
    )
    if rebound_chain is not None:
        expected_aliases = [
            item.model_dump(mode="json")
            for item in rebound_chain.execution_artifact_aliases
        ]
        rebind = rebound_chain.compatibility_rebind
        assert rebind is not None
        output_bindings = {
            item.config: item for item in rebind.bank_bindings if item.role == "output"
        }
        if (
            lock.get("bank_sha256s")
            != {
                config: bank.bank_sha256
                for config, bank in rebound_chain.output_banks.items()
            }
            or lock.get("bank_source_file_sha256s")
            != {
                config: output_bindings[config].rebound_bank_file_sha256
                for config in _CONFIG_ORDER
            }
            or lock.get("treatment_record_sha256s")
            != {item.config: item.receipt_sha256 for item in manifest.records}
            or lock.get("execution_artifact_alias_policy_version")
            != PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION
            or lock.get("execution_artifact_aliases") != expected_aliases
            or lock.get("execution_artifact_alias_provider_model_call_count") != 0
        ):
            raise PortfolioTreatmentError(
                "runtime-rebound lock Bank, record, or alias identities differ"
            )
        return VerifiedPortfolioTreatmentRuntime(
            root=root,
            runtime_lock=lock,
            runtime_lock_file_sha256=expected_runtime_lock_file_sha256,
            manifest_file_sha256=manifest_file_sha256,
            chain=rebound_chain,
            _marker=_VERIFIED_RUNTIME_MARKER,
        )

    output_banks: dict[PortfolioTreatmentConfig, StaticBankArtifact] = {}
    candidate_banks: dict[PortfolioTreatmentConfig, StaticBankArtifact] = {}
    mutations: dict[PortfolioTreatmentConfig, PortfolioStageMutation] = {}
    invocations: dict[PortfolioTreatmentConfig, PortfolioModelInvocationReceipt] = {}
    gates: dict[str, PortfolioStageGateReport] = {}
    gate_results: dict[
        str,
        tuple[PortfolioStageGateResultSet, PortfolioStageGateResultSet],
    ] = {}

    bank_binding_by_config = {item.config: item for item in manifest.banks}
    for record in manifest.records:
        binding = bank_binding_by_config[record.config]
        output_content = _read_bound_file(
            root,
            binding.bank_file,
            binding.bank_file_sha256,
            label=f"{record.config} output Bank",
        )
        output = _canonical_model(
            output_content,
            StaticBankArtifact,
            label=f"{record.config} output Bank",
        )
        assert isinstance(output, StaticBankArtifact)
        if output.bank_sha256 != binding.bank_sha256:
            raise PortfolioTreatmentError(
                f"{record.config} output Bank content hash mismatch"
            )
        output_banks[record.config] = output

        candidate_content = _read_bound_file(
            root,
            record.candidate_bank_file,
            record.candidate_bank_file_sha256,
            label=f"{record.config} candidate Bank",
        )
        candidate = _canonical_model(
            candidate_content,
            StaticBankArtifact,
            label=f"{record.config} candidate Bank",
        )
        assert isinstance(candidate, StaticBankArtifact)
        candidate_banks[record.config] = candidate

        raw_output = _read_bound_file(
            root,
            record.raw_model_output_file,
            record.raw_model_output_file_sha256,
            label=f"{record.config} raw model output",
        )
        del raw_output
        invocation_content = _read_bound_file(
            root,
            record.invocation_receipt_file,
            record.invocation_receipt_file_sha256,
            label=f"{record.config} model invocation receipt",
        )
        invocation = _canonical_model(
            invocation_content,
            PortfolioModelInvocationReceipt,
            label=f"{record.config} model invocation receipt",
        )
        assert isinstance(invocation, PortfolioModelInvocationReceipt)
        invocations[record.config] = invocation
        _verify_invocation_sidecars(
            root,
            receipt_relative=record.invocation_receipt_file,
            raw_output_relative=record.raw_model_output_file,
            receipt=invocation,
        )

        gate_content = _read_bound_file(
            root,
            record.gate_report_file,
            record.gate_report_file_sha256,
            label=f"{record.config} gate report",
        )
        if record.config != "llm_static":
            gate = _canonical_model(
                gate_content,
                PortfolioStageGateReport,
                label=f"{record.config} gate report",
            )
            assert isinstance(gate, PortfolioStageGateReport)
            gates[record.config] = gate
            parent_result_content = _read_bound_file(
                root,
                gate.parent_result_file,
                gate.parent_result_file_sha256,
                label=f"{record.config} parent gate results",
            )
            candidate_result_content = _read_bound_file(
                root,
                gate.candidate_result_file,
                gate.candidate_result_file_sha256,
                label=f"{record.config} candidate gate results",
            )
            parent_result = _canonical_model(
                parent_result_content,
                PortfolioStageGateResultSet,
                label=f"{record.config} parent gate results",
            )
            candidate_result = _canonical_model(
                candidate_result_content,
                PortfolioStageGateResultSet,
                label=f"{record.config} candidate gate results",
            )
            assert isinstance(parent_result, PortfolioStageGateResultSet)
            assert isinstance(candidate_result, PortfolioStageGateResultSet)
            verify_portfolio_stage_gate_evidence(
                gate,
                parent_result,
                candidate_result,
            )
            gate_results[record.config] = (parent_result, candidate_result)
        else:
            static_gate = _canonical_object(
                gate_content,
                label="LLMStatic author review/adjudication",
            )
            if (
                record.gate_report_sha256 != sha256_bytes(gate_content)
                and record.gate_report_sha256 not in static_gate.values()
            ):
                raise PortfolioTreatmentError(
                    "LLMStatic author review content identity mismatch"
                )

        if record.mutation_file is not None:
            assert record.mutation_file_sha256 is not None
            mutation_content = _read_bound_file(
                root,
                record.mutation_file,
                record.mutation_file_sha256,
                label=f"{record.config} mutation",
            )
            mutation = _canonical_model(
                mutation_content,
                PortfolioStageMutation,
                label=f"{record.config} mutation",
            )
            assert isinstance(mutation, PortfolioStageMutation)
            mutations[record.config] = mutation

    _verify_input_artifacts(
        root,
        manifest,
        gates=gates,
        gate_results=gate_results,
    )
    _verify_evolution_session_lineage(
        root,
        manifest,
        invocations=invocations,
        output_banks=output_banks,
        candidate_banks=candidate_banks,
        mutations=mutations,
    )
    chain = verify_portfolio_treatment_chain(
        manifest,
        output_banks=output_banks,
        candidate_banks=candidate_banks,
        mutations=mutations,
        invocation_receipts=invocations,
        gate_reports=gates,  # type: ignore[arg-type]
    )

    locked_banks = lock.get("bank_sha256s")
    locked_records = lock.get("treatment_record_sha256s")
    expected_aliases = [
        item.model_dump(mode="json")
        for item in build_portfolio_execution_artifact_aliases(manifest)
    ]
    aliases_match = (
        lock.get("execution_artifact_alias_policy_version")
        == PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION
        and lock.get("execution_artifact_aliases") == expected_aliases
        and lock.get("execution_artifact_alias_provider_model_call_count") == 0
    )
    legacy_aliases_absent = _allow_legacy_missing_execution_aliases and all(
        field not in lock
        for field in (
            "execution_artifact_alias_policy_version",
            "execution_artifact_aliases",
            "execution_artifact_alias_provider_model_call_count",
        )
    )
    if (
        locked_banks != {item.config: item.bank_sha256 for item in manifest.banks}
        or locked_records
        != {item.config: item.receipt_sha256 for item in manifest.records}
        or not (aliases_match or legacy_aliases_absent)
    ):
        raise PortfolioTreatmentError(
            "runtime lock Bank, treatment-record, or execution-alias identities differ"
        )
    return VerifiedPortfolioTreatmentRuntime(
        root=root,
        runtime_lock=lock,
        runtime_lock_file_sha256=expected_runtime_lock_file_sha256,
        manifest_file_sha256=manifest_file_sha256,
        chain=chain,
        _marker=_VERIFIED_RUNTIME_MARKER,
    )


def require_verified_portfolio_treatment_runtime(
    value: object,
) -> VerifiedPortfolioTreatmentRuntime:
    if (
        type(value) is not VerifiedPortfolioTreatmentRuntime
        or value._marker is not _VERIFIED_RUNTIME_MARKER
    ):
        raise TypeError("Portfolio matrix execution requires a verified runtime")
    require_verified_portfolio_treatment_chain(value.chain)
    return load_verified_portfolio_treatment_runtime(
        value.root,
        expected_runtime_lock_file_sha256=value.runtime_lock_file_sha256,
    )


__all__ = [
    "VerifiedPortfolioTreatmentRuntime",
    "load_verified_portfolio_treatment_runtime",
    "require_verified_portfolio_treatment_runtime",
]
