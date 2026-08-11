"""Prepare one create-only, zero-call S1 Creator input package.

The command rebuilds ``CodexAuthoringInput`` from the semantic AuthoringInput
inside one deeply verified Static-opt runtime and an explicitly SHA-bound Codex
model-access evidence file.  It binds that current contract to one parent
LLMStatic Bank and one typed Feedback bundle, then publishes the exact future
runner command without launching Codex.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from skillchain.codex_authoring import (  # noqa: E402
    CodexAuthoringInput,
    VerifiedCodexModelAccessEvidence,
    build_codex_authoring_input,
    load_codex_authoring_input,
    load_codex_model_access_evidence,
)
from skillchain.evaluation.portfolio_s1_feedback import (  # noqa: E402
    PortfolioS1FeedbackBundleV1,
    PortfolioS1FeedbackBundleV2,
    PortfolioS1FeedbackBundleV3,
    PortfolioS1FeedbackBundleV4,
    PortfolioS1FeedbackBundleV5,
    PortfolioS1FeedbackBundleV6,
    PortfolioS1FeedbackBundleV7,
    load_portfolio_s1_feedback_bundle,
    load_portfolio_s1_feedback_bundle_v2,
    load_portfolio_s1_feedback_bundle_v3,
    load_portfolio_s1_feedback_bundle_v4,
    load_portfolio_s1_feedback_bundle_v5,
    load_portfolio_s1_feedback_bundle_v6,
    load_portfolio_s1_feedback_bundle_v7,
)
from skillchain.evaluation.portfolio_s1_feedback_retry_v3 import (  # noqa: E402
    PortfolioS1FeedbackBundleV8,
    load_portfolio_s1_feedback_bundle_v8,
)
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (  # noqa: E402
    PortfolioS1FeedbackBundleV9,
    load_portfolio_s1_feedback_bundle_v9,
)
from skillchain.evaluation.portfolio_s1_feedback_round3_v1 import (  # noqa: E402
    PortfolioS1FeedbackBundleV10,
    load_portfolio_s1_feedback_bundle_v10,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (  # noqa: E402
    VerifiedPortfolioStaticOptRuntime,
    load_verified_portfolio_static_opt_runtime_evidence,
    require_verified_portfolio_static_opt_runtime_evidence,
)
from skillchain.static_authoring import (  # noqa: E402
    AuthoringInput,
    StaticBankArtifact,
    load_authoring_packet,
)
from skillchain.synthesis.store import (  # noqa: E402
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


PREPARATION_POLICY_VERSION = "portfolio-s1-creator-input-package-v1"
SPARSE_PREPARATION_POLICY_VERSION = "portfolio-s1-creator-input-package-v2"
RUNNER_POLICY_VERSION = "portfolio-evolution-typed-feedback-whole-bank-clean-turn-v6"
SPARSE_RUNNER_POLICY_VERSION = (
    "portfolio-evolution-policy-filtered-sparse-patch-clean-turn-v7"
)
RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "run_portfolio_evolution_model.py"

PARENT_BANK_FILE = "parent-static-bank.json"
SEMANTIC_INPUT_FILE = "semantic-authoring-input.json"
CODEX_INPUT_FILE = "codex-authoring-input.json"
FEEDBACK_BUNDLE_FILE = "s1-feedback-bundle.json"
MODEL_ACCESS_EVIDENCE_FILE = "codex-model-access-evidence.json"
RUNNER_ARGV_FILE = "runner-argv.json"
INPUT_MANIFEST_FILE = "input-manifest.json"

_PACKAGE_FILES = frozenset(
    {
        PARENT_BANK_FILE,
        SEMANTIC_INPUT_FILE,
        CODEX_INPUT_FILE,
        FEEDBACK_BUNDLE_FILE,
        MODEL_ACCESS_EVIDENCE_FILE,
        RUNNER_ARGV_FILE,
        INPUT_MANIFEST_FILE,
    }
)
_INPUT_ROLES = (
    "codex_authoring_input",
    "feedback_bundle",
    "parent_static_bank",
    "semantic_authoring_input",
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PortfolioS1FeedbackBundle = (
    PortfolioS1FeedbackBundleV1
    | PortfolioS1FeedbackBundleV2
    | PortfolioS1FeedbackBundleV3
    | PortfolioS1FeedbackBundleV4
    | PortfolioS1FeedbackBundleV5
    | PortfolioS1FeedbackBundleV6
    | PortfolioS1FeedbackBundleV7
    | PortfolioS1FeedbackBundleV8
    | PortfolioS1FeedbackBundleV9
    | PortfolioS1FeedbackBundleV10
)


class PortfolioS1CreatorInputError(ValueError):
    """A prepared S1 Creator input package failed closed."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class PreparedInputBinding(_StrictFrozenModel):
    role: Literal[
        "codex_authoring_input",
        "feedback_bundle",
        "parent_static_bank",
        "semantic_authoring_input",
    ]
    file: str
    file_sha256: Sha256
    artifact_sha256: Sha256


class PortfolioS1CreatorRunnerArgv(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-creator-runner-argv"] = (
        "portfolio-s1-creator-runner-argv"
    )
    policy_version: Literal[
        "portfolio-evolution-typed-feedback-whole-bank-clean-turn-v6",
        "portfolio-evolution-policy-filtered-sparse-patch-clean-turn-v7",
    ] = RUNNER_POLICY_VERSION
    command: tuple[str, ...] = Field(min_length=24)
    requested_model: Literal["gpt-5.6-sol"] = "gpt-5.6-sol"
    reasoning_effort: Literal["high"] = "high"
    timeout_seconds: Literal[600] = 600
    invocation_sha256: Sha256

    @field_validator("command", mode="before")
    @classmethod
    def _command_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_invocation(self) -> Self:
        if (
            self.command.count("--stage") != 1
            or self.command[self.command.index("--stage") + 1] != "s1_creator"
            or self.command.count("--session-mode") != 1
            or self.command[self.command.index("--session-mode") + 1] != "ephemeral"
            or self.command.count("--timeout-seconds") != 1
            or self.command[self.command.index("--timeout-seconds") + 1] != "600"
        ):
            raise ValueError("prepared S1 runner command drifted")
        payload = self.model_dump(mode="json", exclude={"invocation_sha256"})
        if self.invocation_sha256 != sha256_bytes(canonical_json_bytes(payload)):
            raise ValueError("prepared S1 runner command self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioS1CreatorInputManifest(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-creator-input-manifest"] = (
        "portfolio-s1-creator-input-manifest"
    )
    policy_version: Literal[
        "portfolio-s1-creator-input-package-v1",
        "portfolio-s1-creator-input-package-v2",
    ] = PREPARATION_POLICY_VERSION
    status: Literal["prepared_not_invoked"] = "prepared_not_invoked"
    provider_calls: Literal[0] = 0
    source_runtime_root: str
    runtime_lock_file_sha256: Sha256
    runtime_lock_sha256: Sha256
    tool_registry_sha256: Sha256
    tool_registry_runtime_sha256: Sha256
    codex_executable: str
    codex_executable_sha256: Sha256
    codex_executable_bytes: int = Field(ge=1)
    future_creator_output_dir: str
    preparer_source_file_sha256: Sha256
    runner_source_file_sha256: Sha256
    input_files: tuple[PreparedInputBinding, ...]
    model_access_evidence_file: Literal["codex-model-access-evidence.json"] = (
        MODEL_ACCESS_EVIDENCE_FILE
    )
    model_access_evidence_file_sha256: Sha256
    runner_argv_file: Literal["runner-argv.json"] = RUNNER_ARGV_FILE
    runner_argv_file_sha256: Sha256
    runner_invocation_sha256: Sha256
    manifest_sha256: Sha256

    @field_validator("input_files", mode="before")
    @classmethod
    def _bindings_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        roles = tuple(item.role for item in self.input_files)
        files = tuple(item.file for item in self.input_files)
        if roles != _INPUT_ROLES or len(set(files)) != len(files):
            raise ValueError("prepared S1 input roles or files drifted")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != sha256_bytes(canonical_json_bytes(payload)):
            raise ValueError("prepared S1 input manifest self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class VerifiedPortfolioS1CreatorInputs:
    root: Path
    manifest: PortfolioS1CreatorInputManifest
    manifest_file_sha256: str
    runner_argv: PortfolioS1CreatorRunnerArgv
    runtime: VerifiedPortfolioStaticOptRuntime
    parent_bank: StaticBankArtifact
    semantic_input: AuthoringInput
    codex_input: CodexAuthoringInput
    feedback_bundle: PortfolioS1FeedbackBundle


def _canonical_model(model_type, content: bytes, *, label: str):
    try:
        value = model_type.model_validate_json(content, strict=True)
    except ValidationError as error:
        raise PortfolioS1CreatorInputError(f"{label} is invalid") from error
    if value.canonical_bytes() != content:
        raise PortfolioS1CreatorInputError(f"{label} is not canonical")
    return value


def _load_typed_s1_feedback_bundle(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1FeedbackBundle:
    """Route only exact V1-V10 identities to strict canonical loaders."""

    content = read_stable_regular_file(
        path,
        label="S1 Feedback bundle",
        max_bytes=32 * 1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1CreatorInputError("S1 Feedback bundle file SHA drifted")
    try:
        raw = parse_canonical_json(content, label="S1 Feedback bundle")
    except ValueError as error:
        raise PortfolioS1CreatorInputError("S1 Feedback bundle is invalid") from error
    identity = (
        (
            raw.get("schema_version"),
            raw.get("kind"),
            raw.get("policy_version"),
        )
        if isinstance(raw, dict)
        else None
    )
    if identity == (
        1,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v1",
    ):
        bundle = load_portfolio_s1_feedback_bundle(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        2,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v2",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v2(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        3,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v3",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v3(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        4,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v4",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v4(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        5,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v5",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v5(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        6,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v6",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v6(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        7,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v7",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v7(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        8,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v8",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v8(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        9,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v9",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v9(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        10,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v10",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v10(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    else:
        raise PortfolioS1CreatorInputError(
            "S1 requires an exact PortfolioS1FeedbackBundleV1, "
            "PortfolioS1FeedbackBundleV2, PortfolioS1FeedbackBundleV3, or "
            "PortfolioS1FeedbackBundleV4/V5/V6/V7/V8/V9/V10 identity"
        )
    if bundle.canonical_bytes() != content:
        raise PortfolioS1CreatorInputError("S1 Feedback bundle is not canonical")
    if isinstance(bundle, PortfolioS1FeedbackBundleV10):
        retry_count = bundle.provider_call_count - bundle.selected_count
        projection = bundle.model_projection_payload()
        if (
            bundle.provider_call_count not in {240, 241, 242, 243}
            or retry_count != bundle.retry_claim_count
            or bundle.retry_claim_count != len(bundle.retry_claim_sha256s)
            or bundle.fresh_output_count != 240
            or bundle.historical_feedback_outputs_imported != 0
            or bundle.run_sha256 != bundle.round3_run_sha256
            or bundle.run_file_sha256 != bundle.round3_run_file_sha256
            or bundle.authorization_sha256 != bundle.round3_authorization_sha256
            or bundle.control_sha256 != bundle.round3_control_sha256
            or not bundle.round3_artifact_set_sha256
            or len(bundle.entry_provenance) != 240
            or len(
                {item.bound_artifact_sha256 for item in bundle.entry_provenance}
            )
            != 240
            or len(
                {item.feedback_result_sha256 for item in bundle.entry_provenance}
            )
            != 240
            or len(
                {item.final_global_call_ordinal for item in bundle.entry_provenance}
            )
            != 240
            or projection.get("schema_version") != 5
            or projection.get("selected_count") != 240
            or projection.get("parsed_count") != 240
        ):
            raise PortfolioS1CreatorInputError(
                "BundleV10 fresh Round3 lineage or V5 projection drifted"
            )
    return bundle


def _style_tool_version(semantic: AuthoringInput) -> str:
    raw = parse_canonical_json(
        semantic.tool_registry.canonical_json.encode("utf-8"),
        label="current semantic Tool Registry",
    )
    if not isinstance(raw, dict) or not isinstance(raw.get("tools"), list):
        raise PortfolioS1CreatorInputError("semantic Tool Registry is invalid")
    matches = [
        item
        for item in raw["tools"]
        if isinstance(item, dict) and item.get("name") == "style_similar_search"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("tool_version"), str):
        raise PortfolioS1CreatorInputError("semantic Style ToolSpec is missing")
    return matches[0]["tool_version"]


def require_current_s1_codex_authoring_input(
    runtime: VerifiedPortfolioStaticOptRuntime,
    codex_input: CodexAuthoringInput,
) -> None:
    """Reject every Codex packet not rebuilt from this exact Style-2.3 runtime."""

    verified = require_verified_portfolio_static_opt_runtime_evidence(runtime)
    semantic = verified.semantic_authoring_input
    if (
        _style_tool_version(semantic) != "2.3.0"
        or codex_input.semantic_source_packet_file_sha256
        != verified.semantic_authoring_input_file_sha256
        or codex_input.taxonomy != semantic.taxonomy
        or codex_input.task_specification != semantic.task_specification
        or codex_input.tool_registry != semantic.tool_registry
        or codex_input.tool_registry_runtime_binding
        != semantic.tool_registry_runtime_binding
        or codex_input.tool_registry_runtime_sha256
        != semantic.tool_registry_runtime_sha256
        or codex_input.compiler != semantic.compiler
        or codex_input.public_sources != semantic.public_sources
        or codex_input.reference_skill_bundle != semantic.reference_skill_bundle
    ):
        raise PortfolioS1CreatorInputError(
            "Codex AuthoringInput is stale or does not match the current Style-2.3 runtime"
        )


def _verify_codex_executable(
    path: str | Path,
    evidence: VerifiedCodexModelAccessEvidence,
) -> tuple[Path, str, int]:
    executable = Path(path).absolute()
    try:
        before = executable.lstat()
    except OSError as error:
        raise PortfolioS1CreatorInputError(
            "Codex executable cannot be inspected"
        ) from error
    reparse = int(getattr(before, "st_file_attributes", 0)) & int(
        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or reparse:
        raise PortfolioS1CreatorInputError("Codex executable must be a regular file")
    try:
        with executable.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            after_read = os.fstat(handle.fileno())
        after = executable.lstat()
    except OSError as error:
        raise PortfolioS1CreatorInputError(
            "Codex executable cannot be hashed"
        ) from error
    expected = evidence.value.binary
    if (
        opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
        or after_read.st_size != opened.st_size
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or digest != expected.get("sha256")
        or before.st_size != expected.get("bytes")
    ):
        raise PortfolioS1CreatorInputError(
            "Codex executable differs from the explicit model-access evidence"
        )
    return executable, digest, int(before.st_size)


def _build_runner_argv(
    *,
    package_root: Path,
    future_output_dir: Path,
    runtime: VerifiedPortfolioStaticOptRuntime,
    feedback_file_sha256: str,
    codex_file_sha256: str,
    codex_executable: Path,
    sparse_feedback: bool,
) -> PortfolioS1CreatorRunnerArgv:
    command = (
        sys.executable,
        str(RUNNER_PATH.resolve(strict=True)),
        "--stage",
        "s1_creator",
        "--output-dir",
        str(future_output_dir),
        "--stage-input",
        str(package_root / FEEDBACK_BUNDLE_FILE),
        "--stage-input-file-sha256",
        feedback_file_sha256,
        "--semantic-authoring-input",
        str(package_root / SEMANTIC_INPUT_FILE),
        "--semantic-authoring-input-file-sha256",
        runtime.semantic_authoring_input_file_sha256,
        "--codex-authoring-input",
        str(package_root / CODEX_INPUT_FILE),
        "--codex-authoring-input-file-sha256",
        codex_file_sha256,
        "--parent-bank",
        str(package_root / PARENT_BANK_FILE),
        "--parent-bank-file-sha256",
        runtime.bank_file_sha256,
        "--tool-registry-runtime-sha256",
        runtime.bank.tool_registry_runtime_sha256,
        "--codex-executable",
        str(codex_executable),
        "--session-mode",
        "ephemeral",
        "--timeout-seconds",
        "600",
    )
    payload = {
        "schema_version": 1,
        "kind": "portfolio-s1-creator-runner-argv",
        "policy_version": (
            SPARSE_RUNNER_POLICY_VERSION if sparse_feedback else RUNNER_POLICY_VERSION
        ),
        "command": list(command),
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "timeout_seconds": 600,
    }
    return PortfolioS1CreatorRunnerArgv.model_validate(
        {
            **payload,
            "invocation_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )


def _binding(role: str, file: str, content: bytes, artifact_sha256: str):
    return PreparedInputBinding(
        role=role,
        file=file,
        file_sha256=sha256_bytes(content),
        artifact_sha256=artifact_sha256,
    )


def _safe_cleanup_staging(staging: Path, destination: Path) -> None:
    if not staging.exists():
        return
    resolved = staging.resolve(strict=True)
    parent = destination.absolute().parent.resolve(strict=True)
    if resolved.parent != parent or not resolved.name.startswith(
        f".{destination.name}.staging-"
    ):
        raise RuntimeError("refusing to clean an unexpected S1 staging directory")
    shutil.rmtree(resolved)


def prepare_portfolio_s1_creator_inputs(
    *,
    static_runtime_root: str | Path,
    expected_runtime_lock_file_sha256: str,
    feedback_bundle_path: str | Path,
    expected_feedback_bundle_file_sha256: str,
    model_access_evidence_path: str | Path,
    expected_model_access_evidence_file_sha256: str,
    codex_executable_path: str | Path,
    future_creator_output_dir: str | Path,
    output_dir: str | Path,
) -> VerifiedPortfolioS1CreatorInputs:
    """Publish and reload the exact S1 inputs without invoking any provider."""

    destination = Path(output_dir).absolute()
    future_output = Path(future_creator_output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"create-only output already exists: {destination}")
    if os.path.lexists(future_output):
        raise FileExistsError(
            f"future Creator output must remain create-only: {future_output}"
        )
    if destination == future_output:
        raise PortfolioS1CreatorInputError(
            "input package and future Creator output must be different directories"
        )

    runtime = load_verified_portfolio_static_opt_runtime_evidence(
        static_runtime_root,
        expected_runtime_lock_file_sha256=expected_runtime_lock_file_sha256,
    )
    feedback = _load_typed_s1_feedback_bundle(
        feedback_bundle_path,
        expected_file_sha256=expected_feedback_bundle_file_sha256,
    )
    if feedback.parent_static_bank_sha256 != runtime.bank.bank_sha256:
        raise PortfolioS1CreatorInputError(
            "Feedback bundle does not bind the verified parent Static Bank"
        )
    access = load_codex_model_access_evidence(
        model_access_evidence_path,
        expected_file_sha256=expected_model_access_evidence_file_sha256,
    )
    executable, executable_sha256, executable_bytes = _verify_codex_executable(
        codex_executable_path,
        access,
    )
    codex_input = build_codex_authoring_input(
        semantic_source=runtime.semantic_authoring_input,
        semantic_source_packet_file_sha256=(
            runtime.semantic_authoring_input_file_sha256
        ),
        model_access_evidence=access,
    )
    require_current_s1_codex_authoring_input(runtime, codex_input)

    parent_bytes = runtime.bank.canonical_bytes()
    semantic_bytes = runtime.semantic_authoring_input.canonical_bytes()
    codex_bytes = codex_input.canonical_bytes()
    feedback_bytes = feedback.canonical_bytes()
    evidence_bytes = read_stable_regular_file(
        access.path,
        label="Codex model-access evidence",
    )
    if (
        sha256_bytes(parent_bytes) != runtime.bank_file_sha256
        or sha256_bytes(semantic_bytes) != runtime.semantic_authoring_input_file_sha256
        or sha256_bytes(feedback_bytes) != expected_feedback_bundle_file_sha256
        or sha256_bytes(evidence_bytes) != expected_model_access_evidence_file_sha256
    ):
        raise PortfolioS1CreatorInputError("prepared source bytes drifted")

    package_root = destination.resolve(strict=False)
    runner_argv = _build_runner_argv(
        package_root=package_root,
        future_output_dir=future_output,
        runtime=runtime,
        feedback_file_sha256=expected_feedback_bundle_file_sha256,
        codex_file_sha256=sha256_bytes(codex_bytes),
        codex_executable=executable,
        sparse_feedback=feedback.schema_version in {5, 6, 7, 8, 9, 10},
    )
    argv_bytes = runner_argv.canonical_bytes()
    preparer_source_sha256 = sha256_bytes(
        read_stable_regular_file(Path(__file__), label="S1 input preparer source")
    )
    runner_source_sha256 = sha256_bytes(
        read_stable_regular_file(RUNNER_PATH, label="S1 Creator runner source")
    )
    input_files = tuple(
        sorted(
            (
                _binding(
                    "parent_static_bank",
                    PARENT_BANK_FILE,
                    parent_bytes,
                    runtime.bank.bank_sha256,
                ),
                _binding(
                    "semantic_authoring_input",
                    SEMANTIC_INPUT_FILE,
                    semantic_bytes,
                    runtime.semantic_authoring_input.input_sha256,
                ),
                _binding(
                    "codex_authoring_input",
                    CODEX_INPUT_FILE,
                    codex_bytes,
                    codex_input.input_sha256,
                ),
                _binding(
                    "feedback_bundle",
                    FEEDBACK_BUNDLE_FILE,
                    feedback_bytes,
                    feedback.bundle_sha256,
                ),
            ),
            key=lambda item: item.role,
        )
    )
    manifest_payload = {
        "schema_version": 1,
        "kind": "portfolio-s1-creator-input-manifest",
        "policy_version": (
            SPARSE_PREPARATION_POLICY_VERSION
            if feedback.schema_version in {5, 6, 7, 8, 9, 10}
            else PREPARATION_POLICY_VERSION
        ),
        "status": "prepared_not_invoked",
        "provider_calls": 0,
        "source_runtime_root": str(runtime.root),
        "runtime_lock_file_sha256": runtime.runtime_lock_file_sha256,
        "runtime_lock_sha256": runtime.runtime_lock["runtime_lock_sha256"],
        "tool_registry_sha256": runtime.bank.tool_registry_sha256,
        "tool_registry_runtime_sha256": runtime.bank.tool_registry_runtime_sha256,
        "codex_executable": str(executable),
        "codex_executable_sha256": executable_sha256,
        "codex_executable_bytes": executable_bytes,
        "future_creator_output_dir": str(future_output),
        "preparer_source_file_sha256": preparer_source_sha256,
        "runner_source_file_sha256": runner_source_sha256,
        "input_files": [item.model_dump(mode="json") for item in input_files],
        "model_access_evidence_file": MODEL_ACCESS_EVIDENCE_FILE,
        "model_access_evidence_file_sha256": (
            expected_model_access_evidence_file_sha256
        ),
        "runner_argv_file": RUNNER_ARGV_FILE,
        "runner_argv_file_sha256": sha256_bytes(argv_bytes),
        "runner_invocation_sha256": runner_argv.invocation_sha256,
    }
    manifest = PortfolioS1CreatorInputManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest_payload)),
        },
        strict=True,
    )

    staging = new_staging_directory(destination)
    try:
        for name, content in {
            PARENT_BANK_FILE: parent_bytes,
            SEMANTIC_INPUT_FILE: semantic_bytes,
            CODEX_INPUT_FILE: codex_bytes,
            FEEDBACK_BUNDLE_FILE: feedback_bytes,
            MODEL_ACCESS_EVIDENCE_FILE: evidence_bytes,
            RUNNER_ARGV_FILE: argv_bytes,
            INPUT_MANIFEST_FILE: manifest.canonical_bytes(),
        }.items():
            atomic_create_file(staging / name, content)
        atomic_publish_new_directory(staging, destination)
    finally:
        _safe_cleanup_staging(staging, destination)
    return load_verified_portfolio_s1_creator_inputs(
        destination,
        expected_manifest_file_sha256=sha256_bytes(manifest.canonical_bytes()),
    )


def load_verified_portfolio_s1_creator_inputs(
    root: str | Path,
    *,
    expected_manifest_file_sha256: str,
) -> VerifiedPortfolioS1CreatorInputs:
    package_root = Path(root).absolute().resolve(strict=True)
    observed = frozenset(item.name for item in package_root.iterdir())
    if observed != _PACKAGE_FILES or any(
        item.is_symlink() or not item.is_file() for item in package_root.iterdir()
    ):
        raise PortfolioS1CreatorInputError("prepared S1 package layout drifted")
    manifest_bytes = read_stable_regular_file(
        package_root / INPUT_MANIFEST_FILE,
        label="S1 Creator input manifest",
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_file_sha256:
        raise PortfolioS1CreatorInputError("S1 Creator manifest file SHA drifted")
    manifest = _canonical_model(
        PortfolioS1CreatorInputManifest,
        manifest_bytes,
        label="S1 Creator input manifest",
    )
    runtime = load_verified_portfolio_static_opt_runtime_evidence(
        manifest.source_runtime_root,
        expected_runtime_lock_file_sha256=manifest.runtime_lock_file_sha256,
    )
    binding_by_role = {item.role: item for item in manifest.input_files}
    contents: dict[str, bytes] = {}
    for role, binding in binding_by_role.items():
        content = read_stable_regular_file(
            package_root / binding.file,
            label=f"prepared S1 input {role}",
            max_bytes=64 * 1024 * 1024,
        )
        if sha256_bytes(content) != binding.file_sha256:
            raise PortfolioS1CreatorInputError(f"prepared S1 {role} file drifted")
        contents[role] = content
    parent = _canonical_model(
        StaticBankArtifact,
        contents["parent_static_bank"],
        label="prepared S1 parent Bank",
    )
    semantic = load_authoring_packet(
        package_root / binding_by_role["semantic_authoring_input"].file,
        expected_file_sha256=binding_by_role["semantic_authoring_input"].file_sha256,
    )
    codex = load_codex_authoring_input(
        package_root / binding_by_role["codex_authoring_input"].file,
        expected_file_sha256=binding_by_role["codex_authoring_input"].file_sha256,
    )
    feedback = _load_typed_s1_feedback_bundle(
        package_root / binding_by_role["feedback_bundle"].file,
        expected_file_sha256=binding_by_role["feedback_bundle"].file_sha256,
    )
    access = load_codex_model_access_evidence(
        package_root / manifest.model_access_evidence_file,
        expected_file_sha256=manifest.model_access_evidence_file_sha256,
    )
    executable, executable_sha256, executable_bytes = _verify_codex_executable(
        manifest.codex_executable,
        access,
    )
    require_current_s1_codex_authoring_input(runtime, codex)
    if (
        parent != runtime.bank
        or semantic != runtime.semantic_authoring_input
        or feedback.parent_static_bank_sha256 != parent.bank_sha256
        or executable_sha256 != manifest.codex_executable_sha256
        or executable_bytes != manifest.codex_executable_bytes
        or manifest.runtime_lock_sha256 != runtime.runtime_lock["runtime_lock_sha256"]
        or manifest.tool_registry_sha256 != parent.tool_registry_sha256
        or manifest.tool_registry_runtime_sha256 != parent.tool_registry_runtime_sha256
        or binding_by_role["parent_static_bank"].artifact_sha256 != parent.bank_sha256
        or binding_by_role["semantic_authoring_input"].artifact_sha256
        != semantic.input_sha256
        or binding_by_role["codex_authoring_input"].artifact_sha256
        != codex.input_sha256
        or binding_by_role["feedback_bundle"].artifact_sha256 != feedback.bundle_sha256
        or codex.model_access_evidence_file_sha256 != access.file_sha256
    ):
        raise PortfolioS1CreatorInputError("prepared S1 identities drifted")
    argv_bytes = read_stable_regular_file(
        package_root / RUNNER_ARGV_FILE,
        label="prepared S1 runner argv",
    )
    if sha256_bytes(argv_bytes) != manifest.runner_argv_file_sha256:
        raise PortfolioS1CreatorInputError("prepared S1 runner argv file drifted")
    runner_argv = _canonical_model(
        PortfolioS1CreatorRunnerArgv,
        argv_bytes,
        label="prepared S1 runner argv",
    )
    sparse_feedback = feedback.schema_version in {5, 6, 7, 8, 9, 10}
    expected_preparation_policy = (
        SPARSE_PREPARATION_POLICY_VERSION
        if sparse_feedback
        else PREPARATION_POLICY_VERSION
    )
    expected_runner_policy = (
        SPARSE_RUNNER_POLICY_VERSION if sparse_feedback else RUNNER_POLICY_VERSION
    )
    if (
        runner_argv.invocation_sha256 != manifest.runner_invocation_sha256
        or manifest.policy_version != expected_preparation_policy
        or runner_argv.policy_version != expected_runner_policy
        or manifest.preparer_source_file_sha256
        != sha256_bytes(
            read_stable_regular_file(Path(__file__), label="S1 preparer source")
        )
        or manifest.runner_source_file_sha256
        != sha256_bytes(read_stable_regular_file(RUNNER_PATH, label="S1 runner source"))
        or os.path.lexists(manifest.future_creator_output_dir)
        or str(executable) not in runner_argv.command
    ):
        raise PortfolioS1CreatorInputError("prepared S1 execution contract drifted")
    return VerifiedPortfolioS1CreatorInputs(
        root=package_root,
        manifest=manifest,
        manifest_file_sha256=expected_manifest_file_sha256,
        runner_argv=runner_argv,
        runtime=runtime,
        parent_bank=parent,
        semantic_input=semantic,
        codex_input=codex,
        feedback_bundle=feedback,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-lock-file-sha256", required=True)
    parser.add_argument("--feedback-bundle", type=Path, required=True)
    parser.add_argument("--feedback-bundle-file-sha256", required=True)
    parser.add_argument("--model-access-evidence", type=Path, required=True)
    parser.add_argument("--model-access-evidence-file-sha256", required=True)
    parser.add_argument("--codex-executable", type=Path, required=True)
    parser.add_argument("--future-creator-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepared = prepare_portfolio_s1_creator_inputs(
            static_runtime_root=args.static_runtime_root,
            expected_runtime_lock_file_sha256=args.runtime_lock_file_sha256,
            feedback_bundle_path=args.feedback_bundle,
            expected_feedback_bundle_file_sha256=(args.feedback_bundle_file_sha256),
            model_access_evidence_path=args.model_access_evidence,
            expected_model_access_evidence_file_sha256=(
                args.model_access_evidence_file_sha256
            ),
            codex_executable_path=args.codex_executable,
            future_creator_output_dir=args.future_creator_output_dir,
            output_dir=args.output_dir,
        )
    except (
        FileExistsError,
        OSError,
        PortfolioS1CreatorInputError,
        ValueError,
    ) as error:
        print(f"prepare-s1-creator-inputs: {error}", file=sys.stderr)
        return 2
    print(prepared.root / INPUT_MANIFEST_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
