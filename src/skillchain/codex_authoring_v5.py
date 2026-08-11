"""Prompt-contract repair for the fifth Codex authoring candidate.

Candidate v5 preserves the complete v4 runtime and transactional boundary.
It changes only the Author-visible prompt: object-detector predictions must be
described as a ``predicted class name`` instead of copying schema field
vocabulary that the trusted compiler reserves for experiment metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from skillchain import codex_authoring_v4 as v4
from skillchain.codex_authoring_v4 import *  # noqa: F403
from skillchain.static_authoring import PromptIdentity
from skillchain.tools.serialization import (
    canonical_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
)


CODEX_AUTHOR_CANDIDATE_ID = "authoring-codex-high-20260724-v5"
CODEX_AUTHOR_RUN_ID = "llm-static-codex-primary-20260724-high-v5"
CODEX_EXECUTION_TYPE = v4.CODEX_EXECUTION_TYPE
CODEX_EVIDENCE_TIER = v4.CODEX_EVIDENCE_TIER
CODEX_COMMAND_SHAPE = v4.CODEX_COMMAND_SHAPE
CODEX_FROZEN_BINARY_PATH = v4.CODEX_FROZEN_BINARY_PATH
CODEX_FROZEN_BINARY_BYTES = v4.CODEX_FROZEN_BINARY_BYTES
CODEX_FROZEN_BINARY_SHA256 = v4.CODEX_FROZEN_BINARY_SHA256
CODEX_FROZEN_CLI_VERSION = v4.CODEX_FROZEN_CLI_VERSION
CODEX_INHERITED_ENV_NAMES = v4.CODEX_INHERITED_ENV_NAMES
CODEX_CONSTRUCTED_ENV_NAMES = v4.CODEX_CONSTRUCTED_ENV_NAMES
CODEX_ENV_POLICY_VERSION = v4.CODEX_ENV_POLICY_VERSION

V5_PROMPT_ID = "static-author-v5-codex-compiler-vocabulary"
V5_PROMPT_VERSION = "5.0.1-codex"
V5_PROMPT_SUFFIX = (
    "\n\nIn authored prose, describe an object detector's class prediction "
    'only as a "predicted class name". Do not copy schema field-name '
    "vocabulary into instructions. This wording rule applies to objectives, "
    "steps, and fallback instructions."
)

CODEX_V5_TRUSTED_SOURCE_FILES = (
    "scripts/run_codex_authoring_v5.py",
    "src/skillchain/codex_authoring_v5.py",
)


class CreateOnceConflictError(FileExistsError):
    """A create-only destination exists but is not this caller's exact bytes."""


@dataclass(frozen=True)
class CreateOnceResult:
    """Classified result of a create-once commit."""

    committed: bool
    recovered_after_exception: bool
    cleanup_error: str | None


def _exception_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _real_directory(path: Path, label: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a real directory: {path}")


def _read_exact_destination(path: Path, content: bytes, label: str) -> bool:
    """Return whether a stable regular destination contains the exact bytes."""
    if not os.path.lexists(path):
        return False
    try:
        observed = read_stable_regular_file(path, label=label)
    except BaseException as error:
        raise CreateOnceConflictError(
            f"{label} exists but cannot be verified as this caller's commit"
        ) from error
    if observed != content:
        raise CreateOnceConflictError(
            f"{label} is occupied by different bytes: {path}"
        )
    return True


def classified_atomic_create(
    path: str | Path,
    content: bytes,
    *,
    label: str,
) -> CreateOnceResult:
    """Create a file once and classify ambiguous post-link exceptions.

    The content must contain a caller-unique ownership nonce when this primitive
    is used for an authorization claim.  If ``os.link`` commits and a later
    operation raises, an exact stable reread classifies the destination as
    committed by this caller.  A different destination is always a conflict.
    Temporary-file cleanup failures are returned as evidence instead of
    obscuring a successful commit.
    """

    destination = Path(path)
    if not isinstance(content, bytes) or not content:
        raise ValueError(f"{label} content must be non-empty bytes")
    _real_directory(destination.parent, f"{label} parent")
    if os.path.lexists(destination):
        raise CreateOnceConflictError(f"{label} already exists: {destination}")

    temporary: Path | None = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    link_returned = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        link_returned = True
    except BaseException as error:
        primary_error = error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as error:
                cleanup_error = error

    committed = False
    try:
        committed = _read_exact_destination(destination, content, label)
    except CreateOnceConflictError:
        if link_returned:
            raise RuntimeError(
                f"{label} changed after this caller committed it"
            ) from None
        raise

    if committed:
        warning = cleanup_error or primary_error
        return CreateOnceResult(
            committed=True,
            recovered_after_exception=primary_error is not None,
            cleanup_error=_exception_text(warning) if warning is not None else None,
        )
    if primary_error is not None:
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error
    raise RuntimeError(f"{label} create returned without a committed destination")


def build_codex_runtime_dependency_snapshot_v5(
    root: str | Path,
) -> dict[str, object]:
    """Extend the v4 dependency closure with the immutable v5 sources."""

    repository = Path(root).resolve(strict=True)
    snapshot = v4.build_codex_runtime_dependency_snapshot_v4(repository)
    trusted = list(snapshot["trusted_sources"])
    known = {item["file"] for item in trusted}
    for relative in CODEX_V5_TRUSTED_SOURCE_FILES:
        if relative in known:
            raise v4.CodexAuthoringContractError(
                f"duplicate Codex v5 trusted source: {relative}"
            )
        path = repository / relative
        content = read_stable_regular_file(
            path,
            label=f"Codex v5 trusted source {relative}",
        )
        trusted.append(
            {
                "file": relative,
                "bytes": len(content),
                "file_sha256": sha256_bytes(content),
            }
        )
    snapshot["schema_version"] = 5
    snapshot["snapshot_policy"] = (
        "v4_runtime_closure_plus_v5_explicit_compiler_vocabulary_prompt"
    )
    snapshot["trusted_sources"] = sorted(trusted, key=lambda item: item["file"])
    return snapshot


# Explicit aliases make the inherited runtime contract visible to static
# readers and avoid relying on wildcard-import implementation details.
CodexAuthoringContractError = v4.CodexAuthoringContractError
build_codex_authoring_request = v4.build_codex_authoring_request
build_codex_output_contract = v4.build_codex_output_contract
build_codex_v5_environment_policy = v4.build_codex_v4_environment_policy
construct_codex_v5_environment = v4.construct_codex_v4_environment
freeze_codex_binary_identity = v4.freeze_codex_binary_identity
load_codex_authoring_input = v4.load_codex_authoring_input
load_codex_model_access_evidence = v4.load_codex_model_access_evidence
normalize_codex_authoring_output = v4.normalize_codex_authoring_output
parse_codex_authoring_request = v4.parse_codex_authoring_request
render_codex_authoring_stdin = v4.render_codex_authoring_stdin
resolve_frozen_codex_binary = v4.resolve_frozen_codex_binary
validate_codex_cli_output_schema = v4.validate_codex_cli_output_schema


def build_codex_authoring_input(*args: Any, **kwargs: Any):
    """Build the v4 input and replace only its Author-visible prompt identity."""

    base = v4.build_codex_authoring_input(*args, **kwargs)
    payload = base.model_dump(mode="json")
    template = base.prompt.template + V5_PROMPT_SUFFIX
    prompt = PromptIdentity(
        prompt_id=V5_PROMPT_ID,
        prompt_version=V5_PROMPT_VERSION,
        template=template,
        prompt_sha256=sha256_bytes(template.encode("utf-8")),
    )
    payload["prompt"] = prompt.model_dump(mode="json")
    payload.pop("input_sha256")
    payload["input_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return type(base).model_validate(payload, strict=True)


__all__ = [
    "CODEX_AUTHOR_CANDIDATE_ID",
    "CODEX_AUTHOR_RUN_ID",
    "CODEX_COMMAND_SHAPE",
    "CODEX_CONSTRUCTED_ENV_NAMES",
    "CODEX_ENV_POLICY_VERSION",
    "CODEX_EVIDENCE_TIER",
    "CODEX_EXECUTION_TYPE",
    "CODEX_FROZEN_BINARY_BYTES",
    "CODEX_FROZEN_BINARY_PATH",
    "CODEX_FROZEN_BINARY_SHA256",
    "CODEX_FROZEN_CLI_VERSION",
    "CODEX_INHERITED_ENV_NAMES",
    "CODEX_V5_TRUSTED_SOURCE_FILES",
    "CodexAuthoringContractError",
    "CreateOnceConflictError",
    "CreateOnceResult",
    "build_codex_authoring_input",
    "build_codex_authoring_request",
    "build_codex_output_contract",
    "build_codex_runtime_dependency_snapshot_v5",
    "build_codex_v5_environment_policy",
    "classified_atomic_create",
    "construct_codex_v5_environment",
    "freeze_codex_binary_identity",
    "load_codex_authoring_input",
    "load_codex_model_access_evidence",
    "normalize_codex_authoring_output",
    "parse_codex_authoring_request",
    "render_codex_authoring_stdin",
    "resolve_frozen_codex_binary",
    "validate_codex_cli_output_schema",
]
