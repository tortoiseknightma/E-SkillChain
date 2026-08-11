"""Deterministic runtime policy for the third Codex-CLI authoring candidate.

The v2 contract remains byte-for-byte historical evidence.  This module reuses
its authoring payload and compiler contracts, but replaces ambient executable
discovery and environment inheritance with a machine-bound absolute binary and
a child environment constructed from an empty mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

from skillchain import codex_authoring as v2
from skillchain.codex_authoring import (  # Re-export the unchanged data contract.
    CanonicalCodexAuthoringRequest,
    CodexAuthoringContractError,
    CodexAuthoringInput,
    CodexAuthoringOutputContract,
    CodexSessionBudget,
    build_codex_authoring_input,
    build_codex_authoring_request,
    build_codex_output_contract,
    load_codex_authoring_input,
    load_codex_model_access_evidence,
    normalize_codex_authoring_output,
    parse_codex_authoring_request,
    project_codex_cli_output_schema,
    render_codex_authoring_stdin,
    validate_codex_cli_output_schema,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
)


CODEX_AUTHOR_CANDIDATE_ID = "authoring-codex-high-20260724-v3"
CODEX_AUTHOR_RUN_ID = "llm-static-codex-primary-20260724-high-v3"
CODEX_EXECUTION_TYPE = v2.CODEX_EXECUTION_TYPE
CODEX_EVIDENCE_TIER = v2.CODEX_EVIDENCE_TIER
CODEX_COMMAND_SHAPE = v2.CODEX_COMMAND_SHAPE
CODEX_FROZEN_BINARY_PATH = Path(
    r"C:\Users\torto\AppData\Local\Programs\OpenAI\Codex\bin\codex.exe"
)
CODEX_FROZEN_BINARY_BYTES = 359245096
CODEX_FROZEN_BINARY_SHA256 = (
    "83751f15cb6a0a7b97df67752c001e3fe1c20e18ffbfec3ff63567296205eb6c"
)
CODEX_FROZEN_CLI_VERSION = "0.145.0"

# Canonical output names.  Lower-case proxy aliases are accepted only when they
# are the sole spelling in the parent mapping and are canonicalized to upper
# case.  PATH, PATHEXT, TEMP, and TMP are deliberately absent.
CODEX_INHERITED_ENV_NAMES = (
    "ALL_PROXY",
    "APPDATA",
    "CODEX_HOME",
    "COMSPEC",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LOCALAPPDATA",
    "NO_PROXY",
    "PROGRAMDATA",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "USERPROFILE",
    "WINDIR",
)
CODEX_CONSTRUCTED_ENV_NAMES = ("PATH", "TEMP", "TMP")
CODEX_ENV_POLICY_VERSION = "codex-author-deterministic-env-v2"

CODEX_V3_TRUSTED_SOURCE_FILES = (
    "scripts/run_codex_authoring_v3.py",
    "src/skillchain/codex_authoring_v3.py",
)


def _regular_file_identity(path: Path) -> tuple[int, str]:
    """Hash a stable regular file without following a symbolic-link endpoint."""
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CodexAuthoringContractError(
            f"Codex binary is not a regular non-symlink file: {path}"
        )
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after_read = os.fstat(handle.fileno())
    after = path.lstat()
    snapshots = (opened, after_read, after)
    identity = (
        before.st_size,
        before.st_mtime_ns,
        getattr(before, "st_ino", 0),
    )
    if any(
        (
            snapshot.st_size,
            snapshot.st_mtime_ns,
            getattr(snapshot, "st_ino", 0),
        )
        != identity
        for snapshot in snapshots
    ):
        raise CodexAuthoringContractError(
            "Codex binary changed while its identity was being verified"
        )
    return before.st_size, digest


def freeze_codex_binary_identity(path: str | Path) -> dict[str, object]:
    """Build the exact machine-bound binary identity used by the v3 runtime."""
    raw = Path(path)
    if not raw.is_absolute():
        raise CodexAuthoringContractError("Codex binary path must be absolute")
    resolved = raw.resolve(strict=True)
    if resolved.name.casefold() != "codex.exe":
        raise CodexAuthoringContractError("frozen Codex binary must be codex.exe")
    size, digest = _regular_file_identity(resolved)
    if (
        size != CODEX_FROZEN_BINARY_BYTES
        or digest != CODEX_FROZEN_BINARY_SHA256
    ):
        raise CodexAuthoringContractError(
            "absolute Codex binary differs from the model-access evidence"
        )
    return {
        "path": str(resolved),
        "path_policy": "frozen_absolute_path_no_path_lookup",
        "bytes": size,
        "sha256": digest,
        "cli_version": CODEX_FROZEN_CLI_VERSION,
    }


def resolve_frozen_codex_binary(runtime: Mapping[str, Any]) -> Path:
    """Resolve only the absolute path recorded in the runtime lock."""
    binary = runtime.get("binary")
    if not isinstance(binary, dict) or set(binary) != {
        "path",
        "path_policy",
        "bytes",
        "sha256",
        "cli_version",
    }:
        raise CodexAuthoringContractError(
            "Codex v3 runtime lacks an exact absolute binary identity"
        )
    if binary.get("path_policy") != "frozen_absolute_path_no_path_lookup":
        raise CodexAuthoringContractError("Codex binary path policy drifted")
    path_value = binary.get("path")
    if not isinstance(path_value, str):
        raise CodexAuthoringContractError("Codex binary path is invalid")
    path = Path(path_value)
    if not path.is_absolute():
        raise CodexAuthoringContractError("Codex binary path is not absolute")
    resolved = path.resolve(strict=True)
    if os.path.normcase(str(resolved)) != os.path.normcase(path_value):
        raise CodexAuthoringContractError("Codex binary path was not canonical")
    size, digest = _regular_file_identity(resolved)
    if (
        resolved.name.casefold() != "codex.exe"
        or size != binary.get("bytes")
        or digest != binary.get("sha256")
        or binary.get("cli_version") != CODEX_FROZEN_CLI_VERSION
    ):
        raise CodexAuthoringContractError(
            "live absolute Codex binary differs from the runtime lock"
        )
    return resolved


def _canonical_parent_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Select inherited values with Windows case-insensitive ambiguity checks."""
    spellings: dict[str, str] = {}
    values: dict[str, str] = {}
    for raw_name, raw_value in environment.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            raise CodexAuthoringContractError(
                "Codex parent environment must contain string names and values"
            )
        folded = raw_name.casefold()
        previous = spellings.get(folded)
        if previous is not None and previous != raw_name:
            raise CodexAuthoringContractError(
                "Codex parent environment has case-insensitive duplicate names"
            )
        spellings[folded] = raw_name
        values[folded] = raw_value

    selected: dict[str, str] = {}
    for canonical in CODEX_INHERITED_ENV_NAMES:
        value = values.get(canonical.casefold())
        if value is not None:
            selected[canonical] = value
    return selected


def _constructed_path(binary: Mapping[str, object]) -> str:
    path_value = binary.get("path")
    if not isinstance(path_value, str):
        raise CodexAuthoringContractError("Codex binary path is invalid")
    parent = str(Path(path_value).parent)
    if not parent or os.pathsep in parent or "\n" in parent or "\r" in parent:
        raise CodexAuthoringContractError(
            "Codex binary parent cannot form a deterministic PATH"
        )
    return parent


def build_codex_v3_environment_policy(
    *,
    binary: Mapping[str, object],
    parent_environment: Mapping[str, str],
) -> dict[str, object]:
    """Freeze inherited commitments and deterministic value derivations."""
    inherited = _canonical_parent_environment(parent_environment)
    path_value = _constructed_path(binary)
    return {
        "policy_version": CODEX_ENV_POLICY_VERSION,
        "inherited_names": list(CODEX_INHERITED_ENV_NAMES),
        "inherited_present_value_sha256": {
            name: sha256_bytes(value.encode("utf-8"))
            for name, value in sorted(inherited.items())
        },
        "constructed_values": {
            "PATH": {
                "derivation": "parent_directory_of_frozen_codex_binary",
                "value_sha256": sha256_bytes(path_value.encode("utf-8")),
            },
            "TEMP": {
                "derivation": "new_external_invocation_temp_directory",
            },
            "TMP": {
                "derivation": "same_as_TEMP",
            },
        },
        "parent_path_inherited": False,
        "parent_pathext_inherited": False,
        "parent_temp_inherited": False,
        "all_other_environment_variables_removed": True,
        "credential_values_recorded": False,
    }


def construct_codex_v3_environment(
    *,
    runtime: Mapping[str, Any],
    executable: Path,
    invocation_temp: Path,
    parent_environment: Mapping[str, str],
) -> tuple[dict[str, str], str, str]:
    """Construct the exact child environment and return policy/instance hashes."""
    binary = runtime.get("binary")
    policy = runtime.get("environment_policy")
    if not isinstance(binary, dict) or not isinstance(policy, dict):
        raise CodexAuthoringContractError(
            "Codex v3 runtime environment policy is missing"
        )
    if os.path.normcase(str(executable)) != os.path.normcase(
        str(binary.get("path"))
    ):
        raise CodexAuthoringContractError(
            "Codex executable differs from the environment policy binary"
        )
    expected_policy = build_codex_v3_environment_policy(
        binary=binary,
        parent_environment=parent_environment,
    )
    if canonical_json_bytes(policy) != canonical_json_bytes(expected_policy):
        raise CodexAuthoringContractError(
            "live inherited Codex environment differs from the v3 commitments"
        )

    temp = invocation_temp.resolve(strict=True)
    if not temp.is_dir() or any(temp.iterdir()):
        raise CodexAuthoringContractError(
            "Codex invocation TEMP must be a new empty directory"
        )
    inherited = _canonical_parent_environment(parent_environment)
    child = dict(inherited)
    child["PATH"] = _constructed_path(binary)
    child["TEMP"] = str(temp)
    child["TMP"] = str(temp)
    if set(child) - (
        set(CODEX_INHERITED_ENV_NAMES) | set(CODEX_CONSTRUCTED_ENV_NAMES)
    ):
        raise CodexAuthoringContractError(
            "Codex child environment contains an unapproved name"
        )
    policy_sha256 = sha256_bytes(canonical_json_bytes(policy))
    instance = {
        "policy_sha256": policy_sha256,
        "present_value_sha256": {
            name: sha256_bytes(value.encode("utf-8"))
            for name, value in sorted(child.items())
        },
        "temp_and_tmp_same": child["TEMP"] == child["TMP"],
    }
    return child, policy_sha256, sha256_bytes(canonical_json_bytes(instance))


def build_codex_runtime_dependency_snapshot_v3(root: str | Path) -> dict[str, object]:
    """Extend the v2 dependency closure with the immutable v3 policy and runner."""
    repository = Path(root).resolve(strict=True)
    snapshot = v2.build_codex_runtime_dependency_snapshot(repository)
    trusted = list(snapshot["trusted_sources"])
    known = {item["file"] for item in trusted}
    for relative in CODEX_V3_TRUSTED_SOURCE_FILES:
        if relative in known:
            raise CodexAuthoringContractError(
                f"duplicate Codex v3 trusted source: {relative}"
            )
        path = repository / relative
        content = read_stable_regular_file(
            path,
            label=f"Codex v3 trusted source {relative}",
        )
        trusted.append(
            {
                "file": relative,
                "bytes": len(content),
                "file_sha256": sha256_bytes(content),
            }
        )
    snapshot["schema_version"] = 3
    snapshot["snapshot_policy"] = (
        "v2_recursive_runtime_closure_plus_v3_deterministic_environment_sources"
    )
    snapshot["trusted_sources"] = sorted(trusted, key=lambda item: item["file"])
    return snapshot


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
    "CODEX_V3_TRUSTED_SOURCE_FILES",
    "CanonicalCodexAuthoringRequest",
    "CodexAuthoringContractError",
    "CodexAuthoringInput",
    "CodexAuthoringOutputContract",
    "CodexSessionBudget",
    "build_codex_authoring_input",
    "build_codex_authoring_request",
    "build_codex_output_contract",
    "build_codex_runtime_dependency_snapshot_v3",
    "build_codex_v3_environment_policy",
    "construct_codex_v3_environment",
    "freeze_codex_binary_identity",
    "load_codex_authoring_input",
    "load_codex_model_access_evidence",
    "normalize_codex_authoring_output",
    "parse_codex_authoring_request",
    "project_codex_cli_output_schema",
    "render_codex_authoring_stdin",
    "resolve_frozen_codex_binary",
    "validate_codex_cli_output_schema",
]
