"""Build, deploy, probe, and lock the formal authoring container runtime."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = ROOT / "deploy" / "authoring" / "locks"
IMAGE_TAG = "skillchain-authoring:formal-v1"
INTERNAL_NETWORK = "skillchain-authoring-internal-v1"
EXTERNAL_NETWORK = "skillchain-authoring-egress-v1"
PROXY_CONTAINER = "skillchain-authoring-egress-proxy-v1"


class DeploymentError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        check=False,
        capture_output=True,
        shell=False,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise DeploymentError(f"command failed: {arguments[:3]}: {stderr}")
    return completed


def docker_json(docker: str, *arguments: str) -> object:
    output = run(docker, *arguments).stdout
    try:
        return json.loads(output)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentError(
            f"Docker returned invalid JSON for {arguments}"
        ) from error


def remove_existing(docker: str) -> None:
    run(docker, "rm", "--force", PROXY_CONTAINER, check=False)
    for network in (INTERNAL_NETWORK, EXTERNAL_NETWORK):
        inspect = run(docker, "network", "inspect", network, check=False)
        if inspect.returncode == 0:
            raw = json.loads(inspect.stdout)
            if len(raw) != 1 or raw[0].get("Name") != network:
                raise DeploymentError(
                    f"refusing to replace ambiguous network {network}"
                )
            containers = raw[0].get("Containers") or {}
            if containers:
                raise DeploymentError(
                    f"network {network} still has attached containers"
                )
            run(docker, "network", "rm", network)


def image_digest(docker: str, reference: str) -> str:
    raw = docker_json(docker, "image", "inspect", reference)
    if not isinstance(raw, list) or len(raw) != 1:
        raise DeploymentError(f"image inspection is ambiguous: {reference}")
    image_id = raw[0].get("Id")
    if not isinstance(image_id, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", image_id
    ):
        raise DeploymentError(f"image ID is invalid: {reference}")
    return image_id.removeprefix("sha256:")


def probe(docker: str, image_reference: str, script: str) -> dict[str, object]:
    command = [
        docker,
        "run",
        "--rm",
        "--pull=never",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--network",
        INTERNAL_NETWORK,
        image_reference,
        "python",
        "-I",
        "-c",
        script,
    ]
    completed = run(*command, check=False)
    return {
        "command_sha256": digest(canonical_bytes(command)),
        "exit_code": completed.returncode,
        "stdout_sha256": digest(completed.stdout),
        "stderr_sha256": digest(completed.stderr),
    }


def encoded_lock(payload: object) -> tuple[bytes, str]:
    content = canonical_bytes(payload)
    return content, digest(content)


def publish_lock_bundle(files: dict[str, bytes]) -> None:
    LOCK_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{LOCK_ROOT.name}.staging-",
            dir=LOCK_ROOT.parent,
        )
    )
    try:
        for name, content in sorted(files.items()):
            with (staging_root / name).open("xb") as handle:
                handle.write(content)
        staging_root.replace(LOCK_ROOT)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def main() -> int:
    global LOCK_ROOT, IMAGE_TAG, INTERNAL_NETWORK, EXTERNAL_NETWORK, PROXY_CONTAINER

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider-endpoint",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--credential-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--runtime-version", default="formal-v1")
    parser.add_argument("--lock-directory", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"formal-v[1-9][0-9]*", args.runtime_version):
        raise DeploymentError("runtime version must match formal-vN")
    lock_relative = Path(args.lock_directory)
    if (
        lock_relative.is_absolute()
        or ".." in lock_relative.parts
        or lock_relative.as_posix() != args.lock_directory
        or not re.fullmatch(
            r"deploy/authoring/locks-v[1-9][0-9]*",
            lock_relative.as_posix(),
        )
    ):
        raise DeploymentError(
            "lock directory must be a new deploy/authoring/locks-vN namespace"
        )
    LOCK_ROOT = (ROOT / lock_relative).resolve(strict=False)
    try:
        LOCK_ROOT.relative_to(ROOT.resolve(strict=True))
    except ValueError as error:
        raise DeploymentError("lock directory escapes the repository") from error
    if os.path.lexists(LOCK_ROOT):
        raise DeploymentError(
            "lock directory already exists; formal lock bundles are create-only"
        )
    suffix = args.runtime_version.removeprefix("formal-")
    IMAGE_TAG = f"skillchain-authoring:{args.runtime_version}"
    INTERNAL_NETWORK = f"skillchain-authoring-internal-{suffix}"
    EXTERNAL_NETWORK = f"skillchain-authoring-egress-{suffix}"
    PROXY_CONTAINER = f"skillchain-authoring-egress-proxy-{suffix}"
    endpoint = urlparse(args.provider_endpoint)
    if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username:
        raise DeploymentError("provider endpoint must be credential-free HTTPS")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", args.credential_env):
        raise DeploymentError("credential environment variable name is invalid")

    docker_path = shutil.which("docker")
    if docker_path is None:
        raise DeploymentError("Docker CLI is not installed")
    docker = str(Path(docker_path).resolve(strict=True))
    run(docker, "info")

    base_tag = "python:3.12-slim"
    run(docker, "pull", base_tag)
    base_raw = docker_json(docker, "image", "inspect", base_tag)
    repo_digests = (
        base_raw[0].get("RepoDigests") if isinstance(base_raw, list) else None
    )
    if not isinstance(repo_digests, list):
        raise DeploymentError("Python base image has no repository digest")
    pinned_bases = sorted(
        value
        for value in repo_digests
        if isinstance(value, str) and value.startswith("python@sha256:")
    )
    if not pinned_bases:
        raise DeploymentError("Python base image digest could not be pinned")
    pinned_base = pinned_bases[0]
    run(
        docker,
        "build",
        "--pull=false",
        "--build-arg",
        f"PYTHON_BASE={pinned_base}",
        "--tag",
        IMAGE_TAG,
        "--file",
        str(ROOT / "deploy" / "authoring" / "Dockerfile"),
        str(ROOT),
    )
    locked_image_digest = image_digest(docker, IMAGE_TAG)
    image_reference = f"sha256:{locked_image_digest}"
    import_probe = run(
        docker,
        "run",
        "--rm",
        "--pull=never",
        image_reference,
        "python",
        "-I",
        "-c",
        "from skillchain.runners.authoring_worker import main; print('worker-import-ok')",
    )

    remove_existing(docker)
    run(docker, "network", "create", "--driver", "bridge", EXTERNAL_NETWORK)
    run(
        docker,
        "network",
        "create",
        "--driver",
        "bridge",
        "--internal",
        INTERNAL_NETWORK,
    )
    run(
        docker,
        "run",
        "--detach",
        "--name",
        PROXY_CONTAINER,
        "--pull=never",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=256m",
        "--cpus=0.5",
        "--restart=unless-stopped",
        "--network",
        EXTERNAL_NETWORK,
        "--tmpfs=/tmp:rw,noexec,nosuid,size=4m",
        "--env",
        f"SKILLCHAIN_ALLOWED_HOST={endpoint.hostname}",
        "--env",
        "SKILLCHAIN_ALLOWED_PORT=443",
        image_reference,
        "python",
        "-I",
        "-m",
        "skillchain.runners.egress_proxy",
    )
    run(
        docker,
        "network",
        "connect",
        "--alias",
        PROXY_CONTAINER,
        INTERNAL_NETWORK,
        PROXY_CONTAINER,
    )
    ready = False
    for _attempt in range(30):
        logs = run(docker, "logs", PROXY_CONTAINER, check=False).stdout
        if b'"event":"ready"' in logs:
            ready = True
            break
        time.sleep(0.5)
    if not ready:
        raise DeploymentError("egress proxy did not become ready")

    network_raw = docker_json(docker, "network", "inspect", INTERNAL_NETWORK)
    external_raw = docker_json(docker, "network", "inspect", EXTERNAL_NETWORK)
    proxy_raw = docker_json(docker, "container", "inspect", PROXY_CONTAINER)
    network = network_raw[0]
    external_network = external_raw[0]
    proxy = proxy_raw[0]
    if (
        network.get("Internal") is not True
        or external_network.get("Internal") is not False
    ):
        raise DeploymentError("deployed Docker network modes are invalid")
    if set((proxy.get("NetworkSettings") or {}).get("Networks") or {}) != {
        INTERNAL_NETWORK,
        EXTERNAL_NETWORK,
    }:
        raise DeploymentError("proxy is not attached to exactly two locked networks")
    proxy_environment = (proxy.get("Config") or {}).get("Env") or []
    if any(value.startswith(f"{args.credential_env}=") for value in proxy_environment):
        raise DeploymentError("provider credential leaked into the egress proxy")

    connect_prefix = (
        "import socket; s=socket.create_connection(('"
        + PROXY_CONTAINER
        + "',3128),5); "
    )
    allowed_script = (
        connect_prefix
        + "s.sendall(b'CONNECT "
        + endpoint.hostname.encode("ascii").decode("ascii")
        + ":443 HTTP/1.1\\r\\nHost: "
        + endpoint.hostname.encode("ascii").decode("ascii")
        + ":443\\r\\n\\r\\n'); r=s.recv(128); print(r.split(b'\\r\\n',1)[0].decode()); "
        + "raise SystemExit(0 if r.startswith(b'HTTP/1.1 200') else 1)"
    )
    denied_script = (
        connect_prefix
        + "s.sendall(b'CONNECT example.com:443 HTTP/1.1\\r\\nHost: example.com:443\\r\\n\\r\\n'); "
        + "r=s.recv(128); print(r.split(b'\\r\\n',1)[0].decode()); "
        + "raise SystemExit(0 if r.startswith(b'HTTP/1.1 403') else 1)"
    )
    direct_script = (
        "import socket;\ntry: socket.create_connection(('1.1.1.1',443),3)\n"
        "except OSError: print('direct-egress-blocked'); raise SystemExit(0)\n"
        "raise SystemExit(1)"
    )
    dns_script = (
        "import socket;\ntry: socket.getaddrinfo('example.com',443)\n"
        "except socket.gaierror: print('external-dns-blocked'); raise SystemExit(0)\n"
        "raise SystemExit(1)"
    )
    probes = {
        "allowlisted_connect_succeeds": probe(docker, image_reference, allowed_script),
        "non_allowlisted_connect_denied": probe(docker, image_reference, denied_script),
        "direct_ip_egress_blocked": probe(docker, image_reference, direct_script),
        "external_dns_blocked": probe(docker, image_reference, dns_script),
    }
    if any(value["exit_code"] != 0 for value in probes.values()):
        raise DeploymentError("one or more network policy probes failed")

    deployed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    policy_unsigned = {
        "allowed_hostname": endpoint.hostname,
        "allowed_port": 443,
        "authoring_network_id": network["Id"],
        "authoring_network_internal": True,
        "authoring_network_name": INTERNAL_NETWORK,
        "denies_direct_egress": True,
        "denies_ip_literal_proxy_targets": True,
        "denies_non_allowlisted_proxy_targets": True,
        "enforcement": "docker-internal-network-single-dual-homed-connect-proxy-v1",
        "proxy_container_name": PROXY_CONTAINER,
        "proxy_external_network_id": external_network["Id"],
        "proxy_external_network_name": EXTERNAL_NETWORK,
        "proxy_image_digest": locked_image_digest,
        "proxy_url": f"http://{PROXY_CONTAINER}:3128",
        "schema_version": 1,
    }
    policy = {
        **policy_unsigned,
        "policy_sha256": digest(canonical_bytes(policy_unsigned)),
    }
    policy_bytes, policy_file_sha = encoded_lock(policy)
    policy_path = LOCK_ROOT / "network-policy.json"

    engine_bytes = Path(docker).read_bytes()
    profile_unsigned = {
        "allowed_provider_endpoint": args.provider_endpoint,
        "cpu_count_milli": 1000,
        "credential_env_name": args.credential_env,
        "engine": "docker",
        "engine_binary_sha256": digest(engine_bytes),
        "engine_path": docker,
        "image_digest": locked_image_digest,
        "image_reference": image_reference,
        "memory_megabytes": 512,
        "network_id": network["Id"],
        "network_name": INTERNAL_NETWORK,
        "network_policy_sha256": policy_file_sha,
        "pids_limit": 64,
        "proxy_container_name": PROXY_CONTAINER,
        "proxy_external_network_name": EXTERNAL_NETWORK,
        "proxy_image_digest": locked_image_digest,
        "proxy_url": f"http://{PROXY_CONTAINER}:3128",
        "schema_version": 2,
        "timeout_seconds": 600,
    }
    profile = {
        **profile_unsigned,
        "profile_sha256": digest(canonical_bytes(profile_unsigned)),
    }
    profile_bytes, profile_file_sha = encoded_lock(profile)
    profile_path = LOCK_ROOT / "sandbox-profile.json"

    receipt_unsigned = {
        "base_image_reference": pinned_base,
        "deployed_at_utc": deployed_at,
        "docker_server_version": docker_json(
            docker, "version", "--format", "{{json .}}"
        )["Server"]["Version"],
        "engine_binary_sha256": digest(engine_bytes),
        "image_digest": locked_image_digest,
        "import_probe_stdout_sha256": digest(import_probe.stdout),
        "network_id": network["Id"],
        "network_policy_file_sha256": policy_file_sha,
        "probes": probes,
        "proxy_container_id": proxy["Id"],
        "proxy_image_digest": proxy["Image"].removeprefix("sha256:"),
        "runtime_version": args.runtime_version,
        "sandbox_profile_file_sha256": profile_file_sha,
        "schema_version": 1,
    }
    receipt = {
        **receipt_unsigned,
        "receipt_sha256": digest(canonical_bytes(receipt_unsigned)),
    }
    receipt_bytes, receipt_file_sha = encoded_lock(receipt)
    receipt_path = LOCK_ROOT / "deployment-receipt.json"
    manifest = {
        "deployment_receipt_file_sha256": receipt_file_sha,
        "network_policy_file_sha256": policy_file_sha,
        "sandbox_profile_file_sha256": profile_file_sha,
        "schema_version": 1,
    }
    manifest_bytes, manifest_file_sha = encoded_lock(manifest)
    manifest_path = LOCK_ROOT / "lock-manifest.json"
    publish_lock_bundle(
        {
            "deployment-receipt.json": receipt_bytes,
            "lock-manifest.json": manifest_bytes,
            "network-policy.json": policy_bytes,
            "sandbox-profile.json": profile_bytes,
        }
    )
    print(
        json.dumps(
            {
                "deployment_receipt": str(receipt_path),
                "image_reference": image_reference,
                "lock_manifest": str(manifest_path),
                "lock_manifest_sha256": manifest_file_sha,
                "network_policy": str(policy_path),
                "sandbox_profile": str(profile_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
