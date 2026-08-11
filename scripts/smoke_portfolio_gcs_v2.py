"""Run the GCS-v2 artifact path with a deterministic, network-denied model.

This smoke deliberately writes into a disposable execution-root clone.  It
uses the verified Core launch, Static runtime, real Query objects, real tool
registry, and materialized runtime sources, but replaces ``skillchain.llm.chat``
before any Assistant execution.  It therefore exercises checkpoint/sidecar
production and the read-only analyzer without a provider or budget-ledger call.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
from typing import Callable, Iterator, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.analyze_portfolio_matrix import analyze_portfolio_matrix  # noqa: E402
from scripts.run_portfolio_assistant_smoke import (  # noqa: E402
    _hash,
    _model,
    _self_model,
    _treatment,
)
from scripts.run_portfolio_shard import (  # noqa: E402
    _assistant_result,
    _assistant_row,
    _build_runner,
    _inputs_for_launch,
    _load_control,
    _request,
    _verify_scorer_sidecar_binding,
)
from skillchain import config  # noqa: E402
from skillchain import llm as llm_module  # noqa: E402
from skillchain.evaluation.assistant_runs import (  # noqa: E402
    AssistantRequestSnapshot,
    BackboneLock,
    InferenceBudget,
    _registry_lock,
)
from skillchain.evaluation.portfolio_gcs import (  # noqa: E402
    GCS_CAPABILITY_ORDER,
    GCS_V2_POLICY_SHA256,
    GCS_V2_POLICY_VERSION,
)
from skillchain.evaluation.portfolio_gcs_evidence import (  # noqa: E402
    GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    classify_style_query_v2,
    expected_multi_items_from_call_v2,
    make_public_scorer_evidence_v2,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    load_portfolio_launch_package,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (  # noqa: E402
    load_verified_portfolio_static_opt_runtime,
    validate_portfolio_static_opt_execution_control,
)
from skillchain.llm import LLMResponse, LLMToolCall, LLMUsage  # noqa: E402
from skillchain.runners.assistant import (  # noqa: E402
    GCS_V2_MODEL_RESPONSE_CONTRACT_VERSION,
    PortfolioStaticOptAssistantRunner,
    RunnerOwnedAssistantExecution,
)
from skillchain.schemas import Query  # noqa: E402
from skillchain.synthesis.store import atomic_create_file  # noqa: E402
from skillchain.tools.portfolio_runtime import PORTFOLIO_SYSTEM_PROMPT  # noqa: E402
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
)


_STATIC_SCOPE = "static_opt_rollout"
_PREFERRED_TOOL = {
    "knowledge.visual_encyclopedia": "encyclopedia_lookup",
    "product.exact_match": "image_product_search",
    "product.multi_search": "multi_product_search",
    "product.style_recommendation": "style_similar_search",
    "utility.document_reading": "document_ocr",
    "utility.recipe_guidance": "recipe_lookup",
}


class OfflineSmokeError(ValueError):
    """Raised when the zero-provider smoke is incomplete or non-deterministic."""


class OfflineNetworkAccessError(RuntimeError):
    """Raised on any attempted socket use during the offline smoke."""


class _OfflineStaticOptRunner(PortfolioStaticOptAssistantRunner):
    """Reuse the production path while intentionally bypassing paid-ledger entry.

    The production exact type requires a ``PortfolioAssistantBudgetContext``.
    This private smoke-only subclass is never accepted by the production shard
    entrypoint; its inherited entry guard consequently accepts ``None`` and the
    replacement chat adapter can run without reserving a provider call.
    """

    __slots__ = ()


@dataclass(frozen=True)
class ProbeRecord:
    label: str
    query_id: str
    split: str
    capability_id: str
    checkpoint_file_sha256: str
    checkpoint_row_sha256: str
    scorer_evidence_sha256: str
    style_submode: str | None
    style_support_status: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preauthorization-zero-provider",
        action="store_true",
        help=(
            "Use a blocked preauthorization launch and create every fake artifact "
            "under the system temporary directory."
        ),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        help="Prepared Static opt execution root; this source is never modified.",
    )
    parser.add_argument(
        "--smoke-execution-root",
        type=Path,
        help="New disposable root receiving fake-model schema-v2 checkpoints.",
    )
    parser.add_argument(
        "--analysis-output-parent",
        type=Path,
        help="New/existing external parent for deterministic analyzer replays.",
    )
    parser.add_argument(
        "--launch-root",
        type=Path,
        help="Blocked Static opt launch root for preauthorization mode.",
    )
    parser.add_argument(
        "--launch-plan-file-sha256",
        help="External file digest of the blocked preauthorization launch plan.",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help="Verified one-Bank Static runtime root for preauthorization mode.",
    )
    parser.add_argument(
        "--runtime-lock-file-sha256",
        help="External file digest of the verified Static runtime lock.",
    )
    parser.add_argument(
        "--batch-id",
        help="Static 25-query batch to smoke; defaults to the first launch batch.",
    )
    parser.add_argument(
        "--legacy-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Repeat for each immutable schema-v1 checkpoint (for example v5/v7).",
    )
    parser.add_argument(
        "--legacy-runtime-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Repeat for an immutable legacy runtime (for example treatment v7) "
            "that must not load as the one-Bank GCS-v2 Static runtime."
        ),
    )
    parser.add_argument(
        "--max-style-attempts",
        type=int,
        default=128,
        help="Maximum real Style Queries tried while locating each outcome.",
    )
    return parser


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def verify_legacy_checkpoint_rejected(path: Path) -> dict[str, object]:
    """Prove v2 rejects a schema-v1 row without changing one input byte."""

    path = path.absolute()
    before = read_stable_regular_file(
        path, label="legacy Assistant checkpoint", max_bytes=16 * 1024 * 1024
    )
    before_sha256 = sha256_bytes(before)
    try:
        _assistant_row(path, require_schema_v2=True, include_scorer_evidence=True)
    except ValueError as error:
        if "rejects legacy Assistant checkpoint" not in str(error):
            raise OfflineSmokeError(
                f"legacy checkpoint failed for a different reason: {path}"
            ) from error
        reason = "legacy_schema_v1_checkpoint_rejected"
    else:  # pragma: no cover - fail-closed guard
        raise OfflineSmokeError(f"GCS v2 accepted a legacy checkpoint: {path}")
    after = read_stable_regular_file(
        path, label="legacy Assistant checkpoint recheck", max_bytes=16 * 1024 * 1024
    )
    after_sha256 = sha256_bytes(after)
    if after != before or after_sha256 != before_sha256:
        raise OfflineSmokeError(f"legacy checkpoint changed during audit: {path}")
    return {
        "path": path.as_posix(),
        "status": "passed_immutable_rejection",
        "reason_code": reason,
        "bytes": len(before),
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
    }


def verify_legacy_runtime_rejected(root: Path) -> dict[str, object]:
    """Prove an old treatment runtime cannot enter the GCS-v2 Static chain."""

    root = root.absolute()
    lock_path = root / "runtime-lock.json"
    lock_bytes = read_stable_regular_file(
        lock_path, label="legacy runtime lock", max_bytes=16 * 1024 * 1024
    )
    lock_file_sha256 = sha256_bytes(lock_bytes)
    before = {
        name: sha256_bytes(content)
        for name, content in artifact_tree_bytes(root).items()
    }
    try:
        load_verified_portfolio_static_opt_runtime(
            root,
            expected_runtime_lock_file_sha256=lock_file_sha256,
        )
    except (OSError, ValueError) as error:
        rejection_type = type(error).__name__
    else:  # pragma: no cover - fail-closed guard
        raise OfflineSmokeError(f"GCS v2 accepted a legacy runtime: {root}")
    after = {
        name: sha256_bytes(content)
        for name, content in artifact_tree_bytes(root).items()
    }
    if after != before or sha256_bytes(lock_path.read_bytes()) != lock_file_sha256:
        raise OfflineSmokeError(f"legacy runtime changed during audit: {root}")
    return {
        "root": root.as_posix(),
        "status": "passed_immutable_rejection",
        "reason_code": "legacy_runtime_rejected_by_static_gcs_v2_loader",
        "rejection_type": rejection_type,
        "runtime_lock_file_sha256": lock_file_sha256,
        "artifact_count": len(before),
        "artifact_set_sha256": _hash(before),
    }


def _model_arguments(tool_name: str) -> dict[str, object]:
    if tool_name in {
        "image_product_search",
        "multi_product_search",
        "style_similar_search",
        "object_detect",
        "document_ocr",
    }:
        return {"asset_id": "query_asset"}
    if tool_name == "text_product_search":
        return {}
    if tool_name == "encyclopedia_lookup":
        return {"entity": "offline smoke entity"}
    if tool_name == "recipe_lookup":
        return {"dish": "offline smoke dish"}
    raise OfflineSmokeError(f"offline smoke has no arguments for tool: {tool_name}")


class DeterministicOfflineChat:
    """Stateful fake for one route, one tool action, and one final action."""

    def __init__(self) -> None:
        self.provider_calls = 0
        self.fake_model_calls = 0
        self.response_contract_observations = 0
        self._execution_index = 0
        self._action_turn = 0
        self._capability = ""
        self._tool_name = ""
        self._arguments: dict[str, object] = {}

    def configure(self, *, capability_id: str, tool_name: str) -> None:
        self._execution_index += 1
        self._action_turn = 0
        self._capability = capability_id
        self._tool_name = tool_name
        self._arguments = _model_arguments(tool_name)

    def __call__(self, provider: str, messages: list[dict], **kwargs) -> LLMResponse:
        if not self._capability or not self._tool_name:
            raise OfflineSmokeError("offline chat was used before configure()")
        self.fake_model_calls += 1
        model = kwargs.get("model")
        if provider != "qwen" or not isinstance(model, str) or not model:
            raise OfflineSmokeError("offline chat received an unexpected backbone")
        json_mode = kwargs.get("json_mode") is True
        tools = kwargs.get("tools")
        request_id = (
            f"offline-gcs-v2-{self._execution_index:04d}-{self.fake_model_calls:05d}"
        )
        if json_mode:
            text = canonical_json_bytes(
                {"selected_capability": self._capability}
            ).decode("utf-8")
            tool_calls: tuple[LLMToolCall, ...] = ()
            finish_reason = "stop"
            output_tokens = 4
        elif self._action_turn == 0:
            offered = {
                str(item.get("function", {}).get("name"))
                for item in (tools or [])
                if isinstance(item, dict)
            }
            if self._tool_name not in offered:
                raise OfflineSmokeError(
                    f"selected Skill did not expose {self._tool_name}"
                )
            self._action_turn = 1
            text = ""
            tool_calls = (
                LLMToolCall(
                    call_id=f"offline-tool-{self._execution_index:04d}",
                    name=self._tool_name,
                    arguments_json=canonical_json_bytes(self._arguments).decode(
                        "utf-8"
                    ),
                ),
            )
            finish_reason = "tool_calls"
            output_tokens = 4
        else:
            self._action_turn += 1
            tool_message = json.loads(messages[-1]["content"])
            response_contract = tool_message.get("final_response_contract")
            if (
                not isinstance(response_contract, dict)
                or response_contract.get("policy_version")
                != GCS_V2_MODEL_RESPONSE_CONTRACT_VERSION
            ):
                raise OfflineSmokeError(
                    "production tool message omitted the frozen GCS-v2 response contract"
                )
            self.response_contract_observations += 1
            text = (
                "Offline zero-provider artifact smoke. The local tool result is "
                "retained only to validate the frozen checkpoint contract."
            )
            tool_calls = ()
            finish_reason = "stop"
            output_tokens = 16
        return LLMResponse(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            requested_model=model,
            response_model=model,
            request_id=request_id,
            text=text,
            tool_calls=tool_calls,
            usage=LLMUsage(input_tokens=16, output_tokens=output_tokens),
            finish_reason=finish_reason,
            latency_ms=0,
        )


@contextmanager
def offline_chat_and_network_guard(
    chat: DeterministicOfflineChat,
) -> Iterator[None]:
    """Replace the only model adapter and deny Python socket access."""

    original_chat = llm_module.chat
    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyaddr = socket.gethostbyaddr
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex

    def denied(*_args, **_kwargs):
        raise OfflineNetworkAccessError(
            "network access is forbidden in the GCS-v2 artifact smoke"
        )

    class GuardedSocket(original_socket):
        def connect(self, *_args, **_kwargs):
            return denied()

        def connect_ex(self, *_args, **_kwargs):
            return denied()

    llm_module.chat = chat
    socket.socket = GuardedSocket
    socket.create_connection = denied
    socket.getaddrinfo = denied
    socket.gethostbyaddr = denied
    socket.gethostbyname = denied
    socket.gethostbyname_ex = denied
    try:
        yield
    finally:
        llm_module.chat = original_chat
        socket.socket = original_socket
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
        socket.gethostbyaddr = original_gethostbyaddr
        socket.gethostbyname = original_gethostbyname
        socket.gethostbyname_ex = original_gethostbyname_ex


def _clone_execution_control(source_root: Path, target_root: Path) -> dict:
    source_root = source_root.absolute()
    target_root = target_root.absolute()
    if target_root.exists():
        raise OfflineSmokeError("smoke execution root must not already exist")
    control = _load_control(source_root)
    target_root.mkdir(parents=True)
    for name in ("execution-control.json", "blinding-key.bin"):
        content = read_stable_regular_file(
            source_root / name, label=f"source {name}", max_bytes=4 * 1024 * 1024
        )
        atomic_create_file(target_root / name, content)
    if _load_control(target_root) != control:
        raise OfflineSmokeError("cloned execution control differs from its source")
    return control


def preauthorization_smoke_control_payload(
    *,
    launch,
    launch_root: Path,
    launch_plan_file_sha256: str,
    runtime,
    runtime_root: Path,
    runtime_lock_file_sha256: str,
    blinding_key: bytes,
) -> dict[str, object]:
    """Build a smoke-only control that production execution must reject."""

    if len(blinding_key) != 32:
        raise OfflineSmokeError("preauthorization smoke key must contain 32 bytes")
    plan = launch.plan
    lock = runtime.runtime_lock
    if (
        getattr(plan, "kind", None) != "portfolio-core-static-opt-800x1-launch-plan"
        or getattr(plan, "execution_mode", None) != _STATIC_SCOPE
        or getattr(plan, "execution_ready", None) is not False
        or not tuple(getattr(plan, "blockers", ()))
        or getattr(plan, "query_count", None) != 800
        or getattr(plan, "instance_count", None) != 800
        or getattr(plan, "shard_count", None) != 32
        or tuple(getattr(plan, "config_order", ())) != ("llm_static",)
        or tuple(getattr(plan, "selected_splits", ())) != ("opt_pool",)
    ):
        raise OfflineSmokeError(
            "preauthorization mode requires the exact blocked Static opt800 launch"
        )
    launch_artifacts = {
        str(item.artifact_id): item for item in getattr(plan, "artifacts", ())
    }
    runtime_binding = launch_artifacts.get("assistant_runtime_lock")
    if (
        runtime_binding is None
        or getattr(runtime_binding, "status", None) != "verified"
        or getattr(runtime_binding, "file_sha256", None) != runtime_lock_file_sha256
        or getattr(runtime_binding, "content_sha256", None)
        != lock["runtime_lock_sha256"]
    ):
        raise OfflineSmokeError(
            "blocked launch does not bind the supplied verified Static runtime"
        )
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "kind": "portfolio-preauthorization-zero-provider-smoke-control",
        "track": "portfolio",
        "formal_eligible": False,
        "status": "preauthorization_zero_provider_smoke_only",
        "preauthorization_zero_provider": True,
        "execution_authorized": False,
        "provider_calls_permitted": False,
        "budget_ledger_permitted": False,
        "execution_scope": _STATIC_SCOPE,
        "matrix_run_id": plan.matrix_run_id,
        "launch_root": launch_root.absolute().as_posix(),
        "launch_plan_file_sha256": launch_plan_file_sha256,
        "launch_plan_sha256": plan.launch_plan_sha256,
        "runtime_root": runtime_root.absolute().as_posix(),
        "runtime_lock_file_sha256": runtime_lock_file_sha256,
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "execution_artifact_aliases": [],
        "execution_artifact_alias_provider_model_call_count": 0,
        "blinding_key_sha256": sha256_bytes(blinding_key),
        # This deliberate mismatch is the production-entry kill switch.  The
        # verified runtime requires ["assistant", "gcs_v2"] for an authorized
        # execution control; the read-only analyzer does not consume this field.
        "evaluation_stages": ["preauthorization_zero_provider_smoke"],
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": GCS_V2_POLICY_VERSION,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "gcs_scorer_evidence_policy_version": (GCS_SCORER_EVIDENCE_V2_POLICY_VERSION),
        "static_bank_file_sha256": runtime.bank_file_sha256,
        "static_bank_sha256": runtime.bank.bank_sha256,
        "static_contract_refresh_receipt_file_sha256": (
            runtime.refresh_receipt_file_sha256
        ),
        "static_contract_refresh_receipt_sha256": runtime.refresh_receipt[
            "receipt_sha256"
        ],
        "semantic_authoring_input_file_sha256": (
            runtime.semantic_authoring_input_file_sha256
        ),
        "semantic_authoring_input_sha256": (
            runtime.semantic_authoring_input.input_sha256
        ),
        "core_runtime_sources_receipt_file_sha256": (
            runtime.core_source_receipt_file_sha256
        ),
        "core_runtime_sources_receipt_sha256": runtime.core_source_receipt[
            "receipt_sha256"
        ],
        "runtime_data_sha256": lock["runtime_data_sha256"],
        "task_spec_version": lock["task_spec_version"],
        "task_spec_sha256": lock["task_spec_sha256"],
        "task_spec_file_sha256": lock["task_spec_file_sha256"],
        "pairwise_judge_enabled": False,
        "legacy_final_judge_enabled": False,
        "analyzer_provider_call_count": 0,
        "authorized_shard_ids": [],
        "authorized_shards": [],
        "model_calls_performed": 0,
    }
    return {**unsigned, "control_sha256": _hash(unsigned)}


def prepare_preauthorization_smoke_root(
    *,
    launch_root: Path,
    launch_plan_file_sha256: str,
    runtime_root: Path,
    runtime_lock_file_sha256: str,
) -> tuple[Path, Path, dict[str, object]]:
    """Create a non-executable smoke control under the system TEMP root."""

    launch_root = launch_root.absolute()
    runtime_root = runtime_root.absolute()
    launch = load_portfolio_launch_package(
        launch_root,
        expected_plan_file_sha256=launch_plan_file_sha256,
    )
    runtime = load_verified_portfolio_static_opt_runtime(
        runtime_root,
        expected_runtime_lock_file_sha256=runtime_lock_file_sha256,
    )
    base = Path(tempfile.mkdtemp(prefix="portfolio-gcs-v2-preauth-"))
    system_temp = Path(tempfile.gettempdir()).absolute().resolve(strict=True)
    resolved_base = base.absolute().resolve(strict=True)
    if system_temp not in resolved_base.parents:
        raise OfflineSmokeError("preauthorization smoke root escaped system TEMP")
    execution_root = base / "smoke-execution"
    analysis_parent = base / "analysis"
    execution_root.mkdir()
    key = b"portfolio-gcs-v2-preauth-smoke!!"
    if len(key) != 32:  # pragma: no cover - source literal invariant
        raise AssertionError("preauthorization smoke key literal changed")
    control = preauthorization_smoke_control_payload(
        launch=launch,
        launch_root=launch_root,
        launch_plan_file_sha256=launch_plan_file_sha256,
        runtime=runtime,
        runtime_root=runtime_root,
        runtime_lock_file_sha256=runtime_lock_file_sha256,
        blinding_key=key,
    )
    atomic_create_file(
        execution_root / "execution-control.json", canonical_json_bytes(control)
    )
    atomic_create_file(execution_root / "blinding-key.bin", key)
    if _load_control(execution_root) != control:
        raise OfflineSmokeError("preauthorization smoke control failed self-check")
    try:
        validate_portfolio_static_opt_execution_control(control, runtime)
    except ValueError:
        pass
    else:  # pragma: no cover - critical kill-switch guard
        raise OfflineSmokeError(
            "production execution validator accepted a preauthorization control"
        )
    return execution_root, analysis_parent, control


def _backbone_budget(runtime) -> tuple[BackboneLock, InferenceBudget]:
    backbone_payload = {
        "provider": "qwen",
        "model": "qwen3-vl-flash-2026-01-22",
        "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": None,
        "system_prompt_sha256": sha256_bytes(PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")),
    }
    budget_payload = {
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "max_tool_calls": 3,
        "max_turns": 5,
        "timeout_ms": 180000,
    }
    return (
        _self_model(BackboneLock, backbone_payload, "identity_sha256"),
        _self_model(InferenceBudget, budget_payload, "budget_sha256"),
    )


def _style_observation(
    execution: RunnerOwnedAssistantExecution,
) -> tuple[str | None, str | None]:
    style_calls = tuple(
        item
        for item in execution.scorer_calls
        if item.payload_kind == "style_candidates_v2"
    )
    if not style_calls:
        return None, None
    if len(style_calls) != 1:
        raise OfflineSmokeError("Style probe produced multiple scorer calls")
    payload = style_calls[0].payload
    submode = payload.get("style_submode")
    status = payload.get("support_status")
    if not isinstance(submode, str) or not isinstance(status, str):
        raise OfflineSmokeError("Style scorer payload lacks mode or support status")
    return submode, status


def build_schema_v2_checkpoint(
    *,
    launch,
    member,
    request: AssistantRequestSnapshot,
    query: Query,
    execution: RunnerOwnedAssistantExecution,
) -> tuple[dict[str, object], str, str | None, str | None]:
    """Build the same atomic row envelope used by the production shard."""

    response, receipt = execution.response, execution.receipt
    if receipt.response_sha256 != _hash(_model(response)):
        raise OfflineSmokeError("runner response differs from its immutable receipt")
    result = _assistant_result(launch, request, response)
    multi_sets = tuple(
        expected
        for call in execution.scorer_calls
        if (expected := expected_multi_items_from_call_v2(call)) is not None
    )
    if len(multi_sets) > 1:
        raise OfflineSmokeError("one execution produced multiple multi item sets")
    sidecar = make_public_scorer_evidence_v2(
        matrix_run_id=launch.plan.matrix_run_id,
        instance_id=member.instance_sha256,
        request_sha256=request.request_sha256,
        query_id=member.query_id,
        config=member.config,
        query_artifact_sha256=launch.plan.query_artifact_sha256,
        assistant_result_sha256=_hash(_model(result)),
        assistant_receipt_sha256=receipt.receipt_sha256,
        calls=execution.scorer_calls,
        expected_multi_items=multi_sets[0] if multi_sets else None,
    )
    _verify_scorer_sidecar_binding(
        sidecar,
        launch=launch,
        member=member,
        request=request,
        result=result,
        receipt=receipt,
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "kind": "portfolio-assistant-checkpoint",
        "instance_sha256": member.instance_sha256,
        "query_ordinal": member.query_ordinal,
        "request": _model(request),
        "response": _model(response),
        "receipt": _model(receipt),
        "public_scorer_evidence": _model(sidecar),
    }
    row = {**payload, "row_sha256": _hash(payload)}
    style_submode, style_status = _style_observation(execution)
    return row, sidecar.evidence_sha256, style_submode, style_status


def publish_verified_checkpoint(
    path: Path,
    row: Mapping[str, object],
    *,
    verifier: Callable[[Path], object] | None = None,
) -> bytes:
    """Create once, or reuse only an exactly identical already-created row."""

    if row.get("schema_version") != 2 or "public_scorer_evidence" not in row:
        raise OfflineSmokeError("checkpoint is not a schema-v2 sidecar envelope")
    supplied = row.get("row_sha256")
    unsigned = dict(row)
    unsigned.pop("row_sha256", None)
    if supplied != _hash(unsigned):
        raise OfflineSmokeError("checkpoint row self hash is invalid")
    content = canonical_json_bytes(dict(row))
    if path.exists():
        existing = read_stable_regular_file(
            path, label="existing smoke checkpoint", max_bytes=16 * 1024 * 1024
        )
        if existing != content:
            raise OfflineSmokeError("create-only checkpoint has conflicting bytes")
    else:
        atomic_create_file(path, content)
    check = verifier or (
        lambda candidate: _assistant_row(
            candidate,
            require_schema_v2=True,
            include_scorer_evidence=True,
        )
    )
    check(path)
    if path.read_bytes() != content:
        raise OfflineSmokeError("verified checkpoint changed after publication")
    return content


def _execute_one(
    *,
    runner: _OfflineStaticOptRunner,
    chat: DeterministicOfflineChat,
    launch,
    shard,
    member,
    query: Query,
    public_query,
    treatment,
    backbone: BackboneLock,
    budget: InferenceBudget,
    registry_lock,
) -> tuple[dict[str, object], str, str | None, str | None]:
    capability = query.canonical_capability
    if capability not in _PREFERRED_TOOL:
        raise OfflineSmokeError(f"query lacks a supported capability: {query.query_id}")
    tool_name = _PREFERRED_TOOL[capability]
    request = _request(
        launch=launch,
        shard=shard,
        member=member,
        query=public_query,
        treatment=treatment,
        backbone=backbone,
        budget=budget,
        registry_lock=registry_lock,
    )
    chat.configure(capability_id=capability, tool_name=tool_name)
    execution = runner.execute(request, scorer_query=query)
    if execution.scorer_capture_policy_version != (
        GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
    ):
        raise OfflineSmokeError("runner omitted the GCS-v2 scorer capture policy")
    return build_schema_v2_checkpoint(
        launch=launch,
        member=member,
        request=request,
        query=query,
        execution=execution,
    )


def select_capability_probes(queries: Sequence[object]) -> tuple[object, ...]:
    """Choose six real dev/opt probes with both splits represented."""

    selected: list[object] = []
    for index, capability in enumerate(GCS_CAPABILITY_ORDER):
        preferred_split = "dev_mini" if index % 2 == 0 else "opt_pool"
        candidates = tuple(
            item
            for item in queries
            if getattr(item, "canonical_capability", None) == capability
            and getattr(item, "split", None) in {"dev_mini", "opt_pool"}
        )
        chosen = next(
            (
                item
                for item in candidates
                if getattr(item, "split", None) == preferred_split
            ),
            candidates[0] if candidates else None,
        )
        if chosen is None:
            raise OfflineSmokeError(f"Core dev/opt lacks capability: {capability}")
        selected.append(chosen)
    if {str(getattr(item, "split")) for item in selected} != {
        "dev_mini",
        "opt_pool",
    }:
        raise OfflineSmokeError("six-capability probes do not cover dev and opt")
    return tuple(selected)


def style_probe_pools(
    queries: Sequence[object],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    style_queries = tuple(
        item
        for item in queries
        if getattr(item, "canonical_capability", None) == "product.style_recommendation"
        and getattr(item, "split", None) in {"dev_mini", "opt_pool"}
    )
    same: list[object] = []
    cross: list[object] = []
    for item in style_queries:
        classification = classify_style_query_v2(str(getattr(item, "text")))
        if classification.style_submode == "same_category_alternative":
            same.append(item)
        else:
            cross.append(item)
    if not same or not cross:
        raise OfflineSmokeError("Core dev/opt lacks both Style submodes")
    return tuple(same), tuple(cross)


def require_probe_coverage(records: Sequence[ProbeRecord]) -> None:
    capabilities = {item.capability_id for item in records}
    if capabilities != set(GCS_CAPABILITY_ORDER):
        missing = sorted(set(GCS_CAPABILITY_ORDER) - capabilities)
        raise OfflineSmokeError(f"probe coverage lacks capabilities: {missing}")
    style_pairs = {
        (item.style_submode, item.style_support_status)
        for item in records
        if item.capability_id == "product.style_recommendation"
    }
    required = {
        ("same_category_alternative", "candidates"),
        ("cross_category_coordination", "candidates"),
        ("cross_category_coordination", "unsupported"),
    }
    if not required.issubset(style_pairs):
        raise OfflineSmokeError(
            f"Style smoke lacks same/mapped/unsupported outcomes: {sorted(style_pairs)}"
        )


def artifact_tree_bytes(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        raise OfflineSmokeError(f"artifact tree is missing: {root}")
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def compare_artifact_trees(first: Path, second: Path) -> dict[str, str]:
    first_tree = artifact_tree_bytes(first)
    second_tree = artifact_tree_bytes(second)
    if first_tree != second_tree:
        names = sorted(set(first_tree) | set(second_tree))
        drift = [
            name for name in names if first_tree.get(name) != second_tree.get(name)
        ]
        raise OfflineSmokeError(f"analyzer replay bytes drifted: {drift}")
    return {name: sha256_bytes(content) for name, content in first_tree.items()}


def run_deterministic_analysis_replay(
    execution_root: Path,
    *,
    batch_id: str,
    output_parent: Path,
    analyzer: Callable[..., object] = analyze_portfolio_matrix,
) -> dict[str, str]:
    output_parent = output_parent.absolute()
    first = output_parent / "analysis-replay-a"
    second = output_parent / "analysis-replay-b"
    if first.exists() or second.exists():
        raise OfflineSmokeError("analyzer replay output directories already exist")
    output_parent.mkdir(parents=True, exist_ok=True)
    analyzer(execution_root, batch_id=batch_id, output_dir=first)
    analyzer(execution_root, batch_id=batch_id, output_dir=second)
    return compare_artifact_trees(first, second)


def _selected_batch(launch, batch_id: str | None):
    batch_ids = tuple(
        dict.fromkeys(item.accepted_batch_id for item in launch.plan.shards)
    )
    selected = batch_id or (batch_ids[0] if batch_ids else None)
    if selected is None or selected not in batch_ids:
        raise OfflineSmokeError(f"unknown or empty Static batch: {selected}")
    shards = tuple(
        item for item in launch.plan.shards if item.accepted_batch_id == selected
    )
    if len(shards) != 1 or shards[0].config != "llm_static":
        raise OfflineSmokeError(
            "Static smoke batch is not exactly one llm_static shard"
        )
    shard = shards[0]
    members = tuple(
        sorted(
            (item for item in launch.instances if item.shard_id == shard.shard_id),
            key=lambda item: item.query_ordinal,
        )
    )
    if len(members) != 25:
        raise OfflineSmokeError("Static smoke batch does not contain 25 members")
    return selected, shard, members


def _probe_member(label: str, query: Query, ordinal: int):
    return SimpleNamespace(
        instance_sha256=_hash(
            {"kind": "offline-gcs-v2-probe", "label": label, "query_id": query.query_id}
        ),
        query_id=query.query_id,
        query_ordinal=ordinal,
        config="llm_static",
    )


def _record_for(
    *,
    label: str,
    query: Query,
    path: Path,
    content: bytes,
    sidecar_sha256: str,
    style_submode: str | None,
    style_status: str | None,
) -> ProbeRecord:
    raw = json.loads(content)
    return ProbeRecord(
        label=label,
        query_id=query.query_id,
        split=query.split,
        capability_id=str(query.canonical_capability),
        checkpoint_file_sha256=sha256_bytes(content),
        checkpoint_row_sha256=str(raw["row_sha256"]),
        scorer_evidence_sha256=sidecar_sha256,
        style_submode=style_submode,
        style_support_status=style_status,
    )


def run_zero_provider_smoke(
    *,
    control_root: Path | None,
    smoke_execution_root: Path,
    analysis_output_parent: Path,
    batch_id: str | None,
    legacy_checkpoints: Sequence[Path],
    legacy_runtime_roots: Sequence[Path],
    max_style_attempts: int,
    preauthorization_zero_provider: bool = False,
) -> dict[str, object]:
    if max_style_attempts < 1:
        raise OfflineSmokeError("max_style_attempts must be positive")
    legacy_audits = [
        verify_legacy_checkpoint_rejected(path) for path in legacy_checkpoints
    ]
    legacy_runtime_audits = [
        verify_legacy_runtime_rejected(path) for path in legacy_runtime_roots
    ]
    if preauthorization_zero_provider:
        if control_root is not None:
            raise OfflineSmokeError(
                "preauthorization smoke cannot clone an authorized control"
            )
        control = _load_control(smoke_execution_root)
        if (
            control.get("kind")
            != "portfolio-preauthorization-zero-provider-smoke-control"
            or control.get("preauthorization_zero_provider") is not True
            or control.get("execution_authorized") is not False
            or control.get("provider_calls_permitted") is not False
            or control.get("budget_ledger_permitted") is not False
            or control.get("evaluation_stages")
            != ["preauthorization_zero_provider_smoke"]
        ):
            raise OfflineSmokeError(
                "preauthorization smoke root lacks its non-executable control"
            )
    else:
        if control_root is None:
            raise OfflineSmokeError("authorized-control smoke requires control_root")
        control = _clone_execution_control(control_root, smoke_execution_root)
    if (
        control.get("execution_scope") != _STATIC_SCOPE
        or control.get("assistant_checkpoint_schema_version") != 2
        or control.get("gcs_policy_version") != GCS_V2_POLICY_VERSION
        or control.get("gcs_policy_sha256") != GCS_V2_POLICY_SHA256
        or control.get("gcs_scorer_evidence_policy_version")
        != GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
    ):
        raise OfflineSmokeError(
            "source execution control is not the exact GCS-v2 scope"
        )

    launch_root = _repository_path(str(control["launch_root"])).absolute()
    runtime_root = _repository_path(str(control["runtime_root"])).absolute()
    launch = load_portfolio_launch_package(
        launch_root,
        expected_plan_file_sha256=str(control["launch_plan_file_sha256"]),
    )
    static_runtime = load_verified_portfolio_static_opt_runtime(
        runtime_root,
        expected_runtime_lock_file_sha256=str(control["runtime_lock_file_sha256"]),
    )
    if preauthorization_zero_provider:
        try:
            validate_portfolio_static_opt_execution_control(control, static_runtime)
        except ValueError:
            pass
        else:  # pragma: no cover - critical kill-switch guard
            raise OfflineSmokeError(
                "production validator accepted preauthorization smoke control"
            )
    else:
        validate_portfolio_static_opt_execution_control(control, static_runtime)
    selected_batch_id, shard, members = _selected_batch(launch, batch_id)
    inputs = _inputs_for_launch(launch.plan)
    private_by_id = {item.query_id: item for item in inputs.queries}
    public_by_id = {item.query_id: item for item in inputs.assistant_queries}

    chat = DeterministicOfflineChat()
    with offline_chat_and_network_guard(chat):
        runtime, banks, catalog, _production_runner = _build_runner(
            runtime_root,
            str(control["runtime_lock_file_sha256"]),
            inputs=inputs,
            static_opt=True,
        )
        runner = _OfflineStaticOptRunner(
            registry=runtime.registry,
            system_prompt=PORTFOLIO_SYSTEM_PROMPT,
            bank=banks["llm_static"],
            asset_catalog=catalog,
            runtime_lock=static_runtime.runtime_lock,
            runtime_lock_file_sha256=str(control["runtime_lock_file_sha256"]),
        )
        backbone, budget = _backbone_budget(runtime)
        registry_lock = _registry_lock(runtime.registry)
        treatment = _treatment("llm_static", banks["llm_static"].bank_sha256)

        for member in members:
            query = private_by_id[member.query_id]
            row, _sidecar_sha, _submode, _status = _execute_one(
                runner=runner,
                chat=chat,
                launch=launch,
                shard=shard,
                member=member,
                query=query,
                public_query=public_by_id[member.query_id],
                treatment=treatment,
                backbone=backbone,
                budget=budget,
                registry_lock=registry_lock,
            )
            publish_verified_checkpoint(
                smoke_execution_root / member.assistant_output_relpath, row
            )

        probe_root = smoke_execution_root / "offline-probes"
        records: list[ProbeRecord] = []
        probe_ordinal = 0

        def execute_probe(label: str, query: Query, *, publish: bool = True):
            nonlocal probe_ordinal
            member = _probe_member(label, query, probe_ordinal)
            probe_ordinal += 1
            probe_shard = SimpleNamespace(config="llm_static")
            row, sidecar_sha, submode, status = _execute_one(
                runner=runner,
                chat=chat,
                launch=launch,
                shard=probe_shard,
                member=member,
                query=query,
                public_query=public_by_id[query.query_id],
                treatment=treatment,
                backbone=backbone,
                budget=budget,
                registry_lock=registry_lock,
            )
            if not publish:
                return row, sidecar_sha, submode, status
            path = probe_root / f"{label}-{query.query_id}.json"
            content = publish_verified_checkpoint(path, row)
            record = _record_for(
                label=label,
                query=query,
                path=path,
                content=content,
                sidecar_sha256=sidecar_sha,
                style_submode=submode,
                style_status=status,
            )
            records.append(record)
            return row, sidecar_sha, submode, status

        for query in select_capability_probes(inputs.queries):
            execute_probe(f"capability-{query.canonical_capability}", query)

        same_pool, cross_pool = style_probe_pools(inputs.queries)
        style_targets = (
            (
                "style-same-candidates",
                same_pool,
                "same_category_alternative",
                "candidates",
            ),
            (
                "style-mapped-cross",
                cross_pool,
                "cross_category_coordination",
                "candidates",
            ),
            (
                "style-unsupported-cross",
                cross_pool,
                "cross_category_coordination",
                "unsupported",
            ),
        )
        for label, pool, expected_submode, expected_status in style_targets:
            matched: Query | None = None
            matched_result = None
            for query in pool[:max_style_attempts]:
                result = execute_probe(label, query, publish=False)
                if result[2:] == (expected_submode, expected_status):
                    matched = query
                    matched_result = result
                    break
            if matched is None or matched_result is None:
                raise OfflineSmokeError(
                    f"no real Core Query produced {label} within "
                    f"{min(len(pool), max_style_attempts)} attempts"
                )
            # Re-execute the selected query so the published record is a complete,
            # independently verified terminal checkpoint.
            execute_probe(label, matched)

        require_probe_coverage(records)
        analyzer_hashes = run_deterministic_analysis_replay(
            smoke_execution_root,
            batch_id=selected_batch_id,
            output_parent=analysis_output_parent,
        )

    if chat.provider_calls != 0:
        raise OfflineSmokeError("offline adapter recorded a provider call")
    summary_unsigned: dict[str, object] = {
        "schema_version": 1,
        "kind": "portfolio-gcs-v2-zero-provider-artifact-smoke",
        "status": "passed",
        "execution_scope": _STATIC_SCOPE,
        "preauthorization_zero_provider": preauthorization_zero_provider,
        "execution_authorized": not preauthorization_zero_provider,
        "smoke_execution_root": smoke_execution_root.absolute().as_posix(),
        "selected_batch_id": selected_batch_id,
        "batch_checkpoint_count": len(members),
        "probe_checkpoint_count": len(records),
        "capabilities": list(GCS_CAPABILITY_ORDER),
        "legacy_checkpoint_audits": legacy_audits,
        "legacy_runtime_audits": legacy_runtime_audits,
        "probe_records": [item.__dict__ for item in records],
        "analyzer_output_file_sha256s": analyzer_hashes,
        "deterministic_fake_model_calls": chat.fake_model_calls,
        "response_contract_observations": chat.response_contract_observations,
        "provider_calls": chat.provider_calls,
        "budget_ledger_calls": 0,
        "network_policy": "python_sockets_fail_closed",
    }
    summary = {
        **summary_unsigned,
        "smoke_sha256": _hash(summary_unsigned),
    }
    atomic_create_file(
        smoke_execution_root / "gcs-v2-zero-provider-smoke.json",
        canonical_json_bytes(summary),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preauthorization_zero_provider:
        forbidden = (
            args.control_root,
            args.smoke_execution_root,
            args.analysis_output_parent,
        )
        required = (
            args.launch_root,
            args.launch_plan_file_sha256,
            args.runtime_root,
            args.runtime_lock_file_sha256,
        )
        if any(item is not None for item in forbidden) or any(
            item is None for item in required
        ):
            raise OfflineSmokeError(
                "preauthorization mode requires launch/runtime roots+digests and "
                "forbids caller-selected smoke/output roots"
            )
        smoke_execution_root, analysis_output_parent, _control = (
            prepare_preauthorization_smoke_root(
                launch_root=args.launch_root,
                launch_plan_file_sha256=args.launch_plan_file_sha256,
                runtime_root=args.runtime_root,
                runtime_lock_file_sha256=args.runtime_lock_file_sha256,
            )
        )
        control_root = None
    else:
        if any(
            item is None
            for item in (
                args.control_root,
                args.smoke_execution_root,
                args.analysis_output_parent,
            )
        ) or any(
            item is not None
            for item in (
                args.launch_root,
                args.launch_plan_file_sha256,
                args.runtime_root,
                args.runtime_lock_file_sha256,
            )
        ):
            raise OfflineSmokeError(
                "authorized-control mode requires control/smoke/analysis roots and "
                "does not accept standalone launch/runtime arguments"
            )
        control_root = args.control_root
        smoke_execution_root = args.smoke_execution_root
        analysis_output_parent = args.analysis_output_parent
    summary = run_zero_provider_smoke(
        control_root=control_root,
        smoke_execution_root=smoke_execution_root,
        analysis_output_parent=analysis_output_parent,
        batch_id=args.batch_id,
        legacy_checkpoints=args.legacy_checkpoint,
        legacy_runtime_roots=args.legacy_runtime_root,
        max_style_attempts=args.max_style_attempts,
        preauthorization_zero_provider=args.preauthorization_zero_provider,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
