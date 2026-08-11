"""Run the zero-provider Static-opt -> Feedback -> S1 contract smoke.

This is deliberately not an experiment result.  It deeply loads the immutable
Static opt800 execution and frozen Discovery600 fold, runs all 240 Feedback rows
through a deterministic local substitute, prepares the current Style-2.3
Creator package, invokes the real S1 runner exactly once with a deterministic
local Codex process substitute, and verifies the derived two-Bank runtime.

No Assistant population is executed.  Replay and body-gate status is therefore
reported as ``not_run_requires_paid_candidate_execution``.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import socket
import sys
from types import TracebackType
from typing import Literal


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.prepare_portfolio_s1_creator_inputs import (  # noqa: E402
    PARENT_BANK_FILE,
    SEMANTIC_INPUT_FILE,
    CODEX_INPUT_FILE,
    prepare_portfolio_s1_creator_inputs,
)
from scripts.run_portfolio_evolution_model import (  # noqa: E402
    CodexProcessResult,
    run_portfolio_evolution_model,
)
from scripts.run_portfolio_s1_feedback import (  # noqa: E402
    PreparedPortfolioS1FeedbackRun,
    execute_run,
    prepare_run,
)
from skillchain import config  # noqa: E402
from skillchain.evaluation.evaluator_outputs import (  # noqa: E402
    GroundedFeedbackFinding,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
    VisualFeedbackOutput,
)
from skillchain.evaluation.feedback_runtime import (  # noqa: E402
    FeedbackEvaluationResult,
    VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6,
)
from skillchain.evaluation.packets import (  # noqa: E402
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
)
from skillchain.evaluation.portfolio_s1_experiment_runtime import (  # noqa: E402
    create_portfolio_s1_experiment_runtime,
)
from skillchain.llm import LLMUsage  # noqa: E402
from skillchain.synthesis.store import (  # noqa: E402
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import read_stable_regular_file  # noqa: E402


SMOKE_POLICY_VERSION = "portfolio-s1-feedback-loop-zero-provider-smoke-v2"
REPLAY_NOT_RUN = "not_run_requires_paid_candidate_execution"


class PortfolioS1ZeroProviderSmokeError(RuntimeError):
    """The local, network-denied S1 smoke failed closed."""


class _NetworkDenied(AbstractContextManager["_NetworkDenied"]):
    """Fail immediately if any code path attempts an outbound connection."""

    def __init__(self) -> None:
        self.attempts = 0
        self._create_connection = socket.create_connection
        self._socket_connect = socket.socket.connect
        self._socket_connect_ex = socket.socket.connect_ex

    def _deny(self, *_args, **_kwargs):
        self.attempts += 1
        raise PortfolioS1ZeroProviderSmokeError(
            "zero-provider smoke attempted a network connection"
        )

    def __enter__(self) -> "_NetworkDenied":
        socket.create_connection = self._deny
        socket.socket.connect = self._deny
        socket.socket.connect_ex = self._deny
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        socket.create_connection = self._create_connection
        socket.socket.connect = self._socket_connect
        socket.socket.connect_ex = self._socket_connect_ex
        return False


@dataclass
class _FakeFeedbackRunner:
    calls: int = 0

    def __call__(
        self,
        packet,
        _isolation,
        *,
        remote_runtime,
        max_tokens,
        max_completion_tokens,
        timeout_seconds,
        record_usage,
    ):
        if record_usage is not True:
            raise PortfolioS1ZeroProviderSmokeError(
                "Feedback smoke must exercise usage-recording semantics"
            )
        if (
            max_tokens is not None
            or max_completion_tokens != 4096
            or timeout_seconds != 600
        ):
            raise PortfolioS1ZeroProviderSmokeError(
                "Feedback smoke did not receive the frozen Qwen completion contract"
            )
        self.calls += 1
        feedback = VisualFeedbackOutput(
            schema_version=1,
            summary=(
                "Deterministic local smoke feedback; this is contract evidence, "
                "not a model judgment."
            ),
            rule_violations=(
                GroundedFeedbackFinding(
                    dimension="tool_use",
                    severity="low",
                    grounded_in_image=False,
                    description="Preserve the visible evidence contract.",
                    evidence=("Locally generated smoke evidence.",),
                ),
            ),
            ideal_response_gaps=(),
            skill_suggestions=(
                "[policy_compatible] Keep claims, cards, citations, and tool "
                "evidence mutually closed.",
            ),
        )
        raw_text = canonical_json_bytes(feedback.model_dump(mode="json")).decode(
            "utf-8"
        )
        payload = {
            "schema_version": 5,
            "cache_namespace": "feedback-evaluator-v11",
            "parser_policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
            "parser_policy_sha256": VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
            "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
            "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
            "transport_policy_version": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6,
            "transport_policy_sha256": VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V6,
            "requested_response_format": "json_schema",
            "requested_json_schema_sha256": VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
            "requested_thinking": True,
            "requested_thinking_budget": 2048,
            "requested_timeout_seconds": timeout_seconds,
            "requested_temperature": None,
            "requested_top_p": None,
            "query_id": packet.query_id,
            "packet_sha256": packet.packet_sha256,
            "prompt_sha256": sha256_bytes(
                b"portfolio-s1-zero-provider-feedback-prompt\x00"
                + packet.packet_sha256.encode("ascii")
            ),
            "image_sha256": packet.image.sha256,
            "wire_sha256": sha256_bytes(
                b"portfolio-s1-zero-provider-feedback-wire\x00"
                + packet.packet_sha256.encode("ascii")
            ),
            "asset_catalog_sha256": remote_runtime.catalog.catalog_sha256,
            "remote_authorization_id": remote_runtime.authorization.authorization_id,
            "remote_authorization_file_sha256": (
                remote_runtime.authorization_file_sha256
            ),
            "remote_receipt_file_sha256": remote_runtime.receipt_file_sha256,
            "remote_receipt_sha256": remote_runtime.receipt.receipt_sha256,
            "provider": "qwen",
            "model": "qwen3.8-max",
            "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "status": "parsed",
            "request_id": f"local-smoke-{packet.packet_sha256[:20]}",
            "raw_response_text": raw_text,
            "raw_response_sha256": sha256_bytes(raw_text.encode("utf-8")),
            "raw_response_bytes": len(raw_text.encode("utf-8")),
            "tool_calls": (),
            "tool_call_count": 0,
            "parsed_feedback": feedback,
            "usage": LLMUsage(input_tokens=0, output_tokens=0),
            "finish_reason": "stop",
            "latency_ms": 0,
            "reasoning_present": False,
            "reasoning_tokens": None,
            "reasoning_bytes": 0,
            "reasoning_sha256": None,
            "error_code": None,
        }
        unsigned = FeedbackEvaluationResult.model_construct(
            **payload, result_sha256="0" * 64
        )
        result_hash = sha256_bytes(
            canonical_json_bytes(
                unsigned.model_dump(mode="json", exclude={"result_sha256"})
            )
        )
        return FeedbackEvaluationResult.model_validate(
            {**payload, "result_sha256": result_hash}, strict=True
        )


@dataclass
class _FakeCodexProcess:
    response: bytes
    calls: int = 0

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        stdin: bytes,
        cwd: Path,
        environment,
        timeout_seconds: int,
    ) -> CodexProcessResult:
        del environment
        if self.calls != 0:
            raise PortfolioS1ZeroProviderSmokeError(
                "fake Codex process was invoked more than once"
            )
        if timeout_seconds != 600 or tuple(cwd.iterdir()):
            raise PortfolioS1ZeroProviderSmokeError(
                "real S1 runner did not preserve its clean-turn contract"
            )
        self.calls += 1
        final_path = Path(command[command.index("--output-last-message") + 1])
        final_path.write_bytes(self.response)
        text = self.response.decode("utf-8")
        thread_id = "portfolio-s1-local-zero-provider-smoke"
        events = (
            {
                "type": "thread.started",
                "thread_id": thread_id,
            },
            {
                "type": "turn.started",
                "thread_id": thread_id,
            },
            {
                "type": "item.completed",
                "thread_id": thread_id,
                "item": {
                    "id": "local-smoke-message",
                    "type": "agent_message",
                    "text": text,
                },
            },
            {
                "type": "turn.completed",
                "thread_id": thread_id,
                "usage": {
                    # Positive deterministic local counters satisfy the same
                    # receipt schema as a real clean turn. They are not billable
                    # usage and never enter the external-provider call totals.
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 30,
                },
            },
        )
        return CodexProcessResult(
            returncode=0,
            stdout=b"".join(canonical_json_bytes(item) for item in events),
            stderr=b"",
        )


def _file_sha(path: Path) -> str:
    return sha256_bytes(read_stable_regular_file(path, label=path.name))


def _feedback_arguments(
    args: argparse.Namespace, output_dir: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        execution_root=args.execution_root,
        expected_execution_control_sha256=args.expected_execution_control_sha256,
        artifact_repository_root=args.artifact_repository_root,
        selection_profile="discovery240",
        creator_selection_index=None,
        expected_creator_selection_sha256=None,
        fold_manifest=args.fold_manifest,
        expected_fold_manifest_sha256=args.expected_fold_manifest_sha256,
        fold_mapping=args.fold_mapping,
        expected_fold_mapping_sha256=args.expected_fold_mapping_sha256,
        rubric_file=args.rubric_file,
        expected_rubric_sha256=args.expected_rubric_sha256,
        rubric_id=args.rubric_id,
        rubric_version=args.rubric_version,
        output_dir=output_dir,
        authorization_id=args.authorization_id,
        reviewer_id=args.reviewer_id,
        reviewed_at=args.reviewed_at,
        owner_statement=args.owner_statement,
        approved_phase_hard_cap_cny="94.000000000000",
        run_id=args.run_id,
        model_source_lock=args.model_source_lock,
        expected_model_source_lock_sha256=args.expected_model_source_lock_sha256,
        pricing_lock=args.pricing_lock,
        expected_pricing_lock_sha256=args.expected_pricing_lock_sha256,
        role_selection_file=args.role_selection_file,
        expected_role_selection_file_sha256=(args.expected_role_selection_file_sha256),
        expected_role_selection_sha256=args.expected_role_selection_sha256,
    )


def _binding_file(package, role: str) -> Path:
    binding = next(item for item in package.manifest.input_files if item.role == role)
    return package.root / binding.file


def run_smoke_once(args: argparse.Namespace, root: Path) -> dict[str, object]:
    feedback_runner = _FakeFeedbackRunner()
    fake_codex = _FakeCodexProcess(
        read_stable_regular_file(
            args.fake_creator_response,
            label="deterministic fake Creator response",
            max_bytes=64 * 1024,
        )
    )
    feedback_root = root / "feedback"
    prepared: PreparedPortfolioS1FeedbackRun = prepare_run(
        _feedback_arguments(args, feedback_root)
    )
    bundle = execute_run(prepared, feedback_runner=feedback_runner)
    if bundle is None or feedback_runner.calls != 240:
        raise PortfolioS1ZeroProviderSmokeError(
            "deterministic Feedback substitute did not complete exact240"
        )
    bundle_path = feedback_root / "portfolio-s1-feedback-bundle.json"

    package = prepare_portfolio_s1_creator_inputs(
        static_runtime_root=args.static_runtime_root,
        expected_runtime_lock_file_sha256=args.static_runtime_lock_file_sha256,
        feedback_bundle_path=bundle_path,
        expected_feedback_bundle_file_sha256=_file_sha(bundle_path),
        model_access_evidence_path=args.model_access_evidence,
        expected_model_access_evidence_file_sha256=(
            args.model_access_evidence_file_sha256
        ),
        codex_executable_path=args.codex_executable,
        future_creator_output_dir=root / "creator-output",
        output_dir=root / "creator-inputs",
    )
    creator = run_portfolio_evolution_model(
        stage="s1_creator",
        output_dir=root / "creator-output",
        stage_input_path=_binding_file(package, "feedback_bundle"),
        expected_stage_input_file_sha256=_file_sha(
            _binding_file(package, "feedback_bundle")
        ),
        semantic_authoring_input_path=(package.root / SEMANTIC_INPUT_FILE),
        expected_semantic_authoring_input_file_sha256=_file_sha(
            package.root / SEMANTIC_INPUT_FILE
        ),
        codex_authoring_input_path=package.root / CODEX_INPUT_FILE,
        expected_codex_authoring_input_file_sha256=_file_sha(
            package.root / CODEX_INPUT_FILE
        ),
        parent_bank_path=package.root / PARENT_BANK_FILE,
        expected_parent_bank_file_sha256=_file_sha(package.root / PARENT_BANK_FILE),
        tool_registry_runtime_sha256=(package.manifest.tool_registry_runtime_sha256),
        codex_executable=args.codex_executable,
        process_runner=fake_codex,
    )
    if fake_codex.calls != 1:
        raise PortfolioS1ZeroProviderSmokeError(
            "real S1 runner did not invoke fake Codex exactly once"
        )
    candidate_path = creator.candidate_bank_path
    runtime = create_portfolio_s1_experiment_runtime(
        parent_static_runtime_root=args.static_runtime_root,
        expected_parent_runtime_lock_file_sha256=(args.static_runtime_lock_file_sha256),
        creator_output_root=creator.output_dir,
        expected_creator_invocation_receipt_file_sha256=_file_sha(creator.receipt_path),
        expected_creator_model_invocation_receipt_file_sha256=_file_sha(
            creator.model_invocation_receipt_path
        ),
        expected_candidate_bank_file_sha256=_file_sha(candidate_path),
        output_dir=root / "two-bank-runtime",
    )
    return {
        "selection_sha256": prepared.selection.selection_sha256,
        "control_sha256": prepared.control.control_sha256,
        "feedback_bundle_sha256": bundle.bundle_sha256,
        "feedback_model_projection_sha256": sha256_bytes(
            canonical_json_bytes(bundle.model_projection_payload())
        ),
        "creator_candidate_bank_sha256": creator.candidate_bank.bank_sha256,
        "creator_candidate_bank_file_sha256": _file_sha(candidate_path),
        "two_bank_runtime_lock_sha256": runtime.runtime_lock["runtime_lock_sha256"],
        "two_bank_runtime_lock_file_sha256": runtime.runtime_lock_file_sha256,
        "parent_static_bank_sha256": package.parent_bank.bank_sha256,
        "style_tool_version": "2.3.0",
        "selected_rows": 240,
        "local_feedback_evaluations": feedback_runner.calls,
        "local_fake_codex_invocations": fake_codex.calls,
        "external_feedback_provider_calls": 0,
        "external_codex_provider_calls": 0,
        "external_assistant_provider_calls": 0,
        "replay_status": REPLAY_NOT_RUN,
        "body_gate_status": REPLAY_NOT_RUN,
    }


def _invariant_view(run: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in run.items()
        if key
        not in {
            "local_feedback_evaluations",
            "local_fake_codex_invocations",
            # The runtime deliberately embeds the Creator receipt, whose input
            # paths are absolute create-only paths.  Runtime identities may
            # therefore differ across fresh output roots even though the
            # selection, bundle, and compiled candidate bytes are identical.
            "two_bank_runtime_lock_sha256",
            "two_bank_runtime_lock_file_sha256",
        }
    }


def run_smoke_twice(args: argparse.Namespace) -> dict[str, object]:
    output = args.output_root.absolute()
    if output.exists():
        raise FileExistsError(f"create-only smoke output exists: {output}")
    output.mkdir(parents=True)
    with _NetworkDenied() as network:
        first = run_smoke_once(args, output / "run-1")
        second = run_smoke_once(args, output / "run-2")
    invariant_equal = _invariant_view(first) == _invariant_view(second)
    if not invariant_equal:
        raise PortfolioS1ZeroProviderSmokeError(
            "two fresh smoke runs produced different canonical identities"
        )
    if network.attempts != 0:
        raise PortfolioS1ZeroProviderSmokeError(
            "zero-provider smoke observed a network attempt"
        )
    payload = {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-loop-zero-provider-smoke",
        "policy_version": SMOKE_POLICY_VERSION,
        "status": "passed",
        "run_count": 2,
        "network_denied": True,
        "network_attempt_count": 0,
        "external_provider_call_count": 0,
        "canonical_invariants_equal": True,
        "invariants": _invariant_view(first),
        "path_bound_runtime_identities_equal": (
            first["two_bank_runtime_lock_sha256"]
            == second["two_bank_runtime_lock_sha256"]
            and first["two_bank_runtime_lock_file_sha256"]
            == second["two_bank_runtime_lock_file_sha256"]
        ),
        "path_bound_runtime_identity_note": (
            "Runtime locks include Creator evidence with absolute create-only "
            "input paths; those lock identities may legitimately differ across "
            "fresh roots while candidate Bank bytes remain identical."
        ),
        "per_run": [first, second],
        "paid_evidence_claimed": False,
        "replay_status": REPLAY_NOT_RUN,
        "body_gate_status": REPLAY_NOT_RUN,
    }
    report = {
        **payload,
        "report_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }
    atomic_create_file(output / "smoke-report.json", canonical_json_bytes(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--expected-execution-control-sha256", required=True)
    parser.add_argument("--artifact-repository-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--expected-fold-manifest-sha256", required=True)
    parser.add_argument("--fold-mapping", type=Path, required=True)
    parser.add_argument("--expected-fold-mapping-sha256", required=True)
    parser.add_argument("--rubric-file", type=Path, required=True)
    parser.add_argument("--expected-rubric-sha256", required=True)
    parser.add_argument("--rubric-id", default="portfolio-s1-feedback-v1")
    parser.add_argument("--rubric-version", default="1")
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--owner-statement", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-source-lock", type=Path, required=True)
    parser.add_argument("--expected-model-source-lock-sha256", required=True)
    parser.add_argument("--pricing-lock", type=Path, required=True)
    parser.add_argument("--expected-pricing-lock-sha256", required=True)
    parser.add_argument("--role-selection-file", type=Path, required=True)
    parser.add_argument("--expected-role-selection-file-sha256", required=True)
    parser.add_argument("--expected-role-selection-sha256", required=True)
    parser.add_argument("--static-runtime-root", type=Path, required=True)
    parser.add_argument("--static-runtime-lock-file-sha256", required=True)
    parser.add_argument("--model-access-evidence", type=Path, required=True)
    parser.add_argument("--model-access-evidence-file-sha256", required=True)
    parser.add_argument("--codex-executable", type=Path, required=True)
    parser.add_argument("--fake-creator-response", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        datetime.fromisoformat(args.reviewed_at.replace("Z", "+00:00"))
        report = run_smoke_twice(args)
    except (OSError, TypeError, ValueError) as error:
        print(f"portfolio-s1-zero-provider-smoke: {error}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(report).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
