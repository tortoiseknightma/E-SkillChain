"""Run the offline Core S1 replay/body GCS gate without provider calls."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import TypeVar


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pydantic import BaseModel, ValidationError  # noqa: E402

from skillchain.evaluation.portfolio_gcs import (  # noqa: E402
    GCSQueryScoreV2,
    build_gcs_population_v2,
)
from skillchain.evaluation.portfolio_s1_gcs_artifacts import (  # noqa: E402
    PortfolioS1GCSArtifactError,
    load_verified_portfolio_s1_gcs_gate_export,
)
from skillchain.evaluation.portfolio_s1_experiment_runtime import (  # noqa: E402
    load_verified_portfolio_s1_experiment_runtime,
)
from skillchain.evolution.s1_gcs_gate import (  # noqa: E402
    S1BankDispositionPublication,
    S1DevelopmentCompositeReport,
    S1GCSGateError,
    S1GCSGatePhase,
    S1GCSGateReport,
    S1GCSEvidenceBinding,
    S1Round2CandidateFreeze,
    S1Round2DevelopmentScreen,
    S1Round2ResponseContractDiagnostics,
    build_s1_bank_disposition,
    build_s1_round2_bank_disposition,
    evaluate_s1_body_gate,
    evaluate_s1_development_composite,
    evaluate_s1_replay,
    evaluate_s1_round2_body_gate,
    make_s1_round2_candidate_freeze,
    screen_s1_round2_development_patches,
)
from skillchain.evolution.s1_sparse_patch import (  # noqa: E402
    SparseCompilationReceiptV1,
    SparseScreenedBankReceiptV1,
    compose_screened_sparse_bank,
)
from skillchain.schemas import Query  # noqa: E402
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.synthesis.store import (  # noqa: E402
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.serialization import (  # noqa: E402
    ArtifactFormatError,
    canonical_jsonl_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)


_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSONL_BYTES = 256 * 1024 * 1024
ModelT = TypeVar("ModelT", bound=BaseModel)


def _read(path: Path, *, label: str, max_bytes: int) -> bytes:
    return read_stable_regular_file(path, label=label, max_bytes=max_bytes)


def _load_model(
    path: Path, model_type: type[ModelT], *, label: str
) -> tuple[bytes, ModelT]:
    content = _read(path, label=label, max_bytes=_MAX_JSON_BYTES)
    return _load_model_bytes(content, model_type, label=label)


def _load_model_with_sha(
    path: Path,
    model_type: type[ModelT],
    *,
    label: str,
    expected_file_sha256: str,
) -> tuple[bytes, ModelT]:
    content = _read(path, label=label, max_bytes=_MAX_JSON_BYTES)
    if sha256_bytes(content) != expected_file_sha256:
        raise S1GCSGateError(f"{label} file SHA-256 drifted")
    return _load_model_bytes(content, model_type, label=label)


def _load_model_bytes(
    content: bytes, model_type: type[ModelT], *, label: str
) -> tuple[bytes, ModelT]:
    try:
        model = model_type.model_validate_json(content, strict=True)
    except ValidationError as error:
        raise S1GCSGateError(f"{label} is invalid") from error
    canonical = model.canonical_bytes() if hasattr(model, "canonical_bytes") else None
    if canonical is not None and canonical != content:
        raise S1GCSGateError(f"{label} is not canonical")
    return content, model


def _load_queries(path: Path) -> tuple[bytes, tuple[Query, ...]]:
    content = _read(path, label="S1 gate queries", max_bytes=_MAX_JSONL_BYTES)
    return _load_query_bytes(content)


def _load_query_bytes(content: bytes) -> tuple[bytes, tuple[Query, ...]]:
    try:
        rows = parse_canonical_jsonl(content, label="S1 gate queries")
        queries = tuple(Query.model_validate(item, strict=True) for item in rows)
    except (ArtifactFormatError, ValidationError) as error:
        raise S1GCSGateError("S1 gate query artifact is invalid") from error
    canonical = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in queries)
    )
    if canonical != content:
        raise S1GCSGateError("S1 gate query artifact is not canonical")
    return content, queries


def _load_scores(
    path: Path, *, label: str
) -> tuple[bytes, tuple[GCSQueryScoreV2, ...]]:
    content = _read(path, label=label, max_bytes=_MAX_JSONL_BYTES)
    return _load_score_bytes(content, label=label)


def _load_score_bytes(
    content: bytes, *, label: str
) -> tuple[bytes, tuple[GCSQueryScoreV2, ...]]:
    try:
        rows = parse_canonical_jsonl(content, label=label)
        scores = tuple(
            GCSQueryScoreV2.model_validate(item, strict=True) for item in rows
        )
    except (ArtifactFormatError, ValidationError) as error:
        raise S1GCSGateError(f"{label} is invalid") from error
    canonical = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in scores)
    )
    if canonical != content:
        raise S1GCSGateError(f"{label} is not canonical")
    return content, scores


def _load_bank(path: Path, *, label: str) -> tuple[bytes, StaticBankArtifact]:
    return _load_model(path, StaticBankArtifact, label=label)


def _gate_inputs(arguments: argparse.Namespace, phase: S1GCSGatePhase):
    try:
        export = load_verified_portfolio_s1_gcs_gate_export(
            arguments.export_root,
            expected_manifest_file_sha256=arguments.export_manifest_file_sha256,
            expected_phase=phase,
        )
    except PortfolioS1GCSArtifactError as error:
        raise S1GCSGateError("S1 gate export verification failed") from error
    queries_bytes, queries = _load_query_bytes(export.content("queries.jsonl"))
    baseline_bytes, baseline = _load_score_bytes(
        export.content("baseline-scores.jsonl"),
        label=f"{phase} baseline GCS rows",
    )
    candidate_bytes, candidate = _load_score_bytes(
        export.content("candidate-scores.jsonl"),
        label=f"{phase} candidate GCS rows",
    )
    parent_bytes, parent = _load_model_bytes(
        export.content("parent-bank.json"), StaticBankArtifact, label="S1 parent Bank"
    )
    candidate_bank_bytes, candidate_bank = _load_model_bytes(
        export.content("candidate-bank.json"),
        StaticBankArtifact,
        label="S1 candidate Bank",
    )
    _evidence_bytes, evidence = _load_model_bytes(
        export.content("evidence-binding.json"),
        S1GCSEvidenceBinding,
        label=f"{phase} evidence binding",
    )
    try:
        population_binding = parse_canonical_json(
            export.content("population-binding.json"),
            label=f"{phase} population binding",
        )
    except ArtifactFormatError as error:
        raise S1GCSGateError(f"{phase} population binding is invalid") from error
    population = build_gcs_population_v2(queries)
    if (
        not isinstance(population_binding, dict)
        or population_binding.get("query_ids") != [query.query_id for query in queries]
        or population_binding.get("population") != population.model_dump(mode="json")
        or population_binding.get("parent_bank_sha256") != parent.bank_sha256
        or population_binding.get("candidate_bank_sha256") != candidate_bank.bank_sha256
        or export.manifest.query_count != len(queries)
    ):
        raise S1GCSGateError(
            f"{phase} population binding does not close over queries and Banks"
        )
    raw_diagnostics = population_binding.get(
        "round2_response_contract_diagnostics"
    )
    diagnostics = None
    if raw_diagnostics is not None:
        try:
            diagnostics = S1Round2ResponseContractDiagnostics.model_validate(
                raw_diagnostics, strict=True
            )
        except ValidationError as error:
            raise S1GCSGateError(
                f"{phase} response contract diagnostics are invalid"
            ) from error
        if diagnostics.query_ids != tuple(item.query_id for item in queries):
            raise S1GCSGateError(
                f"{phase} response contract diagnostics differ from queries"
            )
    return (
        queries,
        baseline,
        candidate,
        parent_bytes,
        parent,
        candidate_bank_bytes,
        candidate_bank,
        evidence,
        population_binding,
        diagnostics,
    )


def _publish(output_dir: Path, files: dict[str, bytes]) -> None:
    staging = new_staging_directory(output_dir)
    try:
        for name, content in files.items():
            atomic_create_file(staging / name, content)
        atomic_publish_new_directory(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _publication_files(
    publication: S1BankDispositionPublication,
    *,
    reports: dict[str, bytes],
) -> dict[str, bytes]:
    return {
        **reports,
        "output-bank.json": publication.output_bank_bytes,
        "disposition-receipt.json": publication.receipt.canonical_bytes(),
    }


def _run_replay(arguments: argparse.Namespace) -> None:
    (
        queries,
        baseline,
        candidate,
        _parent_bytes,
        parent,
        _candidate_bank_bytes,
        candidate_bank,
        evidence,
        _population_binding,
        _diagnostics,
    ) = _gate_inputs(arguments, "replay")
    report = evaluate_s1_replay(
        queries=queries,
        baseline_scores=baseline,
        candidate_scores=candidate,
        evidence=evidence,
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate_bank.bank_sha256,
    )
    _publish(
        arguments.output_dir,
        {
            "replay-evidence-binding.json": evidence.canonical_bytes(),
            "replay-report.json": report.canonical_bytes(),
        },
    )


def _run_body_gate(arguments: argparse.Namespace) -> None:
    replay_bytes, replay = _load_model(
        arguments.replay_report, S1GCSGateReport, label="S1 replay report"
    )
    (
        queries,
        baseline,
        candidate,
        parent_bytes,
        parent,
        candidate_bank_bytes,
        candidate_bank,
        evidence,
        _population_binding,
        _diagnostics,
    ) = _gate_inputs(arguments, "body_gate")
    body = evaluate_s1_body_gate(
        replay_report=replay,
        queries=queries,
        baseline_scores=baseline,
        candidate_scores=candidate,
        evidence=evidence,
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate_bank.bank_sha256,
    )
    publication = build_s1_bank_disposition(
        replay_report=replay,
        body_gate_report=body,
        parent_bank=parent,
        parent_bank_bytes=parent_bytes,
        candidate_bank=candidate_bank,
        candidate_bank_bytes=candidate_bank_bytes,
    )
    _publish(
        arguments.output_dir,
        _publication_files(
            publication,
            reports={
                "replay-report.json": replay_bytes,
                "body-gate-evidence-binding.json": evidence.canonical_bytes(),
                "body-gate-report.json": body.canonical_bytes(),
            },
        ),
    )


def _run_finalize_replay(arguments: argparse.Namespace) -> None:
    replay_bytes, replay = _load_model(
        arguments.replay_report, S1GCSGateReport, label="S1 replay report"
    )
    parent_bytes, parent = _load_bank(arguments.parent_bank, label="S1 parent Bank")
    candidate_bytes, candidate = _load_bank(
        arguments.candidate_bank, label="S1 candidate Bank"
    )
    publication = build_s1_bank_disposition(
        replay_report=replay,
        body_gate_report=None,
        parent_bank=parent,
        parent_bank_bytes=parent_bytes,
        candidate_bank=candidate,
        candidate_bank_bytes=candidate_bytes,
    )
    _publish(
        arguments.output_dir,
        _publication_files(
            publication,
            reports={"replay-report.json": replay_bytes},
        ),
    )


def _run_round2_screen(arguments: argparse.Namespace) -> None:
    """Screen Creator patches on development replay and compose parent bytes."""

    (
        queries,
        baseline,
        candidate,
        _parent_bytes,
        parent,
        _candidate_bank_bytes,
        candidate_bank,
        evidence,
        _population_binding,
        diagnostics,
    ) = _gate_inputs(arguments, "replay")
    if diagnostics is None:
        raise S1GCSGateError(
            "Round 2 screen requires paired checkpoint contract diagnostics"
        )
    _receipt_bytes, receipt = _load_model_with_sha(
        arguments.sparse_compilation_receipt,
        SparseCompilationReceiptV1,
        label="S1 sparse compilation receipt",
        expected_file_sha256=(arguments.sparse_compilation_receipt_file_sha256),
    )
    screen = screen_s1_round2_development_patches(
        queries=queries,
        baseline_scores=baseline,
        raw_candidate_scores=candidate,
        response_contract_diagnostics=diagnostics,
        source_evidence=evidence,
        parent_bank_sha256=parent.bank_sha256,
        raw_candidate_bank_sha256=candidate_bank.bank_sha256,
    )
    screen_bytes = screen.canonical_bytes()
    patch_actions = {item.capability_id: item.action for item in receipt.bindings}
    retained = tuple(
        item.capability_id
        for item in screen.decisions
        if item.decision == "retain_patch"
        and patch_actions.get(item.capability_id) == "patch"
    )
    screened = compose_screened_sparse_bank(
        parent_bank=parent,
        creator_candidate_bank=candidate_bank,
        creator_compilation_receipt=receipt,
        development_screen_sha256=sha256_bytes(screen_bytes),
        retained_capability_ids=retained,
    )
    _publish(
        arguments.output_dir,
        {
            "development-screen.json": screen_bytes,
            "screened-bank.json": screened.bank.canonical_bytes(),
            "screened-bank-receipt.json": screened.receipt.canonical_bytes(),
        },
    )


def _verified_round2_runtime(
    arguments: argparse.Namespace,
    *,
    parent_bytes: bytes,
    parent: StaticBankArtifact,
    candidate_bytes: bytes,
    candidate: StaticBankArtifact,
    population_binding: dict[str, object],
):
    runtime = load_verified_portfolio_s1_experiment_runtime(
        arguments.runtime_root,
        expected_runtime_lock_file_sha256=arguments.runtime_lock_file_sha256,
    )
    if (
        runtime.banks.get("llm_static") != parent
        or runtime.banks.get("s1") != candidate
        or runtime.bank_file_sha256s.get("llm_static") != sha256_bytes(parent_bytes)
        or runtime.bank_file_sha256s.get("s1") != sha256_bytes(candidate_bytes)
    ):
        raise S1GCSGateError("Round 2 export Banks differ from the execution runtime")
    runtime_self = runtime.runtime_lock.get("runtime_lock_sha256")
    if not isinstance(runtime_self, str):
        raise S1GCSGateError("Round 2 runtime lacks its self hash")
    if (
        population_binding.get("runtime_lock_sha256") != runtime_self
        or population_binding.get("runtime_lock_file_sha256")
        != runtime.runtime_lock_file_sha256
    ):
        raise S1GCSGateError(
            "Round 2 scored population differs from the execution runtime"
        )
    return runtime, runtime_self


def _validate_screened_lineage(
    *,
    screen_bytes: bytes,
    screen: S1Round2DevelopmentScreen,
    receipt_bytes: bytes,
    receipt: SparseScreenedBankReceiptV1,
    parent: StaticBankArtifact,
    candidate: StaticBankArtifact,
) -> None:
    decisions = {item.capability_id: item.decision for item in screen.decisions}
    expected_retained = tuple(
        item.capability_id
        for item in receipt.bindings
        if item.creator_candidate_skill_sha256 != item.parent_skill_sha256
        and decisions.get(item.capability_id) == "retain_patch"
    )
    if (
        screen.parent_bank_sha256 != parent.bank_sha256
        or receipt.parent_bank_sha256 != parent.bank_sha256
        or receipt.creator_candidate_bank_sha256 != screen.raw_candidate_bank_sha256
        or receipt.development_screen_sha256 != sha256_bytes(screen_bytes)
        or receipt.screened_bank_sha256 != candidate.bank_sha256
        or receipt.retained_capability_ids != expected_retained
        or sha256_bytes(receipt_bytes) != sha256_bytes(receipt.canonical_bytes())
    ):
        raise S1GCSGateError(
            "Round 2 screen, retained set, or screened receipt lineage drifted"
        )


def _run_round2_freeze(arguments: argparse.Namespace) -> None:
    """Evaluate the composed development replay and freeze its sole candidate."""

    (
        queries,
        baseline,
        candidate_scores,
        parent_bytes,
        parent,
        candidate_bytes,
        candidate,
        evidence,
        population_binding,
        diagnostics,
    ) = _gate_inputs(arguments, "replay")
    if diagnostics is None:
        raise S1GCSGateError(
            "Round 2 freeze requires paired checkpoint contract diagnostics"
        )
    screen_bytes, screen = _load_model_with_sha(
        arguments.development_screen,
        S1Round2DevelopmentScreen,
        label="S1 Round 2 development screen",
        expected_file_sha256=arguments.development_screen_file_sha256,
    )
    screened_receipt_bytes, screened_receipt = _load_model_with_sha(
        arguments.screened_bank_receipt,
        SparseScreenedBankReceiptV1,
        label="S1 Round 2 screened Bank receipt",
        expected_file_sha256=arguments.screened_bank_receipt_file_sha256,
    )
    _validate_screened_lineage(
        screen_bytes=screen_bytes,
        screen=screen,
        receipt_bytes=screened_receipt_bytes,
        receipt=screened_receipt,
        parent=parent,
        candidate=candidate,
    )
    report = evaluate_s1_development_composite(
        queries=queries,
        baseline_scores=baseline,
        candidate_scores=candidate_scores,
        response_contract_diagnostics=diagnostics,
    )
    if evidence.queries_file_sha256 != screen.source_evidence.queries_file_sha256:
        raise S1GCSGateError(
            "Round 2 composite query file differs from the raw patch screen"
        )
    runtime, runtime_self = _verified_round2_runtime(
        arguments,
        parent_bytes=parent_bytes,
        parent=parent,
        candidate_bytes=candidate_bytes,
        candidate=candidate,
        population_binding=population_binding,
    )
    freeze = make_s1_round2_candidate_freeze(
        development_report=report,
        development_screen=screen,
        development_screen_file_sha256=sha256_bytes(screen_bytes),
        screened_bank_receipt_sha256=screened_receipt.receipt_sha256,
        screened_bank_receipt_file_sha256=sha256_bytes(screened_receipt_bytes),
        retained_capability_ids=screened_receipt.retained_capability_ids,
        parent_bank_sha256=parent.bank_sha256,
        parent_bank_file_sha256=sha256_bytes(parent_bytes),
        candidate_bank_sha256=candidate.bank_sha256,
        candidate_bank_file_sha256=sha256_bytes(candidate_bytes),
        runtime_lock_sha256=runtime_self,
        runtime_lock_file_sha256=runtime.runtime_lock_file_sha256,
    )
    _publish(
        arguments.output_dir,
        {
            "development-screen.json": screen_bytes,
            "screened-bank-receipt.json": screened_receipt_bytes,
            "development-evidence-binding.json": evidence.canonical_bytes(),
            "development-report.json": report.canonical_bytes(),
            "candidate-freeze.json": freeze.canonical_bytes(),
        },
    )


def _run_round2_body_gate(arguments: argparse.Namespace) -> None:
    """Apply the one-shot fresh body gate to the pre-frozen Round 2 candidate."""

    development_bytes, development = _load_model_with_sha(
        arguments.development_report,
        S1DevelopmentCompositeReport,
        label="S1 Round 2 development report",
        expected_file_sha256=arguments.development_report_file_sha256,
    )
    freeze_bytes, freeze = _load_model_with_sha(
        arguments.candidate_freeze,
        S1Round2CandidateFreeze,
        label="S1 Round 2 candidate freeze",
        expected_file_sha256=arguments.candidate_freeze_file_sha256,
    )
    (
        queries,
        baseline,
        candidate_scores,
        parent_bytes,
        parent,
        candidate_bytes,
        candidate,
        evidence,
        population_binding,
        diagnostics,
    ) = _gate_inputs(arguments, "body_gate")
    if diagnostics is None:
        raise S1GCSGateError(
            "Round 2 body gate requires paired checkpoint contract diagnostics"
        )
    runtime, runtime_self = _verified_round2_runtime(
        arguments,
        parent_bytes=parent_bytes,
        parent=parent,
        candidate_bytes=candidate_bytes,
        candidate=candidate,
        population_binding=population_binding,
    )
    report = evaluate_s1_round2_body_gate(
        development_report=development,
        candidate_freeze=freeze,
        queries=queries,
        baseline_scores=baseline,
        candidate_scores=candidate_scores,
        response_contract_diagnostics=diagnostics,
        evidence=evidence,
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate.bank_sha256,
        runtime_lock_sha256=runtime_self,
        runtime_lock_file_sha256=runtime.runtime_lock_file_sha256,
    )
    publication = build_s1_round2_bank_disposition(
        body_gate_report=report,
        parent_bank=parent,
        parent_bank_bytes=parent_bytes,
        candidate_bank=candidate,
        candidate_bank_bytes=candidate_bytes,
    )
    _publish(
        arguments.output_dir,
        {
            "development-report.json": development_bytes,
            "candidate-freeze.json": freeze_bytes,
            "body-gate-evidence-binding.json": evidence.canonical_bytes(),
            "body-gate-report.json": report.canonical_bytes(),
            "output-bank.json": publication.output_bank_bytes,
            "disposition-receipt.json": publication.receipt.canonical_bytes(),
        },
    )


def _add_gate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--export-manifest-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay = subparsers.add_parser("replay", help="evaluate the opt replay200 screen")
    _add_gate_arguments(replay)
    replay.set_defaults(handler=_run_replay)

    body = subparsers.add_parser(
        "body-gate", help="evaluate body_gate75 and publish accept/rollback"
    )
    _add_gate_arguments(body)
    body.add_argument("--replay-report", type=Path, required=True)
    body.set_defaults(handler=_run_body_gate)

    finalize = subparsers.add_parser(
        "finalize-replay",
        help="publish a byte-exact rollback after a failed replay screen",
    )
    finalize.add_argument("--replay-report", type=Path, required=True)
    finalize.add_argument("--parent-bank", type=Path, required=True)
    finalize.add_argument("--candidate-bank", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.set_defaults(handler=_run_finalize_replay)

    round2_screen = subparsers.add_parser(
        "round2-screen",
        help=(
            "screen sparse capability patches on development replay and "
            "compose the parent-protected candidate"
        ),
    )
    _add_gate_arguments(round2_screen)
    round2_screen.add_argument("--sparse-compilation-receipt", type=Path, required=True)
    round2_screen.add_argument(
        "--sparse-compilation-receipt-file-sha256", required=True
    )
    round2_screen.set_defaults(handler=_run_round2_screen)

    round2_freeze = subparsers.add_parser(
        "round2-freeze",
        help="evaluate the composed development replay and freeze one candidate",
    )
    _add_gate_arguments(round2_freeze)
    round2_freeze.add_argument("--development-screen", type=Path, required=True)
    round2_freeze.add_argument("--development-screen-file-sha256", required=True)
    round2_freeze.add_argument("--screened-bank-receipt", type=Path, required=True)
    round2_freeze.add_argument("--screened-bank-receipt-file-sha256", required=True)
    round2_freeze.add_argument("--runtime-root", type=Path, required=True)
    round2_freeze.add_argument("--runtime-lock-file-sha256", required=True)
    round2_freeze.set_defaults(handler=_run_round2_freeze)

    round2_body = subparsers.add_parser(
        "round2-body-gate",
        help=(
            "evaluate the fresh body_gate75 for the frozen Round 2 candidate "
            "and publish accept/rollback"
        ),
    )
    _add_gate_arguments(round2_body)
    round2_body.add_argument("--development-report", type=Path, required=True)
    round2_body.add_argument("--development-report-file-sha256", required=True)
    round2_body.add_argument("--candidate-freeze", type=Path, required=True)
    round2_body.add_argument("--candidate-freeze-file-sha256", required=True)
    round2_body.add_argument("--runtime-root", type=Path, required=True)
    round2_body.add_argument("--runtime-lock-file-sha256", required=True)
    round2_body.set_defaults(handler=_run_round2_body_gate)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
