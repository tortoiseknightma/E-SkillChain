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
from skillchain.evolution.s1_gcs_gate import (  # noqa: E402
    S1BankDispositionPublication,
    S1GCSGateError,
    S1GCSGatePhase,
    S1GCSGateReport,
    S1GCSEvidenceBinding,
    build_s1_bank_disposition,
    evaluate_s1_body_gate,
    evaluate_s1_replay,
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
)


_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSONL_BYTES = 256 * 1024 * 1024
ModelT = TypeVar("ModelT", bound=BaseModel)


def _read(path: Path, *, label: str, max_bytes: int) -> bytes:
    return read_stable_regular_file(path, label=label, max_bytes=max_bytes)


def _load_model(path: Path, model_type: type[ModelT], *, label: str) -> tuple[bytes, ModelT]:
    content = _read(path, label=label, max_bytes=_MAX_JSON_BYTES)
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


def _load_scores(path: Path, *, label: str) -> tuple[bytes, tuple[GCSQueryScoreV2, ...]]:
    content = _read(path, label=label, max_bytes=_MAX_JSONL_BYTES)
    return _load_score_bytes(content, label=label)


def _load_score_bytes(
    content: bytes, *, label: str
) -> tuple[bytes, tuple[GCSQueryScoreV2, ...]]:
    try:
        rows = parse_canonical_jsonl(content, label=label)
        scores = tuple(GCSQueryScoreV2.model_validate(item, strict=True) for item in rows)
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
        or population_binding.get("query_ids")
        != [query.query_id for query in queries]
        or population_binding.get("population") != population.model_dump(mode="json")
        or population_binding.get("parent_bank_sha256") != parent.bank_sha256
        or population_binding.get("candidate_bank_sha256")
        != candidate_bank.bank_sha256
        or export.manifest.query_count != len(queries)
    ):
        raise S1GCSGateError(
            f"{phase} population binding does not close over queries and Banks"
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
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
