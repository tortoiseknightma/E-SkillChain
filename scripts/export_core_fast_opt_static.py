#!/usr/bin/env python3
"""Export the verified Static opt800 corpus as Core Fast observations.

This is an offline, create-only format adapter.  It delegates all historical
checkpoint, launch, runtime, and GCS verification to
``load_verified_static_gcs_corpus`` and performs no provider calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from skillchain.evaluation.core_fast.models import (  # noqa: E402
    AssistantObservation,
    ToolTraceItem,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import (  # noqa: E402
    STATIC_GCS_QUERY_COUNT,
    StaticGCSCorpusError,
    VerifiedStaticGCSCorpus,
    VerifiedStaticGCSRow,
    load_verified_static_gcs_corpus,
)
from skillchain.synthesis.store import atomic_create_file  # noqa: E402
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


class CoreFastOptStaticExportError(RuntimeError):
    """The verified corpus cannot be represented as one exact opt800 file."""


def observation_from_verified_row(row: VerifiedStaticGCSRow) -> AssistantObservation:
    """Map one deeply verified historical row without inventing evidence."""

    response = row.response
    score = row.score
    full_trace = tuple(item.model_dump(mode="json") for item in response.tool_trace)
    return AssistantObservation(
        query_id=row.query.query_id,
        response_text=response.response_text,
        selected_capability=response.selected_capability,
        route_trace_key=response.route_trace_sha256,
        tool_trace_key=sha256_bytes(canonical_json_bytes(list(full_trace))),
        tool_trace=tuple(
            ToolTraceItem(
                tool_name=item.tool_name,
                # Historical checkpoints intentionally retain the argument hash,
                # not private/raw tool arguments.  Preserve that fact explicitly.
                arguments={"arguments_sha256": item.arguments_sha256},
                status=item.status,
                result_sha256=item.result_sha256,
                error_code=item.error_code,
            )
            for item in response.tool_trace
        ),
        replay_context={
            "response": response.model_dump(mode="json"),
            "receipt": row.receipt.model_dump(mode="json"),
            "scorer_calls": [
                item.model_dump(mode="json") for item in row.sidecar.calls
            ],
            "scorer_capture_policy_version": row.sidecar.policy_version,
            "assistant_result": row.result.model_dump(mode="json"),
        },
        answer_mode=score.answer_mode,
        oracle_available=bool(score.oracle_available),
        gcs_components={
            "route_acceptable": bool(score.route_acceptable),
            "no_hard_error": bool(score.no_hard_error),
            "tool_contract_pass": bool(score.tool_contract_pass),
            "evidence_grounded": bool(score.evidence_grounded),
            "output_contract_pass": bool(score.output_contract_pass),
        },
        gcs_score=float(score.gcs),
        gcs_reason_codes=score.reason_codes,
        hard_error=bool(score.hard_error),
        card_violation=(
            row.query.requires_card and not bool(score.output_contract_pass)
        ),
        evidence_violation=not bool(score.evidence_grounded),
        tool_violation=not bool(score.tool_contract_pass),
        source=row.strata.source_dataset,
        repair=row.strata.repair_status,
        style_submode=row.strata.style_submode,
    )


def build_observations(
    corpus: VerifiedStaticGCSCorpus,
) -> tuple[AssistantObservation, ...]:
    """Build a deterministic, ordinal-ordered, exact opt800 projection."""

    rows = tuple(sorted(corpus.rows, key=lambda row: row.query_ordinal))
    ordinals = tuple(row.query_ordinal for row in rows)
    if ordinals != tuple(range(STATIC_GCS_QUERY_COUNT)):
        raise CoreFastOptStaticExportError(
            "verified Static rows must have contiguous query ordinals 0..799"
        )
    expected_ids = {
        query.query_id
        for query in corpus.core_inputs.queries
        if query.split == "opt_pool"
    }
    observations = tuple(observation_from_verified_row(row) for row in rows)
    actual_ids = tuple(item.query_id for item in observations)
    if (
        len(observations) != STATIC_GCS_QUERY_COUNT
        or len(set(actual_ids)) != STATIC_GCS_QUERY_COUNT
        or set(actual_ids) != expected_ids
        or len(expected_ids) != STATIC_GCS_QUERY_COUNT
    ):
        raise CoreFastOptStaticExportError(
            "Core Fast observations must cover the verified opt800 exactly"
        )
    return observations


def export(arguments: argparse.Namespace) -> dict[str, object]:
    corpus = load_verified_static_gcs_corpus(
        arguments.execution_root,
        expected_control_file_sha256=(arguments.expected_execution_control_file_sha256),
        artifact_repository_root=arguments.artifact_repository_root,
    )
    observations = build_observations(corpus)
    payloads = tuple(item.model_dump(mode="json") for item in observations)
    content = canonical_jsonl_bytes(payloads)
    output = arguments.output.absolute()
    atomic_create_file(output, content)
    if output.read_bytes() != content:
        raise CoreFastOptStaticExportError("created output differs from projection")
    return {
        "kind": "core-fast-opt-static-export",
        "schema_version": 1,
        "provider_calls_performed": 0,
        "source_corpus_sha256": corpus.corpus_sha256,
        "source_execution_control_file_sha256": corpus.control_file_sha256,
        "row_count": len(observations),
        "gcs_success_count": sum(int(item.gcs_score) for item in observations),
        "hard_error_count": sum(int(item.hard_error) for item in observations),
        "output_path": str(output),
        "output_bytes": len(content),
        "output_sha256": sha256_bytes(content),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--artifact-repository-root", type=Path, required=True)
    parser.add_argument(
        "--expected-execution-control-file-sha256",
        required=True,
        help="expected byte SHA-256 of execution-control.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new create-only Core Fast AssistantObservation JSONL",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = export(arguments)
    except (
        CoreFastOptStaticExportError,
        StaticGCSCorpusError,
        OSError,
        ValueError,
    ) as error:
        print(f"Core Fast opt Static export error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
