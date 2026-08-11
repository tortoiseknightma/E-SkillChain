"""Create the one Kimi v3 partial11 Portfolio S1 Feedback bundle.

This command performs no model or provider calls.  It loads every input by an
operator-supplied file SHA, deeply reloads and re-scores the frozen Static
opt800 corpus, validates the exact terminal Kimi 12/11/1 run, and writes one
canonical bundle with create-only semantics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from skillchain.evaluation.portfolio_s1_feedback import (
    BoundFeedbackArtifact,
    PortfolioS1FeedbackError,
    build_portfolio_s1_feedback_bundle_v3,
    load_bound_feedback_artifact,
    load_portfolio_s1_feedback_control,
    load_portfolio_s1_feedback_run,
    load_portfolio_s1_feedback_selection,
    load_selected_kimi_feedback_authorization,
    write_portfolio_s1_feedback_bundle_v3,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import (
    StaticGCSCorpusError,
    load_verified_static_gcs_corpus,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


class PortfolioS1FeedbackBundleV3MaterializationError(RuntimeError):
    """The exact Kimi v3 bundle inputs or create-only output are invalid."""


def _load_terminal_artifacts(
    directory: Path,
    *,
    expected_count: int,
) -> tuple[BoundFeedbackArtifact, ...]:
    root = directory.absolute().resolve(strict=True)
    if not root.is_dir():
        raise PortfolioS1FeedbackBundleV3MaterializationError(
            "bound Feedback path is not a directory"
        )
    children = tuple(sorted(root.iterdir(), key=lambda item: item.name))
    if (
        len(children) != expected_count
        or any(not item.is_file() or item.suffix != ".json" for item in children)
    ):
        raise PortfolioS1FeedbackBundleV3MaterializationError(
            f"bound Feedback directory must contain exactly {expected_count} JSON files"
        )
    artifacts = tuple(load_bound_feedback_artifact(item) for item in children)
    if len({item.selection_entry_sha256 for item in artifacts}) != expected_count:
        raise PortfolioS1FeedbackBundleV3MaterializationError(
            "bound Feedback directory repeats a selection identity"
        )
    return artifacts


def materialize(arguments: argparse.Namespace) -> dict[str, object]:
    selection = load_portfolio_s1_feedback_selection(
        arguments.selection,
        expected_file_sha256=arguments.expected_selection_file_sha256,
    )
    authorization = load_selected_kimi_feedback_authorization(
        arguments.authorization,
        expected_file_sha256=arguments.expected_authorization_file_sha256,
    )
    control = load_portfolio_s1_feedback_control(
        arguments.control,
        expected_file_sha256=arguments.expected_control_file_sha256,
    )
    run = load_portfolio_s1_feedback_run(
        arguments.run,
        expected_file_sha256=arguments.expected_run_file_sha256,
    )
    if (
        run.completed_count != 12
        or run.parsed_count != 11
        or run.error_count != 1
        or run.provider_calls != 12
        or run.status != "stopped_nonparsed"
    ):
        raise PortfolioS1FeedbackBundleV3MaterializationError(
            "run is not the terminal Kimi 12/11/1 identity"
        )
    artifacts = _load_terminal_artifacts(
        arguments.bound_feedback_dir,
        expected_count=run.completed_count,
    )
    corpus = load_verified_static_gcs_corpus(
        arguments.execution_root,
        expected_control_file_sha256=(
            arguments.expected_execution_control_file_sha256
        ),
        artifact_repository_root=arguments.artifact_repository_root,
    )
    bundle = build_portfolio_s1_feedback_bundle_v3(
        selection,
        control,
        authorization,
        artifacts,
        run,
        corpus=corpus,
    )
    output = arguments.output.absolute()
    if not output.parent.is_dir():
        raise PortfolioS1FeedbackBundleV3MaterializationError(
            "bundle output parent directory does not exist"
        )
    write_portfolio_s1_feedback_bundle_v3(output, bundle)
    content = output.read_bytes()
    if content != bundle.canonical_bytes():
        raise PortfolioS1FeedbackBundleV3MaterializationError(
            "created bundle differs from the verified in-memory bundle"
        )
    return {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-bundle-v3-materialization-summary",
        "provider_calls_performed": 0,
        "selection_file_sha256": arguments.expected_selection_file_sha256,
        "authorization_file_sha256": arguments.expected_authorization_file_sha256,
        "control_file_sha256": arguments.expected_control_file_sha256,
        "run_file_sha256": arguments.expected_run_file_sha256,
        "corpus_sha256": corpus.corpus_sha256,
        "selected_count": bundle.selected_count,
        "attempted_count": bundle.attempted_count,
        "parsed_count": bundle.parsed_count,
        "parse_error_count": bundle.parse_error_count,
        "unattempted_count": bundle.unattempted_count,
        "missing_feedback_count": bundle.missing_feedback_count,
        "bundle_file_sha256": sha256_bytes(content),
        "bundle_sha256": bundle.bundle_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument(
        "--expected-execution-control-file-sha256", required=True
    )
    parser.add_argument("--artifact-repository-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-file-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-file-sha256", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--expected-control-file-sha256", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-run-file-sha256", required=True)
    parser.add_argument("--bound-feedback-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = materialize(arguments)
        print(canonical_json_bytes(summary).decode("utf-8"), end="")
        return 0
    except (
        FileExistsError,
        OSError,
        PortfolioS1FeedbackBundleV3MaterializationError,
        PortfolioS1FeedbackError,
        StaticGCSCorpusError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
