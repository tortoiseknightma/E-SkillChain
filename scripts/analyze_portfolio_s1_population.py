"""Export zero-model GCS-v2 rows from a completed S1 replay/body population."""

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

from skillchain.evaluation.portfolio_s1_gcs_artifacts import (  # noqa: E402
    export_portfolio_s1_gcs_gate_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--execution-control-file-sha256", required=True)
    parser.add_argument("--artifact-repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--static-execution-root", type=Path)
    parser.add_argument("--static-execution-control-file-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        published = export_portfolio_s1_gcs_gate_inputs(
            arguments.execution_root,
            expected_control_file_sha256=(
                arguments.execution_control_file_sha256
            ),
            artifact_repository_root=arguments.artifact_repository_root,
            output_dir=arguments.output_dir,
            static_execution_root=arguments.static_execution_root,
            static_expected_control_file_sha256=(
                arguments.static_execution_control_file_sha256
            ),
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"analyze-portfolio-s1-population: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "phase": published.phase,
                "query_count": published.query_count,
                "output_dir": published.root.as_posix(),
                "population_binding_file_sha256": (
                    published.population_binding_sha256
                ),
                "evidence_binding_sha256": published.evidence_binding_sha256,
                "baseline_scores_file_sha256": (
                    published.baseline_scores_file_sha256
                ),
                "candidate_scores_file_sha256": (
                    published.candidate_scores_file_sha256
                ),
                "export_manifest_path": (
                    published.root / "export-manifest.json"
                ).as_posix(),
                "export_manifest_file_sha256": (
                    published.export_manifest_file_sha256
                ),
                "provider_model_call_count": 0,
                "pairwise_judge_call_count": 0,
                "legacy_final_judge_call_count": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
