"""Compile the real Portfolio LLMStatic Bank from verified Codex artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from skillchain.evaluation.portfolio_treatments import (
    compile_verified_codex_llm_static_bank,
    load_verified_codex_draft_rebind,
)
from skillchain.synthesis.store import atomic_create_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-authoring-input", type=Path, required=True)
    parser.add_argument(
        "--codex-authoring-input-file-sha256",
        required=True,
    )
    parser.add_argument("--semantic-authoring-input", type=Path, required=True)
    parser.add_argument(
        "--semantic-authoring-input-file-sha256",
        required=True,
    )
    parser.add_argument("--pre-review-draft", type=Path, required=True)
    parser.add_argument("--pre-review-draft-file-sha256", required=True)
    parser.add_argument("--tool-registry-runtime-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.output.exists():
        print(
            "compile-portfolio-llm-static-bank: output already exists",
            file=sys.stderr,
        )
        return 2
    try:
        verified = load_verified_codex_draft_rebind(
            codex_input_path=arguments.codex_authoring_input,
            expected_codex_input_file_sha256=(
                arguments.codex_authoring_input_file_sha256
            ),
            semantic_input_path=arguments.semantic_authoring_input,
            expected_semantic_input_file_sha256=(
                arguments.semantic_authoring_input_file_sha256
            ),
            draft_path=arguments.pre_review_draft,
            expected_draft_file_sha256=(
                arguments.pre_review_draft_file_sha256
            ),
        )
        bank = compile_verified_codex_llm_static_bank(
            verified,
            tool_registry_runtime_sha256=(
                arguments.tool_registry_runtime_sha256
            ),
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_create_file(arguments.output, bank.canonical_bytes())
    except Exception as error:
        print(f"compile-portfolio-llm-static-bank: {error}", file=sys.stderr)
        return 2
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
