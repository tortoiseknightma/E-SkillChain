"""DIAGNOSTIC ONLY: score raw offline outputs without a formal trust claim."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any, Literal

from skillchain.tools.benchmark import (
    BenchmarkBindings,
    BenchmarkError,
    load_diagnostic_benchmark_gold,
    run_offline_benchmark,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "This command always writes assurance_level=diagnostic. Raw predictions "
            "and caller-entered hashes are not formal evaluation evidence."
        ),
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def _query_ids(path: Path) -> tuple[list[str], str]:
    content = read_stable_regular_file(path, label="benchmark query ids")
    rows = parse_canonical_jsonl(
        content,
        label="benchmark query ids",
    )
    values: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"query_id"}:
            raise BenchmarkError("query-id rows must contain only query_id")
        query_id = row["query_id"]
        if not isinstance(query_id, str) or not query_id.strip():
            raise BenchmarkError("query_id must be a non-blank string")
        values.append(query_id)
    if not values or len(values) != len(set(values)):
        raise BenchmarkError("query-id artifact must be non-empty and unique")
    return values, sha256_bytes(content)


class _FrozenOutputExecutor:
    network_policy: Literal["offline"] = "offline"

    def __init__(self, outputs: dict[tuple[str, str], object]) -> None:
        self._outputs = outputs

    def __call__(self, tool_name: str, arguments: Mapping[str, Any]) -> object:
        key = (tool_name, sha256_bytes(canonical_json_bytes(dict(arguments))))
        try:
            return self._outputs[key]
        except KeyError:
            raise BenchmarkError("frozen output is missing for a gold case") from None


def _predictions(path: Path) -> tuple[dict[tuple[str, str], object], str]:
    content = read_stable_regular_file(path, label="frozen tool outputs")
    rows = parse_canonical_jsonl(
        content,
        label="frozen tool outputs",
    )
    outputs: dict[tuple[str, str], object] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "arguments_sha256",
            "output",
            "tool_name",
        }:
            raise BenchmarkError("frozen output row has an invalid field set")
        tool_name = row["tool_name"]
        arguments_sha256 = row["arguments_sha256"]
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise BenchmarkError("frozen output tool_name must be non-blank")
        if not isinstance(arguments_sha256, str):
            raise BenchmarkError("frozen output arguments_sha256 must be a string")
        if len(arguments_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in arguments_sha256
        ):
            raise BenchmarkError(
                "frozen output arguments_sha256 must be a lowercase sha256"
            )
        key = (tool_name, arguments_sha256)
        if key in outputs:
            raise BenchmarkError("frozen outputs contain a duplicate call identity")
        outputs[key] = row["output"]
    if not outputs:
        raise BenchmarkError("frozen outputs must not be empty")
    return outputs, sha256_bytes(content)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        query_ids, query_ids_sha256 = _query_ids(arguments.query_ids)
        gold = load_diagnostic_benchmark_gold(
            arguments.gold,
            expected_query_ids=query_ids,
        )
        outputs, predictions_sha256 = _predictions(arguments.predictions)
        expected_keys = {
            (
                case.tool_name,
                sha256_bytes(canonical_json_bytes(case.arguments)),
            )
            for case in gold.cases
        }
        if set(outputs) != expected_keys:
            missing = sorted(expected_keys - set(outputs))
            extra = sorted(set(outputs) - expected_keys)
            raise BenchmarkError(
                f"frozen output call set mismatch; missing={missing}, extra={extra}"
            )
        diagnostic = run_offline_benchmark(
            gold,
            _FrozenOutputExecutor(outputs),
            arguments.output,
            run_id=arguments.run_id,
            bindings=BenchmarkBindings(
                registry_sha256=sha256_bytes(b"diagnostic:registry:unverified"),
                tool_binding_sha256=predictions_sha256,
                runtime_binding_sha256=gold.sha256,
                run_binding_sha256=query_ids_sha256,
            ),
        )
        print(
            json.dumps(
                {
                    "assurance_level": "diagnostic",
                    "manifest": diagnostic.manifest.model_dump(mode="json"),
                    "warning": (
                        "raw prediction benchmark; not formal or verified evidence"
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(f"run-tool-benchmark: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
