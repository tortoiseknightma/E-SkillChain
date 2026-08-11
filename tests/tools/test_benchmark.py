from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from skillchain.tools.benchmark import (
    BenchmarkBindings,
    BenchmarkError,
    load_benchmark_gold,
    load_diagnostic_benchmark_gold,
    load_benchmark_result,
    require_diagnostic_benchmark_result,
    require_verified_benchmark_result,
    run_offline_benchmark,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


def _ranking(
    query_id: str,
    split: str,
    group: str,
    *,
    relevant: list[str] | None = None,
    k: int = 2,
) -> dict:
    return {
        "arguments": {"query": query_id},
        "k": k,
        "leakage_group_id": group,
        "query_id": query_id,
        "relevant_ids": relevant or ["a", "b"],
        "result_id_path": "citation.entry_id",
        "result_list_path": "hits",
        "schema_version": 1,
        "split": split,
        "task": "ranking",
        "tool_name": "encyclopedia_lookup",
    }


def _detection(query_id: str, split: str, group: str, *, two: bool = True) -> dict:
    expected = [{"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "label": "cat"}]
    if two:
        expected.append({"bbox_xyxy": [20.0, 20.0, 30.0, 30.0], "label": "dog"})
    return {
        "arguments": {"asset_id": query_id},
        "expected_detections": expected,
        "iou_threshold": 0.5,
        "leakage_group_id": group,
        "query_id": query_id,
        "schema_version": 1,
        "split": split,
        "task": "detection",
        "tool_name": "object_detect",
    }


def _ocr(
    query_id: str,
    split: str,
    group: str,
    *,
    text: str = "A😀B",
    fields: list[dict] | None = None,
) -> dict:
    return {
        "arguments": {"asset_id": query_id},
        "expected_fields": fields
        or [
            {"expected_value": "Alice", "field_name": "name"},
            {"expected_value": "42", "field_name": "amount"},
        ],
        "expected_text": text,
        "leakage_group_id": group,
        "query_id": query_id,
        "schema_version": 1,
        "split": split,
        "task": "ocr",
        "tool_name": "document_ocr",
    }


def _write_gold(path: Path, rows: list[dict]) -> Path:
    path.write_bytes(canonical_jsonl_bytes(tuple(rows)))
    return path


def _assignments(rows: list[dict]) -> list[dict]:
    return [
        {
            "leakage_group_id": row["leakage_group_id"],
            "query_id": row["query_id"],
            "split": row["split"],
        }
        for row in rows
    ]


def _write_review_ledger(path: Path, gold_path: Path, rows: list[dict]) -> Path:
    ledger = {
        "schema_version": 1,
        "kind": "formal-gold-review-ledger",
        "policy_version": "formal-gold-human-review-v1",
        "gold_sha256": sha256_bytes(gold_path.read_bytes()),
        "cases": [
            {
                "query_id": row["query_id"],
                "gold_case_sha256": sha256_bytes(canonical_json_bytes(row)),
                "decision": "approved",
                "reviewer_kind": "human",
                "reviewer_id": "benchmark-human-reviewer",
                "reviewed_at_utc": "2026-07-20T12:00:00Z",
                "blind_to_system_results": True,
                "review_basis": "Checked the expected output against source evidence.",
                "evidence": [
                    {
                        "evidence_id": f"evidence-{row['query_id']}",
                        "evidence_uri": f"urn:benchmark-test:{row['query_id']}",
                        "evidence_revision": "fixture-v1",
                        "evidence_sha256": sha256_bytes(
                            f"evidence:{row['query_id']}".encode()
                        ),
                        "basis": "Immutable fixture evidence for this expected output.",
                    }
                ],
            }
            for row in sorted(rows, key=lambda item: item["query_id"])
        ],
    }
    path.write_bytes(canonical_json_bytes(ledger))
    return path


def _load_formal_gold(path: Path, rows: list[dict]):
    review_path = _write_review_ledger(path.with_suffix(".review.json"), path, rows)
    return load_benchmark_gold(
        path,
        expected_gold_sha256=sha256_bytes(path.read_bytes()),
        authoritative_assignments=_assignments(rows),
        review_ledger_path=review_path,
        expected_review_ledger_sha256=sha256_bytes(review_path.read_bytes()),
    )


@pytest.fixture
def bindings() -> BenchmarkBindings:
    return BenchmarkBindings(
        registry_sha256="a" * 64,
        tool_binding_sha256="b" * 64,
        runtime_binding_sha256="c" * 64,
        run_binding_sha256="d" * 64,
    )


@pytest.fixture
def full_gold(tmp_path: Path):
    rows = [
        _ranking("rank-ok", "tool_dev", "group-rank-ok"),
        _ranking(
            "rank-error",
            "tool_test_frozen",
            "group-rank-error",
            relevant=["z"],
        ),
        _detection("detect-ok", "tool_dev", "group-detect-ok"),
        _detection(
            "detect-error",
            "tool_test_frozen",
            "group-detect-error",
            two=False,
        ),
        _ocr("ocr-ok", "tool_dev", "group-ocr-ok"),
        _ocr(
            "ocr-error",
            "tool_test_frozen",
            "group-ocr-error",
            text="商品",
            fields=[{"expected_value": "X", "field_name": "sku"}],
        ),
    ]
    path = _write_gold(tmp_path / "gold.jsonl", rows)
    return _load_formal_gold(path, rows)


class _Executor:
    network_policy = "offline"

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, tool_name: str, arguments: dict):
        self.calls.append((tool_name, arguments))
        query_id = arguments.get("query") or arguments["asset_id"]
        if query_id == "rank-ok":
            return {
                "hits": [
                    {"citation": {"entry_id": "x"}},
                    {"citation": {"entry_id": "a"}},
                    {"citation": {"entry_id": "c"}},
                ]
            }
        if query_id == "rank-error":
            raise RuntimeError("offline fixture failed")
        if query_id == "detect-ok":
            return {
                "detections": [
                    {
                        "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                        "confidence": 0.9,
                        "label": "cat",
                    },
                    {
                        "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                        "confidence": 0.8,
                        "label": "cat",
                    },
                ]
            }
        if query_id == "detect-error":
            return {"detections": [{"label": "cat"}]}
        if query_id == "ocr-ok":
            return {
                "fields": [
                    {
                        "evidence_line_ids": ["line-1"],
                        "field_name": "name",
                        "value": "Alice",
                    },
                    {
                        "evidence_line_ids": [],
                        "field_name": "amount",
                        "value": "41",
                    },
                ],
                "full_text": "A😃B",
                "lines": [{"line_id": "line-1", "text": "Customer: Alice"}],
            }
        if query_id == "ocr-error":
            raise OSError("fixture unavailable")
        raise AssertionError(f"unexpected query: {query_id}")


def test_gold_rejects_cross_split_leakage_and_duplicate_queries(tmp_path: Path) -> None:
    crossed = [
        _ranking("one", "tool_dev", "shared"),
        _ranking("two", "tool_test_frozen", "shared"),
    ]
    path = _write_gold(tmp_path / "crossed.jsonl", crossed)
    with pytest.raises(BenchmarkError, match="leakage group crosses"):
        load_benchmark_gold(
            path,
            expected_gold_sha256=sha256_bytes(path.read_bytes()),
            authoritative_assignments=_assignments(crossed),
        )

    duplicate = [
        _ranking("same", "tool_dev", "first"),
        _ranking("same", "tool_dev", "second"),
    ]
    path = _write_gold(tmp_path / "duplicate.jsonl", duplicate)
    with pytest.raises(BenchmarkError, match="duplicate gold query_id"):
        load_benchmark_gold(
            path,
            expected_gold_sha256=sha256_bytes(path.read_bytes()),
            authoritative_assignments=_assignments(duplicate),
        )


def test_gold_requires_exact_authoritative_query_set(tmp_path: Path) -> None:
    rows = [_ranking("one", "tool_dev", "one"), _ranking("extra", "tool_dev", "extra")]
    path = _write_gold(tmp_path / "gold.jsonl", rows)
    with pytest.raises(BenchmarkError, match="missing=.*missing.*extra=.*extra"):
        assignments = _assignments(rows)
        assignments[1] = {
            "query_id": "missing",
            "leakage_group_id": "missing",
            "split": "tool_dev",
        }
        load_benchmark_gold(
            path,
            expected_gold_sha256=sha256_bytes(path.read_bytes()),
            authoritative_assignments=assignments,
        )
    with pytest.raises(BenchmarkError, match="contains duplicates"):
        load_benchmark_gold(
            path,
            expected_gold_sha256=sha256_bytes(path.read_bytes()),
            authoritative_assignments=[_assignments(rows)[0]] * 2,
        )


def test_formal_gold_requires_external_hash_assignment_and_frozen_coverage(
    tmp_path: Path,
) -> None:
    rows = [
        _ranking("rank", "tool_test_frozen", "rank"),
        _detection("detect", "tool_test_frozen", "detect"),
        _ocr("ocr", "tool_test_frozen", "ocr"),
    ]
    path = _write_gold(tmp_path / "gold.jsonl", rows)
    with pytest.raises(BenchmarkError, match="expected_gold_sha256"):
        load_benchmark_gold(
            path,
            expected_gold_sha256="0" * 64,
            authoritative_assignments=_assignments(rows),
        )

    wrong_assignment = _assignments(rows)
    wrong_assignment[0]["leakage_group_id"] = "forged-group"
    with pytest.raises(BenchmarkError, match="assignment mismatch"):
        load_benchmark_gold(
            path,
            expected_gold_sha256=sha256_bytes(path.read_bytes()),
            authoritative_assignments=wrong_assignment,
        )

    incomplete = rows[:2]
    incomplete_path = _write_gold(tmp_path / "incomplete.jsonl", incomplete)
    with pytest.raises(BenchmarkError, match="frozen coverage is incomplete"):
        _load_formal_gold(incomplete_path, incomplete)


def test_formal_gold_requires_external_human_review_ledger(tmp_path: Path) -> None:
    rows = [
        _ranking("rank", "tool_test_frozen", "rank"),
        _detection("detect", "tool_test_frozen", "detect"),
        _ocr("ocr", "tool_test_frozen", "ocr"),
    ]
    path = _write_gold(tmp_path / "gold.jsonl", rows)
    with pytest.raises(BenchmarkError, match="GoldReviewLedger"):
        load_benchmark_gold(
            path,
            expected_gold_sha256=sha256_bytes(path.read_bytes()),
            authoritative_assignments=_assignments(rows),
        )

    review_path = _write_review_ledger(tmp_path / "review.json", path, rows)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["cases"][0]["blind_to_system_results"] = False
    review_path.write_bytes(canonical_json_bytes(review))
    with pytest.raises(BenchmarkError, match="blind_to_system_results|literal"):
        load_benchmark_gold(
            path,
            expected_gold_sha256=sha256_bytes(path.read_bytes()),
            authoritative_assignments=_assignments(rows),
            review_ledger_path=review_path,
            expected_review_ledger_sha256=sha256_bytes(review_path.read_bytes()),
        )


def test_verified_gold_deeply_rechecks_review_ledger(
    tmp_path: Path,
    bindings: BenchmarkBindings,
) -> None:
    rows = [
        _ranking("rank", "tool_test_frozen", "rank"),
        _detection("detect", "tool_test_frozen", "detect"),
        _ocr("ocr", "tool_test_frozen", "ocr"),
    ]
    path = _write_gold(tmp_path / "gold.jsonl", rows)
    verified = _load_formal_gold(path, rows)
    ledger = json.loads(verified.review_ledger_path.read_text(encoding="utf-8"))
    ledger["cases"][0]["review_basis"] = "Changed after verification."
    verified.review_ledger_path.write_bytes(canonical_json_bytes(ledger))
    with pytest.raises(BenchmarkError, match="review ledger"):
        run_offline_benchmark(
            verified,
            _Executor(),
            tmp_path / "changed-ledger-run",
            run_id="changed-ledger-v1",
            bindings=bindings,
        )


def test_gold_requires_canonical_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "pretty.jsonl"
    path.write_text(json.dumps(_ranking("one", "tool_dev", "one"), indent=2) + "\n")
    with pytest.raises(BenchmarkError, match="invalid UTF-8 JSON|canonical JSON"):
        load_benchmark_gold(
            path,
            expected_gold_sha256=sha256_bytes(path.read_bytes()),
            authoritative_assignments=[
                {"query_id": "one", "leakage_group_id": "one", "split": "tool_dev"}
            ],
        )

    link = tmp_path / "link.jsonl"
    canonical = _write_gold(
        tmp_path / "canonical.jsonl",
        [_ranking("one", "tool_dev", "one")],
    )
    try:
        link.symlink_to(canonical)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(BenchmarkError, match="non-symlink"):
        load_benchmark_gold(
            link,
            expected_gold_sha256=sha256_bytes(canonical.read_bytes()),
            authoritative_assignments=[
                {"query_id": "one", "leakage_group_id": "one", "split": "tool_dev"}
            ],
        )


def test_run_computes_all_metrics_and_keeps_errors_in_denominators(
    tmp_path: Path,
    full_gold,
    bindings: BenchmarkBindings,
) -> None:
    executor = _Executor()
    verified = run_offline_benchmark(
        full_gold,
        executor,
        tmp_path / "benchmark",
        run_id="fixture-run-v1",
        bindings=bindings,
    )
    verified = require_diagnostic_benchmark_result(verified)
    with pytest.raises(BenchmarkError, match="diagnostic only"):
        require_verified_benchmark_result(verified)  # type: ignore[arg-type]
    assert verified.manifest.assurance_level == "diagnostic"

    assert len(executor.calls) == 6
    assert verified.manifest.case_count == 6
    assert verified.manifest.success_count == 3
    assert verified.manifest.error_count == 3
    assert verified.manifest.split_counts == {
        "tool_dev": 3,
        "tool_test_frozen": 3,
    }
    assert verified.manifest.task_counts == {
        "detection": 2,
        "ocr": 2,
        "ranking": 2,
    }
    assert verified.manifest.metrics_by_split["tool_dev"].ranking is not None
    frozen_ranking = verified.manifest.metrics_by_split["tool_test_frozen"].ranking
    assert frozen_ranking is not None
    assert frozen_ranking.error_count == 1
    assert frozen_ranking.recall_at_k == 0.0

    ranking = verified.manifest.metrics.ranking
    assert ranking is not None
    expected_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ranking.case_count == 2
    assert ranking.error_count == 1
    assert ranking.recall_at_k == pytest.approx(0.25)
    assert ranking.mrr == pytest.approx(0.25)
    assert ranking.ndcg_at_k == pytest.approx(expected_ndcg / 2)

    detection = verified.manifest.metrics.detection
    assert detection is not None
    assert detection.metric_definition.endswith("not-coco-map")
    assert detection.case_count == 2
    assert detection.error_count == 1
    assert detection.gold_count == 3
    assert detection.prediction_count == 2
    assert (
        detection.true_positive,
        detection.false_positive,
        detection.false_negative,
    ) == (
        1,
        1,
        2,
    )
    assert detection.precision == pytest.approx(0.5)
    assert detection.recall == pytest.approx(1 / 3)
    assert detection.ap50 == pytest.approx(1 / 3)

    ocr = verified.manifest.metrics.ocr
    assert ocr is not None
    assert ocr.cer_unit == "unicode-codepoint"
    assert ocr.unicode_normalization == "none"
    assert ocr.reference_codepoints == 5  # The emoji is one code point.
    assert ocr.edit_distance == 3
    assert ocr.cer == pytest.approx(0.6)
    assert ocr.expected_field_count == 3
    assert ocr.field_exact == pytest.approx(1 / 3)
    assert ocr.evidence_coverage == pytest.approx(1 / 3)

    rows = {item.query_id: item for item in verified.results}
    assert rows["rank-error"].status == "error"
    assert rows["rank-error"].metrics.relevant_count == 1
    assert rows["detect-error"].status == "error"  # malformed output is retained
    assert rows["detect-error"].metrics.false_negative == 1
    assert rows["ocr-error"].metrics.reference_codepoints == 2
    assert rows["ocr-error"].metrics.edit_distance == 2


def test_result_is_create_only_and_rejects_tampering(
    tmp_path: Path,
    full_gold,
    bindings: BenchmarkBindings,
) -> None:
    destination = tmp_path / "benchmark"
    run_offline_benchmark(
        full_gold,
        _Executor(),
        destination,
        run_id="fixture-run-v1",
        bindings=bindings,
    )
    with pytest.raises(FileExistsError):
        run_offline_benchmark(
            full_gold,
            _Executor(),
            destination,
            run_id="fixture-run-v2",
            bindings=bindings,
        )

    results_path = destination / "results.jsonl"
    results_path.write_bytes(results_path.read_bytes() + b"\n")
    with pytest.raises(BenchmarkError):
        load_benchmark_result(
            destination,
            gold=full_gold,
            expected_bindings=bindings,
        )


def test_ocr_dangling_evidence_is_rejected_as_a_case_error(
    tmp_path: Path,
    full_gold,
    bindings: BenchmarkBindings,
) -> None:
    class DanglingEvidenceExecutor(_Executor):
        def __call__(self, tool_name: str, arguments: dict):
            output = super().__call__(tool_name, arguments)
            query_id = arguments.get("query") or arguments["asset_id"]
            if query_id == "ocr-ok":
                output["lines"] = [
                    {"line_id": "different-line", "text": "Customer: Alice"}
                ]
            return output

    result = run_offline_benchmark(
        full_gold,
        DanglingEvidenceExecutor(),
        tmp_path / "dangling-evidence",
        run_id="dangling-evidence",
        bindings=bindings,
    )
    by_query = {item.query_id: item for item in result.results}
    assert by_query["ocr-ok"].status == "error"
    assert by_query["ocr-ok"].metrics.evidenced_field_count == 0


def test_ocr_placeholder_evidence_does_not_count_as_grounded(
    tmp_path: Path,
    full_gold,
    bindings: BenchmarkBindings,
) -> None:
    class PlaceholderEvidenceExecutor(_Executor):
        def __call__(self, tool_name: str, arguments: dict):
            output = super().__call__(tool_name, arguments)
            query_id = arguments.get("query") or arguments["asset_id"]
            if query_id == "ocr-ok":
                output["lines"] = [{"line_id": "line-1", "text": "unrelated"}]
            return output

    result = run_offline_benchmark(
        full_gold,
        PlaceholderEvidenceExecutor(),
        tmp_path / "placeholder-evidence",
        run_id="placeholder-evidence",
        bindings=bindings,
    )
    by_query = {item.query_id: item for item in result.results}
    assert by_query["ocr-ok"].status == "ok"
    assert by_query["ocr-ok"].metrics.exact_field_count == 1
    assert by_query["ocr-ok"].metrics.evidenced_field_count == 0


def test_result_rejects_binding_mismatch(
    tmp_path: Path,
    full_gold,
    bindings: BenchmarkBindings,
) -> None:
    destination = tmp_path / "benchmark"
    run_offline_benchmark(
        full_gold,
        _Executor(),
        destination,
        run_id="fixture-run-v1",
        bindings=bindings,
    )
    wrong = bindings.model_copy(update={"registry_sha256": "e" * 64})
    with pytest.raises(BenchmarkError, match="registry_sha256 binding mismatch"):
        load_benchmark_result(
            destination,
            gold=full_gold,
            expected_bindings=wrong,
        )


def test_gold_mutation_during_execution_aborts_publication(
    tmp_path: Path,
    bindings: BenchmarkBindings,
) -> None:
    row = _ranking("one", "tool_dev", "one")
    path = _write_gold(tmp_path / "gold.jsonl", [row])
    gold = load_diagnostic_benchmark_gold(path, expected_query_ids=["one"])

    def mutate(_tool_name: str, _arguments: dict):
        path.write_bytes(canonical_jsonl_bytes((row, row)))
        return {"hits": []}

    mutate.network_policy = "offline"  # type: ignore[attr-defined]
    destination = tmp_path / "benchmark"
    with pytest.raises(BenchmarkError, match="gold changed"):
        run_offline_benchmark(
            gold,
            mutate,
            destination,
            run_id="mutated-run",
            bindings=bindings,
        )
    assert not destination.exists()


def test_duplicate_ranking_ids_are_a_case_error_not_a_skipped_case(
    tmp_path: Path,
    bindings: BenchmarkBindings,
) -> None:
    path = _write_gold(
        tmp_path / "gold.jsonl",
        [_ranking("one", "tool_dev", "one")],
    )
    gold = load_diagnostic_benchmark_gold(path, expected_query_ids=["one"])

    def duplicate(_tool_name: str, _arguments: dict):
        hit = {"citation": {"entry_id": "a"}}
        return {"hits": [hit, hit]}

    duplicate.network_policy = "offline"  # type: ignore[attr-defined]
    result = run_offline_benchmark(
        gold,
        duplicate,
        tmp_path / "benchmark",
        run_id="duplicate-output-run",
        bindings=bindings,
    )
    assert result.manifest.case_count == 1
    assert result.manifest.error_count == 1
    ranking = result.manifest.metrics.ranking
    assert ranking is not None
    assert ranking.recall_at_k == 0.0


def test_executor_must_declare_offline_policy(
    tmp_path: Path,
    bindings: BenchmarkBindings,
) -> None:
    path = _write_gold(
        tmp_path / "gold.jsonl",
        [_ranking("one", "tool_dev", "one")],
    )
    gold = load_diagnostic_benchmark_gold(path, expected_query_ids=["one"])

    with pytest.raises(BenchmarkError, match="network_policy='offline'"):
        run_offline_benchmark(
            gold,
            lambda _tool, _arguments: {"hits": []},
            tmp_path / "benchmark",
            run_id="online-not-allowed",
            bindings=bindings,
        )
