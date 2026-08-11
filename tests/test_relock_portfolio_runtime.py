from __future__ import annotations

from pathlib import Path

import pytest

from scripts import relock_portfolio_runtime as relock
from scripts.prepare_portfolio_runtime import _bank
from skillchain.tools.portfolio_runtime import (
    PORTFOLIO_SYSTEM_PROMPT,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


_CAPABILITIES = (
    "knowledge.visual_encyclopedia",
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
)


def _source_runtime(root: Path) -> tuple[Path, str, dict[str, bytes], str]:
    root.mkdir()
    recipe = canonical_json_bytes(
        {
            "author": "fixture",
            "category": "Fixture",
            "dish": "fixture dish",
            "license_id": "LicenseRef-Fixture",
            "name": "fixture recipe",
            "recipeIngredient": ["one ingredient"],
            "recipeInstructions": ["one instruction"],
            "source_archive_sha256": "a" * 64,
            "source_record_id": "fixture:1",
            "source_uri": "https://example.invalid/fixture",
        }
    )
    (root / "recipe-evidence.jsonl").write_bytes(recipe)
    runtime = build_portfolio_tool_runtime(
        relock._runtime_sources(root / "recipe-evidence.jsonl")
    )
    drafts = [
        {
            "capability_id": capability,
            "objective": f"Fixture objective for {capability}",
            "steps": [
                {
                    "tool_name": "text_product_search",
                    "instruction": "Obtain one grounded result.",
                }
            ],
            "fallback_instruction": "Return a grounded partial result.",
        }
        for capability in _CAPABILITIES
    ]
    banks = {
        name: _bank(drafts, name, runtime.registry)
        for name in ("llm_static", "s1", "s1s2", "full")
    }
    preserved: dict[str, bytes] = {}
    for name, bank in banks.items():
        content = bank.canonical_bytes()
        filename = f"bank-{name}.json"
        (root / filename).write_bytes(content)
        preserved[filename] = content
    # The accepted Windows artifact uses CRLF bytes while its lock binds the
    # normalized prompt string. Relock must preserve bytes without treating
    # platform newline translation as semantic drift.
    prompt = PORTFOLIO_SYSTEM_PROMPT.encode("utf-8").replace(b"\n", b"\r\n")
    (root / "system-prompt.txt").write_bytes(prompt)
    preserved["system-prompt.txt"] = prompt
    preserved["recipe-evidence.jsonl"] = recipe
    report_payload = {
        "schema_version": 1,
        "kind": "portfolio-post-smoke-bank-iteration",
        "formal_eligible": False,
    }
    report = {
        **report_payload,
        "report_sha256": sha256_bytes(canonical_json_bytes(report_payload)),
    }
    report_bytes = canonical_json_bytes(report)
    (root / "iteration-report.json").write_bytes(report_bytes)
    preserved["iteration-report.json"] = report_bytes
    lock_payload = {
        "schema_version": 1,
        "kind": "portfolio-assistant-runtime-lock",
        "policy_version": relock.PORTFOLIO_TOOL_RUNTIME_POLICY,
        "track": "portfolio",
        "formal_eligible": False,
        "formal_ineligible_reason": "public-runtime-fixture",
        "stage": "post_smoke_candidates",
        "tool_registry_sha256": runtime.registry.registry_sha256,
        "tool_registry_runtime_sha256": runtime.registry.registry_runtime_sha256,
        "runtime_data_sha256": runtime.index.runtime_data_sha256,
        "source_sha256s": list(runtime.index.source_sha256s),
        "system_prompt_sha256": sha256_bytes(
            PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")
        ),
        "iteration_report_sha256": report["report_sha256"],
        "bank_sha256s": {
            name: bank.bank_sha256 for name, bank in banks.items()
        },
        "bank_policy": "portfolio-post-smoke-candidate-v1",
    }
    lock = {
        **lock_payload,
        "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(lock_payload)),
    }
    lock_bytes = canonical_json_bytes(lock)
    (root / "runtime-lock.json").write_bytes(lock_bytes)
    return root, sha256_bytes(lock_bytes), preserved, lock["runtime_lock_sha256"]


def test_relock_rejects_scaffold_runtime_without_publishing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, lock_file_sha, _, _ = _source_runtime(
        tmp_path / "source"
    )
    output = tmp_path / "output"

    assert (
        relock.main(
            [
                "--source-runtime-root",
                str(source),
                "--expected-runtime-lock-file-sha256",
                lock_file_sha,
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
    assert "does not directly bind" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field",
    [
        "portfolio_tool_runtime_file_sha256",
        "tool_registry_file_sha256",
    ],
)
def test_parent_runtime_must_bind_each_current_tool_source(field: str) -> None:
    current = relock._current_tool_source_file_sha256s()
    relock._require_parent_tool_source_bindings(current)

    missing = dict(current)
    missing.pop(field)
    with pytest.raises(ValueError, match="does not directly bind"):
        relock._require_parent_tool_source_bindings(missing)

    drifted = {**current, field: "0" * 64}
    with pytest.raises(ValueError, match="does not directly bind"):
        relock._require_parent_tool_source_bindings(drifted)


def test_active_execution_contract_binds_tool_evaluator_and_judge_runtime() -> None:
    contract = relock._active_execution_contract()
    expected_files = {
        "portfolio_tool_runtime_file_sha256": (
            relock.SOURCE_ROOT / "skillchain" / "tools" / "portfolio_runtime.py"
        ),
        "tool_registry_file_sha256": (
            relock.SOURCE_ROOT / "skillchain" / "tools" / "registry.py"
        ),
        "config_file_sha256": relock.SOURCE_ROOT / "skillchain" / "config.py",
        "packets_file_sha256": (
            relock.SOURCE_ROOT / "skillchain" / "evaluation" / "packets.py"
        ),
        "evaluator_isolation_file_sha256": (
            relock.SOURCE_ROOT
            / "skillchain"
            / "evaluation"
            / "evaluator_isolation.py"
        ),
    }
    for field, path in expected_files.items():
        assert contract[field] == relock._file_sha(path)

    assert contract["final_judge_result_schema_version"] == (
        relock.FINAL_JUDGE_RESULT_SCHEMA_VERSION
    )
    assert contract["final_judge_cache_namespace"] == (
        relock.FINAL_JUDGE_CACHE_NAMESPACE
    )
    assert contract["final_judge_max_attempts"] == relock.FINAL_JUDGE_MAX_ATTEMPTS
    assert contract["final_judge_retry_policy_version"] == (
        relock.FINAL_JUDGE_RETRY_POLICY_VERSION
    )
    assert contract["final_judge_retry_policy_sha256"] == (
        relock.FINAL_JUDGE_RETRY_POLICY_SHA256
    )
    assert contract["final_judge_thinking_budget"] == (
        relock.FINAL_JUDGE_THINKING_BUDGET
    )
    assert contract["final_judge_provider_input_token_reserve"] == (
        relock.FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE
    )
    assert contract["final_judge_provider_output_token_reserve"] == (
        relock.FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE
    )
    assert contract["portfolio_budget_policy_version"] == (
        relock.PORTFOLIO_BUDGET_POLICY_VERSION
    )
    assert contract["portfolio_budget_policy_sha256"] == (
        relock.PORTFOLIO_BUDGET_POLICY_SHA256
    )
    assert contract["provider_pricing_contract_sha256"] == (
        relock.PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
    )
    assert contract["card_requirement_guard_policy_version"] == (
        relock.CARD_REQUIREMENT_GUARD_POLICY_VERSION
    )
    assert contract["card_requirement_guard_policy_sha256"] == (
        relock.CARD_REQUIREMENT_GUARD_POLICY_SHA256
    )


def test_relock_rejects_tampered_bank_without_publishing(tmp_path: Path) -> None:
    source, lock_file_sha, _, _ = _source_runtime(tmp_path / "source")
    bank_path = source / "bank-full.json"
    bank_path.write_bytes(bank_path.read_bytes() + b"\n")
    output = tmp_path / "output"

    assert (
        relock.main(
            [
                "--source-runtime-root",
                str(source),
                "--expected-runtime-lock-file-sha256",
                lock_file_sha,
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_relock_rejects_wrong_external_parent_lock_hash(tmp_path: Path) -> None:
    source, _, _, _ = _source_runtime(tmp_path / "source")
    output = tmp_path / "output"

    assert (
        relock.main(
            [
                "--source-runtime-root",
                str(source),
                "--expected-runtime-lock-file-sha256",
                "0" * 64,
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
