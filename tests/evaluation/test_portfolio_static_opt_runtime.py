from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from scripts import refresh_portfolio_llm_static_style_contract as refresh
from scripts import prepare_portfolio_static_opt_runtime as prepare_runtime
from skillchain.evaluation.portfolio_static_opt_runtime import (
    PortfolioStaticOptRuntimeError,
    build_portfolio_static_opt_runtime_lock,
    load_verified_portfolio_static_opt_runtime,
    require_verified_portfolio_static_opt_runtime,
    validate_portfolio_static_opt_execution_control,
)
from skillchain.evaluation.portfolio_treatments import (
    compile_verified_codex_llm_static_bank,
    load_verified_codex_draft_rebind,
)
from skillchain.runners.assistant import (
    GCS_V2_MODEL_RESPONSE_CONTRACT_SHA256,
    GCS_V2_MODEL_RESPONSE_CONTRACT_VERSION,
)
from skillchain.tools.portfolio_runtime import (
    PORTFOLIO_SYSTEM_PROMPT,
    PORTFOLIO_TOOL_RUNTIME_POLICY,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
CODEX_INPUT = ROOT / "specs" / "authoring" / "authoring-packet-codex-high-v5.json"
SEMANTIC_INPUT = (
    ROOT / "specs" / "authoring" / "authoring-packet-primary-v5-candidate.json"
)
SOURCE_DRAFT = (
    ROOT
    / "runs"
    / "formal-authoring"
    / "llm-static-codex-primary-20260724-high-v5"
    / "pre-review-draft.json"
)


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@pytest.fixture(scope="module")
def refreshed_static():
    verified = load_verified_codex_draft_rebind(
        codex_input_path=CODEX_INPUT,
        expected_codex_input_file_sha256=_sha(CODEX_INPUT),
        semantic_input_path=SEMANTIC_INPUT,
        expected_semantic_input_file_sha256=_sha(SEMANTIC_INPUT),
        draft_path=SOURCE_DRAFT,
        expected_draft_file_sha256=_sha(SOURCE_DRAFT),
    )
    old_bank = compile_verified_codex_llm_static_bank(
        verified,
        tool_registry_runtime_sha256="a" * 64,
    )
    semantic, draft, bank, receipt = refresh.build_contract_refresh(
        old_semantic=verified.semantic_input,
        old_draft=verified.source_draft,
        old_semantic_file_sha256=_sha(SEMANTIC_INPUT),
        old_draft_file_sha256=_sha(SOURCE_DRAFT),
        old_bank=old_bank,
        old_bank_file_sha256=sha256_bytes(old_bank.canonical_bytes()),
        tool_registry_runtime_sha256="b" * 64,
    )
    return semantic, draft, bank, receipt


def _runtime_root(tmp_path: Path, refreshed_static) -> Path:
    semantic, _draft, bank, refresh_receipt = refreshed_static
    root = tmp_path / "static-runtime"
    core = root / "core-runtime-sources"
    core.mkdir(parents=True)
    (root / "bank-llm_static.json").write_bytes(bank.canonical_bytes())
    (root / "semantic-authoring-input.json").write_bytes(semantic.canonical_bytes())
    (root / "static-contract-refresh-receipt.json").write_bytes(
        canonical_json_bytes(refresh_receipt)
    )
    (root / "system-prompt.txt").write_text(
        PORTFOLIO_SYSTEM_PROMPT,
        encoding="utf-8",
        newline="",
    )
    source = b'{"source":"fixture"}\n'
    (core / "fixture.jsonl").write_bytes(source)
    source_sha256 = sha256_bytes(source)
    core_payload = {
        "schema_version": 1,
        "kind": "portfolio-core-runtime-sources-receipt",
        "policy_version": "portfolio-core-runtime-sources-v1",
        "provider_call_count": 0,
        "formal_eligible": False,
        "outputs": {
            "fixture.jsonl": {
                "bytes": len(source),
                "rows": 1,
                "sha256": source_sha256,
            }
        },
        "runtime_source_sha256s": [source_sha256],
        "runtime_data_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
                    "source_sha256s": [source_sha256],
                }
            )
        ),
    }
    core_receipt = {
        **core_payload,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(core_payload)),
    }
    (core / "receipt.json").write_bytes(canonical_json_bytes(core_receipt))
    lock = build_portfolio_static_opt_runtime_lock(root, active_contract={})
    (root / "runtime-lock.json").write_bytes(canonical_json_bytes(lock))
    return root


def test_static_runtime_loads_exact_one_bank_package(
    tmp_path: Path,
    refreshed_static,
) -> None:
    root = _runtime_root(tmp_path, refreshed_static)
    lock_file_sha256 = _sha(root / "runtime-lock.json")

    loaded = load_verified_portfolio_static_opt_runtime(
        root,
        expected_runtime_lock_file_sha256=lock_file_sha256,
    )

    assert loaded.runtime_lock["execution_mode"] == "static_opt_rollout"
    assert loaded.runtime_lock["bank_sha256s"] == {
        "llm_static": loaded.bank.bank_sha256
    }
    assert (
        loaded.semantic_authoring_input.input_sha256
        == loaded.runtime_lock["semantic_authoring_input_sha256"]
    )
    assert loaded.runtime_lock["provider_calls"] == 0
    assert (
        loaded.runtime_lock["gcs_v2_model_response_contract_version"]
        == GCS_V2_MODEL_RESPONSE_CONTRACT_VERSION
    )
    assert (
        loaded.runtime_lock["gcs_v2_model_response_contract_sha256"]
        == GCS_V2_MODEL_RESPONSE_CONTRACT_SHA256
    )

    lock = loaded.runtime_lock
    control = {
        "execution_scope": "static_opt_rollout",
        "runtime_lock_file_sha256": loaded.runtime_lock_file_sha256,
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "evaluation_stages": ["assistant", "gcs_v2"],
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": lock["gcs_policy_version"],
        "gcs_policy_sha256": lock["gcs_policy_sha256"],
        "gcs_scorer_evidence_policy_version": lock[
            "gcs_scorer_evidence_policy_version"
        ],
        "static_bank_file_sha256": loaded.bank_file_sha256,
        "static_bank_sha256": loaded.bank.bank_sha256,
        "static_contract_refresh_receipt_file_sha256": (
            loaded.refresh_receipt_file_sha256
        ),
        "static_contract_refresh_receipt_sha256": loaded.refresh_receipt[
            "receipt_sha256"
        ],
        "semantic_authoring_input_file_sha256": (
            loaded.semantic_authoring_input_file_sha256
        ),
        "semantic_authoring_input_sha256": (
            loaded.semantic_authoring_input.input_sha256
        ),
        "core_runtime_sources_receipt_file_sha256": (
            loaded.core_source_receipt_file_sha256
        ),
        "core_runtime_sources_receipt_sha256": loaded.core_source_receipt[
            "receipt_sha256"
        ],
        "runtime_data_sha256": lock["runtime_data_sha256"],
        "task_spec_version": lock["task_spec_version"],
        "task_spec_sha256": lock["task_spec_sha256"],
        "task_spec_file_sha256": lock["task_spec_file_sha256"],
        "execution_artifact_aliases": [],
        "execution_artifact_alias_provider_model_call_count": 0,
        "pairwise_judge_enabled": False,
        "legacy_final_judge_enabled": False,
        "analyzer_provider_call_count": 0,
    }
    validate_portfolio_static_opt_execution_control(control, loaded)
    control["rubric_path"] = "legacy-rubric.json"
    with pytest.raises(PortfolioStaticOptRuntimeError, match="execution control"):
        validate_portfolio_static_opt_execution_control(control, loaded)


def test_immutable_evidence_scope_cannot_cross_execution_boundary(
    tmp_path: Path,
    refreshed_static,
) -> None:
    root = _runtime_root(tmp_path, refreshed_static)
    active = load_verified_portfolio_static_opt_runtime(
        root,
        expected_runtime_lock_file_sha256=_sha(root / "runtime-lock.json"),
    )
    evidence = replace(active, verification_scope="immutable_evidence")

    with pytest.raises(TypeError, match="execution requires"):
        require_verified_portfolio_static_opt_runtime(evidence)


def test_static_runtime_rejects_downstream_treatment_bank(
    tmp_path: Path,
    refreshed_static,
) -> None:
    root = _runtime_root(tmp_path, refreshed_static)
    shutil.copyfile(root / "bank-llm_static.json", root / "bank-s1.json")

    with pytest.raises(PortfolioStaticOptRuntimeError, match="forbidden treatment"):
        load_verified_portfolio_static_opt_runtime(
            root,
            expected_runtime_lock_file_sha256=_sha(root / "runtime-lock.json"),
        )


def test_static_runtime_rejects_forged_unchanged_skill_proof(
    tmp_path: Path,
    refreshed_static,
) -> None:
    root = _runtime_root(tmp_path, refreshed_static)
    receipt_path = root / "static-contract-refresh-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    slug = next(iter(receipt["unchanged_skill_sha256s"]))
    receipt["unchanged_skill_sha256s"][slug] = "f" * 64
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    (root / "runtime-lock.json").unlink()

    with pytest.raises(PortfolioStaticOptRuntimeError, match="five unchanged Skills"):
        build_portfolio_static_opt_runtime_lock(root, active_contract={})


def test_prepare_static_runtime_cli_publishes_and_reloads_create_only(
    tmp_path: Path,
    refreshed_static,
) -> None:
    semantic, draft, bank, receipt = refreshed_static
    source = _runtime_root(tmp_path / "source", refreshed_static)
    refresh_root = tmp_path / "refresh"
    refresh_root.mkdir()
    (refresh_root / "bank-llm_static.json").write_bytes(bank.canonical_bytes())
    (refresh_root / "semantic-authoring-input.json").write_bytes(
        semantic.canonical_bytes()
    )
    (refresh_root / "contract-refresh-draft.json").write_bytes(draft.canonical_bytes())
    (refresh_root / "static-contract-refresh-receipt.json").write_bytes(
        canonical_json_bytes(receipt)
    )
    output = tmp_path / "assembled-runtime"

    status = prepare_runtime.main(
        [
            "--refresh-root",
            str(refresh_root),
            "--bank-file-sha256",
            _sha(refresh_root / "bank-llm_static.json"),
            "--semantic-authoring-input-file-sha256",
            _sha(refresh_root / "semantic-authoring-input.json"),
            "--contract-refresh-draft-file-sha256",
            _sha(refresh_root / "contract-refresh-draft.json"),
            "--static-contract-refresh-receipt-file-sha256",
            _sha(refresh_root / "static-contract-refresh-receipt.json"),
            "--core-runtime-sources-root",
            str(source / "core-runtime-sources"),
            "--core-runtime-sources-receipt-file-sha256",
            _sha(source / "core-runtime-sources" / "receipt.json"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    assert set(path.name for path in output.iterdir()) == {
        "bank-llm_static.json",
        "core-runtime-sources",
        "runtime-lock.json",
        "semantic-authoring-input.json",
        "static-contract-refresh-receipt.json",
        "system-prompt.txt",
    }
    loaded = load_verified_portfolio_static_opt_runtime(
        output,
        expected_runtime_lock_file_sha256=_sha(output / "runtime-lock.json"),
    )
    assert loaded.runtime_lock["provider_calls"] == 0
    assert (
        prepare_runtime.main(
            [
                "--refresh-root",
                str(refresh_root),
                "--bank-file-sha256",
                _sha(refresh_root / "bank-llm_static.json"),
                "--semantic-authoring-input-file-sha256",
                _sha(refresh_root / "semantic-authoring-input.json"),
                "--contract-refresh-draft-file-sha256",
                _sha(refresh_root / "contract-refresh-draft.json"),
                "--static-contract-refresh-receipt-file-sha256",
                _sha(refresh_root / "static-contract-refresh-receipt.json"),
                "--core-runtime-sources-root",
                str(source / "core-runtime-sources"),
                "--core-runtime-sources-receipt-file-sha256",
                _sha(source / "core-runtime-sources" / "receipt.json"),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
