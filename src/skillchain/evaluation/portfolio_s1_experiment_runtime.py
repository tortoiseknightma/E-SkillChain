"""Create-only GCS-v2 execution identities for the first S1 experiment.

The existing Core Static opt800 runtime is a completed, one-Bank artifact.  It
must remain byte-exact and cannot be silently widened into a treatment runtime.
This module therefore derives a new, independent two-Bank runtime and two
strict Assistant-only launch geometries:

* ``s1_opt_replay``: the eight frozen opt replay batches, S1 only (200 rows);
* ``s1_body_gate``: the three frozen validation body-gate batches, Static and
  S1 (150 rows).

Neither identity contains a Pairwise or legacy Final-Judge stage.  All
functions in this module are deterministic and perform zero provider calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path, PurePosixPath
import re
from typing import Literal, Mapping

from pydantic import ValidationError

from skillchain.data.asset_catalog import AssetCatalog
from skillchain.evaluation.assistant_runs import AssistantRunConfig
from skillchain.evaluation.portfolio_core_inputs import (
    PortfolioCoreBatch,
    VerifiedPortfolioCoreInputs,
    require_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_gcs import (
    GCS_V2_POLICY_SHA256,
    GCS_V2_POLICY_VERSION,
)
from skillchain.evaluation.portfolio_gcs_evidence import (
    GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION,
)
from skillchain.evaluation.portfolio_launch import (
    MAIN_CONFIG_ORDER,
    PortfolioLaunchInstance,
    PortfolioLaunchShard,
)
from skillchain.evaluation.portfolio_parallel import QWEN_RATE_LIMIT_POLICY
from skillchain.evaluation.portfolio_execution import (
    PortfolioBudgetError,
)
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackBundleV1,
    PortfolioS1FeedbackBundleV2,
    PortfolioS1FeedbackBundleV3,
    PortfolioS1FeedbackBundleV4,
    load_portfolio_s1_feedback_bundle,
    load_portfolio_s1_feedback_bundle_v2,
    load_portfolio_s1_feedback_bundle_v3,
    load_portfolio_s1_feedback_bundle_v4,
)
from skillchain.evaluation.portfolio_treatments import (
    PortfolioModelInvocationReceipt,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (
    STATIC_OPT_CORE_RECEIPT_FILE,
    STATIC_OPT_REFRESH_RECEIPT_FILE,
    STATIC_OPT_SEMANTIC_INPUT_FILE,
    STATIC_OPT_SYSTEM_PROMPT_FILE,
    _active_execution_contract as _active_static_opt_execution_contract,
    load_verified_portfolio_static_opt_runtime_evidence,
    require_verified_portfolio_static_opt_runtime_evidence,
)
from skillchain.static_authoring import (
    AuthoringContractError,
    AuthoringInput,
    StaticBankArtifact,
    load_authoring_packet,
)
from skillchain.codex_authoring import (
    CodexAuthoringContractError,
    CodexAuthoringInput,
    load_codex_authoring_input,
)
from skillchain.runners.assistant import (
    PortfolioAssistantBudgetContext,
    ProductionAssistantRunner,
)
from skillchain.synthesis.store import (
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)
from skillchain.tools.portfolio_runtime import require_portfolio_diagnostic_registry
from skillchain.tools.registry import ToolRegistry

from scripts.run_portfolio_evolution_model import (
    EvolutionInvocationReceipt,
    MODEL as S1_CREATOR_MODEL,
    PortfolioEvolutionModelError,
    REASONING_EFFORT as S1_CREATOR_REASONING_EFFORT,
    S1_IMPLEMENTATION_VERSION,
    _require_common_authoring_semantics,
)


S1_EXPERIMENT_RUNTIME_POLICY_VERSION = "portfolio-s1-experiment-runtime-v1"
S1_EXPERIMENT_RUNTIME_KIND = "portfolio-s1-experiment-runtime-lock"
S1_EXPERIMENT_LAUNCH_POLICY_VERSION = "portfolio-s1-gcs-launch-v1"
S1_EXPERIMENT_LAUNCH_KIND = "portfolio-s1-gcs-launch-plan"
S1_EXPERIMENT_CONTROL_KIND = "portfolio-s1-gcs-execution-control"

S1_OPT_REPLAY_SCOPE = "s1_opt_replay"
S1_BODY_GATE_SCOPE = "s1_body_gate"
S1ExecutionScope = Literal["s1_opt_replay", "s1_body_gate"]

S1_EXPERIMENT_STATIC_BANK_FILE = "bank-llm_static.json"
S1_EXPERIMENT_CANDIDATE_BANK_FILE = "bank-s1.json"
S1_EXPERIMENT_PARENT_LOCK_FILE = "parent-static-runtime-lock.json"
S1_EXPERIMENT_RUNTIME_LOCK_FILE = "runtime-lock.json"
S1_EXPERIMENT_CREATOR_EVIDENCE_DIR = "creator-output-evidence"
S1_EXPERIMENT_CREATOR_RECEIPT_FILE = "creator-invocation-receipt.json"
S1_EXPERIMENT_MODEL_RECEIPT_FILE = "creator-model-invocation-receipt.json"
S1_EXPERIMENT_FEEDBACK_BUNDLE_FILE = "creator-feedback-bundle.json"
S1_EXPERIMENT_CREATOR_SEMANTIC_INPUT_FILE = "creator-semantic-authoring-input.json"
S1_EXPERIMENT_CODEX_INPUT_FILE = "creator-codex-authoring-input.json"
_CREATOR_SOURCE_RECEIPT_FILE = "invocation-receipt.json"
_CREATOR_MODEL_RECEIPT_FILE = "model-invocation-receipt.json"
_CREATOR_CANDIDATE_FILE = "candidate-bank.json"
S1_QWEN_REQUESTS_PER_MINUTE_CAP = 20

OPT_FOLD_POLICY_VERSION = "portfolio-core-opt800-group-folds-v1"
VAL_GATE_POLICY_VERSION = "portfolio-core-r2-text-free-validation-gates-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
_VERIFIED_RUNTIME = object()
_VERIFIED_LAUNCH = object()
_ACTIVE_EXECUTION_SCOPE = "active_execution"
_IMMUTABLE_EVIDENCE_SCOPE = "immutable_evidence"

# The completed replay200 execution is rooted in this exact runtime-v5 lock.
# Its source map records the code that produced the immutable checkpoints and
# is therefore historical evidence, not a capability to execute with stale
# code.  Only the exact file/self identities below may enter the immutable
# evidence scope; Assistant execution continues to require the active scope.
HISTORICAL_S1_RUNTIME_V5_FILE_SHA256 = (
    "464e4a6ba4fa299e260b35f4a7effaf550728ea8559c0d4cc1d2d86e758b749e"
)
_HISTORICAL_S1_RUNTIME_V5_SELF_SHA256 = (
    "db1e122f89d92b8ae638c565686a5f0496c4b1969f9ce21bc5d345e8f1e224da"
)
HISTORICAL_S1_REPLAY_V5_CONTROL_FILE_SHA256 = (
    "eede177cfee57b4297adc6eec0ff68b8475b75ac10aaa0673ffee1bc3f55477f"
)
_HISTORICAL_S1_REPLAY_V5_CONTROL_SELF_SHA256 = (
    "6d56053c6dce009843be9b48440b30dd3c770e9b6584c363889546bf64db2cc4"
)
_HISTORICAL_S1_RUNTIME_V5_SOURCE_HASHES = {
    "assistant_runs_file_sha256": (
        "1fef25284939b3f288c728d01f3734f07e411409cd97a50fa6300ef193cd83a2"
    ),
    "llm_adapter_file_sha256": (
        "086b94bdb010362ebfc04187ee3c6e77db0bb4a49bd5411c977592e5aecfeed7"
    ),
    "matrix_runner_file_sha256": (
        "2004bb6ee47d33caf5a59799c3e77cf6c38e56f94b7152668dfba459677d539b"
    ),
    "portfolio_core_inputs_file_sha256": (
        "623d8fa1025cc3acc77ec40c8cf4f884f405eeccfce4f6e81f356171fbf16af1"
    ),
    "portfolio_core_runtime_sources_file_sha256": (
        "c7545d5c770022e252b4c987464f01e3588cea4f9619d3f1d443b153b42bf68d"
    ),
    "portfolio_evolution_runner_file_sha256": (
        "b021dd6dcd50ce069c00510a75e9cc69234d5c6795a141aac786e30f36b1dd5e"
    ),
    "portfolio_execution_file_sha256": (
        "c25819ad38d15a6941f1c676e4d08f67d3cf9b70715c5e00b01b92058b34c93e"
    ),
    "portfolio_gcs_evidence_file_sha256": (
        "fd912572940529f524eab5120383a523c5f36c197af51a357755e6b055f33f84"
    ),
    "portfolio_gcs_file_sha256": (
        "57c17d597201969900edc6a30204be317df222f4b948a5d3363c324451962148"
    ),
    "portfolio_launch_file_sha256": (
        "41409e7766660e14538c1de6c8e21b62d83bc2282ecdc8e27a1d8276462ae675"
    ),
    "portfolio_parallel_file_sha256": (
        "4c1f6e31959091872d350ebf64eaabcd88d83e6d5718b985c2dbc44c5e23d38a"
    ),
    "portfolio_s1_experiment_runtime_file_sha256": (
        "5069199184aeea46fe111bcfa50c41fcf912d96d904fd1bb207c9c69398fa19c"
    ),
    "portfolio_s1_gcs_artifacts_file_sha256": (
        "d0171d9d668be3ec94200e36e311a819a22d4a39e6bc580688d26a4fb1facadd"
    ),
    "portfolio_static_gcs_corpus_file_sha256": (
        "270234f49f5c612dacb71992d259f3eb4a68f6e2597f4b451148f0ea9d245e9d"
    ),
    "portfolio_tool_runtime_file_sha256": (
        "1888190e4e958101b456d660a75735c91a743bcd1f8d8083e12c211c708bb328"
    ),
    "runner_file_sha256": (
        "da7cab45f63bbade415e9c2286d125bee4cf661ce9e801045840461571862800"
    ),
    "s1_population_analyzer_file_sha256": (
        "8831abd24225b7b9dc037e35a1f7a7b8d6dcb9da72da439b6e64d04c0c1fe62d"
    ),
    "s1_population_runner_file_sha256": (
        "0195e79dd088e8b9581f76efced96751c5f57495a502fcd95564b1e014a35682"
    ),
    "shard_finalizer_file_sha256": (
        "808bfbf11a19c80d9fde59179375e2d0b4c426b932f14781e12a6440729c15ca"
    ),
    "shard_runner_file_sha256": (
        "68dd084328f7496b2c7072b6d332706859bd0d9e52f353ff52fb07ad89046b03"
    ),
    "task_spec_file_sha256": (
        "045646e3eb845de4affff3deb73e71e65f9f27212f530b14de9b04aac4f13616"
    ),
    "tool_registry_file_sha256": (
        "6ef8e0de07ca7a73053285603f6809759b438afb83a6838183f122e20764a5e6"
    ),
}
_HISTORICAL_S1_RUNTIME_V5_EXECUTION_CONTRACT = {
    "assistant_checkpoint_schema_version": 2,
    "gcs_policy_version": "portfolio-grounded-contract-success-v2",
    "gcs_policy_sha256": (
        "ccb838617c857719e8a864f1638b3aa01f2e2aacbb7344629bcd9087b9e20651"
    ),
    "gcs_scorer_evidence_policy_version": "portfolio-gcs-scorer-evidence-v2",
    "gcs_scorer_evidence_schema_version": 2,
    "gcs_v2_model_response_contract_version": (
        "portfolio-gcs-v2-model-visible-response-contract-v1"
    ),
    "gcs_v2_model_response_contract_sha256": (
        "5528c28d375dfbd2830e2015f45ec2cf40881a10c766df20870aa1f76a41c882"
    ),
    "task_spec_version": "ecommerce-task-spec-v1",
    "task_spec_sha256": (
        "8b7b1ea59766e6de4e8c0310e9403f0ea7e550fc2aab39427adf77aa7682aa6b"
    ),
    "task_spec_file_sha256": (
        "045646e3eb845de4affff3deb73e71e65f9f27212f530b14de9b04aac4f13616"
    ),
    "tool_registry_sha256": (
        "ddf5c079ac71b217bd7c4ca22bb3db8d959272155b0c3cf7b782832524f152d0"
    ),
    "tool_registry_runtime_sha256": (
        "7e6551a18e1f79dcddc5c173dc222f69defc6620a77b0531446716074e585ab0"
    ),
    "portfolio_budget_policy_version": "portfolio-call-hard-cap-v3",
    "portfolio_budget_policy_sha256": (
        "dcb6b73462f8f350db53b119aa6428c64502bb6affe3783cdfdd6d6f145e2c7c"
    ),
    "provider_pricing_contract_version": "portfolio-provider-pricing-contract-v3",
    "provider_pricing_contract_sha256": (
        "2888b833a3ef532a05d473e748a82d6d747cf8298b66dfabfb6447e3980b0ad3"
    ),
    "portfolio_router_contract_version": "portfolio-assistant-router-v6",
    "portfolio_router_contract_sha256": (
        "dd6b73405e0507605234e3cdf3621c9361bd2105bb9e49f46c5bdf3b81d1b355"
    ),
    "portfolio_router_request_max_output_tokens": 64,
    "portfolio_router_pricing_reservation_max_output_tokens": 512,
    "portfolio_failure_policy_version": "portfolio-shard-attempt-v4",
    "portfolio_circuit_breaker_threshold": 2,
    "portfolio_max_retryable_attempts_per_query": 2,
}

PortfolioS1FeedbackBundle = (
    PortfolioS1FeedbackBundleV1
    | PortfolioS1FeedbackBundleV2
    | PortfolioS1FeedbackBundleV3
    | PortfolioS1FeedbackBundleV4
)

_RUNTIME_CONTRACT_FIELDS = (
    "assistant_checkpoint_schema_version",
    "gcs_policy_version",
    "gcs_policy_sha256",
    "gcs_scorer_evidence_policy_version",
    "gcs_scorer_evidence_schema_version",
    "gcs_v2_model_response_contract_version",
    "gcs_v2_model_response_contract_sha256",
    "task_spec_version",
    "task_spec_sha256",
    "task_spec_file_sha256",
    "tool_registry_sha256",
    "tool_registry_runtime_sha256",
    "portfolio_budget_policy_version",
    "portfolio_budget_policy_sha256",
    "provider_pricing_contract_version",
    "provider_pricing_contract_sha256",
    "portfolio_router_contract_version",
    "portfolio_router_contract_sha256",
    "portfolio_router_request_max_output_tokens",
    "portfolio_router_pricing_reservation_max_output_tokens",
    "portfolio_failure_policy_version",
    "portfolio_circuit_breaker_threshold",
    "portfolio_max_retryable_attempts_per_query",
)


class PortfolioS1ExperimentError(ValueError):
    """An S1 runtime, selection, launch, or execution control drifted."""


class PortfolioS1ExperimentAssistantRunner(ProductionAssistantRunner):
    """Portfolio runner restricted to the two physical S1 experiment Banks."""

    __slots__ = ("_runtime_lock_sha256",)

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        system_prompt: str,
        banks: Mapping[AssistantRunConfig, StaticBankArtifact],
        asset_catalog: AssetCatalog,
        runtime_lock: Mapping[str, object],
        runtime_lock_file_sha256: str,
        qwen_call_start_waiter=None,
    ) -> None:
        registry = require_portfolio_diagnostic_registry(registry)
        if (
            not system_prompt
            or system_prompt != system_prompt.strip()
            or _SHA256.fullmatch(runtime_lock_file_sha256) is None
            or runtime_lock.get("kind") != S1_EXPERIMENT_RUNTIME_KIND
            or runtime_lock.get("track") != "portfolio"
            or runtime_lock.get("formal_eligible") is not False
            or runtime_lock.get("eligible_configs") != ["llm_static", "s1"]
            or runtime_lock.get("tool_registry_sha256") != registry.registry_sha256
            or runtime_lock.get("tool_registry_runtime_sha256")
            != registry.registry_runtime_sha256
            or runtime_lock.get("system_prompt_file_sha256")
            != sha256_bytes(system_prompt.encode("utf-8"))
            or runtime_lock.get("active_source_file_sha256s") != _active_source_hashes()
            or set(banks) != {"llm_static", "s1"}
        ):
            raise PortfolioS1ExperimentError(
                "S1 Assistant runner differs from its two-Bank runtime"
            )
        runtime_lock_sha256 = runtime_lock.get("runtime_lock_sha256")
        unsigned = dict(runtime_lock)
        unsigned.pop("runtime_lock_sha256", None)
        if runtime_lock_sha256 != sha256_bytes(canonical_json_bytes(unsigned)):
            raise PortfolioS1ExperimentError("S1 runtime lock self hash mismatch")
        if type(asset_catalog) is not AssetCatalog:
            raise TypeError("S1 Assistant runner requires exactly AssetCatalog")
        asset_catalog.require_verified_files()
        held: dict[str, StaticBankArtifact] = {}
        locked_banks = runtime_lock.get("bank_sha256s")
        if not isinstance(locked_banks, Mapping):
            raise PortfolioS1ExperimentError("S1 runtime lacks Bank identities")
        for config, bank in banks.items():
            reparsed = StaticBankArtifact.model_validate(
                bank.model_dump(mode="python"), strict=True
            )
            if (
                locked_banks.get(config) != reparsed.bank_sha256
                or reparsed.tool_registry_sha256 != registry.registry_sha256
                or reparsed.tool_registry_runtime_sha256
                != registry.registry_runtime_sha256
            ):
                raise PortfolioS1ExperimentError(
                    f"S1 runner Bank differs from live registry: {config}"
                )
            held[config] = reparsed
        self._registry = registry
        self._system_prompt = system_prompt
        self._banks = held
        self._asset_catalog = asset_catalog
        self._runtime_lock_sha256 = runtime_lock_sha256
        if qwen_call_start_waiter is not None and not callable(qwen_call_start_waiter):
            raise TypeError("qwen_call_start_waiter must be callable")
        self._qwen_call_start_waiter = qwen_call_start_waiter

    def _budget_context_for_entry(
        self, value: PortfolioAssistantBudgetContext | None
    ) -> PortfolioAssistantBudgetContext:
        if type(value) is not PortfolioAssistantBudgetContext:
            raise PortfolioBudgetError(
                "S1 Assistant provider calls require a hard-budget context"
            )
        return value


_S1_EXPERIMENT_EXECUTE = PortfolioS1ExperimentAssistantRunner.execute


def require_portfolio_s1_experiment_assistant_runner(
    value: object,
) -> PortfolioS1ExperimentAssistantRunner:
    if (
        type(value) is not PortfolioS1ExperimentAssistantRunner
        or PortfolioS1ExperimentAssistantRunner.execute is not _S1_EXPERIMENT_EXECUTE
    ):
        raise TypeError(
            "S1 execution requires exactly its locked two-Bank Assistant runner"
        )
    require_portfolio_diagnostic_registry(object.__getattribute__(value, "_registry"))
    return value


@dataclass(frozen=True)
class VerifiedPortfolioS1ExperimentRuntime:
    root: Path
    runtime_lock: Mapping[str, object]
    runtime_lock_file_sha256: str
    banks: Mapping[AssistantRunConfig, StaticBankArtifact]
    bank_file_sha256s: Mapping[AssistantRunConfig, str]
    semantic_authoring_input: AuthoringInput
    core_source_receipt: Mapping[str, object]
    verification_scope: Literal["active_execution", "immutable_evidence"]
    _marker: object = field(repr=False, compare=False, default=None)


@dataclass(frozen=True)
class VerifiedPortfolioS1ExperimentLaunch:
    root: Path
    plan: Mapping[str, object]
    plan_file_sha256: str
    instances: tuple[PortfolioLaunchInstance, ...]
    shards: tuple[PortfolioLaunchShard, ...]
    _marker: object = field(repr=False, compare=False, default=None)


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PortfolioS1ExperimentError(f"{label} must be lowercase SHA-256")
    return value


def _canonical_object(path: Path, *, label: str) -> tuple[bytes, dict]:
    try:
        content = read_stable_regular_file(path, label=label, max_bytes=64 << 20)
        raw = parse_canonical_json(content, label=label)
    except (OSError, ArtifactFormatError) as error:
        raise PortfolioS1ExperimentError(f"{label} cannot be read") from error
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise PortfolioS1ExperimentError(f"{label} is not a canonical object")
    return content, raw


def _canonical_rows(path: Path, *, label: str) -> tuple[bytes, tuple[dict, ...]]:
    try:
        content = read_stable_regular_file(path, label=label, max_bytes=64 << 20)
        parsed = parse_canonical_jsonl(content, label=label)
    except (OSError, ArtifactFormatError) as error:
        raise PortfolioS1ExperimentError(f"{label} cannot be read") from error
    if not parsed or any(not isinstance(item, dict) for item in parsed):
        raise PortfolioS1ExperimentError(f"{label} must contain canonical objects")
    rows = tuple(parsed)  # type: ignore[arg-type]
    if canonical_jsonl_bytes(rows) != content:
        raise PortfolioS1ExperimentError(f"{label} is not canonical JSONL")
    return content, rows


def _self_hash(raw: Mapping[str, object], field_name: str, *, label: str) -> str:
    supplied = _sha(raw.get(field_name), label=f"{label} {field_name}")
    unsigned = dict(raw)
    unsigned.pop(field_name, None)
    if supplied != sha256_bytes(canonical_json_bytes(unsigned)):
        raise PortfolioS1ExperimentError(f"{label} self hash mismatch")
    return supplied


def _verified_file(path: Path, expected_sha256: str, *, label: str) -> bytes:
    digest = _sha(expected_sha256, label=f"expected {label}")
    try:
        content = read_stable_regular_file(path, label=label, max_bytes=128 << 20)
    except (OSError, ArtifactFormatError) as error:
        raise PortfolioS1ExperimentError(f"{label} cannot be read") from error
    if sha256_bytes(content) != digest:
        raise PortfolioS1ExperimentError(f"{label} external digest mismatch")
    return content


@dataclass(frozen=True)
class _VerifiedS1CreatorLineage:
    receipt: EvolutionInvocationReceipt
    receipt_bytes: bytes
    model_receipt: PortfolioModelInvocationReceipt
    model_receipt_bytes: bytes
    candidate_bank: StaticBankArtifact
    candidate_bytes: bytes
    feedback_bundle: PortfolioS1FeedbackBundle
    feedback_bytes: bytes
    semantic_input: AuthoringInput
    semantic_bytes: bytes
    codex_input: CodexAuthoringInput
    codex_bytes: bytes
    output_file_sha256s: Mapping[str, str]


def _load_typed_s1_feedback_bundle(
    path: Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1FeedbackBundle:
    """Dispatch only exact, canonical V1/V2/V3/V4 Feedback bundle identities."""

    content = _verified_file(
        path,
        expected_file_sha256,
        label="S1 Creator Feedback bundle",
    )
    try:
        raw = parse_canonical_json(content, label="S1 Creator Feedback bundle")
    except ArtifactFormatError as error:
        raise PortfolioS1ExperimentError(
            "S1 Creator Feedback bundle is invalid"
        ) from error
    identity = (
        (
            raw.get("schema_version"),
            raw.get("kind"),
            raw.get("policy_version"),
        )
        if isinstance(raw, dict)
        else None
    )
    if identity == (
        1,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v1",
    ):
        bundle = load_portfolio_s1_feedback_bundle(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        2,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v2",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v2(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        3,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v3",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v3(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    elif identity == (
        4,
        "portfolio-s1-feedback-bundle",
        "portfolio-s1-feedback-bundle-v4",
    ):
        bundle = load_portfolio_s1_feedback_bundle_v4(
            path,
            expected_file_sha256=expected_file_sha256,
        )
    else:
        raise PortfolioS1ExperimentError(
            "S1 Creator requires an exact PortfolioS1FeedbackBundleV1, "
            "PortfolioS1FeedbackBundleV2, PortfolioS1FeedbackBundleV3, or "
            "PortfolioS1FeedbackBundleV4 identity"
        )
    if bundle.canonical_bytes() != content:
        raise PortfolioS1ExperimentError("S1 Creator Feedback bundle is not canonical")
    return bundle


def _load_s1_creator_lineage(
    *,
    creator_output_root: Path,
    expected_invocation_receipt_file_sha256: str,
    expected_model_invocation_receipt_file_sha256: str,
    expected_candidate_bank_file_sha256: str,
    parent_bank: StaticBankArtifact,
    parent_semantic: AuthoringInput,
    parent_semantic_file_sha256: str,
    copied_input_root: Path | None = None,
) -> _VerifiedS1CreatorLineage:
    """Deep-verify one completed S1 v1.6 Creator result and all its lineage."""

    root = creator_output_root.absolute()
    if not root.is_dir() or root.is_symlink():
        raise PortfolioS1ExperimentError("S1 Creator output root is invalid")
    receipt_bytes = _verified_file(
        root / _CREATOR_SOURCE_RECEIPT_FILE,
        expected_invocation_receipt_file_sha256,
        label="S1 Creator invocation receipt",
    )
    model_receipt_bytes = _verified_file(
        root / _CREATOR_MODEL_RECEIPT_FILE,
        expected_model_invocation_receipt_file_sha256,
        label="S1 Creator model invocation receipt",
    )
    candidate_bytes = _verified_file(
        root / _CREATOR_CANDIDATE_FILE,
        expected_candidate_bank_file_sha256,
        label="S1 Creator candidate Bank",
    )
    try:
        receipt = EvolutionInvocationReceipt.model_validate_json(
            receipt_bytes, strict=True
        )
        model_receipt = PortfolioModelInvocationReceipt.model_validate_json(
            model_receipt_bytes, strict=True
        )
        candidate = StaticBankArtifact.model_validate_json(candidate_bytes, strict=True)
    except ValidationError as error:
        raise PortfolioS1ExperimentError(
            "S1 Creator receipt or candidate is invalid"
        ) from error
    if (
        receipt.canonical_bytes() != receipt_bytes
        or model_receipt.canonical_bytes() != model_receipt_bytes
        or candidate.canonical_bytes() != candidate_bytes
        or receipt.status != "completed"
        or receipt.stage != "s1_creator"
        or receipt.implementation_version != S1_IMPLEMENTATION_VERSION
        or receipt.implementation_file_sha256
        != sha256_bytes(
            (_REPOSITORY_ROOT / "scripts/run_portfolio_evolution_model.py").read_bytes()
        )
        or receipt.requested_model != S1_CREATOR_MODEL
        or receipt.reasoning_effort != S1_CREATOR_REASONING_EFFORT
        or receipt.parent_bank_sha256 != parent_bank.bank_sha256
        or receipt.candidate_bank_sha256 != candidate.bank_sha256
        or model_receipt.stage != "s1_creator"
        or model_receipt.requested_model != S1_CREATOR_MODEL
        or model_receipt.effort != S1_CREATOR_REASONING_EFFORT
        or model_receipt.thread_id != receipt.thread_id
    ):
        raise PortfolioS1ExperimentError("S1 Creator completed lineage drifted")

    output_bindings = {item.file: item.file_sha256 for item in receipt.output_files}
    if (
        len(output_bindings) != len(receipt.output_files)
        or _CREATOR_CANDIDATE_FILE not in output_bindings
        or output_bindings[_CREATOR_CANDIDATE_FILE]
        != expected_candidate_bank_file_sha256
        or any(PurePosixPath(name).name != name for name in output_bindings)
    ):
        raise PortfolioS1ExperimentError("S1 Creator output bindings drifted")
    expected_output_names = set(output_bindings) | {
        _CREATOR_SOURCE_RECEIPT_FILE,
        _CREATOR_MODEL_RECEIPT_FILE,
    }
    actual_output_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_output_names != expected_output_names or any(
        path.is_dir() or path.is_symlink() for path in root.iterdir()
    ):
        raise PortfolioS1ExperimentError("S1 Creator output layout drifted")
    for name, expected_sha in output_bindings.items():
        content = read_stable_regular_file(
            root / name, label=f"S1 Creator output {name}", max_bytes=64 << 20
        )
        if sha256_bytes(content) != expected_sha:
            raise PortfolioS1ExperimentError(f"S1 Creator output bytes drifted: {name}")

    try:
        prompt_sha = output_bindings["prompt.txt"]
        raw_sha = output_bindings["raw-model-output.json"]
        event_sha = output_bindings["codex-events.jsonl"]
        stderr_sha = output_bindings["codex-stderr.bin"]
    except KeyError as error:
        raise PortfolioS1ExperimentError(
            "S1 Creator lacks clean-turn evidence outputs"
        ) from error
    if (
        receipt.prompt_sha256 != prompt_sha
        or receipt.raw_model_output_sha256 != raw_sha
        or model_receipt.prompt_sha256 != prompt_sha
        or model_receipt.output_sha256 != raw_sha
        or model_receipt.event_log_sha256 != event_sha
        or model_receipt.stderr_sha256 != stderr_sha
        or model_receipt.model_call_count != 1
    ):
        raise PortfolioS1ExperimentError(
            "S1 Creator model receipt differs from clean-turn evidence"
        )

    input_bindings = {item.role: item for item in receipt.input_files}
    expected_roles = {
        "codex_authoring_input",
        "feedback_bundle",
        "parent_static_bank",
        "semantic_authoring_input",
    }
    if set(input_bindings) != expected_roles:
        raise PortfolioS1ExperimentError("S1 Creator input roles drifted")
    copied_names = {
        "codex_authoring_input": S1_EXPERIMENT_CODEX_INPUT_FILE,
        "feedback_bundle": S1_EXPERIMENT_FEEDBACK_BUNDLE_FILE,
        "parent_static_bank": S1_EXPERIMENT_STATIC_BANK_FILE,
        "semantic_authoring_input": S1_EXPERIMENT_CREATOR_SEMANTIC_INPUT_FILE,
    }
    input_bytes: dict[str, bytes] = {}
    for role, binding in input_bindings.items():
        source = (
            copied_input_root / copied_names[role]
            if copied_input_root is not None
            else Path(binding.path)
        )
        content = read_stable_regular_file(
            source, label=f"S1 Creator input {role}", max_bytes=64 << 20
        )
        if (
            sha256_bytes(content) != binding.file_sha256
            or sha256_bytes(content) != binding.content_sha256
        ):
            raise PortfolioS1ExperimentError(
                f"S1 Creator input binding drifted: {role}"
            )
        input_bytes[role] = content

    if input_bytes["parent_static_bank"] != parent_bank.canonical_bytes():
        raise PortfolioS1ExperimentError(
            "S1 Creator parent Bank differs from the parent runtime"
        )
    try:
        feedback = _load_typed_s1_feedback_bundle(
            (
                copied_input_root / S1_EXPERIMENT_FEEDBACK_BUNDLE_FILE
                if copied_input_root is not None
                else Path(input_bindings["feedback_bundle"].path)
            ),
            expected_file_sha256=input_bindings["feedback_bundle"].file_sha256,
        )
        semantic_source = (
            copied_input_root / S1_EXPERIMENT_CREATOR_SEMANTIC_INPUT_FILE
            if copied_input_root is not None
            else Path(input_bindings["semantic_authoring_input"].path)
        )
        semantic = load_authoring_packet(
            semantic_source,
            expected_file_sha256=input_bindings["semantic_authoring_input"].file_sha256,
        )
        codex_source = (
            copied_input_root / S1_EXPERIMENT_CODEX_INPUT_FILE
            if copied_input_root is not None
            else Path(input_bindings["codex_authoring_input"].path)
        )
        codex = load_codex_authoring_input(
            codex_source,
            expected_file_sha256=input_bindings["codex_authoring_input"].file_sha256,
        )
        _require_common_authoring_semantics(
            semantic,
            input_bindings["semantic_authoring_input"].file_sha256,
            codex,
        )
    except (
        AuthoringContractError,
        CodexAuthoringContractError,
        PortfolioEvolutionModelError,
        ValueError,
    ) as error:
        raise PortfolioS1ExperimentError(
            "S1 Creator typed input lineage is invalid"
        ) from error
    if (
        semantic.canonical_bytes() != input_bytes["semantic_authoring_input"]
        or semantic.canonical_bytes() != parent_semantic.canonical_bytes()
        or input_bindings["semantic_authoring_input"].file_sha256
        != parent_semantic_file_sha256
        or codex.canonical_bytes() != input_bytes["codex_authoring_input"]
        or feedback.canonical_bytes() != input_bytes["feedback_bundle"]
        or feedback.parent_static_bank_sha256 != parent_bank.bank_sha256
        or receipt.s1_feedback_bundle_sha256 != feedback.bundle_sha256
        or parent_bank.compiler != semantic.compiler
        or parent_bank.tool_registry_sha256 != semantic.tool_registry.identity_sha256
    ):
        raise PortfolioS1ExperimentError(
            "S1 Creator Feedback, authoring inputs, and parent runtime diverged"
        )
    _validate_candidate_bank(parent_bank, candidate)
    return _VerifiedS1CreatorLineage(
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        model_receipt=model_receipt,
        model_receipt_bytes=model_receipt_bytes,
        candidate_bank=candidate,
        candidate_bytes=candidate_bytes,
        feedback_bundle=feedback,
        feedback_bytes=input_bytes["feedback_bundle"],
        semantic_input=semantic,
        semantic_bytes=input_bytes["semantic_authoring_input"],
        codex_input=codex,
        codex_bytes=input_bytes["codex_authoring_input"],
        output_file_sha256s={
            name: sha256_bytes((root / name).read_bytes())
            for name in sorted(actual_output_names)
        },
    )


def _safe_relative(value: str, *, label: str) -> PurePosixPath:
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise PortfolioS1ExperimentError(f"{label} path is unsafe")
    return parsed


def _active_source_paths() -> dict[str, Path]:
    return {
        "portfolio_s1_experiment_runtime_file_sha256": Path(__file__).resolve(),
        "portfolio_evolution_runner_file_sha256": _REPOSITORY_ROOT
        / "scripts/run_portfolio_evolution_model.py",
        "runner_file_sha256": _SOURCE_ROOT / "skillchain/runners/assistant.py",
        "shard_runner_file_sha256": _REPOSITORY_ROOT / "scripts/run_portfolio_shard.py",
        "shard_finalizer_file_sha256": _REPOSITORY_ROOT
        / "scripts/finalize_portfolio_shard.py",
        "matrix_runner_file_sha256": _REPOSITORY_ROOT
        / "scripts/run_portfolio_matrix.py",
        "s1_population_runner_file_sha256": _REPOSITORY_ROOT
        / "scripts/run_portfolio_s1_population.py",
        "s1_population_analyzer_file_sha256": _REPOSITORY_ROOT
        / "scripts/analyze_portfolio_s1_population.py",
        "portfolio_s1_gcs_artifacts_file_sha256": _SOURCE_ROOT
        / "skillchain/evaluation/portfolio_s1_gcs_artifacts.py",
        "portfolio_static_gcs_corpus_file_sha256": _SOURCE_ROOT
        / "skillchain/evaluation/portfolio_static_gcs_corpus.py",
        "assistant_runs_file_sha256": _SOURCE_ROOT
        / "skillchain/evaluation/assistant_runs.py",
        "portfolio_execution_file_sha256": _SOURCE_ROOT
        / "skillchain/evaluation/portfolio_execution.py",
        "portfolio_launch_file_sha256": _SOURCE_ROOT
        / "skillchain/evaluation/portfolio_launch.py",
        "portfolio_parallel_file_sha256": _SOURCE_ROOT
        / "skillchain/evaluation/portfolio_parallel.py",
        "portfolio_core_inputs_file_sha256": _SOURCE_ROOT
        / "skillchain/evaluation/portfolio_core_inputs.py",
        "portfolio_core_runtime_sources_file_sha256": _SOURCE_ROOT
        / "skillchain/evaluation/portfolio_core_runtime_sources.py",
        "portfolio_gcs_file_sha256": _SOURCE_ROOT
        / "skillchain/evaluation/portfolio_gcs.py",
        "portfolio_gcs_evidence_file_sha256": _SOURCE_ROOT
        / "skillchain/evaluation/portfolio_gcs_evidence.py",
        "tool_registry_file_sha256": _SOURCE_ROOT / "skillchain/tools/registry.py",
        "portfolio_tool_runtime_file_sha256": _SOURCE_ROOT
        / "skillchain/tools/portfolio_runtime.py",
        "llm_adapter_file_sha256": _SOURCE_ROOT / "skillchain/llm.py",
        "task_spec_file_sha256": _REPOSITORY_ROOT
        / "specs/task_specs/ecommerce-task-spec-v1.json",
    }


def _active_source_hashes() -> dict[str, str]:
    return {
        name: sha256_bytes(path.read_bytes())
        for name, path in _active_source_paths().items()
    }


def _validate_candidate_bank(
    static_bank: StaticBankArtifact,
    candidate_bank: StaticBankArtifact,
) -> None:
    static_capabilities = tuple(
        item.capability_id for item in static_bank.capability_map
    )
    candidate_capabilities = tuple(
        item.capability_id for item in candidate_bank.capability_map
    )
    if (
        candidate_bank.bank_sha256 == static_bank.bank_sha256
        or candidate_bank.skills == static_bank.skills
        or candidate_bank.tool_registry_sha256 != static_bank.tool_registry_sha256
        or candidate_bank.tool_registry_runtime_sha256
        != static_bank.tool_registry_runtime_sha256
        or candidate_capabilities != static_capabilities
        or len(candidate_capabilities) != 6
    ):
        raise PortfolioS1ExperimentError(
            "S1 candidate is unchanged, incomplete, or runtime-incompatible"
        )


def _copy_tree_exact(source_root: Path, target_root: Path) -> None:
    source_root = source_root.resolve(strict=True)
    paths = tuple(sorted(source_root.rglob("*")))
    if not paths:
        raise PortfolioS1ExperimentError("Core runtime-source package is empty")
    for source in paths:
        if source.is_symlink():
            raise PortfolioS1ExperimentError(
                "Core runtime-source package contains a symlink"
            )
        if not source.is_file():
            continue
        relative = _safe_relative(
            source.relative_to(source_root).as_posix(), label="Core runtime source"
        )
        target = target_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            read_stable_regular_file(source, label=f"Core source {relative}")
        )


def _validate_core_sources(root: Path, lock: Mapping[str, object]) -> dict:
    core_root = root / "core-runtime-sources"
    receipt_path = core_root / "receipt.json"
    content, receipt = _canonical_object(receipt_path, label="Core source receipt")
    receipt_sha = _self_hash(receipt, "receipt_sha256", label="Core source receipt")
    outputs = receipt.get("outputs")
    if (
        receipt.get("provider_call_count") != 0
        or receipt.get("formal_eligible") is not False
        or not isinstance(outputs, dict)
        or sha256_bytes(content) != lock.get("core_runtime_sources_receipt_file_sha256")
        or receipt_sha != lock.get("core_runtime_sources_receipt_sha256")
        or receipt.get("runtime_data_sha256") != lock.get("runtime_data_sha256")
    ):
        raise PortfolioS1ExperimentError("Core runtime-source receipt drifted")
    expected_files = {"receipt.json"}
    for relative_text, binding in outputs.items():
        if not isinstance(relative_text, str) or not isinstance(binding, dict):
            raise PortfolioS1ExperimentError("Core source output binding is invalid")
        relative = _safe_relative(relative_text, label="Core source output")
        path = core_root.joinpath(*relative.parts)
        content = read_stable_regular_file(path, label=f"Core output {relative}")
        if sha256_bytes(content) != binding.get("sha256") or len(
            content
        ) != binding.get("bytes"):
            raise PortfolioS1ExperimentError(f"Core source output drifted: {relative}")
        expected_files.add(relative.as_posix())
    actual_files = {
        path.relative_to(core_root).as_posix()
        for path in core_root.rglob("*")
        if path.is_file()
    }
    if any(path.is_symlink() for path in core_root.rglob("*")):
        raise PortfolioS1ExperimentError("Core runtime sources contain a symlink")
    if actual_files != expected_files:
        raise PortfolioS1ExperimentError("Core runtime-source layout drifted")
    return receipt


def create_portfolio_s1_experiment_runtime(
    *,
    parent_static_runtime_root: str | Path,
    expected_parent_runtime_lock_file_sha256: str,
    creator_output_root: str | Path,
    expected_creator_invocation_receipt_file_sha256: str,
    expected_creator_model_invocation_receipt_file_sha256: str,
    expected_candidate_bank_file_sha256: str,
    output_dir: str | Path,
) -> VerifiedPortfolioS1ExperimentRuntime:
    """Derive a zero-call runtime from one externally SHA-bound Creator run."""

    output = Path(output_dir).absolute()
    if output.exists():
        raise PortfolioS1ExperimentError("S1 experiment runtime output exists")
    parent = require_verified_portfolio_static_opt_runtime_evidence(
        load_verified_portfolio_static_opt_runtime_evidence(
            parent_static_runtime_root,
            expected_runtime_lock_file_sha256=(
                expected_parent_runtime_lock_file_sha256
            ),
        )
    )
    lineage = _load_s1_creator_lineage(
        creator_output_root=Path(creator_output_root),
        expected_invocation_receipt_file_sha256=(
            expected_creator_invocation_receipt_file_sha256
        ),
        expected_model_invocation_receipt_file_sha256=(
            expected_creator_model_invocation_receipt_file_sha256
        ),
        expected_candidate_bank_file_sha256=expected_candidate_bank_file_sha256,
        parent_bank=parent.bank,
        parent_semantic=parent.semantic_authoring_input,
        parent_semantic_file_sha256=parent.semantic_authoring_input_file_sha256,
    )
    candidate = lineage.candidate_bank
    candidate_content = lineage.candidate_bytes

    staging: Path | None = None
    try:
        staging = new_staging_directory(output)
        (staging / S1_EXPERIMENT_STATIC_BANK_FILE).write_bytes(
            parent.bank.canonical_bytes()
        )
        (staging / S1_EXPERIMENT_CANDIDATE_BANK_FILE).write_bytes(candidate_content)
        (staging / S1_EXPERIMENT_CREATOR_RECEIPT_FILE).write_bytes(
            lineage.receipt_bytes
        )
        (staging / S1_EXPERIMENT_MODEL_RECEIPT_FILE).write_bytes(
            lineage.model_receipt_bytes
        )
        (staging / S1_EXPERIMENT_FEEDBACK_BUNDLE_FILE).write_bytes(
            lineage.feedback_bytes
        )
        (staging / S1_EXPERIMENT_CREATOR_SEMANTIC_INPUT_FILE).write_bytes(
            lineage.semantic_bytes
        )
        (staging / S1_EXPERIMENT_CODEX_INPUT_FILE).write_bytes(lineage.codex_bytes)
        parent_lock_content = (parent.root / "runtime-lock.json").read_bytes()
        (staging / S1_EXPERIMENT_PARENT_LOCK_FILE).write_bytes(parent_lock_content)
        for name in (
            STATIC_OPT_SEMANTIC_INPUT_FILE,
            STATIC_OPT_REFRESH_RECEIPT_FILE,
            STATIC_OPT_SYSTEM_PROMPT_FILE,
        ):
            (staging / name).write_bytes((parent.root / name).read_bytes())
        core_target = staging / "core-runtime-sources"
        core_target.mkdir()
        _copy_tree_exact(parent.root / "core-runtime-sources", core_target)
        creator_evidence_target = staging / S1_EXPERIMENT_CREATOR_EVIDENCE_DIR
        creator_evidence_target.mkdir()
        _copy_tree_exact(Path(creator_output_root), creator_evidence_target)

        copied_files = {
            "llm_static": sha256_bytes(
                (staging / S1_EXPERIMENT_STATIC_BANK_FILE).read_bytes()
            ),
            "s1": sha256_bytes(
                (staging / S1_EXPERIMENT_CANDIDATE_BANK_FILE).read_bytes()
            ),
        }
        inherited_contract = {
            name: parent.runtime_lock[name] for name in _RUNTIME_CONTRACT_FIELDS
        }
        # runtime-v8 remains immutable evidence for the completed Static opt800
        # population.  New Assistant calls must nevertheless use the active
        # hard-budget/pricing contract, so only those execution fields are
        # forward-bound here; the parent Bank, semantic input, runtime data,
        # TaskSpec, GCS contract, and receipt bytes remain unchanged.
        inherited_contract.update(_active_static_opt_execution_contract())
        payload: dict[str, object] = {
            "schema_version": 1,
            "kind": S1_EXPERIMENT_RUNTIME_KIND,
            "policy_version": S1_EXPERIMENT_RUNTIME_POLICY_VERSION,
            "track": "portfolio",
            "formal_eligible": False,
            "provider_calls": 0,
            "model_calls_performed": 0,
            "execution_modes": [S1_OPT_REPLAY_SCOPE, S1_BODY_GATE_SCOPE],
            "eligible_configs": ["llm_static", "s1"],
            "evaluation_stages": ["assistant", "gcs_v2"],
            "pairwise_judge_enabled": False,
            "legacy_final_judge_enabled": False,
            "analyzer_provider_call_count": 0,
            "execution_artifact_aliases": [],
            "execution_artifact_alias_provider_model_call_count": 0,
            "parent_static_runtime_lock_file": S1_EXPERIMENT_PARENT_LOCK_FILE,
            "parent_static_runtime_lock_file_sha256": sha256_bytes(parent_lock_content),
            "parent_static_runtime_lock_sha256": parent.runtime_lock[
                "runtime_lock_sha256"
            ],
            "bank_files": {
                "llm_static": S1_EXPERIMENT_STATIC_BANK_FILE,
                "s1": S1_EXPERIMENT_CANDIDATE_BANK_FILE,
            },
            "bank_source_file_sha256s": copied_files,
            "bank_sha256s": {
                "llm_static": parent.bank.bank_sha256,
                "s1": candidate.bank_sha256,
            },
            "s1_creator_output_dir": S1_EXPERIMENT_CREATOR_EVIDENCE_DIR,
            "s1_creator_output_file_sha256s": dict(lineage.output_file_sha256s),
            "s1_creator_invocation_receipt_file": (S1_EXPERIMENT_CREATOR_RECEIPT_FILE),
            "s1_creator_invocation_receipt_file_sha256": sha256_bytes(
                lineage.receipt_bytes
            ),
            "s1_creator_invocation_receipt_sha256": lineage.receipt.receipt_sha256,
            "s1_creator_model_invocation_receipt_file": (
                S1_EXPERIMENT_MODEL_RECEIPT_FILE
            ),
            "s1_creator_model_invocation_receipt_file_sha256": sha256_bytes(
                lineage.model_receipt_bytes
            ),
            "s1_creator_model_invocation_receipt_sha256": (
                lineage.model_receipt.invocation_receipt_sha256
            ),
            "s1_feedback_bundle_file": S1_EXPERIMENT_FEEDBACK_BUNDLE_FILE,
            "s1_feedback_bundle_file_sha256": sha256_bytes(lineage.feedback_bytes),
            "s1_feedback_bundle_sha256": lineage.feedback_bundle.bundle_sha256,
            "s1_feedback_bundle_schema_version": (
                lineage.feedback_bundle.schema_version
            ),
            "s1_feedback_bundle_policy_version": (
                lineage.feedback_bundle.policy_version
            ),
            "s1_feedback_bundle_selected_count": (
                lineage.feedback_bundle.selected_count
            ),
            "s1_feedback_bundle_parsed_count": (lineage.feedback_bundle.parsed_count),
            "s1_feedback_bundle_provider_call_count": (
                lineage.feedback_bundle.provider_call_count
            ),
            "s1_creator_semantic_authoring_input_file": (
                S1_EXPERIMENT_CREATOR_SEMANTIC_INPUT_FILE
            ),
            "s1_creator_semantic_authoring_input_file_sha256": sha256_bytes(
                lineage.semantic_bytes
            ),
            "s1_creator_semantic_authoring_input_sha256": (
                lineage.semantic_input.input_sha256
            ),
            "s1_creator_codex_authoring_input_file": S1_EXPERIMENT_CODEX_INPUT_FILE,
            "s1_creator_codex_authoring_input_file_sha256": sha256_bytes(
                lineage.codex_bytes
            ),
            "s1_creator_codex_authoring_input_sha256": lineage.codex_input.input_sha256,
            "semantic_authoring_input_file": STATIC_OPT_SEMANTIC_INPUT_FILE,
            "semantic_authoring_input_file_sha256": (
                parent.semantic_authoring_input_file_sha256
            ),
            "semantic_authoring_input_sha256": (
                parent.semantic_authoring_input.input_sha256
            ),
            "static_contract_refresh_receipt_file": (STATIC_OPT_REFRESH_RECEIPT_FILE),
            "static_contract_refresh_receipt_file_sha256": (
                parent.refresh_receipt_file_sha256
            ),
            "static_contract_refresh_receipt_sha256": parent.refresh_receipt[
                "receipt_sha256"
            ],
            "core_runtime_sources_dir": "core-runtime-sources",
            "core_runtime_sources_receipt_file": STATIC_OPT_CORE_RECEIPT_FILE,
            "core_runtime_sources_receipt_file_sha256": (
                parent.core_source_receipt_file_sha256
            ),
            "core_runtime_sources_receipt_sha256": parent.core_source_receipt[
                "receipt_sha256"
            ],
            "runtime_data_sha256": parent.runtime_lock["runtime_data_sha256"],
            "system_prompt_file": STATIC_OPT_SYSTEM_PROMPT_FILE,
            "system_prompt_file_sha256": parent.runtime_lock[
                "system_prompt_file_sha256"
            ],
            "active_source_file_sha256s": _active_source_hashes(),
            **inherited_contract,
        }
        lock = {
            **payload,
            "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }
        lock_content = canonical_json_bytes(lock)
        (staging / S1_EXPERIMENT_RUNTIME_LOCK_FILE).write_bytes(lock_content)
        atomic_publish_new_directory(staging, output)
        staging = None
        return load_verified_portfolio_s1_experiment_runtime(
            output,
            expected_runtime_lock_file_sha256=sha256_bytes(lock_content),
        )
    finally:
        if staging is not None and staging.exists():
            import shutil

            shutil.rmtree(staging)


def load_verified_portfolio_s1_experiment_runtime(
    root: str | Path,
    *,
    expected_runtime_lock_file_sha256: str,
) -> VerifiedPortfolioS1ExperimentRuntime:
    """Deeply load an S1 runtime that may grant Assistant execution."""

    return _load_verified_portfolio_s1_experiment_runtime(
        root,
        expected_runtime_lock_file_sha256=expected_runtime_lock_file_sha256,
        verification_scope=_ACTIVE_EXECUTION_SCOPE,
    )


def load_verified_portfolio_s1_experiment_runtime_evidence(
    root: str | Path,
    *,
    expected_runtime_lock_file_sha256: str,
) -> VerifiedPortfolioS1ExperimentRuntime:
    """Deeply load active evidence or the exact completed runtime-v5.

    The historical branch is deliberately restricted to one immutable lock.
    Its verified handle cannot cross the Assistant execution boundary.
    """

    expected = _sha(
        expected_runtime_lock_file_sha256,
        label="expected S1 runtime-lock file",
    )
    scope = (
        _IMMUTABLE_EVIDENCE_SCOPE
        if expected == HISTORICAL_S1_RUNTIME_V5_FILE_SHA256
        else _ACTIVE_EXECUTION_SCOPE
    )
    return _load_verified_portfolio_s1_experiment_runtime(
        root,
        expected_runtime_lock_file_sha256=expected,
        verification_scope=scope,
    )


def _load_verified_portfolio_s1_experiment_runtime(
    root: str | Path,
    *,
    expected_runtime_lock_file_sha256: str,
    verification_scope: Literal["active_execution", "immutable_evidence"],
) -> VerifiedPortfolioS1ExperimentRuntime:
    runtime_root = Path(root).absolute()
    expected_lock_sha = _sha(
        expected_runtime_lock_file_sha256, label="expected S1 runtime-lock file"
    )
    lock_content, lock = _canonical_object(
        runtime_root / S1_EXPERIMENT_RUNTIME_LOCK_FILE,
        label="S1 experiment runtime lock",
    )
    if sha256_bytes(lock_content) != expected_lock_sha:
        raise PortfolioS1ExperimentError("S1 experiment runtime-lock file drifted")
    lock_self_sha = _self_hash(
        lock, "runtime_lock_sha256", label="S1 experiment runtime lock"
    )
    immutable_evidence = verification_scope == _IMMUTABLE_EVIDENCE_SCOPE
    if immutable_evidence and (
        expected_lock_sha != HISTORICAL_S1_RUNTIME_V5_FILE_SHA256
        or lock_self_sha != _HISTORICAL_S1_RUNTIME_V5_SELF_SHA256
    ):
        raise PortfolioS1ExperimentError(
            "immutable S1 evidence runtime identity is not recognized"
        )
    expected_root_files = {
        S1_EXPERIMENT_RUNTIME_LOCK_FILE,
        S1_EXPERIMENT_PARENT_LOCK_FILE,
        S1_EXPERIMENT_STATIC_BANK_FILE,
        S1_EXPERIMENT_CANDIDATE_BANK_FILE,
        S1_EXPERIMENT_CREATOR_RECEIPT_FILE,
        S1_EXPERIMENT_MODEL_RECEIPT_FILE,
        S1_EXPERIMENT_FEEDBACK_BUNDLE_FILE,
        S1_EXPERIMENT_CREATOR_SEMANTIC_INPUT_FILE,
        S1_EXPERIMENT_CODEX_INPUT_FILE,
        STATIC_OPT_SEMANTIC_INPUT_FILE,
        STATIC_OPT_REFRESH_RECEIPT_FILE,
        STATIC_OPT_SYSTEM_PROMPT_FILE,
    }
    if (
        lock.get("schema_version") != 1
        or lock.get("kind") != S1_EXPERIMENT_RUNTIME_KIND
        or lock.get("policy_version") != S1_EXPERIMENT_RUNTIME_POLICY_VERSION
        or lock.get("formal_eligible") is not False
        or lock.get("provider_calls") != 0
        or lock.get("model_calls_performed") != 0
        or lock.get("execution_modes") != [S1_OPT_REPLAY_SCOPE, S1_BODY_GATE_SCOPE]
        or lock.get("eligible_configs") != ["llm_static", "s1"]
        or lock.get("evaluation_stages") != ["assistant", "gcs_v2"]
        or lock.get("pairwise_judge_enabled") is not False
        or lock.get("legacy_final_judge_enabled") is not False
        or lock.get("execution_artifact_aliases") != []
        or {path.name for path in runtime_root.iterdir() if path.is_file()}
        != expected_root_files
        or {path.name for path in runtime_root.iterdir() if path.is_dir()}
        != {"core-runtime-sources", S1_EXPERIMENT_CREATOR_EVIDENCE_DIR}
        or any(path.is_symlink() for path in runtime_root.iterdir())
    ):
        raise PortfolioS1ExperimentError("S1 experiment runtime identity drifted")
    expected_source_hashes = (
        _HISTORICAL_S1_RUNTIME_V5_SOURCE_HASHES
        if immutable_evidence
        else _active_source_hashes()
    )
    if lock.get("active_source_file_sha256s") != expected_source_hashes:
        raise PortfolioS1ExperimentError("S1 experiment execution source bytes drifted")
    expected_gcs_contract = (
        _HISTORICAL_S1_RUNTIME_V5_EXECUTION_CONTRACT
        if immutable_evidence
        else {
            "assistant_checkpoint_schema_version": 2,
            "gcs_policy_version": GCS_V2_POLICY_VERSION,
            "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
            "gcs_scorer_evidence_policy_version": (
                GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
            ),
            "gcs_scorer_evidence_schema_version": (
                GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION
            ),
        }
    )
    if any(
        lock.get(name) != expected_gcs_contract[name]
        for name in (
            "assistant_checkpoint_schema_version",
            "gcs_policy_version",
            "gcs_policy_sha256",
            "gcs_scorer_evidence_policy_version",
            "gcs_scorer_evidence_schema_version",
        )
    ):
        raise PortfolioS1ExperimentError("S1 runtime lacks the active GCS v2 contract")

    parent_lock_content, parent_lock = _canonical_object(
        runtime_root / S1_EXPERIMENT_PARENT_LOCK_FILE,
        label="parent Static runtime lock",
    )
    if sha256_bytes(parent_lock_content) != lock.get(
        "parent_static_runtime_lock_file_sha256"
    ) or _self_hash(
        parent_lock, "runtime_lock_sha256", label="parent Static runtime lock"
    ) != lock.get("parent_static_runtime_lock_sha256"):
        raise PortfolioS1ExperimentError("parent Static runtime identity drifted")
    expected_execution_contract = (
        _HISTORICAL_S1_RUNTIME_V5_EXECUTION_CONTRACT
        if immutable_evidence
        else _active_static_opt_execution_contract()
    )
    for name in _RUNTIME_CONTRACT_FIELDS:
        expected = expected_execution_contract.get(name, parent_lock.get(name))
        if lock.get(name) != expected:
            raise PortfolioS1ExperimentError(
                f"S1 runtime changed its frozen or active execution contract: {name}"
            )

    bank_files = lock.get("bank_files")
    bank_file_hashes = lock.get("bank_source_file_sha256s")
    bank_hashes = lock.get("bank_sha256s")
    if (
        bank_files
        != {
            "llm_static": S1_EXPERIMENT_STATIC_BANK_FILE,
            "s1": S1_EXPERIMENT_CANDIDATE_BANK_FILE,
        }
        or not isinstance(bank_file_hashes, dict)
        or set(bank_file_hashes) != {"llm_static", "s1"}
        or not isinstance(bank_hashes, dict)
        or set(bank_hashes) != {"llm_static", "s1"}
    ):
        raise PortfolioS1ExperimentError("S1 runtime Bank map drifted")
    banks: dict[AssistantRunConfig, StaticBankArtifact] = {}
    observed_files: dict[AssistantRunConfig, str] = {}
    for config, filename in (
        ("llm_static", S1_EXPERIMENT_STATIC_BANK_FILE),
        ("s1", S1_EXPERIMENT_CANDIDATE_BANK_FILE),
    ):
        content = read_stable_regular_file(runtime_root / filename, label=filename)
        digest = sha256_bytes(content)
        if digest != bank_file_hashes.get(config):
            raise PortfolioS1ExperimentError(f"{config} Bank file drifted")
        try:
            bank = StaticBankArtifact.model_validate_json(content, strict=True)
        except ValidationError as error:
            raise PortfolioS1ExperimentError(f"{config} Bank is invalid") from error
        if bank.canonical_bytes() != content or bank.bank_sha256 != bank_hashes.get(
            config
        ):
            raise PortfolioS1ExperimentError(f"{config} Bank content drifted")
        banks[config] = bank  # type: ignore[index]
        observed_files[config] = digest  # type: ignore[index]
    _validate_candidate_bank(banks["llm_static"], banks["s1"])
    if banks["llm_static"].bank_sha256 != parent_lock.get(
        "bank_sha256"
    ) or observed_files["llm_static"] != parent_lock.get("bank_file_sha256"):
        raise PortfolioS1ExperimentError("Static parent Bank bytes changed")

    semantic_path = runtime_root / STATIC_OPT_SEMANTIC_INPUT_FILE
    semantic_file_sha = sha256_bytes(semantic_path.read_bytes())
    if semantic_file_sha != lock.get("semantic_authoring_input_file_sha256"):
        raise PortfolioS1ExperimentError("semantic AuthoringInput file drifted")
    try:
        semantic = load_authoring_packet(
            semantic_path, expected_file_sha256=semantic_file_sha
        )
    except AuthoringContractError as error:
        raise PortfolioS1ExperimentError(
            "semantic AuthoringInput is invalid"
        ) from error
    if (
        semantic.input_sha256 != lock.get("semantic_authoring_input_sha256")
        or semantic.tool_registry.identity_sha256
        != banks["llm_static"].tool_registry_sha256
        or (
            semantic.tool_registry_runtime_sha256 is not None
            and semantic.tool_registry_runtime_sha256
            != banks["llm_static"].tool_registry_runtime_sha256
        )
    ):
        raise PortfolioS1ExperimentError("semantic AuthoringInput drifted")

    creator_output_hashes = lock.get("s1_creator_output_file_sha256s")
    if (
        lock.get("s1_creator_output_dir") != S1_EXPERIMENT_CREATOR_EVIDENCE_DIR
        or not isinstance(creator_output_hashes, dict)
        or lock.get("s1_creator_invocation_receipt_file")
        != S1_EXPERIMENT_CREATOR_RECEIPT_FILE
        or lock.get("s1_creator_model_invocation_receipt_file")
        != S1_EXPERIMENT_MODEL_RECEIPT_FILE
        or lock.get("s1_feedback_bundle_file") != S1_EXPERIMENT_FEEDBACK_BUNDLE_FILE
        or lock.get("s1_creator_semantic_authoring_input_file")
        != S1_EXPERIMENT_CREATOR_SEMANTIC_INPUT_FILE
        or lock.get("s1_creator_codex_authoring_input_file")
        != S1_EXPERIMENT_CODEX_INPUT_FILE
    ):
        raise PortfolioS1ExperimentError("S1 Creator runtime lineage map drifted")
    lineage = _load_s1_creator_lineage(
        creator_output_root=(runtime_root / S1_EXPERIMENT_CREATOR_EVIDENCE_DIR),
        expected_invocation_receipt_file_sha256=_sha(
            lock.get("s1_creator_invocation_receipt_file_sha256"),
            label="S1 Creator receipt file SHA-256",
        ),
        expected_model_invocation_receipt_file_sha256=_sha(
            lock.get("s1_creator_model_invocation_receipt_file_sha256"),
            label="S1 Creator model receipt file SHA-256",
        ),
        expected_candidate_bank_file_sha256=observed_files["s1"],
        parent_bank=banks["llm_static"],
        parent_semantic=semantic,
        parent_semantic_file_sha256=semantic_file_sha,
        copied_input_root=runtime_root,
    )
    if (
        dict(lineage.output_file_sha256s) != creator_output_hashes
        or lineage.receipt_bytes
        != (runtime_root / S1_EXPERIMENT_CREATOR_RECEIPT_FILE).read_bytes()
        or lineage.model_receipt_bytes
        != (runtime_root / S1_EXPERIMENT_MODEL_RECEIPT_FILE).read_bytes()
        or lineage.receipt.receipt_sha256
        != lock.get("s1_creator_invocation_receipt_sha256")
        or lineage.model_receipt.invocation_receipt_sha256
        != lock.get("s1_creator_model_invocation_receipt_sha256")
        or sha256_bytes(lineage.feedback_bytes)
        != lock.get("s1_feedback_bundle_file_sha256")
        or lineage.feedback_bundle.bundle_sha256
        != lock.get("s1_feedback_bundle_sha256")
        or lineage.feedback_bundle.schema_version
        != lock.get("s1_feedback_bundle_schema_version")
        or lineage.feedback_bundle.policy_version
        != lock.get("s1_feedback_bundle_policy_version")
        or lineage.feedback_bundle.selected_count
        != lock.get("s1_feedback_bundle_selected_count")
        or lineage.feedback_bundle.parsed_count
        != lock.get("s1_feedback_bundle_parsed_count")
        or lineage.feedback_bundle.provider_call_count
        != lock.get("s1_feedback_bundle_provider_call_count")
        or sha256_bytes(lineage.semantic_bytes)
        != lock.get("s1_creator_semantic_authoring_input_file_sha256")
        or lineage.semantic_input.input_sha256
        != lock.get("s1_creator_semantic_authoring_input_sha256")
        or sha256_bytes(lineage.codex_bytes)
        != lock.get("s1_creator_codex_authoring_input_file_sha256")
        or lineage.codex_input.input_sha256
        != lock.get("s1_creator_codex_authoring_input_sha256")
        or lineage.candidate_bytes
        != (runtime_root / S1_EXPERIMENT_CANDIDATE_BANK_FILE).read_bytes()
    ):
        raise PortfolioS1ExperimentError("S1 Creator runtime lineage drifted")
    for filename, file_field, content_field in (
        (
            STATIC_OPT_REFRESH_RECEIPT_FILE,
            "static_contract_refresh_receipt_file_sha256",
            "static_contract_refresh_receipt_sha256",
        ),
    ):
        content, raw = _canonical_object(runtime_root / filename, label=filename)
        if sha256_bytes(content) != lock.get(file_field) or _self_hash(
            raw, "receipt_sha256", label=filename
        ) != lock.get(content_field):
            raise PortfolioS1ExperimentError(f"{filename} drifted")
    prompt = (runtime_root / STATIC_OPT_SYSTEM_PROMPT_FILE).read_bytes()
    if sha256_bytes(prompt) != lock.get("system_prompt_file_sha256"):
        raise PortfolioS1ExperimentError("system prompt drifted")
    core_receipt = _validate_core_sources(runtime_root, lock)
    return VerifiedPortfolioS1ExperimentRuntime(
        root=runtime_root,
        runtime_lock=lock,
        runtime_lock_file_sha256=expected_lock_sha,
        banks=banks,
        bank_file_sha256s=observed_files,
        semantic_authoring_input=semantic,
        core_source_receipt=core_receipt,
        verification_scope=verification_scope,
        _marker=_VERIFIED_RUNTIME,
    )


def require_verified_portfolio_s1_experiment_runtime(
    value: object,
) -> VerifiedPortfolioS1ExperimentRuntime:
    if (
        type(value) is not VerifiedPortfolioS1ExperimentRuntime
        or value._marker is not _VERIFIED_RUNTIME
        or value.verification_scope != _ACTIVE_EXECUTION_SCOPE
    ):
        raise TypeError("S1 execution requires its dedicated verified runtime")
    return load_verified_portfolio_s1_experiment_runtime(
        value.root,
        expected_runtime_lock_file_sha256=value.runtime_lock_file_sha256,
    )


def require_verified_portfolio_s1_experiment_runtime_evidence(
    value: object,
) -> VerifiedPortfolioS1ExperimentRuntime:
    if (
        type(value) is not VerifiedPortfolioS1ExperimentRuntime
        or value._marker is not _VERIFIED_RUNTIME
        or value.verification_scope
        not in {_ACTIVE_EXECUTION_SCOPE, _IMMUTABLE_EVIDENCE_SCOPE}
    ):
        raise TypeError("S1 evidence requires its dedicated verified runtime")
    return load_verified_portfolio_s1_experiment_runtime_evidence(
        value.root,
        expected_runtime_lock_file_sha256=value.runtime_lock_file_sha256,
    )


def _verify_replay_selection(
    inputs: VerifiedPortfolioCoreInputs,
    *,
    manifest_path: Path,
    expected_manifest_file_sha256: str,
    mapping_path: Path,
    expected_mapping_file_sha256: str,
) -> tuple[tuple[PortfolioCoreBatch, ...], dict[str, object]]:
    manifest_content, manifest = _canonical_object(
        manifest_path, label="opt fold manifest"
    )
    mapping_content, rows = _canonical_rows(mapping_path, label="opt fold mapping")
    if (
        sha256_bytes(manifest_content)
        != _sha(expected_manifest_file_sha256, label="fold manifest SHA-256")
        or sha256_bytes(mapping_content)
        != _sha(expected_mapping_file_sha256, label="fold mapping SHA-256")
        or manifest.get("schema_version") != 1
        or manifest.get("policy_version") != OPT_FOLD_POLICY_VERSION
        or manifest.get("mapping_sha256") != sha256_bytes(mapping_content)
        or manifest.get("plan_sha256") != inputs.expected_plan_sha256
        or manifest.get("source_queries_sha256")
        != inputs.expected_query_artifact_sha256
        or len(rows) != 800
    ):
        raise PortfolioS1ExperimentError("opt replay fold identity drifted")
    query_by_id = {query.query_id: query for query in inputs.queries}
    opt_ids = {query.query_id for query in inputs.queries if query.split == "opt_pool"}
    if len(opt_ids) != 800 or {row.get("query_id") for row in rows} != opt_ids:
        raise PortfolioS1ExperimentError("opt fold mapping does not cover opt_pool")
    roles_by_batch: dict[str, set[object]] = {}
    replay_ids: list[str] = []
    for row in rows:
        if (
            set(row)
            != {
                "schema_version",
                "query_id",
                "atomic_batch_id",
                "fold_id",
                "role",
            }
            or row.get("schema_version") != 1
            or row.get("role")
            not in {
                "discovery",
                "replay",
            }
        ):
            raise PortfolioS1ExperimentError("opt fold row is invalid")
        query_id = row["query_id"]
        assert isinstance(query_id, str)
        query = query_by_id[query_id]
        if row.get("atomic_batch_id") != query.generator_batch_id:
            raise PortfolioS1ExperimentError("opt fold batch binding drifted")
        roles_by_batch.setdefault(query.generator_batch_id, set()).add(row["role"])
        if row["role"] == "replay":
            replay_ids.append(query_id)
    if any(len(roles) != 1 for roles in roles_by_batch.values()):
        raise PortfolioS1ExperimentError("opt fold splits an atomic batch")
    replay_batches = tuple(
        batch
        for batch in inputs.batches
        if batch.split == "opt_pool"
        and roles_by_batch.get(batch.batch_id) == {"replay"}
    )
    if (
        len(replay_ids) != 200
        or len(replay_batches) != 8
        or any(len(batch.query_ids) != 25 for batch in replay_batches)
        or set(replay_ids)
        != {query_id for batch in replay_batches for query_id in batch.query_ids}
        or manifest.get("replay_query_ids_sha256")
        != sha256_bytes(canonical_json_bytes(replay_ids))
        or len(roles_by_batch) != 32
    ):
        raise PortfolioS1ExperimentError("opt replay geometry is not 8x25")
    return replay_batches, {
        "selection_policy_version": OPT_FOLD_POLICY_VERSION,
        "selection_manifest_file_sha256": sha256_bytes(manifest_content),
        "selection_mapping_file_sha256": sha256_bytes(mapping_content),
        "selection_query_ids_sha256": sha256_bytes(canonical_json_bytes(replay_ids)),
    }


def _verify_body_gate_selection(
    inputs: VerifiedPortfolioCoreInputs,
    *,
    gate_path: Path,
    expected_gate_file_sha256: str,
) -> tuple[tuple[PortfolioCoreBatch, ...], dict[str, object]]:
    content, artifact = _canonical_object(gate_path, label="validation gate artifact")
    if sha256_bytes(content) != _sha(
        expected_gate_file_sha256, label="validation gate file SHA-256"
    ):
        raise PortfolioS1ExperimentError("validation gate file drifted")
    manifest = artifact.get("manifest")
    audit = artifact.get("audit")
    if (
        artifact.get("schema_version") != 1
        or not isinstance(manifest, dict)
        or not isinstance(audit, dict)
        or manifest.get("policy_version") != VAL_GATE_POLICY_VERSION
        or manifest.get("plan_sha256") != inputs.expected_plan_sha256
        or manifest.get("capability_assignments_sha256")
        != inputs.expected_capability_assignments_sha256
        or audit.get("source_split") != "val"
        or audit.get("gate_sizes")
        != {"route_gate": 75, "body_gate": 75, "shadow_val": 50}
    ):
        raise PortfolioS1ExperimentError("validation body-gate identity drifted")
    query_to_gate = audit.get("query_id_to_gate")
    val_ids = {query.query_id for query in inputs.queries if query.split == "val"}
    if (
        not isinstance(query_to_gate, dict)
        or set(query_to_gate) != val_ids
        or len(val_ids) != 200
        or any(
            value not in {"route_gate", "body_gate", "shadow_val"}
            for value in query_to_gate.values()
        )
    ):
        raise PortfolioS1ExperimentError("validation gate mapping drifted")
    body_ids = [
        query.query_id
        for query in inputs.queries
        if query.split == "val" and query_to_gate.get(query.query_id) == "body_gate"
    ]
    body_set = set(body_ids)
    body_batches = tuple(
        batch
        for batch in inputs.batches
        if batch.split == "val" and set(batch.query_ids) <= body_set
    )
    if (
        len(body_ids) != 75
        or len(body_batches) != 3
        or any(len(batch.query_ids) != 25 for batch in body_batches)
        or {query_id for batch in body_batches for query_id in batch.query_ids}
        != body_set
        or any(
            set(batch.query_ids) & body_set and not set(batch.query_ids) <= body_set
            for batch in inputs.batches
        )
    ):
        raise PortfolioS1ExperimentError("body_gate is not exactly 3 atomic batches")
    return body_batches, {
        "selection_policy_version": VAL_GATE_POLICY_VERSION,
        "selection_gate_file_sha256": sha256_bytes(content),
        "selection_query_ids_sha256": sha256_bytes(canonical_json_bytes(body_ids)),
    }


def _build_instances_and_shards(
    inputs: VerifiedPortfolioCoreInputs,
    *,
    batches: tuple[PortfolioCoreBatch, ...],
    configs: tuple[AssistantRunConfig, ...],
    matrix_run_id: str,
) -> tuple[tuple[PortfolioLaunchInstance, ...], tuple[PortfolioLaunchShard, ...]]:
    query_order = tuple(query_id for batch in batches for query_id in batch.query_ids)
    query_ordinal = {query_id: index for index, query_id in enumerate(query_order)}
    assistant_by_id = {item.query_id: item for item in inputs.assistant_queries}
    asset_by_id = {item.query_id: item for item in inputs.query_assets}
    instances: list[PortfolioLaunchInstance] = []
    shards: list[PortfolioLaunchShard] = []
    for batch in batches:
        for config in configs:
            config_ordinal = MAIN_CONFIG_ORDER.index(config)
            shard_ordinal = len(shards)
            shard_id = (
                f"{shard_ordinal:03d}-{batch.batch_id}-{config_ordinal:02d}-{config}"
            )
            shard_instances: list[PortfolioLaunchInstance] = []
            for query_id in batch.query_ids:
                assistant = assistant_by_id[query_id]
                asset = asset_by_id[query_id]
                if assistant.asset_binding is None:
                    raise PortfolioS1ExperimentError(
                        "Core Assistant query lacks its opaque asset binding"
                    )
                payload: dict[str, object] = {
                    "schema_version": 1,
                    "kind": "portfolio-launch-instance",
                    "matrix_run_id": matrix_run_id,
                    "instance_ordinal": len(instances),
                    "shard_id": shard_id,
                    "shard_ordinal": shard_ordinal,
                    "query_ordinal": query_ordinal[query_id],
                    "config_ordinal": config_ordinal,
                    "config": config,
                    "accepted_batch_id": batch.batch_id,
                    "query_id": query_id,
                    "query_sha256": assistant.query_sha256,
                    "public_input_sha256": assistant.public_input_sha256,
                    "asset_token": assistant.asset_binding.asset_token,
                    "asset_binding_sha256": assistant.asset_binding.binding_sha256,
                    "image_sha256": asset.image_sha256,
                    "assistant_output_relpath": (
                        f"shards/{shard_id}/assistant/{query_id}.json"
                    ),
                    # Reserved for compatibility with the established typed row;
                    # GCS-only controls explicitly prohibit producing this file.
                    "final_output_relpath": f"shards/{shard_id}/final/{query_id}.json",
                }
                instance = PortfolioLaunchInstance.model_validate(
                    {
                        **payload,
                        "instance_sha256": sha256_bytes(canonical_json_bytes(payload)),
                    },
                    strict=True,
                )
                instances.append(instance)
                shard_instances.append(instance)
            shard_payload = {
                "shard_id": shard_id,
                "shard_ordinal": shard_ordinal,
                "accepted_batch_id": batch.batch_id,
                "config": config,
                "config_ordinal": config_ordinal,
                "query_count": 25,
                "query_ids": [item.query_id for item in shard_instances],
                "instance_sha256s": [item.instance_sha256 for item in shard_instances],
                "output_relpath": f"shards/{shard_id}",
            }
            shards.append(
                PortfolioLaunchShard.model_validate(
                    {
                        **shard_payload,
                        "shard_sha256": sha256_bytes(
                            canonical_json_bytes(shard_payload)
                        ),
                    },
                    strict=True,
                )
            )
    return tuple(instances), tuple(shards)


def _build_launch(
    inputs: VerifiedPortfolioCoreInputs,
    runtime: VerifiedPortfolioS1ExperimentRuntime,
    *,
    execution_scope: S1ExecutionScope,
    matrix_run_id: str,
    replay_manifest_path: Path | None,
    expected_replay_manifest_file_sha256: str | None,
    replay_mapping_path: Path | None,
    expected_replay_mapping_file_sha256: str | None,
    validation_gate_path: Path | None,
    expected_validation_gate_file_sha256: str | None,
    core_input_binding_launch_root: Path | None,
    core_input_binding_launch_plan_file_sha256: str | None,
) -> tuple[dict[str, object], bytes, tuple[PortfolioLaunchInstance, ...]]:
    if _RUN_ID.fullmatch(matrix_run_id) is None:
        raise PortfolioS1ExperimentError("matrix_run_id contains unsafe characters")
    if execution_scope == S1_OPT_REPLAY_SCOPE:
        if (
            None
            in (
                replay_manifest_path,
                expected_replay_manifest_file_sha256,
                replay_mapping_path,
                expected_replay_mapping_file_sha256,
            )
            or validation_gate_path is not None
        ):
            raise PortfolioS1ExperimentError(
                "s1_opt_replay requires only its manifest and mapping"
            )
        assert replay_manifest_path is not None
        assert expected_replay_manifest_file_sha256 is not None
        assert replay_mapping_path is not None
        assert expected_replay_mapping_file_sha256 is not None
        batches, selection = _verify_replay_selection(
            inputs,
            manifest_path=replay_manifest_path,
            expected_manifest_file_sha256=expected_replay_manifest_file_sha256,
            mapping_path=replay_mapping_path,
            expected_mapping_file_sha256=expected_replay_mapping_file_sha256,
        )
        configs: tuple[AssistantRunConfig, ...] = ("s1",)
        selected_split = "opt_pool"
        expected = (200, 8, 200)
    else:
        if (
            validation_gate_path is None
            or expected_validation_gate_file_sha256 is None
            or replay_manifest_path is not None
            or replay_mapping_path is not None
        ):
            raise PortfolioS1ExperimentError(
                "s1_body_gate requires only its validation-gate artifact"
            )
        batches, selection = _verify_body_gate_selection(
            inputs,
            gate_path=validation_gate_path,
            expected_gate_file_sha256=expected_validation_gate_file_sha256,
        )
        configs = ("llm_static", "s1")
        selected_split = "val"
        expected = (75, 6, 150)
    instances, shards = _build_instances_and_shards(
        inputs, batches=batches, configs=configs, matrix_run_id=matrix_run_id
    )
    query_count, shard_count, instance_count = expected
    if (
        len(instances) != instance_count
        or len(shards) != shard_count
        or len({item.query_id for item in instances}) != query_count
        or any(item.query_count != 25 for item in shards)
    ):
        raise PortfolioS1ExperimentError("S1 launch geometry drifted")
    instances_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in instances)
    )
    lock = runtime.runtime_lock
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": S1_EXPERIMENT_LAUNCH_KIND,
        "policy_version": S1_EXPERIMENT_LAUNCH_POLICY_VERSION,
        "track": "portfolio",
        "formal_eligible": False,
        "status": "prepared_ready",
        "execution_ready": True,
        "execution_authorized": False,
        "model_calls_performed": 0,
        "execution_mode": execution_scope,
        "matrix_run_id": matrix_run_id,
        "dataset_profile": "core",
        "selected_split": selected_split,
        "test_frozen_access": False,
        "query_count": query_count,
        "config_count": len(configs),
        "instance_count": instance_count,
        "shard_count": shard_count,
        "config_order": list(configs),
        "selected_batch_ids": [batch.batch_id for batch in batches],
        "portfolio_plan_sha256": inputs.expected_plan_sha256,
        "query_artifact_sha256": inputs.expected_query_artifact_sha256,
        "capability_assignments_sha256": (
            inputs.expected_capability_assignments_sha256
        ),
        "runtime_catalog_sha256": inputs.expected_output_catalog_sha256,
        "core_input_binding_launch_root": (
            None
            if core_input_binding_launch_root is None
            else core_input_binding_launch_root.absolute().as_posix()
        ),
        "core_input_binding_launch_plan_file_sha256": (
            None
            if core_input_binding_launch_plan_file_sha256 is None
            else _sha(
                core_input_binding_launch_plan_file_sha256,
                label="Core input-binding launch-plan file",
            )
        ),
        **selection,
        "runtime_lock_file_sha256": runtime.runtime_lock_file_sha256,
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "bank_file_sha256s": dict(runtime.bank_file_sha256s),
        "bank_sha256s": {
            config: runtime.banks[config].bank_sha256 for config in ("llm_static", "s1")
        },
        "evaluation_stages": ["assistant", "gcs_v2"],
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": GCS_V2_POLICY_VERSION,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "gcs_scorer_evidence_policy_version": (GCS_SCORER_EVIDENCE_V2_POLICY_VERSION),
        "pairwise_judge_enabled": False,
        "legacy_final_judge_enabled": False,
        "final_output_publication_enabled": False,
        "analyzer_provider_call_count": 0,
        "assistant_concurrency": 2,
        "effective_process_assistant_concurrency": 1,
        "final_judge_concurrency": 0,
        "qwen_requests_per_minute_cap": S1_QWEN_REQUESTS_PER_MINUTE_CAP,
        "qwen_rate_limit_policy": QWEN_RATE_LIMIT_POLICY,
        "resume_policy": "create_only_shards_skip_only_verified_complete",
        "checkpoint_policy": "after_every_terminal_query_and_25_query_shard",
        "calls": {
            "assistant_instance_count": instance_count,
            "assistant_call_floor": instance_count * 2,
            "assistant_call_ceiling": instance_count * 5,
            "final_judge_instance_count": 0,
            "final_judge_call_count": 0,
            "pairwise_call_count": 0,
        },
        "instances_file_sha256": sha256_bytes(instances_bytes),
        "shards": [item.model_dump(mode="json") for item in shards],
    }
    plan = {
        **payload,
        "launch_plan_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }
    return plan, instances_bytes, instances


def create_portfolio_s1_experiment_launch_package(
    inputs: VerifiedPortfolioCoreInputs,
    runtime: VerifiedPortfolioS1ExperimentRuntime,
    *,
    execution_scope: S1ExecutionScope,
    matrix_run_id: str,
    output_dir: str | Path,
    replay_manifest_path: str | Path | None = None,
    expected_replay_manifest_file_sha256: str | None = None,
    replay_mapping_path: str | Path | None = None,
    expected_replay_mapping_file_sha256: str | None = None,
    validation_gate_path: str | Path | None = None,
    expected_validation_gate_file_sha256: str | None = None,
    core_input_binding_launch_root: str | Path | None = None,
    core_input_binding_launch_plan_file_sha256: str | None = None,
) -> VerifiedPortfolioS1ExperimentLaunch:
    verified_inputs = require_verified_portfolio_core_inputs(inputs)
    verified_runtime = require_verified_portfolio_s1_experiment_runtime(runtime)
    output = Path(output_dir).absolute()
    if output.exists():
        raise PortfolioS1ExperimentError("S1 launch output exists")
    replay_manifest = (
        None if replay_manifest_path is None else Path(replay_manifest_path)
    )
    replay_mapping = None if replay_mapping_path is None else Path(replay_mapping_path)
    validation_gate = (
        None if validation_gate_path is None else Path(validation_gate_path)
    )
    input_binding_root = (
        None
        if core_input_binding_launch_root is None
        else Path(core_input_binding_launch_root)
    )
    if (input_binding_root is None) != (
        core_input_binding_launch_plan_file_sha256 is None
    ):
        raise PortfolioS1ExperimentError(
            "Core input-binding launch root and SHA-256 must be supplied together"
        )
    plan, instances_bytes, _instances = _build_launch(
        verified_inputs,
        verified_runtime,
        execution_scope=execution_scope,
        matrix_run_id=matrix_run_id,
        replay_manifest_path=replay_manifest,
        expected_replay_manifest_file_sha256=(expected_replay_manifest_file_sha256),
        replay_mapping_path=replay_mapping,
        expected_replay_mapping_file_sha256=(expected_replay_mapping_file_sha256),
        validation_gate_path=validation_gate,
        expected_validation_gate_file_sha256=(expected_validation_gate_file_sha256),
        core_input_binding_launch_root=input_binding_root,
        core_input_binding_launch_plan_file_sha256=(
            core_input_binding_launch_plan_file_sha256
        ),
    )
    staging: Path | None = None
    try:
        staging = new_staging_directory(output)
        selection_root = staging / "selection"
        selection_root.mkdir()
        if execution_scope == S1_OPT_REPLAY_SCOPE:
            assert replay_manifest is not None and replay_mapping is not None
            (selection_root / "fold-manifest.json").write_bytes(
                replay_manifest.read_bytes()
            )
            (selection_root / "fold-mapping.jsonl").write_bytes(
                replay_mapping.read_bytes()
            )
        else:
            assert validation_gate is not None
            (selection_root / "validation-gates.json").write_bytes(
                validation_gate.read_bytes()
            )
        plan_content = canonical_json_bytes(plan)
        (staging / "instances.jsonl").write_bytes(instances_bytes)
        (staging / "launch-plan.json").write_bytes(plan_content)
        state_payload = {
            "schema_version": 1,
            "kind": "portfolio-s1-gcs-launch-state",
            "matrix_run_id": matrix_run_id,
            "launch_plan_sha256": plan["launch_plan_sha256"],
            "status": "not_started",
            "completed_shard_ids": [],
            "failed_shard_ids": [],
            "model_calls_performed": 0,
        }
        state = {
            **state_payload,
            "state_sha256": sha256_bytes(canonical_json_bytes(state_payload)),
        }
        (staging / "launch-state.json").write_bytes(canonical_json_bytes(state))
        atomic_publish_new_directory(staging, output)
        staging = None
        return load_verified_portfolio_s1_experiment_launch(
            output,
            expected_plan_file_sha256=sha256_bytes(plan_content),
        )
    finally:
        if staging is not None and staging.exists():
            import shutil

            shutil.rmtree(staging)


def load_verified_portfolio_s1_experiment_launch(
    root: str | Path,
    *,
    expected_plan_file_sha256: str,
) -> VerifiedPortfolioS1ExperimentLaunch:
    launch_root = Path(root).absolute()
    content, plan = _canonical_object(
        launch_root / "launch-plan.json", label="S1 launch plan"
    )
    expected_sha = _sha(expected_plan_file_sha256, label="expected launch-plan file")
    if sha256_bytes(content) != expected_sha:
        raise PortfolioS1ExperimentError("S1 launch-plan file drifted")
    _self_hash(plan, "launch_plan_sha256", label="S1 launch plan")
    scope = plan.get("execution_mode")
    geometry = {
        S1_OPT_REPLAY_SCOPE: ("opt_pool", ["s1"], 200, 8, 200),
        S1_BODY_GATE_SCOPE: ("val", ["llm_static", "s1"], 75, 6, 150),
    }.get(scope)
    if geometry is None:
        raise PortfolioS1ExperimentError("S1 launch scope is invalid")
    split, configs, query_count, shard_count, instance_count = geometry
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != S1_EXPERIMENT_LAUNCH_KIND
        or plan.get("policy_version") != S1_EXPERIMENT_LAUNCH_POLICY_VERSION
        or plan.get("formal_eligible") is not False
        or plan.get("selected_split") != split
        or plan.get("test_frozen_access") is not False
        or plan.get("config_order") != configs
        or plan.get("query_count") != query_count
        or plan.get("shard_count") != shard_count
        or plan.get("instance_count") != instance_count
        or plan.get("evaluation_stages") != ["assistant", "gcs_v2"]
        or plan.get("assistant_checkpoint_schema_version") != 2
        or plan.get("pairwise_judge_enabled") is not False
        or plan.get("legacy_final_judge_enabled") is not False
        or plan.get("final_output_publication_enabled") is not False
        or plan.get("assistant_concurrency") != 2
        or plan.get("effective_process_assistant_concurrency") != 1
        or plan.get("final_judge_concurrency") != 0
        or plan.get("qwen_requests_per_minute_cap") != S1_QWEN_REQUESTS_PER_MINUTE_CAP
        or plan.get("qwen_rate_limit_policy") != QWEN_RATE_LIMIT_POLICY
    ):
        raise PortfolioS1ExperimentError("S1 launch identity or geometry drifted")
    selection_root = launch_root / "selection"
    if scope == S1_OPT_REPLAY_SCOPE:
        expected_selection_files = {"fold-manifest.json", "fold-mapping.jsonl"}
        selection_hashes = {
            "fold-manifest.json": plan.get("selection_manifest_file_sha256"),
            "fold-mapping.jsonl": plan.get("selection_mapping_file_sha256"),
        }
    else:
        expected_selection_files = {"validation-gates.json"}
        selection_hashes = {
            "validation-gates.json": plan.get("selection_gate_file_sha256")
        }
    if (
        not selection_root.is_dir()
        or {path.name for path in selection_root.iterdir() if path.is_file()}
        != expected_selection_files
        or any(path.is_dir() or path.is_symlink() for path in selection_root.iterdir())
        or any(
            sha256_bytes(
                read_stable_regular_file(
                    selection_root / filename, label=f"launch selection {filename}"
                )
            )
            != digest
            for filename, digest in selection_hashes.items()
        )
    ):
        raise PortfolioS1ExperimentError("S1 launch selection bytes drifted")
    instances_content, raw_instances = _canonical_rows(
        launch_root / "instances.jsonl", label="S1 launch instances"
    )
    if sha256_bytes(instances_content) != plan.get("instances_file_sha256"):
        raise PortfolioS1ExperimentError("S1 launch instances file drifted")
    try:
        instances = tuple(
            PortfolioLaunchInstance.model_validate(item, strict=True)
            for item in raw_instances
        )
        raw_shards = plan.get("shards")
        if not isinstance(raw_shards, list):
            raise ValueError("missing shards")
        shards = tuple(
            PortfolioLaunchShard.model_validate(item, strict=True)
            for item in raw_shards
        )
    except (ValidationError, ValueError) as error:
        raise PortfolioS1ExperimentError("S1 launch typed rows are invalid") from error
    if (
        len(instances) != instance_count
        or len(shards) != shard_count
        or tuple(item.shard_ordinal for item in shards) != tuple(range(shard_count))
        or {item.config for item in instances} != set(configs)
        or len({item.query_id for item in instances}) != query_count
        or any(
            item.accepted_batch_id not in plan["selected_batch_ids"]
            for item in instances
        )
        or any(item.query_count != 25 for item in shards)
        or any((launch_root / item.final_output_relpath).exists() for item in instances)
    ):
        raise PortfolioS1ExperimentError("S1 launch membership drifted")
    state_content, state = _canonical_object(
        launch_root / "launch-state.json", label="S1 launch state"
    )
    del state_content
    if (
        _self_hash(state, "state_sha256", label="S1 launch state")
        != state.get("state_sha256")
        or state.get("launch_plan_sha256") != plan["launch_plan_sha256"]
        or state.get("status") != "not_started"
        or state.get("model_calls_performed") != 0
    ):
        raise PortfolioS1ExperimentError("S1 launch state is not fresh")
    return VerifiedPortfolioS1ExperimentLaunch(
        root=launch_root,
        plan=plan,
        plan_file_sha256=expected_sha,
        instances=instances,
        shards=shards,
        _marker=_VERIFIED_LAUNCH,
    )


def require_verified_portfolio_s1_experiment_launch(
    value: object,
) -> VerifiedPortfolioS1ExperimentLaunch:
    if (
        type(value) is not VerifiedPortfolioS1ExperimentLaunch
        or value._marker is not _VERIFIED_LAUNCH
    ):
        raise TypeError("S1 execution requires a verified launch")
    return load_verified_portfolio_s1_experiment_launch(
        value.root, expected_plan_file_sha256=value.plan_file_sha256
    )


def create_portfolio_s1_experiment_execution_control(
    launch: VerifiedPortfolioS1ExperimentLaunch,
    runtime: VerifiedPortfolioS1ExperimentRuntime,
    *,
    output_dir: str | Path,
    approved_dashscope_budget_cny: Decimal,
    phase_cumulative_cap_cny: Decimal,
    prior_dashscope_observed_cost_cny: Decimal = Decimal("0"),
) -> dict[str, object]:
    verified_launch = require_verified_portfolio_s1_experiment_launch(launch)
    verified_runtime = require_verified_portfolio_s1_experiment_runtime(runtime)
    output = Path(output_dir).absolute()
    if output.exists():
        raise PortfolioS1ExperimentError("S1 execution-control output exists")
    quantum = Decimal("0.000000000001")
    values = (
        approved_dashscope_budget_cny,
        phase_cumulative_cap_cny,
        prior_dashscope_observed_cost_cny,
    )
    if any(
        type(value) is not Decimal or value != value.quantize(quantum)
        for value in values
    ):
        raise PortfolioS1ExperimentError(
            "S1 budget values require 12-place Decimal precision"
        )
    if not (
        approved_dashscope_budget_cny > 0
        and Decimal("0")
        <= prior_dashscope_observed_cost_cny
        < phase_cumulative_cap_cny
        <= approved_dashscope_budget_cny
    ):
        raise PortfolioS1ExperimentError("S1 execution budget bounds are invalid")
    lock = verified_runtime.runtime_lock
    plan = verified_launch.plan
    incremental = phase_cumulative_cap_cny - prior_dashscope_observed_cost_cny

    def text(value: Decimal) -> str:
        return format(value, ".12f")

    payload: dict[str, object] = {
        "schema_version": 4,
        "kind": S1_EXPERIMENT_CONTROL_KIND,
        "track": "portfolio",
        "formal_eligible": False,
        "status": "prepared_not_started",
        "execution_scope": plan["execution_mode"],
        "matrix_run_id": plan["matrix_run_id"],
        "launch_root": verified_launch.root.as_posix(),
        "launch_plan_file_sha256": verified_launch.plan_file_sha256,
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "runtime_root": verified_runtime.root.as_posix(),
        "runtime_lock_file_sha256": verified_runtime.runtime_lock_file_sha256,
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "core_input_binding_launch_root": plan["core_input_binding_launch_root"],
        "core_input_binding_launch_plan_file_sha256": plan[
            "core_input_binding_launch_plan_file_sha256"
        ],
        "evaluation_stages": ["assistant", "gcs_v2"],
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": GCS_V2_POLICY_VERSION,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "gcs_scorer_evidence_policy_version": (GCS_SCORER_EVIDENCE_V2_POLICY_VERSION),
        "execution_artifact_aliases": [],
        "execution_artifact_alias_provider_model_call_count": 0,
        "pairwise_judge_enabled": False,
        "legacy_final_judge_enabled": False,
        "final_output_publication_enabled": False,
        "analyzer_provider_call_count": 0,
        "assistant_concurrency": 2,
        "effective_process_assistant_concurrency": 1,
        "final_judge_concurrency": 0,
        "qwen_requests_per_minute_cap": S1_QWEN_REQUESTS_PER_MINUTE_CAP,
        "qwen_rate_limit_policy": QWEN_RATE_LIMIT_POLICY,
        "approved_dashscope_budget_cny": text(approved_dashscope_budget_cny),
        "phase_cumulative_cap_cny": text(phase_cumulative_cap_cny),
        "incremental_authorized_dashscope_budget_cny": text(incremental),
        "prior_dashscope_observed_cost_cny": text(prior_dashscope_observed_cost_cny),
        "budget_ledger_relpath": "budget-ledger",
        "budget_authority_relpath": "budget-ledger/budget-authority.json",
        "budget_authority_creation_policy": "create_only_on_first_execute",
        "portfolio_budget_policy_version": lock["portfolio_budget_policy_version"],
        "portfolio_budget_policy_sha256": lock["portfolio_budget_policy_sha256"],
        "provider_pricing_contract_version": lock["provider_pricing_contract_version"],
        "provider_pricing_contract_sha256": lock["provider_pricing_contract_sha256"],
        "portfolio_failure_policy_version": lock["portfolio_failure_policy_version"],
        "portfolio_circuit_breaker_threshold": lock[
            "portfolio_circuit_breaker_threshold"
        ],
        "portfolio_max_retryable_attempts_per_query": lock[
            "portfolio_max_retryable_attempts_per_query"
        ],
        "resume_policy": "create_only_shards_skip_only_verified_complete",
        "checkpoint_policy": "after_every_terminal_query_and_25_query_shard",
        "authorized_batch_ids": plan["selected_batch_ids"],
        "authorized_shard_ids": [item.shard_id for item in verified_launch.shards],
        "authorized_shards": [
            {
                "shard_id": item.shard_id,
                "shard_ordinal": item.shard_ordinal,
                "shard_sha256": item.shard_sha256,
                "config": item.config,
                "accepted_batch_id": item.accepted_batch_id,
                "query_count": item.query_count,
            }
            for item in verified_launch.shards
        ],
        "local_authorized_shard_count": len(verified_launch.shards),
        "local_authorized_instance_count": len(verified_launch.instances),
        "shard_count": plan["shard_count"],
        "instance_count": plan["instance_count"],
        "model_calls_performed": 0,
    }
    control = {
        **payload,
        "control_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }
    validate_portfolio_s1_experiment_execution_control(
        control, verified_launch, verified_runtime
    )
    output.mkdir(parents=True)
    (output / "execution-control.json").write_bytes(canonical_json_bytes(control))
    return control


def validate_portfolio_s1_experiment_execution_control(
    control: Mapping[str, object],
    launch: VerifiedPortfolioS1ExperimentLaunch,
    runtime: VerifiedPortfolioS1ExperimentRuntime,
) -> None:
    verified_launch = require_verified_portfolio_s1_experiment_launch(launch)
    verified_runtime = require_verified_portfolio_s1_experiment_runtime(runtime)
    _validate_portfolio_s1_experiment_control_binding(
        control, verified_launch, verified_runtime
    )


def validate_portfolio_s1_experiment_evidence_control(
    control: Mapping[str, object],
    launch: VerifiedPortfolioS1ExperimentLaunch,
    runtime: VerifiedPortfolioS1ExperimentRuntime,
) -> None:
    """Validate completed evidence without granting Assistant execution."""

    verified_launch = require_verified_portfolio_s1_experiment_launch(launch)
    verified_runtime = require_verified_portfolio_s1_experiment_runtime_evidence(
        runtime
    )
    if not isinstance(control, Mapping):
        raise PortfolioS1ExperimentError("S1 evidence control is not a mapping")
    if verified_runtime.verification_scope == _IMMUTABLE_EVIDENCE_SCOPE and (
        sha256_bytes(canonical_json_bytes(control))
        != HISTORICAL_S1_REPLAY_V5_CONTROL_FILE_SHA256
        or control.get("control_sha256") != _HISTORICAL_S1_REPLAY_V5_CONTROL_SELF_SHA256
    ):
        raise PortfolioS1ExperimentError(
            "immutable S1 evidence control identity is not recognized"
        )
    _validate_portfolio_s1_experiment_control_binding(
        control, verified_launch, verified_runtime
    )


def _validate_portfolio_s1_experiment_control_binding(
    control: Mapping[str, object],
    verified_launch: VerifiedPortfolioS1ExperimentLaunch,
    verified_runtime: VerifiedPortfolioS1ExperimentRuntime,
) -> None:
    plan = verified_launch.plan
    lock = verified_runtime.runtime_lock
    immutable_evidence = (
        verified_runtime.verification_scope == _IMMUTABLE_EVIDENCE_SCOPE
    )
    forbidden = {
        "rubric_path",
        "rubric_file_sha256",
        "rubric_content_sha256",
        "treatment_chain_manifest_file_sha256",
        "treatment_chain_sha256",
        "pairwise_policy_sha256",
        "test_frozen",
    }
    expected = {
        "kind": S1_EXPERIMENT_CONTROL_KIND,
        "execution_scope": plan["execution_mode"],
        "matrix_run_id": plan["matrix_run_id"],
        "launch_plan_file_sha256": verified_launch.plan_file_sha256,
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "runtime_lock_file_sha256": verified_runtime.runtime_lock_file_sha256,
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "core_input_binding_launch_root": plan["core_input_binding_launch_root"],
        "core_input_binding_launch_plan_file_sha256": plan[
            "core_input_binding_launch_plan_file_sha256"
        ],
        "evaluation_stages": ["assistant", "gcs_v2"],
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": lock["gcs_policy_version"],
        "gcs_policy_sha256": lock["gcs_policy_sha256"],
        "gcs_scorer_evidence_policy_version": lock[
            "gcs_scorer_evidence_policy_version"
        ],
        "execution_artifact_aliases": [],
        "execution_artifact_alias_provider_model_call_count": 0,
        "pairwise_judge_enabled": False,
        "legacy_final_judge_enabled": False,
        "final_output_publication_enabled": False,
        "analyzer_provider_call_count": 0,
        "assistant_concurrency": 2,
        "effective_process_assistant_concurrency": 1,
        "final_judge_concurrency": 0,
        "qwen_requests_per_minute_cap": (
            20 if immutable_evidence else S1_QWEN_REQUESTS_PER_MINUTE_CAP
        ),
        "qwen_rate_limit_policy": (
            "smooth_provider_call_start_v2"
            if immutable_evidence
            else QWEN_RATE_LIMIT_POLICY
        ),
        "authorized_batch_ids": plan["selected_batch_ids"],
        "authorized_shard_ids": [item.shard_id for item in verified_launch.shards],
        "local_authorized_shard_count": len(verified_launch.shards),
        "local_authorized_instance_count": len(verified_launch.instances),
        "shard_count": plan["shard_count"],
        "instance_count": plan["instance_count"],
        "model_calls_performed": 0,
    }
    if (
        not isinstance(control, Mapping)
        or any(control.get(name) != value for name, value in expected.items())
        or any(name in control for name in forbidden)
        or _self_hash(control, "control_sha256", label="S1 execution control")
        != control.get("control_sha256")
    ):
        raise PortfolioS1ExperimentError(
            "S1 execution control differs from its launch/runtime"
        )


__all__ = [
    "HISTORICAL_S1_REPLAY_V5_CONTROL_FILE_SHA256",
    "HISTORICAL_S1_RUNTIME_V5_FILE_SHA256",
    "PortfolioS1ExperimentError",
    "S1_BODY_GATE_SCOPE",
    "S1_EXPERIMENT_CONTROL_KIND",
    "S1_EXPERIMENT_LAUNCH_KIND",
    "S1_EXPERIMENT_LAUNCH_POLICY_VERSION",
    "S1_EXPERIMENT_RUNTIME_KIND",
    "S1_EXPERIMENT_RUNTIME_POLICY_VERSION",
    "S1_OPT_REPLAY_SCOPE",
    "S1_QWEN_REQUESTS_PER_MINUTE_CAP",
    "VerifiedPortfolioS1ExperimentLaunch",
    "VerifiedPortfolioS1ExperimentRuntime",
    "create_portfolio_s1_experiment_execution_control",
    "create_portfolio_s1_experiment_launch_package",
    "create_portfolio_s1_experiment_runtime",
    "load_verified_portfolio_s1_experiment_launch",
    "load_verified_portfolio_s1_experiment_runtime",
    "load_verified_portfolio_s1_experiment_runtime_evidence",
    "PortfolioS1ExperimentAssistantRunner",
    "require_verified_portfolio_s1_experiment_launch",
    "require_verified_portfolio_s1_experiment_runtime",
    "require_verified_portfolio_s1_experiment_runtime_evidence",
    "require_portfolio_s1_experiment_assistant_runner",
    "validate_portfolio_s1_experiment_evidence_control",
    "validate_portfolio_s1_experiment_execution_control",
]
