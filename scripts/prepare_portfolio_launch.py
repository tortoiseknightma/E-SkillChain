"""Prepare a zero-call, 25-query-sharded Portfolio launch package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.evaluation.assistant_runs import AssistantRunConfig  # noqa: E402
from skillchain.evaluation.portfolio_inputs import (  # noqa: E402
    ACTIVE_PORTFOLIO_PROCESSOR_ORDER,
    CORE_PORTFOLIO_PROCESSOR_ORDER,
    ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER,
    PortfolioRemoteProcessingFiles,
    load_verified_portfolio_dev_mini_inputs,
)
from skillchain.evaluation.core_final_config import (  # noqa: E402
    load_core_final_framework_artifact,
)
from skillchain.evaluation.portfolio_core_inputs import (  # noqa: E402
    CORE_R3_MATERIALIZATION_MANIFEST_FILE_SHA256,
    CORE_SPLIT_ORDER,
    CoreSplit,
    PortfolioCoreInputFiles,
    load_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    AUTONOMOUS_DASHSCOPE_BUDGET_CNY,
    DEFAULT_PLANNING_CEILING_CNY,
    PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256,
    PORTFOLIO_ROLE_SELECTION_FILE_SHA256,
    PortfolioLaunchArtifactInputs,
    PortfolioLaunchError,
    build_portfolio_launch_plan,
    create_portfolio_launch_package,
    load_portfolio_launch_package,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.tools.serialization import (  # noqa: E402
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


DEFAULT_MATRIX_RUN_ID = "portfolio-dev-mini-200x5-v1"
DEFAULT_CORE_MATRIX_RUN_ID = "portfolio-core-r3-1500x5-v1"
DEFAULT_CORE_STATIC_OPT_RUN_ID = "portfolio-core-r3-static-opt-800x1-v1"
DEFAULT_CORE_FRAMEWORK = (
    REPOSITORY_ROOT / "specs" / "evaluation" / "core-final-evaluation-framework-v1.json"
)
PLAN_SHA256 = "5e3b0d67545c3d9a311df6c133c427564e4064174158eddf603b3b075e9fa6be"
PLAN_MANIFEST_FILE_SHA256 = (
    "59f63c873f49bc2610df9547ddd5933d9019d6c3eb6dd19b4874f6a89a3f953a"
)
ACCEPTED_LEDGER_SHA256 = (
    "79835fa73bfde60a005ea16480c1efc04635b3fee64f8a01bae893fe5cd4e69a"
)
QUERY_ARTIFACT_SHA256 = (
    "6f8eda4fe663733708d6e797c954e58f098d3938857c6f2b143e24e202437f03"
)
CAPABILITY_ASSIGNMENTS_SHA256 = (
    "b579c616251b1734cd0cf9da4dbc53f9e90dec3c06c42aaac39b60738df2ca2f"
)
SEED_SET_SHA256 = "767a10caa954890088efe85a3b5ba50628e0ab772bafdf06b6b43dc7159c3250"
BASE_CATALOG_SHA256 = "df17da7dd7ad7e1077c2e6d084a5e1e79516021979915c9881314b7815c28c96"
RUNTIME_CATALOG_SHA256 = (
    "d7f44371a03af0e54e6671012bb3ff54ba6181c0dbff3ca8d219da1a362284a8"
)
AUTHORIZATION_FILE_SHA256 = (
    "6454850abfed8a117b8552ea70729d3d7bfca6d80770a15e2bc42055a57da016"
)
RECEIPT_FILE_SHA256 = "6c6e208d17a4bd23bd2f587994a93e14f20437fd5aed698dbf8152262e30c462"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BANK_CONFIGS = {"llm_static", "s1", "s1s2", "full"}


def _artifact_argument(value: str) -> tuple[str, Path, str]:
    config_name, separator, remainder = value.partition("=")
    path_text, hash_separator, digest = remainder.rpartition("@")
    if (
        not separator
        or not hash_separator
        or config_name not in _BANK_CONFIGS
        or not path_text
        or not _SHA256.fullmatch(digest)
    ):
        raise argparse.ArgumentTypeError(
            "--bank must use CONFIG=PATH@SHA256 for llm_static, s1, s1s2, or full"
        )
    return config_name, Path(path_text), digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("dev_mini", "core"),
        default="dev_mini",
        help="Corpus profile. dev_mini remains the compatibility default.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("full_matrix", "static_opt_rollout"),
        default="full_matrix",
        help=(
            "Use static_opt_rollout only for the exact Core opt_pool × "
            "llm_static 800-row Assistant+GCS workload."
        ),
    )
    parser.add_argument("--matrix-run-id")
    parser.add_argument(
        "--core-framework",
        type=Path,
        default=DEFAULT_CORE_FRAMEWORK,
        help="Core-final framework containing externally hashed dataset bindings.",
    )
    parser.add_argument(
        "--core-artifact-root",
        type=Path,
        help="Filesystem root for the framework's skillchain-data bindings.",
    )
    parser.add_argument(
        "--core-asset-root",
        type=Path,
        help="Root beneath which Core catalog local_path values resolve.",
    )
    parser.add_argument(
        "--core-input-binding-launch-root",
        type=Path,
        help=(
            "Existing Core launch package carrying a typed, fail-closed input-file "
            "recipe. Supply with --core-input-binding-launch-plan-file-sha256 "
            "instead of the raw Core path/permission arguments."
        ),
    )
    parser.add_argument(
        "--core-input-binding-launch-plan-file-sha256",
        help="External launch-plan file SHA-256 for the Core input recipe.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("all", *CORE_SPLIT_ORDER),
        help="Core split(s) to schedule; default all. Repeat for multiple splits.",
    )
    parser.add_argument("--core-runtime-catalog-dir", type=Path)
    parser.add_argument("--core-runtime-catalog-sha256")
    parser.add_argument("--core-authorization", type=Path)
    parser.add_argument("--core-authorization-sha256")
    parser.add_argument("--core-receipt", type=Path)
    parser.add_argument("--core-receipt-sha256")
    parser.add_argument("--core-selection-manifest", type=Path)
    parser.add_argument("--core-dataset-assets", type=Path)
    parser.add_argument(
        "--assistant-runtime-lock",
        type=Path,
        help="Canonical Portfolio Assistant runtime lock.",
    )
    parser.add_argument(
        "--assistant-runtime-lock-sha256",
        help="External SHA-256 for --assistant-runtime-lock.",
    )
    parser.add_argument(
        "--treatment-chain-manifest",
        type=Path,
        help="Canonical matrix-ready treatment-chain manifest.",
    )
    parser.add_argument(
        "--treatment-chain-manifest-sha256",
        help="External SHA-256 for --treatment-chain-manifest.",
    )
    parser.add_argument(
        "--static-contract-refresh-receipt",
        type=Path,
        help="Canonical receipt for the one-Bank Static contract refresh.",
    )
    parser.add_argument(
        "--static-contract-refresh-receipt-sha256",
        help="External SHA-256 for --static-contract-refresh-receipt.",
    )
    parser.add_argument(
        "--bank",
        action="append",
        type=_artifact_argument,
        default=[],
        metavar="CONFIG=PATH@SHA256",
    )
    parser.add_argument(
        "--approved-dashscope-budget-cny",
        type=float,
        default=AUTONOMOUS_DASHSCOPE_BUDGET_CNY,
        help=(
            "Operator-approved run cap. The default is only the autonomous "
            "CNY 10 allowance and therefore does not authorize the full run."
        ),
    )
    parser.add_argument(
        "--planning-ceiling-cny",
        type=float,
        default=DEFAULT_PLANNING_CEILING_CNY,
        help="Evidence-based cumulative DashScope ceiling carried by the launch.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return exit 3 after publishing if any launch blocker remains.",
    )
    return parser


def _load_active_inputs():
    permission_root = (
        REPOSITORY_ROOT
        / "specs"
        / "data_sources"
        / "c2"
        / "portfolio-mini-remote-processing-v1"
    )
    clean_root = REPOSITORY_ROOT / "data" / "clean"
    remote_files = PortfolioRemoteProcessingFiles(
        authorization_file=permission_root / "owner-authorization-v3.json",
        expected_authorization_file_sha256=AUTHORIZATION_FILE_SHA256,
        receipt_file=permission_root / "catalog-v3-receipt-v3.json",
        expected_receipt_file_sha256=RECEIPT_FILE_SHA256,
        selection_manifest=clean_root / "query_images" / "selection-manifest.json",
        dataset_assets=clean_root / "query_images" / "dataset-assets.jsonl",
        base_catalog_dir=clean_root / "portfolio-mini-asset-catalog-v1",
        output_catalog_dir=clean_root / "portfolio-mini-asset-catalog-v3",
        asset_root=clean_root,
        processor_order=ACTIVE_PORTFOLIO_PROCESSOR_ORDER,
    )
    return load_verified_portfolio_dev_mini_inputs(
        queries_root=REPOSITORY_ROOT / "data" / "queries",
        plan_path=REPOSITORY_ROOT / "data" / "queries" / "plans" / "dev_mini.json",
        capability_assignments_path=(
            clean_root / "portfolio-mini-capability-assignments-v1.jsonl"
        ),
        expected_plan_sha256=PLAN_SHA256,
        expected_plan_manifest_file_sha256=PLAN_MANIFEST_FILE_SHA256,
        expected_accepted_ledger_sha256=ACCEPTED_LEDGER_SHA256,
        expected_query_artifact_sha256=QUERY_ARTIFACT_SHA256,
        expected_capability_assignments_sha256=CAPABILITY_ASSIGNMENTS_SHA256,
        expected_seed_set_sha256=SEED_SET_SHA256,
        expected_base_catalog_sha256=BASE_CATALOG_SHA256,
        expected_output_catalog_sha256=RUNTIME_CATALOG_SHA256,
        remote_files=remote_files,
    )


def _core_binding(framework, role: str):
    matches = [item for item in framework.artifact_bindings if item.role == role]
    if len(matches) != 1:
        raise ValueError(f"Core framework does not uniquely bind {role}")
    binding = matches[0]
    if (
        binding.status != "verified"
        or binding.path is None
        or binding.file_sha256 is None
        or binding.artifact_root_id != "skillchain-data"
    ):
        raise ValueError(f"Core framework binding is not verified: {role}")
    return binding


def _core_path(root: Path, binding) -> Path:
    relative = PurePosixPath(str(binding.path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Core framework contains an unsafe artifact path")
    return root.joinpath(*relative.parts)


def _core_splits(arguments: argparse.Namespace) -> tuple[CoreSplit, ...]:
    requested = tuple(arguments.split or ("all",))
    if "all" in requested:
        if requested != ("all",):
            raise ValueError("--split all cannot be combined with another split")
        return CORE_SPLIT_ORDER
    if len(set(requested)) != len(requested):
        raise ValueError("Core split selection contains duplicates")
    return tuple(split for split in CORE_SPLIT_ORDER if split in requested)


def _optional_core_remote_files(
    arguments: argparse.Namespace,
    *,
    base_catalog_dir: Path,
    asset_root: Path,
):
    values = (
        arguments.core_authorization,
        arguments.core_authorization_sha256,
        arguments.core_receipt,
        arguments.core_receipt_sha256,
        arguments.core_selection_manifest,
        arguments.core_dataset_assets,
        arguments.core_runtime_catalog_dir,
        arguments.core_runtime_catalog_sha256,
    )
    if not any(value is not None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "Core remote preflight requires authorization/receipt hashes, "
            "selection, dataset-assets, and runtime catalog together"
        )
    return PortfolioRemoteProcessingFiles(
        authorization_file=arguments.core_authorization,
        expected_authorization_file_sha256=arguments.core_authorization_sha256,
        receipt_file=arguments.core_receipt,
        expected_receipt_file_sha256=arguments.core_receipt_sha256,
        selection_manifest=arguments.core_selection_manifest,
        dataset_assets=arguments.core_dataset_assets,
        base_catalog_dir=base_catalog_dir,
        output_catalog_dir=arguments.core_runtime_catalog_dir,
        asset_root=asset_root,
        processor_order=(
            CORE_PORTFOLIO_PROCESSOR_ORDER
            if arguments.execution_mode == "static_opt_rollout"
            else ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER
        ),
    )


def _load_core_inputs(arguments: argparse.Namespace):
    binding_root = arguments.core_input_binding_launch_root
    binding_sha256 = arguments.core_input_binding_launch_plan_file_sha256
    if (binding_root is None) != (binding_sha256 is None):
        raise ValueError(
            "--core-input-binding-launch-root and its plan file SHA-256 "
            "must be supplied together"
        )
    if binding_sha256 is not None and not _SHA256.fullmatch(binding_sha256):
        raise ValueError("Core input-binding launch plan SHA-256 is invalid")
    if binding_root is not None:
        raw_core_options = {
            "--core-artifact-root": arguments.core_artifact_root,
            "--core-asset-root": arguments.core_asset_root,
            "--core-runtime-catalog-dir": arguments.core_runtime_catalog_dir,
            "--core-runtime-catalog-sha256": arguments.core_runtime_catalog_sha256,
            "--core-authorization": arguments.core_authorization,
            "--core-authorization-sha256": arguments.core_authorization_sha256,
            "--core-receipt": arguments.core_receipt,
            "--core-receipt-sha256": arguments.core_receipt_sha256,
            "--core-selection-manifest": arguments.core_selection_manifest,
            "--core-dataset-assets": arguments.core_dataset_assets,
        }
        supplied = [
            name for name, value in raw_core_options.items() if value is not None
        ]
        if supplied:
            raise ValueError(
                "Core input-binding launch cannot be mixed with raw Core inputs: "
                + ", ".join(supplied)
            )
        source = load_portfolio_launch_package(
            binding_root,
            expected_plan_file_sha256=binding_sha256,
            _allow_legacy_budget_contract=True,
        )
        return reconstruct_verified_portfolio_core_inputs(source.plan)

    if arguments.core_artifact_root is None or arguments.core_asset_root is None:
        raise ValueError(
            "--profile core requires --core-artifact-root and --core-asset-root"
        )
    framework = load_core_final_framework_artifact(arguments.core_framework)
    root = arguments.core_artifact_root.absolute()
    plan_binding = _core_binding(framework, "core_query_plan")
    pregen_binding = _core_binding(framework, "core_query_plan_manifest")
    query_binding = _core_binding(framework, "core_queries")
    assignment_binding = _core_binding(framework, "core_capability_assignments")
    catalog_manifest_binding = _core_binding(framework, "core_asset_catalog_manifest")
    plan_path = _core_path(root, plan_binding)
    pregen_path = _core_path(root, pregen_binding)
    query_path = _core_path(root, query_binding)
    assignment_path = _core_path(root, assignment_binding)
    catalog_manifest_path = _core_path(root, catalog_manifest_binding)
    catalog_manifest_bytes = read_stable_regular_file(
        catalog_manifest_path,
        label="Core asset catalog manifest",
        max_bytes=4 * 1024 * 1024,
    )
    if sha256_bytes(catalog_manifest_bytes) != catalog_manifest_binding.file_sha256:
        raise ValueError("Core framework catalog-manifest digest mismatch")
    catalog_manifest = parse_canonical_json(
        catalog_manifest_bytes, label="Core asset catalog manifest"
    )
    if not isinstance(catalog_manifest, dict):
        raise ValueError("Core asset catalog manifest is not one object")
    catalog_sha256 = catalog_manifest.get("catalog_sha256")
    if not isinstance(catalog_sha256, str) or not _SHA256.fullmatch(catalog_sha256):
        raise ValueError("Core asset catalog logical digest is invalid")

    pregen = parse_canonical_json(
        read_stable_regular_file(
            pregen_path,
            label="Core pre-generation manifest",
            max_bytes=4 * 1024 * 1024,
        ),
        label="Core pre-generation manifest",
    )
    if not isinstance(pregen, dict) or not isinstance(pregen.get("files"), list):
        raise ValueError("Core pre-generation manifest has an invalid shape")
    split_rows = [
        item
        for item in pregen["files"]
        if isinstance(item, dict)
        and item.get("relative_path") == "sidecars/final-splits.jsonl"
    ]
    if len(split_rows) != 1 or not isinstance(split_rows[0].get("sha256"), str):
        raise ValueError("Core pre-generation manifest lacks final-splits binding")
    split_path = plan_path.parents[1] / "sidecars" / "final-splits.jsonl"
    materialization_manifest_path = query_path.with_name("repair-manifest.json")
    runtime_catalog_dir = arguments.core_runtime_catalog_dir
    runtime_catalog_sha256 = arguments.core_runtime_catalog_sha256
    remote_files = _optional_core_remote_files(
        arguments,
        base_catalog_dir=catalog_manifest_path.parent,
        asset_root=arguments.core_asset_root.absolute(),
    )
    return load_verified_portfolio_core_inputs(
        PortfolioCoreInputFiles(
            plan_path=plan_path,
            expected_plan_sha256=plan_binding.file_sha256,
            pre_generation_manifest_path=pregen_path,
            expected_pre_generation_manifest_file_sha256=(pregen_binding.file_sha256),
            split_assignment_path=split_path,
            expected_split_assignment_sha256=split_rows[0]["sha256"],
            query_artifact_path=query_path,
            expected_query_artifact_sha256=query_binding.file_sha256,
            materialization_manifest_path=materialization_manifest_path,
            expected_materialization_manifest_file_sha256=(
                CORE_R3_MATERIALIZATION_MANIFEST_FILE_SHA256
            ),
            capability_assignments_path=assignment_path,
            expected_capability_assignments_sha256=(assignment_binding.file_sha256),
            base_catalog_dir=catalog_manifest_path.parent,
            expected_base_catalog_sha256=catalog_sha256,
            asset_root=arguments.core_asset_root,
            runtime_catalog_dir=runtime_catalog_dir,
            expected_runtime_catalog_sha256=runtime_catalog_sha256,
            remote_files=remote_files,
        )
    )


def _artifact_inputs(arguments: argparse.Namespace) -> PortfolioLaunchArtifactInputs:
    runtime_path = arguments.assistant_runtime_lock
    runtime_sha = arguments.assistant_runtime_lock_sha256
    if (runtime_path is None) != (runtime_sha is None):
        raise ValueError(
            "--assistant-runtime-lock and its SHA-256 must be supplied together"
        )
    if runtime_sha is not None and not _SHA256.fullmatch(runtime_sha):
        raise ValueError("Assistant runtime lock SHA-256 is invalid")
    chain_path = arguments.treatment_chain_manifest
    chain_sha = arguments.treatment_chain_manifest_sha256
    if (chain_path is None) != (chain_sha is None):
        raise ValueError(
            "--treatment-chain-manifest and its SHA-256 must be supplied together"
        )
    if chain_sha is not None and not _SHA256.fullmatch(chain_sha):
        raise ValueError("treatment-chain manifest SHA-256 is invalid")
    refresh_path = arguments.static_contract_refresh_receipt
    refresh_sha = arguments.static_contract_refresh_receipt_sha256
    if (refresh_path is None) != (refresh_sha is None):
        raise ValueError(
            "--static-contract-refresh-receipt and its SHA-256 must be "
            "supplied together"
        )
    if refresh_sha is not None and not _SHA256.fullmatch(refresh_sha):
        raise ValueError("Static contract-refresh receipt SHA-256 is invalid")
    paths: dict[AssistantRunConfig, Path] = {}
    hashes: dict[AssistantRunConfig, str] = {}
    for config_name, path, digest in arguments.bank:
        if config_name in paths:
            raise ValueError(f"duplicate --bank config: {config_name}")
        typed_config: AssistantRunConfig = config_name  # type: ignore[assignment]
        paths[typed_config] = path
        hashes[typed_config] = digest
    if arguments.execution_mode == "static_opt_rollout":
        if chain_path is not None:
            raise ValueError("static_opt_rollout forbids a treatment-chain manifest")
        if paths and set(paths) != {"llm_static"}:
            raise ValueError(
                "static_opt_rollout accepts no Bank or exactly one llm_static Bank"
            )
    elif refresh_path is not None:
        raise ValueError("Static contract-refresh receipt requires static_opt_rollout")
    return PortfolioLaunchArtifactInputs(
        assistant_runtime_lock_path=runtime_path,
        assistant_runtime_lock_file_sha256=runtime_sha,
        treatment_chain_manifest_path=chain_path,
        treatment_chain_manifest_file_sha256=chain_sha,
        static_contract_refresh_receipt_path=refresh_path,
        static_contract_refresh_receipt_file_sha256=refresh_sha,
        bank_paths=paths,
        bank_file_sha256s=hashes,
    )


def _reject_core_options_for_dev_mini(arguments: argparse.Namespace) -> None:
    core_only = {
        "--split": arguments.split,
        "--core-artifact-root": arguments.core_artifact_root,
        "--core-asset-root": arguments.core_asset_root,
        "--core-input-binding-launch-root": (arguments.core_input_binding_launch_root),
        "--core-input-binding-launch-plan-file-sha256": (
            arguments.core_input_binding_launch_plan_file_sha256
        ),
        "--core-runtime-catalog-dir": arguments.core_runtime_catalog_dir,
        "--core-runtime-catalog-sha256": arguments.core_runtime_catalog_sha256,
        "--core-authorization": arguments.core_authorization,
        "--core-authorization-sha256": arguments.core_authorization_sha256,
        "--core-receipt": arguments.core_receipt,
        "--core-receipt-sha256": arguments.core_receipt_sha256,
        "--core-selection-manifest": arguments.core_selection_manifest,
        "--core-dataset-assets": arguments.core_dataset_assets,
    }
    supplied = [name for name, value in core_only.items() if value is not None]
    if supplied:
        raise ValueError(
            "Core-only arguments require --profile core: " + ", ".join(supplied)
        )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.profile == "core":
            inputs = _load_core_inputs(arguments)
            selected_splits = _core_splits(arguments)
            if arguments.execution_mode == "static_opt_rollout" and selected_splits != (
                "opt_pool",
            ):
                raise ValueError("static_opt_rollout requires exactly --split opt_pool")
            matrix_run_id = arguments.matrix_run_id or (
                DEFAULT_CORE_STATIC_OPT_RUN_ID
                if arguments.execution_mode == "static_opt_rollout"
                else DEFAULT_CORE_MATRIX_RUN_ID
            )
            if arguments.execution_mode == "static_opt_rollout":
                role_selection_path = (
                    REPOSITORY_ROOT
                    / "specs"
                    / "authoring"
                    / "model-role-selection-v4.json"
                )
                role_selection_file_sha256 = PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256
            else:
                role_selection_path = (
                    REPOSITORY_ROOT
                    / "specs"
                    / "authoring"
                    / "model-role-selection-v7.json"
                )
                role_selection_file_sha256 = PORTFOLIO_ROLE_SELECTION_FILE_SHA256
        else:
            if arguments.execution_mode != "full_matrix":
                raise ValueError("static_opt_rollout requires --profile core")
            _reject_core_options_for_dev_mini(arguments)
            inputs = _load_active_inputs()
            selected_splits = None
            matrix_run_id = arguments.matrix_run_id or DEFAULT_MATRIX_RUN_ID
            role_selection_path = (
                REPOSITORY_ROOT / "specs" / "authoring" / "model-role-selection-v7.json"
            )
            role_selection_file_sha256 = PORTFOLIO_ROLE_SELECTION_FILE_SHA256
        plan, instances = build_portfolio_launch_plan(
            inputs,
            matrix_run_id=matrix_run_id,
            role_selection_path=role_selection_path,
            expected_role_selection_file_sha256=role_selection_file_sha256,
            artifacts=_artifact_inputs(arguments),
            operator_approved_dashscope_budget_cny=(
                arguments.approved_dashscope_budget_cny
            ),
            planning_ceiling_cny=arguments.planning_ceiling_cny,
            selected_splits=selected_splits,
            execution_mode=arguments.execution_mode,
        )
        created = create_portfolio_launch_package(
            arguments.output_dir,
            plan=plan,
            instances_bytes=instances,
        )
    except (
        FileExistsError,
        OSError,
        PortfolioLaunchError,
        TypeError,
        ValueError,
    ) as error:
        print(f"prepare-portfolio-launch: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "blockers": list(plan.blockers),
                "execution_authorized": plan.execution_authorized,
                "execution_mode": plan.execution_mode,
                "execution_ready": plan.execution_ready,
                "instance_count": plan.instance_count,
                "launch_plan_file_sha256": created.plan_file_sha256,
                "launch_plan_sha256": plan.launch_plan_sha256,
                "model_calls_performed": plan.model_calls_performed,
                "output_dir": str(created.root),
                "policy_version": plan.policy_version,
                "profile": arguments.profile,
                "query_count": plan.query_count,
                "selected_splits": list(plan.selected_splits),
                "shard_count": plan.shard_count,
                "status": plan.status,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if arguments.require_ready and not plan.execution_ready:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
