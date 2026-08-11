"""Phase 3 确定性命令行入口；不承担正式语料创作。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from skillchain import config
from skillchain.data.asset_catalog import AssetCatalog, load_asset_catalog
from skillchain.schemas import Query
from skillchain.synthesis.batches import (
    accept_generated_batch,
    corpus_status,
    reject_generated_batch,
    show_next,
    stage_generated_batch,
)
from skillchain.synthesis.labeling import (
    apply_arbitration,
    pending_arbitrations,
    run_cross_review,
)
from skillchain.synthesis.planning import (
    activate_core_plan,
    activate_full_plan,
    activate_dev_plan,
    build_dev_mini_plan,
    discover_image_pools,
    extend_core_plan,
    extend_full_plan,
    load_capability_assignments,
    load_plan,
    read_active_plan,
    write_full_plan,
    write_core_plan,
    write_dev_mini_plan,
)
from skillchain.synthesis.seeds import (
    accept_seed_batch,
    reject_seed_batch,
    stage_seed_batch,
)
from skillchain.synthesis.splitting import (
    FrozenSplitError,
    freeze_test_split,
    load_labeled_queries_for_split,
    stratified_split,
    verify_frozen_split,
)
from skillchain.synthesis.store import (
    atomic_replace_file,
    canonical_jsonl_bytes,
)
from skillchain.taxonomy import capabilities_for_intent

_WRITE_COMMANDS = {
    "plan-dev-mini",
    "plan-core",
    "plan-full",
    "activate-plan",
    "stage-seeds",
    "accept-seeds",
    "reject-seeds",
    "stage-batch",
    "accept-batch",
    "reject-batch",
    "review-labels",
    "arbitrate",
    "split-full",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SkillChain Phase 3 corpus workflow")
    parser.add_argument("--queries-root", type=Path, default=config.QUERIES_DIR)
    parser.add_argument("--debug", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    guard = subparsers.add_parser("guard-model")
    guard.add_argument("--confirmed-model")

    plan = subparsers.add_parser("plan-dev-mini")
    plan.add_argument("--data-root", type=Path, required=True)
    plan.add_argument("--output", type=Path)
    plan.add_argument("--seed", type=int, default=20260711)
    plan.add_argument("--dry-run", action="store_true")
    plan.add_argument("--capability-assignments", type=Path)
    plan.add_argument("--expected-capability-assignments-sha256")
    _add_asset_catalog_gate(plan)

    for profile in ("core", "full"):
        derived_plan = subparsers.add_parser(f"plan-{profile}")
        derived_plan.add_argument("--data-root", type=Path, required=True)
        derived_plan.add_argument("--dev-plan", type=Path)
        derived_plan.add_argument("--output", type=Path)
        derived_plan.add_argument("--seed", type=int, default=20260711)
        derived_plan.add_argument("--dry-run", action="store_true")
        derived_plan.add_argument("--capability-assignments", type=Path)
        derived_plan.add_argument("--expected-capability-assignments-sha256")
        _add_asset_catalog_gate(derived_plan)

    activate = subparsers.add_parser("activate-plan")
    activate.add_argument("scope", choices=("dev_mini", "core", "full"))
    activate.add_argument("--plan", type=Path, required=True)
    _add_asset_catalog_gate(activate)

    for name in ("status", "show-next", "stats"):
        command = subparsers.add_parser(name)
        command.add_argument("--json", action="store_true")

    stage_seeds = subparsers.add_parser("stage-seeds")
    stage_seeds.add_argument("--input", type=Path, required=True)
    stage_seeds.add_argument("--seed-batch-id", required=True)

    accept_seeds = subparsers.add_parser("accept-seeds")
    accept_seeds.add_argument("--seed-batch-id", required=True)
    accept_seeds.add_argument("--confirmation", required=True)

    reject_seeds = subparsers.add_parser("reject-seeds")
    reject_seeds.add_argument("--seed-batch-id", required=True)
    reject_seeds.add_argument("--reason", required=True)

    stage = subparsers.add_parser("stage-batch")
    stage.add_argument("--draft", type=Path, required=True)
    stage.add_argument("--draft-manifest", type=Path, required=True)
    stage.add_argument("--base-batch-id", required=True)
    _add_asset_catalog_gate(stage)

    accept = subparsers.add_parser("accept-batch")
    accept.add_argument("--batch-id", required=True)
    accept.add_argument("--confirmation", required=True)
    _add_asset_catalog_gate(accept)

    reject = subparsers.add_parser("reject-batch")
    reject.add_argument("--batch-id", required=True)
    reject.add_argument("--reason", required=True)

    review = subparsers.add_parser("review-labels")
    review.add_argument("--accepted-source", type=Path)
    review.add_argument("--output-dir", type=Path)
    review.add_argument("--image-root", type=Path)
    review.add_argument("--plan", type=Path)
    _add_asset_catalog_gate(review)

    arbitrate = subparsers.add_parser("arbitrate")
    arbitrate.add_argument("--labels-dir", type=Path)
    arbitrate.add_argument("--image-root", type=Path)
    arbitrate.add_argument("--open-image", action="store_true")
    _add_asset_catalog_gate(arbitrate)

    split_full = subparsers.add_parser("split-full")
    split_full.add_argument("--plan", type=Path)
    split_full.add_argument("--seed", type=int, default=20260711)
    _add_asset_catalog_gate(split_full)
    return parser


def _add_asset_catalog_gate(parser: argparse.ArgumentParser) -> None:
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--asset-catalog", type=Path)
    gate.add_argument(
        "--allow-provisional-asset-groups",
        action="store_true",
        help="debug/tests only: use path-derived identities without a catalog",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="root used to verify catalog local_path (defaults to clean data root)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, FrozenSplitError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        if args.debug:
            raise
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    root: Path = args.queries_root
    asset_catalog = _load_asset_catalog_for_args(args)
    allow_provisional = bool(getattr(args, "allow_provisional_asset_groups", False))
    if args.command in _WRITE_COMMANDS:
        verify_frozen_split(
            root,
            asset_catalog=asset_catalog,
            full_plan_path=(
                getattr(args, "plan", None) or root / "plans" / "full.json"
            ),
            allow_provisional_asset_groups=allow_provisional,
        )
    if args.command == "guard-model":
        if args.confirmed_model != "5.6 Sol Ultra":
            raise ValueError("请先在 Codex UI 选择并明确确认 5.6 Sol Ultra")
        return 0

    if args.command == "plan-dev-mini":
        if (args.capability_assignments is None) != (
            args.expected_capability_assignments_sha256 is None
        ):
            raise ValueError(
                "--capability-assignments and its external expected SHA-256 "
                "must be provided together"
            )
        capability_assignments = (
            load_capability_assignments(
                args.capability_assignments,
                expected_sha256=args.expected_capability_assignments_sha256,
            )
            if args.capability_assignments is not None
            else None
        )
        plan = build_dev_mini_plan(
            args.data_root,
            seed=args.seed,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional,
            capability_assignments=capability_assignments,
            expected_capability_assignments_sha256=(
                args.expected_capability_assignments_sha256
            ),
        )
        summary = {
            "count": len(plan.queries),
            "boundary_count": sum(item.is_boundary for item in plan.queries),
            "batches": len({item.batch_id for item in plan.queries}),
            "asset_catalog_sha256": plan.asset_catalog_sha256,
            "capability_assignments_sha256": plan.capability_assignments_sha256,
            "leakage_policy_version": plan.leakage_policy_version,
        }
        if args.dry_run:
            _print(summary)
            return 0
        output = args.output or root / "plans" / "dev_mini.json"
        plan_path, manifest_path = write_dev_mini_plan(plan, output)
        pointer = activate_dev_plan(
            plan_path,
            root,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional,
        )
        _print(
            {
                **summary,
                "plan_path": str(plan_path),
                "manifest_path": str(manifest_path),
                "plan_sha256": pointer.plan_sha256,
            }
        )
        return 0

    if args.command in {"plan-core", "plan-full"}:
        profile = args.command.removeprefix("plan-")
        if (args.capability_assignments is None) != (
            args.expected_capability_assignments_sha256 is None
        ):
            raise ValueError(
                "--capability-assignments and its external expected SHA-256 "
                "must be provided together"
            )
        capability_assignments = (
            load_capability_assignments(
                args.capability_assignments,
                expected_sha256=args.expected_capability_assignments_sha256,
            )
            if args.capability_assignments is not None
            else None
        )
        dev_path = args.dev_plan or root / "plans" / "dev_mini.json"
        dev_plan, dev_manifest = load_plan(dev_path)
        if dev_plan.scope != "dev_mini":
            raise ValueError(f"plan-{profile} 的 --dev-plan 必须是 dev_mini")
        pools = discover_image_pools(args.data_root)
        derived_plan = (extend_core_plan if profile == "core" else extend_full_plan)(
            dev_plan,
            pools,
            seed=args.seed,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional,
            capability_assignments=capability_assignments,
            expected_capability_assignments_sha256=(
                args.expected_capability_assignments_sha256
            ),
        )
        summary = {
            "profile": profile,
            "count": len(derived_plan.queries),
            "boundary_count": sum(
                item.is_boundary for item in derived_plan.queries
            ),
            "batches": len({item.batch_id for item in derived_plan.queries}),
            "parent_plan_sha256": dev_manifest.plan_sha256,
            "asset_catalog_sha256": derived_plan.asset_catalog_sha256,
            "capability_assignments_sha256": (
                derived_plan.capability_assignments_sha256
            ),
            "leakage_policy_version": derived_plan.leakage_policy_version,
        }
        if args.dry_run:
            _print(summary)
            return 0
        output = args.output or root / "plans" / f"{profile}.json"
        writer = write_core_plan if profile == "core" else write_full_plan
        plan_path, manifest_path = writer(
            derived_plan,
            output,
            parent_plan_sha256=dev_manifest.plan_sha256,
        )
        _print(
            {
                **summary,
                "plan_path": str(plan_path),
                "manifest_path": str(manifest_path),
            }
        )
        return 0

    if args.command == "activate-plan":
        pointer = (
            activate_dev_plan(
                args.plan,
                root,
                asset_catalog=asset_catalog,
                allow_provisional_asset_groups=allow_provisional,
            )
            if args.scope == "dev_mini"
            else (
                activate_core_plan(
                    root,
                    args.plan,
                    asset_catalog=asset_catalog,
                    allow_provisional_asset_groups=allow_provisional,
                )
                if args.scope == "core"
                else activate_full_plan(
                root,
                args.plan,
                asset_catalog=asset_catalog,
                allow_provisional_asset_groups=allow_provisional,
                )
            )
        )
        _print(pointer.model_dump(mode="json"))
        return 0

    if args.command == "status":
        _print(corpus_status(root))
        return 0
    if args.command == "show-next":
        _print(show_next(root))
        return 0
    if args.command == "stats":
        _print(_stats(root))
        return 0

    if args.command == "stage-seeds":
        path = stage_seed_batch(args.input, root, args.seed_batch_id)
        _print({"path": str(path), "seed_batch_id": args.seed_batch_id})
        return 0
    if args.command == "accept-seeds":
        path = accept_seed_batch(
            root, args.seed_batch_id, confirmation=args.confirmation
        )
        _print({"path": str(path), "seed_batch_id": args.seed_batch_id})
        return 0
    if args.command == "reject-seeds":
        path = reject_seed_batch(root, args.seed_batch_id, reason=args.reason)
        _print({"path": str(path), "seed_batch_id": args.seed_batch_id})
        return 0

    if args.command == "review-labels":
        source = args.accepted_source or root / "queries.jsonl"
        output_dir = args.output_dir or root / "labels"
        review_plan = args.plan or root / "plans" / "full.json"
        if allow_provisional and args.plan is None and not review_plan.is_file():
            review_plan = None
        path = run_cross_review(
            source,
            output_dir=output_dir,
            image_root=args.image_root,
            plan_path=review_plan,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional,
        )
        _print(
            {
                "path": str(path),
                "pending_arbitrations": len(pending_arbitrations(path)),
            }
        )
        return 0
    if args.command == "arbitrate":
        return _run_arbitration_cli(
            args,
            root,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional,
        )
    if args.command == "split-full":
        plan_path = args.plan or root / "plans" / "full.json"
        _, full_manifest = load_plan(plan_path)
        candidates, locked_ids = load_labeled_queries_for_split(
            root,
            plan_path,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional,
        )
        assigned = stratified_split(
            candidates,
            locked_dev_query_ids=locked_ids,
            seed=args.seed,
        )
        test_path, test_manifest_path = freeze_test_split(
            root,
            assigned,
            seed=args.seed,
            full_plan_path=plan_path,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional,
        )
        labeled_path = root / "labels" / "labeled_queries.jsonl"
        atomic_replace_file(labeled_path, canonical_jsonl_bytes(assigned))
        split_counts = Counter(query.split for query in assigned)
        frozen_manifest = verify_frozen_split(
            root,
            asset_catalog=asset_catalog,
            full_plan_path=plan_path,
            expected_full_plan_sha256=full_manifest.plan_sha256,
            allow_provisional_asset_groups=allow_provisional,
        )
        if frozen_manifest is None:  # pragma: no cover - freeze contract guard
            raise FrozenSplitError("冻结 manifest 未生成")
        _print(
            {
                "labeled_queries": str(labeled_path),
                "split_assignment": str(root / "split_assignment.jsonl"),
                "test_frozen": str(test_path),
                "test_frozen_manifest": str(test_manifest_path),
                "profile": frozen_manifest.profile,
                "split_counts": dict(sorted(split_counts.items())),
                "assignment_sha256": frozen_manifest.assignment_sha256,
                "full_plan_sha256": frozen_manifest.full_plan_sha256,
                "asset_catalog_sha256": frozen_manifest.asset_catalog_sha256,
                "leakage_policy_version": frozen_manifest.leakage_policy_version,
                "group_component_count": frozen_manifest.component_count,
                "leakage_violations": frozen_manifest.leakage_violations,
            }
        )
        return 0

    plan_path = read_active_plan(root)[0]
    if args.command == "stage-batch":
        path = stage_generated_batch(
            draft_path=args.draft,
            draft_manifest_path=args.draft_manifest,
            plan_path=plan_path,
            queries_root=root,
            base_batch_id=args.base_batch_id,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional,
        )
        _print({"path": str(path), "batch_id": path.name})
        return 0
    if args.command == "accept-batch":
        path = accept_generated_batch(
            root,
            args.batch_id,
            confirmation=args.confirmation,
            plan_path=plan_path,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional,
        )
        _print({"path": str(path), "batch_id": path.name})
        return 0
    if args.command == "reject-batch":
        path = reject_generated_batch(root, args.batch_id, reason=args.reason)
        _print({"path": str(path), "batch_id": path.name})
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _load_asset_catalog_for_args(args: argparse.Namespace) -> AssetCatalog | None:
    catalog_path = getattr(args, "asset_catalog", None)
    if catalog_path is None:
        return None
    asset_root = getattr(args, "asset_root", None)
    if asset_root is None:
        data_root = getattr(args, "data_root", None)
        asset_root = (
            Path(data_root).parent
            if data_root is not None
            else config.DATA_DIR / "clean"
        )
    verify_full_catalog = args.command not in {"stage-batch", "accept-batch"}
    return load_asset_catalog(
        catalog_path,
        asset_root,
        verify_files=verify_full_catalog,
    )


def _stats(root: Path) -> dict:
    path = root / "queries.jsonl"
    queries: list[Query] = []
    if path.exists():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                queries.append(Query.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(
                    f"queries.jsonl 第 {line_number} 行校验失败: {exc}"
                ) from exc
    cross = Counter((q.split, q.canonical_intent, q.is_boundary) for q in queries)
    labels = Counter(q.label_status for q in queries)
    return {
        "total": len(queries),
        "split_intent_boundary": [
            {
                "split": split,
                "intent": intent,
                "is_boundary": boundary,
                "count": count,
            }
            for (split, intent, boundary), count in sorted(cross.items())
        ],
        "label_status": dict(sorted(labels.items())),
        "workflow": corpus_status(root),
    }


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _run_arbitration_cli(
    args: argparse.Namespace,
    root: Path,
    *,
    asset_catalog: AssetCatalog | None,
    allow_provisional_asset_groups: bool,
) -> int:
    labels_dir = args.labels_dir or root / "labels"
    image_root = (args.image_root or config.DATA_DIR / "clean").resolve()
    choices = {
        "exact_match",
        "multi_product",
        "divergent_rec",
        "encyclopedia",
        "utility",
    }
    completed = 0
    for item in pending_arbitrations(labels_dir):
        raw_image_path = Path(item.image_path)
        image_path = (
            raw_image_path.resolve()
            if raw_image_path.is_absolute()
            else (image_root / raw_image_path).resolve()
        )
        _print(
            {
                "query_id": item.query_id,
                "image_path": str(image_path),
                "text": item.text,
                "constructed_intent": item.constructed_intent,
                "constructed_capability": item.constructed_capability,
                "constructed_acceptable_capabilities": (
                    item.constructed_acceptable_capabilities
                ),
                "constructed_requires_card": item.constructed_requires_card,
                "qwen_intent": item.qwen_intent,
                "qwen_reason": item.qwen_reason,
                "qwen_review_error": item.qwen_review_error,
                "deepseek_intent": item.deepseek_intent,
                "deepseek_reason": item.deepseek_reason,
                "deepseek_review_error": item.deepseek_review_error,
                "spot_check": item.spot_check,
            }
        )
        if args.open_image:
            from PIL import Image

            Image.open(image_path).show()
        while True:
            answer = input(
                "intent [exact_match/multi_product/divergent_rec/"
                "encyclopedia/utility] 或 q 退出: "
            ).strip()
            if answer.lower() == "q":
                _print(
                    {
                        "arbitrated": completed,
                        "remaining": len(pending_arbitrations(labels_dir)),
                    }
                )
                return 0
            if answer in choices:
                capability_choice = _prompt_arbitration_capability(answer)
                if capability_choice is None:
                    _print(
                        {
                            "arbitrated": completed,
                            "remaining": len(pending_arbitrations(labels_dir)),
                        }
                    )
                    return 0
                canonical_capability, acceptable_capabilities = capability_choice
                apply_arbitration(
                    labels_dir,
                    query_id=item.query_id,
                    intent=answer,
                    canonical_capability=canonical_capability,
                    acceptable_capabilities=acceptable_capabilities,
                    reason="interactive CLI arbitration",
                    asset_catalog=asset_catalog,
                    allow_provisional_asset_groups=allow_provisional_asset_groups,
                )
                completed += 1
                break
            print("无效 intent；请输入五个合法值之一或 q。", file=sys.stderr)
    _print({"arbitrated": completed, "remaining": 0})
    return 0


def _prompt_arbitration_capability(
    intent: str,
) -> tuple[str, list[str] | None] | None:
    available = tuple(
        capability.capability_id for capability in capabilities_for_intent(intent)
    )
    if len(available) == 1:
        return available[0], None

    choices = set(available)
    while True:
        answer = input(
            f"canonical capability [{'/'.join(available)}] or q to quit: "
        ).strip()
        if answer.lower() == "q":
            return None
        if answer not in choices:
            print("invalid canonical capability for selected intent", file=sys.stderr)
            continue
        acceptable_raw = input(
            "acceptable capabilities (comma-separated; blank means canonical only): "
        ).strip()
        acceptable = (
            [answer]
            if not acceptable_raw
            else sorted({value.strip() for value in acceptable_raw.split(",")})
        )
        if (
            answer not in acceptable
            or not set(acceptable) <= choices
            or "" in acceptable
        ):
            print(
                "acceptable capabilities must include canonical and stay within intent",
                file=sys.stderr,
            )
            continue
        return answer, acceptable


if __name__ == "__main__":
    raise SystemExit(main())
