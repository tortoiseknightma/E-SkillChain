"""Assemble the create-only one-Bank runtime for Core Static opt800.

This command performs no provider calls.  It copies only the deterministic
Static contract-refresh outputs and the frozen Core runtime-source package,
writes the active system prompt, builds the content-addressed runtime lock,
publishes atomically, and reloads the complete package fail closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from skillchain.evaluation.portfolio_static_opt_runtime import (  # noqa: E402
    STATIC_OPT_BANK_FILE,
    STATIC_OPT_REFRESH_RECEIPT_FILE,
    STATIC_OPT_SEMANTIC_INPUT_FILE,
    STATIC_OPT_SYSTEM_PROMPT_FILE,
    build_portfolio_static_opt_runtime_lock,
    load_verified_portfolio_static_opt_runtime,
)
from skillchain.synthesis.store import (  # noqa: E402
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.portfolio_runtime import PORTFOLIO_SYSTEM_PROMPT  # noqa: E402
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_REFRESH_DRAFT_FILE = "contract-refresh-draft.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-root", type=Path, required=True)
    parser.add_argument("--bank-file-sha256", required=True)
    parser.add_argument("--semantic-authoring-input-file-sha256", required=True)
    parser.add_argument("--contract-refresh-draft-file-sha256", required=True)
    parser.add_argument(
        "--static-contract-refresh-receipt-file-sha256",
        required=True,
    )
    parser.add_argument("--core-runtime-sources-root", type=Path, required=True)
    parser.add_argument("--core-runtime-sources-receipt-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _verified_bytes(path: Path, digest: str, *, label: str) -> bytes:
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} external SHA-256 is invalid")
    content = read_stable_regular_file(path, label=label)
    if sha256_bytes(content) != digest:
        raise ValueError(f"{label} external digest mismatch")
    return content


def _copy_core_sources(source_root: Path, target_root: Path) -> None:
    source_root = source_root.resolve(strict=True)
    files = tuple(sorted(source_root.rglob("*")))
    if not files:
        raise ValueError("Core runtime-source package is empty")
    for source in files:
        if source.is_symlink():
            raise ValueError("Core runtime-source package contains a symlink")
        if not source.is_file():
            continue
        relative = PurePosixPath(source.relative_to(source_root).as_posix())
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Core runtime-source path is unsafe")
        target = target_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            read_stable_regular_file(
                source,
                label=f"Core runtime source {relative.as_posix()}",
            )
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        print("prepare-static-opt-runtime: output directory exists", file=sys.stderr)
        return 2
    staging: Path | None = None
    try:
        bank = _verified_bytes(
            args.refresh_root / STATIC_OPT_BANK_FILE,
            args.bank_file_sha256,
            label="refreshed Static Bank",
        )
        semantic = _verified_bytes(
            args.refresh_root / STATIC_OPT_SEMANTIC_INPUT_FILE,
            args.semantic_authoring_input_file_sha256,
            label="refreshed semantic AuthoringInput",
        )
        refresh_receipt = _verified_bytes(
            args.refresh_root / STATIC_OPT_REFRESH_RECEIPT_FILE,
            args.static_contract_refresh_receipt_file_sha256,
            label="Static contract-refresh receipt",
        )
        draft = _verified_bytes(
            args.refresh_root / _CONTRACT_REFRESH_DRAFT_FILE,
            args.contract_refresh_draft_file_sha256,
            label="Static contract-refresh draft",
        )
        refresh_value = parse_canonical_json(
            refresh_receipt,
            label="Static contract-refresh receipt",
        )
        if (
            not isinstance(refresh_value, dict)
            or refresh_value.get("new_draft_file") != _CONTRACT_REFRESH_DRAFT_FILE
            or refresh_value.get("new_draft_file_sha256") != sha256_bytes(draft)
        ):
            raise ValueError("Static contract-refresh draft differs from its receipt")
        _verified_bytes(
            args.core_runtime_sources_root / "receipt.json",
            args.core_runtime_sources_receipt_file_sha256,
            label="Core runtime-source receipt",
        )

        staging = new_staging_directory(args.output_dir)
        (staging / STATIC_OPT_BANK_FILE).write_bytes(bank)
        (staging / STATIC_OPT_SEMANTIC_INPUT_FILE).write_bytes(semantic)
        (staging / STATIC_OPT_REFRESH_RECEIPT_FILE).write_bytes(refresh_receipt)
        (staging / STATIC_OPT_SYSTEM_PROMPT_FILE).write_bytes(
            PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")
        )
        core_target = staging / "core-runtime-sources"
        core_target.mkdir()
        _copy_core_sources(args.core_runtime_sources_root, core_target)

        lock = build_portfolio_static_opt_runtime_lock(
            staging,
            active_contract={},
        )
        lock_content = canonical_json_bytes(lock)
        (staging / "runtime-lock.json").write_bytes(lock_content)
        atomic_publish_new_directory(staging, args.output_dir)
        staging = None

        runtime_lock_file_sha256 = sha256_bytes(lock_content)
        verified = load_verified_portfolio_static_opt_runtime(
            args.output_dir,
            expected_runtime_lock_file_sha256=runtime_lock_file_sha256,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir.resolve()),
                    "runtime_lock_file_sha256": runtime_lock_file_sha256,
                    "runtime_lock_sha256": verified.runtime_lock["runtime_lock_sha256"],
                    "bank_file_sha256": verified.bank_file_sha256,
                    "bank_sha256": verified.bank.bank_sha256,
                    "runtime_data_sha256": verified.runtime_lock["runtime_data_sha256"],
                    "provider_calls": 0,
                    "status": "verified_ready",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError) as error:
        print(f"prepare-static-opt-runtime: {error}", file=sys.stderr)
        return 2
    finally:
        if staging is not None and staging.exists():
            import shutil

            expected_parent = args.output_dir.absolute().parent.resolve(strict=True)
            if staging.resolve(
                strict=True
            ).parent != expected_parent or not staging.name.startswith(
                f".{args.output_dir.name}.staging-"
            ):
                raise RuntimeError("refusing to clean an unexpected staging directory")
            shutil.rmtree(staging)


if __name__ == "__main__":
    raise SystemExit(main())
