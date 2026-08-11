"""Create a Core runtime-source revision by adding one frozen Style graph.

This is a zero-provider contract refresh.  It first reloads an existing
verified Core source package, preserves every prior output byte, validates the
graph against the same primary selection/catalog bytes, and publishes a new
create-only receipt and directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from skillchain.evaluation.portfolio_core_runtime_sources import (  # noqa: E402
    CORE_RUNTIME_SOURCE_RECEIPT,
    CORE_RUNTIME_STYLE_COORDINATION_GRAPH,
    _output_descriptor,
    _validate_style_coordination_graph,
    load_verified_portfolio_core_runtime_sources,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    load_portfolio_launch_package,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.synthesis.store import (  # noqa: E402
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_TOOL_RUNTIME_POLICY,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_ANCHORS = 130
_EXPECTED_EDGES = 872
_EXPECTED_CANDIDATES = 9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-launch-root", type=Path, required=True)
    parser.add_argument("--core-launch-plan-file-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--style-coordination-graph", type=Path, required=True)
    parser.add_argument("--style-coordination-graph-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _canonical_object(content: bytes, label: str) -> dict:
    value = parse_canonical_json(content, label=label)
    if not isinstance(value, dict) or canonical_json_bytes(value) != content:
        raise ValueError(f"{label} must be a canonical object")
    return value


def _refresh_receipt(
    source: dict,
    *,
    source_receipt_file_sha256: str,
    graph: dict,
    graph_content: bytes,
) -> dict:
    if "style_coordination_graph" in source.get("auxiliary_inputs", {}):
        raise ValueError("source runtime package already contains a Style graph")
    if CORE_RUNTIME_STYLE_COORDINATION_GRAPH in source.get("outputs", {}):
        raise ValueError("source runtime outputs already contain a Style graph")
    if source.get("provider_call_count") != 0:
        raise ValueError("source runtime package is not zero-provider")
    graph_sha256 = sha256_bytes(graph_content)
    graph_record = {
        "bytes": len(graph_content),
        "edge_count": len(graph["edges"]),
        "graph_sha256": graph["graph_sha256"],
        "output_path": CORE_RUNTIME_STYLE_COORDINATION_GRAPH,
        "policy_version": graph["policy_version"],
        "sha256": graph_sha256,
    }
    auxiliary = dict(source["auxiliary_inputs"])
    auxiliary["style_coordination_graph"] = graph_record
    outputs = dict(source["outputs"])
    outputs[CORE_RUNTIME_STYLE_COORDINATION_GRAPH] = _output_descriptor(
        CORE_RUNTIME_STYLE_COORDINATION_GRAPH,
        graph_content,
    )
    source_hashes = [*source["runtime_source_sha256s"], graph_sha256]
    unsigned = {key: value for key, value in source.items() if key != "receipt_sha256"}
    unsigned.update(
        {
            "auxiliary_inputs": auxiliary,
            "outputs": outputs,
            "runtime_source_sha256s": source_hashes,
            "runtime_data_sha256": sha256_bytes(
                canonical_json_bytes(
                    {
                        "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
                        "source_sha256s": source_hashes,
                    }
                )
            ),
            "contract_refresh": {
                "kind": "portfolio-core-style-graph-contract-refresh",
                "source_receipt_file_sha256": source_receipt_file_sha256,
                "style_graph_file_sha256": graph_sha256,
                "anchor_count": _EXPECTED_ANCHORS,
                "edge_count": _EXPECTED_EDGES,
                "candidate_count": _EXPECTED_CANDIDATES,
                "provider_calls": 0,
            },
        }
    )
    return {
        **unsigned,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    digests = (
        args.core_launch_plan_file_sha256,
        args.source_receipt_file_sha256,
        args.style_coordination_graph_file_sha256,
    )
    if any(_SHA256.fullmatch(item) is None for item in digests):
        print("refresh-core-style-sources: invalid external SHA-256", file=sys.stderr)
        return 2
    if args.output_dir.exists():
        print("refresh-core-style-sources: output directory exists", file=sys.stderr)
        return 2
    staging: Path | None = None
    try:
        launch = load_portfolio_launch_package(
            args.core_launch_root,
            expected_plan_file_sha256=args.core_launch_plan_file_sha256,
            _allow_legacy_budget_contract=True,
        )
        core_inputs = reconstruct_verified_portfolio_core_inputs(launch.plan)
        verified_source = load_verified_portfolio_core_runtime_sources(
            core_inputs,
            output_dir=args.source_root,
            expected_receipt_file_sha256=args.source_receipt_file_sha256,
            _allow_legacy_tool_runtime_policy=True,
        )
        graph_content = read_stable_regular_file(
            args.style_coordination_graph,
            label="Style coordination graph",
        )
        if sha256_bytes(graph_content) != args.style_coordination_graph_file_sha256:
            raise ValueError("Style coordination graph external digest mismatch")
        graph = _canonical_object(graph_content, "Style coordination graph")
        _validate_style_coordination_graph(graph)
        edges = graph["edges"]
        anchor_count = len({item["anchor"]["asset_id"] for item in edges})
        candidate_count = len({item["candidate"]["asset_id"] for item in edges})
        if (anchor_count, len(edges), candidate_count) != (
            _EXPECTED_ANCHORS,
            _EXPECTED_EDGES,
            _EXPECTED_CANDIDATES,
        ):
            raise ValueError("Style coordination graph frozen statistics drifted")
        sources = verified_source.sources
        bindings = graph["source_bindings"]
        if bindings["selection_manifest_sha256"] != sha256_bytes(
            read_stable_regular_file(
                sources.selection_manifest,
                label="verified Core selection manifest",
            )
        ) or bindings["runtime_catalog_assets_sha256"] != sha256_bytes(
            read_stable_regular_file(
                sources.runtime_catalog_assets,
                label="verified Core runtime catalog assets",
            )
        ):
            raise ValueError("Style graph differs from verified Core primary sources")
        source_receipt_content = read_stable_regular_file(
            verified_source.receipt_path,
            label="source Core runtime receipt",
        )
        source_receipt = _canonical_object(
            source_receipt_content,
            "source Core runtime receipt",
        )
        refreshed = _refresh_receipt(
            source_receipt,
            source_receipt_file_sha256=args.source_receipt_file_sha256,
            graph=graph,
            graph_content=graph_content,
        )
        staging = new_staging_directory(args.output_dir)
        for source_path in sorted(args.source_root.rglob("*")):
            if (
                not source_path.is_file()
                or source_path.name == CORE_RUNTIME_SOURCE_RECEIPT
            ):
                continue
            relative = source_path.relative_to(args.source_root)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                read_stable_regular_file(
                    source_path,
                    label=f"source Core runtime output {relative.as_posix()}",
                )
            )
        (staging / CORE_RUNTIME_STYLE_COORDINATION_GRAPH).write_bytes(graph_content)
        receipt_content = canonical_json_bytes(refreshed)
        (staging / CORE_RUNTIME_SOURCE_RECEIPT).write_bytes(receipt_content)
        atomic_publish_new_directory(staging, args.output_dir)
        staging = None
        receipt_file_sha256 = sha256_bytes(receipt_content)
        loaded = load_verified_portfolio_core_runtime_sources(
            core_inputs,
            output_dir=args.output_dir,
            expected_receipt_file_sha256=receipt_file_sha256,
        )
        if loaded.sources.style_coordination_graph is None:
            raise ValueError("refreshed runtime sources lost the Style graph")
        print(
            canonical_json_bytes(
                {
                    "output_dir": str(args.output_dir.resolve()),
                    "receipt_file_sha256": receipt_file_sha256,
                    "runtime_data_sha256": refreshed["runtime_data_sha256"],
                    "anchor_count": anchor_count,
                    "edge_count": len(edges),
                    "candidate_count": candidate_count,
                    "provider_calls": 0,
                }
            ).decode("utf-8")
        )
        return 0
    except Exception as error:
        print(f"refresh-core-style-sources: {error}", file=sys.stderr)
        return 2
    finally:
        if staging is not None and staging.exists():
            import shutil

            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
