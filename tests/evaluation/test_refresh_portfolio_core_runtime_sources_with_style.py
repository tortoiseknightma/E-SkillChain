from scripts import refresh_portfolio_core_runtime_sources_with_style as refresh
from skillchain.tools.portfolio_runtime import PORTFOLIO_TOOL_RUNTIME_POLICY
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _source_receipt() -> dict:
    return {
        "schema_version": 1,
        "kind": "portfolio-core-runtime-sources-receipt",
        "policy_version": "portfolio-core-runtime-sources-v1",
        "provider_call_count": 0,
        "formal_eligible": False,
        "auxiliary_inputs": {"fashioniq_captions": []},
        "outputs": {"rpc-scenes.jsonl": {"bytes": 1, "rows": 1, "sha256": "a" * 64}},
        "runtime_source_sha256s": ["b" * 64],
        "runtime_data_sha256": "c" * 64,
        "receipt_sha256": "d" * 64,
    }


def test_refresh_receipt_adds_only_graph_and_recomputes_runtime_identity(
    monkeypatch,
) -> None:
    graph = {
        "policy_version": "portfolio-style-coordination-graph-v1",
        "graph_sha256": "e" * 64,
        "edges": [{}] * 872,
    }
    graph_content = b"frozen-style-graph"
    graph_file_sha256 = sha256_bytes(graph_content)
    monkeypatch.setattr(
        refresh,
        "_output_descriptor",
        lambda _name, content: {
            "bytes": len(content),
            "rows": 872,
            "sha256": sha256_bytes(content),
        },
    )

    result = refresh._refresh_receipt(
        _source_receipt(),
        source_receipt_file_sha256="f" * 64,
        graph=graph,
        graph_content=graph_content,
    )

    assert result["provider_call_count"] == 0
    assert result["runtime_source_sha256s"] == ["b" * 64, graph_file_sha256]
    assert result["auxiliary_inputs"]["style_coordination_graph"] == {
        "bytes": len(graph_content),
        "edge_count": 872,
        "graph_sha256": "e" * 64,
        "output_path": "style-coordination-graph.json",
        "policy_version": "portfolio-style-coordination-graph-v1",
        "sha256": graph_file_sha256,
    }
    assert result["contract_refresh"] == {
        "kind": "portfolio-core-style-graph-contract-refresh",
        "source_receipt_file_sha256": "f" * 64,
        "style_graph_file_sha256": graph_file_sha256,
        "anchor_count": 130,
        "edge_count": 872,
        "candidate_count": 9,
        "provider_calls": 0,
    }
    assert result["runtime_data_sha256"] == sha256_bytes(
        canonical_json_bytes(
            {
                "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
                "source_sha256s": ["b" * 64, graph_file_sha256],
            }
        )
    )
    unsigned = dict(result)
    supplied = unsigned.pop("receipt_sha256")
    assert supplied == sha256_bytes(canonical_json_bytes(unsigned))


def test_refresh_receipt_rejects_graph_replacement(monkeypatch) -> None:
    source = _source_receipt()
    source["auxiliary_inputs"]["style_coordination_graph"] = {}
    monkeypatch.setattr(refresh, "_output_descriptor", lambda *_args: {})

    try:
        refresh._refresh_receipt(
            source,
            source_receipt_file_sha256="f" * 64,
            graph={"edges": []},
            graph_content=b"graph",
        )
    except ValueError as error:
        assert "already contains" in str(error)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("graph replacement was accepted")
