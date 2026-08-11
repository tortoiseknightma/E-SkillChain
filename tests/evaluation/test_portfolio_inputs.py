from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import prepare_portfolio_smoke_matrix
from skillchain.evaluation.portfolio_inputs import (
    PORTFOLIO_PROCESSOR_ORDER,
    require_verified_portfolio_dev_mini_inputs,
)
from skillchain.synthesis.planning import DEV_MINI_CAPABILITY_COUNTS


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_HAS_LOCAL_PORTFOLIO = (
    _REPOSITORY_ROOT / "data" / "queries" / "queries.jsonl"
).is_file()


@pytest.mark.skipif(
    not _HAS_LOCAL_PORTFOLIO,
    reason="owner-reviewed Portfolio artifacts are not present",
)
def test_current_portfolio_inputs_load_with_exact_external_commitments() -> None:
    inputs = prepare_portfolio_smoke_matrix.load_current_portfolio_inputs()

    assert len(inputs.queries) == 200
    assert len(inputs.ledger) == 8
    assert all(item.count == 25 for item in inputs.ledger)
    assert Counter(
        item.canonical_capability for item in inputs.queries
    ) == Counter(DEV_MINI_CAPABILITY_COUNTS)
    assert tuple(item.processor for item in inputs.remote_runtimes) == (
        PORTFOLIO_PROCESSOR_ORDER
    )
    assert inputs.formal_eligible is False

    query = next(item for item in inputs.queries if item.query_id == "dm-019")
    asset = next(
        item for item in inputs.query_assets if item.query_id == "dm-019"
    )
    assistant_query = next(
        item for item in inputs.assistant_queries if item.query_id == "dm-019"
    )
    assert query.text == "上面写的什么？"
    assert query.canonical_capability == "utility.document_reading"
    assert asset.image_path == "query_images/utility/document-0021.jpg"
    assert (
        asset.image_sha256
        == "e240c2ff6b95c29ac81fe7f56a95d46c497431fcecb12d07736369c64c7c9134"
    )
    assert assistant_query.query_id == query.query_id

    with pytest.raises(TypeError, match="external-digest loader"):
        require_verified_portfolio_dev_mini_inputs(
            replace(inputs, formal_eligible=True)
        )
