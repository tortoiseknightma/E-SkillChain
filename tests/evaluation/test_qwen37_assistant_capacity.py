from skillchain import config
from skillchain.evaluation.core_fast.live_adapter import _assistant_cost
from skillchain.evaluation.core_fast.models import Concurrency


def test_qwen37_assistant_capacity_profile_is_bound() -> None:
    capacity = Concurrency()

    assert config.ASSISTANT_MODEL == "qwen3.7-flash-2026-07-15"
    assert (capacity.assistant, capacity.assistant_requests_per_second) == (
        60,
        20.0,
    )


def test_qwen37_assistant_cost_uses_the_under_32k_tier() -> None:
    assert _assistant_cost(1_400, 30) == 0.000304


def test_qwen35_route_qualification_profile_and_cost_are_bound() -> None:
    capacity = Concurrency(
        assistant=config.QWEN35_ROUTE_QUALIFICATION_CONCURRENCY,
        assistant_requests_per_second=(
            config.QWEN35_ROUTE_QUALIFICATION_REQUESTS_PER_SECOND
        ),
    )

    assert (capacity.assistant, capacity.assistant_requests_per_second) == (16, 8.0)
    assert (
        _assistant_cost(
            1_400,
            30,
            config.QWEN35_ROUTE_QUALIFICATION_MODEL,
        )
        == 0.00034
    )
