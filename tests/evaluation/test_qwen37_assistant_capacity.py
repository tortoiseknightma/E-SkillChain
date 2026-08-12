import pytest

from skillchain import config
from skillchain.evaluation.core_fast.live_adapter import _assistant_cost
from skillchain.evaluation.core_fast.models import Concurrency
from skillchain.evaluation.core_fast.pacing import StartPacer


def test_qwen37_assistant_capacity_profile_is_bound() -> None:
    capacity = Concurrency()

    assert config.ASSISTANT_MODEL == "qwen3.7-flash-2026-07-15"
    assert (capacity.assistant, capacity.assistant_requests_per_second) == (60, 20.0)


def test_qwen37_assistant_cost_uses_the_under_32k_tier() -> None:
    assert _assistant_cost(1_400, 30) == 0.000304


def test_assistant_start_pacer_smooths_provider_calls() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    pacer = StartPacer(20.0, clock=lambda: now[0], sleeper=sleep)
    for _ in range(4):
        pacer.wait()

    assert sleeps == pytest.approx([0.05, 0.05, 0.05])
