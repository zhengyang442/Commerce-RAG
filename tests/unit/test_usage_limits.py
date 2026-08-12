from __future__ import annotations

from datetime import date

import pytest

from app.core.usage_limits import DailyExternalCallBudget


@pytest.mark.anyio
async def test_daily_budget_resets_on_a_new_utc_day() -> None:
    current = date(2026, 8, 13)
    budget = DailyExternalCallBudget(2, today=lambda: current)

    assert await budget.try_acquire() is True
    assert await budget.try_acquire() is True
    assert await budget.try_acquire() is False
    assert budget.used == 2

    current = date(2026, 8, 14)

    assert await budget.try_acquire() is True
    assert budget.used == 1


def test_daily_budget_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="正整数"):
        DailyExternalCallBudget(0)
