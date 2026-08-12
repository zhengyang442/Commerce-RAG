from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, date, datetime


class DailyExternalCallBudget:
    """Single-process UTC-day budget for real provider calls in the public demo."""

    def __init__(
        self,
        limit: int,
        *,
        today: Callable[[], date] | None = None,
    ) -> None:
        if limit <= 0:
            raise ValueError("每日外部调用额度必须是正整数")
        self.limit = limit
        self._today = today or (lambda: datetime.now(UTC).date())
        self._day = self._today()
        self._used = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            current_day = self._today()
            if current_day != self._day:
                self._day = current_day
                self._used = 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    @property
    def used(self) -> int:
        return self._used
