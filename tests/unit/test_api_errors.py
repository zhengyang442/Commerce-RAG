from __future__ import annotations

from app.api.errors import validation_error_handler


def test_validation_error_handler_is_available() -> None:
    assert callable(validation_error_handler)
