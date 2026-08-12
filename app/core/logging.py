from __future__ import annotations

import logging
from collections.abc import Iterable


class SecretRedactionFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str | None] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in self._secrets:
            message = message.replace(secret, "[REDACTED]")
        record.msg = message
        record.args = ()
        return True


def configure_logging(*, api_key: str | None = None) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(SecretRedactionFilter([api_key]))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
