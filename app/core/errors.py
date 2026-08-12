from __future__ import annotations


class CommerceRAGError(Exception):
    """Base class for expected application failures."""


class DataValidationError(CommerceRAGError):
    """Raised when source data violates the frozen WANDS contract."""


class EmptyQueryError(CommerceRAGError):
    """Raised when a query is empty after normalization."""


class NoSearchTokensError(CommerceRAGError):
    """Raised when a query contains no searchable letters or numbers."""


class InvalidTopKError(CommerceRAGError):
    """Raised when top_k is outside the supported range."""


class IndexNotReadyError(CommerceRAGError):
    """Raised when the local search index is unavailable or incompatible."""


class ProductNotFoundError(CommerceRAGError):
    """Raised when a product ID is not present in the local catalog."""
