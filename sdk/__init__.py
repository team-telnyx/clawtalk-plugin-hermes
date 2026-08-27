"""ClawTalk REST SDK - public surface.

Usage::

    from .sdk import ClawTalkClient, ApiError
"""

from .client import ClawTalkClient
from .endpoints import (
    ENDPOINTS,
    IMPLEMENTED_ENDPOINTS,
    READ_ENDPOINTS,
    UNIMPLEMENTED_ENDPOINTS,
    Endpoint,
    resolve,
)
from .errors import ApiError

__all__ = [
    "ENDPOINTS",
    "IMPLEMENTED_ENDPOINTS",
    "READ_ENDPOINTS",
    "UNIMPLEMENTED_ENDPOINTS",
    "ApiError",
    "ClawTalkClient",
    "Endpoint",
    "resolve",
]
