"""Shared MongoDB persistence for enterprise and token-usage logs."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from bson import ObjectId

from app.infrastructure.database import get_database
from app.infrastructure.motor_database import get_motor_database


DEFAULT_TOKEN_USAGE_COLLECTION = "TokenUsageLog"
DEFAULT_ENTERPRISE_LOG_COLLECTION = "EnterpriseLog"


def token_usage_collection_name() -> str:
    """Return the configured token-usage collection name."""

    return (
        os.getenv("TOKEN_USAGE_LOG_COLLECTION_NAME")
        or DEFAULT_TOKEN_USAGE_COLLECTION
    ).strip()


def enterprise_log_collection_name() -> str:
    """Return the configured enterprise-log collection name."""

    return (
        os.getenv("ENTERPRISE_LOG_COLLECTION_NAME")
        or DEFAULT_ENTERPRISE_LOG_COLLECTION
    ).strip()


def _mongo_value(value: Any) -> Any:
    """Convert arbitrary caller values into BSON-safe values."""

    if value is None or isinstance(
        value,
        (str, int, float, bool, datetime, ObjectId),
    ):
        return value

    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, (UUID, Enum)):
        return str(value.value if isinstance(value, Enum) else value)

    if is_dataclass(value) and not isinstance(value, type):
        return _mongo_value(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _mongo_value(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_mongo_value(item) for item in value]

    if isinstance(value, set):
        return [_mongo_value(item) for item in sorted(value, key=str)]

    if hasattr(value, "model_dump"):
        return _mongo_value(value.model_dump(mode="python"))

    return str(value)


def prepare_log_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a log payload and attach a server-side creation timestamp."""

    document = _mongo_value(dict(payload))
    document.setdefault("CreatedAt", datetime.now(timezone.utc))
    return document


async def insert_token_usage(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Insert one token-usage document through the shared Motor client."""

    collection_name = token_usage_collection_name()
    document = prepare_log_document(payload)
    result = await get_motor_database()[collection_name].insert_one(document)
    return {
        "success": True,
        "collection": collection_name,
        "inserted_id": str(result.inserted_id),
    }


def insert_enterprise_log(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Insert one enterprise-log document through the shared PyMongo client."""

    collection_name = enterprise_log_collection_name()
    document = prepare_log_document(payload)
    result = get_database()[collection_name].insert_one(document)
    return {
        "success": True,
        "collection": collection_name,
        "inserted_id": str(result.inserted_id),
    }


__all__ = [
    "DEFAULT_ENTERPRISE_LOG_COLLECTION",
    "DEFAULT_TOKEN_USAGE_COLLECTION",
    "enterprise_log_collection_name",
    "insert_enterprise_log",
    "insert_token_usage",
    "prepare_log_document",
    "token_usage_collection_name",
]
