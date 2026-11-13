"""Shared asynchronous MongoDB connection helpers for the tender flow."""

from __future__ import annotations

import os
from typing import Any

import certifi
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)


_motor_client: AsyncIOMotorClient | None = None


def get_motor_client() -> AsyncIOMotorClient:
    """Return the process-wide Motor client without performing blocking I/O."""

    global _motor_client
    if _motor_client is None:
        mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("MONGODB_URL")
        if not mongo_uri:
            raise ValueError("MONGO_URI, MONGODB_URI, or MONGODB_URL is required")

        client_options = {
            "serverSelectionTimeoutMS": int(
                os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "10000")
            ),
        }
        if os.getenv("MONGO_TLS", "false").lower() in {
            "1", "true", "yes", "on"
        }:
            client_options["tlsCAFile"] = certifi.where()

        _motor_client = AsyncIOMotorClient(mongo_uri, **client_options)
    return _motor_client


def get_motor_database() -> AsyncIOMotorDatabase:
    database_name = (
        os.getenv("MONGO_DB_NAME")
        or os.getenv("MONGODB_DATABASE")
        or os.getenv("DATABASE_NAME")
        or "DocAI"
    )
    return get_motor_client()[database_name]


def get_motor_collection(collection_name: str) -> AsyncIOMotorCollection:
    return get_motor_database()[collection_name]


def get_requirement_extraction_motor_collection() -> AsyncIOMotorCollection:
    return get_motor_collection(
        os.getenv(
            "REQUIREMENT_EXTRACTION_COLLECTION_NAME",
            "TenderRequirements",
        )
    )


def get_deduplication_motor_collection() -> AsyncIOMotorCollection:
    return get_motor_collection(
        os.getenv(
            "DEDUPLICATION_COLLECTION_NAME",
            "TenderRequirementDeduplications",
        )
    )


async def ping_motor() -> dict[str, Any]:
    """Verify the configured MongoDB connection."""

    result = await get_motor_client().admin.command("ping")
    return dict(result)


def close_motor_client() -> None:
    global _motor_client
    if _motor_client is not None:
        _motor_client.close()
        _motor_client = None


__all__ = [
    "close_motor_client",
    "get_motor_client",
    "get_motor_collection",
    "get_motor_database",
    "get_deduplication_motor_collection",
    "get_requirement_extraction_motor_collection",
    "ping_motor",
]
