from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import certifi
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from typing import Any, Optional
class MongoDocumentStore:
    """Persist agent output through Motor without changing sync agent APIs."""

    def __init__(self, collection_name: str) -> None:
        mongo_uri = os.getenv("MONGO_URI")
        database_name = os.getenv("MONGO_DB_NAME")
        if not mongo_uri:
            raise ValueError("MONGO_URI is required for MongoDB persistence.")
        if not database_name:
            raise ValueError("MONGO_DB_NAME is required for MongoDB persistence.")
        client_options = {
            "serverSelectionTimeoutMS": int(
                os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "10000")
            ),
        }
        if os.getenv("MONGO_TLS", "false").lower() in {
            "1", "true", "yes", "on"
        }:
            client_options["tlsCAFile"] = certifi.where()

        self.client = AsyncIOMotorClient(mongo_uri, **client_options)
        self.collection = self.client[database_name][collection_name]

    def upsert(self, identity: dict[str, Any], document: dict[str, Any]) -> str:
        return asyncio.run(self._upsert(identity, document))

    async def _upsert(self, identity: dict[str, Any], document: dict[str, Any]) -> str:
        now = datetime.now(timezone.utc)
        update = dict(document)
        # CreatedAt is insert-only. Keeping it out of $set avoids MongoDB's
        # conflict when the input document already contains CreatedAt.
        update.pop("CreatedAt", None)
        update["UpdatedAt"] = now
        await self.collection.update_one(
            identity,
            {
                "$set": update,
                "$setOnInsert": {"CreatedAt": now},
            },
            upsert=True,
        )
        stored = await self.collection.find_one(identity, {"_id": 1})
        if not stored:
            raise RuntimeError("MongoDB upsert completed without a retrievable document.")
        return str(stored["_id"])

    def insert(self, document: dict[str, Any]) -> str:
        return asyncio.run(self._insert(document))

    async def _insert(self, document: dict[str, Any]) -> str:
        result = await self.collection.insert_one(dict(document))
        return str(result.inserted_id)

    def set_status(self, identity: dict[str, Any], status: str, upsert: bool = False) -> bool:
        return asyncio.run(self._set_status(identity, status, upsert))

    async def _set_status(self, identity: dict[str, Any], status: str, upsert: bool = False) -> bool:
        result = await self.collection.update_one(
            identity,
            {
                "$set": {
                    "Status": status,
                    "UpdatedAt": datetime.now(timezone.utc),
                }
            },
            upsert=upsert,
        )
        return bool(result.matched_count or result.upserted_id)

    def find_one(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        return asyncio.run(self._find_one(identity))

    async def _find_one(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        document = await self.collection.find_one(identity)
        if document and "_id" in document:
            document["_id"] = str(document["_id"])
        return document

    def find_latest(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        return asyncio.run(self._find_latest(identity))

    async def _find_latest(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        document = await self.collection.find_one(identity, sort=[("CreatedAt", -1), ("_id", -1)])
        if document and "_id" in document:
            document["_id"] = str(document["_id"])
        return document

    def close(self) -> None:
        self.client.close()
        


load_dotenv()

_mongo_client: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    global _mongo_client

    if _mongo_client is None:
        mongo_uri = (
            os.getenv("MONGO_URI")
            or os.getenv("MONGODB_URI")
        )

        if not mongo_uri:
            raise ValueError(
                "MONGO_URI is missing from the .env file"
            )

        client_options = {"serverSelectionTimeoutMS": 10000}
        if os.getenv("MONGO_TLS", "false").lower() in {
            "1", "true", "yes", "on"
        }:
            client_options["tlsCAFile"] = certifi.where()

        _mongo_client = MongoClient(mongo_uri, **client_options)

        # Verify the MongoDB connection.
        _mongo_client.admin.command("ping")

    return _mongo_client


def get_database() -> Database:
    database_name = (
        os.getenv("MONGO_DB_NAME")
        or os.getenv("MONGODB_DATABASE")
        or os.getenv("DATABASE_NAME")
    )

    if not database_name:
        raise ValueError(
            "MONGO_DB_NAME is missing from the .env file"
        )

    return get_mongo_client()[database_name]


def get_requirement_extraction_collection() -> Collection:
    """
    Source collection.

    Requirements/files are read from this collection.
    """

    collection_name = os.getenv(
        "REQUIREMENT_EXTRACTION_COLLECTION_NAME",
        "TenderRequirements",
    )

    return get_database()[collection_name]


def get_deduplication_collection() -> Collection:
    """
    Destination collection.

    Deduplication input and LLM output are saved here.
    """

    collection_name = os.getenv(
        "DEDUPLICATION_COLLECTION_NAME",
        "TenderRequirementDeduplications",
    )

    return get_database()[collection_name]


def close_mongo_connection() -> None:
    global _mongo_client

    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
