import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne

from app.infrastructure.motor_database import get_motor_client


class TenderRequirementRepository:

    def __init__(self,mongo_uri: Optional[str] = None,database_name: Optional[str] = None) -> None:
        if mongo_uri or database_name:
            client_options = {"serverSelectionTimeoutMS": 5000}
            if os.getenv("MONGO_TLS", "false").lower() in {
                "1", "true", "yes", "on"
            }:
                client_options["tlsCAFile"] = certifi.where()
            self.client = AsyncIOMotorClient(
                mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
                **client_options,
            )
        else:
            self.client = get_motor_client()
        database = self.client[database_name or os.getenv("MONGODB_DATABASE","DocAI")]
        self.requirements_collection = database[os.getenv("REQUIREMENTS_COLLECTION","tender_requirements")]
        self.chunks_collection = database[os.getenv("CHUNKS_COLLECTION","tender_chunks")]

    async def ping(self) -> None:
        await self.client.admin.command("ping")

    # ==========================================================
    # Single Upsert
    # ==========================================================

    async def upsert_requirement(self,requirement: Dict[str, Any]) -> None:

        now = datetime.now(timezone.utc)
        payload = deepcopy(requirement)
        payload["UpdatedAt"] = now

        await self.requirements_collection.update_one(
            {
                "DocumentId": payload["DocumentId"],
                "RequirementId": payload["RequirementId"],
            },
            {
                "$set": payload,
                "$setOnInsert": {
                    "CreatedAt": now,
                },
            },
            upsert=True,
        )

    # ==========================================================
    # Bulk Upsert
    # ==========================================================

    async def bulk_upsert_requirements(self, requirements: List[Dict[str, Any]]) -> None:

        if not requirements:
            return

        now = datetime.now(timezone.utc)

        operations = []

        for requirement in requirements:

            payload = deepcopy(requirement)
            payload["UpdatedAt"] = now

            operations.append(
                UpdateOne(
                    {
                        "DocumentId": payload["DocumentId"],
                        "RequirementId": payload["RequirementId"],
                    },
                    {
                        "$set": payload,
                        "$setOnInsert": {
                            "CreatedAt": now,
                        },
                    },
                    upsert=True,
                )
            )

        if operations:
            await self.requirements_collection.bulk_write(
                operations,
                ordered=False,
            )

    # ==========================================================
    # Chunk Status
    # ==========================================================

    async def upsert_chunk_status(self,chunk_data: Dict[str, Any]) -> None:

        now = datetime.now(timezone.utc)
        payload = deepcopy(chunk_data)
        payload["UpdatedAt"] = now

        await self.chunks_collection.update_one(
            {
                "DocumentId": payload["DocumentId"],
                "ChunkId": payload["ChunkId"],
            },
            {
                "$set": payload,
                "$setOnInsert": {
                    "CreatedAt": now,
                },
            },
            upsert=True,
        )


# Create Object TenderRequirementRepository Class
requirement_repository = TenderRequirementRepository()
