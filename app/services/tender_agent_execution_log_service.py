"""Persistence for individual tender-agent execution attempts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorCollection

from app.infrastructure.motor_database import get_motor_collection


class TenderAgentExecutionLogService:
    """Store one MongoDB document for each agent execution attempt."""

    COLLECTION_NAME = "TenderAgentExecutionLogs"

    def __init__(
        self,
        collection: AsyncIOMotorCollection | None = None,
    ) -> None:
        self.collection = collection or get_motor_collection(
            self.COLLECTION_NAME
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    async def start(
        self,
        *,
        job_id: str,
        run_id: str,
        company_id: str,
        tender_id: str,
        project_id: str,
        user_id: str,
        user_name: str,
        agent_name: str,
        step: str,
        attempt: int,
    ) -> str:
        execution_id = str(uuid4())
        now = self._now()
        await self.collection.insert_one(
            {
                "ExecutionId": execution_id,
                "JobId": job_id,
                "RunId": run_id,
                "CompanyId": company_id,
                "TenderId": tender_id,
                "ProjectId": project_id,
                "UserId": user_id,
                "UserName": user_name,
                "AgentName": agent_name,
                "Step": step,
                "Attempt": attempt,
                "Status": "Running",
                "StartedAt": now,
                "CompletedAt": None,
                "FailedAt": None,
                "Error": None,
                "CreatedAt": now,
                "UpdatedAt": now,
            }
        )
        return execution_id

    async def complete(self, execution_id: str) -> None:
        now = self._now()
        await self.collection.update_one(
            {"ExecutionId": execution_id},
            {
                "$set": {
                    "Status": "Completed",
                    "CompletedAt": now,
                    "FailedAt": None,
                    "Error": None,
                    "UpdatedAt": now,
                }
            },
        )

    async def fail(self, execution_id: str, error: BaseException) -> None:
        now = self._now()
        error_data: dict[str, Any] = {
            "Type": type(error).__name__,
            "Message": str(error),
            "Timestamp": now,
        }
        await self.collection.update_one(
            {"ExecutionId": execution_id},
            {
                "$set": {
                    "Status": "Failed",
                    "FailedAt": now,
                    "CompletedAt": None,
                    "Error": error_data,
                    "UpdatedAt": now,
                }
            },
        )


__all__ = ["TenderAgentExecutionLogService"]
