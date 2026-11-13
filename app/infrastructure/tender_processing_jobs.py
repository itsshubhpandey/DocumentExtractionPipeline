"""Persistence helpers for RabbitMQ tender-processing jobs."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from app.infrastructure.motor_database import get_motor_collection


DEFAULT_COLLECTION_NAME = "TenderDocumentsProcessingJobRMQ"
DEFAULT_STAGE_LOG_COLLECTION_NAME = "TenderProcessingJobs"


def get_tender_processing_job_collection():
    """Return the Mongo collection used to track RabbitMQ tender jobs."""

    collection_name = os.getenv(
        "TENDER_PROCESSING_JOB_COLLECTION",
        DEFAULT_COLLECTION_NAME,
    )
    return get_motor_collection(collection_name)


def get_tender_processing_stage_collection():
    """Return the collection used for agent/stage execution logging."""

    collection_name = os.getenv(
        "ORCHESTRATION_RUN_COLLECTION",
        DEFAULT_STAGE_LOG_COLLECTION_NAME,
    )
    return get_motor_collection(collection_name)


async def create_tender_processing_job(
    *,
    run_id: str,
    company_id: str,
    tender_id: str | None,
    project_id: str,
    user_id: str,
    user_name: str,
    created_by: str,
    is_regenerated: bool,
    job_type: str = "TENDER_PROCESSING",
) -> tuple[dict[str, Any], Any]:
    """Create the queue execution record before publishing to RabbitMQ."""

    now = datetime.now(timezone.utc)
    document: dict[str, Any] = {
        "RunId": run_id,
        "JobType": job_type,
        "RequestBody": {
            "CompanyId": company_id,
            "TenderId": tender_id,
            "ProjectId": project_id,
            "UserId": user_id,
            "UserName": user_name,
            "CreatedBy": created_by,
            "IsRegenerated": is_regenerated,
        },
        "Status": "PENDING_PUBLISH",
        "MessageId": None,
        "WorkerId": None,
        "Attempt": 0,
        "LeaseUntil": None,
        "Error": None,
        "QueuedAt": None,
        "StartedAt": None,
        "CompletedAt": None,
        "FailedAt": None,
        "CreatedAt": now,
        "UpdatedAt": now,
    }
    result = await get_tender_processing_job_collection().insert_one(document)
    document["_id"] = result.inserted_id
    return document, result.inserted_id


__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_STAGE_LOG_COLLECTION_NAME",
    "create_tender_processing_job",
    "get_tender_processing_job_collection",
    "get_tender_processing_stage_collection",
]
