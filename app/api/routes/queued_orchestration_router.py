"""FastAPI producer endpoint for asynchronous tender processing."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.infrastructure.rabbitmq_service import (
    get_rabbitmq_service,
    rabbitmq_enabled,
)
from app.infrastructure.tender_processing_jobs import (
    create_tender_processing_job,
    get_tender_processing_job_collection,
)


router = APIRouter(
    prefix="/documents",
    tags=["Queued Tender Processing"],
)
logger = logging.getLogger("uvicorn.error")

COMPANY_DOCUMENT_PROCESSING_JOB = "COMPANY_DOCUMENT_PROCESSING"


class QueuedTenderProcessRequest(BaseModel):
    """Request metadata required to enqueue a tender processing job."""

    company_id: str
    tender_id: str | None = None
    created_by: str | None = None
    user_id: str = ""
    user_name: str = ""
    project_id: str = ""
    bearer_token: str | None = None
    is_regenerated: bool | None = False


async def enqueue_company_document_processing_job(
    *,
    run_id: str,
    company_id: str,
    project_id: str = "",
    user_id: str = "",
    user_name: str = "",
    created_by: str = "",
) -> dict[str, Any]:
    """Persist and publish a CompanyId-only document-processing job."""

    queue_document = None
    queue_document_id = None
    try:
        queue_document, queue_document_id = await create_tender_processing_job(
            run_id=run_id,
            company_id=company_id,
            tender_id=None,
            project_id=project_id,
            user_id=user_id,
            user_name=user_name,
            created_by=created_by,
            is_regenerated=False,
            job_type=COMPANY_DOCUMENT_PROCESSING_JOB,
        )

        message = {
            "event_id": run_id,
            "event_type": "company.document.processing.requested",
            "schema_version": "1.0",
            "job_type": COMPANY_DOCUMENT_PROCESSING_JOB,
            "run_id": run_id,
            "company_id": company_id,
            "tender_id": None,
            "created_by": created_by,
            "user_id": user_id,
            "user_name": user_name,
            "project_id": project_id,
        }
        message_id = await get_rabbitmq_service().publish(message)

        queued_at = datetime.now(timezone.utc)
        await get_tender_processing_job_collection().update_one(
            {"_id": queue_document_id, "Status": "PENDING_PUBLISH"},
            {
                "$set": {
                    "MessageId": message_id,
                    "Status": "QUEUED",
                    "QueuedAt": queued_at,
                    "UpdatedAt": queued_at,
                }
            },
        )
        return {
            "run_id": run_id,
            "status": "QUEUED",
            "message_id": message_id,
            "attempt": queue_document["Attempt"],
            "job_type": COMPANY_DOCUMENT_PROCESSING_JOB,
            "message": "Company document processing job queued successfully",
        }
    except Exception as exc:
        now = datetime.now(timezone.utc)
        if queue_document_id is not None:
            await get_tender_processing_job_collection().update_one(
                {"_id": queue_document_id},
                {
                    "$set": {
                        "Status": "FAILED",
                        "Error": {
                            "Phase": "PUBLISH",
                            "Type": type(exc).__name__,
                            "Message": str(exc),
                            "Timestamp": now,
                        },
                        "UpdatedAt": now,
                        "FailedAt": now,
                    }
                },
            )
        raise


@router.post("/process/ocr/using_Tenderid_or_Companyid")
async def queue_tender_processing(
    request: QueuedTenderProcessRequest,
) -> dict[str, Any]:
    """Queue a request without running agents inside the FastAPI process."""

    company_id = str(request.company_id or "").strip()
    tender_id = str(request.tender_id or "").strip()
    is_regenerated = bool(request.is_regenerated)
    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="company_id is required",
        )

    run_id = str(uuid4())
    if not tender_id:
        logger.info(
            "Company document OCR endpoint called | company_id=%s | run_id=%s",
            company_id,
            run_id,
        )
        if not rabbitmq_enabled():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "RabbitMQ must be enabled to queue CompanyId-only "
                    "document processing. Set USING_RABBITMQ=true."
                ),
            )

        try:
            return await enqueue_company_document_processing_job(
                run_id=run_id,
                company_id=company_id,
                project_id=request.project_id,
                user_id=request.user_id,
                user_name=request.user_name,
                created_by=str(
                    request.created_by or request.user_name or ""
                ).strip(),
            )
        except Exception as exc:
            logger.exception(
                "Unable to queue company document job | run_id=%s",
                run_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Unable to queue company document processing job: {exc}",
            ) from exc

    logger.info(
        "Tender OCR endpoint called | company_id=%s | tender_id=%s | run_id=%s",
        company_id,
        tender_id,
        run_id,
    )

    if not rabbitmq_enabled():
        logger.info(
            "RabbitMQ disabled; running tender processing directly | run_id=%s",
            run_id,
        )
        try:
            from app.orchestration.orchestration_layer import (
                process_tender_document_flow,
            )

            result = await process_tender_document_flow(
                company_id=company_id,
                tender_id=tender_id,
                created_by=str(request.created_by or request.user_name or "").strip(),
                user_id=request.user_id,
                user_name=request.user_name,
                project_id=request.project_id,
                bearer_token=request.bearer_token,
                is_regenerated=is_regenerated,
                run_id=run_id,
            )
        except Exception as exc:
            logger.exception(
                "Direct tender processing failed | run_id=%s",
                run_id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Tender processing failed: {exc}",
            ) from exc

        return {
            "run_id": run_id,
            "status": str(result.get("Status") or "Completed").upper(),
            "mode": "direct",
            "message": "Tender processing completed without RabbitMQ",
            "result": result,
        }

    queue_document_id = None

    try:
        queue_document, queue_document_id = await create_tender_processing_job(
            run_id=run_id,
            company_id=company_id,
            tender_id=tender_id,
            project_id=request.project_id,
            user_id=request.user_id,
            user_name=request.user_name,
            created_by=str(request.created_by or request.user_name or "").strip(),
            is_regenerated=is_regenerated,
        )

        # The non-secret request metadata is persisted under RequestBody.
        # The bearer token is carried only in RabbitMQ and is never logged.
        message = {
            "event_id": run_id,
            "event_type": "tender.processing.requested",
            "schema_version": "1.0",
            "run_id": run_id,
            "company_id": company_id,
            "tender_id": tender_id,
            "created_by": request.created_by or request.user_name,
            "user_id": request.user_id,
            "user_name": request.user_name,
            "project_id": request.project_id,
            "bearer_token": request.bearer_token,
            "is_regenerated": is_regenerated,
        }
        message_id = await get_rabbitmq_service().publish(message)

        collection = get_tender_processing_job_collection()
        queued_at = datetime.now(timezone.utc)
        await collection.update_one(
            {"_id": queue_document_id, "Status": "PENDING_PUBLISH"},
            {
                "$set": {
                    "MessageId": message_id,
                    "Status": "QUEUED",
                    "QueuedAt": queued_at,
                    "UpdatedAt": queued_at,
                }
            },
        )
        logger.info(
            "Tender OCR job queued | run_id=%s | message_id=%s",
            run_id,
            message_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        now = datetime.now(timezone.utc)
        if queue_document_id is not None:
            await get_tender_processing_job_collection().update_one(
                {"_id": queue_document_id},
                {
                    "$set": {
                        "Status": "FAILED",
                        "Error": {
                            "Phase": "PUBLISH",
                            "Type": type(exc).__name__,
                            "Message": str(exc),
                            "Timestamp": now,
                        },
                        "UpdatedAt": now,
                        "FailedAt": now,
                    }
                },
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to queue tender processing job: {exc}",
        ) from exc

    return {
        "run_id": run_id,
        "status": "QUEUED",
        "message_id": message_id,
        "attempt": queue_document["Attempt"],
        "message": "Tender processing job queued successfully",
    }


__all__ = [
    "COMPANY_DOCUMENT_PROCESSING_JOB",
    "QueuedTenderProcessRequest",
    "enqueue_company_document_processing_job",
    "queue_tender_processing",
    "router",
]
