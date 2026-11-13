"""RabbitMQ worker for the complete tender processing flow."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from app.infrastructure.environment import load_project_environment

load_project_environment()

from app.infrastructure.motor_database import (
    close_motor_client,
)
from app.infrastructure.rabbitmq_service import RabbitMQService, rabbitmq_enabled
from app.infrastructure.tender_processing_jobs import (
    get_tender_processing_job_collection,
)
from app.orchestration.orchestration_layer import process_tender_document_flow
from app.api.routes.document_extractor_router import (
    MongoDocumentProcessRequest,
    process_documents_from_mongo_direct,
)


COMPANY_DOCUMENT_PROCESSING_JOB = "COMPANY_DOCUMENT_PROCESSING"
TENDER_PROCESSING_JOB = "TENDER_PROCESSING"


class TenderProcessingWorker:
    """Consume tender jobs and execute them one at a time per worker."""

    def __init__(
        self,
        *,
        rabbitmq: RabbitMQService | None = None,
        prefetch_count: int | None = None,
    ) -> None:
        self.rabbitmq = rabbitmq or RabbitMQService()
        self.jobs = get_tender_processing_job_collection()
        self.worker_id = str(uuid4())
        self.prefetch_count = prefetch_count or int(
            os.getenv("RABBITMQ_PREFETCH_COUNT", "1")
        )
        self.lease_seconds = int(
            os.getenv("TENDER_WORKER_LEASE_SECONDS", "3600")
        )
        self.lease_heartbeat_seconds = max(
            5.0,
            float(
                os.getenv(
                    "TENDER_WORKER_LEASE_HEARTBEAT_SECONDS",
                    str(min(300, max(5, self.lease_seconds // 3))),
                )
            ),
        )
        self.consumer_retry_seconds = max(
            1.0,
            float(os.getenv("RABBITMQ_CONSUMER_RETRY_SECONDS", "5")),
        )
        self.logger = logging.getLogger("tender.processing.worker")
        self._stopping = asyncio.Event()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    async def _claim_job(self, run_id: str) -> str:
        """Atomically claim a queued or recoverable processing job."""

        now = self._now()
        lease_until = now + timedelta(seconds=self.lease_seconds)
        result = await self.jobs.update_one(
            {
                "RunId": run_id,
                "$or": [
                    {
                        "Status": {
                            "$in": [
                                "PENDING_PUBLISH",
                                "QUEUED",
                                "PUBLISHED",
                            ]
                        }
                    },
                    {
                        "Status": "PROCESSING",
                        "$or": [
                            {"LeaseUntil": {"$lt": now}},
                            {"LeaseUntil": {"$exists": False}},
                        ],
                    },
                ],
            },
            {
                "$set": {
                    "Status": "PROCESSING",
                    "WorkerId": self.worker_id,
                    "StartedAt": now,
                    "LeaseUntil": lease_until,
                    "Error": None,
                    "UpdatedAt": now,
                },
                "$inc": {"Attempt": 1},
            },
        )
        if result.matched_count:
            return "claimed"

        records = await self.jobs.find(
            {"RunId": run_id},
            {"Status": 1, "LeaseUntil": 1},
        ).to_list(length=None)
        if not records:
            return "missing"
        statuses = {
            str(record.get("Status") or "").upper()
            for record in records
        }
        if statuses and statuses.issubset({"COMPLETED"}):
            return "completed"
        return "already_processing"

    async def _lease_heartbeat(self, run_id: str) -> None:
        """Renew the Mongo processing lease while a job is running."""

        while True:
            await asyncio.sleep(self.lease_heartbeat_seconds)
            now = self._now()
            try:
                result = await self.jobs.update_one(
                    {
                        "RunId": run_id,
                        "Status": "PROCESSING",
                        "WorkerId": self.worker_id,
                    },
                    {
                        "$set": {
                            "LeaseUntil": now + timedelta(
                                seconds=self.lease_seconds
                            ),
                            "UpdatedAt": now,
                        }
                    },
                )
            except Exception:
                self.logger.exception(
                    "Job lease renewal failed; will retry | run_id=%s | "
                    "worker_id=%s",
                    run_id,
                    self.worker_id,
                )
                continue
            if result.matched_count != 1:
                self.logger.warning(
                    "Job lease renewal stopped; active job was not found | "
                    "run_id=%s | worker_id=%s",
                    run_id,
                    self.worker_id,
                )
                return
            self.logger.debug(
                "Job lease renewed | run_id=%s | worker_id=%s",
                run_id,
                self.worker_id,
            )

    async def _mark_completed(self, run_id: str) -> None:
        now = self._now()
        await self.jobs.update_one(
            {"RunId": run_id},
            {
                "$set": {
                    "Status": "COMPLETED",
                    "CompletedAt": now,
                    "LeaseUntil": None,
                    "UpdatedAt": now,
                }
            },
        )

    async def _mark_failed(self, run_id: str, error: BaseException) -> None:
        now = self._now()
        await self.jobs.update_one(
            {
                "RunId": run_id,
                "Status": "PROCESSING",
            },
            {
                "$set": {
                    "Status": "FAILED",
                    "FailedAt": now,
                    "LeaseUntil": None,
                    "Error": {
                        "Phase": "PROCESSING",
                        "Type": type(error).__name__,
                        "Message": str(error),
                        "Timestamp": now,
                    },
                    "UpdatedAt": now,
                }
            },
        )

    async def _handle_message(self, message: Any) -> None:
        run_id = ""
        payload: dict[str, Any] = {}
        job_claimed = False
        self.logger.info(
            "RabbitMQ message received | message_id=%s | redelivered=%s",
            getattr(message, "message_id", None),
            bool(getattr(message, "redelivered", False)),
        )
        try:
            payload = json.loads(message.body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("RabbitMQ payload must be a JSON object")

            # job_id remains a fallback for messages published by older code.
            run_id = str(payload.get("run_id") or payload.get("job_id") or "")
            company_id = str(payload.get("company_id") or "").strip()
            tender_id = str(payload.get("tender_id") or "").strip()
            job_type = str(
                payload.get("job_type") or TENDER_PROCESSING_JOB
            ).strip().upper()
            if not run_id or not company_id:
                raise ValueError(
                    "RabbitMQ payload requires run_id and company_id"
                )
            if job_type not in {
                COMPANY_DOCUMENT_PROCESSING_JOB,
                TENDER_PROCESSING_JOB,
            }:
                raise ValueError(f"Unsupported RabbitMQ job_type: {job_type}")
            if job_type == TENDER_PROCESSING_JOB and not tender_id:
                raise ValueError(
                    "Tender-processing payload requires tender_id"
                )

            claim_status = await self._claim_job(run_id)
            if claim_status == "completed":
                self.logger.info(
                    "Acknowledging completed duplicate job | "
                    "run_id=%s | status=%s",
                    run_id,
                    claim_status,
                )
                await message.ack()
                return
            if claim_status == "already_processing":
                await self.rabbitmq.publish_retry(payload)
                await message.ack()
                self.logger.warning(
                    "Active job deferred to retry queue | run_id=%s | "
                    "retry_in_ms=%s",
                    run_id,
                    self.rabbitmq.retry_delay_ms,
                )
                return
            if claim_status != "claimed":
                raise RuntimeError(
                    "RabbitMQ job record is unavailable: "
                    f"run_id={run_id}, claim_status={claim_status}"
                )
            job_claimed = True

            lease_heartbeat = asyncio.create_task(
                self._lease_heartbeat(run_id)
            )
            try:
                if job_type == COMPANY_DOCUMENT_PROCESSING_JOB:
                    result = await process_documents_from_mongo_direct(
                        MongoDocumentProcessRequest(
                            company_id=company_id,
                            project_id=str(payload.get("project_id") or ""),
                            user_id=str(payload.get("user_id") or ""),
                            user_name=str(payload.get("user_name") or ""),
                            created_by=str(payload.get("created_by") or ""),
                        )
                    )
                    failures = [
                        item
                        for item in result.get("results", [])
                        if str(item.get("status") or "").upper() == "FAILED"
                    ]
                    if failures:
                        raise RuntimeError(
                            "Company document processing failed for "
                            f"{len(failures)} document(s): {failures}"
                        )
                else:
                    await process_tender_document_flow(
                        company_id=company_id,
                        tender_id=tender_id,
                        created_by=payload.get("created_by"),
                        user_id=str(payload.get("user_id") or ""),
                        user_name=str(payload.get("user_name") or ""),
                        project_id=str(payload.get("project_id") or ""),
                        bearer_token=payload.get("bearer_token"),
                        is_regenerated=bool(
                            payload.get("is_regenerated", False)
                        ),
                        run_id=run_id,
                    )
            finally:
                lease_heartbeat.cancel()
                await asyncio.gather(
                    lease_heartbeat,
                    return_exceptions=True,
                )
            await self._mark_completed(run_id)
            await message.ack()
            self.logger.info(
                "Processing job completed and message acknowledged | "
                "run_id=%s | job_type=%s",
                run_id,
                job_type,
            )
        except Exception as exc:
            self.logger.exception("Tender job failed | run_id=%s", run_id or "unknown")
            if run_id and job_claimed:
                try:
                    await self._mark_failed(run_id, exc)
                except Exception:
                    self.logger.exception(
                        "Unable to persist worker failure | run_id=%s",
                        run_id,
                    )
            # Publish to the durable dead-letter queue before removing the
            # original message. If dead-letter publishing fails, preserve the
            # original by requeueing it.
            try:
                await self.rabbitmq.publish_dead_letter(payload, exc)
            except Exception:
                self.logger.exception(
                    "Unable to publish failed job to dead-letter queue; "
                    "requeueing original | run_id=%s",
                    run_id or "unknown",
                )
                await message.nack(requeue=True)
                return

            try:
                await message.reject(requeue=False)
                self.logger.warning(
                    "RabbitMQ message moved to dead-letter queue | "
                    "run_id=%s | requeue=false",
                    run_id or "unknown",
                )
            except Exception:
                self.logger.exception(
                    "Unable to reject RabbitMQ message | run_id=%s",
                    run_id or "unknown",
                )
                raise

    async def run(self) -> None:
        """Start consuming jobs until the worker process is stopped."""

        self.logger.info(
            "Tender processing worker started | worker_id=%s | prefetch=%s",
            self.worker_id,
            self.prefetch_count,
        )
        reconnect_attempt = 0
        while not self._stopping.is_set():
            try:
                await self.rabbitmq.consume(
                    self._handle_message,
                    prefetch_count=self.prefetch_count,
                )
                if not self._stopping.is_set():
                    raise RuntimeError("RabbitMQ consumer exited unexpectedly")
            except asyncio.CancelledError:
                raise
            except Exception:
                reconnect_attempt += 1
                self.logger.exception(
                    "RabbitMQ consumer unavailable; restarting | worker_id=%s | "
                    "attempt=%s | retry_in_seconds=%s",
                    self.worker_id,
                    reconnect_attempt,
                    self.consumer_retry_seconds,
                )
                await self.rabbitmq.close()

                try:
                    await asyncio.wait_for(
                        self._stopping.wait(),
                        timeout=self.consumer_retry_seconds,
                    )
                except asyncio.TimeoutError:
                    self.logger.info(
                        "Retrying RabbitMQ consumer connection | worker_id=%s | attempt=%s",
                        self.worker_id,
                        reconnect_attempt,
                    )
            else:
                reconnect_attempt = 0

    async def close(self) -> None:
        self._stopping.set()
        await self.rabbitmq.close()
        close_motor_client()


async def main() -> None:
    configured_log_level = os.getenv(
        "TENDER_WORKER_LOG_LEVEL",
        "INFO",
    ).upper()
    logging.basicConfig(
        level=getattr(logging, configured_log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("tender.processing.worker")
    if not rabbitmq_enabled():
        logger.info(
            "RabbitMQ disabled by USING_RABBITMQ; worker will not connect."
        )
        return

    worker = TenderProcessingWorker()
    try:
        await worker.run()
    finally:
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main())


__all__ = ["TenderProcessingWorker", "main"]
