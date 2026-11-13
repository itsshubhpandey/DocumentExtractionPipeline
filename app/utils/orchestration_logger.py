"""Persistent logging and checkpointing for the tender orchestration flow."""

from __future__ import annotations

import inspect
import logging
import os
import traceback
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4
from motor.motor_asyncio import AsyncIOMotorCollection

from app.infrastructure.motor_database import get_motor_client
from app.infrastructure.tender_processing_jobs import (
    get_tender_processing_stage_collection,
)
from app.services.tender_agent_execution_log_service import (
    TenderAgentExecutionLogService,
)


ResultType = TypeVar("ResultType")


class PersistentOrchestrationLogger:
    """Store one processing-job document per stage attempt."""

    DEFAULT_COLLECTION = "TenderProcessingJobs"
    STEP_ORDER = (
        "document_processing",
        "requirement_extraction",
        "requirement_deduplication",
        "evaluation_criteria",
    )
    STEP_AGENT_NAMES = {
        "document_processing": "DocExtractionPipeline",
        "requirement_extraction": (
            "Requirements_Detection_Extraction_And_Intent_Agent"
        ),
        "requirement_deduplication": "RequirementDeduplicationAgent",
        "evaluation_criteria": "EvaluationCriteriaAgent",
    }
    STEP_STAGE_NAMES = {
        "document_processing": "DOCUMENT_PROCESSING",
        "requirement_extraction": "REQUIREMENT_EXTRACTION",
        "requirement_deduplication": "DEDUPLICATION",
        "evaluation_criteria": "EVALUATION_CRITERIA",
    }

    def __init__(
        self,
        *,
        company_id: str,
        tender_id: str,
        project_id: str = "",
        user_id: str = "",
        user_name: str = "",
        created_by: str = "",
        is_regenerated: bool = False,
        run_id: str | None = None,
        collection: AsyncIOMotorCollection | None = None,
    ) -> None:
        self.company_id = str(company_id or "").strip()
        self.tender_id = str(tender_id or "").strip()
        self.project_id = str(project_id or "").strip()
        self.user_id = str(user_id or "").strip()
        self.user_name = str(user_name or "").strip()
        self.created_by = str(created_by or "").strip()
        self.is_regenerated = bool(is_regenerated)
        self.run_id = str(run_id or "").strip()
        self._client = None
        self._logger = logging.getLogger("uvicorn.error")
        self._execution_log_service: (
            TenderAgentExecutionLogService | None
        ) = None
        self._active_execution_ids: dict[str, str] = {}
        self._input_documents: list[dict[str, Any]] = []

        if not self.company_id:
            raise ValueError("company_id cannot be empty")
        if not self.tender_id:
            raise ValueError("tender_id cannot be empty")

        if collection is not None:
            self.collection = collection
            return

        database_name = (
            os.getenv("MONGO_DB_NAME")
            or os.getenv("MONGODB_DATABASE")
            or "DocAI"
        )
        collection_name = os.getenv(
            "ORCHESTRATION_RUN_COLLECTION",
            self.DEFAULT_COLLECTION,
        )
        if collection_name == self.DEFAULT_COLLECTION:
            self.collection = get_tender_processing_stage_collection()
        else:
            self._client = get_motor_client()
            self.collection = self._client[database_name][collection_name]
        self._execution_log_service = TenderAgentExecutionLogService()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _identity(self) -> dict[str, Any]:
        return {
            "CompanyId": self.company_id,
            "TenderId": self.tender_id,
        }

    @staticmethod
    def _status(record: dict[str, Any] | None) -> str:
        return str((record or {}).get("Status") or "").upper()

    @staticmethod
    def _normalize_input_documents(documents: Any) -> list[dict[str, Any]]:
        if not isinstance(documents, list):
            return []

        normalized: list[dict[str, Any]] = []
        seen_document_ids: set[str] = set()
        for item in documents:
            if not isinstance(item, dict):
                continue
            document_id = str(
                item.get("DocumentId")
                or item.get("document_id")
                or ""
            ).strip()
            if document_id and document_id in seen_document_ids:
                continue
            if document_id:
                seen_document_ids.add(document_id)
            normalized.append(
                {
                    "DocumentName": str(
                        item.get("DocumentName")
                        or item.get("document_name")
                        or ""
                    ),
                    "FilePath": str(
                        item.get("FilePath")
                        or item.get("file_path")
                        or ""
                    ),
                    "DocumentId": document_id or None,
                    "Status": str(
                        item.get("Status")
                        or item.get("status")
                        or ""
                    ),
                }
            )
        return normalized

    @staticmethod
    def _document_id_value(
        documents: list[dict[str, Any]],
    ) -> str | list[str] | None:
        document_ids = [
            str(item.get("DocumentId") or "").strip()
            for item in documents
            if str(item.get("DocumentId") or "").strip()
        ]
        if not document_ids:
            return None
        if len(document_ids) == 1:
            return document_ids[0]
        return document_ids

    @staticmethod
    def _merge_input_documents(
        existing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        without_id: list[dict[str, Any]] = []
        for item in [*existing, *incoming]:
            document_id = str(item.get("DocumentId") or "").strip()
            if document_id:
                merged[document_id] = item
            else:
                without_id.append(item)
        return [*merged.values(), *without_id]

    def _restore_input_documents(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        document_stage = self._latest_by_stage(records).get(
            "document_processing"
        )
        if not document_stage:
            return
        documents = document_stage.get("InputDocuments")
        self._input_documents = self._normalize_input_documents(documents)

    async def _run_records(self) -> list[dict[str, Any]]:
        if not self.run_id:
            raise RuntimeError("The orchestration run has not been started")
        return await self.collection.find(
            {"RunId": self.run_id}
        ).to_list(length=None)

    def _latest_by_stage(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        stage_to_step = {
            stage: step for step, stage in self.STEP_STAGE_NAMES.items()
        }
        for record in records:
            step_name = stage_to_step.get(
                str(record.get("Stage") or "").upper()
            )
            if step_name is None:
                continue

            previous = latest.get(step_name)
            current_key = (
                int(record.get("Attempt") or 0),
                record.get("UpdatedAt") or datetime.min.replace(
                    tzinfo=timezone.utc
                ),
            )
            previous_key = (
                int((previous or {}).get("Attempt") or 0),
                (previous or {}).get("UpdatedAt")
                or datetime.min.replace(tzinfo=timezone.utc),
            )
            if previous is None or current_key >= previous_key:
                latest[step_name] = record
        return latest

    def _summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        latest = self._latest_by_stage(records)
        steps: dict[str, dict[str, Any]] = {}
        for order, step_name in enumerate(self.STEP_ORDER, start=1):
            record = latest.get(step_name)
            steps[step_name] = {
                "Order": order,
                "Stage": self.STEP_STAGE_NAMES[step_name],
                "AgentName": self.STEP_AGENT_NAMES[step_name],
                "Status": self._status(record) or "PENDING",
                "Attempt": int((record or {}).get("Attempt") or 0),
                "StartedAt": (record or {}).get("StartedAt"),
                "CompletedAt": (record or {}).get("CompletedAt"),
                "FailedAt": (record or {}).get("FailedAt"),
                "Error": (record or {}).get("Error"),
            }
        statuses = [step["Status"] for step in steps.values()]
        if all(status == "COMPLETED" for status in statuses):
            run_status = "COMPLETED"
        elif "FAILED" in statuses:
            run_status = "FAILED"
        elif "PROCESSING" in statuses:
            run_status = "PROCESSING"
        elif "QUEUED" in statuses:
            run_status = "QUEUED"
        else:
            run_status = "PENDING"

        current_step = next(
            (
                step_name
                for step_name in self.STEP_ORDER
                if steps[step_name]["Status"] in {"PROCESSING", "QUEUED"}
            ),
            None,
        )
        latest_record = max(
            records,
            key=lambda record: record.get("UpdatedAt")
            or datetime.min.replace(tzinfo=timezone.utc),
            default={},
        )
        return {
            "_id": latest_record.get("_id"),
            "RunId": self.run_id,
            **self._identity(),
            "Status": run_status,
            "CurrentStep": current_step,
            "InputDocuments": (
                latest.get("document_processing", {}).get(
                    "InputDocuments", []
                )
            ),
            "Steps": steps,
            "UpdatedAt": latest_record.get("UpdatedAt"),
        }

    async def _latest_stage_document(
        self,
        step_name: str,
    ) -> dict[str, Any] | None:
        return self._latest_by_stage(await self._run_records()).get(step_name)

    @staticmethod
    def _new_stage_document(
        *,
        run_id: str,
        identity: dict[str, Any],
        user_id: str,
        user_name: str,
        created_by: str,
        is_regenerated: bool,
        step_name: str,
        attempt: int,
        status: str,
        input_documents: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        normalized_documents = list(input_documents or [])
        return {
            "RunId": run_id,
            **identity,
            "DocumentId": PersistentOrchestrationLogger._document_id_value(
                normalized_documents
            ),
            "Stage": PersistentOrchestrationLogger.STEP_STAGE_NAMES[
                step_name
            ],
            "Attempt": attempt,
            "Status": status,
            "Error": None,
            "InputDocuments": normalized_documents,
            "OutputReference": {"Id": None, "Version": None},
            "CompletedAt": None,
            "CreatedAt": now,
            "UpdatedAt": now,
        }

    async def _find_resumable_run(self) -> list[dict[str, Any]]:
        records = await self.collection.find(self._identity()).to_list(
            length=None
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            run_id = str(record.get("RunId") or "")
            if run_id:
                grouped[run_id].append(record)

        candidates: list[tuple[datetime, list[dict[str, Any]]]] = []
        for run_records in grouped.values():
            latest = self._latest_by_stage(run_records)
            # Ignore legacy or unrelated records that do not contain any of
            # the stages managed by this orchestration flow.
            if not any(latest.get(step_name) for step_name in self.STEP_ORDER):
                continue
            if all(
                self._status(latest.get(step_name)) == "COMPLETED"
                for step_name in self.STEP_ORDER
            ):
                continue
            updated_at = max(
                (
                    record.get("UpdatedAt")
                    or datetime.min.replace(tzinfo=timezone.utc)
                    for record in run_records
                ),
                default=datetime.min.replace(tzinfo=timezone.utc),
            )
            candidates.append((updated_at, run_records))

        if not candidates:
            return []
        return max(candidates, key=lambda item: item[0])[1]

    async def _start_or_resume(self) -> dict[str, Any]:
        if self.run_id:
            records = await self._run_records()
            if records:
                self._restore_input_documents(records)
                summary = self._summary(records)
                if summary["Status"] == "COMPLETED":
                    raise RuntimeError(
                        f"Orchestration run {self.run_id!r} is already completed"
                    )
                self._logger.info(
                    "Orchestration resumed | run_id=%s | company_id=%s | "
                    "tender_id=%s",
                    self.run_id,
                    self.company_id,
                    self.tender_id,
                )
                return summary

            self._logger.info(
                "Orchestration stage log not found; creating first stage | "
                "run_id=%s | company_id=%s | tender_id=%s",
                self.run_id,
                self.company_id,
                self.tender_id,
            )

            # A new queue job may be created when a previous job failed. In
            # that case, continue the latest incomplete orchestration run so
            # completed stages are checkpointed and are not executed again.
            existing_records = await self._find_resumable_run()
            if existing_records:
                previous_run_id = self.run_id
                self.run_id = str(existing_records[0]["RunId"])
                self._restore_input_documents(existing_records)
                summary = self._summary(existing_records)
                self._logger.info(
                    "New queue job mapped to incomplete orchestration run | "
                    "queue_run_id=%s | orchestration_run_id=%s | "
                    "current_step=%s",
                    previous_run_id,
                    self.run_id,
                    summary.get("CurrentStep"),
                )
                return summary

        existing_records = (
            [] if self.run_id else await self._find_resumable_run()
        )
        if existing_records:
            self.run_id = str(existing_records[0]["RunId"])
            self._restore_input_documents(existing_records)
            summary = self._summary(existing_records)
            self._logger.info(
                "Orchestration resumed | run_id=%s | company_id=%s | "
                "tender_id=%s",
                self.run_id,
                self.company_id,
                self.tender_id,
            )
            return summary

        if not self.run_id:
            self.run_id = str(uuid4())
        document = self._new_stage_document(
            run_id=self.run_id,
            identity=self._identity(),
            user_id=self.user_id,
            user_name=self.user_name,
            created_by=self.created_by,
            is_regenerated=self.is_regenerated,
            step_name=self.STEP_ORDER[0],
            attempt=1,
            status="PENDING",
            input_documents=self._input_documents,
        )
        insert_result = await self.collection.insert_one(document)
        document["_id"] = insert_result.inserted_id
        self._logger.info(
            "Orchestration created | run_id=%s | company_id=%s | tender_id=%s",
            self.run_id,
            self.company_id,
            self.tender_id,
        )
        return self._summary([document])

    async def start_or_resume(self) -> dict[str, Any]:
        """Resume the run or create its first stage checkpoint."""

        return await self._start_or_resume()

    async def get_run(self) -> dict[str, Any]:
        records = await self._run_records()
        if not records:
            raise RuntimeError(
                f"Orchestration run {self.run_id!r} was not found"
            )
        return self._summary(records)

    async def set_input_documents(self, documents: Any) -> None:
        """Persist normalized document details on the document stage."""

        incoming_documents = self._normalize_input_documents(documents)
        self._input_documents = self._merge_input_documents(
            self._input_documents,
            incoming_documents,
        )
        stage_document = await self._latest_stage_document(
            "document_processing"
        )
        if stage_document is None:
            raise RuntimeError("Document-processing stage record was not found")
        await self.collection.update_one(
            {"_id": stage_document["_id"]},
            {
                "$set": {
                    "DocumentId": self._document_id_value(
                        self._input_documents
                    ),
                    "InputDocuments": self._input_documents,
                    "UpdatedAt": self._now(),
                }
            },
        )

    async def get_input_documents(self) -> list[dict[str, Any]]:
        """Reload recovery documents from TenderProcessingJobs."""

        records = await self._run_records()
        self._restore_input_documents(records)
        return [dict(item) for item in self._input_documents]

    async def _start_agent_execution_log(
        self,
        step_name: str,
    ) -> None:
        if self._execution_log_service is None:
            return

        try:
            stage_document = await self._latest_stage_document(step_name)
            attempt = int((stage_document or {}).get("Attempt") or 1)
            execution_id = await self._execution_log_service.start(
                job_id=self.run_id,
                run_id=self.run_id,
                company_id=self.company_id,
                tender_id=self.tender_id,
                project_id=self.project_id,
                user_id=self.user_id,
                user_name=self.user_name,
                agent_name=self.STEP_AGENT_NAMES[step_name],
                step=step_name,
                attempt=attempt,
            )
            self._active_execution_ids[step_name] = execution_id
        except Exception:
            # Execution logging must not prevent the actual agent from
            # running. The stage record remains authoritative.
            self._logger.exception(
                "Agent execution log start failed | run_id=%s | step=%s",
                self.run_id,
                step_name,
            )

    async def _complete_agent_execution_log(self, step_name: str) -> None:
        execution_id = self._active_execution_ids.pop(step_name, None)
        if not execution_id or self._execution_log_service is None:
            return

        try:
            await self._execution_log_service.complete(execution_id)
        except Exception:
            self._logger.exception(
                "Agent execution log completion failed | run_id=%s | step=%s",
                self.run_id,
                step_name,
            )

    async def _fail_agent_execution_log(
        self,
        step_name: str,
        error: BaseException,
    ) -> None:
        execution_id = self._active_execution_ids.pop(step_name, None)
        if not execution_id or self._execution_log_service is None:
            return

        try:
            await self._execution_log_service.fail(execution_id, error)
        except Exception:
            self._logger.exception(
                "Agent execution log failure update failed | run_id=%s | step=%s",
                self.run_id,
                step_name,
            )

    async def mark_step_started(self, step_name: str) -> None:
        if step_name not in self.STEP_ORDER:
            raise ValueError(f"Unknown orchestration step: {step_name}")

        now = self._now()
        latest = await self._latest_stage_document(step_name)
        latest_status = self._status(latest)
        if latest is not None and latest_status not in {"COMPLETED", "FAILED"}:
            await self.collection.update_one(
                {"_id": latest["_id"]},
                {
                    "$set": {
                        "Status": "PROCESSING",
                        "DocumentId": self._document_id_value(
                            self._input_documents
                        ),
                        "InputDocuments": self._input_documents,
                        "CompletedAt": None,
                        "Error": None,
                        "UpdatedAt": now,
                    }
                },
            )
            attempt = int(latest.get("Attempt") or 1)
        else:
            attempt = int((latest or {}).get("Attempt") or 0) + 1
            document = self._new_stage_document(
                run_id=self.run_id,
                identity=self._identity(),
                user_id=self.user_id,
                user_name=self.user_name,
                created_by=self.created_by,
                is_regenerated=self.is_regenerated,
                step_name=step_name,
                attempt=attempt,
                status="PROCESSING",
                input_documents=self._input_documents,
            )
            document["UpdatedAt"] = now
            await self.collection.insert_one(document)

        self._logger.info(
            "Orchestration stage started | run_id=%s | stage=%s | attempt=%s",
            self.run_id,
            self.STEP_STAGE_NAMES[step_name],
            attempt,
        )
        await self._start_agent_execution_log(step_name)

    async def mark_step_completed(
        self,
        step_name: str,
        result: Any,
    ) -> None:
        stage_document = await self._latest_stage_document(step_name)
        if stage_document is None:
            raise RuntimeError(f"Stage record not found: {step_name}")
        now = self._now()
        await self.collection.update_one(
            {"_id": stage_document["_id"]},
            {
                "$set": {
                    "Status": "COMPLETED",
                    "CompletedAt": now,
                    "Error": None,
                    "DocumentId": self._document_id_value(
                        self._input_documents
                    ),
                    "InputDocuments": self._input_documents,
                    "OutputReference": (
                        result.get("OutputReference")
                        if isinstance(result, dict)
                        and isinstance(result.get("OutputReference"), dict)
                        else stage_document.get(
                            "OutputReference",
                            {"Id": None, "Version": None},
                        )
                    ),
                    "UpdatedAt": now,
                },
            },
        )
        self._logger.info(
            "Orchestration stage completed | run_id=%s | stage=%s | attempt=%s",
            self.run_id,
            self.STEP_STAGE_NAMES[step_name],
            stage_document.get("Attempt", 1),
        )
        await self._complete_agent_execution_log(step_name)

    async def mark_step_failed(
        self,
        step_name: str,
        error: BaseException,
    ) -> None:
        stage_document = await self._latest_stage_document(step_name)
        if stage_document is None:
            raise RuntimeError(f"Stage record not found: {step_name}")
        now = self._now()
        error_data = {
            "Type": type(error).__name__,
            "Message": str(error),
            "Traceback": traceback.format_exc(),
            "Timestamp": now,
        }
        await self.collection.update_one(
            {"_id": stage_document["_id"]},
            {
                "$set": {
                    "Status": "FAILED",
                    "Error": error_data,
                    "CompletedAt": None,
                    "UpdatedAt": now,
                },
            },
        )
        self._logger.exception(
            "Orchestration stage failed | run_id=%s | stage=%s | error=%s",
            self.run_id,
            self.STEP_STAGE_NAMES[step_name],
            error,
        )
        await self._fail_agent_execution_log(step_name, error)

    async def mark_step_skipped(self, step_name: str) -> None:
        self._logger.info(
            "Orchestration stage skipped | run_id=%s | stage=%s | "
            "reason=already_completed",
            self.run_id,
            self.STEP_STAGE_NAMES[step_name],
        )

    async def run_step(
        self,
        step_name: str,
        operation: Callable[[], Awaitable[ResultType] | ResultType],
    ) -> ResultType:
        """Run a stage, or skip it when already completed."""

        stage_document = await self._latest_stage_document(step_name)
        if self._status(stage_document) == "COMPLETED":
            await self.mark_step_skipped(step_name)
            return None  # type: ignore[return-value]

        await self.mark_step_started(step_name)
        try:
            result = operation()
            if inspect.isawaitable(result):
                result = await result
            await self.mark_step_completed(step_name, result)
            return result
        except Exception as exc:
            await self.mark_step_failed(step_name, exc)
            raise

    async def mark_run_completed(self) -> dict[str, Any]:
        summary = await self.get_run()
        if summary["Status"] != "COMPLETED":
            raise RuntimeError(
                f"Cannot mark run completed; stage status is {summary['Status']}"
            )
        self._logger.info(
            "Orchestration completed | run_id=%s | company_id=%s | "
            "tender_id=%s",
            self.run_id,
            self.company_id,
            self.tender_id,
        )
        return summary

    def close(self) -> None:
        # The Motor client is shared process-wide. It is closed by the
        # application shutdown hook, not after each orchestration run.
        self._client = None


__all__ = ["PersistentOrchestrationLogger"]
