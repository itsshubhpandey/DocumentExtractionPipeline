from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from celery import Celery

from app.infrastructure.database import get_database


_worker_execution: ContextVar[bool] = ContextVar("worker_execution", default=False)

celery_app = Celery(
    "tender_master_agents",
    broker=os.getenv(
        "CELERY_BROKER_URL",
        os.getenv(
            "RABBITMQ_URL",
            "amqp://guest:guest@localhost:5672//",
        ),
    ),
    # Job status and results are persisted in MongoDB by this module.
    # Redis is intentionally not used as a Celery result backend.
    backend=None,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "1800")),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "1650")),
)

JOB_COLLECTION = os.getenv("CELERY_JOB_COLLECTION", "AgentWorkflowJobs")


def is_worker_execution() -> bool:
    return _worker_execution.get()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe(value: Any) -> Any:
    if hasattr(value, "get_secret_value"):
        return _safe(value.get_secret_value())
    if hasattr(value, "model_dump"):
        return _safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _jobs():
    return get_database()[JOB_COLLECTION]


def create_job(agent: str, payload: dict[str, Any]) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    document = {
        "JobId": job_id,
        "Agent": agent,
        "Status": "Queued",
        "Payload": _safe(
            {
                key: value
                for key, value in payload.items()
                if key.casefold().replace("_", "")
                not in {"token", "jwttoken", "authorization"}
            }
        ),
        "CreatedAt": _now(),
        "UpdatedAt": _now(),
    }
    _jobs().insert_one(document)
    return {"JobId": job_id, "Agent": agent, "Status": "Queued"}


def update_job(job_id: str, **fields: Any) -> None:
    fields["UpdatedAt"] = _now()
    _jobs().update_one({"JobId": job_id}, {"$set": _safe(fields)})


def get_job(job_id: str) -> dict[str, Any] | None:
    document = _jobs().find_one({"JobId": job_id}, {"_id": 0, "Payload": 0})
    if document:
        for key in ("CreatedAt", "UpdatedAt"):
            if isinstance(document.get(key), datetime):
                document[key] = document[key].isoformat()
    return document


def get_job_result(job_id: str) -> dict[str, Any] | None:
    document = _jobs().find_one({"JobId": job_id}, {"_id": 0, "Result": 1, "Status": 1, "Error": 1})
    if not document or document.get("Status") != "Completed":
        return None
    return document.get("Result")


def enqueue_job(agent: str, payload: dict[str, Any]) -> dict[str, Any]:
    job = create_job(agent, payload)
    try:
        run_agent_job.apply_async(args=[job["JobId"], agent, _safe(payload)])
    except Exception as exc:
        update_job(job["JobId"], Status="Failed", Error={"Type": type(exc).__name__, "Message": str(exc)})
        raise
    return job


def enqueue_or_continue(agent: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a queued response on HTTP execution, otherwise continue in Celery."""
    if is_worker_execution():
        return None
    return enqueue_job(agent, payload)


def _internal_request() -> Any:
    return SimpleNamespace(headers={})


@celery_app.task(bind=True, name="tender_master_agents.run_agent_job", max_retries=os.getenv("CELERY_MAX_RETRIES", "2"))
def run_agent_job(self, job_id: str, agent: str, payload: dict[str, Any]) -> None:
    update_job(job_id, Status="Running", CeleryTaskId=self.request.id)
    token = _worker_execution.set(True)
    try:
        module_name, function_name = agent.split(":", 1)
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        request_models = getattr(module, "REQUEST_MODELS", {})
        request_model = request_models.get(function_name) or getattr(module, "REQUEST_MODEL")
        request = request_model.model_validate(payload)
        arguments = [request]
        if len(inspect.signature(function).parameters) > 1:
            arguments.append(_internal_request())
        result = function(*arguments)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        update_job(job_id, Status="Completed", Result=_safe(result), CompletedAt=_now())
    except Exception as exc:

        retry_count = self.request.retries

        if retry_count < self.max_retries:

            update_job(
                job_id,
                Status="Retrying",
                Error={
                    "Type": type(exc).__name__,
                    "Message": str(exc)
                },
                RetryCount=retry_count + 1
            )

            raise self.retry(
                exc=exc,
                countdown=60
            )

        else:

            update_job(
                job_id,
                Status="Failed",
                Error={
                    "Type": type(exc).__name__,
                    "Message": str(exc)
                },
                RetryCount=retry_count
            )

            raise

    finally:
        _worker_execution.reset(token)
