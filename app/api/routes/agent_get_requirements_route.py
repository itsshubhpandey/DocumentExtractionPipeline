import time

from fastapi import APIRouter, HTTPException

from app.agents.tender_requirement_agent.schemas.request import RequirementRequest
# Queue dispatch is temporarily disabled. The route runs directly.
# from app.infrastructure.celery_jobs import enqueue_or_continue
from app.infrastructure.logger import Logging


logger = Logging(
    agent_name="Requirements_Detection_Extraction_And_Intent_Agent",
    source_module="agent_requirement_route",
)

router = APIRouter(
    prefix="/api/v1/agents",
    tags=["Requirement Agent"],
)

REQUEST_MODEL = RequirementRequest


def _get_service():
    from app.agents.tender_requirement_agent.services.requirement_service import (
        RequirementService,
    )

    return RequirementService()


@router.get("/get_requirements/health")
def requirement_agent_health() -> dict[str, str]:
    return {
        "agent": "Requirements_Detection_And_Intent_Agent",
        "status": "healthy",
    }


@router.post("/get_requirements", status_code=202)
async def process_requirement(
    request: RequirementRequest,
):
    # queued = enqueue_or_continue(
    #     "app.api.routes.agent_get_requirements_route:process_requirement",
    #     {**request.model_dump(), "JwtToken": request.JwtToken},
    # )
    # if queued is not None:
    #     return queued

    tracking_token = logger.start(
        message="Requirement processing started",
        event_type="RequirementProcessingStarted",
    )
    start_time = time.perf_counter()
    processing_status = "Regenerating" if request.IsRegenerate else "Active"

    try:
        result = await _get_service().process_tender(
            company_id=request.CompanyId,
            tender_id=request.TenderId,
            user_id=request.UserId,
            user_name=request.UserName,
            status=processing_status,
        )

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        try:
            from app.infrastructure.token_usage_logger import TokenUsageService

            await TokenUsageService.log_agent_usage(
                request=request,
                token_usage=result.get("TokenUsage", {}),
                duration_ms=duration_ms,
                correlation_id=request.ProjectId,
                application_name="Tender",
                source_ids=[request.TenderId],
                purpose="Requirement Extraction",
                method="POST",
                agent_name="Requirements_Detection_And_Intent_Agent",
                usage_type="LLM",
            )
        except Exception as log_ex:
            logger.log(
                message="Failed to log token usage",
                event_type="TokenUsageLoggingFailed",
                is_success=False,
                payload={
                    "company_id": request.CompanyId,
                    "tender_id": request.TenderId,
                    "error": str(log_ex),
                },
                correlation_id=tracking_token["correlation_id"],
            )

        logger.end(
            tracking_token=tracking_token,
            is_success=True,
            message="Requirement processing completed",
            event_type="RequirementProcessingCompleted",
            payload={
                "company_id": request.CompanyId,
                "tender_id": request.TenderId,
                "status": result.get("Status"),
                "total_chunks": result.get("TotalChunks"),
                "processed_chunks": result.get("ProcessedChunks"),
            },
        )
        return result
    except Exception as exc:
        logger.end(
            tracking_token=tracking_token,
            is_success=False,
            message="Requirement processing failed",
            event_type="RequirementProcessingFailed",
            payload={
                "company_id": request.CompanyId,
                "tender_id": request.TenderId,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
