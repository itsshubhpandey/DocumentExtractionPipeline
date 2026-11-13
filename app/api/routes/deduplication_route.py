from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.agents.deduplication.schemas.request import DeduplicationRequest
# Queue dispatch is temporarily disabled. The route runs directly.
# from app.infrastructure.celery_jobs import enqueue_or_continue


router = APIRouter(
    prefix="/requirement-deduplication",
    tags=["Requirement Deduplication Agent"],
)

REQUEST_MODEL = DeduplicationRequest


def _get_service():
    from app.agents.deduplication.services.service import (
        RequirementDeduplicationService,
    )

    return RequirementDeduplicationService()


@router.get("/health")
def requirement_deduplication_health() -> dict[str, str]:
    return {
        "agent": "RequirementDeduplicationAgent",
        "status": "healthy",
    }


def clean_bearer_token(token: str | None) -> str | None:
    """Clean a bearer token received in the JSON body."""

    if not token:
        return None
    cleaned_token = token.strip()
    if cleaned_token.lower().startswith("bearer "):
        cleaned_token = cleaned_token[7:].strip()
    return cleaned_token or None


@router.post(
    "/process",
    response_model=None,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_requirement_deduplication(
    request: DeduplicationRequest,
):
    # queued = enqueue_or_continue(
    #     "app.api.routes.deduplication_route:process_requirement_deduplication",
    #     {**request.model_dump(), "Token": request.Token},
    # )
    # if queued is not None:
    #     return queued

    try:
        bearer_token = clean_bearer_token(request.Token)
        if not bearer_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token is required in the API request body.",
            )

        return await _get_service().process(
            request=request,
            bearer_token=bearer_token,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Requirement deduplication failed: {exc}",
        ) from exc


@router.get("/{deduplication_id}")
async def get_deduplication_result(
    deduplication_id: str,
) -> dict[str, Any]:
    result = await _get_service().get_by_id(deduplication_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deduplication record was not found",
        )
    return result
