"""Shared orchestration imports for the tender document flow."""

import asyncio
from typing import Any

from app.api.routes.document_extractor_router import (
    MongoDocumentProcessRequest,
    process_documents_from_mongo_direct,
)
from app.utils.orchestration_logger import PersistentOrchestrationLogger


__all__ = [
    "process_documents_from_mongo_direct",
    "PersistentOrchestrationLogger",
]


async def process_tender_document_flow(
    *,
    company_id: str,
    tender_id: str,
    created_by: str | None = None, 
    user_id: str = "",
    user_name: str = "",
    project_id: str = "",
    bearer_token: str | None = None,
    is_regenerated: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
   

    company_id = str(company_id or "").strip()
    tender_id = str(tender_id or "").strip()
    if not company_id:
        raise ValueError("company_id cannot be empty")
    if not tender_id:
        raise ValueError("tender_id cannot be empty")

    operational_status = (
        "Regenerating" if is_regenerated else "Active"
    )
    resolved_created_by = (
        str(created_by or user_name or "").strip()
    )

    tracker = PersistentOrchestrationLogger(
        company_id=company_id,
        tender_id=tender_id,
        project_id=project_id,
        user_id=user_id,
        user_name=user_name,
        created_by=resolved_created_by,
        is_regenerated=is_regenerated,
        run_id=run_id,
    )

    try:
        await tracker.start_or_resume()
        run_id = tracker.run_id

        async def process_documents_step() -> dict[str, Any]:
            result = await process_documents_from_mongo_direct(
                MongoDocumentProcessRequest(
                    tender_id=tender_id,
                    company_id=company_id,
                    project_id=project_id,
                    user_id=user_id,
                    created_by=resolved_created_by,
                    user_name=user_name,
                    bearer_token=bearer_token,
                )
            )
            await tracker.set_input_documents(
                result.get("results", [])
            )
            failed_documents = [
                item
                for item in result.get("results", [])
                if str(item.get("status", "")).lower() == "failed"
            ]
            if failed_documents:
                raise RuntimeError(
                    "Document processing failed for "
                    f"{len(failed_documents)} document(s): "
                    f"{failed_documents}"
                )
            return result

        async def extract_requirements_step() -> dict[str, Any]:
            from app.agents.tender_requirement_agent.services.conditional_requirement_extraction import (
                process_new_tender_chunks,
            )

            result = await process_new_tender_chunks(
                company_id=company_id,
                tender_id=tender_id,
                user_id=user_id,
                user_name=user_name or resolved_created_by,
                status=operational_status,
            )
            if result.get("Status") in {"Failed", "PartiallyCompleted"}:
                raise RuntimeError(
                    "Requirement extraction did not complete: "
                    f"status={result.get('Status')}, "
                    f"failed_batches={result.get('FailedBatches', [])}"
                )
            return result

        async def deduplicate_requirements_step() -> dict[str, Any]:
            from app.agents.deduplication.services.conditional_requirement_deduplication import (
                process_new_requirements,
            )

            deduplication_context = {
                "CompanyId": company_id,
                "TenderId": tender_id,
                "IsRegenerate": is_regenerated,
                "UserId": user_id,
                "UserName": user_name or resolved_created_by,
                "ProjectId": project_id,
            }
            return await process_new_requirements(
                company_id=company_id,
                tender_id=tender_id,
                status=operational_status,
                context=deduplication_context,
                bearer_token=bearer_token,
                correlation_id=run_id,
            )

        async def evaluation_criteria_step() -> dict[str, Any]:
            from app.agents.EvaluationCriteriaAgent.graph.nodes import (
                EvaluationCriteriaAgent,
            )

            evaluation_usage_context = {
                "applicationName": "Tender",
                "runId": run_id,
                "companyId": company_id,
                "tenderId": tender_id,
                "projectId": project_id,
                "userId": user_id,
                "userName": user_name,
                "bearerToken": bearer_token,
                "run_id": run_id,
                "user_id": user_id,
                "project_id": project_id,
                "jwt_token": bearer_token,
            }
            evaluation_document_ids = sorted(
                {
                    str(item.get("DocumentId") or "").strip()
                    for item in await tracker.get_input_documents()
                    if str(item.get("Status", "")).lower() == "success"
                    and str(item.get("DocumentId") or "").strip()
                }
            )
            if not evaluation_document_ids:
                from app.agents.EvaluationCriteriaAgent.services.output_service import (
                    EvaluationCriteriaOutputService,
                )

                existing_output = EvaluationCriteriaOutputService.get_active(
                    company_id,
                    tender_id,
                )
                if existing_output:
                    return existing_output
                return {
                    "CompanyId": company_id,
                    "TenderId": tender_id,
                    "Status": "Skipped",
                    "Reason": "No newly processed tender documents were available for evaluation.",
                }
            return await asyncio.to_thread(
                EvaluationCriteriaAgent.execute,
                company_id=company_id,
                tender_id=tender_id,
                document_ids=evaluation_document_ids,
                is_regenerated=is_regenerated,
                created_by=resolved_created_by,
                run_id=run_id,
                usage_context=evaluation_usage_context,
                persist=True,
            )

        # Each call is a persistent checkpoint. On retry, run_step returns the
        # saved result for completed steps and starts at the first failed or
        # interrupted step.
        document_processing_result = await tracker.run_step(
            "document_processing",
            process_documents_step,
        )
        requirement_result = await tracker.run_step(
            "requirement_extraction",
            extract_requirements_step,
        )
        deduplication_result = await tracker.run_step(
            "requirement_deduplication",
            deduplicate_requirements_step,
        )
        evaluation_result = await tracker.run_step(
            "evaluation_criteria",
            evaluation_criteria_step,
        )

        run_document = await tracker.mark_run_completed()
        return {
            "RunId": run_id,
            "OrchestrationDocumentId": str(run_document["_id"]),
            "DocumentProcessing": document_processing_result,
            "Status": "Completed",
            "RequirementExtraction": requirement_result,
            "RequirementDeduplication": deduplication_result,
            "EvaluationCriteria": evaluation_result,
        }
    finally:
        tracker.close()


__all__.append("process_tender_document_flow")
