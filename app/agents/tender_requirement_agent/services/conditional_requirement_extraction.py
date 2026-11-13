
from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any

from app.agents.tender_requirement_agent.services.requirement_repository_mongo import (
    requirement_repository,
)
from app.agents.tender_requirement_agent.utils.helper import create_batches



def _value(chunk: Mapping[str, Any], *names: str) -> str:
    """Return the first non-empty value for the supplied field names."""

    for name in names:
        value = chunk.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


async def filter_new_chunks(
    chunks: Iterable[Mapping[str, Any]],
    company_id: str,
    tender_id: str,
    *,
    collection: Any | None = None,
) -> list[dict[str, Any]]:
    """Return only chunks not already processed for this company and tender.

    A processed chunk is identified by the compound key
    ``CompanyId + TenderId + DocumentId + ChunkId`` in the existing MongoDB
    chunk-status collection. Chunks without either source identifier are kept
    because they cannot be safely deduplicated.
    """

    company_id = str(company_id or "").strip()
    tender_id = str(tender_id or "").strip()

    if not company_id:
        raise ValueError("company_id cannot be empty")
    if not tender_id:
        raise ValueError("tender_id cannot be empty")

    candidate_chunks = [dict(chunk) for chunk in chunks]
    identifiable_keys = set()
    for chunk in candidate_chunks:
        document_id = _value(chunk, "DocumentId", "document_id")
        chunk_id = _value(chunk, "ChunkId", "chunk_id")
        if document_id and chunk_id:
            identifiable_keys.add((document_id, chunk_id))

    if not identifiable_keys:
        return candidate_chunks

    mongo_collection = (
        collection
        if collection is not None
        else requirement_repository.chunks_collection
    )
    document_ids = sorted({document_id for document_id, _ in identifiable_keys})
    chunk_ids = sorted({chunk_id for _, chunk_id in identifiable_keys})

    query = {
        "CompanyId": company_id,
        "TenderId": tender_id,
        "DocumentId": {"$in": document_ids},
        "ChunkId": {"$in": chunk_ids},
    }
    projection = {
        "_id": 0,
        "CompanyId": 1,
        "TenderId": 1,
        "DocumentId": 1,
        "ChunkId": 1,
    }

    processed_keys = set()
    async for record in mongo_collection.find(query, projection):
        processed_keys.add(
            (
                _value(record, "DocumentId", "document_id"),
                _value(record, "ChunkId", "chunk_id"),
            )
        )

    new_chunks: list[dict[str, Any]] = []
    for chunk in candidate_chunks:
        key = (
            _value(chunk, "DocumentId", "document_id"),
            _value(chunk, "ChunkId", "chunk_id"),
        )

        if not key[0] or not key[1] or key not in processed_keys:
            new_chunks.append(chunk)

    return new_chunks


async def fetch_new_chunks(
    company_id: str,
    tender_id: str,
    *,
    retriever: Any | None = None,
    collection: Any | None = None,
) -> dict[str, Any]:
    """Retrieve tender chunks from Qdrant and keep only new chunks.

    The retriever still fetches all chunks for the tender. MongoDB is then
    used as the processed-chunk source of truth before any chunk is sent to
    the existing requirement-agent graph.
    """

    if retriever is None:
        from app.agents.tender_requirement_agent.services.tender_retriever_qdrant import (
            tender_retriever,
        )

        active_retriever = tender_retriever
    else:
        active_retriever = retriever
    retrieval_result = active_retriever.retrieve_chunks(tender_id=tender_id)
    retrieved_chunks = retrieval_result.get("chunks", [])
    new_chunks = await filter_new_chunks(
        retrieved_chunks,
        company_id=company_id,
        tender_id=tender_id,
        collection=collection,
    )

    return {
        **retrieval_result,
        "total_chunks": len(retrieved_chunks),
        "new_chunks": len(new_chunks),
        "skipped_chunks": len(retrieved_chunks) - len(new_chunks),
        "chunks": new_chunks,
    }


async def process_new_tender_chunks(
    company_id: str,
    tender_id: str,
    user_id: str,
    user_name: str,
    status: str,
    *,
    processing_batch_size: int = 50,
    max_concurrency: int = 20,
    batch_parallelism: int = 4,
    retriever: Any | None = None,
    collection: Any | None = None,
) -> dict[str, Any]:
    """Process only new chunks through the existing requirement workflow.

    This wrapper does not duplicate or replace the existing LangGraph nodes.
    Each new batch is passed to ``RequirementService.process_batch``; that
    method continues to run the existing sequence:

    ``rule_filter -> detect_and_extract -> process_requirements -> save_chunk_status``

    ``process_requirements`` continues to call ``build_requirement_document``
    and ``bulk_upsert_requirements`` through the existing graph implementation.
    """

    if processing_batch_size <= 0:
        raise ValueError("processing_batch_size must be greater than zero")

    retrieval = await fetch_new_chunks(
        company_id=company_id,
        tender_id=tender_id,
        retriever=retriever,
        collection=collection,
    )
    chunks = retrieval["chunks"]

    base_response = {
        "CompanyId": company_id,
        "TenderId": tender_id,
        "TotalChunks": retrieval["total_chunks"],
        "NewChunks": retrieval["new_chunks"],
        "SkippedChunks": retrieval["skipped_chunks"],
        "ProcessedChunks": 0,
        "TotalRequirements": 0,
        "ChunksWithRequirements": 0,
        "ChunksWithoutRequirements": 0,
        "FailedBatches": [],
        "TokenUsage": {},
    }

    if not chunks:
        return {
            **base_response,
            "Status": (
                "NoChunksFound"
                if retrieval["total_chunks"] == 0
                else "NoNewChunks"
            ),
        }

    # Import lazily because the existing requirement service initializes its
    # LLM configuration during module import. Filtering helpers should remain
    # usable without opening an LLM or MongoDB connection prematurely.
    from app.agents.tender_requirement_agent.services.requirement_service import (
        RequirementService,
    )

    service = RequirementService(
        processing_batch_size=processing_batch_size,
        max_concurrency=max_concurrency,
        batch_parallelism=batch_parallelism,
    )
    tasks = [
        service.process_batch(
            batch=batch,
            company_id=company_id,
            user_id=user_id,
            user_name=user_name,
            status=status,
        )
        for batch in create_batches(chunks, processing_batch_size)
    ]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_results: list[dict[str, Any]] = []
    failed_batches: list[dict[str, Any]] = []
    for index, result in enumerate(batch_results):
        if isinstance(result, Exception):
            failed_batches.append(
                {
                    "batch_index": index,
                    "error": str(result),
                }
            )
        else:
            successful_results.append(result)

    processed_chunks = 0
    total_requirements = 0
    chunks_with_requirements = 0
    chunks_without_requirements = 0

    for batch_result in successful_results:
        for chunk in batch_result.get("chunks", []):
            processed_chunks += 1
            requirement_count = len(chunk.get("requirements", []))
            total_requirements += requirement_count
            if requirement_count:
                chunks_with_requirements += 1
            else:
                chunks_without_requirements += 1

    token_usage = service.aggregate_token_usage(successful_results)
    if processed_chunks == len(chunks):
        business_status = "Completed"
    elif processed_chunks == 0:
        business_status = "Failed"
    else:
        business_status = "PartiallyCompleted"

    return {
        **base_response,
        "ProcessedChunks": processed_chunks,
        "TotalRequirements": total_requirements,
        "ChunksWithRequirements": chunks_with_requirements,
        "ChunksWithoutRequirements": chunks_without_requirements,
        "FailedBatches": failed_batches,
        "Status": business_status,
        "TokenUsage": token_usage,
    }
