"""Detect whether evaluation-related tender documents are available in Qdrant."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

from qdrant_client import QdrantClient, models


DEFAULT_QDRANT_COLLECTION = "CPTenderDoc"

DEFAULT_EVALUATION_DOCUMENT_TYPES = {
    "EvaluationMethodology",
    "AwardCriteria",
    "InstructionsToTenderers",
    "InvitationToTender",
    "TenderSpecification",
    "ResponseQuestionnaire",
    "PricingSchedule",
    "CommercialSchedule",
    "Clarification",
    "Addendum",
    "Appendix",
    "TermsAndConditions",
}

DEFAULT_EVALUATION_MARKERS = (
    "evaluation criteria",
    "award criteria",
    "evaluation methodology",
    "award methodology",
    "scoring methodology",
    "scoring scale",
    "evaluation matrix",
    "quality weighting",
    "price weighting",
    "commercial evaluation",
    "pricing formula",
    "pass/fail",
    "pass fail",
    "minimum threshold",
    "minimum score",
    "marks available",
)


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _text(value: Any) -> str:
    return str(value or "").strip()


def _create_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if not url:
        raise ValueError("QDRANT_URL is required")

    return QdrantClient(
        url=url,
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=float(os.getenv("QDRANT_SEARCH_TIMEOUT", "60")),
    )


def _iter_chunks(
    client: QdrantClient,
    collection_name: str,
    query_filter: models.Filter,
    max_chunks: int,
) -> Iterable[dict[str, Any]]:
    """Scroll matching Qdrant payloads without using embeddings or the LLM."""

    offset: Any = None
    returned = 0

    while returned < max_chunks:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=min(100, max_chunks - returned),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            yield payload
            returned += 1
            if returned >= max_chunks:
                break

        if offset is None:
            break


def _evaluation_match(
    payload: dict[str, Any],
    allowed_document_types: set[str],
    markers: tuple[str, ...],
) -> tuple[bool, list[str]]:
    document_type = _text(payload.get("DocumentType"))
    document_type_normalised = _normalise(document_type)
    reasons: list[str] = []

    if document_type_normalised in allowed_document_types:
        reasons.append("DocumentType")

    searchable_text = " ".join(
        _text(payload.get(field))
        for field in (
            "DocumentName",
            "DocumentType",
            "RelatedSection",
            "SectionHeading",
            "TableCaption",
            "Text",
        )
    ).lower()

    matched_markers = [
        marker
        for marker in markers
        if marker.lower() in searchable_text
    ]
    if matched_markers:
        reasons.extend(
            f"content:{marker}"
            for marker in matched_markers[:5]
        )

    return bool(reasons), reasons


def check_evaluation_document_presence(
    company_id: str,
    tender_id: str,
    *,
    qdrant_client: QdrantClient | None = None,
    collection_name: str | None = None,
    evaluation_document_types: Iterable[str] | None = None,
    evaluation_markers: Iterable[str] | None = None,
    max_chunks: int = 5000,
) -> dict[str, Any]:
    """Check whether a tender has an evaluation-related document in Qdrant.

    Qdrant is the source of truth for this check because the evaluation agent
    consumes processed chunks from ``CPTenderDoc``. A document is considered
    evaluation-related when its ``DocumentType`` is an allowed type or its
    metadata/text contains an explicit evaluation marker.
    """

    company_id = _text(company_id)
    tender_id = _text(tender_id)
    if not company_id:
        raise ValueError("company_id cannot be empty")
    if not tender_id:
        raise ValueError("tender_id cannot be empty")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than zero")

    active_client = qdrant_client or _create_qdrant_client()
    active_collection = (
        collection_name
        or os.getenv("EC_QDRANT_COLLECTION_NAME")
        or DEFAULT_QDRANT_COLLECTION
    )
    allowed_types = {
        _normalise(value)
        for value in (
            evaluation_document_types
            or DEFAULT_EVALUATION_DOCUMENT_TYPES
        )
        if _text(value)
    }
    markers = tuple(
        _text(value).lower()
        for value in (
            evaluation_markers or DEFAULT_EVALUATION_MARKERS
        )
        if _text(value)
    )

    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="CompanyId",
                match=models.MatchValue(value=company_id),
            ),
            models.FieldCondition(
                key="TenderId",
                match=models.MatchValue(value=tender_id),
            ),
        ]
    )

    uploaded_document_ids: set[str] = set()
    evaluation_document_ids: set[str] = set()
    document_types: set[str] = set()
    matched_reasons: dict[str, list[str]] = {}
    matching_chunk_count = 0
    chunk_count = 0

    for payload in _iter_chunks(
        active_client,
        active_collection,
        query_filter,
        max_chunks,
    ):
        chunk_count += 1
        document_id = _text(payload.get("DocumentId"))
        document_type = _text(payload.get("DocumentType"))

        if document_id:
            uploaded_document_ids.add(document_id)
        if document_type:
            document_types.add(document_type)

        is_match, reasons = _evaluation_match(
            payload,
            allowed_types,
            markers,
        )
        if not is_match:
            continue

        matching_chunk_count += 1
        if document_id:
            evaluation_document_ids.add(document_id)
            matched_reasons.setdefault(document_id, [])
            for reason in reasons:
                if reason not in matched_reasons[document_id]:
                    matched_reasons[document_id].append(reason)

    if not uploaded_document_ids:
        status = "NO_DOCUMENTS"
    elif evaluation_document_ids:
        status = "EVALUATION_DOCUMENT_FOUND"
    else:
        status = "NO_EVALUATION_DOCUMENT"

    return {
        "CompanyId": company_id,
        "TenderId": tender_id,
        "Collection": active_collection,
        "Status": status,
        "HasUploadedDocuments": bool(uploaded_document_ids),
        "HasEvaluationDocument": bool(evaluation_document_ids),
        "UploadedDocumentCount": len(uploaded_document_ids),
        "EvaluationDocumentCount": len(evaluation_document_ids),
        "ChunkCountScanned": chunk_count,
        "MatchingChunkCount": matching_chunk_count,
        "UploadedDocumentIds": sorted(uploaded_document_ids),
        "EvaluationDocumentIds": sorted(evaluation_document_ids),
        "DocumentTypes": sorted(document_types),
        "MatchedReasons": matched_reasons,
    }


__all__ = ["check_evaluation_document_presence"]
