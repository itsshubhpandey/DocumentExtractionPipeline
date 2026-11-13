"""State used to skip already evaluated evidence chunks."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from app.infrastructure.database import get_database


DEFAULT_COLLECTION_NAME = "EvaluationCriteriaProcessedChunks"
DEFAULT_STATE_VERSION = "evaluation-criteria-v1"


def _collection():
    collection_name = os.getenv(
        "EC_PROCESSED_CHUNKS_COLLECTION",
        DEFAULT_COLLECTION_NAME,
    )
    return get_database()[collection_name]


def _identity(company_id: str, tender_id: str) -> dict[str, str]:
    return {
        "CompanyId": str(company_id),
        "TenderId": str(tender_id),
        "AgentVersion": os.getenv(
            "EC_INCREMENTAL_STATE_VERSION",
            DEFAULT_STATE_VERSION,
        ),
    }


def chunk_fingerprint(chunk: dict[str, Any]) -> str | None:
    """Return a stable identity that also detects changed chunk text."""

    document_id = chunk.get("DocumentId")
    chunk_id = chunk.get("ChunkId")
    if document_id in (None, "") or chunk_id in (None, ""):
        return None

    content = {
        "Text": chunk.get("Text", ""),
        "TableCaption": chunk.get("TableCaption"),
        "TableHeaders": chunk.get("TableHeaders"),
        "TableJson": chunk.get("TableJson"),
    }
    content_hash = hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"{document_id}::{chunk_id}::{content_hash}"


def get_processed_chunk_fingerprints(
    company_id: str,
    tender_id: str,
) -> set[str]:
    document = _collection().find_one(
        _identity(company_id, tender_id),
        {"ProcessedChunkFingerprints": 1},
    )
    if not document:
        return set()
    return {
        str(value)
        for value in document.get("ProcessedChunkFingerprints", [])
        if value
    }


def get_processed_document_ids(
    company_id: str,
    tender_id: str,
) -> set[str]:
    document = _collection().find_one(
        _identity(company_id, tender_id),
        {"ProcessedDocumentIds": 1},
    )
    if not document:
        return set()
    return {
        str(value)
        for value in document.get("ProcessedDocumentIds", [])
        if value
    }


def mark_processed_chunk_fingerprints(
    company_id: str,
    tender_id: str,
    fingerprints: set[str],
) -> None:
    if not fingerprints:
        return

    now = datetime.now(timezone.utc)
    _collection().update_one(
        _identity(company_id, tender_id),
        {
            "$addToSet": {
                "ProcessedChunkFingerprints": {
                    "$each": sorted(fingerprints),
                }
            },
            "$set": {"UpdatedAt": now},
            "$setOnInsert": {"CreatedAt": now},
        },
        upsert=True,
    )


def mark_processed_document_ids(
    company_id: str,
    tender_id: str,
    document_ids: set[str],
) -> None:
    normalized_document_ids = {
        str(document_id).strip()
        for document_id in document_ids
        if str(document_id).strip()
    }
    if not normalized_document_ids:
        return

    now = datetime.now(timezone.utc)
    _collection().update_one(
        _identity(company_id, tender_id),
        {
            "$addToSet": {
                "ProcessedDocumentIds": {
                    "$each": sorted(normalized_document_ids),
                }
            },
            "$set": {"UpdatedAt": now},
            "$setOnInsert": {"CreatedAt": now},
        },
        upsert=True,
    )


__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "chunk_fingerprint",
    "get_processed_chunk_fingerprints",
    "get_processed_document_ids",
    "mark_processed_chunk_fingerprints",
    "mark_processed_document_ids",
]
