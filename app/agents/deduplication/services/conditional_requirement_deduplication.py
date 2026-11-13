"""Helpers for running deduplication only for new requirements.

This module is intentionally separate from the existing deduplication service.
It reads source requirements, checks the existing
``TenderRequirementDeduplications`` records, and passes only requirements whose
source ``RequirementId`` has not already been deduplicated to the existing
deduplication agent.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from bson import ObjectId
from app.infrastructure.motor_database import (
    get_deduplication_motor_collection,
    get_requirement_extraction_motor_collection,
)

if TYPE_CHECKING:
    from app.agents.deduplication.requirement_deduplication_agent import (
        RequirementDeduplicationAgent,
    )


_REQUIREMENT_ID_FIELDS = (
    "RequirementId",
    "requirement_id",
    "requirementId",
    "Id",
    "id",
)


def _mongo_values(value: str) -> list[Any]:
    """Return string and ObjectId forms used by Mongo identifier fields."""

    values: list[Any] = [value]
    if ObjectId.is_valid(value):
        values.append(ObjectId(value))
    return values


def _company_tender_query(
    company_id: str,
    tender_id: str,
    status: str | None = None,
) -> dict[str, Any]:
    """Build a query that supports the field-name variants used in Mongo."""

    company_values = _mongo_values(company_id)
    tender_values = _mongo_values(tender_id)

    query: dict[str, Any] = {
        "$and": [
            {
                "$or": [
                    {"CompanyId": {"$in": company_values}},
                    {"companyId": {"$in": company_values}},
                    {"company_id": {"$in": company_values}},
                ]
            },
            {
                "$or": [
                    {"TenderId": {"$in": tender_values}},
                    {"tenderId": {"$in": tender_values}},
                    {"tender_id": {"$in": tender_values}},
                ]
            },
        ]
    }

    if status:
        query["$and"].append({"Status": status})

    return query


def _requirement_id(requirement: Any) -> str:
    """Read a source requirement ID without changing the input object."""

    if not isinstance(requirement, Mapping):
        return ""

    for field in _REQUIREMENT_ID_FIELDS:
        value = requirement.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def _add_requirement_ids(value: Any, target: set[str]) -> None:
    """Collect IDs from scalar, list, or nested Mongo values."""

    if isinstance(value, Mapping):
        direct_id = _requirement_id(value)
        if direct_id:
            target.add(direct_id)
            return

        for child in value.values():
            _add_requirement_ids(child, target)
        return

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for child in value:
            _add_requirement_ids(child, target)
        return

    if value is not None and str(value).strip():
        target.add(str(value).strip())


def extract_existing_requirement_ids(document: Mapping[str, Any]) -> set[str]:
    """Extract source IDs already represented by one deduplication document.

    The current output stores them in
    ``JsonOutput.DeduplicatedRequirements[].RequirementIds``. The alternate
    ``SourceRequirementIds`` and top-level forms are supported for older
    documents as well. ``CanonicalRequirementId`` is deliberately ignored.
    """

    existing_ids: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized_key = str(key).replace("_", "").lower()

                if normalized_key in {
                    "requirementids",
                    "sourcerequirementids",
                }:
                    _add_requirement_ids(child, existing_ids)
                elif normalized_key == "requirementid":
                    _add_requirement_ids(child, existing_ids)
                else:
                    walk(child)
            return

        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for child in value:
                walk(child)

    walk(document)
    return existing_ids


def filter_new_requirements(
    requirements: Iterable[Mapping[str, Any] | Any],
    existing_requirement_ids: Iterable[str],
) -> list[Any]:
    """Return only requirements not already in the deduplication collection.

    Requirements without a source ID are retained because they cannot be
    safely matched against an existing deduplication record.
    """

    existing_ids = {
        str(requirement_id).strip()
        for requirement_id in existing_requirement_ids
        if requirement_id is not None and str(requirement_id).strip()
    }

    new_requirements: list[Any] = []
    for requirement in requirements:
        requirement_id = _requirement_id(requirement)
        if not requirement_id or requirement_id not in existing_ids:
            new_requirements.append(requirement)

    return new_requirements


async def read_existing_requirement_ids(
    company_id: str,
    tender_id: str,
    *,
    collection: Any | None = None,
) -> set[str]:
    """Read already deduplicated source IDs for one company and tender."""

    company_id = str(company_id or "").strip()
    tender_id = str(tender_id or "").strip()
    if not company_id:
        raise ValueError("company_id cannot be empty")
    if not tender_id:
        raise ValueError("tender_id cannot be empty")

    mongo_collection = (
        collection
        if collection is not None
        else get_deduplication_motor_collection()
    )
    query = _company_tender_query(company_id, tender_id)

    documents = [document async for document in mongo_collection.find(query)]

    existing_ids: set[str] = set()
    for document in documents:
        existing_ids.update(extract_existing_requirement_ids(document))

    return existing_ids


async def prepare_new_requirements(
    company_id: str,
    tender_id: str,
    *,
    status: str | None = "Active",
    source_collection: Any | None = None,
    deduplication_collection: Any | None = None,
) -> dict[str, Any]:
    """Load source requirements and keep only not-yet-deduplicated IDs."""

    company_id = str(company_id or "").strip()
    tender_id = str(tender_id or "").strip()
    if not company_id:
        raise ValueError("company_id cannot be empty")
    if not tender_id:
        raise ValueError("tender_id cannot be empty")

    source = (
        source_collection
        if source_collection is not None
        else get_requirement_extraction_motor_collection()
    )
    # Keep this helper import lazy so ID filtering can be used without
    # initializing the LLM stack during module import.
    from app.agents.deduplication.services.service import (
        extract_requirements_from_document,
        remove_exact_duplicates,
    )

    source_query = _company_tender_query(
        company_id,
        tender_id,
        status,
    )
    source_documents = [
        document async for document in source.find(source_query)
    ]

    raw_requirements: list[Any] = []
    for document in source_documents:
        raw_requirements.extend(
            extract_requirements_from_document(document)
        )

    raw_requirements = remove_exact_duplicates(raw_requirements)
    existing_ids = await read_existing_requirement_ids(
        company_id,
        tender_id,
        collection=deduplication_collection,
    )
    new_requirements = filter_new_requirements(
        raw_requirements,
        existing_ids,
    )

    return {
        "CompanyId": company_id,
        "TenderId": tender_id,
        "Requirements": new_requirements,
        "ExistingRequirementIds": sorted(existing_ids),
        "TotalRequirements": len(raw_requirements),
        "NewRequirements": len(new_requirements),
        "SkippedRequirements": (
            len(raw_requirements) - len(new_requirements)
        ),
        "Status": (
            "Ready" if new_requirements else "NoNewRequirements"
        ),
    }


async def process_new_requirements(
    company_id: str,
    tender_id: str,
    *,
    status: str | None = "Active",
    context: Mapping[str, Any] | None = None,
    bearer_token: str | None = None,
    correlation_id: str | None = None,
    agent: RequirementDeduplicationAgent | None = None,
    source_collection: Any | None = None,
    deduplication_collection: Any | None = None,
) -> dict[str, Any]:
    """Send only new requirements to the agent and save a new Mongo document.

    The existing ``TenderRequirementDeduplications`` documents are never
    updated or replaced. The inserted document keeps the existing document
    structure used by the deduplication service.
    """

    prepared = await prepare_new_requirements(
        company_id=company_id,
        tender_id=tender_id,
        status=status,
        source_collection=source_collection,
        deduplication_collection=deduplication_collection,
    )

    if not prepared["Requirements"]:
        return {
            **prepared,
            "Result": None,
        }

    destination = (
        deduplication_collection
        if deduplication_collection is not None
        else get_deduplication_motor_collection()
    )

    if agent is None:
        from app.agents.deduplication.requirement_deduplication_agent import (
            RequirementDeduplicationAgent,
        )

        agent_instance = RequirementDeduplicationAgent()
    else:
        agent_instance = agent
    agent_context = dict(context or {})
    agent_context.setdefault("CompanyId", company_id)
    agent_context.setdefault("TenderId", tender_id)

    result = await agent_instance.deduplicate(
        raw_requirements=prepared["Requirements"],
        context=agent_context,
        bearer_token=bearer_token,
        correlation_id=correlation_id,
    )

    saved_document = await save_new_deduplication_document(
        collection=destination,
        company_id=company_id,
        tender_id=tender_id,
        result=result,
        context=agent_context,
    )

    return {
        **prepared,
        "Status": "Completed",
        "DeduplicationId": str(saved_document["_id"]),
        "Result": result,
    }


async def save_new_deduplication_document(
    *,
    collection: Any,
    company_id: str,
    tender_id: str,
    result: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save the filtered run, updating the active result when appropriate."""

    summary = result.get("Summary", {})
    deduplicated_requirements = result.get(
        "DeduplicatedRequirements",
        [],
    )

    if not isinstance(summary, Mapping):
        raise ValueError("Deduplication result contains an invalid Summary")
    if not isinstance(deduplicated_requirements, list):
        raise ValueError(
            "Deduplication result contains an invalid "
            "DeduplicatedRequirements list"
        )

    context_data = dict(context or {})
    is_regenerate = bool(
        context_data.get("IsRegenerate", False)
    )
    created_at = datetime.now(timezone.utc)

    document = {
        "CompanyId": company_id,
        "TenderId": tender_id,
        "IsRegenerate": is_regenerate,
        "UserId": str(
            context_data.get("UserId", "") or ""
        ),
        "UserName": str(
            context_data.get("UserName", "") or ""
        ),
        "ProjectId": str(
            context_data.get("ProjectId", "") or ""
        ),
        "Summary": dict(summary),
        "JsonOutput": {
            "DeduplicatedRequirements": deduplicated_requirements,
        },
        "Status": (
            "Regenerating" if is_regenerate else "Active"
        ),
        "Error": None,
        "CreatedAt": created_at,
        "UpdatedAt": created_at,
        "CompletedAt": created_at,
    }

    existing_active = None
    if not is_regenerate:
        existing_active = await collection.find_one(
            _company_tender_query(company_id, tender_id, status="Active"),
            sort=[("UpdatedAt", -1), ("_id", -1)],
        )

    if existing_active:
        existing_output = existing_active.get("JsonOutput") or {}
        existing_requirements = existing_output.get(
            "DeduplicatedRequirements",
            [],
        )
        if not isinstance(existing_requirements, list):
            existing_requirements = []

        merged_requirements: list[Any] = []
        seen_requirement_ids: set[str] = set()
        seen_fallback_values: set[str] = set()

        for requirement in [
            *existing_requirements,
            *deduplicated_requirements,
        ]:
            if not isinstance(requirement, Mapping):
                fallback_key = json.dumps(
                    requirement,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                if fallback_key in seen_fallback_values:
                    continue
                seen_fallback_values.add(fallback_key)
                merged_requirements.append(requirement)
                continue

            requirement_ids = requirement.get("RequirementIds") or []
            if not isinstance(requirement_ids, list):
                requirement_ids = [requirement_ids]
            normalized_ids = {
                str(value).strip()
                for value in requirement_ids
                if value is not None and str(value).strip()
            }
            if normalized_ids and normalized_ids & seen_requirement_ids:
                continue

            fallback_key = json.dumps(
                requirement,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            if not normalized_ids and fallback_key in seen_fallback_values:
                continue

            seen_requirement_ids.update(normalized_ids)
            seen_fallback_values.add(fallback_key)
            merged_requirements.append(dict(requirement))

        # Canonical IDs are generated per LLM batch, so renumber them after
        # merging to keep the single active document internally consistent.
        for index, requirement in enumerate(merged_requirements, start=1):
            if isinstance(requirement, Mapping):
                requirement["CanonicalRequirementId"] = f"CR-{index:04d}"

        previous_summary = existing_active.get("Summary") or {}
        if not isinstance(previous_summary, Mapping):
            previous_summary = {}
        previous_input_count = int(
            previous_summary.get("TotalInputRequirements", 0) or 0
        )
        current_input_count = int(
            summary.get(
                "TotalInputRequirements",
                len(deduplicated_requirements),
            )
            or 0
        )
        total_input_count = previous_input_count + current_input_count
        document["Summary"] = {
            **dict(previous_summary),
            "TotalInputRequirements": total_input_count,
            "TotalDeduplicatedRequirements": len(merged_requirements),
            "DuplicatesRemoved": max(
                0,
                total_input_count - len(merged_requirements),
            ),
        }
        document["JsonOutput"] = {
            "DeduplicatedRequirements": merged_requirements,
        }
        document["CreatedAt"] = (
            existing_active.get("CreatedAt") or document["CreatedAt"]
        )

        await collection.replace_one(
            {"_id": existing_active["_id"]},
            document,
        )
        return {
            **document,
            "_id": existing_active["_id"],
        }

    insert_result = await collection.insert_one(document)

    saved_document = {
        **document,
        "_id": insert_result.inserted_id,
    }
    return saved_document


__all__ = [
    "extract_existing_requirement_ids",
    "filter_new_requirements",
    "read_existing_requirement_ids",
    "prepare_new_requirements",
    "process_new_requirements",
    "save_new_deduplication_document",
]
