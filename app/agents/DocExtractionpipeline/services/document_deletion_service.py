"""Coordinated deletion of Mongo source documents and Qdrant chunks."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from bson import ObjectId

from app.agents.DocExtractionpipeline.services.collection_names import (
    DocumentCollectionNames,
)
from app.agents.DocExtractionpipeline.services.qdrant.qdrant_services import (
    QdrantService,
)
from app.infrastructure.motor_database import get_motor_collection


class DocumentNotFoundError(LookupError):
    """Raised when an exact Mongo document does not exist."""


class DocumentQdrantDeleteError(RuntimeError):
    """Raised when Qdrant deletion fails before Mongo is changed."""


class DocumentMongoDeleteError(RuntimeError):
    """Raised when Mongo deletion fails after Qdrant deletion."""


class DocumentDeleteConflictError(RuntimeError):
    """Raised when a verified Mongo record changes before deletion."""


class TenderDocumentNotFoundError(DocumentNotFoundError):
    """Raised when the exact Mongo tender document does not exist."""


class TenderDocumentQdrantDeleteError(DocumentQdrantDeleteError):
    """Raised when Qdrant deletion fails before Mongo is changed."""


class TenderDocumentMongoDeleteError(DocumentMongoDeleteError):
    """Raised when Mongo deletion fails after Qdrant deletion."""


class TenderDocumentDeleteConflictError(DocumentDeleteConflictError):
    """Raised when the verified Mongo record changes before deletion."""


def _mongo_identifier_values(value: str) -> list[Any]:
    values: list[Any] = [value]
    if ObjectId.is_valid(value):
        values.append(ObjectId(value))
    return values


def _mongo_identifier_condition(
    field_names: tuple[str, ...],
    value: str,
) -> dict[str, Any]:
    values = _mongo_identifier_values(value)
    return {
        "$or": [
            {field_name: {"$in": values}}
            for field_name in field_names
        ]
    }


def tender_document_identity_query(
    *,
    company_id: str,
    tender_id: str,
    document_id: str,
) -> dict[str, Any]:
    """Match one Mongo tender document using all three identifiers."""

    return {
        "$and": [
            _mongo_identifier_condition(
                (
                    "CompanyProfileId",
                    "companyProfileId",
                    "CompanyProfileID",
                    "companyProfileID",
                    "CompanyID",
                    "companyID",
                    "CompanyId",
                    "companyId",
                    "companyid",
                ),
                company_id,
            ),
            _mongo_identifier_condition(
                ("TenderId", "tenderId", "TenderID", "tenderID"),
                tender_id,
            ),
            _mongo_identifier_condition(
                ("DocumentId", "DocumentID", "documentId", "fileId"),
                document_id,
            ),
        ]
    }


COMPANY_ID_FIELDS = (
    "CompanyProfileId",
    "companyProfileId",
    "CompanyProfileID",
    "companyProfileID",
    "CompanyID",
    "companyID",
    "CompanyId",
    "companyId",
    "companyid",
)
DOCUMENT_ID_FIELDS = ("DocumentId", "DocumentID", "documentId", "fileId")
SECTION_ID_FIELDS = ("SectionId", "SectionID", "sectionId", "sectionID")


def company_documents_identity_query(*, company_id: str) -> dict[str, Any]:
    """Match every Mongo company document belonging to one company."""

    return _mongo_identifier_condition(COMPANY_ID_FIELDS, company_id)


def section_document_identity_query(
    *,
    company_id: str,
    section_id: str,
    document_id: str,
) -> dict[str, Any]:
    """Match a SectionDocument parent containing one exact child document."""

    return {
        "$and": [
            _mongo_identifier_condition(COMPANY_ID_FIELDS, company_id),
            _mongo_identifier_condition(SECTION_ID_FIELDS, section_id),
            _mongo_identifier_condition(
                tuple(f"Documents.{field}" for field in DOCUMENT_ID_FIELDS),
                document_id,
            ),
        ]
    }


def _section_document_pull_condition(document_id: str) -> dict[str, Any]:
    values = _mongo_identifier_values(document_id)
    return {
        "$or": [
            {field_name: {"$in": values}}
            for field_name in DOCUMENT_ID_FIELDS
        ]
    }


class CompanyDocumentDeletionService:
    """Delete all normal company documents for one company."""

    def __init__(
        self,
        *,
        collection_names: DocumentCollectionNames | None = None,
        mongo_collection=None,
        qdrant_client=None,
    ) -> None:
        self.collection_names = (
            collection_names or DocumentCollectionNames.from_environment()
        )
        self.mongo_collection = (
            mongo_collection
            if mongo_collection is not None
            else get_motor_collection(self.collection_names.company_mongo)
        )
        self.qdrant_client = qdrant_client

    async def delete(self, *, company_id: str) -> dict[str, Any]:
        mongo_query = company_documents_identity_query(company_id=company_id)
        try:
            matched_documents = await self.mongo_collection.count_documents(
                mongo_query
            )
        except Exception as exc:
            raise DocumentMongoDeleteError(
                f"Unable to verify company documents in Mongo: {exc}"
            ) from exc
        if matched_documents == 0:
            raise DocumentNotFoundError(
                "No company documents matched the supplied company_id."
            )

        try:
            qdrant_result = await asyncio.to_thread(
                QdrantService.delete_company_document_chunks,
                company_id=company_id,
                client=self.qdrant_client,
                collection_name=self.collection_names.company_qdrant,
            )
        except Exception as exc:
            raise DocumentQdrantDeleteError(
                "Unable to delete company-document chunks from Qdrant; "
                f"Mongo was not changed: {exc}"
            ) from exc

        try:
            mongo_result = await self.mongo_collection.delete_many(mongo_query)
        except Exception as exc:
            raise DocumentMongoDeleteError(
                "Qdrant chunks were deleted, but company documents could not "
                f"be deleted from Mongo: {exc}"
            ) from exc
        if mongo_result.deleted_count == 0:
            raise DocumentDeleteConflictError(
                "Qdrant chunks were deleted, but the matching Mongo documents "
                "changed before deletion."
            )

        return {
            "success": True,
            "company_id": company_id,
            "mongo": {
                "collection": self.collection_names.company_mongo,
                "matched_count": matched_documents,
                "deleted_count": mongo_result.deleted_count,
            },
            "qdrant": qdrant_result,
        }


class SectionDocumentDeletionService:
    """Delete one child from a company SectionDocument and its chunks."""

    def __init__(
        self,
        *,
        collection_names: DocumentCollectionNames | None = None,
        mongo_collection=None,
        qdrant_client=None,
    ) -> None:
        self.collection_names = (
            collection_names or DocumentCollectionNames.from_environment()
        )
        self.mongo_collection = (
            mongo_collection
            if mongo_collection is not None
            else get_motor_collection(self.collection_names.section_mongo)
        )
        self.qdrant_client = qdrant_client

    async def delete(
        self,
        *,
        company_id: str,
        section_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        mongo_query = section_document_identity_query(
            company_id=company_id,
            section_id=section_id,
            document_id=document_id,
        )
        try:
            parent = await self.mongo_collection.find_one(
                mongo_query,
                {"_id": 1},
            )
        except Exception as exc:
            raise DocumentMongoDeleteError(
                f"Unable to verify the section document in Mongo: {exc}"
            ) from exc
        if parent is None:
            raise DocumentNotFoundError(
                "No section document matched the supplied company_id, "
                "section_id, and document_id."
            )

        try:
            qdrant_result = await asyncio.to_thread(
                QdrantService.delete_section_document_chunks,
                company_id=company_id,
                section_id=section_id,
                document_id=document_id,
                client=self.qdrant_client,
                collection_name=self.collection_names.company_qdrant,
            )
        except Exception as exc:
            raise DocumentQdrantDeleteError(
                "Unable to delete section-document chunks from Qdrant; "
                f"Mongo was not changed: {exc}"
            ) from exc

        try:
            mongo_result = await self.mongo_collection.update_one(
                {"$and": [{"_id": parent["_id"]}, mongo_query]},
                {
                    "$pull": {
                        "Documents": _section_document_pull_condition(
                            document_id
                        )
                    },
                    "$set": {"ModifiedOn": datetime.utcnow()},
                },
            )
        except Exception as exc:
            raise DocumentMongoDeleteError(
                "Qdrant chunks were deleted, but the nested section document "
                f"could not be deleted from Mongo: {exc}"
            ) from exc
        if mongo_result.modified_count != 1:
            raise DocumentDeleteConflictError(
                "Qdrant chunks were deleted, but the nested Mongo document "
                "changed before deletion."
            )

        return {
            "success": True,
            "company_id": company_id,
            "section_id": section_id,
            "document_id": document_id,
            "mongo": {
                "collection": self.collection_names.section_mongo,
                "deleted_count": 1,
                "parent_id": str(parent["_id"]),
            },
            "qdrant": qdrant_result,
        }


class TenderDocumentDeletionService:
    def __init__(
        self,
        *,
        collection_names: DocumentCollectionNames | None = None,
        mongo_collection=None,
        qdrant_client=None,
    ) -> None:
        self.collection_names = (
            collection_names or DocumentCollectionNames.from_environment()
        )
        self.mongo_collection = (
            mongo_collection
            if mongo_collection is not None
            else get_motor_collection(self.collection_names.tender_mongo)
        )
        self.qdrant_client = qdrant_client

    async def delete(
        self,
        *,
        company_id: str,
        tender_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        mongo_query = tender_document_identity_query(
            company_id=company_id,
            tender_id=tender_id,
            document_id=document_id,
        )

        try:
            mongo_document = await self.mongo_collection.find_one(
                mongo_query,
                {"_id": 1},
            )
        except Exception as exc:
            raise TenderDocumentMongoDeleteError(
                f"Unable to verify the Mongo tender document: {exc}"
            ) from exc
        if mongo_document is None:
            raise TenderDocumentNotFoundError(
                "No tender document matched the supplied company_id, "
                "tender_id, and document_id."
            )

        try:
            qdrant_result = await asyncio.to_thread(
                QdrantService.delete_tender_document_chunks,
                company_id=company_id,
                tender_id=tender_id,
                document_id=document_id,
                client=self.qdrant_client,
                collection_name=self.collection_names.tender_qdrant,
            )
        except Exception as exc:
            raise TenderDocumentQdrantDeleteError(
                "Unable to delete tender-document chunks from Qdrant; "
                f"the Mongo document was not deleted: {exc}"
            ) from exc

        try:
            mongo_result = await self.mongo_collection.delete_one(
                {
                    "$and": [
                        {"_id": mongo_document["_id"]},
                        mongo_query,
                    ]
                }
            )
        except Exception as exc:
            raise TenderDocumentMongoDeleteError(
                "Qdrant chunks were deleted, but the Mongo document "
                f"could not be deleted: {exc}"
            ) from exc

        if mongo_result.deleted_count != 1:
            raise TenderDocumentDeleteConflictError(
                "Qdrant chunks were deleted, but the Mongo document changed "
                "before deletion and was not removed."
            )

        return {
            "success": True,
            "company_id": company_id,
            "tender_id": tender_id,
            "document_id": document_id,
            "mongo": {
                "collection": self.collection_names.tender_mongo,
                "deleted_count": mongo_result.deleted_count,
            },
            "qdrant": qdrant_result,
        }


__all__ = [
    "CompanyDocumentDeletionService",
    "DocumentDeleteConflictError",
    "DocumentMongoDeleteError",
    "DocumentNotFoundError",
    "DocumentQdrantDeleteError",
    "SectionDocumentDeletionService",
    "TenderDocumentDeleteConflictError",
    "TenderDocumentDeletionService",
    "TenderDocumentMongoDeleteError",
    "TenderDocumentNotFoundError",
    "TenderDocumentQdrantDeleteError",
    "company_documents_identity_query",
    "section_document_identity_query",
    "tender_document_identity_query",
]
