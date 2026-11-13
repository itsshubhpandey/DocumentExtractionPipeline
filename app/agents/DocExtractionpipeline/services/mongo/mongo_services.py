import os
import re
import ntpath
from typing import Optional
from datetime import datetime
from bson import ObjectId
import certifi
from dotenv import load_dotenv

from app.agents.DocExtractionpipeline.services.collection_names import (
    DocumentCollectionNames,
)
from app.infrastructure.motor_database import get_motor_database, get_motor_client

load_dotenv()

############################################### Helper Functions ###############################################
def _to_document_metadata(metadata: dict) -> dict:
    return {
        "documentTitle": metadata.get("document_title"),
        "DocumentClassification": metadata.get("document_type"),
        "appendixNumber": metadata.get("appendix_number"),
        "summary": metadata.get("summary"),
        "pageCount": metadata.get("page_count"),
        "tableCount": metadata.get("table_count"),
        # "imageCount": metadata.get("image_count"),
        "mainHeadings": metadata.get("main_headings", []),
        "sectionTitles": metadata.get("section_titles", []),
        "keywords": metadata.get("keywords", []),
        "topNgrams": metadata.get("top_ngrams", []),
        # "namedEntities": metadata.get("named_entities", []),
        "tableHeaders": metadata.get("table_headers", []),
        "processingStatus": metadata.get("processing_status"),
    }


def _first_value(document: dict, *keys: str):
    for key in keys:
        value = document.get(key)
        if value is not None:
            return value
    return None


def _first_present_key(document: dict, *keys: str) -> str | None:
    for key in keys:
        if key in document:
            return key
    return None


def _document_id_query(document_id) -> dict | None:
    """Build a DocumentId-only query that supports Mongo ObjectId and strings."""

    if document_id is None:
        return None

    values = [document_id]
    string_value = str(document_id)
    if string_value not in values:
        values.append(string_value)

    if ObjectId.is_valid(string_value):
        object_id_value = ObjectId(string_value)
        if object_id_value not in values:
            values.append(object_id_value)

    return {"DocumentId": {"$in": values}}


def _company_id_conditions(company_profile_id: str | None) -> list[dict]:
    return [
        {"CompanyProfileId": company_profile_id},
        {"companyProfileId": company_profile_id},
        {"CompanyProfileID": company_profile_id},
        {"companyProfileID": company_profile_id},
        {"CompanyID": company_profile_id},
        {"companyID": company_profile_id},
        {"CompanyId": company_profile_id},
        {"companyId": company_profile_id},
        {"companyid": company_profile_id},
    ]


def _tender_id_conditions(tender_id: str | None) -> list[dict]:
    return [
        {"TenderId": tender_id},
        {"tenderId": tender_id},
        {"TenderID": tender_id},
        {"tenderID": tender_id},
    ]


def _cpv_set_for_conditions(*values: str) -> list[dict]:
    fields = (
        "CPVsetfor",
        "CPVSetFor",
        "cpvSetFor",
        "cpvsetfor",
    )
    return [{field: value} for field in fields for value in values]


def _document_lookup_query(
    tender_id: str | None,
    company_profile_id: str | None,
) -> tuple[str, dict]:
    if not tender_id and not company_profile_id:
        raise ValueError("Pass tender_id, company_profile_id, or both")

    collection_names = DocumentCollectionNames.from_environment()
    if tender_id:
        filters = [{"$or": _tender_id_conditions(tender_id)}]
        if company_profile_id:
            filters.append({"$or": _company_id_conditions(company_profile_id)})

        query = filters[0] if len(filters) == 1 else {"$and": filters}
        return collection_names.tender_mongo, query

    return (
        collection_names.company_mongo,
        {"$or": _company_id_conditions(company_profile_id)},
    )


def _eligible_document_conditions() -> list[dict]:
    return [
        {
            "$or": [
                {"IsActive": False},
                {"IsActive": 0},
                {"isActive": False},
                {"isActive": 0},
            ],
        },
    ]


def _join_values(values: list) -> str:
    return ", ".join(str(value) for value in values if value)


def _to_document_description(metadata: dict) -> str:
    rows = [
        ("File Name", metadata.get("source_file_name") or metadata.get("file_name")),
        ("Document Title", metadata.get("document_title")),
        ("Main Headings", _join_values(metadata.get("main_headings", []))),
        ("Section Titles", _join_values(metadata.get("section_titles", []))),
        ("Keywords", _join_values(metadata.get("keywords", []))),
        ("Top Ngrams", _join_values(metadata.get("top_ngrams", []))),
        ("Table Headers", _join_values(metadata.get("table_headers", []))),
        ("Page Count", metadata.get("page_count")),
        ("Table Count", metadata.get("table_count")),
        # ("Image Count", metadata.get("image_count")),
    ]
    return "\n".join(
        f"{label}: {value}"
        for label, value in rows
        if value not in (None, "", [])
    )


def _to_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return value
    return value


def _find_matching_local_file(directory: str, document_name: str | None) -> str | None:
    if not document_name or not os.path.isdir(directory):
        return None

    schedule_match = re.search(r"\bD\.\d+(?:\.\d+)*\b", document_name, re.IGNORECASE)
    version_match = re.search(r"\bv\d+(?:\.\d+)*\b", document_name, re.IGNORECASE)

    if not schedule_match:
        return None

    schedule_token = schedule_match.group(0).lower()
    version_token = version_match.group(0).lower() if version_match else None
    matches = []

    for filename in os.listdir(directory):
        if not filename.lower().endswith(".pdf"):
            continue

        normalized_filename = filename.lower()
        if schedule_token not in normalized_filename:
            continue

        if version_token and version_token not in normalized_filename:
            continue

        matches.append(os.path.join(directory, filename))

    return matches[0] if len(matches) == 1 else None


def _has_drive_or_unc_root(file_path: str) -> bool:
    windows_drive, _ = ntpath.splitdrive(file_path)
    return bool(windows_drive) or file_path.startswith("\\\\")


def _apply_document_base_path(file_path: str) -> str:
    document_base_path = os.getenv("DOCUMENT_BASE_PATH", "").strip()
    has_windows_root = _has_drive_or_unc_root(file_path)

    # Drive-qualified and UNC paths are authoritative. A leading slash from
    # the wrapper (for example ``/Downloads/file.pdf``) can be root-relative
    # rather than an actual machine path, so resolve it against the configured
    # document base when the direct path does not exist.
    if has_windows_root:
        return file_path

    if not document_base_path:
        return file_path

    if os.path.isabs(file_path):
        if os.path.exists(file_path):
            return file_path

        base_candidate = os.path.join(
            document_base_path,
            file_path.lstrip("/\\"),
        )
        if os.path.exists(base_candidate) or os.path.exists(
            os.path.dirname(base_candidate)
        ):
            return base_candidate
        return file_path

    relative_file_path = file_path.lstrip("/\\")
    return os.path.join(document_base_path, relative_file_path)


def _find_matching_document_name(
    directory: str,
    document_name: str | None,
) -> str | None:
    """Find the requested document name in a local directory case-insensitively."""

    if not document_name or not os.path.isdir(directory):
        return None

    requested_name = document_name.casefold()
    for filename in os.listdir(directory):
        if filename.casefold() == requested_name:
            return os.path.join(directory, filename)
    return None


def _looks_like_file_path(file_path: str, document_name: str | None, blob_id: str | None) -> bool:
    if os.path.isdir(file_path):
        return False

    file_name = os.path.basename(file_path.rstrip("/\\"))
    if document_name and file_name.lower() == document_name.lower():
        return True

    if blob_id and file_name.lower() == os.path.basename(blob_id).lower():
        return True

    return bool(os.path.splitext(file_name)[1])


def _resolve_local_file_path(file_path: str, document_name: str | None, blob_id: str | None) -> str:
    if not file_path:
        return file_path

    file_path = _apply_document_base_path(file_path)

    if os.path.isfile(file_path):
        # The .NET wrapper stores files under generated names while preserving
        # the original upload name in DocumentName. FilePath is authoritative
        # when it resolves to an existing file; differing basenames are normal.
        return file_path

    if _looks_like_file_path(file_path, document_name, blob_id):
        return file_path

    if document_name:
        candidate = os.path.join(file_path, document_name)
        if os.path.exists(candidate):
            return candidate

    if blob_id:
        candidate = os.path.join(file_path, os.path.basename(blob_id))
        if os.path.exists(candidate):
            return candidate

    matching_file = _find_matching_local_file(file_path, document_name)
    if matching_file:
        return matching_file

    if document_name:
        return os.path.join(file_path, document_name)

    if blob_id:
        return os.path.join(file_path, os.path.basename(blob_id))

    return file_path


class MongoService:
    """
    Service for retrieving document taxonomy from MongoDB.
    """

    DATABASE_NAME = "DocAI"
    COLLECTION_NAME = "DocumentTaxonomies"

    def __init__(self):
        self.client = get_motor_client()
        # Use the same configured database as the upload route and all other
        # Mongo services. This keeps source inserts and post-Qdrant updates in
        # the same database when MONGO_DB_NAME is configured.
        self.db = get_motor_database()
        self.collection = self.db[self.COLLECTION_NAME]

    async def get_classification_set(
        self,
        tender_id: Optional[str] = None,
        company_profile_id: Optional[str] = None,
    ) -> list[dict]:
        if not tender_id and not company_profile_id:
            return []

        projection = {
            "_id": 0,
            "classificationSet.code": 1,
            "classificationSet.name": 1,
            "classificationSet.expectedSignals": 1,
        }

        filters = []
        if tender_id:
            filters.append({"$or": _tender_id_conditions(tender_id)})
            filters.append({"$or": _cpv_set_for_conditions("Tender", "tender")})
        else:
            filters.append({"$or": _cpv_set_for_conditions("Company", "company", "Compay", "compay")})

        if company_profile_id:
            filters.append({"$or": _company_id_conditions(company_profile_id)})

        query = filters[0] if len(filters) == 1 else {"$and": filters}
        document = await self.collection.find_one(query, projection)

        if not document:
            return []

        return document.get("classificationSet", [])

    async def save_document_classification(
        self,
        *,
        tender_id: str,
        company_id: Optional[str],
        document_metadata: dict,
        classifier_response: dict,
        mongo_document_id: Optional[str] = None,
        blob_id: Optional[str] = None,
        document_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        existing_document: Optional[dict] = None,
        created_by: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        """
        Saves the document metadata along with the classifier output.

        Args:
            tender_id: Tender ID.
            company_id: Company ID.
            document_metadata: Metadata extracted from the markdown.
            classifier_response: Output from the Document Classification Agent.

        Returns:
            Updated document id or lookup key.
        """

        collection_names = DocumentCollectionNames.from_environment()
        collection_name = collection_name or (
            collection_names.tender_mongo
            if tender_id
            else collection_names.company_mongo
        )
        collection = self.db[collection_name]
        now = datetime.utcnow()
        existing_document = existing_document or {}
        extracted_text_preview = document_metadata.get("summary")
        ocr_summary = _to_document_description(document_metadata)
        document_name = (
            document_metadata.get("source_file_name")
            or document_metadata.get("file_name")
            or _first_value(existing_document, "DocumentName", "documentName", "FileName", "fileName")
        )
        document_file_type = (
            document_metadata.get("source_file_extension") or ""
        ).upper()
        document_id_value = (
            document_id
            or _first_value(existing_document, "DocumentId", "DocumentID", "documentId", "fileId")
            or document_metadata.get("file_stem")
        )
        resolved_blob_id = blob_id or _first_value(existing_document, "BlobId", "blobId")
        created_by_value = (
            _first_value(existing_document, "CreatedBy", "createdBy")
            or created_by
        )

        query = _document_id_query(document_id_value)
        # BlobId is deliberately not used for updates because multiple
        # CPTenderDocuments records can reference the same blob.
        # if blob_id:
        #     query = {"BlobId": blob_id}

        if query:
            update_document = {
                "DocumentDescription": extracted_text_preview,
                "OCRSummary": ocr_summary,
                # "SummarizeContent": document_metadata.get("summary"),
                "DocumentClassification": classifier_response.get("document_type"),
                "DocumentFileType": document_file_type,
                "ClassificationScore": classifier_response.get("classification_confidence"),
                "DocumentRelevanceScore": classifier_response.get("relevance_score"),
                "ClassificationReason": classifier_response.get("reason"),
                "Status": "Processing",
                "IsLoadedByAI": False,
                "ModifiedAt": now,
            }
            if created_by_value:
                update_document["ModifiedBy"] = created_by_value
            if tender_id:
                update_document["TenderId"] = tender_id
            if company_id:
                company_id_field = _first_present_key(
                    existing_document,
                    "CompanyProfileId",
                    "companyProfileId",
                    "CompanyProfileID",
                    "companyProfileID",
                    "CompanyID",
                    "companyID",
                    "CompanyId",
                    "companyId",
                    "companyid",
                ) or "CompanyId"
                update_document[company_id_field] = company_id
            if resolved_blob_id:
                update_document["BlobId"] = resolved_blob_id
            if document_name:
                update_document["DocumentName"] = document_name
            update_document = {
                key: value for key, value in update_document.items() if value is not None
            }
            result = await collection.update_one(
                query,
                {
                    "$set": update_document,
                    # "$unset": {
                    #     "AIIsLoadByAgent": "",
                    #     "aiIsLoadByAgent": "",
                    # },
                },
                upsert=False,
            )
            if result.matched_count:
                return mongo_document_id or blob_id or ""

        raise ValueError(
            f"Mongo update matched 0 documents in {collection_name}. "
            f"mongo_document_id={mongo_document_id}, document_id={document_id_value}, blob_id={blob_id}"
        )

    async def mark_document_processed(
        self,
        *,
        tender_id: str | None,
        company_id: str | None,
        mongo_document_id: Optional[str] = None,
        blob_id: Optional[str] = None,
        document_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        modified_by: Optional[str] = None,
        document_metadata: Optional[dict] = None,
        classifier_response: Optional[dict] = None,
    ) -> bool:
        collection_names = DocumentCollectionNames.from_environment()
        collection_name = collection_name or (
            collection_names.tender_mongo
            if tender_id
            else collection_names.company_mongo
        )
        collection = self.db[collection_name]
        document_query = _document_id_query(document_id)
        # BlobId is deliberately not used for updates because multiple
        # CPTenderDocuments records can reference the same blob.
        # if blob_id:
        #     query = {"BlobId": blob_id}

        if not document_query:
            return False

        identity_filters = [document_query]
        if tender_id:
            identity_filters.append({"TenderId": tender_id})
        if company_id:
            identity_filters.append({"$or": _company_id_conditions(company_id)})
        query = (
            identity_filters[0]
            if len(identity_filters) == 1
            else {"$and": identity_filters}
        )

        document_metadata = document_metadata or {}
        classifier_response = classifier_response or {}
        document_description = document_metadata.get("summary") or ""
        ocr_summary = _to_document_description(document_metadata)
        document_file_type = (
            document_metadata.get("source_file_extension") or ""
        ).upper()

        update_document = {
            "DocumentDescription": document_description,
            "OCRSummary": ocr_summary,
            "IsLoadedByAI": False,
            "IsActive": True,
            "Status": "Processed",
            "DocumentClassification": (
                classifier_response.get("document_type")
                or "UNKNOWN_OR_UNCLEAR"
            ),
            "DocumentFileType": document_file_type,
            "ClassificationScore": (
                classifier_response.get("classification_confidence")
                if classifier_response.get("classification_confidence") is not None
                else 0
            ),
            "DocumentRelevanceScore": (
                classifier_response.get("relevance_score")
                if classifier_response.get("relevance_score") is not None
                else 0
            ),
            "ClassificationReason": classifier_response.get("reason") or "",
            "ModifiedAt": datetime.utcnow(),
        }
        if modified_by:
            update_document["ModifiedBy"] = modified_by

        result = await collection.update_one(
            query,
            {
                "$set": update_document,
                # "$unset": {
                #     "AIIsLoadByAgent": "",
                #     "aiIsLoadByAgent": "",
                # },
            },
            upsert=False,
        )
        if not result.matched_count:
            raise ValueError(
                f"Final Mongo update matched 0 documents in {collection_name}. "
                f"mongo_document_id={mongo_document_id}, document_id={document_id}, blob_id={blob_id}"
            )

        return True

    async def get_documents_for_processing(
        self,
        *,
        tender_id: str | None = None,
        company_profile_id: str | None = None,
    ) -> list[dict]:
        collection_name, base_query = _document_lookup_query(
            tender_id=tender_id,
            company_profile_id=company_profile_id,
        )
        query = {
            "$and": [
                base_query,
                *_eligible_document_conditions(),
            ],
        }

        processed_documents = []

        async for document in self.db[collection_name].find(query):
            file_path = _first_value(document, "FilePath", "filePath", "file_path")
            if not file_path:
                continue

            resolved_company_id = _first_value(
                document,
                "CompanyProfileId",
                "companyProfileId",
                "CompanyID",
                "companyID",
                "CompanyId",
                "companyId",
                "companyid",
            )
            resolved_tender_id = _first_value(
                document,
                "TenderId",
                "tenderId",
                "TenderID",
                "tenderID",
            )
            document_name = _first_value(
                document,
                "DocumentName",
                "documentName",
                "FileName",
                "fileName",
            )
            blob_id = _first_value(document, "BlobId", "blobId")
            document_id = _first_value(
                document,
                "DocumentId",
                "DocumentID",
                "documentId",
                "fileId",
            )
            created_by = _first_value(document, "CreatedBy", "createdBy")
            resolved_file_path = _resolve_local_file_path(
                file_path=file_path,
                document_name=document_name,
                blob_id=blob_id,
            )

            processed_documents.append(
                {
                    "mongo_document_id": str(document.get("_id")),
                    "blob_id": str(blob_id) if blob_id is not None else None,
                    "document_id": str(document_id) if document_id is not None else None,
                    "file_path": resolved_file_path,
                    "document_name": document_name or os.path.basename(file_path),
                    "company_id": str(resolved_company_id) if resolved_company_id is not None else None,
                    "tender_id": str(resolved_tender_id) if resolved_tender_id is not None else tender_id,
                    "created_by": str(created_by) if created_by is not None else None,
                    "collection_name": collection_name,
                    "existing_document": document,
                }
            )

        return processed_documents

    async def get_processing_match_counts(
        self,
        *,
        tender_id: str | None = None,
        company_profile_id: str | None = None,
    ) -> dict:
        collection_name, base_query = _document_lookup_query(
            tender_id=tender_id,
            company_profile_id=company_profile_id,
        )

        eligible_query = {
            "$and": [
                base_query,
                *_eligible_document_conditions(),
            ],
        }

        return {
            "collection": collection_name,
            "base_matches": await self.db[collection_name].count_documents(base_query),
            "eligible_matches": await self.db[collection_name].count_documents(eligible_query),
        }

    async def get_document_blob_paths(
        self,
        company_profile_id: str | None = None,
        tender_id: str | None = None,
    ):
        documents = []
        collection_names = DocumentCollectionNames.from_environment()

        if company_profile_id:
            # Replaced undefined self.company_collection with self.db lookups
            async for document in self.db[collection_names.company_mongo].find(
                    {
                        "$or": _company_id_conditions(company_profile_id),
                        "IsActive": False,
                        "IsLoadByAI": False
                        
                        
                    },
                    {
                        "_id": 0,
                        "BlobId": '1',
                        "FilePath": 1,
                        "DocumentName": 1
                    }
                ):
                documents.append(document)

        if tender_id:
            # Replaced undefined self.tender_collection with self.db lookups
            async for document in self.db[collection_names.tender_mongo].find(
                    {
                        "TenderId": tender_id,
                        "IsActive": False,
                        "IsLoadByAI": False
                    },
                    {
                        "_id": 0,
                        "BlobId": 1,
                        "FilePath": 1,
                        "DocumentName": 1
                    }
                ):
                documents.append(document)

        return [doc["BlobId"] for doc in documents]
    async def get_section_documents(self, company_id: str, section_id: str) -> list[dict]:
        collection_names = DocumentCollectionNames.from_environment()
        collection = self.db[collection_names.section_mongo]
        query = {
            "$and": [
                {"$or": _company_id_conditions(company_id)},
                {"SectionId": section_id},
                {"Documents.Status": False},
            ],
        }

        results = []
        async for section_doc in collection.find(query):
            mongo_id = str(section_doc.get("_id"))
            for file_entry in section_doc.get("Documents", []):
                if file_entry.get("Status") is not False:
                    continue
                raw_path = file_entry.get("Path")
                if not raw_path:
                    continue
                resolved_path = _apply_document_base_path(raw_path)
                results.append(
                    {
                        "mongo_id": mongo_id,
                        "section_id": section_doc.get("SectionId"),
                        "company_id": company_id,
                        "file_name": file_entry.get("FileName"),
                        "document_id": _first_value(
                            file_entry,
                            "DocumentId",
                            "DocumentID",
                            "documentId",
                            "fileId",
                        ),
                        "extension": file_entry.get("Extension"),
                        "path": resolved_path,
                    }
                )
        return results

    async def mark_section_document_processed(
        self,
        mongo_id: str,
        file_name: str,
        document_id: str,
    ) -> bool:
        collection_names = DocumentCollectionNames.from_environment()
        collection = self.db[collection_names.section_mongo]
        id_filters = [{"_id": mongo_id}]
        try:
            id_filters.append({"_id": ObjectId(mongo_id)})
        except Exception:
            pass

        result = await collection.update_one(
            {
                "$and": [
                    {"$or": id_filters},
                    {"Documents.FileName": file_name},
                ],
            },
            {
                "$set": {
                    "Documents.$.Status": True,
                    "Documents.$.DocumentId": document_id,
                    "ModifiedOn": datetime.utcnow(),
                },
            },
        )
        return result.matched_count > 0




# ################################################Test case #######################################################


# # mongo_service = MongoService()

# # blob_paths = mongo_service.get_document_blob_paths(
# #     tender_id="6a4506d2f4191032014315fd"
# # )

# # print(blob_paths)
#
