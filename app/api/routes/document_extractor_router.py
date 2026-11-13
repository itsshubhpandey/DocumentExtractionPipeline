import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from bson import ObjectId
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Any, Annotated, List, Literal, Optional

from dotenv import load_dotenv
from app.agents.DocExtractionpipeline.services.collection_names import (
    DocumentCollectionNames,
)
from app.agents.DocExtractionpipeline.services.document_deletion_service import (
    CompanyDocumentDeletionService,
    DocumentDeleteConflictError,
    DocumentMongoDeleteError,
    DocumentNotFoundError,
    DocumentQdrantDeleteError,
    SectionDocumentDeletionService,
    TenderDocumentDeleteConflictError,
    TenderDocumentDeletionService,
    TenderDocumentMongoDeleteError,
    TenderDocumentNotFoundError,
    TenderDocumentQdrantDeleteError,
)
from app.infrastructure.motor_database import get_motor_collection


router = APIRouter(
    prefix="/documents",
    tags=["Document Extraction"]
)


class DocumentMetadata(BaseModel):
    pdf_path: str
    document_scope: Literal["company", "tender"]

    created_by: Optional[str] = None
    user_name: Optional[str] = None
    created_at: str
    tender_id: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    purpose: Optional[str] = None

class DocumentBatchRequest(BaseModel):
    documents: List[DocumentMetadata]


class UploadedDocumentMetadata(BaseModel):
    document_scope: Literal["company", "tender"]
    company_id: Optional[str] = None

    created_by: Optional[str] = None
    user_name: Optional[str] = None
    created_at: str
    tender_id: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    purpose: Optional[str] = None
    bearer_token: Optional[str] = None


class MongoDocumentProcessRequest(BaseModel):
    tender_id: Optional[str] = None
    company_id: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    created_by: Optional[str] = None
    user_name: Optional[str] = None
    bearer_token: Optional[str] = None


class TenderDocumentDeleteRequest(BaseModel):
    company_id: str
    tender_id: str
    document_id: str


class CompanyDocumentDeleteRequest(BaseModel):
    company_id: str


class SectionDocumentDeleteRequest(BaseModel):
    company_id: str
    section_id: str
    document_id: str


def clean_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _mongo_json(value: Any) -> Any:
    """Convert Mongo values to the Extended JSON shape shown in the UI."""

    if isinstance(value, ObjectId):
        return {"$oid": str(value)}
    if isinstance(value, datetime):
        return {"$date": value.isoformat().replace("+00:00", "Z")}
    if isinstance(value, dict):
        return {key: _mongo_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mongo_json(item) for item in value]
    return value


def _uploaded_document_payload(
    *,
    document_id: str,
    tender_id: str,
    company_id: str,
    project_id: str,
    blob_id: str,
    ocid: str | None,
    document_name: str,
    file_path: str,
    created_by: str,
    document_description: str,
    ocr_summary: str,
    now: datetime,
) -> dict[str, Any]:
    """Build the inactive source record consumed by the OCR pipeline."""

    return {
        "DocumentId": document_id,
        "TenderId": tender_id,
        "CompanyId": company_id,
        "ProjectId": project_id,
        "BlobId": blob_id,
        "Ocid": ocid,
        "DocumentName": document_name,
        "DocumentDescription": document_description,
        "OCRSummary": ocr_summary,
        "FilePath": file_path,
        "Status": "Uploaded",
        "DocumentClassification": "UNKNOWN_OR_UNCLEAR",
        "DocumentFileType": Path(document_name).suffix.lstrip(".").upper(),
        "ClassificationScore": 0,
        "DocumentRelevanceScore": 0,
        "IsLoadedByAI": False,
        "SourceBaseURL": os.getenv("DOCUMENT_SOURCE_BASE_URL") or None,
        "ClassificationReason": "Awaiting OCR processing.",
        "IsActive": False,
        "CreatedAt": now,
        "ModifiedAt": now,
        "CreatedBy": created_by,
        "ModifiedBy": created_by,
    }


# Old code: this worked when the API received pdf_path values in JSON.
# @router.post("/process")
# def process_documents(request: DocumentBatchRequest):
#
#     results = []
#
#     for document in request.documents:
#         try:
#             document_id = process_document(
#                 pdf_path=document.pdf_path,
#                 document_scope=document.document_scope,
#                 company_id=document.company_id,
#                 tender_id=document.tender_id,
#                 document_name=document.document_name,
#                 title=document.title,
#                 document_type=document.document_type,
#                 created_by=document.created_by,
#                 created_at=document.created_at,
#             )
#
#             results.append({
#                 "document_name": document.document_name,
#                 "document_id": document_id,
#                 "status": "Success"
#             })
#
#         except Exception as e:
#             results.append({
#                 "document_name": document.document_name,
#                 "status": "Failed",
#                 "error": str(e)
#             })
#
#     return {
#         "success": True,
#         "total_documents": len(request.documents),
#         "results": results
#     }

def parse_documents_metadata(documents: str) -> List[UploadedDocumentMetadata]:
    try:
        documents_data = json.loads(documents)
        if isinstance(documents_data, dict):
            documents_data = [documents_data]
        if not isinstance(documents_data, list):
            raise ValueError("documents must be a JSON object or a JSON array")
        return [UploadedDocumentMetadata(**document) for document in documents_data]
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid documents metadata JSON: {e}",
        )


@router.post("/process")
async def process_documents(
    files: Annotated[List[UploadFile], File()],
    documents: Annotated[str, Form()],
):
    run_id = str(uuid4())
    print(f"RunId: {run_id}")

    metadata_list = parse_documents_metadata(documents)

    if len(metadata_list) == 1 and len(files) > 1:
        metadata_list = metadata_list * len(files)

    if len(files) != len(metadata_list):
        raise HTTPException(
            status_code=400,
            detail=(
                "Number of uploaded files must match number of document metadata records, "
                "or pass a single metadata object to apply it to every file."
            ),
        )

    input_dir = Path(
        os.getenv(
            "DOCUMENT_UPLOAD_DIRECTORY",
            "app/infrastrature/documents/input_doc",
        )
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for file, document in zip(files, metadata_list):
        source_filename = Path(file.filename or "document.pdf").name
        saved_path = input_dir / f"{uuid4().hex}_{source_filename}"

        try:
            from app.agents.DocExtractionpipeline.pipeline.document_extraction_pipeline import (
                process_document,
            )

            with saved_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            source_id_type = (
                "tender"
                if document.tender_id
                else "company"
                if document.company_id
                else None
            )
            source_id = document.tender_id or document.company_id
            usage_context = {
                "applicationName": "Tender",
                "runId": run_id,
                "companyId": document.company_id,
                "tenderId": document.tender_id,
                "projectId": document.project_id,
                "userId": document.user_id,
                "userName": clean_optional_string(document.user_name),
                "purpose": document.purpose or "document_upload",
                "bearerToken": clean_optional_string(document.bearer_token),
                "sourceIds": [
                    {
                        "sourceIdType": source_id_type,
                        "id": source_id,
                    }
                ],
            }

            document_id = await process_document(
                pdf_path=str(saved_path),
                document_scope=document.document_scope,
                company_id=document.company_id,
                tender_id=document.tender_id,
                # document_type=document.document_type,
                created_by=(
                    clean_optional_string(document.created_by)
                    or clean_optional_string(document.user_name)
                ),
                created_at=document.created_at,
                source_filename=source_filename,
                usage_context=usage_context,
            )

            results.append({
                "document_name": source_filename,
                "document_id": document_id,
                "status": "Success"
            })

        except Exception as e:
            results.append({
                "document_name": source_filename,
                "status": "Failed",
                "error": str(e)
            })

        finally:
            await file.close()

    return {
        "success": True,
        "runId": run_id,
        "total_documents": len(metadata_list),
        "results": results
    }


@router.post("/upload/tender-document")
async def upload_tender_document(
    file: UploadFile = File(...),
    tender_id: str = Form(...),
    company_id: str = Form(...),
    document_id: str = Form(""),
    ocid: str = Form(""),
    project_id: str = Form(""),
    created_by: str = Form(""),
    user_name: str = Form(""),
    document_description: str = Form(""),
    ocr_summary: str = Form(""),
    blob_id: str = Form(""),
) -> dict[str, Any]:
    """Save an uploaded file and create its inactive Mongo source record."""

    tender_id = clean_optional_string(tender_id)
    company_id = clean_optional_string(company_id)
    document_id = clean_optional_string(document_id) or str(uuid4())
    ocid = clean_optional_string(ocid)
    project_id = clean_optional_string(project_id) or ""
    created_by = (
        clean_optional_string(created_by)
        or clean_optional_string(user_name)
        or ""
    )
    if not tender_id or not company_id:
        raise HTTPException(
            status_code=400,
            detail="tender_id and company_id are required",
        )

    document_name = Path(file.filename or "uploaded-document.pdf").name
    input_dir = Path(
        os.getenv(
            "DOCUMENT_UPLOAD_DIRECTORY",
            "app/infrastrature/documents/input_doc",
        )
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    saved_path = input_dir / f"{uuid4().hex}_{document_name}"
    resolved_blob_id = clean_optional_string(blob_id) or (
        f"{company_id}/TenderDoc/{document_name}"
    )
    now = datetime.utcnow()
    payload = _uploaded_document_payload(
        document_id=document_id,
        tender_id=tender_id,
        company_id=company_id,
        project_id=project_id,
        blob_id=resolved_blob_id,
        ocid=ocid,
        document_name=document_name,
        file_path=str(saved_path.resolve()),
        created_by=created_by,
        document_description=document_description.strip(),
        ocr_summary=ocr_summary.strip(),
        now=now,
    )

    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        collection_name = DocumentCollectionNames.from_environment().tender_mongo
        collection = get_motor_collection(collection_name)
        insert_result = await collection.insert_one(payload)
        payload["_id"] = insert_result.inserted_id
    except Exception as exc:
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=503,
            detail=f"Unable to save uploaded tender document: {exc}",
        ) from exc
    finally:
        await file.close()

    return {
        "success": True,
        "document_id": document_id,
        "collection": collection_name,
        "document_payload": _mongo_json(payload),
    }


@router.get("/tender-document/{document_id}")
async def get_uploaded_tender_document(document_id: str) -> dict[str, Any]:
    """Return the current Mongo payload so the upload UI can verify completion."""

    document_id = clean_optional_string(document_id)
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")

    collection_name = DocumentCollectionNames.from_environment().tender_mongo
    collection = get_motor_collection(collection_name)
    document = await collection.find_one({"DocumentId": document_id})
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tender document {document_id!r} was not found",
        )

    return {
        "success": True,
        "collection": collection_name,
        "document_payload": _mongo_json(document),
    }


@router.delete("/tender-document")
async def delete_tender_document(
    request: TenderDocumentDeleteRequest,
) -> dict[str, Any]:
    """Delete one tender document and its exact Qdrant chunks."""

    company_id = clean_optional_string(request.company_id)
    tender_id = clean_optional_string(request.tender_id)
    document_id = clean_optional_string(request.document_id)
    if not company_id or not tender_id or not document_id:
        raise HTTPException(
            status_code=400,
            detail="company_id, tender_id, and document_id are required",
        )

    try:
        return await TenderDocumentDeletionService().delete(
            company_id=company_id,
            tender_id=tender_id,
            document_id=document_id,
        )
    except TenderDocumentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except TenderDocumentDeleteConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
    except (
        TenderDocumentMongoDeleteError,
        TenderDocumentQdrantDeleteError,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@router.delete("/company-documents")
async def delete_company_documents(
    request: CompanyDocumentDeleteRequest,
) -> dict[str, Any]:
    """Delete all normal company documents and chunks for one company."""

    company_id = clean_optional_string(request.company_id)
    if not company_id:
        raise HTTPException(status_code=400, detail="company_id is required")

    try:
        return await CompanyDocumentDeletionService().delete(
            company_id=company_id,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentDeleteConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (DocumentMongoDeleteError, DocumentQdrantDeleteError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/section-document")
async def delete_section_document(
    request: SectionDocumentDeleteRequest,
) -> dict[str, Any]:
    """Delete one nested company section document and its exact chunks."""

    company_id = clean_optional_string(request.company_id)
    section_id = clean_optional_string(request.section_id)
    document_id = clean_optional_string(request.document_id)
    if not company_id or not section_id or not document_id:
        raise HTTPException(
            status_code=400,
            detail="company_id, section_id, and document_id are required",
        )

    try:
        return await SectionDocumentDeletionService().delete(
            company_id=company_id,
            section_id=section_id,
            document_id=document_id,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentDeleteConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (DocumentMongoDeleteError, DocumentQdrantDeleteError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def process_documents_from_mongo_direct(
    request: MongoDocumentProcessRequest,
):
    """Process Mongo-backed documents without publishing another queue job."""

    tender_id = clean_optional_string(request.tender_id)
    company_id = clean_optional_string(request.company_id)
    project_id = clean_optional_string(request.project_id)
    user_id = clean_optional_string(request.user_id)
    created_by = (
        clean_optional_string(request.created_by)
        or clean_optional_string(request.user_name)
    )
    bearer_token = clean_optional_string(request.bearer_token)

    if not tender_id and not company_id:
        raise HTTPException(
            status_code=400,
            detail="Pass tender_id, company_id, or both.",
        )

    run_id = str(uuid4())
    document_scope = "tender" if tender_id else "company"
    source_id = tender_id or company_id
    created_at = datetime.utcnow().isoformat()

    try:
        from app.agents.DocExtractionpipeline.pipeline.document_extraction_pipeline import (
            process_document,
        )
        from app.agents.DocExtractionpipeline.services.mongo.mongo_services import (
            MongoService,
        )

        mongo = MongoService()
        mongo_documents = await mongo.get_documents_for_processing(
            tender_id=tender_id,
            company_profile_id=company_id,
        )
        debug_counts = None
        if not mongo_documents:
            debug_counts = await mongo.get_processing_match_counts(
                tender_id=tender_id,
                company_profile_id=company_id,
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to fetch documents from MongoDB: {e}",
        )

    async def process_one(mongo_document: dict) -> dict:
        file_path = Path(mongo_document["file_path"])
        document_name = mongo_document.get("document_name") or file_path.name
        resolved_company_id = (
            clean_optional_string(mongo_document.get("company_id"))
            or company_id
        )
        resolved_tender_id = tender_id if document_scope == "tender" else None

        if document_scope == "tender":
            resolved_tender_id = (
                clean_optional_string(mongo_document.get("tender_id"))
                or tender_id
            )

        if not file_path.exists():
            return {
                "document_name": document_name,
                "file_path": str(file_path),
                "status": "Failed",
                "error": "FilePath does not exist locally.",
            }

        source_ids = []
        if tender_id:
            source_ids.append(
                {
                    "sourceIdType": "tender",
                    "id": tender_id,
                }
            )
        if resolved_company_id:
            source_ids.append(
                {
                    "sourceIdType": "company",
                    "id": resolved_company_id,
                }
            )

        usage_context = {
            "applicationName": "Tender",
            "runId": run_id,
            "companyId": resolved_company_id,
            "tenderId": resolved_tender_id,
            "projectId": project_id,
            "userId": user_id,
            "userName": clean_optional_string(request.user_name),
            "purpose": "document_mongo_processing",
            "bearerToken": bearer_token,
            "sourceIds": source_ids or [
                {
                    "sourceIdType": document_scope,
                    "id": source_id,
                }
            ],
        }

        try:
            processed_document_id = await process_document(
                pdf_path=str(file_path),
                document_scope=document_scope,
                company_id=resolved_company_id,
                tender_id=resolved_tender_id,
                created_by=created_by or mongo_document.get("created_by"),
                created_at=created_at,
                source_filename=document_name,
                usage_context=usage_context,
                delete_source_file=False,
                mongo_document_id=mongo_document.get("mongo_document_id"),
                blob_id=mongo_document.get("blob_id"),
                document_id=mongo_document.get("document_id"),
                collection_name=mongo_document.get("collection_name"),
                existing_document=mongo_document.get("existing_document"),
            )

            return {
                "document_name": document_name,
                "file_path": str(file_path),
                "document_id": processed_document_id,
                "mongo_document_id": mongo_document.get("mongo_document_id"),
                "blob_id": mongo_document.get("blob_id"),
                "status": "Success",
            }
        except Exception as e:
            return {
                "document_name": document_name,
                "file_path": str(file_path),
                "mongo_document_id": mongo_document.get("mongo_document_id"),
                "document_id": mongo_document.get("document_id"),
                "blob_id": mongo_document.get("blob_id"),
                "status": "Failed",
                "error": str(e),
            }

    # Process every document sequentially. The caller receives this response
    # only after all documents have finished, so orchestration can safely start
    # requirement extraction and deduplication afterward.
    results = []
    for mongo_document in mongo_documents:
        results.append(await process_one(mongo_document))

    response = {
        "success": True,
        "runId": run_id,
        "document_scope": document_scope,
        "total_documents": len(mongo_documents),
        "results": results,
    }
    if debug_counts:
        response["debug"] = debug_counts

    return response


@router.get("/health")
def health():
    return {
        "success": True,
        "message": "Document Extraction Service is running."
    }





load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

################################################################  Retrival api code   ################################################################

class RetrievalRequest(BaseModel):
    collection_name: str
    company_id: str
    query: str
    limit: int = 5


@router.post("/retrieve")
async def retrieve_context(request: RetrievalRequest):
    try:
        from app.agents.DocExtractionpipeline.services.embedding.embedding_services import (
            DocumentEmbedder,
        )
        from app.agents.DocExtractionpipeline.services.qdrant.qdrant_services import (
            QdrantService,
        )

        embedding_model = await DocumentEmbedder.get_embedding_model()

        query_vector = embedding_model.embed_query(request.query)

        results = QdrantService.search_similarity(
            collection_name=request.collection_name,
            query_vector=query_vector,
            limit=request.limit,
            metadata_filter={
                "CompanyId": request.company_id
            }
        )

        response = []

        for result in results:
            payload = result["payload"]

            response.append({
                "score": round(result["score"], 4),
                "company_id": payload.get("CompanyId"),
                "tender_id": payload.get("TenderId"),
                "document_id": payload.get("DocumentId"),
                "chunk_id": payload.get("ChunkId"),
                "document_name": payload.get("DocumentName"),
                "text": payload.get("Text")
            })

        return {
            "success": True,
            "query": request.query,
            "total_results": len(response),
            "results": response
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
