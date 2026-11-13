import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv

from app.agents.DocExtractionpipeline.services.document_processing.doc_processing import (
    DoclingService,
)
from app.agents.DocExtractionpipeline.services.embedding.embedding_services import (
    DocumentEmbedder,
)
from app.agents.DocExtractionpipeline.services.mongo.mongo_services import (
    MongoService,
)
from app.agents.DocExtractionpipeline.services.qdrant.qdrant_services import (
    QdrantService,
)

load_dotenv()


SECTION_ID_TO_DOCUMENT_TYPE = {
    "1": "Company Overview",
    "2": "Company Structure and Ownership",
    "3": "Financial Information",
    "4": "Legal and Compliance",
    "5": "Insurance and Risk Management",
    "6": "Operational Capability",
    "7": "Sustainability and Recognition",
    "8": "Win Themes & Positioning",
    "9": "Organisation Chart",
    "10": "Equality, Diversity and Inclusion",
    "11": "GDPR and Data Protection",
}


def _resolve_document_type(section_id: Optional[str]) -> str:
    if section_id is None:
        return "SectionDocument"
    return SECTION_ID_TO_DOCUMENT_TYPE.get(str(section_id), "SectionDocument")


def _requires_classification(section_id: Optional[str]) -> bool:
    """Use the LLM taxonomy for sections not covered by fixed mappings."""

    try:
        return int(str(section_id)) > 11
    except (TypeError, ValueError):
        return False


async def _process_single_file(
    file_entry: Dict[str, Any],
    created_by: Optional[str],
    created_at: str,
    usage_context: Dict[str, Any],
) -> Dict[str, Any]:
    file_path = Path(file_entry["path"])
    file_name = file_entry.get("file_name") or file_path.name
    mongo_id = file_entry["mongo_id"]
    company_id = file_entry["company_id"]
    section_id = file_entry.get("section_id")
    document_id = file_entry.get("document_id") or str(uuid4())

    output_dir = Path("app/infrastrature/documents/out_doc")
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{file_path.stem}_{uuid4().hex}.md"

    try:
        from app.agents.DocExtractionpipeline.doc_classifier_agent import (
            document_classifier_agent,
        )

        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        docling = DoclingService()
        mongo = MongoService()
        markdown_path = docling.convert_pdf_to_markdown(
            input_file=file_path,
            output_md_path=markdown_path,
        )

        file_usage_context = dict(usage_context or {})
        file_usage_context["sectionId"] = section_id

        metadata = docling.extract_markdown_metadata(markdown_path)
        metadata["source_file_name"] = file_name
        metadata["source_file_extension"] = file_path.suffix.lstrip(".").lower()

        if _requires_classification(section_id):
            classification_set = await mongo.get_classification_set(
                tender_id=None,
                company_profile_id=company_id,
            )
            classification_response = await document_classifier_agent(
                document_metadata=metadata,
                classification_set=classification_set,
                usage_context=file_usage_context,
            )
            classification_data = classification_response.get(
                "data",
                classification_response,
            )
            document_type = (
                classification_data.get("document_type")
                or "UNKNOWN_OR_UNCLEAR"
            )
        else:
            document_type = _resolve_document_type(section_id)

        client = QdrantService.get_client()
        document_id = await DocumentEmbedder.embed_markdown_document(
            file_path=str(markdown_path),
            document_scope="company",
            company_id=company_id,
            document_name=file_name,
            document_type=document_type,
            created_by=created_by,
            created_at=created_at,
            client=client,
            usage_context=file_usage_context,
            document_id=document_id,
            section_id=section_id,
        )

        await mongo.mark_section_document_processed(
            mongo_id=mongo_id,
            file_name=file_name,
            document_id=document_id,
        )

        return {
            "file_name": file_name,
            "section_id": section_id,
            "mongo_id": mongo_id,
            "document_id": document_id,
            "status": "Success",
        }

    except Exception as e:
        return {
            "file_name": file_name,
            "section_id": section_id,
            "mongo_id": mongo_id,
            "status": "Failed",
            "error": str(e),
        }

    finally:
        try:
            if markdown_path and Path(markdown_path).exists():
                Path(markdown_path).unlink()
        except OSError as cleanup_error:
            print(f"Could not delete temporary markdown {markdown_path}: {cleanup_error}")


async def process_section_documents(
    company_id: str,
    section_id: str,
    created_by: Optional[str] = None,
    created_at: Optional[str] = None,
    usage_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    mongo = MongoService()
    file_entries = await mongo.get_section_documents(
        company_id=company_id,
        section_id=section_id,
    )

    if not file_entries:
        return []

    tasks = [
        _process_single_file(
            file_entry=entry,
            created_by=created_by,
            created_at=created_at,
            usage_context=usage_context or {},
        )
        for entry in file_entries
    ]

    return await asyncio.gather(*tasks)
