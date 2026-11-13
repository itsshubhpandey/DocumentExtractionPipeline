import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.agents.DocExtractionpipeline.services.collection_names import (
    DocumentCollectionNames,
)

load_dotenv()

DEFAULT_LOCAL_QDRANT_PATH = "./qdrant_storage"
DEFAULT_KEYWORD_INDEX_FIELDS = ("CompanyId", "TenderId", "DocumentId", "ChunkId")


class QdrantService:

    @staticmethod
    def _identity_filter(identifiers: dict[str, Any]) -> Filter:
        cleaned_identifiers = {
            field_name: str(value or "").strip()
            for field_name, value in identifiers.items()
        }
        if not all(cleaned_identifiers.values()):
            required = ", ".join(
                field_name for field_name in cleaned_identifiers
            )
            raise ValueError(f"Required Qdrant identifiers: {required}")

        return Filter(
            must=[
                FieldCondition(
                    key=field_name,
                    match=MatchValue(value=value),
                )
                for field_name, value in cleaned_identifiers.items()
            ]
        )

    @staticmethod
    def company_documents_filter(*, company_id: str) -> Filter:
        """Match all company-document chunks for one company."""

        return QdrantService._identity_filter({"CompanyId": company_id})

    @staticmethod
    def section_document_filter(
        *,
        company_id: str,
        section_id: str,
        document_id: str,
    ) -> Filter:
        """Match one exact company section document's chunks."""

        return QdrantService._identity_filter(
            {
                "CompanyId": company_id,
                "SectionId": section_id,
                "DocumentId": document_id,
            }
        )

    @staticmethod
    def tender_document_filter(
        *,
        company_id: str,
        tender_id: str,
        document_id: str,
    ) -> Filter:
        """Return the strict identity filter for one tender document's chunks."""

        return QdrantService._identity_filter(
            {
                "CompanyId": company_id,
                "TenderId": tender_id,
                "DocumentId": document_id,
            }
        )

    @staticmethod
    def _delete_chunks(
        *,
        document_filter: Filter,
        collection_name: str,
        client: Optional[QdrantClient] = None,
    ) -> dict[str, Any]:
        client = client if client is not None else QdrantService.get_client()
        matched_chunks = client.count(
            collection_name=collection_name,
            count_filter=document_filter,
            exact=True,
        ).count
        result = client.delete(
            collection_name=collection_name,
            points_selector=document_filter,
            wait=True,
        )
        status = getattr(result, "status", None)
        status = getattr(status, "value", status)

        return {
            "collection": collection_name,
            "matched_chunks": matched_chunks,
            "delete_status": str(status) if status is not None else None,
        }

    @staticmethod
    def delete_company_document_chunks(
        *,
        company_id: str,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        collection_name = (
            collection_name
            or DocumentCollectionNames.from_environment().company_qdrant
        )
        return QdrantService._delete_chunks(
            document_filter=QdrantService.company_documents_filter(
                company_id=company_id
            ),
            collection_name=collection_name,
            client=client,
        )

    @staticmethod
    def delete_section_document_chunks(
        *,
        company_id: str,
        section_id: str,
        document_id: str,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        collection_name = (
            collection_name
            or DocumentCollectionNames.from_environment().company_qdrant
        )
        return QdrantService._delete_chunks(
            document_filter=QdrantService.section_document_filter(
                company_id=company_id,
                section_id=section_id,
                document_id=document_id,
            ),
            collection_name=collection_name,
            client=client,
        )

    @staticmethod
    def delete_tender_document_chunks(
        *,
        company_id: str,
        tender_id: str,
        document_id: str,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Delete chunks belonging to one exact tender-document identity."""

        collection_name = (
            collection_name
            or DocumentCollectionNames.from_environment().tender_qdrant
        )
        document_filter = QdrantService.tender_document_filter(
            company_id=company_id,
            tender_id=tender_id,
            document_id=document_id,
        )
        return QdrantService._delete_chunks(
            document_filter=document_filter,
            collection_name=collection_name,
            client=client,
        )

    @staticmethod
    def get_client(location: str = None) -> QdrantClient:
        """
        Returns a Qdrant client instance based on the environment or explicit location.
        Uses QDRANT_URL with or without QDRANT_API_KEY.
        Defaults to a persistent local Qdrant store if no URL is configured.
        """
        if location:
            return QdrantClient(location=location)

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant_local_path = os.getenv("QDRANT_LOCAL_PATH", DEFAULT_LOCAL_QDRANT_PATH)

        if qdrant_url:
            if qdrant_api_key:
                return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            return QdrantClient(url=qdrant_url)

        print(f"Qdrant falling back to local persistent storage: {qdrant_local_path}")
        return QdrantClient(path=qdrant_local_path)

    @staticmethod
    def create_collection(collection_name: str, vector_size: int = 1536) -> None:
        """
        Creates a new collection if it does not already exist.
        Default vector size is 1536 (OpenAI standard).
        """
        client = QdrantService.get_client()
        collections_response = client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]

        if collection_name not in existing_collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")

        QdrantService.ensure_payload_indexes(
            collection_name=collection_name,
            fields=DEFAULT_KEYWORD_INDEX_FIELDS,
            client=client,
        )

    @staticmethod
    def ensure_payload_indexes(
        collection_name: str,
        fields: tuple[str, ...],
        client: Optional[QdrantClient] = None,
    ) -> None:
        """
        Ensures keyword payload indexes exist for fields used in filters.
        Qdrant Cloud requires these indexes before filtering by payload values.
        """
        if client is None:
            client = QdrantService.get_client()

        for field in fields:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as e:
                message = str(e).lower()
                if "already exists" not in message and "already has" not in message:
                    raise

    @staticmethod
    def _get_query_vector_name(
        client: QdrantClient,
        collection_name: str,
    ) -> str | None:
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            vectors_config = getattr(collection_info.config.params, "vectors", None)
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                return "dense"
        except Exception:
            pass

        return None

    @staticmethod
    def upsert_points(collection_name: str, points: List[PointStruct]) -> None:
        """
        Upserts a list of PointStruct tracking items into the specified collection.
        """
        client = QdrantService.get_client()
        response = client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully uploaded {len(points)} records to '{collection_name}'. Status: {response.status}")

    @staticmethod
    def search_similarity(
        collection_name: str,
        query_vector: List[float],
        limit: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a vector search query against the collection.
        Returns clean payload dictionary items alongside their matching scores.
        """
        client = QdrantService.get_client()

        query_filter = None
        if metadata_filter:
            QdrantService.ensure_payload_indexes(
                collection_name=collection_name,
                fields=tuple(metadata_filter.keys()),
                client=client,
            )

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                    for key, value in metadata_filter.items()
                ]
            )

        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=QdrantService._get_query_vector_name(
                client=client,
                collection_name=collection_name,
            ),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        formatted_results = []
        for result in search_results.points:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })

        return formatted_results

    @staticmethod
    def delete_collection(collection_name: str) -> None:
        """
        Deletes a complete collection structurally from the cluster.
        """
        client = QdrantService.get_client()
        client.delete_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' dropped entirely.")
