import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from bson import ObjectId
from fastapi import HTTPException

from app.agents.DocExtractionpipeline.services.collection_names import (
    DocumentCollectionNames,
)
from app.agents.DocExtractionpipeline.services.document_deletion_service import (
    TenderDocumentDeletionService,
    TenderDocumentQdrantDeleteError,
    tender_document_identity_query,
)
from app.agents.DocExtractionpipeline.services.qdrant.qdrant_services import (
    QdrantService,
)
from app.api.routes.document_extractor_router import (
    TenderDocumentDeleteRequest,
    delete_tender_document,
)


class _CountResult:
    count = 7


class _DeleteStatus:
    value = "completed"


class _QdrantDeleteResult:
    status = _DeleteStatus()


class _FakeQdrantClient:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.count_filter = None
        self.delete_filter = None

    def count(self, *, collection_name, count_filter, exact):
        if self.fail:
            raise RuntimeError("Qdrant unavailable")
        self.collection_name = collection_name
        self.count_filter = count_filter
        self.exact = exact
        return _CountResult()

    def delete(self, *, collection_name, points_selector, wait):
        self.collection_name = collection_name
        self.delete_filter = points_selector
        self.wait = wait
        return _QdrantDeleteResult()


class _MongoDeleteResult:
    deleted_count = 1


class _FakeMongoCollection:
    def __init__(self):
        self.find_query = None
        self.delete_query = None
        self.delete_calls = 0
        self.mongo_id = ObjectId()

    async def find_one(self, query, projection):
        self.find_query = query
        self.projection = projection
        return {"_id": self.mongo_id}

    async def delete_one(self, query):
        self.delete_calls += 1
        self.delete_query = query
        return _MongoDeleteResult()


def _collection_names() -> DocumentCollectionNames:
    return DocumentCollectionNames(
        company_mongo="company-mongo",
        tender_mongo="tender-mongo",
        section_mongo="section-mongo",
        company_qdrant="company-qdrant",
        tender_qdrant="tender-qdrant",
    )


class TenderDocumentDeleteTests(unittest.IsolatedAsyncioTestCase):
    def test_collection_names_come_from_environment_variables(self):
        environment = {
            "COMPANYDOCUMENTCOLLECTIONMONGO": "env-company-mongo",
            "TENDERDOCUMENTCOLLECTIONMONGO": "env-tender-mongo",
            "SECTIONDOCUMENTCOLLECTIONMONGO": "env-section-mongo",
            "COMPANYDOCUMENTCOLLECTIONQDRANT": "env-company-qdrant",
            "TENDERDOCUMENTCOLLECTIONQDRANT": "env-tender-qdrant",
        }

        with patch.dict(os.environ, environment, clear=False):
            names = DocumentCollectionNames.from_environment()

        self.assertEqual(names.company_mongo, "env-company-mongo")
        self.assertEqual(names.tender_mongo, "env-tender-mongo")
        self.assertEqual(names.section_mongo, "env-section-mongo")
        self.assertEqual(names.company_qdrant, "env-company-qdrant")
        self.assertEqual(names.tender_qdrant, "env-tender-qdrant")

    def test_qdrant_delete_filter_requires_all_three_identifiers(self):
        client = _FakeQdrantClient()

        result = QdrantService.delete_tender_document_chunks(
            company_id="company-1",
            tender_id="tender-1",
            document_id="document-1",
            client=client,
            collection_name="configured-tender-qdrant",
        )

        conditions = {
            condition.key: condition.match.value
            for condition in client.delete_filter.must
        }
        self.assertEqual(
            conditions,
            {
                "CompanyId": "company-1",
                "TenderId": "tender-1",
                "DocumentId": "document-1",
            },
        )
        self.assertEqual(client.collection_name, "configured-tender-qdrant")
        self.assertIs(client.count_filter, client.delete_filter)
        self.assertTrue(client.exact)
        self.assertTrue(client.wait)
        self.assertEqual(result["matched_chunks"], 7)

    def test_mongo_query_requires_all_three_identifiers(self):
        query = tender_document_identity_query(
            company_id="company-1",
            tender_id="tender-1",
            document_id="document-1",
        )

        self.assertEqual(len(query["$and"]), 3)
        query_text = str(query)
        self.assertIn("CompanyId", query_text)
        self.assertIn("TenderId", query_text)
        self.assertIn("DocumentId", query_text)

    async def test_service_deletes_configured_qdrant_then_mongo_document(self):
        mongo_collection = _FakeMongoCollection()
        qdrant_client = _FakeQdrantClient()
        service = TenderDocumentDeletionService(
            collection_names=_collection_names(),
            mongo_collection=mongo_collection,
            qdrant_client=qdrant_client,
        )

        response = await service.delete(
            company_id="company-1",
            tender_id="tender-1",
            document_id="document-1",
        )

        self.assertTrue(response["success"])
        self.assertEqual(response["mongo"]["collection"], "tender-mongo")
        self.assertEqual(response["qdrant"]["collection"], "tender-qdrant")
        self.assertEqual(qdrant_client.collection_name, "tender-qdrant")
        self.assertEqual(mongo_collection.delete_calls, 1)
        self.assertEqual(
            mongo_collection.delete_query["$and"][0],
            {"_id": mongo_collection.mongo_id},
        )

    async def test_qdrant_failure_keeps_mongo_document(self):
        mongo_collection = _FakeMongoCollection()
        service = TenderDocumentDeletionService(
            collection_names=_collection_names(),
            mongo_collection=mongo_collection,
            qdrant_client=_FakeQdrantClient(fail=True),
        )

        with self.assertRaises(TenderDocumentQdrantDeleteError):
            await service.delete(
                company_id="company-1",
                tender_id="tender-1",
                document_id="document-1",
            )

        self.assertEqual(mongo_collection.delete_calls, 0)

    async def test_route_delegates_to_deletion_service(self):
        expected = {"success": True}
        service = Mock()
        service.delete = AsyncMock(return_value=expected)

        with patch(
            "app.api.routes.document_extractor_router."
            "TenderDocumentDeletionService",
            return_value=service,
        ):
            response = await delete_tender_document(
                TenderDocumentDeleteRequest(
                    company_id="company-1",
                    tender_id="tender-1",
                    document_id="document-1",
                )
            )

        self.assertEqual(response, expected)
        service.delete.assert_awaited_once_with(
            company_id="company-1",
            tender_id="tender-1",
            document_id="document-1",
        )

    async def test_blank_identifier_is_rejected_before_service_creation(self):
        with patch(
            "app.api.routes.document_extractor_router."
            "TenderDocumentDeletionService",
        ) as service_class:
            with self.assertRaises(HTTPException) as raised:
                await delete_tender_document(
                    TenderDocumentDeleteRequest(
                        company_id="company-1",
                        tender_id=" ",
                        document_id="document-1",
                    )
                )

        self.assertEqual(raised.exception.status_code, 400)
        service_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
