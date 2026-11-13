import unittest
from unittest.mock import AsyncMock, Mock, patch

from bson import ObjectId

from app.agents.DocExtractionpipeline.services.collection_names import (
    DocumentCollectionNames,
)
from app.agents.DocExtractionpipeline.services.document_deletion_service import (
    CompanyDocumentDeletionService,
    SectionDocumentDeletionService,
    company_documents_identity_query,
    section_document_identity_query,
)
from app.agents.DocExtractionpipeline.services.qdrant.qdrant_services import (
    QdrantService,
)
from app.api.routes.document_extractor_router import (
    CompanyDocumentDeleteRequest,
    SectionDocumentDeleteRequest,
    delete_company_documents,
    delete_section_document,
)


class _CountResult:
    count = 4


class _DeleteStatus:
    value = "completed"


class _QdrantDeleteResult:
    status = _DeleteStatus()


class _FakeQdrantClient:
    def count(self, *, collection_name, count_filter, exact):
        self.collection_name = collection_name
        self.filter = count_filter
        self.exact = exact
        return _CountResult()

    def delete(self, *, collection_name, points_selector, wait):
        self.delete_filter = points_selector
        self.wait = wait
        return _QdrantDeleteResult()


class _DeleteManyResult:
    deleted_count = 2


class _UpdateResult:
    modified_count = 1


class _FakeCompanyCollection:
    async def count_documents(self, query):
        self.count_query = query
        return 2

    async def delete_many(self, query):
        self.delete_query = query
        return _DeleteManyResult()


class _FakeSectionCollection:
    def __init__(self):
        self.parent_id = ObjectId()

    async def find_one(self, query, projection):
        self.find_query = query
        return {"_id": self.parent_id}

    async def update_one(self, query, update):
        self.update_query = query
        self.update = update
        return _UpdateResult()


def _collection_names() -> DocumentCollectionNames:
    return DocumentCollectionNames(
        company_mongo="company-mongo",
        tender_mongo="tender-mongo",
        section_mongo="section-mongo",
        company_qdrant="company-qdrant",
        tender_qdrant="tender-qdrant",
    )


def _filter_values(qdrant_filter):
    return {
        condition.key: condition.match.value
        for condition in qdrant_filter.must
    }


class CompanyAndSectionDocumentDeleteTests(unittest.IsolatedAsyncioTestCase):
    def test_company_filters_use_only_company_id(self):
        mongo_query = company_documents_identity_query(company_id="company-1")
        qdrant_filter = QdrantService.company_documents_filter(
            company_id="company-1"
        )

        self.assertIn("CompanyId", str(mongo_query))
        self.assertEqual(
            _filter_values(qdrant_filter),
            {"CompanyId": "company-1"},
        )

    def test_section_filters_require_all_three_ids(self):
        mongo_query = section_document_identity_query(
            company_id="company-1",
            section_id="section-1",
            document_id="document-1",
        )
        qdrant_filter = QdrantService.section_document_filter(
            company_id="company-1",
            section_id="section-1",
            document_id="document-1",
        )

        query_text = str(mongo_query)
        self.assertIn("CompanyId", query_text)
        self.assertIn("SectionId", query_text)
        self.assertIn("Documents.DocumentId", query_text)
        self.assertEqual(
            _filter_values(qdrant_filter),
            {
                "CompanyId": "company-1",
                "SectionId": "section-1",
                "DocumentId": "document-1",
            },
        )

    async def test_company_service_deletes_qdrant_and_all_mongo_matches(self):
        mongo = _FakeCompanyCollection()
        qdrant = _FakeQdrantClient()
        service = CompanyDocumentDeletionService(
            collection_names=_collection_names(),
            mongo_collection=mongo,
            qdrant_client=qdrant,
        )

        result = await service.delete(company_id="company-1")

        self.assertEqual(result["mongo"]["deleted_count"], 2)
        self.assertEqual(result["qdrant"]["matched_chunks"], 4)
        self.assertEqual(qdrant.collection_name, "company-qdrant")
        self.assertEqual(mongo.count_query, mongo.delete_query)

    async def test_section_service_pulls_only_the_matching_child(self):
        mongo = _FakeSectionCollection()
        qdrant = _FakeQdrantClient()
        service = SectionDocumentDeletionService(
            collection_names=_collection_names(),
            mongo_collection=mongo,
            qdrant_client=qdrant,
        )

        result = await service.delete(
            company_id="company-1",
            section_id="section-1",
            document_id="document-1",
        )

        self.assertEqual(result["mongo"]["collection"], "section-mongo")
        self.assertIn("$pull", mongo.update)
        self.assertIn("Documents", mongo.update["$pull"])
        self.assertEqual(
            _filter_values(qdrant.delete_filter),
            {
                "CompanyId": "company-1",
                "SectionId": "section-1",
                "DocumentId": "document-1",
            },
        )

    async def test_company_and_section_routes_are_separate(self):
        company_service = Mock()
        company_service.delete = AsyncMock(return_value={"kind": "company"})
        section_service = Mock()
        section_service.delete = AsyncMock(return_value={"kind": "section"})

        with patch(
            "app.api.routes.document_extractor_router."
            "CompanyDocumentDeletionService",
            return_value=company_service,
        ), patch(
            "app.api.routes.document_extractor_router."
            "SectionDocumentDeletionService",
            return_value=section_service,
        ):
            company_result = await delete_company_documents(
                CompanyDocumentDeleteRequest(company_id="company-1")
            )
            section_result = await delete_section_document(
                SectionDocumentDeleteRequest(
                    company_id="company-1",
                    section_id="section-1",
                    document_id="document-1",
                )
            )

        self.assertEqual(company_result, {"kind": "company"})
        self.assertEqual(section_result, {"kind": "section"})
        company_service.delete.assert_awaited_once_with(
            company_id="company-1"
        )
        section_service.delete.assert_awaited_once_with(
            company_id="company-1",
            section_id="section-1",
            document_id="document-1",
        )


if __name__ == "__main__":
    unittest.main()
