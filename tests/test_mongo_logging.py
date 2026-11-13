import os
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from bson import ObjectId

from app.infrastructure.logger import Logging
from app.infrastructure.mongo_log_store import (
    insert_enterprise_log,
    insert_token_usage,
)
from app.infrastructure.token_usage import TokenUsageService as SharedTokenUsageService
from app.infrastructure.token_usage_logger import (
    TokenUsageService as RequirementTokenUsageService,
)
from app.utils.token_usage import TokenUsageService as DocumentTokenUsageService


class _InsertResult:
    def __init__(self):
        self.inserted_id = ObjectId()


class _AsyncCollection:
    def __init__(self):
        self.document = None

    async def insert_one(self, document):
        self.document = document
        return _InsertResult()


class _SyncCollection:
    def __init__(self):
        self.document = None

    def insert_one(self, document):
        self.document = document
        return _InsertResult()


class _Database:
    def __init__(self, collection):
        self.collection = collection
        self.requested_collection = None

    def __getitem__(self, name):
        self.requested_collection = name
        return self.collection


class MongoLogStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_usage_is_inserted_into_correctly_spelled_collection(self):
        collection = _AsyncCollection()
        database = _Database(collection)

        with patch.dict(
            os.environ,
            {"TOKEN_USAGE_LOG_COLLECTION_NAME": "TokenUsageLog"},
            clear=False,
        ), patch(
            "app.infrastructure.mongo_log_store.get_motor_database",
            return_value=database,
        ):
            result = await insert_token_usage(
                {
                    "agentName": "Requirement Agent",
                    "totalTokens": 25,
                }
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["collection"], "TokenUsageLog")
        self.assertEqual(database.requested_collection, "TokenUsageLog")
        self.assertEqual(collection.document["totalTokens"], 25)
        self.assertIsInstance(collection.document["CreatedAt"], datetime)

    async def test_enterprise_log_is_inserted_into_correctly_spelled_collection(self):
        collection = _SyncCollection()
        database = _Database(collection)

        with patch.dict(
            os.environ,
            {"ENTERPRISE_LOG_COLLECTION_NAME": "EnterpriseLog"},
            clear=False,
        ), patch(
            "app.infrastructure.mongo_log_store.get_database",
            return_value=database,
        ):
            result = insert_enterprise_log(
                {
                    "eventType": "PipelineCompleted",
                    "isSuccess": True,
                }
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["collection"], "EnterpriseLog")
        self.assertEqual(database.requested_collection, "EnterpriseLog")
        self.assertEqual(collection.document["eventType"], "PipelineCompleted")
        self.assertIsInstance(collection.document["CreatedAt"], datetime)

    async def test_enterprise_logger_no_longer_requires_log_url(self):
        insert_result = {
            "success": True,
            "collection": "EnterpriseLog",
            "inserted_id": "log-id",
        }

        with patch.dict(
            os.environ,
            {"LOG_ENABLED": "true"},
            clear=False,
        ), patch(
            "app.infrastructure.logger.insert_enterprise_log",
            return_value=insert_result,
        ) as insert:
            logger = Logging("Test Agent", "test_module")
            result = logger.log(
                message="Completed",
                event_type="PipelineCompleted",
                payload={"count": 3},
                correlation_id="run-1",
            )

        self.assertEqual(result, insert_result)
        saved = insert.call_args.args[0]
        self.assertEqual(saved["correlationId"], "run-1")
        self.assertEqual(saved["payload"], {"count": 3})
        self.assertIn('"count": 3', saved["payloadJson"])

    async def test_document_token_logger_inserts_without_url_or_bearer_token(self):
        with patch(
            "app.utils.token_usage.insert_token_usage",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "collection": "TokenUsageLog",
                    "inserted_id": "usage-id",
                }
            ),
        ) as insert:
            result = await DocumentTokenUsageService.log_usage(
                {
                    "agentName": "Document Classifier",
                    "companyId": "company-1",
                    "tenderId": "tender-1",
                    "totalTokens": 10,
                }
            )

        self.assertTrue(result["success"])
        saved = insert.await_args.args[0]
        self.assertEqual(saved["companyId"], "company-1")
        self.assertEqual(saved["tenderId"], "tender-1")

    async def test_shared_token_logger_inserts_without_url_or_bearer_token(self):
        with patch.dict(
            os.environ,
            {"AI_USAGE_LOG_ENABLED": "true"},
            clear=False,
        ), patch(
            "app.infrastructure.token_usage.insert_token_usage",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "collection": "TokenUsageLog",
                    "inserted_id": "usage-id",
                }
            ),
        ) as insert:
            result = await SharedTokenUsageService.log_usage(
                {
                    "agentName": "Deduplication Agent",
                    "companyId": "company-1",
                    "tenderId": "tender-1",
                    "totalTokens": 12,
                },
                bearer_token=None,
            )

        self.assertTrue(result["success"])
        insert.assert_awaited_once()

    async def test_requirement_token_logger_ignores_legacy_url_arguments(self):
        with patch(
            "app.infrastructure.token_usage_logger.insert_token_usage",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "collection": "TokenUsageLog",
                    "inserted_id": "usage-id",
                }
            ),
        ) as insert:
            result = await RequirementTokenUsageService.log_usage(
                api_url="https://unused.example/log",
                payload={"totalTokens": 5},
                token="unused-token",
            )

        self.assertTrue(result["success"])
        insert.assert_awaited_once_with({"totalTokens": 5})


if __name__ == "__main__":
    unittest.main()
