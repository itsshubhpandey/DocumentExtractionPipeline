import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bson import ObjectId
from starlette.datastructures import UploadFile

from app.api.routes.document_extractor_router import upload_tender_document


class _InsertResult:
    def __init__(self) -> None:
        self.inserted_id = ObjectId()


class _FakeCollection:
    def __init__(self) -> None:
        self.inserted_payload = None

    async def insert_one(self, payload):
        self.inserted_payload = dict(payload)
        return _InsertResult()


class TenderDocumentUploadTests(unittest.IsolatedAsyncioTestCase):
    def test_frontend_hides_ocr_results_until_processing_completes(self):
        frontend = (
            Path(__file__).parents[1] / "frontend" / "tender_upload.html"
        ).read_text(encoding="utf-8")

        self.assertIn('id="mongoResultsCard" class="card" hidden', frontend)
        self.assertIn('id="apiResultsCard" class="card" hidden', frontend)
        self.assertIn("showCompletedResults(payload);", frontend)
        self.assertIn("if (active && processed)", frontend)
        self.assertNotIn('id="summary"', frontend)

    async def test_upload_saves_file_and_creates_inactive_mongo_record(self):
        collection = _FakeCollection()

        with tempfile.TemporaryDirectory() as upload_directory:
            uploaded_file = UploadFile(
                filename="../../tender.pdf",
                file=BytesIO(b"%PDF-1.7 test document"),
            )
            with (
                patch.dict(
                    os.environ,
                    {"DOCUMENT_UPLOAD_DIRECTORY": upload_directory},
                    clear=False,
                ),
                patch(
                    "app.api.routes.document_extractor_router."
                    "DocumentCollectionNames.from_environment",
                    return_value=SimpleNamespace(tender_mongo="tender-documents"),
                ),
                patch(
                    "app.api.routes.document_extractor_router.get_motor_collection",
                    return_value=collection,
                ),
            ):
                response = await upload_tender_document(
                    file=uploaded_file,
                    tender_id=" tender-1 ",
                    company_id=" company-1 ",
                    document_id="document-1",
                    ocid="ocid-1",
                    project_id="project-1",
                    created_by="Uploader",
                    user_name="",
                    document_description="Tender document",
                    ocr_summary="",
                    blob_id="",
                )

            payload = collection.inserted_payload
            self.assertIsNotNone(payload)
            self.assertEqual(payload["DocumentId"], "document-1")
            self.assertEqual(payload["TenderId"], "tender-1")
            self.assertEqual(payload["CompanyId"], "company-1")
            self.assertEqual(payload["ProjectId"], "project-1")
            self.assertEqual(payload["DocumentName"], "tender.pdf")
            self.assertEqual(payload["Status"], "Uploaded")
            self.assertFalse(payload["IsActive"])
            self.assertEqual(
                payload["BlobId"],
                "company-1/TenderDoc/tender.pdf",
            )
            saved_path = Path(payload["FilePath"])
            self.assertEqual(saved_path.parent, Path(upload_directory).resolve())
            self.assertEqual(saved_path.read_bytes(), b"%PDF-1.7 test document")

            self.assertTrue(response["success"])
            self.assertEqual(response["collection"], "tender-documents")
            self.assertEqual(response["document_id"], "document-1")
            self.assertIn("$oid", response["document_payload"]["_id"])


if __name__ == "__main__":
    unittest.main()
