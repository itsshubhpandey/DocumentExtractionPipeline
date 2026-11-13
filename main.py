from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes.deduplication_route import router as deduplication_router
from app.api.routes.document_extractor_router import router
from app.api.routes.evaluation_criteria_route import (
    router as evaluation_criteria_router,
)
from app.api.routes.queued_orchestration_router import (
    router as queued_orchestration_router,
)
from app.infrastructure.rabbitmq_service import close_rabbitmq_service


app = FastAPI(
    title="Document Extraction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
        "http://localhost:9201",
        "http://127.0.0.1:9201",
        "http://0.0.0.0:9201",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|\d{1,3}(?:\.\d{1,3}){3})(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(evaluation_criteria_router)
app.include_router(deduplication_router, prefix="/api/v1/agents")
app.include_router(queued_orchestration_router)


@app.get("/", include_in_schema=False)
async def tender_upload_frontend() -> FileResponse:
    """Serve the upload-and-process browser console."""

    frontend_path = Path(__file__).parent / "frontend" / "tender_upload.html"
    return FileResponse(frontend_path)


@app.on_event("shutdown")
async def shutdown_rabbitmq() -> None:
    await close_rabbitmq_service()
