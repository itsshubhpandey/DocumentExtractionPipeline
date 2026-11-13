# Tender Document Processing Pipeline

> **Status: Working ✅**

An end-to-end tender intelligence pipeline that reads tender documents, performs
OCR and structural extraction, classifies and indexes their contents, extracts
requirements, removes duplicate requirements, and produces grounded evaluation
criteria.

The project exposes a FastAPI application and can execute the complete workflow
directly or asynchronously through RabbitMQ.

## What the Pipeline Does

1. Finds unprocessed company or tender documents in MongoDB.
2. Uses Docling to read each PDF and convert its text, headings, tables, and page
   boundaries into Markdown.
3. Classifies the document against a taxonomy stored in MongoDB.
4. Splits the document into page-aware, heading-aware chunks while preserving
   tables.
5. Generates dense embeddings and optional BM25 sparse vectors, then stores the
   chunks and their source metadata in Qdrant.
6. Retrieves tender chunks and filters out content that is unlikely to contain a
   requirement.
7. Uses an LLM to extract structured requirements with source document, page,
   heading, strength, priority, confidence, and intent metadata.
8. Deduplicates newly extracted requirements with an LLM and deterministic
   validation safeguards.
9. Retrieves supporting evidence and extracts validated evaluation criteria.
10. Saves stage results and processing state in MongoDB so incomplete runs can be
    resumed without repeating completed work.

## End-to-End Flow

```mermaid
flowchart LR
    A[PDF documents] --> B[MongoDB source records]
    B --> C[Docling OCR and Markdown extraction]
    C --> D[LLM document classification]
    D --> E[Page and heading-aware chunking]
    E --> F[Dense and sparse embeddings]
    F --> G[Qdrant document index]
    G --> H[Rule-based requirement filtering]
    H --> I[LLM requirement extraction]
    I --> J[MongoDB requirements]
    J --> K[LLM deduplication]
    K --> L[Deterministic merge validation]
    L --> M[MongoDB deduplicated requirements]
    G --> N[Evaluation evidence retrieval]
    N --> O[LLM evaluation criteria extraction]
    O --> P[Schema and evidence validation]
    P --> Q[MongoDB evaluation criteria]
```

## Processing Stages

### 1. Document OCR and Indexing

The document-processing stage:

- selects inactive MongoDB document records for a company or tender;
- resolves the local document path, including an optional `DOCUMENT_BASE_PATH`;
- converts PDF content into Markdown with page markers, headings, paragraphs,
  and Markdown tables;
- derives metadata such as page count, headings, keywords, and table count;
- classifies the document using the configured LLM and MongoDB taxonomy;
- splits content into chunks of up to 5,000 characters with 300 characters of
  overlap;
- keeps page numbers, heading hierarchy, table flags, and document identifiers
  in every Qdrant payload; and
- marks the MongoDB source record as `Processed` and active after successful
  indexing.

### 2. Requirement Extraction

The requirement agent scrolls all Qdrant chunks for the tender and processes only
chunks that have not already been recorded in MongoDB. Its LangGraph workflow is:

```text
rule_filter -> detect_and_extract -> process_requirements -> save_chunk_status
```

The rule filter scores headings and words such as `shall`, `must`, `mandatory`,
`submit`, `provide`, and `comply`. Likely requirement chunks are sent to the LLM
in batches. Extracted requirements are bulk-upserted into MongoDB, while every
processed chunk receives a status record so later runs can skip it safely.

### 3. Requirement Deduplication

Only requirement IDs that have not already appeared in a deduplication result are
sent to the deduplication agent. The agent proposes duplicate groups, and the
application validates them before saving the result.

The deterministic validator deliberately favors keeping requirements separate.
It checks:

- numbers, percentages, monetary values, and clause references;
- negation and conflicting conditions;
- materially different sentence endings;
- token overlap, containment, and sequence similarity; and
- complete coverage so every input requirement appears exactly once.

Validated groups receive canonical IDs such as `CR-0001`. Normal runs merge new
results into the active tender result; regeneration runs can be retained
separately.

### 4. Evaluation Criteria

The evaluation agent performs profile-driven retrieval from Qdrant, expands the
surrounding context, removes near-duplicate evidence, and sends grounded evidence
to the LLM. The result is validated against a JSON schema and checked for valid
source references and numeric values before being saved to MongoDB.

Evaluation processing is incremental: unchanged evidence chunks are skipped and
the last successful active result is preserved when no new evidence is available.

### 5. Orchestration and Recovery

The main tender workflow runs these stages in order:

```text
document_processing
    -> requirement_extraction
    -> requirement_deduplication
    -> evaluation_criteria
```

Each stage has a persistent MongoDB checkpoint. If processing is interrupted, a
later attempt resumes from the first incomplete or failed stage instead of
re-running completed stages.

When RabbitMQ is enabled, the worker also provides job claiming, leases,
heartbeats, retry handling, duplicate-message protection, and dead-letter
publishing.

## Technology Stack

- Python 3.11
- FastAPI and Uvicorn
- Docling for PDF extraction and OCR
- LangChain and LangGraph for agent workflows
- OpenAI, Mistral, Groq, Gemini, NVIDIA, or LiteLLM-compatible LLM providers
- MongoDB for source documents, requirements, results, and processing state
- Qdrant for dense and sparse document retrieval
- RabbitMQ for asynchronous processing
- FastEmbed BM25 for optional sparse vectors

## Project Structure

```text
.
├── main.py                              # FastAPI application
├── app/
│   ├── api/routes/                      # HTTP endpoints
│   ├── orchestration/                   # Complete tender workflow
│   ├── workers/                         # RabbitMQ worker
│   ├── infrastructure/                  # MongoDB, RabbitMQ, LLM, logging
│   └── agents/
│       ├── DocExtractionpipeline/       # OCR, classification, chunking, Qdrant
│       ├── tender_requirement_agent/    # Requirement extraction
│       ├── deduplication/               # Requirement deduplication
│       └── EvaluationCriteriaAgent/     # Evaluation criteria extraction
├── frontend/                            # Upload and processing interface
├── docs/                                # Workflow diagrams
└── tests/                               # Automated tests
```

## Prerequisites

- Python 3.11 or newer
- MongoDB
- Qdrant
- An API key for the configured LLM and embedding provider
- RabbitMQ when asynchronous processing is enabled

The document path stored in MongoDB must be readable by the API or worker process.

## Installation

```bash
git clone https://github.com/itsshubhpandey/DocumentExtractionPipeline.git
cd DocumentExtractionPipeline

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Add your own credentials and service connection details to `.env`. Never commit
the `.env` file.

## Core Configuration

The exact values depend on your local or hosted services. The main configuration
has the following shape:

```env
# MongoDB
MONGO_URI="mongodb://localhost:27017"
MONGO_DB_NAME="DocAI"
MONGO_TLS="false"

# Document collections
COMPANYDOCUMENTCOLLECTIONMONGO="CPDocuments"
TENDERDOCUMENTCOLLECTIONMONGO="CPTenderDocuments"
SECTIONDOCUMENTCOLLECTIONMONGO="SectionDocument"
COMPANYDOCUMENTCOLLECTIONQDRANT="CPDocuments"
TENDERDOCUMENTCOLLECTIONQDRANT="CPTenderDoc"
REQUIREMENT_QDRANT_COLLECTION_NAME="CPTenderDoc"

# Requirement and deduplication state
REQUIREMENTS_COLLECTION="TenderRequirements"
REQUIREMENT_EXTRACTION_COLLECTION_NAME="TenderRequirements"
CHUNKS_COLLECTION="TenderRequirementChunks"
DEDUPLICATION_COLLECTION_NAME="TenderRequirementDeduplications"

# Qdrant
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY=""

# LLM and embeddings
LLM_PROVIDER="openai"
OPENAI_MODEL="gpt-4.1-mini"
OPENAI_API_KEY="your-key"

# Processing mode
USING_RABBITMQ="false"
RABBITMQ_URL="amqp://guest:guest@localhost:5672/"

# Direct MongoDB logging
TOKEN_USAGE_LOG_COLLECTION_NAME="TokenUsageLog"
ENTERPRISE_LOG_COLLECTION_NAME="EnterpriseLog"
AI_USAGE_LOG_ENABLED="true"
LOG_ENABLED="true"

# Optional local root for relative document paths
DOCUMENT_BASE_PATH=""
```

The application can also read the active LLM and embedding model configuration
from MongoDB. Set `LLM_PROVIDER_COLLECTION` when your LLM configuration collection
uses a custom name. OpenAI credentials may be read from the local environment or
from the configured encrypted MongoDB secret record.

## Running the Application

Start the API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Application: [http://localhost:8000](http://localhost:8000)
- Swagger API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Document service health: [http://localhost:8000/documents/health](http://localhost:8000/documents/health)

If RabbitMQ processing is enabled, start the worker in a separate terminal:

```bash
python -m app.workers.tender_processing_worker
```

## Run the Complete Tender Pipeline

Use the main orchestration endpoint:

```http
POST /documents/process/ocr/using_Tenderid_or_Companyid
Content-Type: application/json
```

Example request:

```json
{
  "company_id": "your-company-id",
  "tender_id": "your-tender-id",
  "project_id": "your-project-id",
  "user_id": "your-user-id",
  "user_name": "Your Name",
  "created_by": "Your Name",
  "bearer_token": null,
  "is_regenerated": false
}
```

- With `USING_RABBITMQ=false`, the API runs all four stages and returns the final
  result directly.
- With `USING_RABBITMQ=true`, the API returns a `run_id` and queue status while the
  worker processes the tender.
- If `tender_id` is omitted, the endpoint queues document processing for company
  documents only; this mode requires RabbitMQ.

## Main API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/documents/upload/tender-document` | Upload a tender PDF and create its MongoDB source record |
| `GET` | `/documents/tender-document/{document_id}` | Read the current tender document state |
| `POST` | `/documents/process/ocr/using_Tenderid_or_Companyid` | Run or queue the complete pipeline |
| `POST` | `/documents/process` | Process uploaded files directly as multipart form data |
| `POST` | `/documents/process/section-documents` | Process company section documents |
| `POST` | `/documents/retrieve` | Run dense similarity retrieval against Qdrant |
| `DELETE` | `/documents/tender-document` | Delete one tender document and its Qdrant chunks |
| `DELETE` | `/documents/company-documents` | Delete company documents and their Qdrant chunks |
| `DELETE` | `/documents/section-document` | Delete one section document and its chunks |
| `POST` | `/api/v1/agents/requirement-deduplication/process` | Run deduplication independently |
| `POST` | `/api/v1/agents/evaluation-criteria/evaluate` | Run evaluation extraction independently |

## Data and Idempotency

- Documents are linked by `CompanyId`, `TenderId`, `DocumentId`, and `ChunkId`.
- Qdrant payloads retain source document, page, heading, classification, and table
  metadata for traceability.
- Requirement extraction skips chunk IDs already marked as processed.
- Deduplication skips source requirement IDs already included in an existing
  result.
- Evaluation processing fingerprints both chunk identity and content, so changed
  evidence can be processed again.
- Token-usage and enterprise events are inserted directly into the MongoDB
  `TokenUsageLog` and `EnterpriseLog` collections. No logging URL or bearer token
  is required.

## Tests

Run the automated tests with:

```bash
python -m unittest discover -s tests -v
```

## Current State

The tender document pipeline is working end to end: PDF OCR and indexing,
requirement extraction, safe requirement deduplication, evaluation-criteria
extraction, persistent stage tracking, and direct or queued execution are all
implemented.
