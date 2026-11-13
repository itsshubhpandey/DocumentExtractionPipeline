"""
Evaluation Criteria Agent
=========================

End-to-end orchestration for evaluation criteria extraction:

1. Retrieve evidence chunks from Qdrant using the profile-driven retriever.
2. Normalise retrieved chunks into the extraction prompt input contract.
3. Ask OpenAI for schema-shaped JSON.
4. Run deterministic validation checks.
5. Write both the retrieval evidence pack and final Evaluation_Evidence.json.

Example:
    py evaluation_criteria_agent.py --company-id <id> --tender-id <id>

Dry run:
    py evaluation_criteria_agent.py --company-id <id> --tender-id <id> --dry-run
"""

# =============================================================================
# 1. Imports
# =============================================================================

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from app.agents.EvaluationCriteriaAgent.services.output_service import (
    EvaluationCriteriaOutputService,
)
from app.agents.EvaluationCriteriaAgent.services.processing_state import (
    chunk_fingerprint,
    get_processed_chunk_fingerprints,
    mark_processed_document_ids,
    mark_processed_chunk_fingerprints,
)


# =============================================================================
# 2. Constants
# =============================================================================

DEFAULT_PROFILE_PATH = "app/agents/EvaluationCriteriaAgent/retrievalProfile/evaluation_retrieval_profile_complete (1).json"
DEFAULT_SPEC_PATH = "app/agents/EvaluationCriteriaAgent/resources/specification.json"
DEFAULT_CONSTITUTION_PATH = "app/agents/EvaluationCriteriaAgent/resources/constitution.md"
DEFAULT_PROMPT_PATH = "app/agents/EvaluationCriteriaAgent/prompts/runtime_prompt.md"
DEFAULT_RETRIEVAL_OUTPUT_PATH = "evaluation_evidence_debug.json"
DEFAULT_FINAL_OUTPUT_PATH = "Evaluation_Evidence.json"
DEFAULT_QDRANT_URL = "https://b18a6879-f119-47c1-8a5e-17af45713616.eu-west-2-0.aws.cloud.qdrant.io"
DEFAULT_QDRANT_COLLECTION = "CPTenderDoc"
DEFAULT_DENSE_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_MODEL = "gpt-4.1"


def _ec_env(name: str, default: Any = None) -> Any:
    """Read an Evaluation Criteria-specific environment variable."""
    return os.environ.get(f"EC_{name}", default)


# =============================================================================
# 3. Dataclasses
# =============================================================================

@dataclass(frozen=True)
class AgentPaths:
    profile: Path
    specification: Path
    constitution: Path
    runtime_prompt: Path
    retrieval_output: Path
    final_output: Path


@dataclass(frozen=True)
class OpenAIUsageRecord:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    latency: float
    company_id: str | None
    tender_id: str | None
    agent_name: str
    success: bool


# =============================================================================
# 4. Configuration
# =============================================================================


class ConfigurationError(Exception):
    """Raised for missing or invalid environment configuration."""


def load_environment() -> None:
    """Load the project .env while preserving explicitly exported values."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    path = Path(__file__).resolve().with_name(".env")
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip():
            value = value.strip().strip("'\"")
            os.environ.setdefault(key.strip(), value)


load_environment()


class Configuration:
    load_environment = staticmethod(load_environment)

    @staticmethod
    def parse_args() -> argparse.Namespace:
        return parse_args()

    @staticmethod
    def require_args(args: argparse.Namespace) -> None:
        require_args(args)


# =============================================================================
# 5. Logging
# =============================================================================


class LoggerError(Exception):
    """Raised internally when usage telemetry cannot be delivered."""


def _separator(character: str = "=") -> None:
    print(character * 60)


def _print_header(args: argparse.Namespace) -> None:
    _separator()
    print(f"Evaluation Criteria Agent {_ec_env('AGENT_VERSION', 'v1.0')}")
    _separator()
    print(f"Company:        {args.company_id}")
    print(f"Tender:         {args.tender_id}")
    print(f"Collection:     {args.qdrant_collection}")
    print(f"Retrieval Mode: {_ec_env('QDRANT_RETRIEVAL_MODE', 'AUTO')}")
    print(f"Vector Mode:    {_ec_env('QDRANT_VECTOR_MODE', 'AUTO')}")
    print(f"Dense Model:    {os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')}")
    print(f"Sparse Model:   {args.sparse_model_name if args.enable_sparse else 'disabled'}")
    _separator()


class AgentLogging:
    separator = staticmethod(_separator)
    print_header = staticmethod(_print_header)


# =============================================================================
# 6. Usage Tracker
# =============================================================================


class OpenAIUsageLogger:
    """Best-effort client for the central .NET OpenAI usage API."""

    def __init__(self) -> None:
        self.url = _ec_env("OPENAI_USAGE_LOGGER_URL", "").strip()
        self.timeout = float(_ec_env("OPENAI_USAGE_LOGGER_TIMEOUT", "5"))

    def post(self, record: OpenAIUsageRecord) -> None:
        if not self.url:
            logging.warning("OPENAI_USAGE_LOGGER_URL is not configured; usage telemetry skipped.")
            return
        payload = json.dumps(record.__dict__).encode("utf-8")
        request = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if not 200 <= response.status < 300:
                    raise LoggerError(f"Usage logger returned HTTP {response.status}.")
        except Exception as exc:
            logging.warning("OpenAI usage telemetry unavailable: %s", exc)


# =============================================================================
# 7. File Loader
# =============================================================================


class FileLoader:
    @staticmethod
    def load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def load_text(path: Path) -> str:
        return path.read_text(encoding="utf-8")


load_json = FileLoader.load_json
load_text = FileLoader.load_text


# =============================================================================
# 8. Retriever Service
# =============================================================================


class RetrieverService:
    """Namespace for the existing retriever construction and execution API."""

    @staticmethod
    def build(*args: Any, **kwargs: Any) -> Any:
        return build_retriever(*args, **kwargs)

    @staticmethod
    def retrieve(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return run_retrieval(*args, **kwargs)


# =============================================================================
# 9. LLM Service
# =============================================================================


class OpenAIError(Exception):
    """Raised when the extraction request fails."""


def coerce_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _mongo_database_name() -> str:
    return os.getenv("MONGO_DB_NAME") or os.getenv("MONGODB_DATABASE") or "DocAI"


def _candidate_tender_identity_queries(tender_id: str) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = [
        {"TenderId": tender_id},
        {"tenderId": tender_id},
        {"tender_id": tender_id},
        {"Id": tender_id},
        {"id": tender_id},
        {"_id": tender_id},
    ]

    try:
        from bson import ObjectId

        if ObjectId.is_valid(tender_id):
            queries.append({"_id": ObjectId(tender_id)})
    except Exception:
        pass

    return queries


def _candidate_company_queries(company_id: str | None) -> list[dict[str, Any]]:
    if not company_id:
        return []

    return [
        {"CompanyProfileId": company_id},
        {"companyProfileId": company_id},
        {"company_profile_id": company_id},
        {"CompanyId": company_id},
        {"companyId": company_id},
        {"company_id": company_id},
    ]


def _candidate_cp_tender_queries(
    tender_id: str,
    company_id: str | None,
) -> list[dict[str, Any]]:
    tender_queries = _candidate_tender_identity_queries(tender_id)
    company_queries = _candidate_company_queries(company_id)

    if not company_queries:
        return tender_queries

    return [
        {"$and": [tender_query, {"$or": company_queries}]}
        for tender_query in tender_queries
    ] + tender_queries


def _normalise_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalise_for_json(child)
            for key, child in value.items()
        }

    if isinstance(value, list):
        return [_normalise_for_json(child) for child in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _find_key_recursive(value: Any, names: set[str]) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in names and child not in (None, ""):
                return child

        for child in value.values():
            found = _find_key_recursive(child, names)
            if found not in (None, ""):
                return found

    if isinstance(value, list):
        for child in value:
            found = _find_key_recursive(child, names)
            if found not in (None, ""):
                return found

    return None


def _extract_ocid(tender_document: dict[str, Any]) -> str:
    ocid = _find_key_recursive(
        tender_document,
        {
            "ocid",
            "opencontractingid",
            "open_contracting_id",
        },
    )
    return str(ocid or "").strip()


def _chunk_text(text: str, chunk_size: int, max_chunks: int) -> list[str]:
    if not text:
        return []

    chunks = [
        text[index : index + chunk_size]
        for index in range(0, len(text), chunk_size)
    ]
    return chunks[:max_chunks]


def build_tender_details_evidence_pack(
    company_id: str | None,
    tender_id: str,
) -> dict[str, Any]:
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        logging.warning("Mongo fallback skipped: MONGO_URI is not configured.")
        return {
            "retrieval_profile_id": "mongo_tender_details_fallback",
            "evidence_chunk_count": 0,
            "evidence_chunks": [],
        }

    import certifi
    from pymongo import MongoClient

    cp_tenders_collection = _ec_env("CP_TENDERS_COLLECTION", "CPTenders")
    tender_details_collection = _ec_env(
        "TENDER_DETAILS_COLLECTION",
        "TenderDetails",
    )
    chunk_size = int(_ec_env("TENDER_DETAILS_CHUNK_CHAR_LIMIT", "6000"))
    max_chunks = int(_ec_env("TENDER_DETAILS_MAX_CHUNKS", "12"))

    client_options = {}
    if os.getenv("MONGO_TLS", "false").lower() in {
        "1", "true", "yes", "on"
    }:
        client_options["tlsCAFile"] = certifi.where()
    client = MongoClient(mongo_uri, **client_options)
    try:
        db = client[_mongo_database_name()]
        cp_tender = None
        for query in _candidate_cp_tender_queries(tender_id, company_id):
            cp_tender = db[cp_tenders_collection].find_one(query)
            if cp_tender:
                break

        if not cp_tender:
            logging.info(
                "Mongo fallback skipped: no CPTenders document found | tender_id=%s",
                tender_id,
            )
            return {
                "retrieval_profile_id": "mongo_tender_details_fallback",
                "evidence_chunk_count": 0,
                "evidence_chunks": [],
            }

        ocid = _extract_ocid(cp_tender)
        if not ocid:
            logging.info(
                "Mongo fallback skipped: CPTenders document has no ocid | tender_id=%s",
                tender_id,
            )
            return {
                "retrieval_profile_id": "mongo_tender_details_fallback",
                "evidence_chunk_count": 0,
                "evidence_chunks": [],
            }

        tender_details = db[tender_details_collection].find_one(
            {
                "$or": [
                    {"ocid": ocid},
                    {"OCID": ocid},
                    {"Ocid": ocid},
                    {"releases.ocid": ocid},
                    {"Releases.OCID": ocid},
                    {"tender.ocid": ocid},
                    {"Tender.Ocid": ocid},
                ]
            }
        )

        if not tender_details:
            logging.info(
                "Mongo fallback skipped: no TenderDetails document found | ocid=%s",
                ocid,
            )
            return {
                "retrieval_profile_id": "mongo_tender_details_fallback",
                "evidence_chunk_count": 0,
                "evidence_chunks": [],
            }

        tender_details_json = json.dumps(
            _normalise_for_json(tender_details),
            ensure_ascii=False,
            default=str,
        )
        chunks = _chunk_text(tender_details_json, chunk_size, max_chunks)
        document_id = f"mongo-tender-details:{ocid}"
        evidence_chunks = []

        for index, chunk in enumerate(chunks, start=1):
            chunk_id = f"{document_id}:chunk:{index}"
            payload = {
                "DocumentId": document_id,
                "ChunkId": chunk_id,
                "DocumentName": "TenderDetails",
                "DocumentType": "MongoTenderDetails",
                "PageID": f"{document_id}:page:1",
                "PageNumber": 1,
                "ChunkIndex": index,
                "RelatedSection": "TenderDetails",
                "Text": chunk,
                "OriginalSource": "MongoTenderDetails",
                "Ocid": ocid,
            }
            evidence_chunks.append(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "document_name": "TenderDetails",
                    "document_type": "MongoTenderDetails",
                    "page_id": payload["PageID"],
                    "page_number": 1,
                    "chunk_index": index,
                    "related_section": "TenderDetails",
                    "text": chunk,
                    "payload": payload,
                }
            )

        logging.info(
            "Mongo fallback retrieved TenderDetails evidence | ocid=%s | chunks=%s",
            ocid,
            len(evidence_chunks),
        )
        return {
            "retrieval_profile_id": "mongo_tender_details_fallback",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "collection_name": tender_details_collection,
            "retrieval_mode": "mongo_tender_details_fallback",
            "company_id": company_id,
            "tender_id": tender_id,
            "ocid": ocid,
            "evidence_chunk_count": len(evidence_chunks),
            "evidence_chunks": evidence_chunks,
            "retrieval_statistics": {
                "qdrant_evidence_missing": True,
                "mongo_fallback_chunks": len(evidence_chunks),
            },
        }
    finally:
        client.close()


def payload_alias(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def normalise_evidence_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a retriever chunk into the canonical EvidenceChunks contract.

    The retriever preserves original Qdrant payload under "payload"; prefer
    that for PascalCase fields and fall back to the serialised snake_case keys.
    """
    payload = dict(chunk.get("payload") or {})
    return {
        "DocumentId": first_present(
            payload_alias(payload, "DocumentId", "document_id"),
            chunk.get("document_id"),
        ),
        "ChunkId": first_present(
            payload_alias(payload, "ChunkId", "chunk_id"),
            chunk.get("chunk_id"),
        ),
        "DocumentName": first_present(
            payload_alias(payload, "DocumentName", "document_name"),
            chunk.get("document_name"),
        ),
        "DocumentType": first_present(
            payload_alias(payload, "DocumentType", "document_type"),
            chunk.get("document_type"),
        ),
        "PageID": first_present(
            payload_alias(payload, "PageID", "page_id"),
            chunk.get("page_id"),
        ),
        "PageNumber": coerce_int_or_none(
            first_present(payload_alias(payload, "PageNumber", "page_number"), chunk.get("page_number"))
        ),
        "ChunkIndex": coerce_int_or_none(
            first_present(payload_alias(payload, "ChunkIndex", "chunk_index"), chunk.get("chunk_index"))
        ),
        "RelatedSection": first_present(
            payload_alias(payload, "RelatedSection", "related_section", "SectionHeading", "section_heading"),
            chunk.get("related_section"),
        ),
        "TableId": payload_alias(payload, "TableId", "table_id"),
        "TableCaption": payload_alias(payload, "TableCaption", "table_caption"),
        "TableHeaders": payload_alias(payload, "TableHeaders", "table_headers"),
        "TableJson": payload_alias(payload, "TableJson", "table_json"),
        "Text": first_present(payload_alias(payload, "Text", "text"), chunk.get("text"), ""),
        "OriginalPayload": payload,
    }


def build_llm_input(
    evidence_pack: dict[str, Any],
    company_id: str | None,
    tender_id: str,
    *,
    existing_evaluation: dict[str, Any] | None = None,
    previously_processed_source_ids: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    evidence_chunks = [
        normalise_evidence_chunk(chunk)
        for chunk in evidence_pack.get("evidence_chunks", [])
        if isinstance(chunk, dict)
    ]
    llm_input: dict[str, Any] = {
        "TenderContext": {
            "CompanyId": company_id,
            "TenderId": tender_id,
            "RetrievalProfileId": evidence_pack.get("retrieval_profile_id"),
        },
        "EvidenceChunks": evidence_chunks,
    }
    if existing_evaluation is not None:
        llm_input["ExistingEvaluation"] = existing_evaluation
    if previously_processed_source_ids:
        llm_input["PreviouslyProcessedSourceIds"] = previously_processed_source_ids
    return llm_input


def full_not_found_output(company_id: str | None, tender_id: str) -> dict[str, Any]:
    return {
        "SchemaVersion": "1.0",
        "CompanyId": company_id,
        "TenderId": tender_id,
        "EvaluationModelDetected": False,
        "ExtractionStatus": "NOT_FOUND",
        "OverallScoring": {
            "AwardMethod": None,
            "QualityWeightPercent": None,
            "PriceWeightPercent": None,
            "OtherOverallWeightings": [],
            "ScoringScale": {
                "Minimum": None,
                "Maximum": None,
                "Description": None,
                "SourceEvidence": [],
            },
            "PriceScoringFormula": None,
            "SourceEvidence": [],
        },
        "Criteria": [],
        "MandatoryConditions": [],
        "Conflicts": [],
        "MissingInformation": [
            {
                "Field": "EvaluationCriteria",
                "Reason": (
                    "No explicit evaluation, award, weighting, scoring, threshold, "
                    "pass/fail, or price-scoring evidence was found in the supplied evidence chunks."
                ),
                "RelatedCriterionId": None,
                "SourceEvidence": [],
            }
        ],
        "Validation": {
            "SourceCoverage": {
                "Status": "NOT_APPLICABLE",
                "Explanation": "No criteria or conditions were extracted.",
            },
            "NumericValidity": {
                "Status": "NOT_APPLICABLE",
                "Explanation": "No numeric evaluation values were extracted.",
            },
            "OverallWeightingReconciliation": {
                "Status": "NOT_ASSESSABLE",
                "CalculatedTotalPercent": None,
                "ExpectedTotalPercent": None,
                "Explanation": "No complete explicit overall weighting allocation was found.",
            },
            "ParentChildReconciliation": [],
            "PassFailSeparation": {
                "Status": "NOT_APPLICABLE",
                "Explanation": "No criteria or conditions were extracted.",
            },
            "SupersessionControl": {
                "Status": "NOT_APPLICABLE",
                "Explanation": "No supersession evidence was found.",
            },
            "MissingDataDiscipline": {
                "Status": "VALID",
                "Explanation": "No unsupported evaluation facts were emitted.",
            },
        },
        "ReviewFlags": [],
        "RequiresHumanReview": False,
        "SourceDocuments": [],
    }


class LLMService:
    """Namespace for the existing LLM input and extraction functions."""

    normalise_evidence_chunk = staticmethod(normalise_evidence_chunk)
    build_llm_input = staticmethod(build_llm_input)
    full_not_found_output = staticmethod(full_not_found_output)

    @staticmethod
    def extract(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return create_openai_extraction(*args, **kwargs)


# =============================================================================
# 10. Validation
# =============================================================================


class AgentValidationError(RuntimeError):
    """Raised when the final extraction violates schema or grounding rules."""


def make_retrieved_source_index(llm_input: dict[str, Any]) -> set[tuple[str, str]]:
    source_index: set[tuple[str, str]] = set()
    for chunk in llm_input.get("EvidenceChunks", []):
        document_id = chunk.get("DocumentId")
        chunk_id = chunk.get("ChunkId")
        if document_id is not None and chunk_id is not None:
            source_index.add((str(document_id), str(chunk_id)))
    for source in llm_input.get("PreviouslyProcessedSourceIds", []):
        if not isinstance(source, dict):
            continue
        document_id = source.get("DocumentId")
        chunk_id = source.get("ChunkId")
        if document_id is not None and chunk_id is not None:
            source_index.add((str(document_id), str(chunk_id)))
    return source_index


def _normalise_evidence_text(value: Any) -> str:
    """Normalise text for deterministic evidence-excerpt matching."""
    if not isinstance(value, str):
        return ""
    normalised = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalised.split())


def _evidence_tokens(value: Any) -> list[str]:
    """Return comparable words from an evidence excerpt or chunk."""

    return re.findall(r"[a-z0-9]+", _normalise_evidence_text(value))


def _fuzzy_evidence_score(excerpt: str, chunk_text: str) -> float:
    """Score a faithful-but-not-byte-identical excerpt against a chunk."""

    excerpt_tokens = _evidence_tokens(excerpt)
    chunk_tokens = _evidence_tokens(chunk_text)
    if len(excerpt_tokens) < 8 or not chunk_tokens:
        return 0.0

    matching_tokens = len(set(excerpt_tokens) & set(chunk_tokens))
    token_coverage = matching_tokens / len(set(excerpt_tokens))
    sequence_match = SequenceMatcher(
        None,
        excerpt_tokens,
        chunk_tokens,
        autojunk=False,
    ).find_longest_match(0, len(excerpt_tokens), 0, len(chunk_tokens))
    sequence_coverage = sequence_match.size / len(excerpt_tokens)
    return max(token_coverage, sequence_coverage)


def _evidence_search_text(chunk: dict[str, Any]) -> str:
    """Build searchable text from text and supplied table context."""
    def structured_text(value: Any) -> str:
        if isinstance(value, dict):
            fragments = [json.dumps(value, ensure_ascii=False, sort_keys=True)]
            for key, child in value.items():
                child_text = structured_text(child)
                fragments.extend(
                    [
                        str(key),
                        child_text,
                        f"{key} | {child_text}",
                    ]
                )
            return " ".join(fragments)
        if isinstance(value, list):
            return " ".join([json.dumps(value, ensure_ascii=False), *(structured_text(item) for item in value)])
        return str(value)

    parts: list[str] = []
    for field in ("Text", "TableCaption", "TableHeaders", "TableJson"):
        value = chunk.get(field)
        if value is None:
            continue
        parts.append(structured_text(value))
    return _normalise_evidence_text(" ".join(parts))


def canonicalize_source_evidence(
    output: dict[str, Any],
    llm_input: dict[str, Any],
) -> dict[str, Any]:
    """Replace LLM citation IDs with IDs from the retrieved evidence pack.

    The model sometimes copies a UUID incorrectly even when its excerpt came
    from a retrieved chunk.  Source IDs must therefore be canonicalized from
    the retrieved data, never guessed from the model output.  A citation is
    changed only when its normalized excerpt matches exactly one retrieved
    chunk.  Ambiguous and unmatched excerpts are left untouched so the normal
    grounding validator can reject them.
    """
    retrieved_source_index = make_retrieved_source_index(llm_input)
    retrieved_chunks: list[dict[str, Any]] = []
    for chunk in llm_input.get("EvidenceChunks", []):
        if not isinstance(chunk, dict):
            continue
        document_id = chunk.get("DocumentId")
        chunk_id = chunk.get("ChunkId")
        text = _evidence_search_text(chunk)
        if document_id is None or chunk_id is None or not text:
            continue
        retrieved_chunks.append(
            {
                "DocumentId": str(document_id),
                "ChunkId": str(chunk_id),
                "PageID": chunk.get("PageID"),
                "PageNumber": chunk.get("PageNumber"),
                "ChunkIndex": chunk.get("ChunkIndex"),
                "DocumentName": chunk.get("DocumentName"),
                "Text": text,
            }
        )

    # Incremental runs retrieve only new chunks. Keep the text of previously
    # validated evidence available as a conservative fallback so a model that
    # copies an old citation with a changed UUID can still be remapped to the
    # persisted source. The prior source itself was already accepted by this
    # agent's validation before it was saved.
    known_source_keys = {
        (chunk["DocumentId"], chunk["ChunkId"])
        for chunk in retrieved_chunks
    }
    for evidence in iter_source_evidence(
        llm_input.get("ExistingEvaluation", {})
    ):
        document_id = evidence.get("DocumentId")
        chunk_id = evidence.get("ChunkId")
        text = _normalise_evidence_text(evidence.get("EvidenceExcerpt"))
        if document_id is None or chunk_id is None or not text:
            continue
        source_key = (str(document_id), str(chunk_id))
        if source_key in known_source_keys:
            continue
        retrieved_chunks.append(
            {
                "DocumentId": source_key[0],
                "ChunkId": source_key[1],
                "PageID": evidence.get("PageID"),
                "PageNumber": evidence.get("PageNumber"),
                "ChunkIndex": evidence.get("ChunkIndex"),
                "DocumentName": evidence.get("DocumentName"),
                "Text": text,
            }
        )
        known_source_keys.add(source_key)

    mapped_count = 0
    fuzzy_mapped_count = 0
    ambiguous_count = 0
    unmatched_count = 0

    def map_evidence(value: Any) -> None:
        nonlocal mapped_count, fuzzy_mapped_count, ambiguous_count, unmatched_count
        if isinstance(value, dict):
            evidence_items = value.get("SourceEvidence")
            if isinstance(evidence_items, list):
                for evidence in evidence_items:
                    if not isinstance(evidence, dict):
                        continue

                    excerpt = _normalise_evidence_text(evidence.get("EvidenceExcerpt"))
                    if not excerpt:
                        continue

                    existing_source = (
                        str(evidence.get("DocumentId")),
                        str(evidence.get("ChunkId")),
                    )
                    if existing_source in retrieved_source_index:
                        continue

                    candidates = [
                        chunk
                        for chunk in retrieved_chunks
                        if excerpt in chunk["Text"]
                    ]

                    # Existing locator fields are useful disambiguators, but
                    # never create a match by themselves.
                    for field in ("DocumentId", "PageID", "PageNumber", "ChunkIndex"):
                        hint = evidence.get(field)
                        if hint is None:
                            continue
                        narrowed = [
                            chunk
                            for chunk in candidates
                            if chunk.get(field) is not None
                            and str(chunk.get(field)) == str(hint)
                        ]
                        if narrowed:
                            candidates = narrowed

                    if len(candidates) == 1:
                        canonical = candidates[0]
                        for field in (
                            "DocumentId",
                            "ChunkId",
                            "PageID",
                            "PageNumber",
                            "ChunkIndex",
                        ):
                            if canonical.get(field) is not None:
                                evidence[field] = canonical[field]
                        mapped_count += 1
                    elif len(candidates) > 1:
                        ambiguous_count += 1
                    else:
                        # The model is allowed to return a short faithful
                        # excerpt, so punctuation/OCR differences can prevent
                        # an exact substring match. Restrict fuzzy matching to
                        # known document/page hints where possible, then
                        # require a strong and unique match.
                        fuzzy_pool = retrieved_chunks
                        for field in (
                            "DocumentId",
                            "PageID",
                            "PageNumber",
                            "ChunkIndex",
                        ):
                            hint = evidence.get(field)
                            if hint is None:
                                continue
                            hinted = [
                                chunk
                                for chunk in fuzzy_pool
                                if chunk.get(field) is not None
                                and str(chunk.get(field)) == str(hint)
                            ]
                            if hinted:
                                fuzzy_pool = hinted

                        scored: list[tuple[float, dict[str, Any]]] = []
                        for chunk in fuzzy_pool:
                            score = _fuzzy_evidence_score(
                                excerpt,
                                chunk["Text"],
                            )
                            if score >= 0.80:
                                scored.append((score, chunk))
                        scored.sort(key=lambda item: item[0], reverse=True)

                        if scored:
                            best_score, best_chunk = scored[0]
                            second_score = scored[1][0] if len(scored) > 1 else 0.0
                            if (
                                best_score >= 0.80
                                and best_score - second_score >= 0.05
                            ):
                                for field in (
                                    "DocumentId",
                                    "ChunkId",
                                    "PageID",
                                    "PageNumber",
                                    "ChunkIndex",
                                ):
                                    if best_chunk.get(field) is not None:
                                        evidence[field] = best_chunk[field]
                                mapped_count += 1
                                fuzzy_mapped_count += 1
                            else:
                                ambiguous_count += 1
                        else:
                            unmatched_count += 1

            for child in value.values():
                map_evidence(child)
        elif isinstance(value, list):
            for child in value:
                map_evidence(child)

    map_evidence(output)
    if mapped_count:
        logging.info(
            "Canonicalized evaluation evidence source IDs | mapped=%s | "
            "fuzzy_mapped=%s | ambiguous=%s | unmatched=%s",
            mapped_count,
            fuzzy_mapped_count,
            ambiguous_count,
            unmatched_count,
        )
    elif ambiguous_count or unmatched_count:
        logging.warning(
            "Could not canonicalize evaluation evidence source IDs | "
            "ambiguous=%s | unmatched=%s",
            ambiguous_count,
            unmatched_count,
        )
    return output


def iter_source_evidence(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if set(("DocumentId", "ChunkId", "EvidenceExcerpt")).issubset(value.keys()):
            yield value
        for child in value.values():
            yield from iter_source_evidence(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_source_evidence(child)


def iter_percentage_fields(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    percentage_names = {
        "QualityWeightPercent",
        "PriceWeightPercent",
        "WeightPercent",
        "MinimumPercent",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            next_path = f"{path}.{key}" if path else key
            if key in percentage_names:
                yield next_path, child
            yield from iter_percentage_fields(child, next_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_percentage_fields(child, f"{path}[{index}]")


def run_json_schema_validation(output: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        required = schema.get("required", [])
        return [f"Missing required top-level field: {key}" for key in required if key not in output]

    validator = Draft202012Validator(schema)
    return [error.message for error in sorted(validator.iter_errors(output), key=str)]


def normalize_criterion_categories(output: dict[str, Any]) -> None:
    """Correct an unambiguous category/flag contradiction before validation.

    Models occasionally set ``PassFail`` or ``Mandatory`` while leaving the
    primary category as ``WEIGHTED_CRITERION``.  This is deterministic from
    the already extracted flags, so do not discard the complete evaluation
    for that narrow contradiction.
    """
    for index, item in enumerate(output.get("Criteria", []) or []):
        if item.get("PrimaryCategory") != "WEIGHTED_CRITERION":
            continue
        if item.get("SecondaryCategories"):
            continue

        if item.get("Mandatory") is True:
            item["PrimaryCategory"] = "MANDATORY_REQUIREMENT"
            if item.get("PassFail") is True:
                item["SecondaryCategories"] = ["PASS_FAIL_GATE"]
            logging.warning(
                "Normalized Criteria[%s] category from WEIGHTED_CRITERION "
                "to MANDATORY_REQUIREMENT based on Mandatory=true",
                index,
            )
        elif item.get("PassFail") is True:
            item["PrimaryCategory"] = "PASS_FAIL_GATE"
            logging.warning(
                "Normalized Criteria[%s] category from WEIGHTED_CRITERION "
                "to PASS_FAIL_GATE based on PassFail=true",
                index,
            )


def deterministic_post_checks(
    output: dict[str, Any],
    specification: dict[str, Any],
    llm_input: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    schema = specification["output_contract"]["schema"]

    errors.extend(run_json_schema_validation(output, schema))

    for field in specification["output_contract"].get("required_top_level_fields", []):
        if field not in output:
            errors.append(f"Missing required top-level field: {field}")

    for collection_name in ("Criteria", "MandatoryConditions"):
        for index, item in enumerate(output.get(collection_name, []) or []):
            if not item.get("SourceEvidence"):
                errors.append(f"{collection_name}[{index}] has empty SourceEvidence.")

    for index, item in enumerate(output.get("Criteria", []) or []):
        primary = item.get("PrimaryCategory")
        secondary = set(item.get("SecondaryCategories") or [])
        weighted_only = primary == "WEIGHTED_CRITERION" and not secondary
        if weighted_only and (item.get("PassFail") is True or item.get("Mandatory") is True):
            errors.append(
                "Criteria[%s] is a pass/fail or mandatory item represented only "
                "as WEIGHTED_CRITERION." % index
            )

    for path, value in iter_percentage_fields(output):
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path} must be numeric or null.")
        elif value < 0 or value > 100:
            errors.append(f"{path} must be between 0 and 100.")

    for evidence in iter_source_evidence(output):
        excerpt = evidence.get("EvidenceExcerpt")
        if not isinstance(excerpt, str) or not excerpt.strip():
            errors.append("SourceEvidence contains an empty EvidenceExcerpt.")

    if (output.get("Conflicts") or output.get("ReviewFlags")) and not output.get("RequiresHumanReview"):
        errors.append("RequiresHumanReview must be true when Conflicts or ReviewFlags are present.")

    return errors


def validate_or_raise(
    output: dict[str, Any],
    specification: dict[str, Any],
    llm_input: dict[str, Any],
) -> None:
    errors = deterministic_post_checks(output, specification, llm_input)
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise AgentValidationError(f"Evaluation output failed validation:\n{joined}")


class ValidationService:
    make_retrieved_source_index = staticmethod(make_retrieved_source_index)
    canonicalize_source_evidence = staticmethod(canonicalize_source_evidence)
    iter_source_evidence = staticmethod(iter_source_evidence)
    iter_percentage_fields = staticmethod(iter_percentage_fields)
    run_json_schema_validation = staticmethod(run_json_schema_validation)
    normalize_criterion_categories = staticmethod(normalize_criterion_categories)
    deterministic_post_checks = staticmethod(deterministic_post_checks)
    validate_or_raise = staticmethod(validate_or_raise)


def create_openai_extraction(
    llm_input: dict[str, Any],
    specification: dict[str, Any],
    constitution_text: str,
    runtime_prompt_text: str,
    model: str,
    usage_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from app.infrastructure.load_llms import create_llm
    from app.infrastructure.token_usage import TokenUsageService as _TokenUsageService

    # Structured output now goes through the shared lite-LLM infra
    # (app.infrastructure.load_llms.create_llm) instead of a hand-built
    # openai.OpenAI() client hardcoded to OpenAI's Responses API. litellm
    # passes response_format straight through to whichever provider is
    # actually configured (Mongo LlmProvoderConfig, falling back to
    # LLM_PROVIDER), so this agent now follows the same active
    # provider/model as every other agent on create_llm() instead of being
    # pinned to OpenAI regardless of that config. The `model` parameter
    # kept for signature compatibility is no longer what selects the
    # model -- create_llm() resolves that from the shared config -- it's
    # only used as a fallback label if usage metadata doesn't report one.
    #
    # create_llm() itself is called with zero arguments here, exactly like
    # every other agent -- structured-output enforcement (.with_structured_
    # output()) is applied locally, below, on the object it returns, so
    # this agent's needs stay scoped to this file instead of adding
    # parameters to the shared function every other agent also calls.
    output_schema = copy.deepcopy(specification["output_contract"]["schema"])

    def make_structured_output_compatible(value: Any) -> None:
        if isinstance(value, dict):
            if "type" not in value and "const" in value:
                constant = value["const"]
                value["type"] = ("boolean" if isinstance(constant, bool) else
                                 "number" if isinstance(constant, (int, float)) else
                                 "null" if constant is None else "string")
            elif "type" not in value and "enum" in value and value["enum"]:
                non_null = [item for item in value["enum"] if item is not None]
                if non_null and all(isinstance(item, str) for item in non_null):
                    value["type"] = ["string", "null"] if len(non_null) != len(value["enum"]) else "string"
            # OpenAI/litellm structured-output strict mode requires every
            # key in `properties` to also appear in `required` (a
            # genuinely optional field is expressed via a nullable type,
            # not by omitting it). Fields without a natural "empty" value
            # for their existing type (arrays/objects already have one --
            # [] / {}) get one added here so the model has a valid way to
            # supply "nothing" instead of the request being rejected.
            if value.get("type") == "object" and isinstance(value.get("properties"), dict):
                properties = value["properties"]
                required = list(value.get("required") or [])
                required_set = set(required)
                for key, child in properties.items():
                    if key not in required_set:
                        if isinstance(child, dict) and child.get("type") not in ("array", "object"):
                            existing_type = child.get("type")
                            if isinstance(existing_type, list):
                                if "null" not in existing_type:
                                    existing_type.append("null")
                            elif isinstance(existing_type, str) and existing_type != "null":
                                child["type"] = [existing_type, "null"]
                        required.append(key)
                        required_set.add(key)
                value["required"] = required
            for key, child in list(value.items()):
                if isinstance(child, dict) and not child:
                    value[key] = {"anyOf": [{"type": kind} for kind in ("string", "number", "boolean", "null")]}
                else:
                    make_structured_output_compatible(child)
        elif isinstance(value, list):
            for child in value:
                make_structured_output_compatible(child)

    make_structured_output_compatible(output_schema)

    system_content = "\n\n".join(
        [
            runtime_prompt_text,
            "Mandatory constitution:",
            constitution_text,
            "Return JSON only. Use the supplied EvidenceChunks for new evidence. "
            "If ExistingEvaluation is supplied, return a complete merged evaluation "
            "and preserve previously validated findings unless new evidence explicitly "
            "changes or supersedes them. PreviouslyProcessedSourceIds are already "
            "validated source references.",
        ]
    )
    user_content = json.dumps(llm_input, ensure_ascii=False)

    llm = create_llm().with_structured_output(
        output_schema,
        method="json_schema",
        strict=True,
        include_raw=True,
    )

    started = time.perf_counter()
    success = False
    raw_response = None
    output: dict[str, Any] | None = None
    try:
        result = llm.invoke(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
        )
        raw_response = result.get("raw")
        parsing_error = result.get("parsing_error")
        if parsing_error is not None:
            raise OpenAIError(f"Structured output parsing failed: {parsing_error}")
        output = result.get("parsed")
        if output is None:
            raise OpenAIError("LLM returned no parsed structured output")
        success = True
    except OpenAIError:
        raise
    except Exception as exc:
        raise OpenAIError(f"LLM extraction request failed: {exc}") from exc
    finally:
        latency = time.perf_counter() - started
        usage = (
            _TokenUsageService.extract_token_usage(raw_response)
            if raw_response is not None
            else {}
        )
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        resolved_model = usage.get("model") or model
        input_rate = float(_ec_env("OPENAI_INPUT_COST_PER_1M", "0"))
        output_rate = float(_ec_env("OPENAI_OUTPUT_COST_PER_1M", "0"))
        record = OpenAIUsageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(), model=resolved_model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost=round((input_tokens * input_rate + output_tokens * output_rate) / 1_000_000, 8),
            latency=round(latency, 6),
            company_id=llm_input.get("TenderContext", {}).get("CompanyId"),
            tender_id=llm_input.get("TenderContext", {}).get("TenderId"),
            agent_name=_ec_env("AGENT_NAME", "Evaluation Criteria Agent"), success=success,
        )
        OpenAIUsageLogger().post(record)
        from app.infrastructure.token_usage import TokenUsageService

        usage_context = usage_context or {}
        try:
            import asyncio
            from types import SimpleNamespace

            jwt_token = usage_context.get("jwt_token")
            if jwt_token:
                logging.getLogger("uvicorn.error").info(
                    "[AIUsageLogger] submission started | run_id=%s",
                    usage_context.get("run_id", ""),
                )
                # Route through the shared helper (same one every other agent uses)
                # instead of hand-building the payload, so this agent automatically
                # inherits future shared-format fixes -- e.g. sourceIds only ever
                # carrying Company/Tender entries, with projectId kept as its own
                # top-level field rather than a third sourceIds entry.
                request_like = SimpleNamespace(
                    company_id=record.company_id,
                    tender_id=record.tender_id,
                    user_id=usage_context.get("user_id", ""),
                    project_id=usage_context.get("project_id", ""),
                )
                asyncio.run(
                    TokenUsageService.log_agent_usage(
                        request=request_like,
                        token_usage={
                            "input_tokens": record.input_tokens,
                            "output_tokens": record.output_tokens,
                            "total_tokens": record.total_tokens,
                            "model": record.model,
                        },
                        bearer_token=jwt_token,
                        correlation_id=usage_context.get("run_id", ""),
                        duration_ms=int(round(record.latency * 1000)),
                        application_name=_ec_env("AI_USAGE_APPLICATION_NAME", "Tender"),
                        agent_name=_ec_env("AGENT_NAME", "Evaluation Criteria Agent"),
                        purpose="Evaluation Criteria extraction",
                        usage_type="LLM",
                        method="Lite LLM structured output",
                    )
                )
            else:
                logging.getLogger("uvicorn.error").warning(
                    "[AIUsageLogger] skipped: JwtToken was not provided | run_id=%s",
                    usage_context.get("run_id", ""),
                )
        except Exception as exc:
            logging.warning("Shared token usage logging unavailable: %s", exc)
        logging.info("OpenAI usage | input=%s | output=%s | total=%s | cost=%.6f | latency=%.2fs",
                     record.input_tokens, record.output_tokens, record.total_tokens,
                     record.estimated_cost, record.latency)

    def remove_identical_evidence_within_objects(value: Any) -> None:
        if isinstance(value, dict):
            evidence = value.get("SourceEvidence")
            if isinstance(evidence, list):
                unique: list[Any] = []
                seen: set[str] = set()
                for item in evidence:
                    identity = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if identity not in seen:
                        seen.add(identity)
                        unique.append(item)
                value["SourceEvidence"] = unique
                object_name = value.get("Name")
                if isinstance(object_name, str) and object_name.strip():
                    for item in unique:
                        if not isinstance(item, dict):
                            continue
                        excerpt = item.get("EvidenceExcerpt")
                        if not isinstance(excerpt, str) or "\n|" not in excerpt:
                            continue
                        matching_rows = [
                            line.strip().strip("|").strip()
                            for line in excerpt.splitlines()
                            if object_name.lower() in line.lower()
                            and not set(line.strip()) <= {"|", "-", ":"}
                        ]
                        if matching_rows:
                            item["EvidenceExcerpt"] = matching_rows[0]
            for child in value.values():
                remove_identical_evidence_within_objects(child)
        elif isinstance(value, list):
            for child in value:
                remove_identical_evidence_within_objects(child)

    remove_identical_evidence_within_objects(output)
    return output


def build_retriever(
    profile: dict[str, Any],
    qdrant_url: str,
    qdrant_api_key: str | None,
    collection_name: str,
    dense_model_name: str,
    enable_sparse: bool,
    sparse_model_name: str,
) -> Any:
    from app.agents.EvaluationCriteriaAgent.services import retrieval_service as retriever_module

    retriever_module.FIELDS = retriever_module.configure_payload_fields(profile)
    dense_embedder = retriever_module.build_local_dense_embedder(dense_model_name)

    sparse_embedder = None
    if enable_sparse:
        sparse_embedder = retriever_module.build_fastembed_sparse_embedder(sparse_model_name)

    client = retriever_module.QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=float(_ec_env("QDRANT_SEARCH_TIMEOUT", "60")),
    )
    return retriever_module.EvaluationCriteriaRetriever(
        client=client,
        collection_name=collection_name,
        profile=profile,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )


def run_retrieval(
    company_id: str,
    tender_id: str,
    profile: dict[str, Any],
    qdrant_url: str,
    qdrant_api_key: str | None,
    collection_name: str,
    dense_model_name: str,
    enable_sparse: bool,
    sparse_model_name: str,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    retriever = build_retriever(
        profile=profile,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        dense_model_name=dense_model_name,
        enable_sparse=enable_sparse,
        sparse_model_name=sparse_model_name,
    )
    return retriever.retrieve(
        company_id=company_id,
        tender_id=tender_id,
        document_ids=document_ids,
    )


# =============================================================================
# 11. Output Writer
# =============================================================================


class OutputWriter:
    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def write_compact_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")),
            encoding="utf-8",
        )


write_json = OutputWriter.write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve and extract tender evaluation criteria.")
    parser.add_argument("--company-id", default=_ec_env("COMPANY_ID"))
    parser.add_argument("--tender-id", default=_ec_env("TENDER_ID"))
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL))
    parser.add_argument(
        "--qdrant-api-key",
        default=os.environ.get("QDRANT_API_KEY") or None,
    )
    parser.add_argument(
        "--qdrant-collection",
        default=_ec_env("QDRANT_COLLECTION_NAME", DEFAULT_QDRANT_COLLECTION),
    )
    parser.add_argument(
        "--dense-model-name",
        default=os.environ.get("EMBEDDING_MODEL", DEFAULT_DENSE_MODEL),
    )
    parser.add_argument(
        "--enable-sparse",
        action="store_true",
        default=_ec_env("QDRANT_ENABLE_SPARSE", "false").strip().lower() in {"1", "true", "yes"},
    )
    parser.add_argument("--sparse-model-name", default=_ec_env("QDRANT_SPARSE_MODEL", "Qdrant/bm25"))
    parser.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    parser.add_argument("--profile-path", default=_ec_env("RETRIEVAL_PROFILE_PATH", DEFAULT_PROFILE_PATH))
    parser.add_argument("--spec-path", default=_ec_env("SPECIFICATION_PATH", DEFAULT_SPEC_PATH))
    parser.add_argument(
        "--constitution-path",
        default=_ec_env("CONSTITUTION_PATH", DEFAULT_CONSTITUTION_PATH),
    )
    parser.add_argument("--prompt-path", default=_ec_env("RUNTIME_PROMPT_PATH", DEFAULT_PROMPT_PATH))
    parser.add_argument(
        "--retrieval-output",
        default=_ec_env("RETRIEVAL_OUTPUT_PATH", DEFAULT_RETRIEVAL_OUTPUT_PATH),
    )
    parser.add_argument(
        "--final-output",
        default=_ec_env("FINAL_OUTPUT_PATH", DEFAULT_FINAL_OUTPUT_PATH),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Retrieve Qdrant evidence and write only evaluation_evidence.json.",
    )
    parser.add_argument(
        "--skip-llm-when-no-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit the schema-valid NOT_FOUND output without calling the LLM when retrieval returns no chunks.",
    )
    return parser.parse_args(argv)


def require_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.company_id:
        missing.append("--company-id or COMPANY_ID")
    if not args.tender_id:
        missing.append("--tender-id or TENDER_ID")
    if missing:
        raise SystemExit("Missing required input: " + ", ".join(missing))


def _existing_evaluation_context(
    output: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not output:
        return None

    metadata_fields = {
        "_id",
        "Status",
        "IsRegenerated",
        "CreatedBy",
        "CreatedAt",
        "RunId",
        "UpdatedAt",
    }
    return {
        key: copy.deepcopy(value)
        for key, value in output.items()
        if key not in metadata_fields
    }


def _source_ids_from_evaluation(
    output: dict[str, Any] | None,
) -> list[dict[str, str]]:
    source_ids: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for evidence in iter_source_evidence(output or {}):
        document_id = evidence.get("DocumentId")
        chunk_id = evidence.get("ChunkId")
        if document_id in (None, "") or chunk_id in (None, ""):
            continue
        key = (str(document_id), str(chunk_id))
        if key in seen:
            continue
        seen.add(key)
        source_ids.append({"DocumentId": key[0], "ChunkId": key[1]})
    return source_ids


# =============================================================================
# 12. EvaluationCriteriaAgent class
# =============================================================================


def _execute(
    args: argparse.Namespace,
    usage_context: dict[str, Any] | None = None,
    persist: bool = True,
    is_regenerated: bool = False,
    created_by: str = "",
    run_id: str = "",
    document_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    require_args(args)
    _print_header(args)
    total_started = time.perf_counter()
    scoped_document_ids = (
        sorted(
            {
                str(document_id).strip()
                for document_id in document_ids
                if str(document_id).strip()
            }
        )
        if document_ids is not None
        else None
    )

    debug_enabled = _ec_env("ENABLE_DEBUG_LOGS", "false").lower() in {"1", "true", "yes", "on"}
    artifact_dir = Path("app/agents/EvaluationCriteriaAgent/artifacts")
    paths = AgentPaths(
        profile=Path(args.profile_path),
        specification=Path(args.spec_path),
        constitution=Path(args.constitution_path),
        runtime_prompt=Path(args.prompt_path),
        retrieval_output=(artifact_dir / "retrieval_output.json") if debug_enabled else Path(args.retrieval_output),
        final_output=(artifact_dir / "Evaluation_Evidence.json") if debug_enabled else Path(args.final_output),
    )

    profile = load_json(paths.profile)
    specification = load_json(paths.specification)

    logging.info("Retrieving evaluation evidence...")
    evidence_pack = run_retrieval(
        company_id=args.company_id,
        tender_id=args.tender_id,
        document_ids=scoped_document_ids,
        profile=profile,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.qdrant_collection,
        dense_model_name=args.dense_model_name,
        enable_sparse=args.enable_sparse,
        sparse_model_name=args.sparse_model_name,
    )
    logging.info("✓ Retrieved %s chunks", evidence_pack.get("evidence_chunk_count", 0))
    logging.info("✓ Expanded Context")
    logging.info("✓ Deduplicated")

    if args.dry_run:
        logging.info("Dry run requested; skipping LLM extraction.")
        return None

    llm_input = build_llm_input(evidence_pack, args.company_id, args.tender_id)
    if not llm_input["EvidenceChunks"] and scoped_document_ids is None:
        logging.info(
            "No Qdrant evaluation evidence found; trying Mongo TenderDetails fallback."
        )
        fallback_evidence_pack = build_tender_details_evidence_pack(
            company_id=args.company_id,
            tender_id=args.tender_id,
        )
        fallback_llm_input = build_llm_input(
            fallback_evidence_pack,
            args.company_id,
            args.tender_id,
        )
        if fallback_llm_input["EvidenceChunks"]:
            evidence_pack = fallback_evidence_pack
            llm_input = fallback_llm_input
            logging.info(
                "Mongo TenderDetails fallback will be used for evaluation extraction | chunks=%s",
                len(llm_input["EvidenceChunks"]),
            )

    existing_output = None
    current_chunk_fingerprints = {
        fingerprint
        for chunk in llm_input["EvidenceChunks"]
        if (fingerprint := chunk_fingerprint(chunk)) is not None
    }

    if not is_regenerated:
        existing_output = EvaluationCriteriaOutputService.get_active(
            args.company_id,
            args.tender_id,
        )

    if existing_output and not current_chunk_fingerprints:
        logging.info(
            "Evaluation Criteria skipped; no evidence chunks are currently "
            "available, preserving the last successful output | company_id=%s "
            "| tender_id=%s",
            args.company_id,
            args.tender_id,
        )
        if persist and scoped_document_ids:
            mark_processed_document_ids(
                args.company_id,
                args.tender_id,
                set(scoped_document_ids),
            )
        return existing_output

    if existing_output and current_chunk_fingerprints:
        processed_fingerprints = get_processed_chunk_fingerprints(
            args.company_id,
            args.tender_id,
        )
        if processed_fingerprints:
            new_chunks = [
                chunk
                for chunk in llm_input["EvidenceChunks"]
                if chunk_fingerprint(chunk) not in processed_fingerprints
            ]
            if not new_chunks:
                logging.info(
                    "Evaluation Criteria skipped; no new or changed chunks | "
                    "company_id=%s | tender_id=%s",
                    args.company_id,
                    args.tender_id,
                )
                if persist and scoped_document_ids:
                    mark_processed_document_ids(
                        args.company_id,
                        args.tender_id,
                        set(scoped_document_ids),
                    )
                return existing_output

            incremental_pack = dict(evidence_pack)
            incremental_pack["evidence_chunks"] = new_chunks
            incremental_pack["evidence_chunk_count"] = len(new_chunks)
            llm_input = build_llm_input(
                incremental_pack,
                args.company_id,
                args.tender_id,
                existing_evaluation=_existing_evaluation_context(existing_output),
                previously_processed_source_ids=_source_ids_from_evaluation(
                    existing_output
                ),
            )
            logging.info(
                "Evaluation Criteria incremental run | new_chunks=%s | "
                "previously_processed=%s",
                len(new_chunks),
                len(processed_fingerprints),
            )
        else:
            logging.info(
                "Evaluation Criteria state not found; performing one full run "
                "to initialise processed chunk state"
            )
            if scoped_document_ids:
                llm_input = build_llm_input(
                    evidence_pack,
                    args.company_id,
                    args.tender_id,
                    existing_evaluation=_existing_evaluation_context(
                        existing_output
                    ),
                    previously_processed_source_ids=_source_ids_from_evaluation(
                        existing_output
                    ),
                )

    if args.skip_llm_when_no_evidence and not llm_input["EvidenceChunks"]:
        final_output = full_not_found_output(args.company_id, args.tender_id)
    else:
        constitution_text = load_text(paths.constitution)
        runtime_prompt_text = load_text(paths.runtime_prompt)
        _separator()
        logging.info("Calling OpenAI | model=%s", args.openai_model)
        final_output = create_openai_extraction(
            llm_input=llm_input,
            specification=specification,
            constitution_text=constitution_text,
            runtime_prompt_text=runtime_prompt_text,
            model=args.openai_model,
            usage_context=usage_context,
        )
        logging.info("✓ Request Complete")

    logging.info("Running validation...")
    canonicalize_source_evidence(final_output, llm_input)
    normalize_criterion_categories(final_output)
    validate_or_raise(final_output, specification, llm_input)
    logging.info("✓ Validation Passed")
    if persist:
        mongo_document_id = EvaluationCriteriaOutputService.save(
            output=final_output,
            is_regenerated=is_regenerated,
            created_by=created_by,
            run_id=run_id,
        )
        mark_processed_chunk_fingerprints(
            args.company_id,
            args.tender_id,
            current_chunk_fingerprints,
        )
        if scoped_document_ids:
            mark_processed_document_ids(
                args.company_id,
                args.tender_id,
                set(scoped_document_ids),
            )
        logging.info(
            "Saved evaluation output to MongoDB | "
            "collection=EvaluationCriteria | document_id=%s | status=%s",
            mongo_document_id,
            "Regenerating" if is_regenerated else "Active",
        )
    _separator()
    logging.info("Completed Successfully | total_time=%.2fs", time.perf_counter() - total_started)
    _separator()
    return final_output


def _run_main() -> dict[str, Any] | None:
    return _execute(parse_args())


class EvaluationCriteriaAgent:
    """Coordinates the existing configuration, retrieval, LLM, and validation flow."""

    run = staticmethod(_run_main)

    @staticmethod
    def execute(
        company_id: str,
        tender_id: str,
        document_ids: list[str] | None = None,
        is_regenerated: bool = False,
        created_by: str = "",
        run_id: str = "",
        usage_context: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        args = parse_args([])
        args.company_id = company_id
        args.tender_id = tender_id

        output = _execute(
            args=args,
            usage_context=usage_context,
            persist=persist,
            is_regenerated=is_regenerated,
            created_by=created_by,
            run_id=run_id,
            document_ids=document_ids,
        )

        if output is None:
            raise RuntimeError("Evaluation Criteria execution did not produce a final output.")
        return output


# =============================================================================
# 13. main()
# =============================================================================


def main() -> None:
    debug = _ec_env("ENABLE_DEBUG_LOGS", "false").lower() in {"1", "true", "yes", "on"}
    level_name = "DEBUG" if debug else _ec_env("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format="[%(levelname)s] %(message)s")
    try:
        EvaluationCriteriaAgent.run()
    except SystemExit:
        raise
    except Exception as exc:
        if debug:
            logging.exception("Evaluation Criteria Agent failed")
        else:
            logging.error("Evaluation Criteria Agent failed: %s", exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
