"""Environment-backed collection names for document processing."""

from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


def _required_environment_value(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


@dataclass(frozen=True)
class DocumentCollectionNames:
    company_mongo: str
    tender_mongo: str
    section_mongo: str
    company_qdrant: str
    tender_qdrant: str

    @classmethod
    def from_environment(cls) -> "DocumentCollectionNames":
        return cls(
            company_mongo=_required_environment_value(
                "COMPANYDOCUMENTCOLLECTIONMONGO"
            ),
            tender_mongo=_required_environment_value(
                "TENDERDOCUMENTCOLLECTIONMONGO"
            ),
            section_mongo=_required_environment_value(
                "SECTIONDOCUMENTCOLLECTIONMONGO"
            ),
            company_qdrant=_required_environment_value(
                "COMPANYDOCUMENTCOLLECTIONQDRANT"
            ),
            tender_qdrant=_required_environment_value(
                "TENDERDOCUMENTCOLLECTIONQDRANT"
            ),
        )


__all__ = [
    "DocumentCollectionNames",
]
