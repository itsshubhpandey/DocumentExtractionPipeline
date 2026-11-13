


from __future__ import annotations
from bson import ObjectId
import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from bson import ObjectId
from pymongo import ReturnDocument
from app.agents.deduplication.requirement_deduplication_agent import (
    RequirementDeduplicationAgent,
)
# from app.agents.evidence.services.service import (
#     EvidenceSummaryService,
# )
from app.infrastructure.motor_database import (
    get_deduplication_motor_collection,
    get_requirement_extraction_motor_collection,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def serialize_mongo_value(value: Any) -> Any:
    """
    Convert MongoDB values into JSON-safe values
    for the GET API response.
    """

    if isinstance(value, ObjectId):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): serialize_mongo_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            serialize_mongo_value(item)
            for item in value
        ]

    return value


def is_list_value(value: Any) -> bool:
    return isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    )


def normalize_key(value: Any) -> str:
    return (
        str(value)
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .lower()
    )


def extract_requirements_from_document(
    document: Any,
) -> list[Any]:
    """
    Find requirements in common MongoDB structures.

    Supports:

    1. One requirement per MongoDB document.
    2. Requirement arrays such as RawRequirements.
    3. Nested requirement arrays such as Output.Requirements.

    All MongoDB ObjectId and datetime values are converted
    into JSON-safe values before requirements are passed to
    the agent.
    """

    requirement_keys = {
        "rawrequirements",
        "requirements",
        "extractedrequirements",
        "detectedrequirements",
        "tenderrequirements",
        "finalrequirements",
        "requirementitems",
    }

    extracted_requirements: list[Any] = []

    # --------------------------------------------------
    # One MongoDB document represents one requirement
    # --------------------------------------------------

    if isinstance(document, Mapping):
        requirement_text = (
            document.get("RequirementText")
            or document.get("requirement_text")
            or document.get("requirementText")
            or document.get("Text")
            or document.get("text")
        )

        requirement_id = (
            document.get("RequirementId")
            or document.get("requirement_id")
            or document.get("requirementId")
        )

        if requirement_text or requirement_id:
            requirement_document = dict(document)

            source_mongo_id = requirement_document.pop(
                "_id",
                None,
            )

            if source_mongo_id is not None:
                requirement_document[
                    "SourceMongoId"
                ] = str(source_mongo_id)

            return [
                serialize_mongo_value(
                    requirement_document
                )
            ]

    # --------------------------------------------------
    # Requirement arrays inside MongoDB documents
    # --------------------------------------------------

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child_value in value.items():
                normalized_key = normalize_key(key)

                if normalized_key in requirement_keys:
                    if is_list_value(child_value):
                        for requirement in child_value:
                            if requirement is not None:
                                cleaned_requirement = (
                                    serialize_mongo_value(
                                        requirement
                                    )
                                )

                                if isinstance(
                                    cleaned_requirement,
                                    Mapping,
                                ):
                                    cleaned_requirement = dict(
                                        cleaned_requirement
                                    )

                                    source_mongo_id = (
                                        cleaned_requirement.pop(
                                            "_id",
                                            None,
                                        )
                                    )

                                    if (
                                        source_mongo_id
                                        is not None
                                    ):
                                        cleaned_requirement[
                                            "SourceMongoId"
                                        ] = str(
                                            source_mongo_id
                                        )

                                extracted_requirements.append(
                                    cleaned_requirement
                                )

                    elif child_value is not None:
                        cleaned_requirement = (
                            serialize_mongo_value(
                                child_value
                            )
                        )

                        if isinstance(
                            cleaned_requirement,
                            Mapping,
                        ):
                            cleaned_requirement = dict(
                                cleaned_requirement
                            )

                            source_mongo_id = (
                                cleaned_requirement.pop(
                                    "_id",
                                    None,
                                )
                            )

                            if source_mongo_id is not None:
                                cleaned_requirement[
                                    "SourceMongoId"
                                ] = str(
                                    source_mongo_id
                                )

                        extracted_requirements.append(
                            cleaned_requirement
                        )

                else:
                    walk(child_value)

        elif is_list_value(value):
            for child_value in value:
                walk(child_value)

    walk(document)

    return extracted_requirements
    
def requirement_identity(
    requirement: Any,
) -> str:
    """
    Remove only exact duplicate input objects.

    Similar requirements must remain because the
    LLM is responsible for semantic deduplication.
    """

    if isinstance(requirement, Mapping):
        for key in (
            "RequirementId",
            "requirement_id",
            "Id",
            "id",
            "_id",
        ):
            requirement_id = requirement.get(key)

            if requirement_id is not None:
                requirement_id_text = str(
                    requirement_id
                ).strip()

                if requirement_id_text:
                    return f"id:{requirement_id_text}"

    return json.dumps(
        requirement,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def remove_exact_duplicates(
    requirements: list[Any],
) -> list[Any]:
    result: list[Any] = []
    identities: set[str] = set()

    for requirement in requirements:
        identity = requirement_identity(
            requirement
        )

        if identity in identities:
            continue

        identities.add(identity)
        result.append(requirement)

    return result


class RequirementDeduplicationService:
    def __init__(
        self,
        source_collection: Any | None = None,
        destination_collection: Any | None = None,
        agent: RequirementDeduplicationAgent | None = None,
    ) -> None:
        self._source_collection = source_collection
        self._destination_collection = (
            destination_collection
        )
        self._agent = agent

    @property
    def source_collection(self) -> Any:
        """
        Input/source collection:

        TenderRequirements
        """

        if self._source_collection is None:
            self._source_collection = (
                get_requirement_extraction_motor_collection()
            )

        return self._source_collection

    @property
    def destination_collection(self) -> Any:
        """
        Output/destination collection:

        TenderRequirementDeduplications
        """

        if self._destination_collection is None:
            self._destination_collection = (
                get_deduplication_motor_collection()
            )

        return self._destination_collection

    @property
    def collection(self) -> Any:
        """
        Compatibility property for existing code.
        """

        return self.destination_collection

    @property
    def agent(self) -> RequirementDeduplicationAgent:
        if self._agent is None:
            self._agent = (
                RequirementDeduplicationAgent()
            )

        return self._agent

    def build_source_query(
        self,
        company_id: str,
        tender_id: str,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        Supports both string IDs and Mongo ObjectIds,
        along with common field-name variations.
        """

        company_values: list[Any] = [
            company_id
        ]

        tender_values: list[Any] = [
            tender_id
        ]

        if ObjectId.is_valid(company_id):
            company_values.append(
                ObjectId(company_id)
            )

        if ObjectId.is_valid(tender_id):
            tender_values.append(
                ObjectId(tender_id)
            )

        query = {
            "$and": [
                {
                    "$or": [
                        {
                            "CompanyId": {
                                "$in": company_values
                            }
                        },
                        {
                            "companyId": {
                                "$in": company_values
                            }
                        },
                        {
                            "company_id": {
                                "$in": company_values
                            }
                        },
                    ]
                },
                {
                    "$or": [
                        {
                            "TenderId": {
                                "$in": tender_values
                            }
                        },
                        {
                            "tenderId": {
                                "$in": tender_values
                            }
                        },
                        {
                            "tender_id": {
                                "$in": tender_values
                            }
                        },
                    ]
                },
            ]
        }

        if status:
            query["$and"].append({"Status": status})

        return query

    async def read_source_documents(
        self,
        company_id: str,
        tender_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Read input documents from
        TenderRequirements.
        """

        query = self.build_source_query(
            company_id=company_id,
            tender_id=tender_id,
            status=status,
        )

        print(
            "Source database:",
            self.source_collection.database.name,
        )

        print(
            "Source collection:",
            self.source_collection.name,
        )

        print(
            "Source query:",
            query,
        )

        source_documents = [
            document
            async for document in self.source_collection.find(query).sort("_id", -1)
        ]

        print(
            "Source documents found:",
            len(source_documents),
        )

        if not source_documents:
            raise ValueError(
                "No source documents were found for "
                "the supplied CompanyId and TenderId. "
                f"Database: "
                f"{self.source_collection.database.name}. "
                f"Collection: "
                f"{self.source_collection.name}. "
                f"CompanyId: {company_id}. "
                f"TenderId: {tender_id}."
            )

        return source_documents

    async def process(
        self,
        request: Any,
        bearer_token: str | None = None,
    ) -> dict[str, Any]:
        """
        Main method called by deduplication_route.py.

        Flow:

        1. Read CompanyId and TenderId.
        2. Read source Mongo documents.
        3. Extract RawRequirements.
        4. Insert processing record in destination.
        5. Send requirements to the LLM.
        6. Save LLM output in destination.
        """

        if hasattr(request, "model_dump"):
            payload = request.model_dump(
                mode="python"
            )

        elif isinstance(request, Mapping):
            payload = dict(request)

        else:
            raise ValueError(
                "Request must be a Pydantic model "
                "or dictionary"
            )

        company_id = str(
            payload.get(
                "CompanyId",
                "",
            )
        ).strip()

        tender_id = str(
            payload.get(
                "TenderId",
                "",
            )
        ).strip()

        is_regenerate = bool(
            payload.get(
                "IsRegenerate",
                False,
            )
        )

        final_status = (
            "Regenerating"
            if is_regenerate
            else "Active"
        )


        user_id = str(
            payload.get(
                "UserId",
                "",
            )
        ).strip()

        user_name = str(
            payload.get(
                "UserName",
                "",
            )
        ).strip()

        project_id = str(
            payload.get(
                "ProjectId",
                "",
            )
        ).strip()


        if not company_id:
            raise ValueError(
                "CompanyId is required"
            )

        if not tender_id:
            raise ValueError(
                "TenderId is required"
            )

        # ----------------------------------------------
        # Read source Mongo documents
        # ----------------------------------------------

        source_documents = (
            await self.read_source_documents(
                company_id=company_id,
                tender_id=tender_id,
                status=final_status,
            )
        )

        raw_requirements: list[Any] = []

        for source_document in source_documents:
            requirements = (
                extract_requirements_from_document(
                    source_document
                )
            )

            raw_requirements.extend(
                requirements
            )

        raw_requirements = remove_exact_duplicates(
            raw_requirements
        )

        requirements_with_regeneration: list[Any] = []

        for requirement in raw_requirements:
            if isinstance(
                requirement,
                Mapping,
            ):
                requirement_data = dict(
                    requirement
                )

                requirement_data[
                    "IsRegenerated"
                ] = is_regenerate

                requirements_with_regeneration.append(
                    requirement_data
                )

            else:
                requirements_with_regeneration.append(
                    requirement
                )

        raw_requirements = (
            requirements_with_regeneration
        )


        if not raw_requirements:
            raise ValueError(
                "Source documents were found, but "
                "no requirements were found. "
                "Expected a top-level RequirementText, "
                "RawRequirements, Requirements, "
                "ExtractedRequirements or "
                "Output.Requirements."
            )

        created_at = utc_now()

        destination_document = {
            "CompanyId": company_id,
            "TenderId": tender_id,
            "IsRegenerate": is_regenerate,
            "UserId": user_id,
            "UserName": user_name,
            "ProjectId": project_id,
            "Summary": {
                "TotalInputRequirements": len(
                    raw_requirements
                ),
                "TotalDeduplicatedRequirements": 0,
                "DuplicatesRemoved": 0,
            },
            "JsonOutput": {
                "DeduplicatedRequirements": []
            },
            "Status": "Processing",
            "Error": None,
            "CreatedAt": created_at,
            "UpdatedAt": created_at,
            "CompletedAt": None,
        }

        # ----------------------------------------------
        # Insert destination Mongo document
        # ----------------------------------------------

        # insert_result = await asyncio.to_thread(
        #     self.destination_collection.insert_one,
        #     destination_document,
        # )

        # deduplication_id = (
        #     insert_result.inserted_id
        # )

        saved_document = await self.destination_collection.find_one_and_replace(
            {
                "CompanyId": company_id,
                "TenderId": tender_id,
                "IsRegenerate": is_regenerate,
            },
            destination_document,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        if not saved_document:
            raise RuntimeError(
                "Failed to create or overwrite the "
                "deduplication document."
            )

        deduplication_id = saved_document["_id"]




        try:
            # context = {
            #     "CompanyId": company_id,
            #     "TenderId": tender_id,
            #     "DeduplicationId": str(
            #         deduplication_id
            #     ),
            #     "SourceCollection": (
            #         self.source_collection.name
            #     ),
            #     "SourceDocumentIds": (
            #         source_document_ids
            #     ),
            # }

            context = {
                "CompanyId": company_id,
                "TenderId": tender_id,
                "IsRegenerate": is_regenerate,
                "IsRegenerated": is_regenerate,
                "UserId": user_id,
                "UserName": user_name,
                "ProjectId": project_id,
                "DeduplicationId": str(
                    deduplication_id
                ),
            }

            # ------------------------------------------
            # Send requirements to the LLM
            # ------------------------------------------

            result = await self.agent.deduplicate(
                raw_requirements=raw_requirements,
                context=context,
                bearer_token=bearer_token,
                correlation_id=str(
                    deduplication_id
                ),
            )

            summary = result.get(
                "Summary",
                {},
            )

            deduplicated_requirements = (
                result.get(
                    "DeduplicatedRequirements",
                    [],
                )
            )

            if not isinstance(summary, Mapping):
                raise ValueError(
                    "Agent 1 returned an invalid Summary."
                )

            if not isinstance(
                deduplicated_requirements,
                list,
            ):
                raise ValueError(
                    "Agent 1 returned an invalid "
                    "DeduplicatedRequirements list."
                )

            compact_output = {
                "DeduplicatedRequirements": (
                    deduplicated_requirements
                )
            }

            completed_at = utc_now()

            # ------------------------------------------
            # Save only the approved compact output.
            # ------------------------------------------

            await self.destination_collection.update_one(
                {
                    "_id": deduplication_id,
                },
                {
                    "$set": {
                        "Summary": dict(summary),
                        "JsonOutput": compact_output,
                        "Status": final_status,
                        "Error": None,
                        "UpdatedAt": completed_at,
                        "CompletedAt": completed_at,
                    }
                },
            )

            return {
                "DeduplicationId": str(
                    deduplication_id
                ),
                "Status": "Completed",
                "Result": {
                    "Summary": dict(summary),
                    "JsonOutput": compact_output,
                },
            }

        except Exception as exc:
            failed_at = utc_now()

            await self.destination_collection.update_one(
                {
                    "_id": deduplication_id,
                },
                {
                    "$set": {
                        "Status": "Failed",
                        "Error": str(exc),
                        "UpdatedAt": failed_at,
                        "CompletedAt": failed_at,
                    }
                },
            )

            raise

    async def get_by_id(
        self,
        deduplication_id: str,
    ) -> dict[str, Any] | None:
        """
        Read a saved deduplication result.
        """

        if not ObjectId.is_valid(
            deduplication_id
        ):
            return None

        document = await self.destination_collection.find_one(
            {
                "_id": ObjectId(
                    deduplication_id
                ),
            },
        )

        if document is None:
            return None

        return serialize_mongo_value(
            document
        )

class RequirementDeduplicationWorkflowService:
    """
    Preserve the existing API sequence:

    1. Run Requirement Deduplication.
    2. Run Evidence Summary after Agent 1 succeeds.
    3. Return the original Agent 1 API response.
    """

    # def __init__(
    #     self,
    #     deduplication_service: RequirementDeduplicationService | None = None,
    #     evidence_service: EvidenceSummaryService | None = None,
    # ) -> None:
    #     self._deduplication_service = deduplication_service
    #     self._evidence_service = evidence_service

    @property
    def deduplication_service(
        self,
    ) -> RequirementDeduplicationService:
        if self._deduplication_service is None:
            self._deduplication_service = (
                RequirementDeduplicationService()
            )

        return self._deduplication_service

    # @property
    # def evidence_service(self) -> EvidenceSummaryService:
    #     if self._evidence_service is None:
    #         self._evidence_service = EvidenceSummaryService()

    #     return self._evidence_service

    async def process(
        self,
        request: Any,
        bearer_token: str | None = None,
    ) -> dict[str, Any]:
        print(
            "Starting Agent 1: "
            "Requirement Deduplication Agent"
        )

        deduplication_result = (
            await self.deduplication_service.process(
                request=request,
                bearer_token=bearer_token,
            )
        )

        deduplication_status = str(
            deduplication_result.get(
                "Status",
                "",
            )
            or ""
        ).strip()

        print(
            "Agent 1 returned status:",
            deduplication_status,
        )

        successful_agent_1_statuses = {
            "Completed",
            "IsRegenerated",
        }

        if (
            deduplication_status
            not in successful_agent_1_statuses
        ):
            raise ValueError(
                "Agent 1 did not complete successfully. "
                f"Returned status: "
                f"{deduplication_status or 'EMPTY'}"
            )

        print("Agent 1 completed successfully.")

        if hasattr(request, "model_dump"):
            evidence_input = request.model_dump(
                mode="python",
                exclude={"Token"},
            )
        elif isinstance(request, Mapping):
            evidence_input = {
                key: value
                for key, value in dict(request).items()
                if key != "Token"
            }
        else:
            raise ValueError(
                "Request must be a Pydantic model "
                "or dictionary"
            )

        deduplication_id = str(
            deduplication_result.get(
                "DeduplicationId",
                "",
            )
            or ""
        )

        evidence_input["DeduplicationId"] = deduplication_id

        print(
            "Starting Agent 2: "
            "Evidence Summary Agent"
        )

        await self.evidence_service.process(
            input_data=evidence_input,
            bearer_token=bearer_token,
            correlation_id=deduplication_id,
        )

        print("Agent 2 completed successfully.")

        return deduplication_result

    async def get_by_id(
        self,
        deduplication_id: str,
    ) -> dict[str, Any] | None:
        return await self.deduplication_service.get_by_id(
            deduplication_id
        )
