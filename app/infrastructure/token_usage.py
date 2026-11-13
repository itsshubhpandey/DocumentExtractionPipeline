# import os
# import httpx
# import logging
# from pathlib import Path
# from typing import Any, Optional

# from dotenv import load_dotenv
# from litellm import cost_per_token
# from app.infrastructure.database import get_database

# load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
# logger = logging.getLogger(__name__)


# class TokenUsageService:
#     @staticmethod
#     def _normalize_model_name(model: Any) -> str:
#         return str(model or "").strip().lower()

#     @staticmethod
#     def _detect_vendor(model_name: str) -> str:
#         provider_prefix = model_name.split("/", 1)[0]
#         if provider_prefix in {"openai", "mistral", "groq", "gemini", "nvidia_nim"}:
#             return "nvidia" if provider_prefix == "nvidia_nim" else provider_prefix

#         if model_name.startswith("mistral-"):
#             return "mistral"

#         return "openai"

#     @staticmethod
#     def _model_name_candidates(model_name: str) -> set[str]:
#         candidates = {model_name}
#         if "/" in model_name:
#             candidates.add(model_name.split("/", 1)[1])
#         return {candidate for candidate in candidates if candidate}

#     @staticmethod
#     def _get_model_pricing(vendor: Any, model_name: str) -> Optional[dict]:
#         vendor_name = str(vendor or TokenUsageService._detect_vendor(model_name)).strip().lower()
#         collection_name = os.getenv(
#             "AI_MODEL_CONFIG_COLLECTION",
#             "AIModelConfig",
#         )

#         try:
#             config = get_database()[collection_name].find_one(
#                 {
#                     "$or": [{"vendor": vendor_name}, {"_id": vendor_name}],
#                     "isActive": True,
#                 }
#             )
#         except Exception as ex:
#             logger.warning("Token cost pricing lookup failed: %s", ex)
#             return None

#         if not config:
#             return None

#         model_candidates = TokenUsageService._model_name_candidates(model_name)
#         for model_config in config.get("models") or []:
#             configured_model = TokenUsageService._normalize_model_name(
#                 model_config.get("modelName")
#             )
#             if (
#                 configured_model not in model_candidates
#                 or not model_config.get("isActive", False)
#             ):
#                 continue

#             return {
#                 "input": float(model_config.get("inputCostPer1M", 0) or 0),
#                 "output": float(model_config.get("outputCostPer1M", 0) or 0),
#                 "currency": str(model_config.get("currency") or "USD"),
#             }

#         return None

#     @staticmethod
#     def _format_cost_value(value: float) -> str:
#         return format(round(value, 8), ".8f").rstrip("0").rstrip(".") or "0"

#     @staticmethod
#     def _cost_payload(value: Any, currency: str = "USD") -> dict:
#         try:
#             cost_value = float(value or 0)
#         except (TypeError, ValueError):
#             cost_value = 0

#         return {
#             "currency": currency,
#             "value": TokenUsageService._format_cost_value(cost_value),
#         }

#     @staticmethod
#     def _get_float_value(source: Any, *keys: str) -> Optional[float]:
#         for key in keys:
#             value = TokenUsageService._get_value(source, key, None)
#             if value is None:
#                 continue

#             try:
#                 return float(value)
#             except (TypeError, ValueError):
#                 continue

#         return None

#     @staticmethod
#     def _extract_response_cost(response: Any, metadata: dict, usage: Any) -> Optional[dict]:
#         cost_value = (
#             TokenUsageService._get_float_value(metadata, "response_cost", "cost")
#             or TokenUsageService._get_float_value(usage, "response_cost", "cost")
#             or TokenUsageService._get_float_value(response, "response_cost", "cost")
#         )

#         if cost_value is None:
#             return None

#         return TokenUsageService._cost_payload(cost_value)

#     @staticmethod
#     def _calculate_litellm_cost(
#         *,
#         model: Any,
#         input_tokens: int = 0,
#         output_tokens: int = 0,
#     ) -> Optional[dict]:
#         model_name = str(model or "").strip()
#         if not model_name:
#             return None

#         try:
#             input_cost, output_cost = cost_per_token(
#                 model=model_name,
#                 prompt_tokens=int(input_tokens or 0),
#                 completion_tokens=int(output_tokens or 0),
#             )
#             value = input_cost + output_cost
#         except Exception as ex:
#             logger.warning("LiteLLM cost calculation failed for model %s: %s", model, ex)
#             return None

#         return TokenUsageService._cost_payload(value)

#     @staticmethod
#     def calculate_token_cost(
#         *,
#         model: Any,
#         input_tokens: int = 0,
#         output_tokens: int = 0,
#         currency: str = "USD",
#         vendor: Any = None,
#     ) -> dict:
#         litellm_cost = TokenUsageService._calculate_litellm_cost(
#             model=model,
#             input_tokens=input_tokens,
#             output_tokens=output_tokens,
#         )

#         if litellm_cost:
#             return litellm_cost

#         model_name = TokenUsageService._normalize_model_name(model)
#         rates = TokenUsageService._get_model_pricing(vendor, model_name)

#         if not rates:
#             logger.warning("Token cost pricing not configured for model: %s", model)
#             return {
#                 "currency": currency,
#                 "value": 0,
#             }

#         input_cost = (int(input_tokens or 0) / 1_000_000) * rates["input"]
#         output_cost = (int(output_tokens or 0) / 1_000_000) * rates["output"]

#         return {
#             "currency": rates.get("currency") or currency,
#             "value": TokenUsageService._format_cost_value(input_cost + output_cost),
#         }

#     @staticmethod
#     def _get_usage_log_api_url() -> Optional[str]:
#         api_url = (
#             os.getenv("AI_USAGE_LOG_API")
#             or os.getenv("TOKEN_USAGE_LOG_API")
#             or os.getenv("TOKEN_USAGE")
#             or os.getenv("Token_UASAGE")
#             or os.getenv("Token_USAGE")
#         )

#         if not api_url:
#             return None

#         return api_url.strip().strip("\"'")

#     @staticmethod
#     def _required_string(value: Any) -> str:
#         if value is None:
#             return "NA"

#         value = str(value).strip()
#         return value or "NA"

#     @staticmethod
#     def _optional_string(value: Any) -> Optional[str]:
#         if value is None:
#             return None

#         value = str(value).strip()
#         return value or None

#     @staticmethod
#     def _source_id(payload: dict, source_type: str) -> Optional[str]:
#         sources = payload.get("sourceIds") or payload.get("SourceId") or []

#         for source in sources:
#             if not isinstance(source, dict):
#                 continue

#             current_source_type = str(
#                 source.get("sourceIdType")
#                 or source.get("SourceIdType")
#                 or ""
#             ).strip().lower()

#             if current_source_type == source_type:
#                 return (
#                     TokenUsageService._optional_string(source.get("id"))
#                     or TokenUsageService._optional_string(source.get("_id"))
#                 )

#         return None

#     @staticmethod
#     def _normalize_source_ids(payload: dict) -> list[dict]:
#         normalized_source_ids = []

#         sources = payload.get("sourceIds") or payload.get("SourceId") or []

#         for source in sources:
#             if not isinstance(source, dict):
#                 continue

#             source_type = str(
#                 source.get("sourceIdType")
#                 or source.get("SourceIdType")
#                 or ""
#             ).strip().lower()

#             if source_type not in {"tender", "company"}:
#                 continue

#             source_id = (
#                 TokenUsageService._optional_string(source.get("id"))
#                 or TokenUsageService._optional_string(source.get("_id"))
#             )

#             if not source_id:
#                 continue

#             normalized_source_ids.append(
#                 {
#                     "sourceIdType": source_type,
#                     "id": source_id,
#                 }
#             )

#         return normalized_source_ids

#     @staticmethod
#     def _normalize_usage_payload(payload: dict) -> dict:
#         normalized_payload = dict(payload)
#         normalized_payload["sourceIds"] = TokenUsageService._normalize_source_ids(payload)
#         normalized_payload.pop("SourceId", None)
#         normalized_payload.pop("projectId", None)
#         normalized_payload.pop("ProjectId", None)
#         normalized_payload.pop("project_id", None)

#         company_id = (
#             TokenUsageService._optional_string(payload.get("companyId"))
#             or TokenUsageService._optional_string(payload.get("CompanyId"))
#             or TokenUsageService._source_id(payload, "company")
#         )
#         tender_id_value = (
#             TokenUsageService._optional_string(payload.get("tenderId"))
#             or TokenUsageService._optional_string(payload.get("TenderId"))
#             or TokenUsageService._source_id(payload, "tender")
#         )

#         user_id = TokenUsageService._required_string(
#             payload.get("userId") or payload.get("UserId")
#         )
#         tender_id = TokenUsageService._required_string(tender_id_value)

#         if company_id:
#             normalized_payload["companyId"] = company_id
#             normalized_payload["CompanyId"] = company_id

#         normalized_payload["userId"] = user_id
#         normalized_payload["tenderId"] = tender_id
#         normalized_payload["UserId"] = user_id
#         normalized_payload["TenderId"] = tender_id

#         return normalized_payload

#     @staticmethod
#     def _get_value(source: Any, key: str, default: Any = None) -> Any:
#         if isinstance(source, dict):
#             return source.get(key, default)
#         return getattr(source, key, default)

#     @staticmethod
#     def extract_token_usage(response: Any) -> dict:
#         metadata = (
#             TokenUsageService._get_value(response, "response_metadata", {}) or {}
#         )

#         usage = (
#             TokenUsageService._get_value(response, "usage_metadata", None)
#             or metadata.get("token_usage")
#             or metadata.get("usage")
#             or {}
#         )

#         input_tokens = (
#             TokenUsageService._get_value(usage, "input_tokens", None)
#             or TokenUsageService._get_value(usage, "prompt_tokens", None)
#             or 0
#         )

#         output_tokens = (
#             TokenUsageService._get_value(usage, "output_tokens", None)
#             or TokenUsageService._get_value(usage, "completion_tokens", None)
#             or 0
#         )

#         total_tokens = (
#             TokenUsageService._get_value(usage, "total_tokens", None)
#             or int(input_tokens or 0) + int(output_tokens or 0)
#         )

#         model = (
#             metadata.get("model_name")
#             or metadata.get("model")
#             or TokenUsageService._get_value(response, "model_name", "")
#             or ""
#         )

#         cost = (
#             TokenUsageService._extract_response_cost(response, metadata, usage)
#             or TokenUsageService.calculate_token_cost(
#                 model=model,
#                 input_tokens=int(input_tokens or 0),
#                 output_tokens=int(output_tokens or 0),
#             )
#         )

#         return {
#             "input_tokens": int(input_tokens or 0),
#             "output_tokens": int(output_tokens or 0),
#             "total_tokens": int(total_tokens or 0),
#             "model": model,
#             "cost": cost,
#         }

#     @staticmethod
#     async def log_usage(payload: dict, bearer_token: Optional[str] = None):
#         api_url = TokenUsageService._get_usage_log_api_url()

#         if not api_url:
#             logger.warning("Token usage logging skipped: AI_USAGE_LOG_API is not configured")
#             return None

#         headers = {}
#         if bearer_token:
#             token = bearer_token.strip()
#             if token.lower().startswith("bearer "):
#                 headers["Authorization"] = token
#             else:
#                 headers["Authorization"] = f"Bearer {token}"
#         else:
#             logger.warning("Token usage logging skipped: bearer_token is required")
#             return None

#         async with httpx.AsyncClient(timeout=30) as client:
#             try:
#                 payload = TokenUsageService._normalize_usage_payload(payload)
#                 response = await client.post(api_url, json=payload, headers=headers)

#                 if response.status_code == 404:
#                     alternate_api_url = (
#                         api_url.rstrip("/")
#                         if api_url.endswith("/")
#                         else f"{api_url}/"
#                     )

#                     if alternate_api_url != api_url:
#                         response = await client.post(
#                             alternate_api_url,
#                             json=payload,
#                             headers=headers,
#                         )

#                 response.raise_for_status()
#                 return response.json()

#             except httpx.HTTPStatusError as ex:
#                 logger.warning(
#                     "Token usage logging failed with status %s: %s",
#                     ex.response.status_code,
#                     ex.response.text,
#                 )
#                 return {
#                     "success": False,
#                     "status_code": ex.response.status_code,
#                     "message": "Token usage logging failed",
#                 }

#             except httpx.HTTPError as ex:
#                 logger.warning("Token usage logging request failed: %s", ex)
#                 return {
#                     "success": False,
#                     "message": "Token usage logging request failed",
#                 }

#     @staticmethod
#     async def log_agent_usage(
#         *,
#         request,
#         token_usage: dict,
#         bearer_token: str,
#         correlation_id: str,
#         duration_ms: int,
#         application_name: str,
#         agent_name: str,
#         purpose: str,
#         usage_type: str = "Completion",
#         method: str = "LLM",
#     ):
#         def request_value(*names: str):
#             for name in names:
#                 value = getattr(request, name, None)
#                 if value is not None and str(value).strip():
#                     return value
#             return None

#         tender_id = request_value("tender_id", "tenderId", "TenderId")
#         company_id = request_value("company_id", "companyId", "CompanyId")
#         user_id = request_value("user_id", "userId", "UserId")

#         source_ids = []

#         if tender_id:
#             source_ids.append(
#                 {
#                     "sourceIdType": "tender",
#                     "id": tender_id,
#                 }
#             )

#         if company_id:
#             source_ids.append(
#                 {
#                     "sourceIdType": "company",
#                     "id": company_id,
#                 }
#             )

#         payload = {
#             "applicationName": application_name,
#             "sourceIds": source_ids,
#             "runId": correlation_id,
#             "userId": user_id or os.getenv("AI_USAGE_USER_ID", ""),
#             "purpose": purpose,
#             "method": method,
#             "agentName": agent_name,
#             "usageType": usage_type,
#             "inputToken": token_usage.get("input_tokens", 0),
#             "outputToken": token_usage.get("output_tokens", 0),
#             "totalTokens": token_usage.get("total_tokens", 0),
#             "model": token_usage.get("model", ""),
#             "duration": duration_ms,
#             "cost": token_usage.get("cost") or TokenUsageService.calculate_token_cost(
#                 model=token_usage.get("model"),
#                 input_tokens=token_usage.get("input_tokens", 0),
#                 output_tokens=token_usage.get("output_tokens", 0),
#             ),
#             "companyId": company_id,
#             "tenderId": tender_id,
#         }
#         print(payload)

#         return await TokenUsageService.log_usage(
#             payload=payload,
#             bearer_token=bearer_token,
#         )


# # ######################################## Token Uasge ########################################

# # usage = TokenUsageService.extract_token_usage(response)

# # await TokenUsageService.log_usage(
# #     payload=usage,
# #     bearer_token=token,






from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from dotenv import load_dotenv
from litellm import cost_per_token

from app.infrastructure.database import get_database
from app.infrastructure.mongo_log_store import insert_token_usage


load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    override=False,
)

logger = logging.getLogger(__name__)


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class TokenUsageService:
    @staticmethod
    def _is_enabled() -> bool:
        return _parse_bool(
            os.getenv("AI_USAGE_LOG_ENABLED"),
            default=True,
        )

    @staticmethod
    def _debug_enabled() -> bool:
        return _parse_bool(
            os.getenv("AI_USAGE_DEBUG_PRINT"),
            default=False,
        )

    @staticmethod
    def _normalize_model_name(model: Any) -> str:
        return str(model or "").strip().lower()

    @staticmethod
    def _detect_vendor(model_name: str) -> str:
        provider_prefix = model_name.split("/", 1)[0]

        if provider_prefix in {
            "openai",
            "mistral",
            "groq",
            "gemini",
            "nvidia_nim",
        }:
            return (
                "nvidia"
                if provider_prefix == "nvidia_nim"
                else provider_prefix
            )

        if model_name.startswith("mistral-"):
            return "mistral"

        return "openai"

    @staticmethod
    def _model_name_candidates(model_name: str) -> set[str]:
        candidates = {model_name}

        if "/" in model_name:
            candidates.add(
                model_name.split("/", 1)[1]
            )

        return {
            candidate
            for candidate in candidates
            if candidate
        }

    @staticmethod
    def _get_model_pricing(
        vendor: Any,
        model_name: str,
    ) -> Optional[dict]:
        vendor_name = str(
            vendor
            or TokenUsageService._detect_vendor(model_name)
        ).strip().lower()

        collection_name = os.getenv(
            "AI_MODEL_CONFIG_COLLECTION",
            "AIModelConfig",
        )

        try:
            config = get_database()[collection_name].find_one(
                {
                    "$or": [
                        {"vendor": vendor_name},
                        {"_id": vendor_name},
                    ],
                    "isActive": True,
                }
            )
        except Exception as ex:
            logger.warning(
                "Token cost pricing lookup failed: %s",
                ex,
            )
            return None

        if not config:
            return None

        model_candidates = (
            TokenUsageService._model_name_candidates(
                model_name
            )
        )

        for model_config in (
            config.get("models") or []
        ):
            configured_model = (
                TokenUsageService._normalize_model_name(
                    model_config.get("modelName")
                )
            )

            if (
                configured_model not in model_candidates
                or not model_config.get(
                    "isActive",
                    False,
                )
            ):
                continue

            return {
                "input": float(
                    model_config.get(
                        "inputCostPer1M",
                        0,
                    )
                    or 0
                ),
                "output": float(
                    model_config.get(
                        "outputCostPer1M",
                        0,
                    )
                    or 0
                ),
                "currency": str(
                    model_config.get("currency")
                    or "USD"
                ),
            }

        return None

    @staticmethod
    def _format_cost_value(value: float) -> str:
        return (
            format(
                round(value, 8),
                ".8f",
            )
            .rstrip("0")
            .rstrip(".")
            or "0"
        )

    @staticmethod
    def _cost_payload(
        value: Any,
        currency: str = "USD",
    ) -> dict:
        try:
            cost_value = float(value or 0)
        except (TypeError, ValueError):
            cost_value = 0

        return {
            "currency": currency,
            "value": TokenUsageService._format_cost_value(
                cost_value
            ),
        }

    @staticmethod
    def _get_float_value(
        source: Any,
        *keys: str,
    ) -> Optional[float]:
        for key in keys:
            value = TokenUsageService._get_value(
                source,
                key,
                None,
            )

            if value is None:
                continue

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _extract_response_cost(
        response: Any,
        metadata: dict,
        usage: Any,
    ) -> Optional[dict]:
        cost_value = (
            TokenUsageService._get_float_value(
                metadata,
                "response_cost",
                "cost",
            )
            or TokenUsageService._get_float_value(
                usage,
                "response_cost",
                "cost",
            )
            or TokenUsageService._get_float_value(
                response,
                "response_cost",
                "cost",
            )
        )

        if cost_value is None:
            return None

        return TokenUsageService._cost_payload(
            cost_value
        )

    @staticmethod
    def _calculate_litellm_cost(
        *,
        model: Any,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> Optional[dict]:
        model_name = str(model or "").strip()

        if not model_name:
            return None

        try:
            input_cost, output_cost = cost_per_token(
                model=model_name,
                prompt_tokens=int(input_tokens or 0),
                completion_tokens=int(output_tokens or 0),
            )

            value = input_cost + output_cost

        except Exception as ex:
            logger.warning(
                "LiteLLM cost calculation failed for model %s: %s",
                model,
                ex,
            )
            return None

        return TokenUsageService._cost_payload(
            value
        )

    @staticmethod
    def calculate_token_cost(
        *,
        model: Any,
        input_tokens: int = 0,
        output_tokens: int = 0,
        currency: str = "USD",
        vendor: Any = None,
    ) -> dict:
        litellm_cost = (
            TokenUsageService._calculate_litellm_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

        if litellm_cost:
            return litellm_cost

        model_name = (
            TokenUsageService._normalize_model_name(
                model
            )
        )

        rates = TokenUsageService._get_model_pricing(
            vendor,
            model_name,
        )

        if not rates:
            logger.warning(
                "Token cost pricing not configured for model: %s",
                model,
            )

            return {
                "currency": currency,
                "value": "0",
            }

        input_cost = (
            int(input_tokens or 0)
            / 1_000_000
        ) * rates["input"]

        output_cost = (
            int(output_tokens or 0)
            / 1_000_000
        ) * rates["output"]

        return {
            "currency": (
                rates.get("currency")
                or currency
            ),
            "value": TokenUsageService._format_cost_value(
                input_cost + output_cost
            ),
        }

    @staticmethod
    def _required_string(value: Any) -> str:
        if value is None:
            return "NA"

        value = str(value).strip()
        return value or "NA"

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> Optional[str]:
        if value is None:
            return None

        value = str(value).strip()
        return value or None

    @staticmethod
    def _source_id(
        payload: dict,
        source_type: str,
    ) -> Optional[str]:
        sources = (
            payload.get("sourceIds")
            or payload.get("SourceId")
            or []
        )

        for source in sources:
            if not isinstance(source, dict):
                continue

            current_source_type = str(
                source.get("sourceIdType")
                or source.get("SourceIdType")
                or ""
            ).strip().lower()

            if current_source_type == source_type:
                return (
                    TokenUsageService._optional_string(
                        source.get("id")
                    )
                    or TokenUsageService._optional_string(
                        source.get("_id")
                    )
                )

        return None

    @staticmethod
    def _normalize_source_ids(
        payload: dict,
    ) -> list[dict]:
        normalized_source_ids = []

        sources = (
            payload.get("sourceIds")
            or payload.get("SourceId")
            or []
        )

        for source in sources:
            if not isinstance(source, dict):
                continue

            source_type = str(
                source.get("sourceIdType")
                or source.get("SourceIdType")
                or ""
            ).strip().lower()

            if source_type not in {
                "tender",
                "company",
            }:
                continue

            source_id = (
                TokenUsageService._optional_string(
                    source.get("id")
                )
                or TokenUsageService._optional_string(
                    source.get("_id")
                )
            )

            if not source_id:
                continue

            normalized_source_ids.append(
                {
                    "sourceIdType": source_type,
                    "id": source_id,
                }
            )

        return normalized_source_ids

    @staticmethod
    def _normalize_usage_payload(
        payload: dict,
    ) -> dict:
        normalized_payload = dict(payload)

        normalized_payload["sourceIds"] = (
            TokenUsageService._normalize_source_ids(
                payload
            )
        )

        normalized_payload.pop(
            "SourceId",
            None,
        )

        normalized_payload.pop(
            "projectId",
            None,
        )
        normalized_payload.pop(
            "ProjectId",
            None,
        )
        normalized_payload.pop(
            "project_id",
            None,
        )

        company_id = (
            TokenUsageService._optional_string(
                payload.get("companyId")
            )
            or TokenUsageService._optional_string(
                payload.get("CompanyId")
            )
            or TokenUsageService._source_id(
                payload,
                "company",
            )
        )

        tender_id_value = (
            TokenUsageService._optional_string(
                payload.get("tenderId")
            )
            or TokenUsageService._optional_string(
                payload.get("TenderId")
            )
            or TokenUsageService._source_id(
                payload,
                "tender",
            )
        )

        user_id = (
            TokenUsageService._required_string(
                payload.get("userId")
                or payload.get("UserId")
            )
        )

        tender_id = (
            TokenUsageService._required_string(
                tender_id_value
            )
        )

        if company_id:
            normalized_payload["companyId"] = (
                company_id
            )
            normalized_payload["CompanyId"] = (
                company_id
            )

        normalized_payload["userId"] = (
            user_id
        )
        normalized_payload["tenderId"] = (
            tender_id
        )
        normalized_payload["UserId"] = (
            user_id
        )
        normalized_payload["TenderId"] = (
            tender_id
        )

        return normalized_payload

    @staticmethod
    def _get_value(
        source: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        if isinstance(source, Mapping):
            return source.get(
                key,
                default,
            )

        return getattr(
            source,
            key,
            default,
        )

    @staticmethod
    def extract_token_usage(
        response: Any,
    ) -> dict:
        metadata = (
            TokenUsageService._get_value(
                response,
                "response_metadata",
                {},
            )
            or {}
        )

        usage = (
            TokenUsageService._get_value(
                response,
                "usage_metadata",
                None,
            )
            or metadata.get("token_usage")
            or metadata.get("usage")
            or {}
        )

        input_tokens = (
            TokenUsageService._get_value(
                usage,
                "input_tokens",
                None,
            )
            or TokenUsageService._get_value(
                usage,
                "prompt_tokens",
                None,
            )
            or 0
        )

        output_tokens = (
            TokenUsageService._get_value(
                usage,
                "output_tokens",
                None,
            )
            or TokenUsageService._get_value(
                usage,
                "completion_tokens",
                None,
            )
            or 0
        )

        total_tokens = (
            TokenUsageService._get_value(
                usage,
                "total_tokens",
                None,
            )
            or (
                int(input_tokens or 0)
                + int(output_tokens or 0)
            )
        )

        model = (
            metadata.get("model_name")
            or metadata.get("model")
            or TokenUsageService._get_value(
                response,
                "model_name",
                "",
            )
            or ""
        )

        cost = (
            TokenUsageService._extract_response_cost(
                response,
                metadata,
                usage,
            )
            or TokenUsageService.calculate_token_cost(
                model=model,
                input_tokens=int(
                    input_tokens or 0
                ),
                output_tokens=int(
                    output_tokens or 0
                ),
            )
        )

        result = {
            "input_tokens": int(
                input_tokens or 0
            ),
            "output_tokens": int(
                output_tokens or 0
            ),
            "total_tokens": int(
                total_tokens or 0
            ),
            "model": model,
            "cost": cost,
        }

        if TokenUsageService._debug_enabled():
            print(
                "\n========== TOKEN EXTRACTION DEBUG =========="
            )
            print(
                "Response type:",
                type(response),
            )
            print(
                "usage_metadata:",
                TokenUsageService._get_value(
                    response,
                    "usage_metadata",
                    None,
                ),
            )
            print(
                "response_metadata:",
                metadata,
            )
            print(
                "Extracted usage:",
                result,
            )
            print(
                "============================================\n"
            )

        return result

    @staticmethod
    def aggregate_token_usage(
        usage_records: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate token and cost data from multiple LLM responses."""

        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        model_totals: dict[str, dict[str, int]] = {}
        total_cost = 0.0
        cost_currencies: set[str] = set()

        for record in usage_records:
            if not isinstance(record, Mapping):
                continue

            usage = record.get("token_usage", record)
            if not isinstance(usage, Mapping):
                continue

            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", 0) or 0)

            if total_tokens <= 0:
                total_tokens = input_tokens + output_tokens

            totals["input_tokens"] += input_tokens
            totals["output_tokens"] += output_tokens
            totals["total_tokens"] += total_tokens

            model = str(usage.get("model") or "").strip()
            if model:
                model_total = model_totals.setdefault(
                    model,
                    {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                )
                model_total["input_tokens"] += input_tokens
                model_total["output_tokens"] += output_tokens
                model_total["total_tokens"] += total_tokens

            cost = usage.get("cost")
            if isinstance(cost, Mapping):
                cost_value = cost.get("value")
                currency = str(cost.get("currency") or "USD").strip()
            else:
                cost_value = cost
                currency = "USD"

            try:
                if cost_value is not None:
                    total_cost += float(cost_value)
                    cost_currencies.add(currency or "USD")
            except (TypeError, ValueError):
                continue

        result: dict[str, Any] = {
            **totals,
            "models": model_totals,
            "request_count": len(usage_records),
            "records": [dict(record) for record in usage_records],
            "cost": None,
        }

        if cost_currencies:
            result["cost"] = TokenUsageService._cost_payload(
                total_cost,
                next(iter(cost_currencies))
                if len(cost_currencies) == 1
                else "MIXED",
            )

        return result

    @staticmethod
    async def log_usage(
        payload: dict,
        bearer_token: Optional[str] = None,
    ) -> dict:
        if not TokenUsageService._is_enabled():
            result = {
                "success": False,
                "skipped": True,
                "reason": (
                    "AI_USAGE_LOG_ENABLED=false"
                ),
            }

            print(
                "[TOKEN DEBUG] Token logging skipped:",
                result,
            )

            return result

        debug_enabled = (
            TokenUsageService._debug_enabled()
        )

        if debug_enabled:
            print(
                "\n========== TOKEN USAGE DEBUG =========="
            )
            print("Destination: MongoDB TokenUsageLog")
            print(
                "Payload Before Normalize:",
                payload,
            )
            print(
                "=======================================\n"
            )

        del bearer_token  # Direct MongoDB persistence needs no HTTP token.

        normalized_payload = (
            TokenUsageService._normalize_usage_payload(
                payload
            )
        )

        if debug_enabled:
            print(
                "✅ TOKEN LOG MONGODB INSERT ABOUT TO RUN"
            )
            print(
                "Normalized Payload:",
                normalized_payload,
            )

        try:
            result = await insert_token_usage(normalized_payload)
            if debug_enabled:
                print("✅ TOKEN USAGE SAVED TO MONGODB:", result)
            return result
        except Exception as ex:
            logger.warning(
                "Token usage MongoDB insert failed: %s",
                ex,
            )
            return {
                "success": False,
                "message": "Token usage MongoDB insert failed",
                "error": str(ex),
            }

    @staticmethod
    async def log_agent_usage(
        *,
        request: Any,
        token_usage: dict,
        bearer_token: Optional[str] = None,
        correlation_id: str,
        duration_ms: int,
        application_name: str,
        agent_name: str,
        purpose: str,
        usage_type: str = "Completion",
        method: str = "LLM",
    ) -> dict:
        def request_value(
            *names: str,
        ) -> Any:
            for name in names:
                if isinstance(
                    request,
                    Mapping,
                ):
                    value = request.get(name)
                else:
                    value = getattr(
                        request,
                        name,
                        None,
                    )

                if (
                    value is not None
                    and str(value).strip()
                ):
                    return value

            return None

        tender_id = request_value(
            "tender_id",
            "tenderId",
            "TenderId",
        )

        company_id = request_value(
            "company_id",
            "companyId",
            "CompanyId",
        )

        user_id = request_value(
            "user_id",
            "userId",
            "UserId",
        )

        source_ids = []

        if tender_id:
            source_ids.append(
                {
                    "sourceIdType": "tender",
                    "id": tender_id,
                }
            )

        if company_id:
            source_ids.append(
                {
                    "sourceIdType": "company",
                    "id": company_id,
                }
            )

        resolved_application_name = (
            application_name
            or os.getenv(
                "AI_USAGE_APPLICATION_NAME",
                "",
            )
            or "NA"
        )

        payload = {
            "applicationName": (
                resolved_application_name
            ),
            "sourceIds": source_ids,
            "runId": correlation_id,
            "userId": (
                user_id
                or os.getenv(
                    "AI_USAGE_USER_ID",
                    "",
                )
            ),
            "purpose": purpose,
            "method": method,
            "agentName": agent_name,
            "usageType": usage_type,
            "inputToken": token_usage.get(
                "input_tokens",
                0,
            ),
            "outputToken": token_usage.get(
                "output_tokens",
                0,
            ),
            "totalTokens": token_usage.get(
                "total_tokens",
                0,
            ),
            "model": token_usage.get(
                "model",
                "",
            ),
            "duration": duration_ms,
            "cost": (
                token_usage.get("cost")
                or TokenUsageService.calculate_token_cost(
                    model=token_usage.get(
                        "model"
                    ),
                    input_tokens=token_usage.get(
                        "input_tokens",
                        0,
                    ),
                    output_tokens=token_usage.get(
                        "output_tokens",
                        0,
                    ),
                )
            ),
            "companyId": company_id,
            "tenderId": tender_id,
        }

        if TokenUsageService._debug_enabled():
            print(
                "\n========== AGENT USAGE PAYLOAD DEBUG =========="
            )
            print("Agent:", agent_name)
            print("Run ID:", correlation_id)
            print("Payload:", payload)
            print(
                "Bearer Token Present:",
                bool(bearer_token),
            )
            print(
                "================================================\n"
            )

        return await TokenUsageService.log_usage(
            payload=payload,
            bearer_token=bearer_token,
        )
