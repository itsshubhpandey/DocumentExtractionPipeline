import json
import os
import logging
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from app.infrastructure.mongo_log_store import insert_token_usage

try:
    from litellm import cost_per_token as litellm_cost_per_token
except ImportError:  # The dependency is installed with the LiteLLM integration.
    litellm_cost_per_token = None

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
logger = logging.getLogger(__name__)


DEFAULT_MODEL_PRICING_USD_PER_1M = {
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    # OpenAI chat models. Prices are USD per 1M input/output tokens.
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # OpenAI embedding models. Embeddings charge input tokens only.
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.00},
    # Mistral common hosted models.
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-small-latest": {"input": 0.10, "output": 0.30},
    "mistral-embed": {"input": 0.10, "output": 0.00},
}


class TokenUsageService:
    @staticmethod
    def _normalize_model_name(model: Any) -> str:
        return str(model or "").strip().lower()

    @staticmethod
    def _get_model_pricing() -> dict:
        pricing = dict(DEFAULT_MODEL_PRICING_USD_PER_1M)
        custom_pricing = os.getenv("AI_MODEL_PRICING_USD_PER_1M")

        if not custom_pricing:
            return pricing

        try:
            for model, rates in json.loads(custom_pricing).items():
                model_name = TokenUsageService._normalize_model_name(model)
                pricing[model_name] = {
                    "input": float(rates.get("input", 0) or 0),
                    "output": float(rates.get("output", 0) or 0),
                }
        except (TypeError, ValueError, json.JSONDecodeError) as ex:
            logger.warning("Invalid AI_MODEL_PRICING_USD_PER_1M: %s", ex)

        return pricing

    @staticmethod
    def _calculate_litellm_cost(
        *,
        model: Any,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> Optional[dict]:
        if litellm_cost_per_token is None or not str(model or "").strip():
            return None

        try:
            input_cost, output_cost = litellm_cost_per_token(
                model=str(model),
                prompt_tokens=int(input_tokens or 0),
                completion_tokens=int(output_tokens or 0),
            )
        except Exception:
            # Fall back to the project's configured pricing table when the
            # installed LiteLLM catalog does not know the proxy alias.
            return None

        return {
            "currency": "USD",
            "value": round(float(input_cost or 0) + float(output_cost or 0), 8),
        }

    @staticmethod
    def calculate_token_cost(
        *,
        model: Any,
        input_tokens: int = 0,
        output_tokens: int = 0,
        currency: str = "USD",
    ) -> dict:
        litellm_cost = TokenUsageService._calculate_litellm_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if litellm_cost is not None:
            return litellm_cost

        model_name = TokenUsageService._normalize_model_name(model)
        model_candidates = [model_name]
        if "/" in model_name:
            model_candidates.append(model_name.split("/", 1)[1])

        pricing = TokenUsageService._get_model_pricing()
        rates = next(
            (pricing.get(candidate) for candidate in model_candidates if pricing.get(candidate)),
            None,
        )

        if not rates:
            logger.warning("Token cost pricing not configured for model: %s", model)
            return {
                "currency": currency,
                "value": 0,
            }

        input_cost = (int(input_tokens or 0) / 1_000_000) * rates["input"]
        output_cost = (int(output_tokens or 0) / 1_000_000) * rates["output"]

        return {
            "currency": currency,
            "value": round(input_cost + output_cost, 8),
        }

    @staticmethod
    def _required_string(value: Any) -> str:
        if value is None:
            return "NA"

        value = str(value).strip()
        return value or "NA"

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None

        value = str(value).strip()
        return value or None

    @staticmethod
    def _source_id(payload: dict, source_type: str) -> Optional[str]:
        for source in (
            payload.get("sourceIds")
            or payload.get("SourceId")
            or []
        ):
            if not isinstance(source, dict):
                continue

            current_source_type = str(
                source.get("sourceIdType")
                or source.get("SourceIdType")
                or ""
            ).strip().lower()

            if current_source_type == source_type:
                return (
                    TokenUsageService._optional_string(source.get("id"))
                    or TokenUsageService._optional_string(source.get("_id"))
                )

        return None

    @staticmethod
    def _normalize_source_ids(payload: dict) -> list[dict]:
        """Keep only one company ID and one tender ID in sourceIds."""

        normalized_source_ids = []
        seen_source_types = set()
        sources = payload.get("sourceIds") or payload.get("SourceId") or []

        for source in sources:
            if not isinstance(source, dict):
                continue

            source_type = str(
                source.get("sourceIdType")
                or source.get("SourceIdType")
                or ""
            ).strip().lower()

            if source_type not in {"company", "tender"}:
                continue
            if source_type in seen_source_types:
                continue

            source_id = (
                TokenUsageService._optional_string(source.get("id"))
                or TokenUsageService._optional_string(source.get("_id"))
            )
            if not source_id:
                continue

            normalized_source_ids.append(
                {
                    "sourceIdType": source_type,
                    "id": source_id,
                }
            )
            seen_source_types.add(source_type)

        return normalized_source_ids

    @staticmethod
    def _normalize_usage_payload(payload: dict) -> dict:
        normalized_payload = dict(payload)
        normalized_payload["sourceIds"] = TokenUsageService._normalize_source_ids(
            payload
        )
        normalized_payload.pop("SourceId", None)

        # ProjectId is metadata, not a source identifier. Remove all naming
        # variants because the usage API maps ID fields into SourceId entries.
        normalized_payload.pop("projectId", None)
        normalized_payload.pop("ProjectId", None)
        normalized_payload.pop("project_id", None)

        company_id = (
            TokenUsageService._optional_string(payload.get("companyId"))
            or TokenUsageService._optional_string(payload.get("CompanyId"))
            or TokenUsageService._source_id(payload, "company")
        )
        tender_id_value = (
            TokenUsageService._optional_string(payload.get("tenderId"))
            or TokenUsageService._optional_string(payload.get("TenderId"))
            or TokenUsageService._source_id(payload, "tender")
        )

        user_id = TokenUsageService._required_string(
            payload.get("userId") or payload.get("UserId")
        )
        tender_id = TokenUsageService._required_string(tender_id_value)

        if company_id:
            normalized_payload["companyId"] = company_id
            normalized_payload["CompanyId"] = company_id

        normalized_payload["userId"] = user_id
        normalized_payload["tenderId"] = tender_id
        normalized_payload["UserId"] = user_id
        normalized_payload["TenderId"] = tender_id

        return normalized_payload

    @staticmethod
    def _get_value(source: Any, key: str, default: Any = None) -> Any:
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    @staticmethod
    def extract_token_usage(response: Any) -> dict:
        metadata = (
            TokenUsageService._get_value(response, "response_metadata", {}) or {}
        )

        usage = (
            TokenUsageService._get_value(response, "usage_metadata", None)
            or metadata.get("token_usage")
            or metadata.get("usage")
            or {}
        )

        input_tokens = (
            TokenUsageService._get_value(usage, "input_tokens", None)
            or TokenUsageService._get_value(usage, "prompt_tokens", None)
            or 0
        )

        output_tokens = (
            TokenUsageService._get_value(usage, "output_tokens", None)
            or TokenUsageService._get_value(usage, "completion_tokens", None)
            or 0
        )

        total_tokens = (
            TokenUsageService._get_value(usage, "total_tokens", None)
            or input_tokens + output_tokens
        )

        model = (
            metadata.get("model_name")
            or metadata.get("model")
            or TokenUsageService._get_value(response, "model_name", "")
            or ""
        )

        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "model": model,
        }

    @staticmethod
    async def log_usage(payload: dict, bearer_token: Optional[str] = None):
        del bearer_token  # Authentication is not needed for direct MongoDB inserts.
        normalized_payload = TokenUsageService._normalize_usage_payload(payload)

        try:
            return await insert_token_usage(normalized_payload)
        except Exception as ex:
            logger.warning("Token usage MongoDB insert failed: %s", ex)
            return {
                "success": False,
                "message": "Token usage MongoDB insert failed",
                "error": str(ex),
            }
                
 
 
 
######################################## Token Uasge ########################################
 
                
# usage = TokenUsageService.extract_token_usage(response)

# await TokenUsageService.log_usage(
#     payload=usage,
#     bearer_token=token,
#
