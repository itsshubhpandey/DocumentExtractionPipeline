import logging
import os
from typing import Any

from litellm import cost_per_token
from app.infrastructure.database import get_database
from app.infrastructure.mongo_log_store import insert_token_usage

logger = logging.getLogger(__name__)


class TokenUsageService:
    @staticmethod
    def _normalize_model_name(model: Any) -> str:
        return str(model or "").strip().lower()

    @staticmethod
    def _detect_vendor(model_name: str) -> str:
        provider_prefix = model_name.split("/", 1)[0]
        if provider_prefix in {"openai", "mistral", "groq", "gemini", "nvidia_nim"}:
            return "nvidia" if provider_prefix == "nvidia_nim" else provider_prefix

        if model_name.startswith("mistral-"):
            return "mistral"

        return "openai"

    @staticmethod
    def _model_name_candidates(model_name: str) -> set[str]:
        candidates = {model_name}
        if "/" in model_name:
            candidates.add(model_name.split("/", 1)[1])
        return {candidate for candidate in candidates if candidate}

    @staticmethod
    def _get_model_pricing(vendor: Any, model_name: str) -> dict | None:
        vendor_name = str(vendor or TokenUsageService._detect_vendor(model_name)).strip().lower()

        try:
            config = get_database()["AIModelConfig"].find_one(
                {
                    "$or": [{"vendor": vendor_name}, {"_id": vendor_name}],
                    "isActive": True,
                }
            )
        except Exception as ex:
            logger.warning("Token cost pricing lookup failed: %s", ex)
            return None

        if not config:
            return None

        model_candidates = TokenUsageService._model_name_candidates(model_name)
        for model_config in config.get("models") or []:
            configured_model = TokenUsageService._normalize_model_name(
                model_config.get("modelName")
            )
            if (
                configured_model not in model_candidates
                or not model_config.get("isActive", False)
            ):
                continue

            return {
                "input": float(model_config.get("inputCostPer1M", 0) or 0),
                "output": float(model_config.get("outputCostPer1M", 0) or 0),
            }

        return None

    @staticmethod
    def _format_cost_value(value: float) -> str:
        return format(round(value, 8), ".8f").rstrip("0").rstrip(".") or "0"

    @staticmethod
    def _get_value(source: Any, key: str, default: Any = None) -> Any:
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    @staticmethod
    def _cost_payload(value: Any, currency: str = "USD") -> dict:
        try:
            cost_value = float(value or 0)
        except (TypeError, ValueError):
            cost_value = 0

        return {
            "currency": currency,
            "value": TokenUsageService._format_cost_value(cost_value),
        }

    @staticmethod
    def _get_float_value(source: Any, *keys: str) -> float | None:
        for key in keys:
            value = TokenUsageService._get_value(source, key, None)
            if value is None:
                continue

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _extract_response_cost(response: Any, metadata: dict, usage: Any) -> dict | None:
        cost_value = (
            TokenUsageService._get_float_value(metadata, "response_cost", "cost")
            or TokenUsageService._get_float_value(usage, "response_cost", "cost")
            or TokenUsageService._get_float_value(response, "response_cost", "cost")
        )

        if cost_value is None:
            return None

        return TokenUsageService._cost_payload(cost_value)

    @staticmethod
    def _calculate_litellm_cost(
        *,
        model: Any,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> dict | None:
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
            logger.warning("LiteLLM cost calculation failed for model %s: %s", model, ex)
            return None

        return TokenUsageService._cost_payload(value)

    @staticmethod
    def extract_token_usage(response: Any) -> dict:

        metadata = TokenUsageService._get_value(response, "response_metadata", {}) or {}

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
            or int(input_tokens or 0) + int(output_tokens or 0)
        )

        model = (
            metadata.get("model_name")
            or metadata.get("model")
            or TokenUsageService._get_value(response, "model_name", "")
        )

        cost = (
            TokenUsageService._extract_response_cost(response, metadata, usage)
            or TokenUsageService._calculate_litellm_cost(
                model=model,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
            )
        )

        return {
            "model": model,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "cost": cost,
        }

    @staticmethod
    def merge_usage(total_usage: dict, usage: dict):

        total_usage["input_tokens"] += usage["input_tokens"]
        total_usage["output_tokens"] += usage["output_tokens"]
        total_usage["total_tokens"] += usage["total_tokens"]

        model = usage["model"] or "unknown"

        if model not in total_usage["models"]:
            total_usage["models"][model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        total_usage["models"][model]["input_tokens"] += usage["input_tokens"]
        total_usage["models"][model]["output_tokens"] += usage["output_tokens"]
        total_usage["models"][model]["total_tokens"] += usage["total_tokens"]

    @staticmethod
    def update_state(state: dict, response: Any):

        usage = TokenUsageService.extract_token_usage(response)

        if "token_usage" not in state:
            state["token_usage"] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "models": {},
            }

        TokenUsageService.merge_usage(
            state["token_usage"],
            usage,
        )

        return usage

    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int, vendor: Any = None):

        if not model:
            return 0

        litellm_cost = TokenUsageService._calculate_litellm_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if litellm_cost:
            return litellm_cost["value"]

        model_name = TokenUsageService._normalize_model_name(model)
        rates = TokenUsageService._get_model_pricing(vendor, model_name)

        if not rates:
            logger.warning("Token cost pricing not configured for model: %s", model)
            return 0

        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]

        return TokenUsageService._format_cost_value(input_cost + output_cost)

    @staticmethod
    async def log_usage(
        api_url: str | None = None,
        payload: dict | None = None,
        token: str | None = None,
    ) -> dict:
        del api_url, token  # Retained in the signature for caller compatibility.
        if payload is None:
            raise ValueError("payload is required")

        try:
            return await insert_token_usage(payload)
        except Exception as ex:
            logger.warning("Token usage MongoDB insert failed: %s", ex)
            return {
                "success": False,
                "message": "Token usage MongoDB insert failed",
                "error": str(ex),
            }

    @staticmethod
    async def log_agent_usage(
        request,
        token_usage: dict,
        duration_ms: int,
        correlation_id: str,
        application_name: str,
        source_ids: list,
        purpose: str,
        method: str,
        agent_name: str,
        usage_type: str,
    ):

        total_cost = token_usage.get("cost")
        if isinstance(total_cost, dict):
            total_cost = total_cost.get("value", 0)

        if total_cost is None:
            total_cost = sum(
                float(
                    TokenUsageService.calculate_cost(
                        model=model,
                        input_tokens=model_usage["input_tokens"],
                        output_tokens=model_usage["output_tokens"],
                    )
                    or 0
                )
                for model, model_usage in token_usage.get("models", {}).items()
            )

        payload = {
            "applicationName": application_name,
            "sourceIds": [
                {
                    "sourceIdType": "tender",
                    "id": request.TenderId,
                }
            ],
            "runId": correlation_id,
            "userId": request.UserId,
            "purpose": purpose,
            "method": method,
            "agentName": agent_name,
            "usageType": usage_type,
            "inputToken": token_usage.get("input_tokens", 0),
            "outputToken": token_usage.get("output_tokens", 0),
            "totalTokens": token_usage.get("total_tokens", 0),
            "model": ",".join(token_usage.get("models", {}).keys()),
            "duration": duration_ms,
            "cost": {
                "currency": "USD",
                "value": total_cost,
            },
            "companyId": request.CompanyId,
            "tenderId": request.TenderId,
        }

        return await TokenUsageService.log_usage(
            payload=payload,
            token=getattr(request, "JwtToken", None),
        )
