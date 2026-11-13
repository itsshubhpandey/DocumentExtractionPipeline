from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    override=False,
)

terminal_logger = logging.getLogger("uvicorn.error")


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class Logging:
    def __init__(
        self,
        agent_name: str,
        source_module: str,
        timeout: Optional[int] = None,
        debug: Optional[bool] = None,
    ) -> None:
        self.enabled = _parse_bool(
            os.getenv("LOG_ENABLED"),
            default=True,
        )

        self.base_url = (
            os.getenv("LOG_URL") or ""
        ).strip().strip("\"'")

        self.timeout = (
            timeout
            if timeout is not None
            else _parse_int(
                os.getenv("LOG_TIMEOUT_SECONDS"),
                10,
            )
        )

        self.agent_name = agent_name
        self.source_module = source_module

        self.debug = (
            debug
            if debug is not None
            else _parse_bool(
                os.getenv("DEBUG"),
                default=False,
            )
        )

        self._debug_logger = (
            self._build_debug_logger()
            if self.debug
            else None
        )

        print(
            "[ENTERPRISE LOG DEBUG] Logger initialized:",
            {
                "agentName": self.agent_name,
                "enabled": self.enabled,
                "urlPresent": bool(self.base_url),
                "timeoutSeconds": self.timeout,
            },
        )

        if self.enabled and not self.base_url:
            raise ValueError(
                "LOG_URL is not set while LOG_ENABLED=true."
            )

    def _build_debug_logger(self) -> logging.Logger:
        debug_logger = logging.getLogger(
            f"debug.{self.agent_name}"
        )
        debug_logger.setLevel(logging.DEBUG)

        if not any(
            isinstance(handler, logging.FileHandler)
            for handler in debug_logger.handlers
        ):
            log_dir = Path(
                os.getenv("DEBUG_LOG_DIR", "logs")
            )
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(
                log_dir / "debug.log",
                encoding="utf-8",
            )
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(name)s | %(message)s"
                )
            )
            debug_logger.addHandler(file_handler)

        return debug_logger

    def log(
        self,
        message: str,
        event_type: str,
        is_success: bool = True,
        duration_ms: int = 0,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> dict:
        correlation_id = (
            correlation_id or str(uuid.uuid4())
        )

        if not self.enabled:
            print(
                "[ENTERPRISE LOG DEBUG] SKIPPED: LOG_ENABLED=false"
            )
            return {
                "success": False,
                "skipped": True,
                "reason": "LOG_ENABLED=false",
            }

        log_payload = {
            "agentName": self.agent_name,
            "message": message,
            "eventType": event_type,
            "sourceModule": self.source_module,
            "isSuccess": is_success,
            "durationMs": duration_ms,
            "startTime": start_time,
            "endTime": end_time,
            "payloadJson": json.dumps(
                payload or {},
                default=str,
            ),
            "correlationId": correlation_id,
        }

        print("\n========== ENTERPRISE LOG DEBUG ==========")
        print("Event Type:", event_type)
        print("Correlation ID:", correlation_id)
        print("LOG URL:", self.base_url)
        print("Payload:", payload or {})
        print("==========================================\n")

        terminal_logger.info(
            "[EnterpriseLogger] submission started | event=%s | correlation_id=%s",
            event_type,
            correlation_id,
        )

        try:
            response = requests.post(
                self.base_url,
                json=log_payload,
                timeout=self.timeout,
            )

            print(
                "[ENTERPRISE LOG DEBUG] HTTP status:",
                response.status_code,
            )
            print(
                "[ENTERPRISE LOG DEBUG] HTTP body:",
                response.text[:2000],
            )

            response.raise_for_status()

            print("✅ ENTERPRISE LOG SUBMITTED SUCCESSFULLY")

            terminal_logger.info(
                "[EnterpriseLogger] submitted successfully | event=%s | status=%s | correlation_id=%s",
                event_type,
                response.status_code,
                correlation_id,
            )

            try:
                result = response.json()
            except ValueError:
                result = {"rawResponse": response.text}

            if isinstance(result, dict):
                result.setdefault("success", True)
                return result

            return {"success": True, "response": result}

        except requests.RequestException as exc:
            print(
                "❌ ENTERPRISE LOG SUBMISSION FAILED:",
                str(exc),
            )

            terminal_logger.warning(
                "[EnterpriseLogger] submission failed | event=%s | correlation_id=%s | error=%s",
                event_type,
                correlation_id,
                exc,
            )

            return {
                "success": False,
                "error": str(exc),
            }

    def start(
        self,
        message: str,
        event_type: str,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        token = {
            "message": message,
            "event_type": event_type,
            "correlation_id": (
                correlation_id or str(uuid.uuid4())
            ),
            "start_perf": time.perf_counter(),
            "start_time": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        print(
            "[ENTERPRISE LOG DEBUG] Tracking started:",
            {
                "eventType": event_type,
                "correlationId": token["correlation_id"],
            },
        )

        return token

    def end(
        self,
        tracking_token: Dict[str, Any],
        is_success: bool = True,
        payload: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> dict:
        end_perf = time.perf_counter()
        end_time = datetime.now(timezone.utc).isoformat()

        duration_ms = round(
            (
                end_perf
                - tracking_token["start_perf"]
            )
            * 1000
        )

        final_event_type = (
            event_type
            or tracking_token["event_type"]
        )

        print(
            "[ENTERPRISE LOG DEBUG] Tracking ended:",
            {
                "eventType": final_event_type,
                "correlationId": tracking_token["correlation_id"],
                "durationMs": duration_ms,
                "isSuccess": is_success,
            },
        )

        return self.log(
            message=(
                message
                or tracking_token["message"]
            ),
            event_type=final_event_type,
            is_success=is_success,
            duration_ms=duration_ms,
            start_time=tracking_token["start_time"],
            end_time=end_time,
            payload=payload,
            correlation_id=tracking_token["correlation_id"],
        )

    def debug_step(
        self,
        message: str,
        step: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        if (
            not self.debug
            or self._debug_logger is None
        ):
            return

        self._debug_logger.debug(
            json.dumps(
                {
                    "agentName": self.agent_name,
                    "sourceModule": self.source_module,
                    "step": step,
                    "message": message,
                    "correlationId": correlation_id,
                    "payload": payload or {},
                },
                default=str,
            )
        )
