import os
import json
import logging

from dotenv import load_dotenv
from typing import Any, Dict, Optional

from app.infrastructure.mongo_log_store import insert_enterprise_log

load_dotenv()


class AgentLogger:

    def __init__(self):
        self.logger = logging.getLogger("agent_logger")

    def log_event(
        self,
        agent_name: str,
        message: str,
        event_type: str,
        source_module: str,
        is_success: bool,
        duration_ms: int = 0,
        payload: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:

        log_payload = {
            "agentName": agent_name,
            "message": message,
            "eventType": event_type,
            "sourceModule": source_module,
            "isSuccess": is_success,
            "durationMs": duration_ms,
            "payloadJson": json.dumps(payload or {}),
            "payload": payload or {},
            "correlationId": correlation_id or "",
        }

        try:
            insert_enterprise_log(log_payload)
        except Exception as ex:
            self.logger.error(
                "Failed to save enterprise log event: %s",
                ex,
            )
