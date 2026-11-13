"""Reusable asynchronous RabbitMQ publisher for tender processing jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from app.infrastructure.environment import load_project_environment


load_project_environment(override=True)

logger = logging.getLogger("rabbitmq.service")


class RabbitMQConsumerHealthError(RuntimeError):
    """Raised when the registered RabbitMQ consumer is no longer healthy."""


def rabbitmq_enabled() -> bool:
    """Return whether asynchronous RabbitMQ processing is enabled."""

    return os.getenv("USING_RABBITMQ", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class RabbitMQService:
    """Manage one reusable RabbitMQ connection and publish job messages."""

    def __init__(
        self,
        *,
        url: str | None = None,
        exchange_name: str | None = None,
        queue_name: str | None = None,
        routing_key: str | None = None,
    ) -> None:
        self.url = url or os.getenv(
            "RABBITMQ_URL",
            "amqp://guest:guest@localhost:5672/",
        )
        self.exchange_name = exchange_name or os.getenv(
            "RABBITMQ_EXCHANGE",
            "tender.events",
        )
        self.queue_name = queue_name or os.getenv(
            "RABBITMQ_QUEUE",
            "tender.processing",
        )
        self.routing_key = routing_key or os.getenv(
            "RABBITMQ_ROUTING_KEY",
            "tender.processing.requested",
        )
        self.dead_letter_exchange_name = os.getenv(
            "RABBITMQ_DEAD_LETTER_EXCHANGE",
            "tender.events.dead",
        )
        self.dead_letter_queue_name = os.getenv(
            "RABBITMQ_DEAD_LETTER_QUEUE",
            "tender.processing.dead",
        )
        self.dead_letter_routing_key = os.getenv(
            "RABBITMQ_DEAD_LETTER_ROUTING_KEY",
            "tender.processing.dead",
        )
        self.retry_exchange_name = os.getenv(
            "RABBITMQ_RETRY_EXCHANGE",
            "tender.events.retry",
        )
        self.retry_queue_name = os.getenv(
            "RABBITMQ_RETRY_QUEUE",
            "tender.processing.retry",
        )
        self.retry_routing_key = os.getenv(
            "RABBITMQ_RETRY_ROUTING_KEY",
            "tender.processing.retry",
        )
        self.retry_delay_ms = max(
            1000,
            int(os.getenv("RABBITMQ_RETRY_DELAY_MS", "30000")),
        )
        self.connection_timeout = float(
            os.getenv("RABBITMQ_CONNECTION_TIMEOUT_SECONDS", "10")
        )
        self.heartbeat_seconds = int(
            os.getenv("RABBITMQ_HEARTBEAT_SECONDS", "60")
        )
        self.reconnect_interval = float(
            os.getenv("RABBITMQ_RECONNECT_INTERVAL_SECONDS", "5")
        )
        self.healthcheck_interval = max(
            1.0,
            float(os.getenv("RABBITMQ_HEALTHCHECK_INTERVAL_SECONDS", "30")),
        )
        self.healthcheck_timeout = max(
            1.0,
            float(os.getenv("RABBITMQ_HEALTHCHECK_TIMEOUT_SECONDS", "10")),
        )

        self._connection: Any = None
        self._channel: Any = None
        self._exchange: Any = None
        self._dead_letter_exchange: Any = None
        self._retry_exchange: Any = None
        self._queue: Any = None
        self._consumer_tag: str | None = None
        self._connection_lock = asyncio.Lock()

    async def _on_connection_reconnected(self, _connection: Any) -> None:
        logger.warning(
            "RabbitMQ connection restored | queue=%s",
            self.queue_name,
        )

    async def _on_channel_reopened(self, _channel: Any) -> None:
        logger.warning(
            "RabbitMQ channel restored; robust consumers re-registered | queue=%s",
            self.queue_name,
        )

    async def connect(self) -> None:
        """Connect and declare the durable publishing topology."""

        import aio_pika
        from aio_pika import ExchangeType

        if (
            self._connection is not None
            and not self._connection.is_closed
            and self._channel is not None
            and not self._channel.is_closed
            and self._exchange is not None
            and self._dead_letter_exchange is not None
            and self._retry_exchange is not None
            and self._queue is not None
        ):
            return

        async with self._connection_lock:
            if (
                self._connection is not None
                and not self._connection.is_closed
                and self._channel is not None
                and not self._channel.is_closed
                and self._exchange is not None
                and self._dead_letter_exchange is not None
                and self._retry_exchange is not None
                and self._queue is not None
            ):
                return

            await self.close()
            logger.info(
                "Connecting to RabbitMQ | exchange=%s | queue=%s | routing_key=%s",
                self.exchange_name,
                self.queue_name,
                self.routing_key,
            )
            self._connection = await aio_pika.connect_robust(
                self.url,
                timeout=self.connection_timeout,
                heartbeat=self.heartbeat_seconds,
                reconnect_interval=self.reconnect_interval,
            )
            self._connection.reconnect_callbacks.add(
                self._on_connection_reconnected
            )
            self._channel = await self._connection.channel(
                publisher_confirms=True,
            )
            self._channel.reopen_callbacks.add(self._on_channel_reopened)
            self._exchange = await self._channel.declare_exchange(
                self.exchange_name,
                ExchangeType.TOPIC,
                durable=True,
            )
            self._dead_letter_exchange = await self._channel.declare_exchange(
                self.dead_letter_exchange_name,
                ExchangeType.TOPIC,
                durable=True,
            )
            dead_letter_queue = await self._channel.declare_queue(
                self.dead_letter_queue_name,
                durable=True,
            )
            await dead_letter_queue.bind(
                self._dead_letter_exchange,
                routing_key=self.dead_letter_routing_key,
            )
            self._retry_exchange = await self._channel.declare_exchange(
                self.retry_exchange_name,
                ExchangeType.TOPIC,
                durable=True,
            )
            retry_queue = await self._channel.declare_queue(
                self.retry_queue_name,
                durable=True,
                arguments={
                    "x-message-ttl": self.retry_delay_ms,
                    "x-dead-letter-exchange": self.exchange_name,
                    "x-dead-letter-routing-key": self.routing_key,
                },
            )
            await retry_queue.bind(
                self._retry_exchange,
                routing_key=self.retry_routing_key,
            )
            queue = await self._channel.declare_queue(
                self.queue_name,
                durable=True,
            )
            self._queue = queue
            await queue.bind(
                self._exchange,
                routing_key=self.routing_key,
            )
            logger.info(
                "RabbitMQ topology ready | exchange=%s | queue=%s",
                self.exchange_name,
                self.queue_name,
            )

    async def get_queue_health(self) -> dict[str, Any]:
        """Return broker-reported queue depth and consumer count."""

        if self._connection is None or self._connection.is_closed:
            raise RabbitMQConsumerHealthError("RabbitMQ connection is closed")
        if self._channel is None or self._channel.is_closed:
            raise RabbitMQConsumerHealthError("RabbitMQ channel is closed")
        if self._queue is None:
            raise RabbitMQConsumerHealthError("RabbitMQ queue is unavailable")

        try:
            declaration = await asyncio.wait_for(
                self._queue.declare(timeout=self.healthcheck_timeout),
                timeout=self.healthcheck_timeout + 1,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RabbitMQConsumerHealthError(
                f"RabbitMQ queue health check failed: {exc}"
            ) from exc

        return {
            "queue": self.queue_name,
            "ready_messages": int(declaration.message_count or 0),
            "consumer_count": int(declaration.consumer_count or 0),
            "connected": True,
        }

    async def publish(
        self,
        payload: Mapping[str, Any],
        *,
        routing_key: str | None = None,
    ) -> str:
        """Publish one JSON job message and return its message id."""

        from aio_pika import DeliveryMode, Message

        await self.connect()

        message_id = str(
            payload.get("event_id")
            or payload.get("job_id")
            or uuid4()
        )
        body = json.dumps(dict(payload), default=str).encode("utf-8")
        message = Message(
            body,
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=message_id,
            correlation_id=str(
                payload.get("run_id")
                or payload.get("job_id")
                or message_id
            ),
            type=str(payload.get("event_type") or "tender.processing.requested"),
            timestamp=datetime.now(timezone.utc),
        )

        await self._exchange.publish(
            message,
            routing_key=routing_key or self.routing_key,
        )
        return message_id

    async def publish_retry(self, payload: Mapping[str, Any]) -> str:
        """Publish a message to the delayed retry queue."""

        from aio_pika import DeliveryMode, Message

        await self.connect()
        retry_payload = dict(payload)
        retry_payload["retry_count"] = int(
            retry_payload.get("retry_count") or 0
        ) + 1
        message_id = str(
            retry_payload.get("event_id")
            or retry_payload.get("run_id")
            or uuid4()
        )
        message = Message(
            json.dumps(retry_payload, default=str).encode("utf-8"),
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=message_id,
            correlation_id=str(retry_payload.get("run_id") or message_id),
            type="tender.processing.retry",
            timestamp=datetime.now(timezone.utc),
        )
        await self._retry_exchange.publish(
            message,
            routing_key=self.retry_routing_key,
        )
        return message_id

    async def publish_dead_letter(
        self,
        payload: Mapping[str, Any],
        error: BaseException,
    ) -> str:
        """Persist a failed message without storing its bearer token."""

        from aio_pika import DeliveryMode, Message

        await self.connect()
        dead_payload = dict(payload)
        dead_payload.pop("bearer_token", None)
        dead_payload["dead_letter_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        message_id = str(
            dead_payload.get("event_id")
            or dead_payload.get("run_id")
            or uuid4()
        )
        message = Message(
            json.dumps(dead_payload, default=str).encode("utf-8"),
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=message_id,
            correlation_id=str(dead_payload.get("run_id") or message_id),
            type="tender.processing.failed",
            timestamp=datetime.now(timezone.utc),
        )
        await self._dead_letter_exchange.publish(
            message,
            routing_key=self.dead_letter_routing_key,
        )
        return message_id

    async def consume(
        self,
        callback: Any,
        *,
        prefetch_count: int = 1,
    ) -> None:
        """Consume messages from the configured queue until shutdown."""

        await self.connect()
        await self._channel.set_qos(prefetch_count=prefetch_count)
        self._consumer_tag = await self._queue.consume(
            callback,
            no_ack=False,
            robust=True,
        )
        logger.info(
            "RabbitMQ consumer registered | queue=%s | consumer_tag=%s | prefetch=%s",
            self.queue_name,
            self._consumer_tag,
            prefetch_count,
        )

        try:
            while True:
                await asyncio.sleep(self.healthcheck_interval)
                health = await self.get_queue_health()
                ready_messages = health["ready_messages"]
                consumer_count = health["consumer_count"]

                if consumer_count < 1:
                    raise RabbitMQConsumerHealthError(
                        f"No active consumers registered for queue '{self.queue_name}'"
                    )

                log_method = logger.warning if ready_messages else logger.info
                log_method(
                    "RabbitMQ consumer healthy | queue=%s | ready=%s | consumers=%s",
                    self.queue_name,
                    ready_messages,
                    consumer_count,
                )
        finally:
            consumer_tag = self._consumer_tag
            self._consumer_tag = None
            if (
                consumer_tag
                and self._queue is not None
                and self._channel is not None
                and not self._channel.is_closed
            ):
                try:
                    await self._queue.cancel(consumer_tag)
                except Exception:
                    logger.warning(
                        "Unable to cancel RabbitMQ consumer cleanly | queue=%s",
                        self.queue_name,
                        exc_info=True,
                    )

    async def close(self) -> None:
        """Close the channel and connection if they are open."""

        channel = self._channel
        connection = self._connection
        self._consumer_tag = None
        self._channel = None
        self._exchange = None
        self._dead_letter_exchange = None
        self._retry_exchange = None
        self._queue = None
        self._connection = None

        if channel is not None and not channel.is_closed:
            try:
                await channel.close()
            except Exception:
                logger.warning("Unable to close RabbitMQ channel", exc_info=True)
        if connection is not None and not connection.is_closed:
            try:
                await connection.close()
            except Exception:
                logger.warning("Unable to close RabbitMQ connection", exc_info=True)


_rabbitmq_service: RabbitMQService | None = None


def get_rabbitmq_service() -> RabbitMQService:
    """Return the process-wide RabbitMQ service instance."""

    global _rabbitmq_service
    if _rabbitmq_service is None:
        _rabbitmq_service = RabbitMQService()
    return _rabbitmq_service


async def close_rabbitmq_service() -> None:
    """Close the process-wide RabbitMQ service during application shutdown."""

    global _rabbitmq_service
    if _rabbitmq_service is not None:
        await _rabbitmq_service.close()
        _rabbitmq_service = None


__all__ = [
    "RabbitMQConsumerHealthError",
    "RabbitMQService",
    "close_rabbitmq_service",
    "get_rabbitmq_service",
    "rabbitmq_enabled",
]
