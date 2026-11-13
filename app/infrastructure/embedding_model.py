"""Shared embedding-model configuration and construction."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_openai import OpenAIEmbeddings

from app.infrastructure.database import get_database
from app.infrastructure.secrets import resolve_openai_api_key


class EmbeddingModelService:
    """Load the active embedding model configuration and build its client."""

    DEFAULT_COLLECTION_NAME = "EmbeddingModel"

    def __init__(
        self,
        embedding_model: Any | None = None,
        *,
        database: Any | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        model_config: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Configure the service.

        ``database``, ``model_config`` and ``embedding_model`` are injectable
        so callers can test this service without opening MongoDB or calling an
        embedding provider. Production callers normally provide none of them.
        """

        self.database = database
        self.collection_name = self.DEFAULT_COLLECTION_NAME
        self._embedding_model = embedding_model
        self._model_config = (
            dict(model_config)
            if model_config is not None
            else None
        )
        self._api_key = api_key
        self._model_name = str(model_name or "").strip()

    @property
    def model_config(self) -> dict[str, Any]:
        """Return the active model configuration from MongoDB."""

        if self._model_config is None:
            database = self.database or get_database()
            document = database[self.collection_name].find_one(
                {"is_active": True}
            )

            if not document:
                raise ValueError(
                    "No active embedding model was found in the "
                    f"'{self.collection_name}' collection."
                )

            self._model_config = dict(document)

        return dict(self._model_config)

    def get_model_config(self) -> dict[str, Any]:
        """Return the active embedding configuration."""

        return self.model_config

    @property
    def model_name(self) -> str:
        """Return the configured provider model name."""

        if not self._model_name:
            self._model_name = str(
                self.model_config.get("model") or ""
            ).strip()

        if not self._model_name:
            raise ValueError(
                "The active embedding model configuration has no 'model'."
            )

        return self._model_name

    @property
    def embedding_model(self) -> Any:
        """Return a lazily constructed embedding provider."""

        if self._embedding_model is None:
            self._embedding_model = self._create_embedding_model()

        return self._embedding_model

    def get_embedding_model(self) -> Any:
        """Return the configured embedding provider."""

        return self.embedding_model

    def _create_embedding_model(self) -> OpenAIEmbeddings:
        config = self.model_config
        provider = str(config.get("provider") or "").strip().lower()

        if provider != "openai":
            raise ValueError(
                "Unsupported embedding provider "
                f"'{provider or 'unknown'}'. Only 'openai' is supported."
            )

        embedding_arguments: dict[str, Any] = {
            "model": self.model_name,
            "api_key": self._resolve_api_key(config) or None,
        }

        base_url = config.get("base_url")
        if base_url:
            embedding_arguments["base_url"] = str(base_url).strip()

        dimensions = config.get("dimensions")
        if dimensions is not None:
            try:
                resolved_dimensions = int(dimensions)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "The active embedding model 'dimensions' must be an integer."
                ) from exc

            if resolved_dimensions <= 0:
                raise ValueError(
                    "The active embedding model 'dimensions' must be greater than zero."
                )

            embedding_arguments["dimensions"] = resolved_dimensions

        return OpenAIEmbeddings(**embedding_arguments)

    def _resolve_api_key(
        self,
        model_config: Mapping[str, Any] | None = None,
    ) -> str:
        """Resolve a configured key, then the shared encrypted-key resolver."""

        if self._api_key is not None:
            return str(self._api_key).strip()

        config = model_config or self.model_config
        api_key_env = str(config.get("api_key_env") or "").strip()

        if api_key_env:
            configured_api_key = os.getenv(api_key_env, "").strip()
            if configured_api_key:
                return configured_api_key

        return str(resolve_openai_api_key() or "").strip()

    def embed_query(self, text: str) -> list[float]:
        """Create one embedding for a non-empty search query."""

        query = str(text or "").strip()
        if not query:
            raise ValueError("Embedding text cannot be empty.")

        embedding = self.embedding_model.embed_query(query)
        if not embedding:
            raise RuntimeError("The embedding model returned an empty vector.")

        return [float(value) for value in embedding]

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Create embeddings for multiple search queries in one batch."""

        queries = [str(text or "").strip() for text in texts]
        if any(not query for query in queries):
            raise ValueError("Embedding text cannot be empty.")

        if not queries:
            return []

        embed_documents = getattr(
            self.embedding_model,
            "embed_documents",
            None,
        )
        if callable(embed_documents):
            embeddings = embed_documents(queries)
        else:
            embeddings = [
                self.embedding_model.embed_query(query)
                for query in queries
            ]

        normalised_embeddings = [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]
        if any(not embedding for embedding in normalised_embeddings):
            raise RuntimeError(
                "The embedding model returned an empty vector."
            )

        return normalised_embeddings


__all__ = ["EmbeddingModelService"]
