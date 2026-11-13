import os
from types import SimpleNamespace

from bson import ObjectId  # Added: Needed for ObjectId() mapping

from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI
from app.infrastructure.environment import load_project_environment
# from app.agents.DocExtractionpipeline.services.mongo import _get_database
from app.agents.DocExtractionpipeline.services.mongo.mongo_client import (
    get_document,
)
from app.utils.decrept import decrypt


load_project_environment(override=True)


class OpenAIEmbeddingSDKAdapter:
    def __init__(self, *, model: str, api_key: str, base_url: str | None = None):
        self.model = model
        resolved_base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL_OVERRIDE")
            or "https://api.openai.com/v1"
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=resolved_base_url.rstrip("/"),
        )
        self.last_usage = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            self.last_usage = SimpleNamespace(
                prompt_tokens=0,
                total_tokens=0,
            )
            return []

        max_items = max(
            1,
            int(os.getenv("EMBEDDING_BATCH_MAX_ITEMS", "100")),
        )
        max_characters = max(
            1000,
            int(os.getenv("EMBEDDING_BATCH_MAX_CHARACTERS", "200000")),
        )

        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_characters = 0
        for text in texts:
            text_characters = len(text)
            if current_batch and (
                len(current_batch) >= max_items
                or current_characters + text_characters > max_characters
            ):
                batches.append(current_batch)
                current_batch = []
                current_characters = 0

            current_batch.append(text)
            current_characters += text_characters

        if current_batch:
            batches.append(current_batch)

        vectors: list[list[float]] = []
        prompt_tokens = 0
        total_tokens = 0
        for batch in batches:
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            vectors.extend(item.embedding for item in response.data)
            usage = getattr(response, "usage", None)
            prompt_tokens += int(
                getattr(usage, "prompt_tokens", 0) or 0
            )
            total_tokens += int(
                getattr(usage, "total_tokens", 0) or 0
            )

        self.last_usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class EmbeddingModel:

    @staticmethod
    async def get_embedding_model():
        """Returns the configured embedding model from MongoDB."""

        config = await get_document(
            collection_name="EmbeddingModel",
            filter_query={"is_active": True},
        )

        if not config:
            raise Exception("No active embedding model found.")

        provider = config["provider"].lower()

        if provider in {"openai", "litellm"}:
            if provider == "litellm":
                api_key = os.getenv("LITELLM_API_KEY")
                base_url = os.getenv("LITELLM_BASE_URL")
                if not api_key:
                    raise RuntimeError(
                        "LITELLM_API_KEY is required for a LiteLLM embedding model"
                    )
                if not base_url:
                    raise RuntimeError(
                        "LITELLM_BASE_URL is required for a LiteLLM embedding model"
                    )
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                base_url = None

            if not api_key and provider == "openai":
                security_doc = await get_document(
                    collection_name="Security",
                    filter_query={
                        "_id": ObjectId("6a3944f958430082848fc63d")
                    },
                )

                if not security_doc:
                    raise Exception("Security document not found.")

                encrypted_key = security_doc.get("Security")

                if not encrypted_key:
                    raise Exception("Security field not found.")

                api_key = decrypt(encrypted_key)

            # Old LangChain implementation:
            # return OpenAIEmbeddings(
            #     model=config["model"],
            #     api_key=api_key,
            # )

            return OpenAIEmbeddingSDKAdapter(
                model=config["model"],
                api_key=api_key,
                base_url=base_url,
            )

        elif provider == "mistral":
            return MistralAIEmbeddings(
                model=config["model"],
                api_key=os.getenv(config["api_key_env"]),
            )

        elif provider == "ollama":
            return OllamaEmbeddings(
                model=config["model"],
                base_url=config.get("base_url", "http://localhost:11434"),
            )

        raise ValueError(f"Unsupported provider: {provider}")
