import os
import certifi
from langchain_litellm import ChatLiteLLM
from app.infrastructure.environment import load_project_environment
from app.infrastructure.secrets import (
    resolve_openai_api_key as resolve_encrypted_openai_api_key,
)
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI
from pymongo import MongoClient


load_project_environment(override=True)

if os.getenv("USE_LITELLM_OPENAI_PROXY", "").strip().lower() not in {"1", "true", "yes"}:
    os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
    os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"



# def resolve_llm_provider() -> str:
#     """
#     Resolve active LLM provider from MongoDB.

#     Fallback:
#     LLM_PROVIDER environment variable.
#     """

#     try:
#         client = MongoClient(os.getenv("MONGO_URI"))
#         db=client.get_database(os.getenv("MONGO_DB_NAME", "DocAI"))

#         collection_name = os.getenv(
#          "LLM_PROVIDER_COLLECTION",
#          "LlmProvoderConfig"
#          )

#         collection = db[collection_name]

#         config = collection.find_one({
#         "isActive": True
#         })

#         if config and config.get("provider"):
#             provider = config["provider"].strip().lower()

#             print(f"LLM provider loaded from MongoDB: {provider}")

#             return provider

#     except Exception as exc:
#         print(f"Failed to load LLM provider from MongoDB: {exc}")

#     # fallback
#     provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

#     print(f"Using fallback LLM provider: {provider}")

#     return provider


def resolve_llm_config() -> dict[str, str | None]:
    try:
        client_options = {}
        if os.getenv("MONGO_TLS", "false").lower() in {
            "1", "true", "yes", "on"
        }:
            client_options["tlsCAFile"] = certifi.where()
        client = MongoClient(os.getenv("MONGO_URI"), **client_options)

        try:
            db = client.get_database(
                os.getenv("MONGO_DB_NAME", "DocAI")
            )

            collection_name = os.getenv(
                "LLM_PROVIDER_COLLECTION",
                "LlmProvoderConfig",
            )

            config = db[collection_name].find_one(
                {"isActive": True}
            )

            if config and config.get("provider"):
                return {
                    "provider": str(config["provider"]).strip().lower(),
                    "model": str(config.get("model") or "").strip() or None,
                }

        finally:
            client.close()

    except Exception as exc:
        print(f"Failed to load LLM configuration from MongoDB: {exc}")

    return {
        "provider": os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        "model": None,
    }


def resolve_llm_provider() -> str:
    """Return the active provider for existing callers."""

    return str(resolve_llm_config().get("provider") or "").strip().lower()




def resolve_openai_api_key() -> str:
    """
    Resolve OpenAI API key.

    Priority:
    1. Encrypted key stored in MongoDB
    2. OPENAI_API_KEY loaded from the project .env file
    """

    return resolve_encrypted_openai_api_key()




# def create_llm():
#     # provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
#     provider = resolve_llm_provider()

#     if provider == "mistral":
#         return ChatMistralAI(
#             model=os.getenv("MISTRAL_MODEL"),
#             api_key=os.getenv("MISTRAL_API_KEY"),
#             temperature=0.3,
#         )

#     elif provider == "openai":
#         return ChatOpenAI(
#             model=os.getenv("OPENAI_MODEL"),
#             api_key=resolve_openai_api_key(),
#             temperature=0.3,
#         )

#     elif provider == "groq":
#         return ChatGroq(
#             model=os.getenv("GROQ_MODEL"),
#             api_key=os.getenv("GROQ_API_KEY"),
#             temperature=0.3,
#         )

#     elif provider == "gemini":
#         return ChatGoogleGenerativeAI(
#             model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
#             google_api_key=os.getenv("GEMINI_API_KEY"),
#             temperature=0.3,
#         )

#     elif provider == "nvidia":
#         return ChatOpenAI(
#             model=os.getenv(
#                 "NVIDIA_MODEL",
#                 "meta/llama-3.3-70b-instruct",
#             ),
#             api_key=os.getenv("NVIDIA_API_KEY"),
#             base_url=os.getenv(
#                 "NVIDIA_BASE_URL",
#                 "https://integrate.api.nvidia.com/v1",
#             ),
#             temperature=0.3,
#         )

#     raise ValueError(f"Unsupported provider: {provider}")






def create_llm():
    llm_config = resolve_llm_config()

    provider = str(
        llm_config.get("provider") or ""
    ).strip().lower()

    configured_model = llm_config.get("model")

    if provider not in {
        "mistral",
        "openai",
        "groq",
        "gemini",
        "nvidia",
    }:
        raise ValueError(
            f"Unsupported provider: {provider}"
        )

    model_env_by_provider = {
        "mistral": "MISTRAL_MODEL",
        "openai": "OPENAI_MODEL",
        "groq": "GROQ_MODEL",
        "gemini": "GEMINI_MODEL",
        "nvidia": "NVIDIA_MODEL",
    }

    litellm_prefix_by_provider = {
        "mistral": "mistral",
        "openai": "openai",
        "groq": "groq",
        "gemini": "gemini",
        "nvidia": "nvidia_nim",
    }

    model = (
        str(configured_model or "").strip()
        or os.getenv(model_env_by_provider[provider], "").strip()
    )

    if not model:
        default_models = {
            "mistral": "mistral-large-latest",
            "gemini": "gemini-2.0-flash",
            "nvidia": "meta/llama-3.3-70b-instruct",
        }
        model = default_models.get(provider, "")

    if not model:
        raise ValueError(
            f"A model is required for provider '{provider}'"
        )

    known_prefixes = {
        "openai",
        "mistral",
        "mistralai",
        "groq",
        "gemini",
        "vertex_ai",
        "nvidia_nim",
        "azure",
        "anthropic",
        "openrouter",
        "ollama",
        "hosted_vllm",
    }

    model_prefix = model.split("/", 1)[0].strip().lower()
    if model_prefix not in known_prefixes:
        model = f"{litellm_prefix_by_provider[provider]}/{model}"

    common_kwargs = {
        "temperature": 1,
        "max_retries": 2,
    }

    if provider == "mistral":
        return ChatLiteLLM(
            model=model,
            api_key=os.getenv("MISTRAL_API_KEY"),
            **common_kwargs,
        )

    elif provider == "openai":
        os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

        return ChatLiteLLM(
            model=model,
            api_key=resolve_openai_api_key(),
            api_base="https://api.openai.com/v1",
            **common_kwargs,
        )

    elif provider == "groq":
        return ChatLiteLLM(
            model=model,
            api_key=os.getenv("GROQ_API_KEY"),
            **common_kwargs,
        )

    elif provider == "gemini":
        return ChatLiteLLM(
            model=model,
            api_key=os.getenv("GEMINI_API_KEY"),
            **common_kwargs,
        )

    elif provider == "nvidia":
        return ChatLiteLLM(
            model=model,
            api_key=os.getenv("NVIDIA_API_KEY"),
            api_base=os.getenv(
                "NVIDIA_BASE_URL",
                "https://integrate.api.nvidia.com/v1",
            ),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported provider: {provider}")
