"""Provider-agnostic factories for chat LLMs and embeddings.

Provider is selected at import time via env vars:

    LLM_PROVIDER   = gemini | openai      (default: gemini)
    EMBED_PROVIDER = gemini | openai      (default: gemini)

Each provider reads its own model + key vars. See .env.example.
"""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def _provider(var: str, default: str) -> str:
    return (os.getenv(var) or default).strip().lower()


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    provider = _provider("LLM_PROVIDER", "gemini")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_LLM_MODEL", "gemini-3.1-flash-lite"),
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-mini"),
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def get_embeddings() -> Embeddings:
    provider = _provider("EMBED_PROVIDER", "gemini")

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
            google_api_key=os.environ["GEMINI_API_KEY"],
            task_type="retrieval_document",
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        )

    raise ValueError(f"Unknown EMBED_PROVIDER: {provider!r}")


def get_embed_dim() -> int:
    provider = _provider("EMBED_PROVIDER", "gemini")

    if provider == "gemini":
        return int(os.getenv("GEMINI_EMBED_DIM", "768"))

    if provider == "openai":
        # text-embedding-3-small default
        return int(os.getenv("OPENAI_EMBED_DIM", "1536"))

    raise ValueError(f"Unknown EMBED_PROVIDER: {provider!r}")


def llm_provider_label() -> str:
    """Human-readable label used in the demo banner."""
    provider = _provider("LLM_PROVIDER", "gemini")
    if provider == "gemini":
        return os.getenv("GEMINI_LLM_MODEL", "gemini-3.1-flash-lite")
    return os.getenv("OPENAI_LLM_MODEL", "gpt-5-mini")
