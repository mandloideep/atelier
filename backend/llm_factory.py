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


class _PerQueryEmbeddings(Embeddings):
    """Adapter that fans embed_documents out to per-query calls.

    Workaround for langchain-google-genai 4.2.3, whose batched
    embed_documents against gemini-embedding-2 returns 1 result regardless
    of input count, breaking QdrantVectorStore ingestion. embed_query works
    correctly, so we just call it per chunk. Slower on cold ingest, but the
    CacheBackedEmbeddings layer in vector_store.py memoises per-text.
    """

    def __init__(self, inner: Embeddings):
        self._inner = inner
        # Surface the wrapped model name so caching/namespacing keeps working.
        self.model = getattr(inner, "model", inner.__class__.__name__)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._inner.embed_query(t) for t in texts]


def get_embeddings() -> Embeddings:
    provider = _provider("EMBED_PROVIDER", "gemini")

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # Native output is 3072 dims; honour GEMINI_EMBED_DIM so embeddings
        # match the Qdrant collection size (recommended: 768, 1536, 3072).
        # gemini-embedding-2 doesn't use task_type (the task instruction is
        # baked into the prompt text instead); only pass it for -001.
        model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2")
        kwargs: dict = {
            "model": model,
            "google_api_key": os.environ["GEMINI_API_KEY"],
            "output_dimensionality": int(os.getenv("GEMINI_EMBED_DIM", "768")),
        }
        if model.startswith("gemini-embedding-001"):
            kwargs["task_type"] = "retrieval_document"
        inner = GoogleGenerativeAIEmbeddings(**kwargs)
        # Wrap to fix batched embed_documents (broken for -2 in this version).
        if model.startswith("gemini-embedding-2"):
            return _PerQueryEmbeddings(inner)
        return inner

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


def content_to_text(content) -> str:
    """Normalise LangChain message content (str | list[dict|str]) to a plain string.

    Gemini returns answers as a list of content blocks
    (`[{'type': 'text', 'text': '...'}, ...]`); OpenAI returns strings. We
    flatten blocks to text and drop non-text parts (images, etc.).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content is not None else ""


def llm_provider_label() -> str:
    """Human-readable label used in the demo banner."""
    provider = _provider("LLM_PROVIDER", "gemini")
    if provider == "gemini":
        return os.getenv("GEMINI_LLM_MODEL", "gemini-3.1-flash-lite")
    return os.getenv("OPENAI_LLM_MODEL", "gpt-5-mini")
