import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    settings = get_settings()
    kwargs = {
        "model": settings.LLM_MODEL,
        "api_key": settings.LLM_API_KEY,
        "temperature": temperature,
        "max_retries": 3,
        "request_timeout": 60,
    }
    if settings.LLM_BASE_URL:
        kwargs["base_url"] = settings.LLM_BASE_URL
    return ChatOpenAI(**kwargs)


def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    kwargs = {
        "model": settings.EMBEDDING_MODEL,
        "api_key": settings.LLM_API_KEY,
    }
    if settings.LLM_BASE_URL:
        kwargs["base_url"] = settings.LLM_BASE_URL
    return OpenAIEmbeddings(**kwargs)
