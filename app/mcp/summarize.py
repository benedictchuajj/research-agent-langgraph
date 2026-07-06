import logging
from typing import Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

_llm: Optional[BaseChatModel] = None


def _get_llm() -> BaseChatModel:
    global _llm
    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "ollama":
        logger.info("Using Ollama: model=%s base_url=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
        _llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set (LLM_PROVIDER=openai)")
        logger.info("Using OpenAI: model=%s", OPENAI_MODEL)
        _llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER} (expected 'openai' or 'ollama')")

    return _llm


def summarize_paper(title: str, abstract: str) -> str:
    llm = _get_llm()

    prompt = f"""You are a research assistant. Summarize the following academic paper in 3-5 sentences.
Focus on: the main contribution, methodology, key findings, and significance.

Title: {title}

Abstract: {abstract}

Provide a clear, concise summary:"""

    response = llm.invoke(prompt)
    return response.content.strip()
