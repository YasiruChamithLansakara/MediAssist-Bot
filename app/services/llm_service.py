from __future__ import annotations

import os
from typing import Any, Dict, List


class LLMService:
    """
    Optional LLM integration facade.

    The project can run without an LLM API key. This service reports
    availability and provides a stable import surface for the API layer.
    """

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        self.model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.api_key_configured = bool(os.getenv("LLM_API_KEY", "").strip())
        self.enabled = self.provider == "openai" and self.api_key_configured

    def is_available(self) -> bool:
        return self.enabled

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider or None,
            "model": self.model if self.enabled else None,
            "api_key_configured": self.api_key_configured,
        }

    def generate_response(
        self,
        *,
        message: str,
        disease: str,
        age: int,
        matched_drugs: List[Dict[str, Any]] | None = None,
        conversation_history: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "answer": None,
            "reason": "LLM generation is disabled unless LLM_PROVIDER=openai and LLM_API_KEY is set.",
            "context": {"message": message, "disease": disease, "age": age},
            "matched_drugs_count": len(matched_drugs or []),
            "conversation_turns": len(conversation_history or []),
        }


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def get_rag_status() -> Dict[str, Any]:
    return {
        "enabled": False,
        "reason": "RAG availability is managed by app.state.rag_service during FastAPI startup.",
    }
