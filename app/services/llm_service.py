"""
LLM Integration Service: Advanced chat using Language Models.

Features:
- OpenAI API integration (GPT-3.5-turbo, GPT-4)
- Safe prompt engineering with medical disclaimers
- RAG-aware response generation
- Streaming support preparation
- Fallback to rule-based responses if LLM unavailable
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional, Iterator

logger = logging.getLogger("mediassist.llm")

# Configuration from environment
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()  # "openai" or ""
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_ENABLED = LLM_PROVIDER == "openai" and bool(LLM_API_KEY)

SYSTEM_PROMPT = """You are MediAssist, an educational medication assistant. You help patients understand prescriptions and medications in a safe, grounded way.

CRITICAL RULES:
1. Never provide medical advice or recommendations like "you should take" or "this is safe for you"
2. Always reference the data provided - do not make up information
3. Always include appropriate disclaimers
4. Direct users to healthcare professionals for personal decisions
5. For urgent symptoms, immediately suggest emergency care
6. Keep responses concise and patient-friendly
7. Use plain language, avoid medical jargon when possible

Your role is EDUCATIONAL ONLY. You explain what information is available about a medicine, not whether it's safe for this specific person."""

MEDICAL_SAFETY_ADDENDUM = """

IMPORTANT DISCLAIMERS:
- This is NOT medical advice. I cannot make personalized medical recommendations.
- For medication decisions, always consult your pharmacist or doctor.
- If experiencing urgent symptoms (difficulty breathing, chest pain, severe confusion, fainting), seek emergency care immediately.
- Your healthcare provider knows your full medical history and can advise on your specific situation."""


class LLMService:
    """Service for LLM-powered chat responses."""

    def __init__(self, provider: str = LLM_PROVIDER, api_key: str = LLM_API_KEY):
        self.provider = provider
        self.api_key = api_key
        self.enabled = provider == "openai" and bool(api_key)
        self.client = None

        if self.enabled:
            self._init_openai_client()

    def _init_openai_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai

            openai.api_key = self.api_key
            self.client = openai
            logger.info(f"OpenAI LLM initialized with model {LLM_MODEL}")
        except ImportError:
            logger.warning("OpenAI library not installed. LLM features disabled. Install with: pip install openai")
            self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI: {e}")
            self.enabled = False

    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.enabled

    def build_context_for_llm(
        self,
        message: str,
        disease: str,
        age: int,
        matched_drugs: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build context object for LLM generation.

        Returns dict suitable for passing to LLM.
        """
        drug_context = []
        for match in matched_drugs:
            best_match = match.get("best_match", {})
            if best_match:
                sections = match.get("sections", {})
                drug_context.append({
                    "name": best_match.get("generic_name_clean") or best_match.get("generic_name"),
                    "confidence": match.get("confidence"),
                    "warnings": sections.get("warnings"),
                    "contraindications": sections.get("contraindications"),
                    "dosage": sections.get("dosage_and_administration"),
                    "indications": sections.get("indications"),
                })

        return {
            "user_message": message,
            "disease": disease,
            "age": age,
            "drug_information": drug_context,
            "conversation_turns": min(len(conversation_history), 5),  # Recent turns for context
        }

    def generate_response(
        self,
        message: str,
        disease: str,
        age: int,
        matched_drugs: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Generate an LLM response grounded in drug data.

        Returns the generated response, or None if LLM is unavailable.
        """
        if not self.is_available():
            return None

        try:
            # Build context
            context = self.build_context_for_llm(
                message, disease, age, matched_drugs, conversation_history
            )

            # Build messages for API
            messages = self._build_messages(context, conversation_history)

            # Call OpenAI API
            response = self.client.ChatCompletion.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                timeout=30,
            )

            answer = response.choices[0].message.content.strip()

            # Ensure disclaimers are included
            if "disclaimer" not in answer.lower() and "not medical advice" not in answer.lower():
                answer += MEDICAL_SAFETY_ADDENDUM

            return answer

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def generate_response_stream(
        self,
        message: str,
        disease: str,
        age: int,
        matched_drugs: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
    ) -> Iterator[str]:
        """
        Generate an LLM response as a stream of tokens.

        Yields text chunks as they arrive.
        """
        if not self.is_available():
            return

        try:
            context = self.build_context_for_llm(
                message, disease, age, matched_drugs, conversation_history
            )
            messages = self._build_messages(context, conversation_history)

            # Call OpenAI API with streaming
            response = self.client.ChatCompletion.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=True,
                timeout=30,
            )

            full_response = ""
            for chunk in response:
                delta = chunk.choices[0].delta.get("content", "")
                if delta:
                    full_response += delta
                    yield delta

            # Add disclaimer at end if not present
            if "disclaimer" not in full_response.lower():
                yield MEDICAL_SAFETY_ADDENDUM

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield f"\n\n[Error: Could not generate response. {str(e)}]"

    def _build_messages(
        self, context: Dict[str, Any], conversation_history: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Build messages array for OpenAI API."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add recent conversation history
        for turn in conversation_history[-3:]:  # Last 3 turns
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("text", ""),
            })

        # Build user message with drug context
        drug_section = self._format_drug_context(context.get("drug_information", []))
        user_content = f"""Patient context: {context.get("disease", "unknown disease")}, age {context.get("age", "?")}

Drug information available:
{drug_section}

Question: {context.get("user_message", "")}

Provide an educational explanation based on the drug information above. Remember: this is not medical advice."""

        messages.append({"role": "user", "content": user_content})

        return messages

    def _format_drug_context(self, drugs: List[Dict[str, Any]]) -> str:
        """Format drug information for the LLM prompt."""
        if not drugs:
            return "No specific drug information matched."

        parts = []
        for drug in drugs:
            name = drug.get("name", "Unknown")
            confidence = int(drug.get("confidence", 0) * 100)
            parts.append(f"\n{name} (matched with {confidence}% confidence):")

            if drug.get("warnings"):
                parts.append(f"  Warnings: {drug['warnings'][:200]}...")
            if drug.get("contraindications"):
                parts.append(f"  Contraindications: {drug['contraindications'][:200]}...")
            if drug.get("dosage"):
                parts.append(f"  Dosage: {drug['dosage'][:200]}...")

        return "\n".join(parts)


# Global instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(provider=LLM_PROVIDER, api_key=LLM_API_KEY)
    return _llm_service


def is_llm_available() -> bool:
    """Check if LLM service is available."""
    service = get_llm_service()
    return service.is_available()


def generate_llm_response(
    message: str,
    disease: str,
    age: int,
    matched_drugs: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
) -> Optional[str]:
    """Generate a response using LLM."""
    service = get_llm_service()
    return service.generate_response(message, disease, age, matched_drugs, conversation_history)
