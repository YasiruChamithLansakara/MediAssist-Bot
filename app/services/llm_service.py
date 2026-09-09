"""
LLM Integration Service — MediAssist
=====================================
Provides natural-language answers grounded in the drug knowledge base.

Primary backend: Groq API  →  openai/gpt-oss-20b  (FREE, fast, 131k context)
Alternative providers: Together AI, OpenAI, HuggingFace — each needs its OWN key.

Model history (important)
-------------------------
`llama-3.1-8b-instant` was decommissioned by Groq on 2026-08-16. Requests to it
now fail, and because the old code swallowed every generation error and returned
None, the whole app silently degraded to rule-based answers with no visible
signal. Two things guard against a repeat:

  1. MODEL ROTATION — if the configured model is rejected as unknown or
     decommissioned, the client rotates to the next model in GROQ_MODEL_CHAIN
     and retries once, logging the switch at WARNING.
  2. VISIBLE STATUS — `LLMService.status()` reports the active model and the
     last error, and is surfaced through /api/meta and /api/dashboard, so a
     dead model shows up in the UI instead of hiding.

Environment variables
---------------------
LLM_PROVIDER      groq | together | openai | huggingface   (default: groq)
LLM_API_KEY       your API key for the chosen provider
LLM_MODEL         override the default model name
LLM_MODEL_CHAIN   comma-separated Groq models to try, in order
LLM_TEMPERATURE   sampling temperature (default 0.3 — factual, low creativity)
LLM_MAX_TOKENS    max response tokens (default 800)
LLM_ALWAYS_ON     1 = call LLM even when no drugs matched (default 0)

Getting a FREE Groq key
-----------------------
1. Sign up at https://console.groq.com  (no credit card required)
2. Create an API key
3. Add to .env:   LLM_PROVIDER=groq   LLM_API_KEY=gsk_...
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger("mediassist.llm")

# ── FAISS / RAG imports (optional) ──────────────────────────────────────────
try:
    from app.ml.faiss_store import get_faiss_store
except ImportError:
    get_faiss_store = None

try:
    from .rag_service import get_rag_service, is_rag_available
except ImportError:
    get_rag_service = None
    is_rag_available = lambda: False

# ── ENV ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "groq").strip().lower()
LLM_API_KEY     = os.getenv("LLM_API_KEY", "").strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "800"))

# Default model per provider
_DEFAULT_MODELS: Dict[str, str] = {
    "groq":        "openai/gpt-oss-20b",     # FREE — replaces decommissioned llama-3.1-8b-instant
    "together":    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai":      "gpt-4o-mini",
    "huggingface": "meta-llama/Llama-3.3-70B-Instruct",
}

# Models retired by Groq. Configuring one of these is a hard error at startup
# rather than a 400 on the first live request.
_RETIRED_GROQ_MODELS = {
    "llama-3.1-8b-instant",      # decommissioned 2026-08-16
    "llama-3.1-70b-versatile",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
}

# Tried in order when the active model is rejected. Verified live on GroqCloud.
GROQ_MODEL_CHAIN: List[str] = [
    m.strip()
    for m in os.getenv(
        "LLM_MODEL_CHAIN",
        "openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3.6-27b",
    ).split(",")
    if m.strip()
]

_configured_model = os.getenv("LLM_MODEL", "").strip()
if _configured_model in _RETIRED_GROQ_MODELS:
    logger.error(
        "LLM_MODEL='%s' was decommissioned by the provider and no longer serves "
        "requests. Falling back to '%s'. Update LLM_MODEL in your .env.",
        _configured_model, _DEFAULT_MODELS["groq"],
    )
    _configured_model = ""

LLM_MODEL = _configured_model or _DEFAULT_MODELS.get(LLM_PROVIDER, "openai/gpt-oss-20b")

# ── PROMPTS ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MediAssist, an educational AI assistant that helps chronic disease \
patients understand their prescriptions and medications.

STRICT RULES:
1. EDUCATIONAL ONLY — never give personal medical advice or make clinical decisions.
2. Base answers ONLY on the drug information provided in the context below.
3. Always end with a disclaimer to consult a doctor or pharmacist.
4. EMERGENCY: if the user mentions chest pain, difficulty breathing, severe bleeding, or \
loss of consciousness — immediately tell them to call emergency services and stop.
5. Use plain, patient-friendly language. Avoid jargon; if you must use a medical term, \
explain it in brackets.
6. Adapt tone for age: simpler sentences for patients 65+, more detail for younger adults.
7. NEVER say "you should take X mg", "this is safe for you", or prescribe doses.
8. If drug data is missing or thin, say "I don't have enough data on this" rather than guessing.
9. Keep responses concise — 3 to 5 short bullet points is ideal.
10. OFF-TOPIC REFUSAL (most important): If the user's question is NOT about medications, \
drugs, prescriptions, side effects, dosage, or health — respond with EXACTLY this and nothing else: \
"I can only assist with medication and prescription questions. Please ask about a specific \
medicine — its dosage, side effects, warnings, or interactions." \
Do NOT discuss sports, travel, food, entertainment, relationships, or any other non-medical topic. \
Do NOT try to help with the off-topic request while also mentioning medications.

RESPONSE FORMAT:
**[Drug name]**
• What it's for: …
• Key warning(s): …
• Common side effects: …
• ⚕️ Always confirm with your doctor or pharmacist before making any changes."""

MEDICAL_DISCLAIMER = (
    "\n\n⚕️ *This is educational information only — not medical advice. "
    "Always consult your doctor or pharmacist before making any medication decisions.*"
)

# Keywords that indicate the LLM already included a disclaimer (avoid doubling up)
_DISCLAIMER_SIGNALS = (
    "not medical advice", "educational", "consult your doctor",
    "consult your pharmacist", "healthcare provider", "healthcare professional",
    "speak to your", "talk to your", "always verify",
)

# Intent-specific focus instructions injected into each user turn
_INTENT_FOCUS: Dict[str, str] = {
    "dosage": (
        "Focus on: how and when to take this medication, the usual dose range, "
        "whether to take it with food, and what to do if a dose is missed. "
        "Do NOT specify a personalised dose — that is for the prescriber."
    ),
    "side_effects": (
        "Focus on: common side effects patients should expect, serious adverse "
        "reactions that need urgent attention, and which side effects usually resolve on their own."
    ),
    "safety": (
        "Focus on: important warnings, which patients should NOT take this drug "
        "(contraindications), and key precautions for the patient's disease context."
    ),
    "interaction": (
        "Focus on: known drug interactions, foods or substances to avoid while taking "
        "this medication, and signs of a dangerous interaction."
    ),
    "general": (
        "Give a balanced overview: what the drug treats, the most important warning, "
        "and the most common side effect."
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# BACKEND CLIENTS
# ════════════════════════════════════════════════════════════════════════════

class _GroqClient:
    """
    Groq API — Llama 3.1 8B Instant (FREE tier).
    OpenAI-compatible endpoint.
    """
    name = "groq"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._ready = False
        self.last_error: Optional[str] = None
        # Models still worth trying if `model` is rejected, in order.
        self._chain: List[str] = [model] + [m for m in GROQ_MODEL_CHAIN if m != model]
        self._init()

    def _init(self):
        if not self.api_key:
            logger.warning("LLM_API_KEY not set — Groq LLM disabled")
            self.last_error = "no_api_key"
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
            self._ready = True
            logger.info("LLM: Groq/%s ready", self.model)
        except ImportError:
            # Groq SDK not installed, fall back to requests
            try:
                import requests  # noqa: F401
                self._ready = True
                logger.info("LLM: Groq/%s ready (requests fallback)", self.model)
            except ImportError:
                logger.warning("Groq SDK and requests not installed")
        except Exception as exc:
            logger.warning("Groq init failed: %s", exc)

    @property
    def ready(self) -> bool:
        return self._ready

    @staticmethod
    def _is_model_rejected(exc: Exception) -> bool:
        """
        True when the provider rejected the *model* rather than the request.

        Groq answers a retired or unknown model with HTTP 404 and a body
        containing `model_not_found` / "does not exist" / "decommissioned".
        Those are worth retrying on a different model; a 401 or a rate limit
        is not.
        """
        text = str(exc).lower()
        markers = (
            "model_not_found", "does not exist", "decommissioned",
            "deprecated", "no longer supported", "unknown model",
        )
        return any(m in text for m in markers)

    def _call_once(self, model: str, messages: List[Dict[str, str]]) -> str:
        """One completion call against a specific model. Raises on failure."""
        if self._client is not None:
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()

        # Fallback: raw requests (OpenAI-compatible endpoint)
        import requests
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            },
            timeout=30,
        )
        if resp.status_code >= 400:
            # Surface the body — it carries the model_not_found marker.
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        return resp.json()["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """
        Rate limits, timeouts and 5xx are worth retrying; a bad key is not.

        Two evaluation runs came back with empty answers even though retrieval
        had succeeded — a transient provider failure with no retry, which the
        caller then saw as "the LLM produced nothing".
        """
        text = str(exc).lower()
        markers = (
            "429", "rate limit", "rate_limit", "too many requests",
            "500", "502", "503", "504", "overloaded", "capacity",
            "timeout", "timed out", "connection", "temporarily",
        )
        return any(m in text for m in markers)

    def _call_with_retry(self, model: str, messages: List[Dict[str, str]]) -> str:
        """One model, up to three attempts, exponential backoff."""
        import time

        last: Optional[Exception] = None
        for attempt in range(3):
            try:
                answer = self._call_once(model, messages)
                # An empty completion is a failure, not an answer — returning
                # it verbatim shows the user a blank reply.
                if answer and answer.strip():
                    return answer
                last = RuntimeError("empty completion")
            except Exception as exc:
                last = exc
                if not self._is_transient(exc):
                    raise
            if attempt < 2:
                delay = 0.6 * (2 ** attempt)
                logger.warning(
                    "Groq call failed on '%s' (%s) — retrying in %.1fs",
                    model, last, delay,
                )
                time.sleep(delay)
        raise last or RuntimeError("generation failed")

    def generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if not self._ready:
            return None

        for model in self._chain:
            try:
                answer = self._call_with_retry(model, messages)
                if model != self.model:
                    logger.warning(
                        "Groq model '%s' was rejected; now using '%s'. "
                        "Update LLM_MODEL in your .env to make this permanent.",
                        self.model, model,
                    )
                    self.model = model          # stick with what works
                self.last_error = None
                return answer
            except Exception as exc:
                if self._is_model_rejected(exc) and model != self._chain[-1]:
                    logger.warning("Groq model '%s' rejected (%s) — trying next", model, exc)
                    continue
                logger.error("Groq generation failed on '%s': %s", model, exc)
                self.last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                return None

        self.last_error = "all_models_rejected"
        return None


class _TogetherClient:
    """Together AI — Llama 3.1 8B Instruct (affordable)."""
    name = "together"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._ready = bool(api_key)
        if self._ready:
            logger.info("LLM: Together AI/%s configured", self.model)

    @property
    def ready(self) -> bool:
        return self._ready

    def generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if not self._ready:
            return None
        try:
            import requests, json  # noqa: E401
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            }
            resp = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error("Together AI generation failed: %s", exc)
            return None


class _OpenAIClient:
    """OpenAI GPT (any model)."""
    name = "openai"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._ready = False
        self._init()

    def _init(self):
        if not self.api_key:
            return
        try:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
            self._ready = True
            logger.info("LLM: OpenAI/%s ready", self.model)
        except ImportError:
            logger.warning("openai package not installed")
        except Exception as exc:
            logger.warning("OpenAI LLM init failed: %s", exc)

    @property
    def ready(self) -> bool:
        return self._ready

    def generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if not self._ready or not self._client:
            return None
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                timeout=30,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("OpenAI generation failed: %s", exc)
            return None


class _HuggingFaceClient:
    """HuggingFace Inference API."""
    name = "huggingface"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._ready = bool(api_key)
        if self._ready:
            logger.info("LLM: HuggingFace/%s configured", self.model)

    @property
    def ready(self) -> bool:
        return self._ready

    def generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if not self._ready:
            return None
        try:
            import requests  # noqa: F401
            prompt = "\n\n".join(m.get("content", "") for m in messages if m.get("content"))
            url = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": min(LLM_MAX_TOKENS, 512)},
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return str(data[0].get("generated_text", "")).strip()
            return str(data).strip()
        except Exception as exc:
            logger.error("HuggingFace generation failed: %s", exc)
            return None


# ════════════════════════════════════════════════════════════════════════════
# LLM SERVICE
# ════════════════════════════════════════════════════════════════════════════

def _side_effects_text(best: Dict[str, Any], limit: int = 400) -> str:
    """
    Pull side effects out of a lookup match.

    `build_match()` returns them nested under `side_effects_buckets`
    (common / less_common / rare / postmarketing / unknown), but this module
    used to read a flat `common_side_effects` key that no match has ever
    carried. The result: the model was told nothing about side effects even
    when the dataset held them, and answered "I don't have enough data" to
    the single most common question patients ask.
    """
    buckets = best.get("side_effects_buckets") or {}
    if not isinstance(buckets, dict):
        buckets = {}

    parts: List[str] = []
    for label in ("common", "less_common", "rare"):
        value = str(buckets.get(label) or "").strip()
        if value:
            parts.append(f"{label.replace('_', ' ')}: {value}")

    # Fall back to whatever flat columns exist on the row.
    if not parts:
        for column in ("common_side_effects", "top_all_side_effects", "side_effects_all"):
            value = str(best.get(column) or "").strip()
            if value:
                parts.append(value)
                break

    text = " | ".join(parts)
    return text[:limit] if text else ""


class LLMService:
    """
    Orchestrates context building + response generation.
    Uses whatever backend is configured and available.
    """

    def __init__(self, provider: str = LLM_PROVIDER, api_key: str = LLM_API_KEY):
        self.provider = provider
        self._backend = self._build_backend(provider, api_key)

    # ── backend factory ──────────────────────────────────────────────────

    @staticmethod
    def _build_backend(provider: str, api_key: str):
        model = LLM_MODEL
        mapping = {
            "groq":        _GroqClient,
            "together":    _TogetherClient,
            "openai":      _OpenAIClient,
            "huggingface": _HuggingFaceClient,
        }
        cls = mapping.get(provider)
        if cls:
            b = cls(api_key, model)
            if b.ready:
                return b
            logger.warning("Provider '%s' configured but not ready", provider)
        else:
            logger.warning("Unknown LLM_PROVIDER='%s'", provider)

        # Cross-provider fallback, but ONLY with a key that belongs to that
        # provider. The previous version replayed LLM_API_KEY against every
        # other vendor's endpoint, which sent a Groq key to Together, OpenAI
        # and HuggingFace on each request — a credential leak that also could
        # never have authenticated.
        for name, fallback_cls in mapping.items():
            if name == provider:
                continue
            own_key = os.getenv(f"{name.upper()}_API_KEY", "").strip()
            if not own_key:
                continue
            b = fallback_cls(own_key, _DEFAULT_MODELS[name])
            if b.ready:
                logger.info("LLM: fell back to %s (using %s_API_KEY)", name, name.upper())
                return b

        logger.warning(
            "No LLM provider available. Chat will use rule-based responses only.\n"
            "  → Set LLM_PROVIDER=groq and LLM_API_KEY=<your_groq_key> in .env\n"
            "  → Free key: https://console.groq.com"
        )
        return None

    # ── availability ─────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._backend is not None and self._backend.ready

    def status(self) -> Dict[str, Any]:
        """
        Machine-readable LLM health, surfaced via /api/meta and /api/dashboard.

        `last_error` is the reason the most recent generation failed. Without
        it a decommissioned model looks identical to "no drugs matched" from
        the outside — which is exactly how the llama-3.1 retirement went
        unnoticed for three weeks.
        """
        backend = self._backend
        return {
            "available":  self.is_available(),
            "provider":   self.provider,
            "model":      getattr(backend, "model", None),
            "configured_model": LLM_MODEL,
            "model_chain": GROQ_MODEL_CHAIN if self.provider == "groq" else [],
            "last_error": getattr(backend, "last_error", None),
        }

    # ── context builder ──────────────────────────────────────────────────

    def build_context_for_llm(
        self,
        message: str,
        disease: str,
        age: int,
        matched_drugs: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble all available context for the LLM prompt."""
        drug_context: List[Dict[str, Any]] = []

        for match in matched_drugs:
            best = match.get("best_match") or {}
            sections = match.get("sections") or {}
            name = (
                best.get("generic_name_clean")
                or best.get("generic_name")
                or ""
            )
            if not name:
                continue
            drug_context.append({
                "name":              name,
                "confidence":        match.get("confidence"),
                "drug_class":        best.get("drug_class"),
                "warnings":          sections.get("warnings") or best.get("warnings"),
                "contraindications": sections.get("contraindications") or best.get("contraindications"),
                "dosage":            sections.get("dosage_and_administration") or best.get("dosage_and_administration"),
                "indications":       sections.get("indications") or best.get("indications"),
                "common_side_effects": _side_effects_text(best),
                # Flagged identity: the prompt must say so rather than assert.
                "needs_confirmation": bool(match.get("needs_confirmation")),
                "source": "matched",
            })

        # FAISS semantic enrichment
        rag_context: List[Dict[str, Any]] = []
        query = f"{message} {disease}".strip()
        matched_names_lower = {d["name"].lower() for d in drug_context}

        faiss_store = None
        if get_faiss_store is not None:
            try:
                faiss_store = get_faiss_store()
            except Exception:
                pass

        if faiss_store is not None and faiss_store.is_ready():
            try:
                for r in faiss_store.search(query, top_k=3):
                    if r["drug_name"].lower() not in matched_names_lower:
                        rag_context.append({
                            "name":       r["drug_name"],
                            "similarity": r["similarity"],
                            "metadata":   r["metadata"],
                            "source":     "faiss",
                        })
            except Exception as exc:
                logger.debug("FAISS enrichment error: %s", exc)
        elif is_rag_available and is_rag_available():
            try:
                rag_svc = get_rag_service()
                if rag_svc and rag_svc.is_available():
                    for r in rag_svc.retrieve_context(query, top_k=3):
                        if r["drug_name"].lower() not in matched_names_lower:
                            rag_context.append({
                                "name":       r["drug_name"],
                                "similarity": r["similarity"],
                                "metadata":   r["metadata"],
                                "source":     "rag",
                            })
            except Exception as exc:
                logger.debug("RAG enrichment error: %s", exc)

        return {
            "user_message":     message,
            "disease":          disease,
            "age":              age,
            "drug_information": drug_context,
            "rag_context":      rag_context,
            "has_rag":          bool(rag_context),
            "conversation_turns": min(len(conversation_history), 5),
        }

    # ── message builder ──────────────────────────────────────────────────

    def _build_messages(
        self,
        context: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Last 3 conversation turns (text only — no embedded data objects)
        for turn in conversation_history[-3:]:
            role = turn.get("role", "user")
            text = str(turn.get("text", "")).strip()
            if text:
                messages.append({"role": role, "content": text})

        drug_section = self._format_drug_context(
            context.get("drug_information", []),
            context.get("rag_context", []),
        )

        intent = context.get("intent", "general")
        focus  = _INTENT_FOCUS.get(intent, _INTENT_FOCUS["general"])
        age    = context.get("age", "?")
        age_note = " (use simpler language)" if isinstance(age, int) and age >= 65 else ""

        user_content = (
            f"Patient: {context.get('disease', 'unknown condition')} patient, age {age}{age_note}\n\n"
            f"Drug information from database:\n{drug_section}\n\n"
            f"Question: {context.get('user_message', '')}\n\n"
            f"Focus: {focus}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def _format_drug_context(
        drugs: List[Dict[str, Any]],
        rag_context: List[Dict[str, Any]] = None,
    ) -> str:
        parts: List[str] = []

        if drugs:
            parts.append("=== MATCHED DRUGS FROM DATABASE ===")
            for drug in drugs:
                name = drug.get("name", "Unknown")
                conf = int((drug.get("confidence") or 0) * 100)
                parts.append(f"\n► {name} (match confidence: {conf}%)")
                if drug.get("needs_confirmation"):
                    parts.append(
                        "  NOTE: this drug was NOT matched exactly. Open your answer by "
                        "telling the patient to confirm this is the medicine on their "
                        "prescription before relying on anything below."
                    )
                if drug.get("drug_class"):
                    parts.append(f"  Class: {drug['drug_class']}")
                if drug.get("indications"):
                    parts.append(f"  Used for: {str(drug['indications'])[:250]}")
                if drug.get("warnings"):
                    parts.append(f"  Warnings: {str(drug['warnings'])[:250]}")
                if drug.get("contraindications"):
                    parts.append(f"  Contraindications: {str(drug['contraindications'])[:200]}")
                if drug.get("dosage"):
                    parts.append(f"  Dosage guidance: {str(drug['dosage'])[:200]}")
                if drug.get("common_side_effects"):
                    parts.append(f"  Common side effects: {str(drug['common_side_effects'])[:200]}")

        if rag_context:
            parts.append("\n=== SEMANTICALLY RELATED DRUGS ===")
            for r in rag_context:
                meta = r.get("metadata", {})
                parts.append(f"\n• {r.get('name', 'Unknown')} (similarity: {r.get('similarity', 0):.0%})")
                if meta.get("indications"):
                    parts.append(f"  Used for: {str(meta['indications'])[:150]}")
                if meta.get("warnings"):
                    parts.append(f"  Warnings: {str(meta['warnings'])[:150]}")

        if not parts:
            parts.append("No specific drug information available for this query.")

        return "\n".join(parts)

    # ── main generate ────────────────────────────────────────────────────

    def generate_response(
        self,
        message: str,
        disease: str,
        age: int,
        matched_drugs: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        intent: str = "general",
    ) -> Optional[str]:
        """
        Generate an LLM response grounded in drug data.
        Returns the response string, or None if LLM unavailable.
        """
        if not self.is_available():
            return None

        try:
            context = self.build_context_for_llm(
                message, disease, age, matched_drugs, conversation_history
            )
            context["intent"] = intent  # pass detected intent for focused prompting

            messages = self._build_messages(context, conversation_history)
            answer   = self._backend.generate(messages)

            if answer:
                # Safety net: append disclaimer only if LLM omitted it entirely
                answer_lower = answer.lower()
                if not any(sig in answer_lower for sig in _DISCLAIMER_SIGNALS):
                    answer += MEDICAL_DISCLAIMER
            return answer

        except Exception as exc:
            logger.error("LLM generate_response failed: %s", exc)
            return None

    def generate_response_stream(
        self,
        message: str,
        disease: str,
        age: int,
        matched_drugs: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        intent: str = "general",
    ) -> Iterator[str]:
        """
        Streaming response generator.
        Falls back to a single-shot response if streaming not supported.
        """
        answer = self.generate_response(
            message, disease, age, matched_drugs, conversation_history, intent=intent
        )
        if answer:
            yield answer
        else:
            yield "[LLM unavailable — rule-based response shown above]"


# ════════════════════════════════════════════════════════════════════════════
# SINGLETON + MODULE-LEVEL HELPERS
# ════════════════════════════════════════════════════════════════════════════

_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(provider=LLM_PROVIDER, api_key=LLM_API_KEY)
    return _llm_service


def is_llm_available() -> bool:
    return get_llm_service().is_available()


def get_llm_status() -> Dict[str, Any]:
    """Active provider/model plus the last generation error, for /api/meta."""
    return get_llm_service().status()


def generate_llm_response(
    message: str,
    disease: str,
    age: int,
    matched_drugs: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    intent: str = "general",
) -> Optional[str]:
    return get_llm_service().generate_response(
        message, disease, age, matched_drugs, conversation_history, intent=intent
    )


def get_rag_status() -> Dict[str, Any]:
    try:
        if is_rag_available and is_rag_available():
            rag_svc = get_rag_service()
            if rag_svc:
                return rag_svc.get_status()
    except Exception as exc:
        logger.debug("Error getting RAG status: %s", exc)
    return {
        "rag_enabled":  False,
        "provider":     None,
        "dimension":    None,
        "vector_count": 0,
    }
