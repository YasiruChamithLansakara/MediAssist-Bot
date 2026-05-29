from __future__ import annotations

import re
from typing import Any, Dict, List

from app.services.drug_lookup import (
    SUPPORTED_DISEASES,
    build_context_highlights,
    lookup_drug,
    normalize_text,
)

from app.services.ner_service import extract_medication_entities
from app.services.safety_service import (
    MedicalSafetyGuard,
    detect_emergency_symptoms,
    get_safety_notice,
)

# =========================================
# FAISS SEMANTIC SEARCH
# =========================================
try:
    from app.ml.faiss_store import get_faiss_store
    _FAISS_IMPORT_OK = True
except Exception:
    get_faiss_store = None  # type: ignore[assignment]
    _FAISS_IMPORT_OK = False

SAFETY_NOTICE = get_safety_notice()


# =====================================================
# UTILITIES
# =====================================================
def _unique_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()

    for x in items:
        key = normalize_text(str(x or ""))
        if key and key not in seen:
            seen.add(key)
            out.append(x)

    return out


# =====================================================
# INTENT DETECTION
# =====================================================
def _intent(message: str) -> str:
    msg = message.lower()

    if any(w in msg for w in ["dose", "dosage", "how much"]):
        return "dosage"
    if any(w in msg for w in ["side effect", "reaction"]):
        return "side_effects"
    if any(w in msg for w in ["safe", "warning", "contraindication"]):
        return "safety"
    if any(w in msg for w in ["interaction", "combine"]):
        return "interaction"

    return "general"


# =====================================================
# FAISS CONTEXT BUILDER
# =====================================================
def _get_rag_context(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve semantically similar drugs from the FAISS index."""
    if not _FAISS_IMPORT_OK or get_faiss_store is None:
        return []
    try:
        store = get_faiss_store()
        if not store.is_ready():
            return []
        return store.search(query, top_k=top_k)
    except Exception:
        return []


# =====================================================
# MAIN CHAT ENGINE
# =====================================================
def build_chat_response(
    *,
    message: str,
    disease: str,
    age: int,
    drugs: List[str],
    request_id: str = "",
) -> Dict[str, Any]:

    # -------------------------------------------------
    # 1. SAFETY CHECK
    # -------------------------------------------------
    is_emergency, emergency_symptoms = detect_emergency_symptoms(message)

    safety_check = MedicalSafetyGuard.validate_user_intent(
        message, disease, age
    )

    # -------------------------------------------------
    # 2. NER EXTRACTION (SCI SPACY)
    # -------------------------------------------------
    detected_entities = extract_medication_entities(
        text=message,
        disease=disease,
        age=age,
        max_entities=5,
    )

    detected_names = [
        e.get("text", "") for e in detected_entities if e.get("text")
    ]

    # -------------------------------------------------
    # 3. COMBINE INPUT SOURCES
    # -------------------------------------------------
    explicit_drugs = _unique_keep_order(drugs)
    queries = _unique_keep_order(explicit_drugs + detected_names)

    intent = _intent(message)

    matched = []
    citations = []

    # -------------------------------------------------
    # 4. DRUG LOOKUP + FAISS ENRICHMENT
    # -------------------------------------------------
    for query in queries[:5]:

        # --- Structured DB lookup
        result = lookup_drug(
            query,
            disease=disease,
            age=age,
            top_k=1,
        )

        best = result.get("best_match")

        record = {
            "query": query,
            "confidence": result.get("confidence"),
            "best_match": best,
        }

        if best:
            record["sections"] = best

            citations.append(
                {
                    "drug": best.get("generic_name_clean"),
                    "source": best.get("sources"),
                }
            )

        # --- FAISS semantic enrichment (NEW)
        rag_context = _get_rag_context(query, top_k=2)
        if rag_context:
            record["rag_context"] = rag_context

        matched.append(record)

    # -------------------------------------------------
    # 5. BUILD CONTEXT
    # -------------------------------------------------
    context = build_context_highlights(
        disease=disease,
        age=age,
        match=matched[0].get("best_match") if matched else None,
    )
    context["disease"] = disease
    context["age"] = age

    # -------------------------------------------------
    # 6. EMERGENCY OVERRIDE
    # -------------------------------------------------
    if is_emergency:
        answer = MedicalSafetyGuard.build_emergency_warning(
            emergency_symptoms
        )
    else:
        answer = _format_answer(
            message,
            intent,
            matched,
            context,
        )
        answer = MedicalSafetyGuard.enhance_answer_with_safety(
            answer, disease, age
        )

    # -------------------------------------------------
    # 7. RESPONSE
    # -------------------------------------------------
    return {
        "message": message,
        "intent": intent,
        "answer": answer,
        "detected_entities": detected_entities,
        "matched_drugs": matched,
        "citations": citations,
        "context": context,
        "is_emergency": is_emergency,
        "safety": safety_check,
        "safety_notice": SAFETY_NOTICE,
        "request_id": request_id,
    }


# =====================================================
# RESPONSE FORMATTER
# =====================================================
def _format_answer(message, intent, matched, context):

    lines = [
        SAFETY_NOTICE,
        f"Question: {message}",
        "",
    ]

    if not matched:
        lines.append(
            "No matching medication found. Please specify a drug name."
        )
        return "\n".join(lines)

    for m in matched:

        drug = m.get("best_match", {})
        name = drug.get("generic_name_clean", "Unknown")

        lines.append(f"Drug: {name}")

        if intent in ["safety", "general"]:
            if drug.get("warnings"):
                lines.append(f"- Warnings: {drug['warnings'][:200]}")

        if intent == "dosage":
            if drug.get("dosage_and_administration"):
                lines.append(
                    f"- Dosage: {drug['dosage_and_administration'][:200]}"
                )

        # FAISS enrichment display
        if m.get("rag_context"):
            lines.append("- Related drugs (semantic search):")
            for r in m["rag_context"]:
                lines.append(f"  • {r.get('drug_name', 'unknown')}")

        lines.append("")

    return "\n".join(lines)