from __future__ import annotations

import re
from typing import Any, Dict, List

from app.services.drug_lookup import SUPPORTED_DISEASES, build_context_highlights, lookup_drug, normalize_text
from app.services.medication_ner_enhanced import extract_medication_entities_enhanced as extract_medication_entities
from app.services.medication_ner import entity_names
from app.services.medical_safety import (
    MedicalSafetyGuard,
    detect_emergency_symptoms,
    get_safety_notice,
)


# Use the enhanced safety notice
SAFETY_NOTICE = get_safety_notice()


def _unique_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for x in items:
        raw = str(x or "").strip()
        key = normalize_text(raw)
        if not raw or not key or key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def _first_sentences(text: str, *, max_chars: int = 420, max_sentences: int = 2) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return ""

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean) if p.strip()]
    summary = " ".join(parts[:max_sentences]).strip() if parts else clean
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip() + "..."
    return summary


def _intent_from_message(message: str) -> str:
    msg = (message or "").lower()
    if any(k in msg for k in ["dose", "dosage", "how much", "frequency", "times a day", "take"]):
        return "dosage"
    if any(k in msg for k in ["side effect", "adverse", "reaction"]):
        return "side_effects"
    if any(k in msg for k in ["contraindication", "avoid", "unsafe", "safe", "risk", "warning"]):
        return "safety"
    if any(k in msg for k in ["interact", "interaction", "together", "combine"]):
        return "interaction"
    return "general"


def _sections_from_match(match: Dict[str, Any]) -> Dict[str, Any]:
    side_effects = match.get("side_effects_buckets") or {}
    return {
        "warnings": _first_sentences(match.get("warnings") or ""),
        "contraindications": _first_sentences(match.get("contraindications") or ""),
        "dosage_and_administration": _first_sentences(match.get("dosage_and_administration") or ""),
        "indications": _first_sentences(match.get("indications") or ""),
        "side_effects": side_effects,
    }


def _format_context_highlights(highlights: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for key, items in (highlights.get("highlights") or {}).items():
        if not isinstance(items, list):
            continue
        for item in items:
            if str(item).strip():
                lines.append(f"{key}: {item}")
    return lines


def _build_answer(
    *,
    message: str,
    disease: str,
    age: int,
    intent: str,
    matched: List[Dict[str, Any]],
    highlights: Dict[str, Any],
) -> str:
    lines: List[str] = [
        f"Context used: {disease}, age {age}.",
        SAFETY_NOTICE,
        "",
    ]

    if not matched:
        lines.extend(
            [
                "I could not detect a medicine name with enough confidence.",
                "Please include a generic or brand name, for example: Is diclofenac safe for hypertension?",
            ]
        )
        return "\n".join(lines).strip()

    lines.append(f"Question: {message.strip()}")
    lines.append("")

    for item in matched:
        match = item.get("best_match") or {}
        name = match.get("generic_name_clean") or match.get("generic_name") or item.get("query")
        sections = item.get("sections") or {}
        lines.append(f"Medicine: {name} (confidence {round(float(item.get('confidence') or 0) * 100)}%)")

        if intent in {"safety", "general", "interaction"}:
            if sections.get("warnings"):
                lines.append(f"- Warnings: {sections['warnings']}")
            if sections.get("contraindications"):
                lines.append(f"- Contraindications: {sections['contraindications']}")

        if intent in {"dosage", "general"} and sections.get("dosage_and_administration"):
            lines.append(f"- Dosage information from label: {sections['dosage_and_administration']}")

        if intent in {"side_effects", "general"}:
            side_effects = sections.get("side_effects") or {}
            common = side_effects.get("common") or side_effects.get("unknown")
            if common:
                lines.append(f"- Side effects noted in dataset: {_first_sentences(common, max_chars=280, max_sentences=1)}")

        if not any(
            sections.get(key)
            for key in ["warnings", "contraindications", "dosage_and_administration", "indications"]
        ):
            lines.append("- The local dataset has limited label text for this medicine.")
        lines.append("")

    context_lines = _format_context_highlights(highlights)
    if context_lines:
        lines.append("Context highlights:")
        for line in context_lines[:5]:
            lines.append(f"- {line}")
        lines.append("")

    lines.extend(
        [
            "Next step: compare this with the prescription label and confirm the final decision with a pharmacist or doctor.",
        ]
    )
    return "\n".join(lines).strip()


def build_chat_response(
    *,
    message: str,
    disease: str,
    age: int,
    drugs: List[str],
    request_id: str = "",
) -> Dict[str, Any]:
    """
    Rule-based chat response grounded in the existing lookup engine.
    
    Now includes:
    - Emergency symptom detection
    - Safety validation
    - Enhanced disclaimers
    
    The contract is intentionally stable so an LLM/RAG layer can replace only
    answer generation later while the frontend keeps using the same fields.
    """
    # Check for emergency symptoms first
    is_emergency, emergency_symptoms = detect_emergency_symptoms(message)
    
    # Validate user intent and safety
    safety_check = MedicalSafetyGuard.validate_user_intent(message, disease, age)
    
    explicit_drugs = _unique_keep_order(drugs)
    detected_entities = extract_medication_entities(message, disease=disease, age=age, max_entities=5)
    detected_names = entity_names(detected_entities)
    queries = _unique_keep_order(explicit_drugs + detected_names)
    intent = _intent_from_message(message)

    matched: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []

    for query in queries[:5]:
        result = lookup_drug(query, disease=disease, age=age, top_k=1)
        best_match = result.get("best_match")
        record: Dict[str, Any] = {
            "query": query,
            "status": result.get("status"),
            "confidence": result.get("confidence"),
            "best_score": result.get("best_score"),
            "normalized": result.get("normalized"),
            "best_match": best_match,
        }
        if best_match:
            record["sections"] = _sections_from_match(best_match)
            citations.append(
                {
                    "drug": best_match.get("generic_name_clean") or best_match.get("generic_name") or query,
                    "sources": best_match.get("sources", ""),
                    "last_updated": best_match.get("last_updated", ""),
                    "sections_used": ["warnings", "contraindications", "dosage_and_administration"],
                }
            )
        matched.append(record)

    best_match_for_context = next((m.get("best_match") for m in matched if m.get("best_match")), None)
    highlights = build_context_highlights(disease=disease, age=age, match=best_match_for_context)
    answer = _build_answer(
        message=message,
        disease=disease,
        age=age,
        intent=intent,
        matched=[m for m in matched if m.get("best_match")],
        highlights=highlights,
    )
    
    # If emergency detected, override answer with urgent warning
    if is_emergency:
        answer = MedicalSafetyGuard.build_emergency_warning(emergency_symptoms)
    else:
        # Add safety footer to normal answers
        answer = MedicalSafetyGuard.enhance_answer_with_safety(answer, disease, age)

    out: Dict[str, Any] = {
        "context": {"disease": disease, "age": age},
        "supported_diseases": SUPPORTED_DISEASES,
        "message": message,
        "intent": intent,
        "answer": answer,
        "safety_notice": SAFETY_NOTICE,
        "detected_entities": detected_entities,
        "matched_drugs": matched,
        "citations": citations,
        "context_highlights": highlights,
        # New safety metadata
        "is_emergency": is_emergency,
        "emergency_symptoms": emergency_symptoms,
        "safety_validation": {
            "has_forbidden_patterns": safety_check["has_forbidden_patterns"],
            "forbidden_patterns": safety_check["forbidden_patterns"],
            "has_safe_patterns": safety_check["has_safe_patterns"],
        },
    }

    if request_id:
        out["request_id"] = request_id

    return out
