from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.services.drug_lookup import lookup_drug, normalize_text


DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ug|ml|l|iu|units?|%)\b",
    re.IGNORECASE,
)
FREQUENCY_RE = re.compile(
    r"\b(?:once|twice|daily|nightly|morning|evening|weekly|monthly|"
    r"od|bd|bid|tds|tid|qid|qhs|qam|qpm|stat|prn|as needed|"
    r"every\s+\d+\s*(?:hours?|hrs?|days?)|before meals?|after meals?|with food)\b",
    re.IGNORECASE,
)
ROUTE_RE = re.compile(
    r"\b(?:oral|orally|by mouth|po|iv|intravenous|im|intramuscular|"
    r"sc|subcutaneous|topical|inhaled|inhalation|nasal|ophthalmic|otic)\b",
    re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']*")

STOPWORDS = {
    "a",
    "about",
    "after",
    "age",
    "am",
    "and",
    "are",
    "as",
    "at",
    "before",
    "can",
    "daily",
    "disease",
    "disease",
    "do",
    "does",
    "for",
    "have",
    "how",
    "hypertension",
    "asthma",
    "diabetes",
    "arthritis",
    "migraine",
    "heart",
    "chronic",
    "moderate",
    "persistent",
    "controlled",
    "inhaler",
    "i",
    "in",
    "is",
    "it",
    "instructions",
    "additional",
    "appointment",
    "avoid",
    "better",
    "breath",
    "clinic",
    "current",
    "date",
    "doctor",
    "ensure",
    "experiencing",
    "follow",
    "gender",
    "given",
    "healthcare",
    "keep",
    "label",
    "license",
    "male",
    "female",
    "me",
    "medicine",
    "medicines",
    "medication",
    "medical",
    "mg",
    "ml",
    "months",
    "my",
    "of",
    "on",
    "once",
    "order",
    "ordered",
    "patient",
    "patients",
    "pediatric",
    "please",
    "prescription",
    "proper",
    "provider",
    "rescue",
    "review",
    "safe",
    "section",
    "shortness",
    "sign",
    "signature",
    "table",
    "therapy",
    "visit",
    "years",
    "year",
    "use",
    "used",
    "with",
    "withhold",
    "note",
    "notes",
    "should",
    "take",
    "tablet",
    "tablets",
    "the",
    "this",
    "to",
    "usage",
    "twice",
}


def _clean_piece(text: str) -> str:
    text = re.sub(r"\b(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules|syrup|inj|injection)\b", " ", text, flags=re.IGNORECASE)
    text = DOSAGE_RE.sub(" ", text)
    text = FREQUENCY_RE.sub(" ", text)
    text = ROUTE_RE.sub(" ", text)
    text = re.sub(r"[^A-Za-z0-9\-' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_segments(text: str) -> List[str]:
    raw = re.split(r"[\n;,]|(?:\s+-\s+)", text or "")
    segments = [re.sub(r"\s+", " ", x).strip() for x in raw if x and x.strip()]
    return segments[:80]


def _context_fields(segment: str) -> Dict[str, Optional[str]]:
    dosage = DOSAGE_RE.search(segment or "")
    frequency = FREQUENCY_RE.search(segment or "")
    route = ROUTE_RE.search(segment or "")
    return {
        "dosage": dosage.group(0).strip() if dosage else None,
        "frequency": frequency.group(0).strip() if frequency else None,
        "route": route.group(0).strip() if route else None,
    }


def _candidate_ngrams(segment: str) -> Iterable[str]:
    cleaned = _clean_piece(segment)
    cleaned_tokens = TOKEN_RE.findall(cleaned)
    if cleaned and 1 <= len(cleaned_tokens) <= 5:
        if all(t.lower() not in STOPWORDS for t in cleaned_tokens):
            yield cleaned

    tokens = cleaned_tokens
    tokens = [t for t in tokens if t.lower() not in STOPWORDS and len(t) > 1]
    for n in range(4, 0, -1):
        for i in range(0, max(0, len(tokens) - n + 1)):
            gram = " ".join(tokens[i : i + n]).strip()
            if gram and normalize_text(gram):
                yield gram


def _entity_key(best_match: Dict[str, Any], fallback: str) -> str:
    return (
        str(best_match.get("drug_id") or "").strip()
        or normalize_text(str(best_match.get("generic_name_clean") or best_match.get("generic_name") or fallback))
    )


def extract_medication_entities(
    text: str,
    *,
    disease: str,
    age: int,
    max_entities: int = 5,
    min_confidence: float = 0.84,
) -> List[Dict[str, Any]]:
    """
    Lightweight rule-based medication extraction.

    This is intentionally small and deterministic for the project demo:
    - split prescription/chat text into likely medication segments
    - extract dosage/frequency/route with regexes
    - confirm drug names through the existing lookup engine
    """
    seen_candidates: set[str] = set()
    seen_entities: set[str] = set()
    entities: List[Dict[str, Any]] = []

    for segment in _split_segments(text):
        fields = _context_fields(segment)
        for candidate in _candidate_ngrams(segment):
            candidate_key = normalize_text(candidate)
            if not candidate_key or candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)

            result = lookup_drug(candidate, disease=disease, age=age, top_k=1, min_score=82)
            best_match = result.get("best_match")
            confidence = float(result.get("confidence") or 0.0)
            if not best_match or confidence < min_confidence:
                continue

            entity_key = _entity_key(best_match, candidate)
            if not entity_key or entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)

            name = best_match.get("generic_name_clean") or best_match.get("generic_name") or result.get("normalized")
            entities.append(
                {
                    "text": segment,
                    "candidate": candidate,
                    "drug": name,
                    "normalized": result.get("normalized"),
                    "confidence": result.get("confidence"),
                    "best_score": result.get("best_score"),
                    "dosage": fields["dosage"],
                    "frequency": fields["frequency"],
                    "route": fields["route"],
                    "best_match": best_match,
                }
            )

            if len(entities) >= max_entities:
                return entities

    return entities


def entity_names(entities: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for entity in entities:
        name = str(entity.get("drug") or entity.get("normalized") or entity.get("candidate") or "").strip()
        key = normalize_text(name)
        if name and key and key not in seen:
            seen.add(key)
            names.append(name)
    return names
