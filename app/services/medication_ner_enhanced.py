"""
Enhanced NER Service: Medical medication entity extraction with spaCy fallback.

Features:
- Attempts spaCy-based medical NER first (if model available)
- Falls back to rule-based regex NER
- Better dosage/frequency/route parsing
- Prescription line-item grouping
- Hybrid confidence scoring
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.services.medication_ner import (
    extract_medication_entities as extract_rule_based,
    _split_segments,
    _context_fields,
    _candidate_ngrams,
    _entity_key,
    STOPWORDS,
    TOKEN_RE,
)
from app.services.drug_lookup import lookup_drug, normalize_text

logger = logging.getLogger("mediassist.ner")

# Configuration
SPACY_ENABLED = False
nlp = None

try:
    import spacy

    # Try to load medical NER model
    # Note: Users should install with: python -m spacy download en_core_sci_sm
    try:
        nlp = spacy.load("en_core_sci_sm")
        SPACY_ENABLED = True
        logger.info("spaCy medical model loaded successfully")
    except OSError:
        logger.info(
            "spaCy medical model not found. Install with: python -m spacy download en_core_sci_sm"
        )
        # Fall back to using regular English model
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Using default spaCy model as fallback")
        except OSError:
            logger.info("spaCy model not available. Using rule-based NER only.")

except ImportError:
    logger.info("spaCy not installed. Install with: pip install spacy")


def extract_medication_entities_enhanced(
    text: str,
    *,
    disease: str,
    age: int,
    max_entities: int = 5,
    min_confidence: float = 0.84,
    use_spacy: bool = True,
) -> List[Dict[str, Any]]:
    """
    Enhanced medication extraction using spaCy + rule-based hybrid approach.

    Attempts spaCy-based extraction first, then enhances or falls back to rule-based.

    Args:
        text: Prescription or chat text
        disease: Patient disease context
        age: Patient age
        max_entities: Maximum entities to return
        min_confidence: Minimum confidence threshold
        use_spacy: Whether to try spaCy-based extraction

    Returns:
        List of medication entities with dosage/frequency/route
    """

    if use_spacy and SPACY_ENABLED and nlp:
        entities = _extract_with_spacy_fallback(
            text, disease, age, max_entities, min_confidence
        )
    else:
        # Pure rule-based fallback
        entities = extract_rule_based(
            text, disease=disease, age=age, max_entities=max_entities, min_confidence=min_confidence
        )

    return entities


def _extract_with_spacy_fallback(
    text: str,
    disease: str,
    age: int,
    max_entities: int = 5,
    min_confidence: float = 0.84,
) -> List[Dict[str, Any]]:
    """
    Hybrid NER using spaCy with fallback to rule-based extraction.

    Process:
    1. Use spaCy to identify drug-related entities
    2. Enhance with rule-based dosage/frequency/route extraction
    3. Verify matches in drug database
    4. Combine confidence scores
    """

    if not nlp:
        # Fall back to pure rule-based
        return extract_rule_based(
            text, disease=disease, age=age, max_entities=max_entities, min_confidence=min_confidence
        )

    try:
        # Parse text with spaCy
        doc = nlp(text)

        # Extract entities recognized by spaCy
        spacy_entities = []
        for ent in doc.ents:
            # Focus on relevant entity types
            if ent.label_ in {"DRUG", "CHEMICAL", "PRODUCT"}:
                spacy_entities.append((ent.text, ent.label_))

        # Combine spaCy results with rule-based candidates
        seen_candidates = set()
        entities_list: List[Dict[str, Any]] = []

        # Process spaCy entities first (higher priority)
        for candidate_text, entity_type in spacy_entities:
            candidate_key = normalize_text(candidate_text)
            if candidate_key in seen_candidates:
                continue

            seen_candidates.add(candidate_key)

            # Find context in original text
            match_start = text.lower().find(candidate_text.lower())
            if match_start >= 0:
                match_end = match_start + len(candidate_text)
                # Get surrounding context (100 chars)
                context_start = max(0, match_start - 100)
                context_end = min(len(text), match_end + 100)
                context = text[context_start:context_end]
            else:
                context = text

            fields = _extract_dosage_frequency_route(context)

            result = lookup_drug(
                candidate_text, disease=disease, age=age, top_k=1, min_score=82
            )
            best_match = result.get("best_match")

            if best_match or result.get("confidence", 0) > min_confidence:
                # Boost confidence for spaCy-recognized entities
                confidence = min(
                    1.0, (result.get("confidence", 0.8) * 1.1)
                )  # 10% boost

                entity_record = {
                    "query": candidate_text,
                    "normalized": normalize_text(candidate_text),
                    "confidence": confidence,
                    "best_match": best_match,
                    "source": "spacy",
                    "dosage": fields.get("dosage"),
                    "frequency": fields.get("frequency"),
                    "route": fields.get("route"),
                    "sections": _extract_sections_from_match(best_match) if best_match else {},
                }

                entities_list.append(entity_record)

                if len(entities_list) >= max_entities:
                    return entities_list

        # Fall back to rule-based for remaining candidates
        segments = _split_segments(text)
        for segment in segments:
            if len(entities_list) >= max_entities:
                break

            # Skip if we've seen this segment
            segment_key = normalize_text(segment)
            if segment_key in seen_candidates:
                continue

            fields = _context_fields(segment)
            for candidate in _candidate_ngrams(segment):
                if len(entities_list) >= max_entities:
                    break

                candidate_key = normalize_text(candidate)
                if candidate_key in seen_candidates:
                    continue

                seen_candidates.add(candidate_key)

                result = lookup_drug(
                    candidate, disease=disease, age=age, top_k=1, min_score=82
                )
                best_match = result.get("best_match")

                if best_match or result.get("confidence", 0) > min_confidence:
                    entity_record = {
                        "query": candidate,
                        "normalized": candidate_key,
                        "confidence": result.get("confidence", 0.8),
                        "best_match": best_match,
                        "source": "rule_based",
                        "dosage": fields.get("dosage"),
                        "frequency": fields.get("frequency"),
                        "route": fields.get("route"),
                        "sections": _extract_sections_from_match(best_match) if best_match else {},
                    }

                    entities_list.append(entity_record)

        return entities_list

    except Exception as e:
        logger.error(f"spaCy extraction failed, falling back to rule-based: {e}")
        return extract_rule_based(
            text, disease=disease, age=age, max_entities=max_entities, min_confidence=min_confidence
        )


def _extract_dosage_frequency_route(text: str) -> Dict[str, Optional[str]]:
    """Extract dosage, frequency, and route from context text."""
    from app.services.medication_ner import DOSAGE_RE, FREQUENCY_RE, ROUTE_RE

    dosage = DOSAGE_RE.search(text or "")
    frequency = FREQUENCY_RE.search(text or "")
    route = ROUTE_RE.search(text or "")

    return {
        "dosage": dosage.group(0).strip() if dosage else None,
        "frequency": frequency.group(0).strip() if frequency else None,
        "route": route.group(0).strip() if route else None,
    }


def _extract_sections_from_match(match: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant sections from drug match."""
    side_effects = match.get("side_effects_buckets") or {}
    return {
        "warnings": (match.get("warnings") or "")[:200],
        "contraindications": (match.get("contraindications") or "")[:200],
        "dosage_and_administration": (match.get("dosage_and_administration") or "")[:200],
        "indications": (match.get("indications") or "")[:200],
        "side_effects": side_effects,
    }


def get_ner_status() -> Dict[str, Any]:
    """Get NER system status and capabilities."""
    return {
        "spacy_available": SPACY_ENABLED,
        "spacy_model": "en_core_sci_sm" if SPACY_ENABLED else None,
        "mode": "hybrid" if SPACY_ENABLED else "rule_based",
        "fallback_available": True,
    }
