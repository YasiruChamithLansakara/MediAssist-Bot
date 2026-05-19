from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.services.drug_lookup import lookup_drug, normalize_text

logger = logging.getLogger("mediassist.ner")


# =====================================================
# SCI SPACY INITIALIZATION
# =====================================================
import spacy

nlp = None
SPACY_AVAILABLE = False

try:
    # REQUIRED MODEL:
    # python -m spacy download en_core_sci_md
    nlp = spacy.load("en_core_sci_md")
    SPACY_AVAILABLE = True
    logger.info("SciSpaCy model loaded: en_core_sci_md")

except OSError:
    logger.error(
        "SciSpaCy model not found. Install with:\n"
        "python -m spacy download en_core_sci_md"
    )

except Exception as e:
    logger.error(f"SciSpaCy initialization failed: {e}")


# =====================================================
# MAIN ENTRY FUNCTION
# =====================================================
def extract_medication_entities(
    text: str,
    *,
    disease: str,
    age: int,
    max_entities: int = 5,
    min_confidence: float = 0.80,
) -> List[Dict[str, Any]]:
    """
    SciSpaCy-based medical entity extraction + drug validation
    """

    if not SPACY_AVAILABLE or nlp is None:
        logger.warning("SciSpaCy not available, returning empty result")
        return []

    try:
        doc = nlp(text)

        entities: List[Dict[str, Any]] = []
        seen = set()

        # =====================================================
        # STEP 1: EXTRACT SCIENTIFIC ENTITIES
        # =====================================================
        for ent in doc.ents:

            # Focus ONLY on biomedical entities
            if ent.label_ not in {
                "CHEMICAL",
                "DRUG",
                "GENE_OR_GENE_PRODUCT",
                "DISEASE",
            }:
                continue

            candidate = ent.text.strip()
            key = normalize_text(candidate)

            if not candidate or key in seen:
                continue

            seen.add(key)

            # =====================================================
            # STEP 2: VALIDATE WITH DRUG DATABASE
            # =====================================================
            result = lookup_drug(
                candidate,
                disease=disease,
                age=age,
                top_k=1,
                min_score=80,
            )

            best_match = result.get("best_match")
            confidence = float(result.get("confidence") or 0.0)

            # =====================================================
            # STEP 3: FILTER VALID DRUGS ONLY
            # =====================================================
            if not best_match or confidence < min_confidence:
                continue

            entities.append(
                {
                    "text": candidate,
                    "normalized": key,
                    "label": ent.label_,
                    "confidence": confidence,
                    "best_match": best_match,
                    "drug_name": best_match.get(
                        "generic_name_clean"
                    )
                    or best_match.get("generic_name"),
                    "source": "scispacy_en_core_sci_md",
                }
            )

            if len(entities) >= max_entities:
                break

        return entities

    except Exception as e:
        logger.error(f"NER extraction failed: {e}")
        return []


# =====================================================
# STATUS FUNCTION (FOR DEBUGGING / API)
# =====================================================
def get_ner_status() -> Dict[str, Any]:
    return {
        "scispacy_enabled": SPACY_AVAILABLE,
        "model": "en_core_sci_md" if SPACY_AVAILABLE else None,
        "mode": "biomedical_ner",
    }