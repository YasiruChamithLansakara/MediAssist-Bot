# Improved by Nazifa
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mediassist.ner")

# ── SciSpaCy load ─────────────────────────────────────────────────────────
#
# Model: en_core_sci_sm  (biomedical text, trained on CRAFT corpus)
# Install (matches spacy 3.6.x):
#   pip install "scispacy==0.5.4"
#   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

_nlp = None
_SCISPACY_AVAILABLE = False

try:
    import spacy
    try:
        _nlp = spacy.load("en_core_sci_sm")
        _SCISPACY_AVAILABLE = True
        logger.info("SciSpaCy en_core_sci_sm loaded successfully")
    except OSError:
        logger.warning(
            "en_core_sci_sm model not found. Install with:\n"
            "  pip install scispacy\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
        )
except ImportError:
    logger.warning("spaCy not installed. Run: pip install scispacy")

# ── Entity label mappings ─────────────────────────────────────────────────
#
# en_core_sci_sm (CRAFT corpus) entity types we care about:
#   CHEMICAL / SIMPLE_CHEMICAL  → likely a drug or compound
#   GENE_OR_GENE_PRODUCT        → sometimes drug targets / biologics
#   DISEASE / PATHOLOGICAL_*    → disease mentions
#   ORGANISM                    → bacteria/virus in infection context
#   CELL_TYPE / TISSUE          → anatomy context (lower priority)

_DRUG_ENTITY_LABELS = {
    "CHEMICAL",
    "SIMPLE_CHEMICAL",
    "GENE_OR_GENE_PRODUCT",
    "GGP",
    "DRUG",
}

_DISEASE_ENTITY_LABELS = {
    "DISEASE",
    "PATHOLOGICAL_FORMATION",
    "CANCER",
    "ORGANISM",
}

_CHEMICAL_ENTITY_LABELS = {
    "CHEMICAL",
    "SIMPLE_CHEMICAL",
}

_DISEASE_KEYWORDS = {
    "diabetes": ["diabetes", "diabetic", "blood sugar", "glucose", "hyperglyc", "hypoglyc"],
    "hypertension": ["hypertension", "blood pressure", "bp", "edema", "fluid retention"],
    "asthma": ["asthma", "bronchospasm", "wheezing", "respiratory"],
    "heart disease": ["cardiac", "cardiovascular", "heart failure", "arrhythm", "stroke", "myocard"],
    "arthritis": ["arthritis", "joint", "inflammation", "pain", "ulcer", "gi bleeding"],
    "migraine": ["migraine", "headache", "neurolog"],
}

# ── Regex patterns ────────────────────────────────────────────────────────

DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ug|ml|l|iu|meq|units?|mEq|%)"
    r"(?:/\s*(?:\d+\s*)?(?:mg|g|mcg|ml|l|kg|dose|tab))?\b",
    re.IGNORECASE,
)
FREQUENCY_RE = re.compile(
    r"\b(?:once|twice|daily|nightly|morning|evening|weekly|monthly|nocte|mane|"
    r"od|bd|bid|tds|tid|qid|qhs|qam|qpm|stat|prn|sos|as\s+needed|"
    r"every\s+\d+\s*(?:hours?|hrs?|days?)|before\s+meals?|after\s+meals?|with\s+food|"
    r"[xX]\s*\d+\s*(?:days?|weeks?|doses?)|"        # x 5 days, x 3 doses
    r"[Dd](?:ay)?\s*\d+(?:\s*[-–]\s*[Dd](?:ay)?\s*\d+)?|"  # D1, Day 1, D1-D4
    r"q\.?d\.?|b\.?i\.?d\.?|t\.?i\.?d\.?|q\.?i\.?d\.?)\b",  # q.d. b.i.d. etc.
    re.IGNORECASE,
)
ROUTE_RE = re.compile(
    r"\b(?:oral|orally|by\s+mouth|po|p\.o\.?|"
    r"iv|i\.v\.?|intravenous(?:ly)?|im|i\.m\.?|intramuscular(?:ly)?|"
    r"sc|s\.c\.?|subcutaneous(?:ly)?|subcut|"
    r"topical(?:ly)?|inhaled|inhalation|nasal|ophthalmic|otic|"
    r"sublingual|transdermal|rectal|nebulisation)\b",
    re.IGNORECASE,
)

# Prescription shorthand prefixes — stripped before candidate extraction
# e.g. "T. Metformin", "Inj. Xgeva", "Cap. Amoxicillin", "Syr. Paracetamol"
_PRESCRIPTION_PREFIX_RE = re.compile(
    r"(?:^|\s)(?:T|Tab|Tabs|Cap|Caps|Inj|Syr|Sol|Oint|Cr|Supp|Drops?|Dr)"
    r"\.?\s+",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_prescription_text(text: str) -> str:
    """Strip common prescription shorthands that break drug name extraction."""
    text = _PRESCRIPTION_PREFIX_RE.sub(" ", text or "")
    # Split OCR run-ons like "Metformin500mg" → "Metformin 500mg".
    # Require 4+ letters so legitimate tokens like "HbA1c", "B12", "T3",
    # "COVID19", "Omega3" are never split (they all start with ≤3 letters).
    text = re.sub(r"([A-Za-z]{4,})(\d)", r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()

# Minimum character length for a candidate to be considered a drug name
_MIN_DRUG_LEN = 4


# ── Context extraction ────────────────────────────────────────────────────

def _extract_context(text: str) -> Dict[str, Optional[str]]:
    dosage    = DOSAGE_RE.search(text)
    frequency = FREQUENCY_RE.search(text)
    route     = ROUTE_RE.search(text)
    return {
        "dosage":    dosage.group(0).strip()    if dosage    else None,
        "frequency": frequency.group(0).strip() if frequency else None,
        "route":     route.group(0).strip()     if route     else None,
    }


def _segment_context(text: str, candidate: str) -> Dict[str, Optional[str]]:
    """
    Find the line/segment containing the candidate and extract its context.
    Falls back to whole-text context.
    """
    for line in re.split(r"[\n;,]", text):
        if candidate.lower() in line.lower():
            ctx = _extract_context(line)
            if any(ctx.values()):
                return ctx
    return _extract_context(text)


def _segment_context_strict(text: str, candidate: str) -> Dict[str, Optional[str]]:
    """Return only the context from the line containing the candidate."""
    for line in re.split(r"[\n;,]", text):
        if candidate.lower() in line.lower():
            return _extract_context(line)
    return {"dosage": None, "frequency": None, "route": None}


# ── SciSpaCy entity extraction ────────────────────────────────────────────

def _scispacy_entities(text: str) -> Dict[str, List[str]]:
    """Run SciSpaCy NER and bin entities into drugs / diseases / chemicals."""
    if not _SCISPACY_AVAILABLE or _nlp is None:
        return {"drugs": [], "diseases": [], "chemicals": []}

    doc  = _nlp(text)
    seen = set()
    drugs, diseases, chemicals = [], [], []

    for ent in doc.ents:
        raw   = ent.text.strip()
        label = ent.label_.upper()
        key   = raw.lower()

        if not raw or len(raw) < _MIN_DRUG_LEN or key in seen:
            continue
        seen.add(key)

        if label in _DRUG_ENTITY_LABELS:
            drugs.append(raw)
        if label in _DISEASE_ENTITY_LABELS:
            diseases.append(raw)
        if label in _CHEMICAL_ENTITY_LABELS:
            chemicals.append(raw)

    return {"drugs": drugs, "diseases": diseases, "chemicals": chemicals}


# ── Rule-based candidate extraction ──────────────────────────────────────

def _split_segments(text: str) -> List[str]:
    """Split prescription text into segments by newline, semicolon, comma, or dash."""
    raw = re.split(r"[\n;,]|(?:\s+-\s+)", text or "")
    segments = [re.sub(r"\s+", " ", x).strip() for x in raw if x and x.strip()]
    return segments[:80]


def _candidate_ngrams(segment: str):
    """Generate n-gram candidates from a segment, filtering stopwords."""
    from app.services.drug_lookup import normalize_text
    
    STOPWORDS = {
        "a", "about", "after", "age", "am", "and", "are", "as", "at", "before", "can",
        "daily", "disease", "do", "does", "for", "have", "how", "hypertension", "asthma",
        "diabetes", "arthritis", "migraine", "heart", "chronic", "i", "in", "is", "it",
        "me", "medication", "mg", "ml", "of", "on", "the", "to", "use", "with",
    }
    TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']*")
    
    # Clean the segment
    text = re.sub(r"\b(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules|syrup|inj|injection)\b", " ", segment, flags=re.IGNORECASE)
    text = DOSAGE_RE.sub(" ", text)
    text = FREQUENCY_RE.sub(" ", text)
    text = ROUTE_RE.sub(" ", text)
    text = re.sub(r"[^A-Za-z0-9\-' ]+", " ", text)
    cleaned = re.sub(r"\s+", " ", text).strip()
    
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


def _rule_based_candidates(text: str) -> List[str]:
    """
    Split prescription text into segments, then generate n-gram candidates.
    """
    from app.services.drug_lookup import normalize_text

    candidates: List[str] = []
    seen: set[str] = set()

    for segment in _split_segments(text):
        for candidate in _candidate_ngrams(segment):
            key = normalize_text(candidate)
            if key and len(key) >= _MIN_DRUG_LEN and key not in seen:
                seen.add(key)
                candidates.append(candidate)

    return candidates

def _context_fields(segment: str) -> Dict[str, Optional[str]]:
    """Extract dosage, frequency, and route from a segment."""
    dosage = DOSAGE_RE.search(segment or "")
    frequency = FREQUENCY_RE.search(segment or "")
    route = ROUTE_RE.search(segment or "")
    return {
        "dosage": dosage.group(0).strip() if dosage else None,
        "frequency": frequency.group(0).strip() if frequency else None,
        "route": route.group(0).strip() if route else None,
    }


def _entity_key(best_match: Dict[str, Any], fallback: str) -> str:
    """Generate a unique key for an entity."""
    from app.services.drug_lookup import normalize_text
    return (
        str(best_match.get("drug_id") or "").strip()
        or normalize_text(str(best_match.get("generic_name_clean") or best_match.get("generic_name") or fallback))
    )


_NON_DRUG_FRAGMENTS = {
    "formula", "relief", "management", "supplement", "booster",
    "complex", "blend", "extract", "support", "therapy",
}


def extract_medication_entities(
    text: str,
    *,
    disease: str,
    age: int,
    max_entities: int = 5,
    min_confidence: float = 0.84,
) -> List[Dict[str, Any]]:
    """
    OCR-path medication extraction.

    Lower confidence threshold than chat (0.78 vs 0.82) because OCR text
    contains character-level errors that reduce fuzzy-match scores.
    Prescription abbreviations (T., Cap., Inj.) are stripped first so they
    don't break candidate generation.

    Pipeline:
        1. Clean prescription shorthands ("T. Metformin" → "Metformin")
        2. Split into segments → generate n-gram candidates
        3. Validate each candidate via drug lookup (min_score=70 for OCR)
    """
    from app.services.drug_lookup import lookup_drug, normalize_text

    # Strip prescription prefixes that break candidate extraction
    cleaned_text = _clean_prescription_text(text)

    seen_candidates: set[str] = set()
    seen_entities: set[str] = set()
    entities: List[Dict[str, Any]] = []

    for segment in _split_segments(cleaned_text):
        fields = _context_fields(segment)
        for candidate in _candidate_ngrams(segment):
            candidate_key = normalize_text(candidate)
            if not candidate_key or candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)

            # Lower min_score=70 for OCR — fuzzy matching must tolerate OCR errors
            result = lookup_drug(candidate, disease=disease, age=age, top_k=1, min_score=70)
            best_match = result.get("best_match")
            confidence = float(result.get("confidence") or 0.0)
            if not best_match or confidence < min_confidence:
                continue

            entity_key = _entity_key(best_match, candidate)
            if not entity_key or entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)

            name = (
                best_match.get("generic_name_clean")
                or best_match.get("generic_name")
                or result.get("normalized")
                or ""
            )

            # Reject very short resolved names (e.g. "ibu", "tin") and
            # product-description phrases that aren't real drug names.
            name_lower = name.lower()
            if len(name_lower) < 4:
                continue
            if any(frag in name_lower for frag in _NON_DRUG_FRAGMENTS):
                continue

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

# ── Drug validation against lookup ────────────────────────────────────────

def _validate_candidates(
    candidates: List[str],
    *,
    disease: str,
    age: int,
    source_tag: str,
    max_drugs: int,
    min_confidence: float,
    seen_ids: set,
    text: str,
) -> List[Dict[str, Any]]:
    """Validate candidate strings against drug_lookup. Returns confirmed drug entities."""
    from app.services.drug_lookup import lookup_drug, normalize_text

    validated: List[Dict[str, Any]] = []

    for candidate in candidates:
        if len(validated) >= max_drugs:
            break

        token_count = len(normalize_text(candidate).split())

        result     = lookup_drug(candidate, disease=disease, age=age, top_k=1, min_score=82)
        best_match = result.get("best_match")
        confidence = float(result.get("confidence") or 0.0)
        best_score = float(result.get("best_score") or 0.0)
        strict_ctx = _segment_context_strict(text, candidate)
        ctx = _segment_context(text, candidate)

        if not best_match or confidence < min_confidence:
            continue

        candidate_tokens = set(normalize_text(candidate).split())
        best_name = best_match.get("generic_name_clean") or best_match.get("generic_name") or best_match.get("brand_names") or ""
        best_tokens = set(normalize_text(str(best_name)).split())
        shared_tokens = candidate_tokens & best_tokens

        disease_blob = " ".join(
            [
                str(best_match.get("drug_class") or ""),
                str(best_match.get("indications") or ""),
                str(best_match.get("warnings") or ""),
                str(best_match.get("contraindications") or ""),
            ]
        ).lower()
        disease_kws = _DISEASE_KEYWORDS.get((disease or "").strip().lower(), [])
        disease_matches = any(kw in disease_blob for kw in disease_kws)

        if token_count > 1 and not shared_tokens and result.get("match_type") not in {"exact", "alias"} and best_score < 95.0:
            continue

        if token_count > 3 and best_score < 90.0:
            continue

        if token_count > 5:
            continue

        if not any(strict_ctx.values()) and result.get("match_type") not in {"exact", "alias"} and not shared_tokens:
            continue

        # Only apply disease-relevance filter for borderline-confidence matches.
        # High-confidence matches (≥ 0.90) are accepted regardless — sparse drug
        # data may not mention the disease but the drug is still valid.
        if (
            confidence < 0.90
            and disease_kws
            and result.get("match_type") not in {"exact", "alias"}
            and not disease_matches
        ):
            continue

        drug_id = best_match.get("drug_id", "")
        if drug_id and drug_id in seen_ids:
            continue
        if drug_id:
            seen_ids.add(drug_id)

        validated.append({
            "text":       candidate,
            "drug_name":  (
                best_match.get("generic_name_clean")
                or best_match.get("generic_name")
                or candidate
            ),
            "drug_id":    drug_id,
            "confidence": confidence,
            "best_score": result.get("best_score"),
            "best_match": best_match,
            "dosage":     ctx.get("dosage"),
            "frequency":  ctx.get("frequency"),
            "route":      ctx.get("route"),
            "source":     source_tag,
        })

    return validated


# ── Main NER entry point ──────────────────────────────────────────────────

def extract_medical_entities(
    text: str,
    *,
    disease: str,
    age: int,
    max_drugs: int = 8,
    min_confidence: float = 0.82,
) -> Dict[str, Any]:
    """
    Extract DRUGS, DISEASES, and CHEMICALS from clean prescription text.

    Pipeline:
        1. SciSpaCy en_core_sci_sm  → biomedical entity recognition
        2. Rule-based regex         → dosage / frequency / route
        3. Drug lookup validation   → confirm each candidate is a real drug

    Args:
        text:           Clean text from OCR or direct input
        disease:        Patient's chronic disease context
        age:            Patient age
        max_drugs:      Maximum number of drug entities to return
        min_confidence: Minimum lookup confidence to accept a drug match

    Returns:
        {
          "drugs":              [list of validated drug entities],
          "diseases":           [disease strings from SciSpaCy],
          "chemicals":          [chemical strings from SciSpaCy],
          "context":            {dosage, frequency, route} from full text,
          "ner_source":         "scispacy" | "rule_based",
          "scispacy_available": bool
        }
    """
    from app.services.drug_lookup import normalize_text

    if not text or not text.strip():
        return {
            "drugs": [], "diseases": [], "chemicals": [],
            "context": {"dosage": None, "frequency": None, "route": None},
            "ner_source": "none",
            "scispacy_available": _SCISPACY_AVAILABLE,
        }

    # Strip prescription shorthands before NER passes
    text = _clean_prescription_text(text)
    context_global = _extract_context(text)
    seen_ids: set[str] = set()
    drugs: List[Dict[str, Any]] = []

    # ── Pass 1: SciSpaCy candidates ───────────────────────────────────
    if _SCISPACY_AVAILABLE:
        sci = _scispacy_entities(text)
        sci_drug_candidates = sci["drugs"] + sci["chemicals"]
        diseases  = sci["diseases"]
        chemicals = sci["chemicals"]
        ner_source = "scispacy"

        sci_validated = _validate_candidates(
            sci_drug_candidates,
            disease=disease, age=age,
            source_tag="scispacy",
            max_drugs=max_drugs,
            min_confidence=min_confidence,
            seen_ids=seen_ids,
            text=text,
        )
        drugs.extend(sci_validated)
    else:
        diseases   = []
        chemicals  = []
        ner_source = "rule_based"

    # ── Pass 2: Rule-based candidates (supplement / fallback) ─────────
    remaining = max_drugs - len(drugs)
    if remaining > 0:
        rule_candidates = _rule_based_candidates(text)

        # skip candidates already validated via SciSpaCy
        existing_keys = {normalize_text(d["text"]) for d in drugs}
        rule_candidates = [
            c for c in rule_candidates
            if normalize_text(c) not in existing_keys
        ]

        rule_validated = _validate_candidates(
            rule_candidates,
            disease=disease, age=age,
            source_tag="rule_based",
            max_drugs=remaining,
            min_confidence=min_confidence,
            seen_ids=seen_ids,
            text=text,
        )
        drugs.extend(rule_validated)

    return {
        "drugs":              drugs,
        "diseases":           diseases,
        "chemicals":          chemicals,
        "context":            context_global,
        "ner_source":         ner_source,
        "scispacy_available": _SCISPACY_AVAILABLE,
    }


def get_drug_names(ner_result: Dict[str, Any]) -> List[str]:
    """Pull validated drug name strings out of an extract_medical_entities result."""
    names: List[str] = []
    seen:  set[str]  = set()
    for drug in ner_result.get("drugs", []):
        name = str(drug.get("drug_name") or drug.get("text") or "").strip()
        key  = name.lower()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def ner_status() -> Dict[str, Any]:
    return {
        "scispacy_available": _SCISPACY_AVAILABLE,
        "model": "en_core_sci_sm" if _SCISPACY_AVAILABLE else None,
        "mode":  "scispacy+rule_based" if _SCISPACY_AVAILABLE else "rule_based_only",
    }
