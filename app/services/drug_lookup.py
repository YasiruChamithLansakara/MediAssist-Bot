# Improved by Nazifa
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from rapidfuzz import process, fuzz


# ---------------------------------------------------------------------
# PATHS / CONFIG
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../MediAssist-Bot
DEFAULT_CSV_PATH = os.getenv(
    "DRUG_DATASET_PATH",
    str(PROJECT_ROOT / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv"),
)

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 80.0
MAX_BRANDS_RETURNED = 20

ALIASES = {
    "paracetamol": "acetaminophen",
    "panadol": "acetaminophen",
    "tylenol": "acetaminophen",
    "apap": "acetaminophen",
    "salbutamol": "albuterol",
}
ALIAS_SCORE = 95.0

OK_THRESHOLD = 0.85
LOW_THRESHOLD = 0.60
PARACETAMOL_TYPO_THRESHOLD = 85

# ---------------------------------------------------------------------
# INGREDIENT IDENTITY
# ---------------------------------------------------------------------
# Fuzzy string similarity CANNOT decide drug identity. Measured on this
# dataset with rapidfuzz:
#
#     prednisone   vs prednisolone   ratio 90.9   <- DIFFERENT drugs
#     metfarmin    vs metformin      ratio 88.9   <- a typo for the same drug
#     lisinopril   vs fosinopril     ratio 80.0   <- DIFFERENT drugs
#
# The distributions overlap, so no cutoff separates "typo" from "different
# molecule". Before this guard existed, `lookup_drug("lisinopril")` returned
# fosinopril at confidence 0.80 and the chat layer presented that drug's
# dosing and warnings as the answer — a wrong-drug response for a drug that
# is simply absent from the knowledge base.
#
# Policy: identity is established by an EXACT base-ingredient match (after
# salt/ester stripping) or by an alias. Anything else is a *suggestion*,
# returned without clinical fields for the user to confirm.
INGREDIENT_TYPO_RATIO = 94.0     # near-certain typo, still needs confirmation
AMBIGUITY_MARGIN = 4.0           # runner-up must be this far behind

# Salt, ester, hydrate and formulation tokens. Stripping these makes
# "amlodipine besylate" and "amlodipine" the same ingredient, which is the
# single biggest recall win available on this dataset.
_SALT_TOKENS = {
    "hydrochloride", "hcl", "hydrobromide", "hbr", "hydroiodide",
    "sodium", "potassium", "calcium", "magnesium", "zinc", "aluminum",
    "besylate", "mesylate", "maleate", "tartrate", "bitartrate",
    "succinate", "fumarate", "citrate", "acetate", "phosphate",
    "sulfate", "sulphate", "nitrate", "bromide", "chloride", "iodide",
    "valerate", "propionate", "furoate", "dipropionate", "butyrate",
    "mesilate", "tosylate", "oxalate", "pamoate", "embonate",
    "gluconate", "lactate", "malate", "carbonate", "bicarbonate",
    "benzoate", "salicylate", "stearate", "palmitate", "decanoate",
    "lysine", "arginine", "meglumine", "trometamol", "tromethamine",
    "monohydrate", "dihydrate", "trihydrate", "hemihydrate", "anhydrous",
    "micronized", "micronised", "axetil", "estolate", "proxetil",
    "disodium", "dipotassium", "hemifumarate", "xinafoate", "aceponate",
}

# Formulation adjectives that are not part of the ingredient identity.
_FORM_TOKENS = {
    "film", "coated", "extended", "release", "delayed", "immediate",
    "oral", "nasal", "topical", "inhalation", "ophthalmic", "otic",
    "chewable", "dispersible", "effervescent", "sustained", "modified",
    "powder", "solution", "suspension", "gel", "cream", "ointment",
    "spray", "aerosol", "concentrate", "kit", "usp", "er", "xr", "sr", "dr",
}

_INGREDIENT_SPLIT_RE = re.compile(r"\s*(?:,|/|\+|;|\band\b|\bwith\b)\s*", re.IGNORECASE)


def split_ingredients(name: str) -> List[str]:
    """Split a product name into its active-ingredient parts."""
    if not name:
        return []
    parts = [p.strip() for p in _INGREDIENT_SPLIT_RE.split(str(name)) if p and p.strip()]
    return parts or [str(name).strip()]


def base_ingredient(name: str) -> str:
    """
    Reduce one ingredient name to its base molecule.

        "amlodipine besylate"        -> "amlodipine"
        "atorvastatin calcium coated"-> "atorvastatin"
        "fluticasone propionate"     -> "fluticasone"

    If stripping would remove everything, the normalised name is kept — some
    drugs genuinely are salts (e.g. "potassium chloride").
    """
    norm = normalize_text(name)
    if not norm:
        return ""
    tokens = [t for t in norm.split() if t not in _SALT_TOKENS and t not in _FORM_TOKENS]
    return " ".join(tokens) if tokens else norm


def ingredient_key(name: str) -> str:
    """Canonical multi-ingredient key: base of every part, order preserved."""
    bases = [base_ingredient(p) for p in split_ingredients(name)]
    return " ".join(b for b in bases if b)


def ingredient_set(name: str) -> set[str]:
    """The set of base ingredients in a product name."""
    return {b for b in (base_ingredient(p) for p in split_ingredients(name)) if b}

SUPPORTED_DISEASES = [
    "diabetes",
    "hypertension",
    "asthma",
    "heart disease",
    "arthritis",
    "migraine",
]


# ---------------------------------------------------------------------
# INTERNAL STORE
# ---------------------------------------------------------------------
_df: pd.DataFrame | None = None
_index: Dict[str, Dict[str, Any]] = {}
_keys_all: List[str] = []
_keys_primary: List[str] = []
_store_lock = threading.Lock()

# Ingredient-identity indexes (see INGREDIENT IDENTITY above).
#   _base_index       full canonical key ("amlodipine", "ibuprofen famotidine") -> rows
#   _ingredient_index single ingredient  ("ibuprofen") -> every row containing it
_base_index: Dict[str, List[Dict[str, Any]]] = {}
_ingredient_index: Dict[str, List[Dict[str, Any]]] = {}
_base_keys: List[str] = []


# ---------------------------------------------------------------------
# NORMALIZATION + DOSAGE STRIPPING
# ---------------------------------------------------------------------
_non_alnum_re = re.compile(r"[^a-z0-9\s\-]+")

_DOSAGE_PATTERNS = [
    r"\b\d+(\.\d+)?\s*(mg|g|mcg|µg|ug|ml|l|iu|%)\b",
    r"\b\d+\s*(tablet|tab|tabs|capsule|cap|caps|pill|pills)\b",
    r"\b(tablet|tab|tabs|capsule|cap|caps|pill|pills)\b",
    r"\b(syrup|suspension|susp|injection|injectable|oral|iv|im)\b",
    r"\b(once|twice|daily|bd|tds|qid|od)\b",
    r"\b\d+\b",
]
_dosage_re = re.compile("|".join(_DOSAGE_PATTERNS), re.IGNORECASE)


def strip_dosage_and_form(s: str) -> str:
    if not s:
        return ""
    s = _dosage_re.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower().strip()
    s = s.replace("_", " ")
    s = _non_alnum_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = strip_dosage_and_form(s)
    return s


def apply_alias_only(q: str) -> Tuple[str, bool, Optional[str]]:
    qn = normalize_text(q)

    # direct alias
    if qn in ALIASES:
        return qn, True, ALIASES[qn]

    # safe typo handling (paracetamol family only)
    if qn:
        score = fuzz.WRatio(qn, "paracetamol")
        looks_like_para = ("paracet" in qn) or qn.startswith("par") or ("para" in qn)
        if looks_like_para and score >= PARACETAMOL_TYPO_THRESHOLD:
            return qn, True, "acetaminophen"

    return qn, False, None


# ---------------------------------------------------------------------
# RESPONSE STANDARDIZATION
# ---------------------------------------------------------------------
def score_to_confidence(score: float) -> float:
    s = float(score or 0.0)
    s = max(0.0, min(100.0, s))
    return round(s / 100.0, 4)


def classify_status(confidence: float, has_best: bool) -> str:
    if has_best and confidence >= OK_THRESHOLD:
        return "ok"
    if has_best and confidence >= LOW_THRESHOLD:
        return "low_confidence"
    return "no_match"


def _dedupe_path(items: List[str]) -> List[str]:
    if not items:
        return []
    out: List[str] = []
    seen: set[str] = set()

    for raw_item in items:
        raw = "" if raw_item is None else str(raw_item)
        key = normalize_text(raw)

        if not key:
            # keep the first raw element even if it normalizes empty
            if not out:
                out.append(raw)
            continue

        if key in seen:
            continue
        seen.add(key)
        out.append(raw)

    if not out:
        out = [str(items[0])]

    return out


# ---------------------------------------------------------------------
# AGE + DISEASE TAILORING (RULE-BASED, NO AGE DATA IN DATASET NEEDED)
# ---------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _age_group(age: int) -> str:
    if age < 12:
        return "pediatric"
    if age < 18:
        return "adolescent"
    if age < 65:
        return "adult"
    return "elderly"


def _contains(text: str, *keywords: str) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def _extract_relevant_sentences(text: str, keywords: List[str], max_sentences: int = 4) -> List[str]:
    if not text:
        return []
    chunks = [c.strip() for c in _SENT_SPLIT_RE.split(str(text)) if c.strip()]
    keys = [k.lower() for k in keywords if k]
    out: List[str] = []
    for c in chunks:
        cl = c.lower()
        if any(k in cl for k in keys):
            out.append(c)
            if len(out) >= max_sentences:
                break
    return out


def build_context_highlights(*, disease: str, age: int, match: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produces a small, conservative set of highlights to help the UI/chatbot focus.

    Output is intentionally simple:
      - age_group
      - highlights: {age: [...], disease: [...], general: [...]}
      - recommended_sections
    """
    ag = _age_group(int(age))
    d = (disease or "").strip().lower()

    highlights: Dict[str, List[str]] = {"age": [], "disease": [], "general": []}

    if not match:
        highlights["general"].append("No drug matched yet. Provide a drug name for disease/age-based highlights.")
        return {
            "age_group": ag,
            "highlights": {k: v for k, v in highlights.items() if v},
            "recommended_sections": ["warnings", "contraindications", "dosage_and_administration"],
        }

    drug_class = (match.get("drug_class") or "").strip().lower()
    warnings = (match.get("warnings") or "")
    contraindications = (match.get("contraindications") or "")

    blob = " ".join(
        [
            drug_class,
            str(match.get("indications") or ""),
            str(match.get("dosage_and_administration") or ""),
            str(warnings or ""),
            str(contraindications or ""),
        ]
    )
    blob = _MULTI_SPACE_RE.sub(" ", blob).strip().lower()

    # ---- Age highlights (keyword-based)
    if ag in {"pediatric", "adolescent"}:
        if _contains(blob, "pediatric", "children", "child", "infant", "neonate", "under 12", "under 18"):
            highlights["age"].append("Label mentions pediatric/children considerations — verify suitability and dosing for this age.")
    if ag == "elderly":
        if _contains(blob, "elderly", "geriatric", "older patients", "65 years", "age 65"):
            highlights["age"].append("Label mentions elderly/geriatric considerations — older adults may need extra caution or dose review.")

    # ---- Disease highlights (conservative heuristics)
    disease_keywords: Dict[str, List[str]] = {
        "diabetes": ["diabetes", "diabetic", "blood sugar", "glucose", "hypoglyc", "hyperglyc"],
        "hypertension": ["hypertension", "blood pressure", "bp", "fluid retention", "edema"],
        "asthma": ["asthma", "bronchospasm", "wheezing", "respiratory"],
        "heart disease": ["cardiovascular", "cardiac", "stroke", "myocard", "thrombot", "heart failure", "arrhythm"],
        "arthritis": ["arthritis", "inflammation", "pain", "joint", "gi bleeding", "ulcer"],
        "migraine": ["migraine", "headache", "triptan", "serotonin", "aura", "nausea", "photophobia", "neurological"],
    }
    kws = disease_keywords.get(d, [])

    if kws and any(k in blob for k in kws):
        highlights["disease"].append(
            f"Label includes terms related to {d}. Review warnings/contraindications for condition-specific cautions."
        )

    # ---- Disease-specific drug class interaction rules ----------------------

    # NSAID + cardiovascular risk (hypertension / heart disease)
    if d in {"hypertension", "heart disease"}:
        if ("nsaid" in drug_class or "nonsteroidal" in blob) and _contains(blob, "cardiovascular", "thrombotic", "stroke", "myocardial"):
            highlights["disease"].append(
                "⚠️ Cardiovascular risk: NSAIDs can raise blood pressure and increase thrombotic risk — "
                "verify this is appropriate for your condition with your doctor."
            )

    # Beta-blockers are contraindicated in asthma (can trigger bronchospasm)
    if d == "asthma":
        if "beta" in drug_class and ("blocker" in drug_class or "antagonist" in drug_class):
            highlights["disease"].append(
                "⚠️ Beta-blockers are generally contraindicated in asthma — they can cause "
                "bronchospasm. Confirm with your doctor before use."
            )
        elif _contains(blob, "bronchospasm", "bronchoconstriction") and _contains(blob, "asthma", "respiratory"):
            highlights["disease"].append(
                "⚠️ Bronchospasm risk noted in label — important for asthma patients. Review with your doctor."
            )

    # Metformin + alcohol / contrast media for diabetes
    if d == "diabetes":
        if "metformin" in drug_class.lower() or "biguanide" in drug_class.lower():
            highlights["disease"].append(
                "Note: If this is metformin, avoid excessive alcohol use (raises lactic acidosis risk) "
                "and inform your doctor before any contrast imaging procedure."
            )

    # NSAIDs + GI risk for arthritis patients (long-term use is common)
    if d == "arthritis":
        if "nsaid" in drug_class or "nonsteroidal" in blob:
            if _contains(blob, "gastrointestinal", "gi bleeding", "ulcer", "stomach"):
                highlights["disease"].append(
                    "⚠️ GI bleeding/ulcer risk noted for this drug class — important for long-term "
                    "arthritis treatment. Ask your doctor about stomach protection."
                )

    # Triptans + serotonin syndrome risk for migraine
    if d == "migraine":
        if "triptan" in drug_class.lower() or _contains(blob, "serotonin syndrome", "5-ht"):
            highlights["disease"].append(
                "⚠️ Serotonin syndrome risk: triptans interact with SSRIs/SNRIs and some other "
                "migraine medicines. Tell your doctor all medicines you are taking."
            )

    # ---- General highlights
    if str(warnings).strip():
        highlights["general"].append("Warnings section exists — review key safety notes before use.")
    if str(contraindications).strip():
        highlights["general"].append("Contraindications section exists — check if any apply.")

    # Optional: add small evidence snippets (kept short)
    disease_snips = _extract_relevant_sentences(
        " ".join([warnings or "", contraindications or "", str(match.get("dosage_and_administration") or "")]),
        kws,
        max_sentences=3,
    )
    age_snips = _extract_relevant_sentences(warnings or "", ["pediatric", "children", "elderly", "geriatric"], max_sentences=2)

    snippets: Dict[str, List[str]] = {}
    if age_snips:
        snippets["age_relevant"] = age_snips
    if disease_snips:
        snippets["disease_relevant"] = disease_snips

    out = {
        "age_group": ag,
        "highlights": {k: v for k, v in highlights.items() if v},
        "recommended_sections": ["warnings", "contraindications", "dosage_and_administration"],
    }
    if snippets:
        out["snippets"] = snippets

    return out


def build_tailored_context(*, disease: str, age: int, match: Dict[str, Any]) -> Dict[str, Any]:
    highlights = build_context_highlights(disease=disease, age=age, match=match)
    notes: List[str] = []

    for items in (highlights.get("highlights") or {}).values():
        if isinstance(items, list):
            notes.extend(str(item) for item in items if str(item).strip())

    return {
        "disease": disease,
        "age": int(age),
        "age_group": highlights.get("age_group") or _age_group(int(age)),
        "notes": notes,
        "snippets": highlights.get("snippets") or {},
        "recommended_sections": highlights.get("recommended_sections") or [],
    }


def make_response(
    *,
    query: str,
    normalized: str,
    message: str,
    matches: List[Dict[str, Any]],
    suggestions: List[str],
    match_type: str,
    resolution_path: List[str],
    disease: Optional[str] = None,
    age: Optional[int] = None,
    needs_confirmation: bool = False,
    identity_note: str = "",
) -> Dict[str, Any]:
    best_match = matches[0] if matches else None
    best_score = float(best_match.get("score", 0.0)) if best_match else 0.0
    confidence = score_to_confidence(best_score)
    status = classify_status(confidence, bool(best_match))

    path = _dedupe_path(resolution_path)
    if not path:
        path = [query]

    resp: Dict[str, Any] = {
        "query": query,
        "normalized": normalized or "",
        "status": status,
        "confidence": confidence,
        "best_score": round(best_score, 2),
        "best_match": best_match,
        "matches": matches,
        "suggestions": suggestions,
        "message": message,
        "match_type": match_type,
        "resolution_path": path,
        # True when the match was not established by exact ingredient identity
        # or alias. Callers must present these as "did you mean …?" rather
        # than as the answer.
        "needs_confirmation": bool(needs_confirmation),
    }
    if identity_note:
        resp["identity_note"] = identity_note

    # include context if provided (service-level contract)
    if disease is not None or age is not None:
        resp["context"] = {"disease": disease, "age": age}
        resp["supported_diseases"] = SUPPORTED_DISEASES

        # NEW: include context_highlights if we have a match + full context
        if best_match is not None and disease is not None and age is not None:
            context_highlights = build_context_highlights(
                disease=str(disease),
                age=int(age),
                match=best_match,
            )
            resp["context_highlights"] = context_highlights
            resp["tailored"] = build_tailored_context(
                disease=str(disease),
                age=int(age),
                match=best_match,
            )

    return resp


# ---------------------------------------------------------------------
# LOAD + CACHE
# ---------------------------------------------------------------------
_BRAND_SUFFIX_RE = re.compile(r",\s*\.\.\.\s*\(\+\d+\s+more\)\s*$", re.IGNORECASE)


def _strip_brand_suffix(s: str) -> str:
    return _BRAND_SUFFIX_RE.sub("", (s or "").strip())


def init_store(csv_path: str = DEFAULT_CSV_PATH) -> None:
    global _df, _index, _keys_all, _keys_primary, _base_index, _ingredient_index, _base_keys

    if _df is not None:
        return

    with _store_lock:
        if _df is not None:
            return

        csv_path = str(Path(csv_path))
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Drug dataset not found: {csv_path}\n"
                f"Tip: set DRUG_DATASET_PATH env var or ensure the file exists under data/processed/."
            )

        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]

        required = {"drug_id", "generic_name", "generic_name_clean"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        index: Dict[str, Dict[str, Any]] = {}
        primary_keys: set[str] = set()
        all_keys: set[str] = set()

        def _safe_set_key(k: str, row_dict: Dict[str, Any]) -> None:
            if not k:
                return
            existing = index.get(k)
            if not existing:
                index[k] = row_dict
                all_keys.add(k)
                return
            if (existing.get("drug_id") or "") == (row_dict.get("drug_id") or ""):
                all_keys.add(k)
                return
            # collision -> skip
            return

        base_index: Dict[str, List[Dict[str, Any]]] = {}
        ingredient_index: Dict[str, List[Dict[str, Any]]] = {}
        rows: List[Dict[str, Any]] = []

        # ---- Pass 1: generic names and ingredient identity -------------
        # Generics are indexed before brands so that a brand name can never
        # claim a key an ingredient needs. Previously "ibuprofen" was claimed
        # as a BRAND of a diphenhydramine combination product, so looking up
        # ibuprofen returned that combination at match_type="exact",
        # confidence 1.0 — indistinguishable from a correct answer.
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            rows.append(row_dict)

            gen_clean = normalize_text(row_dict.get("generic_name_clean", ""))
            gen_name = normalize_text(row_dict.get("generic_name", ""))

            primary_key = gen_clean or gen_name
            if not primary_key:
                continue

            _safe_set_key(primary_key, row_dict)
            primary_keys.add(primary_key)

            if gen_name:
                _safe_set_key(gen_name, row_dict)
                primary_keys.add(gen_name)

            # Ingredient identity, from the cleanest name available.
            source_name = row_dict.get("generic_name_clean") or row_dict.get("generic_name") or ""
            canonical = ingredient_key(source_name)
            if canonical:
                base_index.setdefault(canonical, []).append(row_dict)
                for ing in ingredient_set(source_name):
                    ingredient_index.setdefault(ing, []).append(row_dict)

        # ---- Pass 2: brand names, only into keys nothing else claimed ---
        for row_dict in rows:
            raw_brands = _strip_brand_suffix(row_dict.get("brand_names") or "")
            if not raw_brands:
                continue
            for b in [b.strip() for b in raw_brands.split(",") if b.strip()][:10]:
                bn = normalize_text(b)
                if not bn or bn in {"...", "more"}:
                    continue
                # A brand that collides with some drug's base ingredient is
                # ambiguous by construction — leave it to the ingredient path.
                if bn in ingredient_index and row_dict not in ingredient_index[bn]:
                    continue
                _safe_set_key(bn, row_dict)

        _df = df
        _index = index
        _keys_primary = sorted(primary_keys)
        _keys_all = sorted(all_keys)
        _base_index = base_index
        _ingredient_index = ingredient_index
        _base_keys = sorted(base_index.keys())

        print(
            f"Loaded {len(df):,} drugs | {len(_keys_all):,} lookup keys "
            f"| {len(_base_index):,} ingredient keys"
        )
        print(f"Dataset path: {os.path.abspath(csv_path)}")


# ---------------------------------------------------------------------
# HELPERS (formatting safety)
# ---------------------------------------------------------------------
def normalize_route(route: str) -> str:
    if not route:
        return ""
    s = str(route).strip()
    if not s:
        return ""
    parts = re.split(r"[,\|/;]+", s)
    cleaned: List[str] = []
    seen: set[str] = set()
    keep_upper = {"iv", "im", "sc", "sq", "po"}

    for p in parts:
        p = p.strip()
        if not p:
            continue
        key = p.lower()
        val = key.upper() if key in keep_upper else p.lower().title()
        dk = val.lower()
        if dk in seen:
            continue
        seen.add(dk)
        cleaned.append(val)

    return ", ".join(cleaned)


def trim_brand_names(raw: str, max_items: int = MAX_BRANDS_RETURNED) -> str:
    if not raw:
        return ""
    raw = _strip_brand_suffix(str(raw))
    text = raw.replace("|", ",").replace(";", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]

    out: List[str] = []
    seen: set[str] = set()

    for p in parts:
        p2 = re.sub(r"\s+", " ", p).strip()
        if not p2:
            continue
        k = p2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p2)
        if len(out) >= max_items:
            break

    total_unique = len({re.sub(r"\s+", " ", x).strip().lower() for x in parts if x.strip()})
    remaining = max(0, total_unique - len(out))
    if remaining > 0:
        return ", ".join(out) + f", ... (+{remaining} more)"
    return ", ".join(out)


def clean_long_text(text: str, *, max_chars: int = 1200) -> str:
    if not text:
        return ""
    t = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return ""
    t = t.replace("\\n", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip() + "…"
    return t


def clean_side_effects(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    def _clean_list(text: str, max_items: int = 80) -> str:
        if not text:
            return ""
        t = re.sub(r"\s+", " ", str(text)).strip()
        if not t:
            return ""
        parts = [p.strip(" .;") for p in t.split(",") if p.strip()]
        out: List[str] = []
        seen: set[str] = set()
        for p in parts:
            k = p.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
            if len(out) >= max_items:
                break
        return ", ".join(out)

    buckets = {
        "common": _clean_list(row.get("common_side_effects", "") or ""),
        "less_common": _clean_list(row.get("less_common_side_effects", "") or ""),
        "rare": _clean_list(row.get("rare_side_effects", "") or ""),
        "postmarketing": _clean_list(row.get("postmarketing_side_effects", "") or ""),
        "unknown": _clean_list(row.get("unknown_frequency_side_effects", "") or ""),
    }
    buckets = {k: v for k, v in buckets.items() if v.strip()}
    return buckets or None


def build_match(row: Dict[str, Any], key: str, score: float) -> Dict[str, Any]:
    warnings_raw = row.get("warnings", "") or ""
    warnings_clean = clean_long_text(warnings_raw)
    
    # Fallback: if warnings are empty, indicate data is not available
    if not warnings_clean or warnings_clean.strip() in ("-", ""):
        warnings_clean = "[Warnings data not available in dataset. Consult the official FDA label or your pharmacist.]"
    
    match: Dict[str, Any] = {
        "match": key,
        "score": float(score),
        "drug_id": row.get("drug_id", "") or "",
        "generic_name": row.get("generic_name", "") or "",
        "generic_name_clean": row.get("generic_name_clean", "") or "",
        "brand_names": trim_brand_names(row.get("brand_names", "") or ""),
        "drug_class": row.get("drug_class", "") or "",
        "route": normalize_route(row.get("route", "") or ""),
        "indications": clean_long_text(row.get("indications", "") or ""),
        "dosage_and_administration": clean_long_text(row.get("dosage_and_administration", "") or ""),
        "warnings": warnings_clean,
        "contraindications": clean_long_text(row.get("contraindications", "") or ""),
        "sources": row.get("sources", "") or "",
        "last_updated": row.get("last_updated", "") or "",
    }

    se = clean_side_effects(row)
    if se:
        match["side_effects_buckets"] = se

    return match


def _suggestion_pool() -> List[str]:
    return list(dict.fromkeys(_keys_primary + list(ALIASES.keys())))


def _clean_suggestion_list(items: List[str], top_k: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    for s in items:
        if not isinstance(s, str):
            continue
        display = re.sub(r"\s+", " ", s).strip()
        key = normalize_text(display)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(display)
        if len(out) >= top_k:
            break

    return out


def suggestions_from_pool(results: List[tuple], top_k: int) -> List[str]:
    raw: List[str] = []
    for key, score, _ in results:
        if isinstance(key, str) and key in ALIASES:
            raw.append(key)
            continue
        row = _index.get(key)
        if not row:
            continue
        name = row.get("generic_name_clean") or row.get("generic_name") or key
        raw.append(str(name))
    return _clean_suggestion_list(raw, top_k)


# ---------------------------------------------------------------------
# LOOKUP
# ---------------------------------------------------------------------
def lookup_drug(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    *,
    disease: Optional[str] = None,
    age: Optional[int] = None,
) -> Dict[str, Any]:
    if _df is None:
        init_store()

    q_original = "" if query is None else str(query)
    q_norm, used_alias, alias_target = apply_alias_only(q_original)
    base_path = [q_original, q_norm] if q_norm else [q_original]

    if not q_norm:
        return make_response(
            query=q_original,
            normalized="",
            message="Empty query",
            matches=[],
            suggestions=[],
            match_type="none",
            resolution_path=base_path,
            disease=disease,
            age=age,
        )

    q_key = ingredient_key(q_norm)
    q_ings = ingredient_set(q_norm)

    def _ingredients_agree(row: Dict[str, Any]) -> bool:
        """The indexed row must actually contain what the user asked for."""
        source = row.get("generic_name_clean") or row.get("generic_name") or ""
        return bool(q_ings) and q_ings.issubset(ingredient_set(source))

    # 1. Exact key hit — accepted only when the ingredients agree. A raw key
    #    hit alone is not identity: brand names and combination products can
    #    own a key that names a different molecule.
    if q_norm in _index and _ingredients_agree(_index[q_norm]):
        row = _index[q_norm]
        matches = [build_match(row, q_norm, 100.0)]
        return make_response(
            query=q_original,
            normalized=q_norm,
            message="OK",
            matches=matches,
            suggestions=[],
            match_type="exact",
            resolution_path=base_path,
            disease=disease,
            age=age,
        )

    if used_alias and alias_target and alias_target in _index:
        row = _index[alias_target]
        alias_cluster = [alias_target] + [k for k, v in ALIASES.items() if v == alias_target]
        alias_cluster = _clean_suggestion_list(alias_cluster, top_k)

        matches = [build_match(row, alias_target, ALIAS_SCORE)]
        return make_response(
            query=q_original,
            normalized=alias_target,
            message="OK",
            matches=matches,
            suggestions=alias_cluster,
            match_type="alias",
            resolution_path=[q_original, q_norm, alias_target],
            disease=disease,
            age=age,
        )

    # 3. Exact ingredient identity, after salt/ester stripping.
    #    "amlodipine" now resolves to "amlodipine besylate" and "atorvastatin"
    #    to "atorvastatin calcium" — previously both were fuzzy guesses. When
    #    several products share the ingredient, prefer the one with the fewest
    #    ingredients so a plain form always beats a combination.
    def _ingredient_count(row: Dict[str, Any]) -> int:
        return len(ingredient_set(row.get("generic_name_clean") or row.get("generic_name") or ""))

    if q_key and q_key in _base_index:
        row = sorted(_base_index[q_key], key=_ingredient_count)[0]
        key = normalize_text(row.get("generic_name_clean") or row.get("generic_name") or q_norm)
        return make_response(
            query=q_original,
            normalized=key or q_norm,
            message="OK",
            matches=[build_match(row, key or q_norm, 100.0)],
            suggestions=[],
            match_type="ingredient",
            resolution_path=[q_original, q_norm, q_key],
            disease=disease,
            age=age,
        )

    # 4. The ingredient is known, but only inside combination products.
    #    "ibuprofen" is genuinely absent from this dataset as a standalone
    #    product — it exists only combined with famotidine, diphenhydramine
    #    or phenylephrine. Saying that is safer than answering with one of
    #    those combinations as though it were plain ibuprofen.
    if len(q_ings) == 1:
        only = next(iter(q_ings))
        combos = _ingredient_index.get(only) or []
        if combos:
            ordered = sorted(combos, key=_ingredient_count)[:top_k]
            matches = [
                build_match(
                    r,
                    normalize_text(r.get("generic_name_clean") or r.get("generic_name") or only),
                    80.0,
                )
                for r in ordered
            ]
            names = [str(r.get("generic_name_clean") or r.get("generic_name") or "") for r in ordered]
            return make_response(
                query=q_original,
                normalized=only,
                message="Only combination products contain this ingredient",
                matches=matches,
                suggestions=_clean_suggestion_list(names, top_k),
                match_type="combination_only",
                resolution_path=[q_original, q_norm, only],
                disease=disease,
                age=age,
                needs_confirmation=True,
                identity_note=(
                    f"'{only}' is not held as a single-ingredient product. The results "
                    f"below are combination products that contain it — check the exact "
                    f"product named on your prescription before relying on this."
                ),
            )

    # 5. No ingredient identity could be established. From here fuzzy matching
    #    may only SUGGEST. It must never answer: on this dataset
    #    "lisinopril" scores 80 against "fosinopril" and would otherwise be
    #    served as a confident answer for a completely different molecule.
    candidate_pool = _base_keys or _keys_primary
    ranked = process.extract(
        q_key or q_norm,
        candidate_pool,
        scorer=fuzz.ratio,
        limit=max(top_k, 15),
        score_cutoff=45.0,
    )

    top_score = float(ranked[0][1]) if ranked else 0.0
    runner_up = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    unambiguous = (top_score - runner_up) >= AMBIGUITY_MARGIN

    # A near-certain typo, and nothing else close: return the match but keep
    # it flagged so the caller asks for confirmation.
    if ranked and top_score >= INGREDIENT_TYPO_RATIO and unambiguous:
        best_key = ranked[0][0]
        row = sorted(_base_index.get(best_key, []), key=_ingredient_count)
        if row:
            chosen = row[0]
            key = normalize_text(
                chosen.get("generic_name_clean") or chosen.get("generic_name") or best_key
            )
            return make_response(
                query=q_original,
                normalized=key or best_key,
                message="Probable spelling variant — confirm before use",
                matches=[build_match(chosen, key or best_key, min(top_score, 84.0))],
                suggestions=_clean_suggestion_list([best_key], top_k),
                match_type="probable_typo",
                resolution_path=[q_original, q_norm, best_key],
                disease=disease,
                age=age,
                needs_confirmation=True,
                identity_note=(
                    f"No exact match for '{q_norm}'. This looks like a misspelling of "
                    f"'{best_key}', but spelling alone cannot confirm a medicine — check "
                    f"the name on your prescription."
                ),
            )

    # 6. Not confident enough to name a drug. Suggest, and return no data.
    suggestions = _clean_suggestion_list([k for k, _s, _i in ranked], top_k)
    return make_response(
        query=q_original,
        normalized=q_norm,
        message="No confident match found",
        matches=[],
        suggestions=suggestions,
        match_type="none",
        resolution_path=base_path,
        disease=disease,
        age=age,
        identity_note=(
            f"'{q_norm}' is not in the knowledge base. Similar names are listed as "
            f"suggestions, but they may be different medicines — do not assume a match."
            if suggestions else ""
        ),
    )
