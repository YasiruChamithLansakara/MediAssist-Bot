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

SUPPORTED_DISEASES = [
    "diabetes",
    "hypertension",
    "asthma",
    "heart disease",
    "arthritis",
]


# ---------------------------------------------------------------------
# INTERNAL STORE
# ---------------------------------------------------------------------
_df: pd.DataFrame | None = None
_index: Dict[str, Dict[str, Any]] = {}
_keys_all: List[str] = []
_keys_primary: List[str] = []
_store_lock = threading.Lock()


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
    }
    kws = disease_keywords.get(d, [])

    if kws and any(k in blob for k in kws):
        highlights["disease"].append(
            f"Label includes terms related to {d}. Review warnings/contraindications for condition-specific cautions."
        )

    # Extra helpful rule: NSAID + CV language for hypertension/heart disease
    if d in {"hypertension", "heart disease"}:
        if ("nsaid" in drug_class or "nonsteroidal" in blob) and _contains(blob, "cardiovascular", "thrombotic", "stroke", "myocardial"):
            highlights["disease"].append(
                "Cardiovascular risk language detected (often relevant to some pain/anti-inflammatory medicines) — verify appropriateness for this condition."
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
    }

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
    global _df, _index, _keys_all, _keys_primary

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

        for _, row in df.iterrows():
            row_dict = row.to_dict()

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

            # brand indexing: split + ignore "...(+X more)" suffix
            raw_brands = _strip_brand_suffix(row_dict.get("brand_names") or "")
            if raw_brands:
                brand_parts = [b.strip() for b in raw_brands.split(",") if b.strip()]
                for b in brand_parts[:10]:
                    bn = normalize_text(b)
                    if bn and bn not in {"...", "more"}:
                        _safe_set_key(bn, row_dict)

        _df = df
        _index = index
        _keys_primary = sorted(primary_keys)
        _keys_all = sorted(all_keys)

        print(f"Loaded {len(df):,} drugs | {len(_keys_all):,} lookup keys")
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

    if q_norm in _index:
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

    results = process.extract(
        q_norm,
        _keys_all,
        scorer=fuzz.WRatio,
        limit=max(top_k, 10),
        score_cutoff=min_score,
    )

    if not results:
        loose = process.extract(
            q_norm,
            _suggestion_pool(),
            scorer=fuzz.WRatio,
            limit=max(top_k, 15),
            score_cutoff=max(0.0, min_score - 35),
        )
        suggestions = suggestions_from_pool(loose, top_k)
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
        )

    matches: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for key, score, _ in results:
        row = _index.get(key)
        if not row:
            continue

        drug_id = row.get("drug_id", "") or ""
        if drug_id and drug_id in seen_ids:
            continue
        if drug_id:
            seen_ids.add(drug_id)

        matches.append(build_match(row, key, float(score)))
        if len(matches) >= top_k:
            break

    if not matches:
        return make_response(
            query=q_original,
            normalized=q_norm,
            message="No confident match found",
            matches=[],
            suggestions=[],
            match_type="none",
            resolution_path=base_path,
            disease=disease,
            age=age,
        )

    top_score = float(matches[0].get("score", 0.0))
    match_type = "fuzzy" if top_score < 99.9 else "exact"

    suggest_candidates = process.extract(
        q_norm,
        _suggestion_pool(),
        scorer=fuzz.WRatio,
        limit=max(top_k, 15),
        score_cutoff=max(0.0, min_score - 35),
    )
    suggestions = suggestions_from_pool(suggest_candidates, top_k)

    normalized_out = matches[0].get("generic_name_clean") or q_norm

    return make_response(
        query=q_original,
        normalized=normalized_out,
        message="OK",
        matches=matches,
        suggestions=suggestions,
        match_type=match_type,
        resolution_path=base_path,
        disease=disease,
        age=age,
    )
