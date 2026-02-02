from __future__ import annotations

import os
import re
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from rapidfuzz import process, fuzz

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DEFAULT_CSV_PATH = os.getenv(
    "DRUG_DATASET_PATH",
    # ✅ cleaned dataset is now the default
    os.path.join("data", "processed", "drug_knowledge_bot_ready_clean.csv"),
)

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 80.0
MAX_BRANDS_RETURNED = 20

ALIASES = {
    "paracetamol": "acetaminophen",
    "panadol": "acetaminophen",
    "tylenol": "acetaminophen",
    "apap": "acetaminophen",
}

ALIAS_SCORE = 95.0

# Status thresholds (0..1 confidence)
OK_THRESHOLD = 0.85
LOW_THRESHOLD = 0.60

# ✅ Paracetamol-family typo threshold (high to avoid false positives)
PARACETAMOL_TYPO_THRESHOLD = 85

# -----------------------------------------------------------------------------
# INTERNAL STORE
# -----------------------------------------------------------------------------
_df: pd.DataFrame | None = None
_index: Dict[str, Dict[str, Any]] = {}
_keys_all: List[str] = []
_keys_primary: List[str] = []

# -----------------------------------------------------------------------------
# NORMALIZATION + DOSAGE STRIPPING
# -----------------------------------------------------------------------------
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
    """
    Returns: (normalized_query, used_alias, alias_target)

    ✅ Includes safe typo rule for paracetamol-family.
    """
    qn = normalize_text(q)

    # Direct alias
    if qn in ALIASES:
        return qn, True, ALIASES[qn]

    # ✅ Extra-safe paracetamol-family typo handling
    if qn:
        score = fuzz.WRatio(qn, "paracetamol")

        # Guard: apply only if query looks like paracetamol-family
        looks_like_para = (
            "paracet" in qn
            or qn.startswith("par")
            or "para" in qn
        )

        if looks_like_para and score >= PARACETAMOL_TYPO_THRESHOLD:
            return qn, True, "acetaminophen"

    return qn, False, None


# -----------------------------------------------------------------------------
# BACKEND RESPONSE STANDARDIZATION
# -----------------------------------------------------------------------------
def score_to_confidence(score: float) -> float:
    """Convert 0..100 score to 0..1 confidence (rounded)."""
    s = float(score or 0.0)
    s = max(0.0, min(100.0, s))
    return round(s / 100.0, 4)


def classify_status(confidence: float, has_best: bool) -> str:
    """
    status:
      - ok: confidence >= 0.85
      - low_confidence: 0.60..0.85
      - no_match: < 0.60 OR no best match
    """
    if has_best and confidence >= OK_THRESHOLD:
        return "ok"
    if has_best and confidence >= LOW_THRESHOLD:
        return "low_confidence"
    return "no_match"


def _dedupe_path(items: List[str]) -> List[str]:
    """Remove duplicates but keep order (cleaner resolution_path)."""
    out: List[str] = []
    seen: set[str] = set()

    for x in items:
        raw = str(x)
        key = normalize_text(raw)  # ✅ stable dedupe key
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)

    return out


def make_response(
    *,
    query: str,
    normalized: str,
    message: str,
    matches: List[Dict[str, Any]],
    suggestions: List[str],
    match_type: str,
    resolution_path: List[str],
) -> Dict[str, Any]:
    best_match = matches[0] if matches else None
    best_score = float(best_match.get("score", 0.0)) if best_match else 0.0
    confidence = score_to_confidence(best_score)
    status = classify_status(confidence, bool(best_match))

    return {
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
        "resolution_path": _dedupe_path(resolution_path),
    }


# -----------------------------------------------------------------------------
# LOAD + CACHE
# -----------------------------------------------------------------------------
def init_store(csv_path: str = DEFAULT_CSV_PATH) -> None:
    global _df, _index, _keys_all, _keys_primary

    if _df is not None:
        return

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

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        gen_clean = normalize_text(row_dict.get("generic_name_clean", ""))
        gen_name = normalize_text(row_dict.get("generic_name", ""))

        primary_key = gen_clean or gen_name
        if not primary_key:
            continue

        index.setdefault(primary_key, row_dict)
        primary_keys.add(primary_key)
        all_keys.add(primary_key)

        if gen_name:
            index.setdefault(gen_name, row_dict)
            primary_keys.add(gen_name)
            all_keys.add(gen_name)

        # keep short brand key recall
        brand_names_norm = normalize_text(row_dict.get("brand_names", ""))
        if brand_names_norm:
            short_brand = " ".join(brand_names_norm.split()[:3])
            if short_brand:
                index.setdefault(short_brand, row_dict)
                all_keys.add(short_brand)

    _df = df
    _index = index
    _keys_primary = sorted(primary_keys)
    _keys_all = sorted(all_keys)

    print(f"✅ Loaded {len(df):,} drugs | {len(_keys_all):,} lookup keys")
    print(f"📄 Dataset path: {os.path.abspath(csv_path)}")


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def clean_side_effects(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    # Keep lightweight cleanup: whitespace + dedupe list items
    def _clean_list(text: str, max_items: int = 80) -> str:
        if not text:
            return ""
        t = str(text).replace("\r\n", "\n").replace("\r", "\n")
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            return ""

        parts = [p.strip(" .;") for p in t.split(",") if p.strip()]
        if not parts:
            return t

        out: List[str] = []
        seen: set[str] = set()
        for p in parts:
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
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


def normalize_route(route: str) -> str:
    # Lightweight normalization (should already be clean in CSV, but safe)
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
    # Lightweight trim (CSV should be pre-cleaned, but keep safety)
    if not raw:
        return ""
    text = str(raw).replace("|", ",").replace(";", ",")
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


def clean_long_text(
    text: str,
    *,
    max_chars: int = 1500,
    min_line_len: int = 20,
) -> str:
    """
    CSV is pre-cleaned, but we still do TWO important things here:

    1) ✅ Preserve newlines (so FDA sections stay readable in UI)
    2) ✅ Remove duplicated section labels like:
         "Liver Warning:\\nLiver Warning This product..." -> "Liver Warning:\\nThis product..."

    Also:
    - remove stray lines that are only ":" or empty
    - trim leading punctuation/spaces in lines
    - hard cap length
    """
    if not text:
        return ""

    t = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return ""

    # If the CSV contains literal "\n" sequences (two chars), convert them to real newlines
    t = t.replace("\\n", "\n")

    # Normalize spaces/tabs but keep newlines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # Remove lines that are only ":" or only punctuation
    lines = [ln.strip() for ln in t.split("\n")]
    cleaned_lines: List[str] = []
    for ln in lines:
        if not ln:
            cleaned_lines.append("")  # keep blank lines for paragraph spacing
            continue
        if re.fullmatch(r"[:;\-–—•\.\s]+", ln):
            continue
        # Trim weird leading punctuation like ": text"
        ln = re.sub(r"^\s*[:;\-–—•]+\s*", "", ln).strip()
        if ln:
            cleaned_lines.append(ln)

    t = "\n".join(cleaned_lines)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # Remove duplicated section label at the start of the next line after "Label:"
    # e.g.
    # Liver Warning:
    # Liver Warning This product contains ...
    # ->
    # Liver Warning:
    # This product contains ...
    label_map = [
        "Liver Warning",
        "Allergy Alert",
        "Do not use",
        "Ask a doctor before use",
        "Ask a doctor or pharmacist before use",
        "Stop use and ask a doctor if",
        "Stop use and ask a doctor",
    ]
    for label in sorted(label_map, key=len, reverse=True):
        # If a line is "LABEL:" then next line starts with "LABEL" again, remove it.
        pat = rf"(?mi)^(?:{re.escape(label)}:)\s*\n\s*(?:{re.escape(label)}\b[: ]*)"
        t = re.sub(pat, f"{label}:\n", t)

    # Also remove repeated "Label:" on its own line twice in a row
    for label in sorted(label_map, key=len, reverse=True):
        pat2 = rf"(?mi)^(?:{re.escape(label)}:)\s*\n\s*(?:{re.escape(label)}:)\s*$"
        t = re.sub(pat2, f"{label}:", t)

    # Final: collapse excessive internal spaces again (but keep newlines)
    t = "\n".join([re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]).strip()
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # Safety: drop super-short non-empty lines (noise), but keep blank lines
    final_lines: List[str] = []
    for ln in t.split("\n"):
        if not ln:
            final_lines.append("")
            continue
        if len(ln) < min_line_len and not ln.endswith(":"):
            # very short line that's not a section header -> drop
            continue
        final_lines.append(ln)

    t = "\n".join(final_lines)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # Hard cap (preserve structure as much as possible)
    if len(t) > max_chars:
        t = t[:max_chars].rstrip() + "…"

    return t


def build_match(row: Dict[str, Any], key: str, score: float) -> Dict[str, Any]:
    match: Dict[str, Any] = {
        "match": key,
        "score": float(score),
        "drug_id": row.get("drug_id", "") or "",
        "generic_name": row.get("generic_name", "") or "",
        "generic_name_clean": row.get("generic_name_clean", "") or "",
        "brand_names": trim_brand_names(row.get("brand_names", "") or ""),
        "drug_class": row.get("drug_class", "") or "",
        "route": normalize_route(row.get("route", "") or ""),

        # ✅ Trust cleaned CSV, but keep safe formatting + caps
        "indications": clean_long_text(row.get("indications", "")),
        "dosage_and_administration": clean_long_text(row.get("dosage_and_administration", "")),
        "warnings": clean_long_text(row.get("warnings", "")),
        "contraindications": clean_long_text(row.get("contraindications", "")),

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
        s2 = normalize_text(s)
        if not s2 or s2 in seen:
            continue
        seen.add(s2)
        out.append(s2)
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


# -----------------------------------------------------------------------------
# LOOKUP
# -----------------------------------------------------------------------------
def lookup_drug(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Dict[str, Any]:
    if _df is None:
        init_store()

    q_original = query
    q_norm, used_alias, alias_target = apply_alias_only(query)

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
        )

    # Alias hit (includes typo → acetaminophen)
    if used_alias and alias_target and alias_target in _index:
        row = _index[alias_target]

        alias_cluster = [alias_target]
        for k, v in ALIASES.items():
            if v == alias_target:
                alias_cluster.append(k)

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
    )
