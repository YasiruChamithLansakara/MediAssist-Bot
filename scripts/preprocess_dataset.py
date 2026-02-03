from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple
from difflib import SequenceMatcher

import pandas as pd


# -------------------------
# Cleaning helpers
# -------------------------

def fix_known_fda_truncations(text: str) -> str:
    """
    Fix known openFDA truncation + encoding (mojibake) artifacts BEFORE any dedupe/splitting.
    This must run first for warnings/indications/etc.
    """
    if not text:
        return ""

    t = str(text)

    # ---- Known truncation/missing-letter glitches ----
    # "cetaminophen" -> "acetaminophen"
    t = re.sub(r"\bcetaminophen\b", "acetaminophen", t, flags=re.IGNORECASE)
    t = re.sub(r"\bcetamino\b", "acetaminophen", t, flags=re.IGNORECASE)

    # ---- Mojibake cleanup (UTF-8 read as Windows-1252) ----
    replacements = {
        "â€¦": "...",   # ellipsis
        "â¦": "...",    # ellipsis fragment
        "â€¯": " ",     # narrow no-break space
        "Â": " ",       # stray
        "\u00a0": " ",  # non-breaking space
        "â€“": "-",     # en dash
        "â€”": "-",     # em dash
        "â€˜": "'",     # left single quote
        "â€™": "'",     # right single quote
        "â€œ": '"',     # left double quote
        "â€ ": '"',     # right double quote (common broken token)
        "â€¢": "•",     # bullet
    }
    for bad, good in replacements.items():
        t = t.replace(bad, good)

    # Fix the exact artifact you saw: "câ¦"
    t = t.replace("câ¦", "...")

    # Any remaining "âX" sequences are garbage — collapse them safely.
    t = re.sub(r"â.{1,2}", "...", t)

    # Final whitespace normalize (keep newlines if any)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n\s+", "\n", t)
    return t.strip()


def normalize_route(route: str) -> str:
    """
    Normalize route strings:
    - supports commas/pipes/slashes/semicolons
    - dedupes
    - keeps common abbreviations uppercase
    """
    if not route:
        return ""

    s = str(route).strip()
    if not s:
        return ""

    parts = re.split(r"[,\|/;]+", s)
    keep_upper = {"iv", "im", "sc", "sq", "po"}
    out: List[str] = []
    seen: set[str] = set()

    for p in parts:
        p = p.strip()
        if not p:
            continue

        k = p.lower()
        if k in keep_upper:
            val = k.upper()
        else:
            val = " ".join(w.capitalize() for w in k.split())

        dk = val.lower()
        if dk in seen:
            continue
        seen.add(dk)
        out.append(val)

    return ", ".join(out)


def trim_brand_names(raw: str, max_items: int = 20) -> str:
    """
    - supports separators: comma, pipe, semicolon
    - case-insensitive dedupe
    - normalizes internal whitespace
    - caps to max_items and shows remaining count
    """
    if not raw:
        return ""

    s = str(raw)
    s = s.replace("|", ",")
    s = re.sub(r"\s*;\s*", ",", s)

    parts = [re.sub(r"\s+", " ", p).strip() for p in s.split(",")]
    parts = [p for p in parts if p]

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

    if not out:
        return ""

    total_unique = len({p.lower() for p in parts})
    remaining = max(0, total_unique - len(out))

    if remaining > 0:
        return ", ".join(out) + f", ... (+{remaining} more)"
    return ", ".join(out)


def clean_side_effects_text(text: str, max_items: int = 120) -> str:
    """
    Side effects are usually comma-separated lists.
    - normalizes separators
    - dedupes items
    - caps number of items
    """
    if not text:
        return ""

    s = fix_known_fda_truncations(text)
    s = str(s).replace("|", ",")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    items = [x.strip(" .;") for x in s.split(",") if x.strip()]
    out: List[str] = []
    seen: set[str] = set()

    for it in items:
        key = re.sub(r"\s+", " ", it).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= max_items:
            break

    return ", ".join(out)


# -------------------------
# Strong dedupe / caps (warnings + long text)
# -------------------------

def _normalize_for_dupe(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def _hard_cap(s: str, max_chars: int) -> str:
    """
    Guarantees len(result) <= max_chars (including ellipsis).
    """
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    if max_chars <= 1:
        return "…"
    return s[: max_chars - 1].rstrip() + "…"


def _cut_repeated_prefix(raw: str) -> str:
    """
    Handles the classic 'big block repeated twice' even if the second copy is slightly different.
    """
    s = (raw or "").strip()
    if len(s) < 350:
        return s

    low = s.lower()
    snippet = re.sub(r"\s+", " ", low[:220]).strip()
    snippet = re.sub(r"[^a-z0-9 ]+", "", snippet)[:180].strip()
    if len(snippet) < 80:
        return s

    idx = low.find(snippet, 120)
    if idx == -1:
        return s

    if 0 < idx < int(len(s) * 0.75):
        cut = s[:idx].strip(" .")
        if len(cut) > 120:
            return cut

    return s


def _near_duplicate(a: str, b: str, threshold: float = 0.92) -> bool:
    na = _normalize_for_dupe(a)
    nb = _normalize_for_dupe(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


# unit split + unit dedupe (OTC starter phrases)
_UNIT_STARTERS_RE = re.compile(
    r"(?i)(?<!^)(?<!\n)\s*(?=("
    r"keep out of reach of children"
    r"|keep away from fire"
    r"|for external use only"
    r"|warning\b"
    r"|warnings\b"
    r"|precautions\b"
    r"|do not use\b"
    r"|when using this product\b"
    r"|ask a doctor before use\b"
    r"|ask a doctor or pharmacist before use\b"
    r"|stop use and ask (?:a )?doctor\b"
    r"|if swallowed\b"
    r"|in case of overdose\b"
    r"|poison control\b"
    r"))"
)


def _split_and_dedupe_units(text: str) -> str:
    """
    Inserts line breaks before common OTC starter phrases, splits into units,
    and removes repeated units (exact or near-duplicates).
    """
    if not text:
        return ""

    s = str(text).strip()
    if not s:
        return ""

    s = _UNIT_STARTERS_RE.sub("\n", s)
    units = [u.strip(" \t•-") for u in re.split(r"\n+", s) if u.strip()]
    if len(units) <= 1:
        return s.strip()

    out: List[str] = []
    seen: set[str] = set()

    for u in units:
        key = _normalize_for_dupe(u)
        if not key:
            continue
        if key in seen:
            continue
        if out and any(_near_duplicate(u, prev) for prev in out[-4:]):
            continue
        seen.add(key)
        out.append(u)

    return "\n".join(out).strip()


def collapse_duplicate_blocks(text: str) -> str:
    """
    Final pass to catch remaining repeated long-block cases.
    """
    if not text:
        return ""

    raw = str(text).strip()
    if not raw:
        return ""

    raw = _cut_repeated_prefix(raw)

    blocks = [b.strip() for b in re.split(r"\n{2,}", raw) if b.strip()]
    out_blocks: List[str] = []
    seen: set[str] = set()

    for b in blocks:
        key = _normalize_for_dupe(b)
        if not key or key in seen:
            continue
        seen.add(key)
        out_blocks.append(b)

    text2 = "\n\n".join(out_blocks).strip()

    # A + A detector
    norm = _normalize_for_dupe(text2)
    if len(norm) >= 320:
        mid = len(norm) // 2
        left = norm[:mid].strip()
        right = norm[mid:].strip()
        sim = SequenceMatcher(None, left, right).ratio()
        if sim >= 0.94:
            half = text2[: len(text2) // 2].strip()
            if len(half) > 120:
                text2 = half

    text2 = _split_and_dedupe_units(text2)
    text2 = _cut_repeated_prefix(text2)

    return text2.strip()


def cap_sections(formatted: str, max_chars: int = 1200) -> str:
    """
    Caps each section so warnings don't exceed max_chars.
    Input format expected:
      Title:
      content...

      Title2:
      content...
    """
    if not formatted:
        return ""

    caps = {
        "Liver Warning": 280,
        "Allergy Alert": 260,
        "Do not use": 360,
        "Ask a doctor before use": 260,
        "Ask a doctor or pharmacist before use": 260,
        "Stop use and ask a doctor if": 520,
        "Pregnancy / Breastfeeding": 220,

        "Boxed Warning": 520,
        "Warnings": 520,
        "Warnings And Precautions": 520,
        "Precautions": 420,
        "Adverse Reactions": 420,
        "Drug Interactions": 320,
        "Use In Specific Populations": 320,
        "Pregnancy": 220,
        "Lactation": 220,
        "Overdosage": 280,
        "Dosage And Administration": 320,
        "Contraindications": 320,
    }

    blocks = [b.strip() for b in re.split(r"\n{2,}", formatted.strip()) if b.strip()]
    new_blocks: List[str] = []

    for b in blocks:
        if ":\n" not in b:
            new_blocks.append(_hard_cap(b, 420))
            continue

        title, body = b.split(":\n", 1)
        title = title.strip()
        body = body.strip()
        cap = caps.get(title, 420)

        if "•" in body:
            lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
            kept: List[str] = []
            total = 0
            for ln in lines:
                if total + len(ln) + 1 > cap:
                    break
                kept.append(ln)
                total += len(ln) + 1
            body2 = "\n".join(kept).strip()
            if not body2:
                body2 = _hard_cap(body, cap)
        else:
            body2 = _hard_cap(body, cap)

        new_blocks.append(f"{title}:\n{body2}")

    out = "\n\n".join(new_blocks).strip()
    return _hard_cap(out, max_chars)


def drop_duplicate_section_bodies(formatted: str, threshold: float = 0.90) -> str:
    """
    Removes sections whose BODY is a near-duplicate of another section body.
    Keeps the longest version when duplicates exist.

    Upgrade:
    - Full similarity (ratio)
    - Subsumption similarity (shorter contained in longer)
    - OVERLAP/COVERAGE similarity for truncated fragments (like leucovorin calcium)
      -> especially for (Warnings <-> Overdosage)
    """
    if not formatted:
        return ""

    blocks = [b.strip() for b in re.split(r"\n{2,}", formatted.strip()) if b.strip()]

    parsed: List[Tuple[str, str]] = []
    for b in blocks:
        if ":\n" not in b:
            parsed.append(("", b.strip()))
            continue
        title, body = b.split(":\n", 1)
        parsed.append((title.strip(), body.strip()))

    def _subsumption_ratio(a: str, b: str) -> float:
        na = _normalize_for_dupe(a)
        nb = _normalize_for_dupe(b)
        if not na or not nb:
            return 0.0

        short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
        if len(short) < 80:
            return 0.0

        sm = SequenceMatcher(None, short, long)
        matched = sum(bl.size for bl in sm.get_matching_blocks())
        return matched / max(1, len(short))

    def _overlap_coverage(a: str, b: str) -> float:
        """
        Measures how much of the SHORTER string overlaps with the LONGER string,
        even if neither fully contains the other (truncated fragments).

        We compute coverage = (sum of matching blocks) / (len(shorter)).
        This is similar to subsumption, but works better when both are fragments.
        """
        na = _normalize_for_dupe(a)
        nb = _normalize_for_dupe(b)
        if not na or not nb:
            return 0.0

        short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
        if len(short) < 80:
            return 0.0

        sm = SequenceMatcher(None, short, long)
        matched = sum(bl.size for bl in sm.get_matching_blocks())
        return matched / max(1, len(short))

    def _is_dup(title_a: str, body_a: str, title_b: str, body_b: str) -> bool:
        na = _normalize_for_dupe(body_a)
        nb = _normalize_for_dupe(body_b)
        if not na or not nb:
            return False

        # 1) Full similarity
        full = SequenceMatcher(None, na, nb).ratio()
        if full >= threshold:
            return True

        # 2) Subsumption (works for "same text twice" cases)
        sub = _subsumption_ratio(body_a, body_b)
        if sub >= 0.78:
            return True

        # 3) Overlap/Coverage (works for truncated fragment duplicates)
        cov = _overlap_coverage(body_a, body_b)

        # Apply a LOWER coverage threshold only when headings are likely duplicates.
        # This targets your exact leucovorin calcium case safely.
        ta = title_a.strip().lower()
        tb = title_b.strip().lower()
        pair = {ta, tb}

        if pair == {"warnings", "overdosage"}:
            # These are often truncated fragments of the same paragraph.
            # Lower threshold + require a decent longest common match.
            na = _normalize_for_dupe(body_a)
            nb = _normalize_for_dupe(body_b)

            short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
            sm = SequenceMatcher(None, short, long)
            longest = sm.find_longest_match(0, len(short), 0, len(long)).size

            # Require at least ~60 chars of continuous overlap
            if longest < 60:
                return False

            # Coverage threshold lowered only for this pair
            return cov >= 0.35

        # For other headings, keep stricter
        return cov >= 0.75

    kept: List[Tuple[str, str]] = []

    for title, body in parsed:
        if not title:
            kept.append((title, body))
            continue

        dup_idx = -1
        for i, (kt, kb) in enumerate(kept):
            if not kt:
                continue
            if _is_dup(title, body, kt, kb):
                dup_idx = i
                break

        if dup_idx == -1:
            kept.append((title, body))
        else:
            # keep the longer body version
            kt, kb = kept[dup_idx]
            if len(body) > len(kb):
                kept[dup_idx] = (title, body)

    out_blocks: List[str] = []
    for title, body in kept:
        if title:
            out_blocks.append(f"{title}:\n{body}")
        else:
            out_blocks.append(body)

    return "\n\n".join(out_blocks).strip()

# -------------------------
# openFDA heading fallback (Rx-ish labels)
# -------------------------

def clean_openfda_heading_blocks(text: str, max_chars: int = 1200) -> str:
    """
    Fallback cleaner for warnings when OTC anchors aren't found.
    Uses openFDA-style headings like WARNINGS/PRECAUTIONS/etc.
    """
    if not text:
        return ""

    t = fix_known_fda_truncations(text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return ""

    headings = [
        "boxed warning",
        "warnings and precautions",
        "warnings",
        "precautions",
        "contraindications",
        "adverse reactions",
        "drug interactions",
        "use in specific populations",
        "pregnancy",
        "lactation",
        "overdosage",
        "dosage and administration",
    ]

    heading_re = re.compile(
        r"(?i)\b(" + "|".join(re.escape(h) for h in headings) + r")\b\s*[:]?[\s]*"
    )

    matches = list(heading_re.finditer(t))
    if not matches:
        flat = re.sub(r"\s+", " ", t).strip()
        flat = collapse_duplicate_blocks(flat)
        return _hard_cap(flat, max_chars)

    blocks: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        chunk = t[start:end].strip()
        blocks.append((title, chunk))

    best: dict[str, str] = {}
    for title, chunk in blocks:
        k = title.lower()
        if len(chunk) > len(best.get(k, "")):
            best[k] = chunk

    def dedupe_sentences(title: str, chunk: str) -> List[str]:
        # remove duplicated heading token at start
        chunk = re.sub(rf"(?i)^\s*{re.escape(title)}\s*:?\s*[,.\-]*\s*", "", chunk).strip()
        chunk = _cut_repeated_prefix(chunk)

        parts = re.split(r"(?<=[.!?])\s+|\n+", chunk)
        out: List[str] = []
        seen: set[str] = set()

        for p in parts:
            p = p.strip(" .")
            if len(p) < 20:
                continue

            key = _normalize_for_dupe(p)
            if not key or key in seen:
                continue

            if out and any(_near_duplicate(p, prev) for prev in out[-3:]):
                continue

            seen.add(key)
            out.append(p + ".")
        return out

    formatted_blocks: List[str] = []
    for h in headings:
        chunk = best.get(h.lower())
        if not chunk:
            continue

        lines = dedupe_sentences(h, chunk)
        body = " ".join(lines).strip()
        if not body:
            continue

        formatted_blocks.append(f"{h.title()}:\n{body}")

    out = "\n\n".join(formatted_blocks).strip()
    out = collapse_duplicate_blocks(out)

    # ✅ Key fix for leucovorin calcium: remove duplicate bodies across headings
    out = drop_duplicate_section_bodies(out, threshold=0.90)

    out = cap_sections(out, max_chars=max_chars)
    return out


# -------------------------
# OTC-style sections + fallback
# -------------------------

def clean_fda_warnings_section_level(text: str, max_chars: int = 1200) -> str:
    """
    OTC-style FDA warnings cleaner with SECTION-LEVEL formatting.
    If OTC anchors aren't found, falls back to openFDA heading-block cleaner.
    """
    if not text:
        return ""

    t = fix_known_fda_truncations(text)

    # Normalize whitespace / bullets
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("■", " ").replace("▪", " ").replace("•", " ")
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"\s*-\s*", " ", t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    if not t:
        return ""

    sections = [
        ("Liver Warning", r"\bliver warning\b"),
        ("Allergy Alert", r"\ballergy alert\b"),
        ("Do not use", r"\bdo not use\b"),
        ("Ask a doctor before use", r"\bask a doctor before use\b"),
        ("Ask a doctor or pharmacist before use", r"\bask a doctor or pharmacist before use\b"),
        ("Stop use and ask a doctor if", r"\bstop use\b.*?\bask a doctor\b"),
        ("Pregnancy / Breastfeeding", r"\bif pregnant\b|\bpregnan|\bbreast[\s-]?feed\b"),
    ]

    hits: list[tuple[int, str]] = []
    for title, pat in sections:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            hits.append((m.start(), title))

    # fallback if no OTC anchors
    if not hits:
        return clean_openfda_heading_blocks(t, max_chars=max_chars)

    hits.sort(key=lambda x: x[0])

    extracted: dict[str, str] = {}
    for i, (start, title) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(t)
        chunk = t[start:end].strip(" .")
        if len(chunk) > len(extracted.get(title, "")):
            extracted[title] = chunk

    def dedupe_sentences(title: str, chunk: str) -> List[str]:
        chunk = re.sub(rf"(?i)^\s*{re.escape(title)}\s*:?\s*", "", chunk).strip()
        chunk = _cut_repeated_prefix(chunk)

        parts = re.split(r"(?<=[.!?])\s+", chunk)
        seen = set()
        out: List[str] = []

        for p in parts:
            p = p.strip(" .")
            if len(p) < 20:
                continue

            k = _normalize_for_dupe(p)
            if not k or k in seen:
                continue

            if out and any(_near_duplicate(p, prev) for prev in out[-3:]):
                continue

            seen.add(k)
            out.append(p)
        return out

    blocks: List[str] = []
    for title, _ in sections:
        content = extracted.get(title)
        if not content:
            continue

        lines = dedupe_sentences(title, content)

        if title.startswith("Stop use"):
            formatted = "\n".join(f"• {l}" for l in lines)
        else:
            formatted = " ".join(lines)

        blocks.append(f"{title}:\n{formatted}")

    out = "\n\n".join(blocks).strip()
    out = collapse_duplicate_blocks(out)

    # ✅ also remove duplicate bodies across OTC sections (rare but safe)
    out = drop_duplicate_section_bodies(out, threshold=0.90)

    out = cap_sections(out, max_chars=max_chars)
    return out


def clean_long_text(text: str, *, max_chars: int = 1200) -> str:
    """
    For cleaned dataset: light cap + whitespace normalize,
    plus truncation/encoding fix + repeat killer.
    """
    if not text:
        return ""

    t = fix_known_fda_truncations(text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("■", " ").replace("▪", " ").replace("•", " ")
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # repeat killer helps indications/dosage/contra too
    t = collapse_duplicate_blocks(t)
    return _hard_cap(t, max_chars)


# -------------------------
# Main
# -------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    in_path = os.getenv(
        "DRUG_DATASET_IN",
        str(project_root / "data" / "processed" / "drug_knowledge_bot_ready_final.csv"),
    )

    out_path = os.getenv(
        "DRUG_DATASET_OUT",
        str(project_root / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv"),
    )

    in_path_p = Path(in_path)
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    if not in_path_p.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path_p}")

    df = pd.read_csv(
        in_path_p,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
    df.columns = [c.strip() for c in df.columns]

    # Long text columns
    long_cols = ["indications", "dosage_and_administration", "contraindications"]
    for col in long_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_long_text(x, max_chars=1200))

    # Warnings
    if "warnings" in df.columns:
        df["warnings"] = df["warnings"].apply(
            lambda x: clean_fda_warnings_section_level(x, max_chars=1200)
        )

    # Brand names
    if "brand_names" in df.columns:
        df["brand_names"] = df["brand_names"].apply(
            lambda x: trim_brand_names(x, max_items=20)
        )

    # Route
    if "route" in df.columns:
        df["route"] = df["route"].apply(normalize_route)

    # Side effects buckets
    se_cols = [
        "common_side_effects",
        "less_common_side_effects",
        "rare_side_effects",
        "postmarketing_side_effects",
        "unknown_frequency_side_effects",
    ]
    for col in se_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_side_effects_text)

    df.to_csv(out_path_p, index=False, encoding="utf-8")

    print("✅ Preprocess complete")
    print(f"Input : {in_path_p}")
    print(f"Output: {out_path_p}")
    print(f"Rows  : {len(df):,}")


if __name__ == "__main__":
    main()
