from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
from difflib import SequenceMatcher

import pandas as pd


# =============================================================================
# Cleaning helpers
# =============================================================================

def fix_known_fda_truncations(text: str) -> str:
    """
    Fix known openFDA truncation + encoding (mojibake) artifacts BEFORE any dedupe/splitting.
    Must run first for warnings/indications/etc.
    """
    if not text:
        return ""

    t = str(text)

    # Remove Unicode replacement character introduced by encoding_errors="replace"
    # (DO NOT remove normal spaces)
    t = t.replace("\ufffd", "")

    # ---- Known truncation/missing-letter glitches ----
    t = re.sub(r"\bcetaminophen\b", "acetaminophen", t, flags=re.IGNORECASE)
    t = re.sub(r"\bcetamino\b", "acetaminophen", t, flags=re.IGNORECASE)

    # ---- Mojibake cleanup (UTF-8 read as Windows-1252) ----
    replacements = {
        "â€¦": "...",
        "â¦": "...",
        "â€¯": " ",
        "Â": " ",
        "\u00a0": " ",
        "â€“": "-",
        "â€”": "-",
        "â€˜": "'",
        "â€™": "'",
        "â€œ": '"',
        "â€ ": '"',
        "â€¢": "•",
    }
    for bad, good in replacements.items():
        t = t.replace(bad, good)

    # Fix exact artifact you saw
    t = t.replace("câ¦", "...")

    # Any remaining "âX" sequences are garbage — collapse safely.
    t = re.sub(r"â.{1,2}", "...", t)

    # Whitespace normalize (keep newlines if any)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n\s+", "\n", t)

    return t.strip()


def normalize_route(route: str) -> str:
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
    Keep this light: normalize separators + dedupe items + cap item count.
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


# =============================================================================
# Strong dedupe / caps core
# =============================================================================

def _normalize_for_dupe(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def _hard_cap(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    if max_chars <= 1:
        return "…"
    return s[: max_chars - 1].rstrip() + "…"


def _near_duplicate(a: str, b: str, threshold: float = 0.92) -> bool:
    na = _normalize_for_dupe(a)
    nb = _normalize_for_dupe(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def _coverage(a: str, b: str) -> float:
    """
    Coverage of shorter by matches inside longer.
    Handles truncation + small edits better than pure ratio.
    """
    na = _normalize_for_dupe(a)
    nb = _normalize_for_dupe(b)
    if not na or not nb:
        return 0.0
    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(short) < 60:
        return 0.0
    sm = SequenceMatcher(None, short, long)
    matched = sum(bl.size for bl in sm.get_matching_blocks())
    return matched / max(1, len(short))


def _cut_repeated_prefix(raw: str) -> str:
    """
    Handles classic 'big block repeated twice' even if second copy differs a bit.
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


def _has_repeated_window(text: str, window_words: int = 18) -> bool:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(words) < window_words * 2:
        return False
    seen = set()
    for i in range(0, len(words) - window_words):
        chunk = " ".join(words[i:i + window_words])
        if chunk in seen:
            return True
        seen.add(chunk)
    return False


def _cut_at_first_repeated_window(text: str, window_words: int = 18, min_keep: int = 180) -> str:
    """
    Cut at the start of the *second* occurrence of a repeated 18-word window.
    """
    s = (text or "").strip()
    if not s:
        return s

    low = re.sub(r"\s+", " ", s.lower()).strip()
    words = re.findall(r"[a-z0-9]+", low)
    if len(words) < window_words * 2:
        return s

    seen: Dict[str, int] = {}
    repeat_phrase = None

    for i in range(0, len(words) - window_words):
        w = " ".join(words[i:i + window_words])
        if w in seen:
            repeat_phrase = w
            break
        seen[w] = i

    if not repeat_phrase:
        return s

    matches = list(re.finditer(re.escape(repeat_phrase), low))
    if len(matches) < 2:
        return s

    second_start = matches[1].start()
    if second_start <= 0:
        return s

    cut = s[:second_start].strip(" .")
    return cut if len(cut) >= min_keep else s


# =============================================================================
# NEW: repeated-window deduper that preserves most punctuation
# =============================================================================

def _dedupe_repeated_word_windows_preserve_punct(text: str, window_words: int = 18, max_passes: int = 4) -> str:
    """
    Final safety net for 'repeated long-block' rows:
    - Tokenizes into [word] and [punct] tokens
    - Removes later repeated 18-word windows by skipping the duplicate window region
    Keeps most punctuation (not perfect spacing, but stays readable enough).
    """
    s = (text or "").strip()
    if not s:
        return s

    toks = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]+", s)

    word_ti: List[int] = []
    words: List[str] = []
    for ti, tok in enumerate(toks):
        if re.fullmatch(r"[A-Za-z0-9]+", tok):
            word_ti.append(ti)
            words.append(tok.lower())

    if len(words) < window_words * 2:
        return s

    rebuilt = s
    for _ in range(max_passes):
        seen: set[str] = set()
        out_toks: List[str] = []
        i = 0
        changed = False

        while i < len(words):
            if i <= len(words) - window_words:
                key = " ".join(words[i:i + window_words])
                if key in seen:
                    changed = True
                    i += window_words
                    continue
                seen.add(key)

            ti = word_ti[i]
            ti_next = word_ti[i + 1] if i + 1 < len(word_ti) else len(toks)
            out_toks.extend(toks[ti:ti_next])
            i += 1

        rebuilt = " ".join(out_toks)
        rebuilt = re.sub(r"\s+([,.;:!?])", r"\1", rebuilt)
        rebuilt = re.sub(r"\s+([\)\]\}])", r"\1", rebuilt)
        rebuilt = re.sub(r"([\(\[\{])\s+", r"\1", rebuilt)
        rebuilt = re.sub(r"\s{2,}", " ", rebuilt).strip()

        if not changed:
            return rebuilt

        toks = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]+", rebuilt)
        word_ti = []
        words = []
        for ti, tok in enumerate(toks):
            if re.fullmatch(r"[A-Za-z0-9]+", tok):
                word_ti.append(ti)
                words.append(tok.lower())

        if len(words) < window_words * 2:
            return rebuilt

    return rebuilt


# =============================================================================
# WARNING/WARNINGS headline dedupe (keeps warnings stable and helps leaks)
# =============================================================================

_WARNING_HEADLINE_RE = re.compile(
    r"(?i)(?<!^)\s+(?=(?:boxed warning\b|warning\s*:|warnings\s*:))"
)


def dedupe_warning_headlines(text: str) -> str:
    if not text:
        return ""

    s = str(text).strip()
    if not s:
        return ""

    s2 = _WARNING_HEADLINE_RE.sub("\n", s)
    units = [u.strip() for u in re.split(r"\n+", s2) if u.strip()]
    if len(units) <= 1:
        return s.strip()

    out: List[str] = []
    for u in units:
        ul = u.lower().strip()
        is_headline = ul.startswith("warning:") or ul.startswith("warnings:") or ul.startswith("boxed warning")
        if not is_headline:
            out.append(u)
            continue

        dup = False
        for prev in out[-6:]:
            pl = prev.lower().strip()
            if not (pl.startswith("warning:") or pl.startswith("warnings:") or pl.startswith("boxed warning")):
                continue
            if _near_duplicate(u, prev, threshold=0.88) or _coverage(u, prev) >= 0.55:
                dup = True
                break

        if not dup:
            out.append(u)

    return "\n".join(out).strip()


# =============================================================================
# Unit split + unit dedupe (generic)
# =============================================================================

_UNIT_STARTERS_RE = re.compile(
    r"(?i)(?<!^)(?<!\n)\s*(?=(?:"
    r"keep out of reach of children"
    r"|keep away from fire"
    r"|for external use only"
    r"|boxed warning\b"
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

        ul = u.lower().strip()
        if ul.startswith("warning") or ul.startswith("warnings") or ul.startswith("boxed warning"):
            for prev in out[-6:]:
                if _near_duplicate(u, prev, threshold=0.86) or _coverage(u, prev) >= 0.55:
                    key = ""
                    break
            if not key:
                continue

        if out and any(_near_duplicate(u, prev) for prev in out[-4:]):
            continue

        seen.add(key)
        out.append(u)

    return "\n".join(out).strip()


# =============================================================================
# Aggressive sentence/unit dedupe for long text
# =============================================================================

_DIRECTIONS_SPLIT_RE = re.compile(r"\bDirections\b\s*:?", flags=re.IGNORECASE)
_SUBSECTION_MARK_RE = re.compile(r"\(\s*\d+(?:\.\d+)*\s*\)")  # ( 1.1 ) style
_SECTION_NUM_RE = re.compile(r"(?<!\w)(\d+(?:\.\d+)*)(?=\s+[A-Z])")  # 1.1 Title

# Long-text noise removers
_JUNK_TEXT_RE = re.compile(r"(?i)\bclick (?:or tap )?here to enter text\.?\b")
_SECTION_REF_RE = re.compile(r"\(\s*\d+(?:\.\d+)*\s*\)")  # (2) (2.1) etc
_BRACKET_SEE_RE = re.compile(r"\[\s*see [^\]]+\]", flags=re.IGNORECASE)  # [see ...]
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _dedupe_sentences_fallback(text: str) -> str:
    """
    Aggressive dedupe for FDA-ish text that often has weak punctuation.
    Keeps shorter useful lines too (fixes repeating short statements).
    """
    s = (text or "").strip()
    if not s:
        return s

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    s = _DIRECTIONS_SPLIT_RE.sub("\nDirections:\n", s)
    s = _SUBSECTION_MARK_RE.sub(lambda m: "\n" + m.group(0) + " ", s)
    s = _SECTION_NUM_RE.sub(lambda m: "\n" + m.group(1), s)

    parts = re.split(r"(?<=[.!?])\s+|\n+", s)
    parts = [p.strip() for p in parts if p.strip()]

    out: List[str] = []
    seen: set[str] = set()

    for p in parts:
        words = re.findall(r"[A-Za-z0-9]+", p)
        if len(p) < 18 and len(words) < 4:
            continue

        k = _normalize_for_dupe(p)
        if not k or k in seen:
            continue

        recent = out[-10:]
        if any(_near_duplicate(p, prev, threshold=0.88) for prev in recent):
            continue
        if any(_coverage(p, prev) >= 0.70 for prev in recent):
            continue

        seen.add(k)
        out.append(p)

    rebuilt = " ".join(out).strip()
    rebuilt = re.sub(r"\s+", " ", rebuilt).strip()
    return rebuilt


# =============================================================================
# Heading-block dedupe (ALL CAPS + Title Case) + numeric prefix support
# =============================================================================

def _dedupe_heading_blocks_allcaps(text: str, headings_caps: List[str]) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    heading_re = re.compile(
        r"(?:(?<=^)|(?<=\n)|(?<=[.!?])\s+)\s*(?:\d+(?:\.\d+)*\s+)?(?P<h>"
        + "|".join(re.escape(h) for h in headings_caps)
        + r")\b\s*:?\s*"
    )

    matches = list(heading_re.finditer(t))
    if not matches:
        return t

    blocks: List[Tuple[str, str, int]] = []
    for i, m in enumerate(matches):
        h = m.group("h").strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        chunk = t[start:end].strip()
        blocks.append((h, chunk, start))

    best: Dict[str, Tuple[str, int]] = {}
    for h, chunk, start in blocks:
        key = h.upper()
        prev = best.get(key)
        if prev is None or len(chunk) > len(prev[0]):
            best[key] = (chunk, start)

    ordered = sorted(best.items(), key=lambda kv: kv[1][1])
    out_chunks = [kv[1][0] for kv in ordered]
    return "\n\n".join(out_chunks).strip()


def _dedupe_heading_blocks_case_insensitive(text: str, headings: List[str]) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    heading_re = re.compile(
        r"(?:(?<=^)|(?<=\n)|(?<=[.!?])\s+)\s*(?:\d+(?:\.\d+)*\s+)?(?P<h>"
        + "|".join(re.escape(h) for h in headings)
        + r")\b\s*:?\s*",
        flags=re.IGNORECASE,
    )

    matches = list(heading_re.finditer(t))
    if not matches:
        return t

    blocks: List[Tuple[str, str, int]] = []
    for i, m in enumerate(matches):
        h = m.group("h").strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        chunk = t[start:end].strip()
        blocks.append((h, chunk, start))

    best: Dict[str, Tuple[str, int]] = {}
    for h, chunk, start in blocks:
        key = h.lower()
        prev = best.get(key)
        if prev is None or len(chunk) > len(prev[0]):
            best[key] = (chunk, start)

    ordered = sorted(best.items(), key=lambda kv: kv[1][1])
    out_chunks = [kv[1][0] for kv in ordered]
    return "\n\n".join(out_chunks).strip()


# =============================================================================
# Core repeat killer used everywhere
# =============================================================================

def collapse_duplicate_blocks(text: str) -> str:
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
    text2 = dedupe_warning_headlines(text2)
    text2 = _cut_repeated_prefix(text2)

    if _has_repeated_window(text2, window_words=18):
        text2 = _dedupe_sentences_fallback(text2)

    if _has_repeated_window(text2, window_words=18):
        text2 = _cut_at_first_repeated_window(text2, window_words=18, min_keep=180)

    return text2.strip()


# =============================================================================
# Warnings formatting & caps (keep stable clean warnings)
# =============================================================================

def _hard_cap_section_body(body: str, cap: int) -> str:
    body = (body or "").strip()
    if len(body) <= cap:
        return body
    if cap <= 1:
        return "…"
    return body[: cap - 1].rstrip() + "…"


def cap_sections(formatted: str, max_chars: int = 1200) -> str:
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
            new_blocks.append(_hard_cap_section_body(b, 420))
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
                body2 = _hard_cap_section_body(body, cap)
        else:
            body2 = _hard_cap_section_body(body, cap)

        new_blocks.append(f"{title}:\n{body2}")

    out = "\n\n".join(new_blocks).strip()
    return _hard_cap(out, max_chars)


def drop_duplicate_section_bodies(formatted: str, threshold: float = 0.90) -> str:
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

    kept: List[Tuple[str, str]] = []

    for title, body in parsed:
        if not title:
            kept.append((title, body))
            continue

        dup_idx = -1
        for i, (kt, kb) in enumerate(kept):
            if not kt:
                continue

            a = _normalize_for_dupe(body)
            b2 = _normalize_for_dupe(kb)
            if not a or not b2:
                continue

            if SequenceMatcher(None, a, b2).ratio() >= threshold:
                dup_idx = i
                break

            if _coverage(body, kb) >= 0.70:
                dup_idx = i
                break

            pair = {title.lower(), kt.lower()}
            if pair == {"warnings", "overdosage"} and _coverage(body, kb) >= 0.40:
                dup_idx = i
                break

        if dup_idx == -1:
            kept.append((title, body))
        else:
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


def clean_openfda_heading_blocks(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""

    t = fix_known_fda_truncations(text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t).strip()
    if not t:
        return ""

    t = dedupe_warning_headlines(t)

    headings_caps = [
        "BOXED WARNING",
        "WARNINGS AND PRECAUTIONS",
        "WARNINGS",
        "PRECAUTIONS",
        "CONTRAINDICATIONS",
        "ADVERSE REACTIONS",
        "DRUG INTERACTIONS",
        "USE IN SPECIFIC POPULATIONS",
        "PREGNANCY",
        "LACTATION",
        "OVERDOSAGE",
        "DOSAGE AND ADMINISTRATION",
    ]

    t = _dedupe_heading_blocks_allcaps(t, headings_caps)

    heading_re = re.compile(
        r"(?:(?<=^)|(?<=\n)|(?<=[.!?])\s+)\s*(?:\d+(?:\.\d+)*\s+)?(?P<h>"
        + "|".join(re.escape(h) for h in headings_caps)
        + r")\b\s*:?\s*"
    )

    matches = list(heading_re.finditer(t))
    if not matches:
        flat = re.sub(r"\s+", " ", t).strip()
        flat = collapse_duplicate_blocks(flat)
        return _hard_cap(flat, max_chars)

    blocks: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        h = m.group("h").strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        chunk = t[start:end].strip()
        blocks.append((h, chunk))

    best: Dict[str, str] = {}
    for h, chunk in blocks:
        key = h.upper()
        if len(chunk) > len(best.get(key, "")):
            best[key] = chunk

    formatted_blocks: List[str] = []
    for h in headings_caps:
        chunk = best.get(h)
        if not chunk:
            continue

        chunk2 = re.sub(
            rf"^\s*(?:\d+(?:\.\d+)*\s+)?{re.escape(h)}\s*:?\s*[,.\-]*\s*",
            "",
            chunk,
            flags=re.IGNORECASE,
        ).strip()

        chunk2 = collapse_duplicate_blocks(chunk2)
        chunk2 = re.sub(r"\s+", " ", chunk2).strip()
        if not chunk2:
            continue

        formatted_blocks.append(f"{h.title()}:\n{chunk2}")

    out = "\n\n".join(formatted_blocks).strip()
    out = collapse_duplicate_blocks(out)
    out = drop_duplicate_section_bodies(out, threshold=0.90)
    out = cap_sections(out, max_chars=max_chars)
    return out


def clean_fda_warnings_section_level(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""

    t = fix_known_fda_truncations(text)

    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("■", " ").replace("▪", " ").replace("•", " ")
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"\s*-\s*", " ", t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    if not t:
        return ""

    t = dedupe_warning_headlines(t)

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
        chunk = re.sub(rf"^\s*{re.escape(title)}\s*:?\s*", "", chunk, flags=re.IGNORECASE).strip()
        chunk = _cut_repeated_prefix(chunk)
        parts = re.split(r"(?<=[.!?])\s+", chunk)

        out: List[str] = []
        seen: set[str] = set()
        for p in parts:
            p = p.strip(" .")
            words = re.findall(r"[A-Za-z0-9]+", p)
            if len(p) < 18 and len(words) < 4:
                continue

            k = _normalize_for_dupe(p)
            if not k or k in seen:
                continue
            if out and (
                any(_near_duplicate(p, prev) for prev in out[-4:])
                or any(_coverage(p, prev) >= 0.70 for prev in out[-4:])
            ):
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
    out = drop_duplicate_section_bodies(out, threshold=0.90)
    out = cap_sections(out, max_chars=max_chars)
    return out


# =============================================================================
# Long text cleaning (indications / dosage / contraindications)
# =============================================================================

_LONGTEXT_HEADINGS_CAPS = [
    "INDICATIONS",
    "INDICATIONS AND USAGE",
    "DOSAGE AND ADMINISTRATION",
    "CONTRAINDICATIONS",
    "WARNINGS",
    "WARNINGS AND PRECAUTIONS",
    "PRECAUTIONS",
    "ADVERSE REACTIONS",
    "CLINICAL PHARMACOLOGY",
    "DESCRIPTION",
    "HOW SUPPLIED",
]

_LONGTEXT_HEADINGS_TITLECASE = [
    "Indications",
    "Indications and Usage",
    "Dosage and Administration",
    "Contraindications",
    "Warnings",
    "Warnings and Precautions",
    "Precautions",
    "Adverse Reactions",
    "Clinical Pharmacology",
    "Description",
    "How Supplied",
]


def _clean_long_text_generic(text: str, *, max_chars: int = 1200) -> str:
    """
    Strong against:
    - repeated full blocks (A then A again)
    - repeated heading blocks inside the same cell
    - low-punctuation FDA label text
    While avoiding wiping short 'see ...' cells (prevents non-empty drop).
    """
    if not text:
        return ""

    t = fix_known_fda_truncations(text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("■", " ").replace("▪", " ").replace("•", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return ""

    # Remove common template/junk strings
    t = _JUNK_TEXT_RE.sub(" ", t)

    # IMPORTANT: Only strip refs if they appear repeatedly (prevents emptying short cells)
    if len(_SECTION_REF_RE.findall(t)) >= 3:
        t = _SECTION_REF_RE.sub(" ", t)
    if len(_BRACKET_SEE_RE.findall(t)) >= 2:
        t = _BRACKET_SEE_RE.sub(" ", t)

    t = _MULTI_SPACE_RE.sub(" ", t).strip()

    # Remove repeated heading blocks first (ALLCAPS + TitleCase), now supports numeric prefixes
    t = _dedupe_heading_blocks_allcaps(t, _LONGTEXT_HEADINGS_CAPS)
    t = _dedupe_heading_blocks_case_insensitive(t, _LONGTEXT_HEADINGS_TITLECASE)

    # If WARNING/WARNINGS headlines leaked in, dedupe them
    t = dedupe_warning_headlines(t)

    # Strong repeat killer
    t = collapse_duplicate_blocks(t)

    # Always run sentence dedupe for medium/long text (kills repeated short statements too)
    if len(t) >= 220:
        t = _dedupe_sentences_fallback(t)

    # Final safety net for your remaining repeated long-block rows
    if _has_repeated_window(t, window_words=18):
        t = _dedupe_repeated_word_windows_preserve_punct(t, window_words=18, max_passes=4)

    # If STILL repeated, hard cut at second occurrence (rare)
    if _has_repeated_window(t, window_words=18):
        t = _cut_at_first_repeated_window(t, window_words=18, min_keep=180)

    # Flatten for display
    t = re.sub(r"\s+", " ", t).strip()

    return _hard_cap(t, max_chars)


def clean_indications_text(text: str, *, max_chars: int = 1200) -> str:
    return _clean_long_text_generic(text, max_chars=max_chars)


def clean_dosage_text(text: str, *, max_chars: int = 1200) -> str:
    return _clean_long_text_generic(text, max_chars=max_chars)


def clean_contraindications_text(text: str, *, max_chars: int = 1200) -> str:
    return _clean_long_text_generic(text, max_chars=max_chars)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # Always read RAW, write CLEAN
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

    # Long text columns (stronger cleaning)
    if "indications" in df.columns:
        df["indications"] = df["indications"].apply(lambda x: clean_indications_text(x, max_chars=1200))

    if "dosage_and_administration" in df.columns:
        df["dosage_and_administration"] = df["dosage_and_administration"].apply(
            lambda x: clean_dosage_text(x, max_chars=1200)
        )

    if "contraindications" in df.columns:
        df["contraindications"] = df["contraindications"].apply(
            lambda x: clean_contraindications_text(x, max_chars=1200)
        )

    # Warnings (keep stable clean behavior)
    if "warnings" in df.columns:
        df["warnings"] = df["warnings"].apply(lambda x: clean_fda_warnings_section_level(x, max_chars=1200))

    # Brand names
    if "brand_names" in df.columns:
        df["brand_names"] = df["brand_names"].apply(lambda x: trim_brand_names(x, max_items=20))

    # Route
    if "route" in df.columns:
        df["route"] = df["route"].apply(normalize_route)

    # Side effects buckets (already clean; light normalization + dedupe only)
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
