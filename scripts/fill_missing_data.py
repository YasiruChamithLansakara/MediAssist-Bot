"""
fill_missing_data.py — Hybrid Data Enrichment (Option 3)
=========================================================
Strategy: Combine  (1) existing dataset  +  (2) LLM-generated fills
          for null fields  +  (3) rule-based medical safety validation.

Run modes
---------
  python scripts/fill_missing_data.py --dry-run        # analyse gaps, no writes
  python scripts/fill_missing_data.py --llm            # fill with LLM (needs API key)
  python scripts/fill_missing_data.py --rules-only     # fill with rules only (no API key needed)
  python scripts/fill_missing_data.py --llm --rules-only   # both passes

Output
------
  data/processed/drug_knowledge_enriched.csv

Requirements
------------
  pip install groq pandas tqdm
  Set LLM_API_KEY and LLM_PROVIDER=groq in .env (or environment)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ── load .env if present ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV  = ROOT / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "drug_knowledge_enriched.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("fill_missing_data")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — ANALYSE GAPS
# ════════════════════════════════════════════════════════════════════════════

CRITICAL_FIELDS = [
    "drug_class",
    "indications",
    "dosage_and_administration",
    "warnings",
    "contraindications",
    "common_side_effects",
]


def analyse_gaps(df: pd.DataFrame) -> Dict[str, int]:
    """Print and return null counts for critical fields."""
    log.info("=== Dataset gap analysis ===")
    log.info("  Total rows: %d", len(df))
    gaps: Dict[str, int] = {}
    for col in CRITICAL_FIELDS:
        if col not in df.columns:
            log.warning("  Column '%s' not found in dataset", col)
            continue
        n_null = int(df[col].isna().sum() | (df[col] == "").sum())
        pct = n_null / len(df) * 100
        gaps[col] = n_null
        log.info("  %-40s %5d null  (%4.1f%%)", col, n_null, pct)
    return gaps


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — RULE-BASED FILLS  (no API needed, safe defaults)
# ════════════════════════════════════════════════════════════════════════════

# Drug-class inference from generic name (keyword-based)
_CLASS_RULES: List[tuple] = [
    # antidiabetics
    (r"\bmetformin\b",                       "Biguanide antidiabetic"),
    (r"\b(glipizide|glyburide|glimepiride)\b","Sulfonylurea antidiabetic"),
    (r"\b(sitagliptin|saxagliptin|alogliptin|linagliptin)\b", "DPP-4 inhibitor antidiabetic"),
    (r"\b(empagliflozin|dapagliflozin|canagliflozin)\b",      "SGLT-2 inhibitor antidiabetic"),
    (r"\b(semaglutide|liraglutide|exenatide|dulaglutide)\b",  "GLP-1 receptor agonist"),
    (r"\binsulin\b",                         "Insulin / antidiabetic hormone"),
    (r"\bpioglitazone\b",                    "Thiazolidinedione antidiabetic"),
    # antihypertensives
    (r"\b(lisinopril|enalapril|ramipril|captopril|benazepril|quinapril|fosinopril|perindopril)\b", "ACE inhibitor"),
    (r"\b(losartan|valsartan|irbesartan|candesartan|olmesartan|telmisartan|azilsartan)\b",          "ARB antihypertensive"),
    (r"\b(amlodipine|nifedipine|felodipine|diltiazem|verapamil|nicardipine)\b",                     "Calcium channel blocker"),
    (r"\b(hydrochlorothiazide|chlorthalidone|furosemide|spironolactone|torsemide)\b",                "Diuretic antihypertensive"),
    (r"\b(metoprolol|atenolol|bisoprolol|carvedilol|nebivolol|propranolol|labetalol)\b",             "Beta-blocker"),
    # asthma / COPD
    (r"\b(albuterol|salbutamol|levalbuterol|pirbuterol)\b",   "Short-acting beta-2 agonist (SABA)"),
    (r"\b(salmeterol|formoterol|arformoterol|vilanterol|indacaterol)\b", "Long-acting beta-2 agonist (LABA)"),
    (r"\b(fluticasone|budesonide|beclomethasone|mometasone|ciclesonide)\b", "Inhaled corticosteroid"),
    (r"\b(montelukast|zafirlukast|zileuton)\b",               "Leukotriene modifier"),
    (r"\b(tiotropium|ipratropium|umeclidinium|glycopyrronium|aclidinium)\b", "Anticholinergic bronchodilator"),
    # statins / cardiac
    (r"\b(atorvastatin|rosuvastatin|simvastatin|pravastatin|lovastatin|fluvastatin|pitavastatin)\b", "Statin / HMG-CoA reductase inhibitor"),
    (r"\b(aspirin|clopidogrel|ticagrelor|prasugrel|warfarin|rivaroxaban|apixaban|dabigatran|edoxaban)\b", "Antiplatelet / anticoagulant"),
    (r"\b(digoxin)\b",                       "Cardiac glycoside"),
    (r"\b(amiodarone|sotalol|flecainide|propafenone)\b",      "Antiarrhythmic"),
    # pain / arthritis
    (r"\b(ibuprofen|naproxen|diclofenac|indomethacin|celecoxib|meloxicam|ketorolac)\b", "NSAID"),
    (r"\b(acetaminophen|paracetamol)\b",      "Analgesic / antipyretic"),
    (r"\b(prednisone|prednisolone|methylprednisolone|dexamethasone|hydrocortisone)\b",   "Corticosteroid"),
    (r"\b(methotrexate|sulfasalazine|hydroxychloroquine|leflunomide)\b",                 "DMARD / disease-modifying agent"),
    (r"\b(adalimumab|etanercept|infliximab|tocilizumab|abatacept|certolizumab|golimumab)\b", "Biologic DMARD"),
    # migraine
    (r"\b(sumatriptan|rizatriptan|naratriptan|zolmitriptan|eletriptan|almotriptan|frovatriptan)\b", "Triptan / 5-HT1 agonist (migraine)"),
    (r"\b(topiramate|valproate|valproic acid|amitriptyline|propranolol|verapamil)\b",    "Migraine prophylaxis agent"),
    (r"\b(ergotamine|dihydroergotamine)\b",   "Ergot alkaloid (migraine)"),
    (r"\b(ubrogepant|rimegepant|atogepant)\b","CGRP receptor antagonist (migraine)"),
]

_CLASS_RE = [(re.compile(pat, re.IGNORECASE), cls) for pat, cls in _CLASS_RULES]


# Generic safety warnings keyed by drug class fragment
_CLASS_WARNINGS: Dict[str, str] = {
    "metformin":          "Avoid in severe renal impairment (eGFR <30). May cause lactic acidosis rarely. Hold before contrast procedures.",
    "sulfonylurea":       "Risk of hypoglycemia, especially in elderly or with missed meals. Monitor blood glucose regularly.",
    "sglt-2":             "Risk of UTI, genital mycotic infections, and ketoacidosis. Hold before surgery.",
    "glp-1":              "May cause nausea/vomiting. Rare risk of pancreatitis. Not recommended in medullary thyroid carcinoma history.",
    "insulin":            "Risk of hypoglycemia. Monitor blood glucose. Do not share needles.",
    "ace inhibitor":      "May cause dry cough, angioedema. Avoid in pregnancy. Monitor potassium and renal function.",
    "arb":                "Avoid in pregnancy. Monitor potassium and renal function. Risk of hypotension.",
    "calcium channel":    "May cause peripheral edema, flushing, headache. Avoid grapefruit juice.",
    "beta-blocker":       "Do not stop abruptly — may worsen angina. May mask hypoglycemia symptoms in diabetics.",
    "diuretic":           "Monitor electrolytes (potassium). Risk of dehydration and orthostatic hypotension.",
    "saba":               "Overuse may indicate poorly controlled asthma. Can cause tachycardia, tremor.",
    "laba":               "Must be used with inhaled corticosteroid, not as monotherapy for asthma.",
    "inhaled corticosteroid": "Rinse mouth after use to prevent oral candidiasis.",
    "nsaid":              "Risk of GI bleeding, peptic ulcer. Avoid in chronic kidney disease and heart failure. Can increase blood pressure.",
    "statin":             "Risk of myopathy/rhabdomyolysis (especially at high doses). Monitor liver enzymes. Avoid grapefruit juice.",
    "antiplatelet":       "Increased bleeding risk. Do not stop without consulting doctor before surgery.",
    "anticoagulant":      "Increased bleeding risk. Monitor INR regularly (for warfarin). Avoid NSAIDs.",
    "corticosteroid":     "Long-term use: risk of osteoporosis, adrenal suppression, weight gain, glucose elevation. Do not stop abruptly.",
    "dmard":              "Requires regular monitoring (CBC, liver function). Risk of infection. Takes weeks to show effect.",
    "triptan":            "Do not use in ischaemic heart disease or uncontrolled hypertension. Max 2 doses/24h.",
    "migraine prophylaxis":"Regular daily dosing required for prevention. Do not use as acute rescue medication.",
}

# Contraindication templates by drug class
_CLASS_CONTRAINDICATIONS: Dict[str, str] = {
    "metformin":          "Severe renal impairment (eGFR <30), hepatic impairment, metabolic acidosis, contrast dye procedures.",
    "sulfonylurea":       "Type 1 diabetes, diabetic ketoacidosis, severe renal or hepatic impairment, pregnancy.",
    "sglt-2":             "Severe renal impairment (eGFR <30), end-stage renal disease, Type 1 diabetes, recurrent UTIs.",
    "ace inhibitor":      "Pregnancy, history of angioedema with ACE inhibitor, bilateral renal artery stenosis, concomitant aliskiren in diabetes/renal impairment.",
    "arb":                "Pregnancy, bilateral renal artery stenosis. Avoid with ACE inhibitor in most patients.",
    "beta-blocker":       "Severe bradycardia, heart block (2nd/3rd degree), decompensated heart failure, uncontrolled asthma/COPD.",
    "nsaid":              "Active GI ulcer/bleeding, severe renal impairment, severe heart failure, third trimester of pregnancy.",
    "statin":             "Active liver disease, pregnancy, breastfeeding. Caution with concomitant medications that raise statin levels.",
    "triptan":            "Ischaemic heart disease, Prinzmetal angina, uncontrolled hypertension, stroke/TIA history, hemiplegic or basilar migraine.",
    "corticosteroid":     "Systemic fungal infections. Long-term use in patients with osteoporosis, uncontrolled diabetes, or psychiatric history requires careful monitoring.",
}


def _infer_drug_class(row: pd.Series) -> Optional[str]:
    """Infer drug class from generic name using keyword rules."""
    name = str(row.get("generic_name_clean") or row.get("generic_name") or "").lower()
    for pattern, drug_class in _CLASS_RE:
        if pattern.search(name):
            return drug_class
    return None


def _rule_warning(drug_class: str) -> Optional[str]:
    """Return a conservative safety warning for a drug class."""
    dc_lower = drug_class.lower()
    for key, warning in _CLASS_WARNINGS.items():
        if key in dc_lower:
            return warning
    return None


def _rule_contraindication(drug_class: str) -> Optional[str]:
    dc_lower = drug_class.lower()
    for key, contra in _CLASS_CONTRAINDICATIONS.items():
        if key in dc_lower:
            return contra
    return None


def _is_empty(value) -> bool:
    """
    Return True when a pandas cell is null/empty.

    Handles: None, float NaN, the literal string "nan", blank strings.
    pandas null values are float('nan'), so bool(NaN) is True in Python
    and str(NaN) == 'nan' — both need special treatment.
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return not s or s.lower() == "nan"


def _safe_str(value) -> str:
    """Convert a potentially-null pandas cell to a clean string."""
    if _is_empty(value):
        return ""
    return str(value).strip()


def apply_rule_fills(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass 1 — fill missing fields using drug-class keyword rules.
    Only fills cells that are currently null/empty.
    """
    df = df.copy()
    filled: Dict[str, int] = {f: 0 for f in CRITICAL_FIELDS}

    for idx, row in df.iterrows():
        # Infer drug class first
        current_class = _safe_str(row.get("drug_class"))
        if not current_class:
            inferred = _infer_drug_class(row)
            if inferred:
                df.at[idx, "drug_class"] = inferred
                current_class = inferred
                filled["drug_class"] += 1

        if not current_class:
            continue

        # Fill warnings
        if _is_empty(row.get("warnings")):
            w = _rule_warning(current_class)
            if w:
                df.at[idx, "warnings"] = w
                filled["warnings"] += 1

        # Fill contraindications
        if _is_empty(row.get("contraindications")):
            c = _rule_contraindication(current_class)
            if c:
                df.at[idx, "contraindications"] = c
                filled["contraindications"] += 1

        # Fill common_side_effects from top_label_confirmed_side_effects if available
        if _is_empty(row.get("common_side_effects")):
            src = _safe_str(row.get("top_label_confirmed_side_effects"))
            if src and src.lower() != "nan":
                df.at[idx, "common_side_effects"] = src
                filled["common_side_effects"] += 1

    log.info("Rule-based fills:")
    for field, count in filled.items():
        log.info("  %-40s %5d cells filled", field, count)

    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — LLM FILLS  (Llama 3.1 8B via Groq)
# ════════════════════════════════════════════════════════════════════════════

FILL_SYSTEM_PROMPT = """You are a pharmaceutical reference assistant.
Your task is to provide concise, factual drug information.
Respond ONLY with the requested field content — no preamble, no JSON wrapper.
Keep responses under 120 words. Be conservative and accurate.
This information is for educational purposes only."""

FILL_PROMPTS: Dict[str, str] = {
    "drug_class": (
        "What is the pharmacological drug class of {name}? "
        "Give the class name only (e.g., 'ACE inhibitor', 'SSRI antidepressant', 'Statin'). "
        "Maximum 10 words."
    ),
    "indications": (
        "What chronic conditions or diseases is {name} most commonly prescribed for? "
        "List 2-4 indications in one sentence. Maximum 50 words."
    ),
    "warnings": (
        "What are the key safety warnings for {name}? "
        "Focus on the most clinically important (organ toxicity, serious interactions, special populations). "
        "Maximum 80 words."
    ),
    "contraindications": (
        "What are the main contraindications for {name}? "
        "List conditions or situations where it must NOT be used. Maximum 60 words."
    ),
    "dosage_and_administration": (
        "What is the typical adult dosage and route of administration for {name} "
        "for its primary chronic disease indication? Maximum 60 words."
    ),
    "common_side_effects": (
        "What are the most common side effects of {name} (affecting >1% of patients)? "
        "List 4-6 side effects briefly. Maximum 50 words."
    ),
}


def _call_llm_groq(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant") -> Optional[str]:
    """Call Groq API with Llama 3.1 8B and return the response text."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FILL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,   # low temperature for factual content
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        # Fall back to requests
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": FILL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens":  200,
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        log.warning("LLM call failed: %s", exc)
        return None


# ── VALIDATION LAYER (Step 3 of Option 3) ───────────────────────────────────

_UNSAFE_PATTERNS = re.compile(
    r"\b(cure|guaranteed|always|never fails|100%|miracle|diagnose|prescribe)\b",
    re.IGNORECASE,
)
_DISCLAIMER_REQUIRED = ["consult", "healthcare", "doctor", "pharmacist", "professional", "physician"]


def validate_llm_fill(field: str, text: str) -> bool:
    """
    Rule-based safety validation of LLM-generated content.
    Returns True if the fill passes validation.
    """
    if not text or len(text.strip()) < 5:
        return False
    # Reject overconfident / unsafe language
    if _UNSAFE_PATTERNS.search(text):
        log.debug("Validation rejected (unsafe pattern): %s", text[:80])
        return False
    # Warnings / contraindications must not be just "None" or "No warnings"
    if field in ("warnings", "contraindications"):
        lower = text.strip().lower()
        if lower in {"none", "no warnings", "no contraindications", "n/a", "not applicable"}:
            return False
    return True


def apply_llm_fills(
    df: pd.DataFrame,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    max_rows: int = 300,
    delay: float = 2.0,
    output_path: Optional[str] = None,
    save_interval: int = 25,
) -> pd.DataFrame:
    """
    Pass 2 — use LLM to fill remaining null fields.

    Args:
        df:            DataFrame (after rule fills)
        api_key:       Groq API key
        model:         Groq model name
        max_rows:      Safety cap (Groq free tier: 30 RPM / 14,400 RPD)
        delay:         Seconds between API calls  (2.0 = safe for free tier)
        output_path:   If set, save a checkpoint every `save_interval` rows
        save_interval: How often to checkpoint (default every 25 processed rows)
    """
    df = df.copy()
    processed = 0
    filled: Dict[str, int] = {f: 0 for f in FILL_PROMPTS}

    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=min(len(df), max_rows), desc="LLM fills")
    except ImportError:
        iterator = df.iterrows()

    for idx, row in iterator:
        if processed >= max_rows:
            log.info("Reached max_rows=%d cap — stopping LLM fills", max_rows)
            break

        name = (
            _safe_str(row.get("generic_name_clean"))
            or _safe_str(row.get("generic_name"))
        )
        if not name:
            continue

        row_needs_fill = any(_is_empty(row.get(f)) for f in FILL_PROMPTS)
        if not row_needs_fill:
            continue

        processed += 1

        for field, template in FILL_PROMPTS.items():
            if not _is_empty(row.get(field)):
                continue  # already has data

            prompt = template.format(name=name)
            answer = _call_llm_groq(prompt, api_key, model)

            if answer and validate_llm_fill(field, answer):
                df.at[idx, field] = answer
                filled[field] += 1
                log.debug("Filled %s[%s] = %s…", field, name, answer[:60])

            time.sleep(delay)  # rate limit safety (free tier: 30 RPM)

        # ── Checkpoint save ──────────────────────────────────────────────
        if output_path and processed % save_interval == 0:
            df.to_csv(output_path, index=False)
            log.info("✓ Checkpoint saved at row %d → %s", processed, output_path)

    log.info("LLM fills completed (processed %d rows):", processed)
    for field, count in filled.items():
        log.info("  %-40s %5d cells filled", field, count)

    return df


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid data enrichment for MediAssist drug knowledge base"
    )
    parser.add_argument("--dry-run",       action="store_true", help="Analyse gaps only, no writes")
    parser.add_argument("--llm",           action="store_true", help="Fill nulls with LLM (requires LLM_API_KEY)")
    parser.add_argument("--rules-only",    action="store_true", help="Apply rule-based fills only")
    parser.add_argument("--max-rows",      type=int,   default=300,  help="Max rows for LLM fills (default 300)")
    parser.add_argument("--delay",         type=float, default=2.0,  help="Delay between LLM calls in seconds (default 2.0 for free tier)")
    parser.add_argument("--save-interval", type=int,   default=25,   help="Checkpoint save every N processed rows (default 25)")
    parser.add_argument("--model",         type=str,   default="llama-3.1-8b-instant", help="Groq model name")
    parser.add_argument("--input",         type=str,   default=str(INPUT_CSV),  help="Input CSV path")
    parser.add_argument("--output",        type=str,   default=str(OUTPUT_CSV), help="Output CSV path")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    log.info("Loading dataset: %s", args.input)
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    log.info("Loaded %d rows × %d columns", *df.shape)

    # Replace empty strings with NaN for consistent null handling
    df.replace("", pd.NA, inplace=True)

    # ── Analyse gaps ───────────────────────────────────────────────────────
    gaps = analyse_gaps(df)

    if args.dry_run:
        log.info("--dry-run: analysis complete, no output written.")
        return

    # ── Rule-based fills ───────────────────────────────────────────────────
    if args.rules_only or not args.llm:
        log.info("\n=== Pass 1: Rule-based fills ===")
        df = apply_rule_fills(df)

    # ── LLM fills ──────────────────────────────────────────────────────────
    if args.llm:
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            log.error(
                "LLM_API_KEY not set. "
                "Get a FREE Groq key at https://console.groq.com and set it in .env"
            )
            sys.exit(1)

        # Rule fills first (reduces LLM calls needed)
        if not args.rules_only:
            log.info("\n=== Pass 1: Rule-based fills (before LLM) ===")
            df = apply_rule_fills(df)

        log.info("\n=== Pass 2: LLM fills (Groq/%s) ===", args.model)
        log.info(
            "  Rate limit note: free tier ~30 RPM. "
            "Saving checkpoint every %d rows to %s",
            args.save_interval, args.output,
        )
        df = apply_llm_fills(
            df,
            api_key=api_key,
            model=args.model,
            max_rows=args.max_rows,
            delay=args.delay,
            output_path=args.output,
            save_interval=args.save_interval,
        )

    # ── Post-validation summary ────────────────────────────────────────────
    log.info("\n=== Post-enrichment gap analysis ===")
    analyse_gaps(df)

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Enriched dataset saved → %s", out_path)
    log.info(
        "\n✅ Done. To use the enriched dataset, set:\n"
        "   DRUG_DATASET_PATH=%s\n"
        "   (or rename it to drug_knowledge_bot_ready_clean.csv)\n",
        out_path,
    )


if __name__ == "__main__":
    main()
