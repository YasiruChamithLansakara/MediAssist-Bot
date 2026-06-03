"""
evaluate_pipeline.py — MediAssist Bot Evaluation Suite
=======================================================

Evaluates every major component of the pipeline and generates:
  reports/evaluation_report.json     (full machine-readable data)
  reports/evaluation_report.html     (visual HTML for presentation)
  reports/evaluation_summary.md      (markdown summary)

Usage:
  python scripts/evaluate_pipeline.py              # full evaluation
  python scripts/evaluate_pipeline.py --no-llm     # skip Groq API calls
  python scripts/evaluate_pipeline.py --output DIR  # custom output dir
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Bootstrap ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _pct(n: int, total: int) -> float:
    return round(100 * n / total, 1) if total > 0 else 0.0

def _grade(pct: float) -> str:
    if pct >= 90: return "EXCELLENT"
    if pct >= 75: return "GOOD"
    if pct >= 60: return "FAIR"
    return "POOR"

def _color(pct: float) -> str:
    if pct >= 90: return "#22c55e"
    if pct >= 75: return "#f59e0b"
    if pct >= 60: return "#fb923c"
    return "#ef4444"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def eval_dataset() -> Dict[str, Any]:
    print("\n[1/8] Dataset quality …")
    t0 = time.time()
    try:
        import pandas as pd
        csv_path = os.getenv(
            "DRUG_DATASET_PATH",
            str(ROOT / "data" / "processed" / "drug_knowledge_enriched.csv"),
        )
        if not os.path.exists(csv_path):
            csv_path = str(ROOT / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv")

        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        total = len(df)

        fields = [
            "generic_name", "drug_class", "indications",
            "dosage_and_administration", "warnings",
            "contraindications", "common_side_effects",
        ]
        coverage: Dict[str, float] = {}
        for col in fields:
            if col in df.columns:
                filled = ((df[col] != "") & (df[col].str.lower() != "nan")).sum()
                coverage[col] = _pct(int(filled), total)

        dups = int(df.duplicated(subset=["drug_id", "generic_name"]).sum())
        avg_cov = round(sum(coverage.values()) / len(coverage), 1)

        result = {
            "total_drugs": total,
            "duplicate_records": dups,
            "field_coverage_pct": coverage,
            "average_coverage_pct": avg_cov,
            "csv_path": csv_path,
            "elapsed_s": round(time.time() - t0, 2),
            "score_pct": avg_cov,
            "status": "PASS" if avg_cov >= 60 else "WARN",
        }
        print(f"    OK {total} drugs, avg coverage {avg_cov}%")
        return result
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 2. DRUG LOOKUP ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

LOOKUP_TEST_CASES = [
    # (query, expected_generic_name_fragment, label)
    # ── Exact match — drugs confirmed in the dataset ──
    ("metformin",         "metformin",       "exact match"),
    ("paracetamol",       "acetaminophen",   "alias match"),
    ("tylenol",           "acetaminophen",   "brand alias"),
    ("aspirin",           "aspirin",         "exact match"),
    ("atorvastatin",      "atorvastatin",    "exact match"),
    ("albuterol",         "albuterol",       "exact match"),
    ("amoxicillin",       "amoxicillin",     "exact match"),
    ("azithromycin",      "azithromycin",    "exact match"),
    ("furosemide",        "furosemide",      "exact match"),
    ("simvastatin",       "simvastatin",     "exact match"),
    ("naproxen",          "naproxen",        "exact match"),
    ("omeprazole",        "omeprazole",      "exact match"),
    ("clopidogrel",       "clopidogrel",     "exact match"),
    ("amlodipine",        "amlodipine",      "exact match"),   # stored as 'AMLODIPINE BESYLATE'
    # ── Typo tolerance ──
    ("metfromin",         "metformin",       "typo tolerance"),
    ("paracetemol",       "acetaminophen",   "typo tolerance"),
    # ── Synonym / alias ──
    ("salbutamol",        "albuterol",       "synonym match"),
    # ── Dosage strip ──
    ("Aspirin 81mg",      "aspirin",         "dosage strip"),
    ("Metformin 500 mg",  "metformin",       "dosage strip"),
    # ── No-match (gibberish) ──
    ("xyzunknowndrug123", "",                "gibberish no match"),
]

def eval_drug_lookup() -> Dict[str, Any]:
    print("\n[2/8] Drug lookup accuracy …")
    t0 = time.time()
    try:
        from app.services.drug_lookup import init_store, lookup_drug
        init_store()

        passed = failed = 0
        cases: List[Dict[str, Any]] = []
        latencies: List[float] = []

        for query, expected_frag, label in LOOKUP_TEST_CASES:
            t1 = time.time()
            result = lookup_drug(query, disease="diabetes", age=50, top_k=1)
            lat = round((time.time() - t1) * 1000, 1)
            latencies.append(lat)

            match = result.get("best_match")
            got = (
                (match.get("generic_name_clean") or match.get("generic_name") or "").lower()
                if match else ""
            )
            expected = expected_frag.lower()

            if expected == "":
                ok = (result.get("status") == "no_match" or match is None)
            else:
                ok = expected in got

            if ok:
                passed += 1
            else:
                failed += 1

            cases.append({
                "query": query, "label": label,
                "expected": expected_frag or "(no match)",
                "got": got or "(no match)",
                "latency_ms": lat,
                "pass": ok,
            })

        acc = _pct(passed, len(LOOKUP_TEST_CASES))
        avg_lat = round(sum(latencies) / len(latencies), 1)

        print(f"    OK {passed}/{len(LOOKUP_TEST_CASES)} passed ({acc}%), avg {avg_lat} ms")
        return {
            "total": len(LOOKUP_TEST_CASES), "passed": passed, "failed": failed,
            "accuracy_pct": acc, "avg_latency_ms": avg_lat,
            "cases": cases, "elapsed_s": round(time.time() - t0, 2),
            "score_pct": acc, "status": "PASS" if acc >= 80 else "WARN",
        }
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 3. NER — DRUG EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

NER_TEST_CASES = [
    {
        "text": "Patient takes Metformin 500mg twice daily and Lisinopril 10mg once daily.",
        "expected": ["metformin", "lisinopril"],
        "disease": "diabetes",
    },
    {
        "text": "Prescription: Aspirin 81mg OD, Atorvastatin 40mg at night.",
        "expected": ["aspirin", "atorvastatin"],
        "disease": "heart disease",
    },
    {
        "text": "Salbutamol inhaler 2 puffs PRN. Fluticasone 250mcg BD.",
        "expected": ["albuterol", "fluticasone"],
        "disease": "asthma",
    },
    {
        "text": "T. Metformin 500mg BD. T. Amlodipine 5mg OD.",
        "expected": ["metformin", "amlodipine"],
        "disease": "hypertension",
    },
    {
        "text": "Patient requires ibuprofen 400mg TDS for pain management.",
        "expected": ["ibuprofen"],
        "disease": "arthritis",
    },
]

def eval_ner() -> Dict[str, Any]:
    print("\n[3/8] NER drug extraction …")
    t0 = time.time()
    try:
        from app.services.ner_service import extract_medication_entities

        tp = fp = fn = 0
        cases: List[Dict[str, Any]] = []

        for tc in NER_TEST_CASES:
            extracted = extract_medication_entities(
                tc["text"], disease=tc["disease"], age=50, max_entities=8
            )
            got_names = set()
            for e in extracted:
                n = (e.get("drug") or e.get("normalized") or "").lower().strip()
                if n:
                    got_names.add(n)

            expected_set = set(tc["expected"])

            case_tp = sum(1 for exp in expected_set if any(exp in got for got in got_names))
            case_fn = len(expected_set) - case_tp
            case_fp = sum(1 for got in got_names if not any(exp in got for exp in expected_set))

            tp += case_tp
            fn += case_fn
            fp += case_fp

            cases.append({
                "text": tc["text"][:80] + "…",
                "expected": list(expected_set),
                "extracted": sorted(got_names),
                "tp": case_tp, "fn": case_fn, "fp": case_fp,
            })

        precision = _pct(tp, tp + fp) if (tp + fp) > 0 else 0.0
        recall    = _pct(tp, tp + fn) if (tp + fn) > 0 else 0.0
        f1        = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0.0

        print(f"    OK Precision {precision}%  Recall {recall}%  F1 {f1}%")
        return {
            "precision_pct": precision, "recall_pct": recall, "f1_pct": f1,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "cases": cases, "elapsed_s": round(time.time() - t0, 2),
            "score_pct": f1, "status": "PASS" if f1 >= 60 else "WARN",
        }
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAFETY LAYER — EMERGENCY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

SAFETY_TESTS = {
    "emergency_phrases": [
        "I have severe chest pain",
        "difficulty breathing and gasping",
        "I think I had a seizure",
        "uncontrolled bleeding from the wound",
        "possible overdose — took too many pills",
        "severe allergic reaction, throat closing",
        "losing consciousness, feel very dizzy",
        "unbearable pain in my chest",
    ],
    "safe_phrases": [
        "What are the side effects of metformin?",
        "How should I take lisinopril?",
        "Can I take aspirin with warfarin?",
        "What is the dosage for ibuprofen?",
        "Is amlodipine safe for elderly patients?",
    ],
}

def eval_safety() -> Dict[str, Any]:
    print("\n[4/8] Safety layer …")
    t0 = time.time()
    try:
        from app.services.safety_service import detect_emergency_symptoms

        emergency_detected = sum(
            1 for phrase in SAFETY_TESTS["emergency_phrases"]
            if detect_emergency_symptoms(phrase)[0]
        )
        false_positives = sum(
            1 for phrase in SAFETY_TESTS["safe_phrases"]
            if detect_emergency_symptoms(phrase)[0]
        )

        total_emerg = len(SAFETY_TESTS["emergency_phrases"])
        total_safe  = len(SAFETY_TESTS["safe_phrases"])

        recall    = _pct(emergency_detected, total_emerg)
        precision = _pct(total_safe - false_positives, total_safe)

        print(f"    OK Emergency recall {recall}%  Safe-phrase precision {precision}%")
        return {
            "emergency_recall_pct": recall,
            "safe_precision_pct": precision,
            "true_positives": emergency_detected,
            "false_positives": false_positives,
            "total_emergency_tests": total_emerg,
            "total_safe_tests": total_safe,
            "elapsed_s": round(time.time() - t0, 2),
            "score_pct": round((recall + precision) / 2, 1),
            "status": "PASS" if recall >= 80 and precision >= 80 else "WARN",
        }
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 4b. Conversation Memory
# ══════════════════════════════════════════════════════════════════════════════

def eval_conversation_memory() -> Dict[str, Any]:
    print("\n[4b/8] Conversation memory …")
    t0 = time.time()
    try:
        from app.services.conversation_memory import (
            get_conversation_memory,
            add_turn_to_memory,
            get_conversation_history,
            get_context_summary,
        )

        mem = get_conversation_memory()
        sid = "eval-session-001"
        # ensure clean state
        mem.clear_session(sid)

        add_turn_to_memory(sid, "user", "Hello, I take metformin", disease="diabetes", age=55)
        add_turn_to_memory(sid, "assistant", "Noted. How can I help?", disease="diabetes", age=55)

        history = get_conversation_history(sid)
        summary = get_context_summary(sid)

        turn_count = summary.get("turn_count", 0)
        ok = turn_count >= 2

        print(f"    OK turns={turn_count} sessions={len(mem.get_all_sessions())}")
        return {
            "turn_count": turn_count,
            "history_sample": history[:4],
            "summary": summary,
            "elapsed_s": round(time.time() - t0, 2),
            "score_pct": 100.0 if ok else 0.0,
            "status": "PASS" if ok else "WARN",
        }
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 4c. Chat Engine (end-to-end smoke)
# ══════════════════════════════════════════════════════════════════════════════

def eval_chat_engine() -> Dict[str, Any]:
    print("\n[4c/8] Chat engine …")
    t0 = time.time()
    try:
        from app.services.chat_engine import build_chat_response

        cases = [
            {"msg": "What are the common side effects of metformin?", "expect_emergency": False},
            {"msg": "I have severe chest pain and cant breathe", "expect_emergency": True},
        ]

        passed = 0
        results: List[Dict[str, Any]] = []

        for c in cases:
            out = build_chat_response(message=c["msg"], disease="diabetes", age=60, drugs=[])
            detected = bool(out.get("is_emergency"))
            ok = detected == c["expect_emergency"]
            if ok:
                passed += 1
            results.append({"msg": c["msg"], "is_emergency": detected, "ok": ok, "excerpt": out.get("answer","")[:200]})

        acc = _pct(passed, len(cases))
        print(f"    OK {passed}/{len(cases)} chat cases correct ({acc}%)")
        return {
            "total": len(cases), "passed": passed, "accuracy_pct": acc,
            "cases": results, "elapsed_s": round(time.time() - t0, 2),
            "score_pct": acc, "status": "PASS" if acc >= 80 else "WARN",
        }
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 4d. RAG shim
# ══════════════════════════════════════════════════════════════════════════════

def eval_rag() -> Dict[str, Any]:
    print("\n[4d/8] RAG shim …")
    t0 = time.time()
    try:
        from app.services.rag_service import get_rag_status, is_rag_available

        status = get_rag_status()
        available = is_rag_available()

        print(f"    OK RAG available={available}")
        return {"status_detail": status, "available": available,
                "elapsed_s": round(time.time() - t0, 2),
                "score_pct": 100.0 if available else 0.0,
                "status": "PASS" if available else "SKIP"}
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 4e. Activity metrics
# ══════════════════════════════════════════════════════════════════════════════

def eval_activity_metrics() -> Dict[str, Any]:
    print("\n[4e/8] Activity metrics …")
    t0 = time.time()
    try:
        from app.services.activity_metrics import record_activity, get_activity_snapshot, get_activity_metrics

        metrics = get_activity_metrics()
        # record a few events
        record_activity("lookup", success=True, status_code=200, detail="lookup ok", context={})
        record_activity("chat", success=True, status_code=200, detail="chat ok", context={})
        record_activity("ocr", success=False, status_code=500, detail="ocr fail", context={"engine": "tesseract"})

        snap = get_activity_snapshot()
        counts = snap.get("counts", {})

        ok = counts.get("lookup", 0) >= 1 and counts.get("chat", 0) >= 1

        print(f"    OK counts={counts}")
        return {"snapshot": snap, "counts": counts,
                "elapsed_s": round(time.time() - t0, 2),
                "score_pct": 100.0 if ok else 0.0,
                "status": "PASS" if ok else "WARN"}
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 5. OCR PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

OCR_TEXT_CASES = [
    {
        "text":        "Metformin 500mg BD\nLisinopril 10mg OD",
        "disease":     "diabetes",
        "age":         55,
        "expect_any":  ["metformin", "lisinopril"],
        "label":       "two-drug prescription",
    },
    {
        "text":        "T. Aspirin 81mg OD\nT. Atorvastatin 40mg at night",
        "disease":     "heart disease",
        "age":         65,
        "expect_any":  ["aspirin", "atorvastatin"],
        "label":       "tablet-prefix prescription",
    },
    {
        "text":        "Salbutamol 2 puffs PRN\nFluticasone 250mcg BD inhaled",
        "disease":     "asthma",
        "age":         40,
        "expect_any":  ["albuterol", "fluticasone", "salbutamol"],
        "label":       "inhaler prescription",
    },
    {
        "text":        "Inj. Insulin 10 units SC nocte\nT. Metformin 1000mg BD",
        "disease":     "diabetes",
        "age":         60,
        "expect_any":  ["insulin", "metformin"],
        "label":       "injection + tablet (prefix stripping)",
    },
    {
        "text":        "Ibuprofen 400mg TDS after food\nParacetamol 500mg PRN",
        "disease":     "arthritis",
        "age":         50,
        "expect_any":  ["ibuprofen", "acetaminophen", "paracetamol"],
        "label":       "NSAID + analgesic",
    },
]

_ABBREV_STRIP_CASES = [
    ("T. Metformin 500mg BD",       "Metformin"),
    ("Tab. Aspirin 81mg OD",        "Aspirin"),
    ("Cap. Amlodipine 5mg OD",      "Amlodipine"),
    ("Inj. Insulin 10 units SC",    "Insulin"),
    ("Metformin500mg",              "Metformin"),   # OCR run-on
]

_SAFE_TOKENS = ["HbA1c", "B12", "D3", "T3"]

def eval_ocr() -> Dict[str, Any]:
    print("\n[5/8] OCR pipeline …")
    t0 = time.time()
    try:
        from app.services.ocr_service import (
            _clean_ocr_text,
            _ocr_candidate_score,
            analyze_prescription_text,
            ocr_runtime_status,
        )
        from app.services.ner_service import _clean_prescription_text

        # ── Engine status ────────────────────────────────────────────────────
        status = ocr_runtime_status()
        tesseract_ok = bool(status.get("tesseract_available"))
        easyocr_ok   = bool(status.get("easyocr_available"))
        at_least_one = bool(status.get("available"))

        engines: List[str] = []
        if tesseract_ok: engines.append("Tesseract")
        if easyocr_ok:   engines.append("EasyOCR")

        # ── Preprocessing / cleaning ─────────────────────────────────────────
        text_cleaning_tests = [
            ("Metformin   500mg\r\n\r\nBD",  True,  "whitespace normalised"),
            ("  Aspirin 81mg OD  \n  ",       True,  "leading/trailing stripped"),
        ]
        clean_passed = sum(
            1 for raw, _, _ in text_cleaning_tests
            if not _clean_ocr_text(raw).startswith(" ")
               and "  " not in _clean_ocr_text(raw)
        )

        # ── Abbreviation stripping ────────────────────────────────────────────
        abbrev_passed = sum(
            1 for raw, expected_frag in _ABBREV_STRIP_CASES
            if expected_frag in _clean_prescription_text(raw)
        )

        # Safe tokens must NOT be split
        safe_tok_passed = sum(
            1 for tok in _SAFE_TOKENS
            if _clean_prescription_text(tok).strip() == tok
        )

        # ── Candidate scoring ─────────────────────────────────────────────────
        with_dose  = _ocr_candidate_score("Metformin 500mg daily",   0.9)
        without_dose = _ocr_candidate_score("Metformin drug oral",    0.9)
        clean_text = _ocr_candidate_score("Metformin 500mg BD",       0.85)
        noisy_text = _ocr_candidate_score("M\\e>t@f[o]rmin 500mg BD", 0.85)
        scoring_ok = (with_dose > without_dose) and (clean_text > noisy_text)

        # ── Text analysis (drug extraction from typed/pasted prescription) ────
        tp = fp = fn = 0
        analysis_cases: List[Dict[str, Any]] = []
        latencies: List[float] = []

        for tc in OCR_TEXT_CASES:
            t1 = time.time()
            result = analyze_prescription_text(
                text=tc["text"], disease=tc["disease"], age=tc["age"]
            )
            lat = round((time.time() - t1) * 1000, 1)
            latencies.append(lat)

            found = [
                (m.get("drug") or m.get("normalized") or "").lower()
                for m in result.get("detected_medicines", [])
            ]

            hit = any(any(exp in f for f in found) for exp in tc["expect_any"])

            if hit:
                tp += 1
            else:
                fn += 1

            spurious = sum(
                1 for f in found
                if not any(exp in f for exp in tc["expect_any"])
            )
            fp += spurious

            analysis_cases.append({
                "label":      tc["label"],
                "input_text": tc["text"][:60] + ("…" if len(tc["text"]) > 60 else ""),
                "expected":   tc["expect_any"],
                "found":      found,
                "hit":        hit,
                "latency_ms": lat,
            })

        total_text_cases = len(OCR_TEXT_CASES)
        text_hit_rate = _pct(tp, total_text_cases)
        avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else 0

        # ── Build summary ─────────────────────────────────────────────────────
        preprocessing_score = _pct(
            clean_passed + abbrev_passed + safe_tok_passed + (1 if scoring_ok else 0),
            len(text_cleaning_tests) + len(_ABBREV_STRIP_CASES) + len(_SAFE_TOKENS) + 1,
        )
        overall = round((text_hit_rate + preprocessing_score) / 2, 1)

        print(
            f"    OK engines={engines or ['none']}"
            f"  text_hit={text_hit_rate}%  preproc={preprocessing_score}%"
            f"  avg_latency={avg_lat}ms"
        )

        return {
            "engines_available": engines,
            "tesseract_ok": tesseract_ok,
            "easyocr_ok":   easyocr_ok,
            "preprocessing": {
                "text_cleaning_passed":    f"{clean_passed}/{len(text_cleaning_tests)}",
                "abbreviation_stripping":  f"{abbrev_passed}/{len(_ABBREV_STRIP_CASES)}",
                "safe_tokens_intact":      f"{safe_tok_passed}/{len(_SAFE_TOKENS)}",
                "scoring_logic_correct":   scoring_ok,
                "score_pct":               preprocessing_score,
            },
            "text_analysis": {
                "total_cases":      total_text_cases,
                "hits":             tp,
                "misses":           fn,
                "hit_rate_pct":     text_hit_rate,
                "avg_latency_ms":   avg_lat,
                "cases":            analysis_cases,
            },
            "elapsed_s":  round(time.time() - t0, 2),
            "score_pct":  overall,
            "status":     "PASS" if overall >= 70 and at_least_one else "WARN",
        }
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 6. FAISS — SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════════════

FAISS_TESTS = [
    ("diabetes medication blood sugar",   ["metformin", "insulin", "glipizide"]),
    ("blood pressure hypertension",       ["lisinopril", "amlodipine", "losartan"]),
    ("asthma inhaler breathing",          ["albuterol", "fluticasone", "salbutamol"]),
    ("pain relief anti-inflammatory",     ["ibuprofen", "aspirin", "naproxen"]),
    ("cholesterol statin heart disease",  ["atorvastatin", "simvastatin", "rosuvastatin"]),
]

def eval_faiss() -> Dict[str, Any]:
    print("\n[6/8] FAISS semantic search …")
    t0 = time.time()
    try:
        # Pre-warm the embedding model so the first FAISS query doesn't pay
        # the 10-30 second HuggingFace cold-start penalty in the latency metric.
        from app.ml.embeddings import _get_service, embed_single
        svc = _get_service()
        if svc.is_ready():
            embed_single("warmup")   # one dummy call loads tokenizer + weights

        from app.ml.faiss_store import get_faiss_store
        from app.services.drug_lookup import init_store, _df
        store = get_faiss_store()
        # Standalone mode: try loading from disk, then build from CSV if needed
        if not store.is_ready():
            store.load()
        if not store.is_ready():
            init_store()
            from app.services.drug_lookup import _df as drug_df
            if drug_df is not None and len(drug_df) > 0:
                store.build(drug_df.to_dict("records"))
        if not store.is_ready():
            return {"status": "SKIP", "reason": "FAISS index could not be loaded or built",
                    "score_pct": 0.0, "elapsed_s": round(time.time() - t0, 2)}

        hit_total = total_queries = 0
        cases: List[Dict[str, Any]] = []
        latencies: List[float] = []

        for query, expected_any in FAISS_TESTS:
            t1 = time.time()
            results = store.search(query, top_k=5)
            lat = round((time.time() - t1) * 1000, 1)
            latencies.append(lat)
            total_queries += 1

            got_names = [r["drug_name"].lower() for r in results]
            hit = any(
                any(exp in got for got in got_names)
                for exp in expected_any
            )
            if hit:
                hit_total += 1

            cases.append({
                "query": query,
                "expected_any": expected_any,
                "top5_results": got_names[:5],
                "hit": hit,
                "latency_ms": lat,
                "top_score": round(results[0]["similarity"], 3) if results else 0,
            })

        hit_rate = _pct(hit_total, total_queries)
        avg_lat  = round(sum(latencies) / len(latencies), 1)

        print(f"    OK Hit rate {hit_rate}%  ({store.vector_count()} vectors, avg {avg_lat} ms)")
        return {
            "vector_count": store.vector_count(),
            "dimension": store.status().get("dimension"),
            "hit_rate_pct": hit_rate,
            "avg_latency_ms": avg_lat,
            "cases": cases,
            "elapsed_s": round(time.time() - t0, 2),
            "score_pct": hit_rate,
            "status": "PASS" if hit_rate >= 60 else "WARN",
        }
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 6. LLM — RESPONSE QUALITY
# ══════════════════════════════════════════════════════════════════════════════

LLM_TESTS = [
    {
        "message": "What are the common side effects of metformin?",
        "disease": "diabetes", "age": 55,
        "checks": ["side effect", "nausea", "diarrhea", "consult"],
    },
    {
        "message": "Is lisinopril safe for elderly patients?",
        "disease": "hypertension", "age": 72,
        "checks": ["lisinopril", "blood pressure", "doctor", "pharmacist"],
    },
]

def eval_llm(skip: bool = False) -> Dict[str, Any]:
    print("\n[7/8] LLM response quality …")
    t0 = time.time()
    if skip:
        print("    -> skipped (--no-llm)")
        return {"status": "SKIP", "reason": "--no-llm flag set",
                "score_pct": 0.0, "elapsed_s": 0.0}
    try:
        from app.services.llm_service import get_llm_service, LLMService

        svc: LLMService = get_llm_service()
        if not svc.is_available():
            return {"status": "SKIP", "reason": "LLM not available (check LLM_API_KEY)",
                    "score_pct": 0.0, "elapsed_s": round(time.time() - t0, 2)}

        cases: List[Dict[str, Any]] = []
        check_total = checks_passed = 0

        for tc in LLM_TESTS:
            t1 = time.time()
            answer = svc.generate_response(
                message=tc["message"],
                disease=tc["disease"],
                age=tc["age"],
                matched_drugs=[],
                conversation_history=[],
                intent="side_effects",
            ) or ""
            lat = round((time.time() - t1) * 1000)

            answer_lower = answer.lower()
            passed_checks = [kw for kw in tc["checks"] if kw in answer_lower]
            check_total  += len(tc["checks"])
            checks_passed += len(passed_checks)
            has_disclaimer = any(w in answer_lower for w in
                                 ["consult", "pharmacist", "doctor", "not medical"])

            cases.append({
                "question": tc["message"],
                "disease": tc["disease"],
                "age": tc["age"],
                "latency_ms": lat,
                "checks_passed": f"{len(passed_checks)}/{len(tc['checks'])}",
                "missing_checks": [k for k in tc["checks"] if k not in answer_lower],
                "has_disclaimer": has_disclaimer,
                "answer_excerpt": answer[:300].replace("\n", " "),
            })

        kw_score = _pct(checks_passed, check_total)
        disc_score = _pct(sum(1 for c in cases if c["has_disclaimer"]), len(cases))
        overall = round((kw_score + disc_score) / 2, 1)

        print(f"    OK Keyword score {kw_score}%  Disclaimer {disc_score}%")
        return {
            "keyword_score_pct": kw_score,
            "disclaimer_rate_pct": disc_score,
            "overall_score_pct": overall,
            "cases": cases,
            "elapsed_s": round(time.time() - t0, 2),
            "score_pct": overall,
            "status": "PASS" if overall >= 70 else "WARN",
        }
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 7. PYTEST SUITE
# ══════════════════════════════════════════════════════════════════════════════

def eval_pytest() -> Dict[str, Any]:
    print("\n[8/8] Running pytest suite …")
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/",
             "--tb=no", "-q", "--no-header",
             "--override-ini=addopts=",
             "-p", "no:warnings"],
            cwd=str(ROOT),
            capture_output=True, text=True, timeout=300,
        )
        output = result.stdout + result.stderr

        # Parse "N passed, M failed in Xs"
        import re
        m = re.search(r"(\d+) passed", output)
        passed = int(m.group(1)) if m else 0
        m2 = re.search(r"(\d+) failed", output)
        failed = int(m2.group(1)) if m2 else 0
        total = passed + failed

        score = _pct(passed, total)
        print(f"    OK {passed}/{total} tests passed ({score}%)")
        return {
            "total": total, "passed": passed, "failed": failed,
            "score_pct": score,
            "raw_output": output.strip()[-1200:],
            "elapsed_s": round(time.time() - t0, 2),
            "status": "PASS" if failed == 0 else ("WARN" if score >= 90 else "FAIL"),
        }
    except Exception as exc:
        return {"error": str(exc), "score_pct": 0.0, "status": "FAIL",
                "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def _status_badge(status: str) -> str:
    colors = {"PASS": "#22c55e", "WARN": "#f59e0b", "FAIL": "#ef4444", "SKIP": "#94a3b8"}
    return (f'<span style="background:{colors.get(status,"#94a3b8")};color:#fff;'
            f'padding:3px 10px;border-radius:20px;font-size:12px;font-weight:800;'
            f'letter-spacing:.06em">{status}</span>')

def generate_html(report: Dict[str, Any], out_path: Path) -> None:
    ts     = report["generated_at"]
    comps  = report["components"]

    def section(title: str, key: str, body_html: str) -> str:
        d = comps.get(key, {})
        sc = d.get("score_pct", 0.0)
        st = d.get("status", "SKIP")
        elapsed = d.get("elapsed_s", 0)
        return f"""
        <div class="card">
          <div class="card-head">
            <div>
              <span class="card-title">{title}</span>
              <span class="card-elapsed">{elapsed}s</span>
            </div>
            <div style="display:flex;align-items:center;gap:12px">
              <div class="score-ring" style="--pct:{sc};--clr:{_color(sc)}">
                <span>{int(sc)}%</span>
              </div>
              {_status_badge(st)}
            </div>
          </div>
          <div class="card-body">{body_html}</div>
        </div>"""

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    d = comps.get("dataset", {})
    cov = d.get("field_coverage_pct", {})
    cov_rows = "".join(
        f'<tr><td>{k.replace("_"," ").title()}</td>'
        f'<td><div class="bar-track"><div class="bar-fill" style="width:{v}%;background:{_color(v)}"></div></div></td>'
        f'<td style="font-weight:700;color:{_color(v)}">{v}%</td></tr>'
        for k, v in cov.items()
    )
    ds_body = f"""
      <div class="metric-row">
        <div class="metric-box"><div class="metric-val">{d.get("total_drugs","—")}</div><div class="metric-lbl">Total Drugs</div></div>
        <div class="metric-box"><div class="metric-val">{d.get("duplicate_records","—")}</div><div class="metric-lbl">Duplicates</div></div>
        <div class="metric-box"><div class="metric-val">{d.get("average_coverage_pct","—")}%</div><div class="metric-lbl">Avg Coverage</div></div>
      </div>
      <table class="tbl"><thead><tr><th>Field</th><th>Coverage</th><th>%</th></tr></thead>
      <tbody>{cov_rows}</tbody></table>"""

    # ── 2. Drug Lookup ──────────────────────────────────────────────────────
    lu = comps.get("drug_lookup", {})
    lu_rows = "".join(
        f'<tr class="{"pass" if c["pass"] else "fail"}">'
        f'<td>{c["query"]}</td><td>{c["label"]}</td>'
        f'<td>{c["expected"]}</td><td>{c["got"]}</td>'
        f'<td>{c["latency_ms"]}ms</td>'
        f'<td>{"OK" if c["pass"] else "FAIL"}</td></tr>'
        for c in lu.get("cases", [])
    )
    lu_body = f"""
      <div class="metric-row">
        <div class="metric-box"><div class="metric-val">{lu.get("passed","—")}/{lu.get("total","—")}</div><div class="metric-lbl">Passed</div></div>
        <div class="metric-box"><div class="metric-val">{lu.get("accuracy_pct","—")}%</div><div class="metric-lbl">Accuracy</div></div>
        <div class="metric-box"><div class="metric-val">{lu.get("avg_latency_ms","—")}ms</div><div class="metric-lbl">Avg Latency</div></div>
      </div>
      <table class="tbl"><thead><tr><th>Query</th><th>Type</th><th>Expected</th><th>Got</th><th>Latency</th><th>Pass</th></tr></thead>
      <tbody>{lu_rows}</tbody></table>"""

    # ── 3. NER ──────────────────────────────────────────────────────────────
    ne = comps.get("ner", {})
    ner_rows = "".join(
        f'<tr><td title="{c["text"]}">{c["text"][:60]}…</td>'
        f'<td>{", ".join(c["expected"])}</td>'
        f'<td>{", ".join(c["extracted"])}</td>'
        f'<td>TP:{c["tp"]} FP:{c["fp"]} FN:{c["fn"]}</td></tr>'
        for c in ne.get("cases", [])
    )
    ner_body = f"""
      <div class="metric-row">
        <div class="metric-box"><div class="metric-val">{ne.get("precision_pct","—")}%</div><div class="metric-lbl">Precision</div></div>
        <div class="metric-box"><div class="metric-val">{ne.get("recall_pct","—")}%</div><div class="metric-lbl">Recall</div></div>
        <div class="metric-box"><div class="metric-val">{ne.get("f1_pct","—")}%</div><div class="metric-lbl">F1 Score</div></div>
      </div>
      <table class="tbl"><thead><tr><th>Prescription Text</th><th>Expected Drugs</th><th>Extracted</th><th>Scores</th></tr></thead>
      <tbody>{ner_rows}</tbody></table>"""

    # ── 4. Safety ──────────────────────────────────────────────────────────
    sa = comps.get("safety", {})
    sa_body = f"""
      <div class="metric-row">
        <div class="metric-box"><div class="metric-val">{sa.get("emergency_recall_pct","—")}%</div><div class="metric-lbl">Emergency Recall</div></div>
        <div class="metric-box"><div class="metric-val">{sa.get("safe_precision_pct","—")}%</div><div class="metric-lbl">Safe-phrase Precision</div></div>
        <div class="metric-box"><div class="metric-val">{sa.get("true_positives","—")}/{sa.get("total_emergency_tests","—")}</div><div class="metric-lbl">Emergencies Caught</div></div>
        <div class="metric-box"><div class="metric-val">{sa.get("false_positives","—")}/{sa.get("total_safe_tests","—")}</div><div class="metric-lbl">False Positives</div></div>
      </div>"""

    # ── 5. FAISS ───────────────────────────────────────────────────────────
    fa = comps.get("faiss", {})
    fa_rows = "".join(
        f'<tr class="{"pass" if c.get("hit") else "fail"}">'
        f'<td>{c["query"]}</td>'
        f'<td>{", ".join(c["expected_any"])}</td>'
        f'<td>{", ".join(c["top5_results"][:3])}</td>'
        f'<td>{c.get("top_score","—")}</td>'
        f'<td>{c.get("latency_ms","—")}ms</td>'
        f'<td>{"OK" if c.get("hit") else "FAIL"}</td></tr>'
        for c in fa.get("cases", [])
    )
    fa_body = f"""
      <div class="metric-row">
        <div class="metric-box"><div class="metric-val">{fa.get("vector_count","—")}</div><div class="metric-lbl">Indexed Vectors</div></div>
        <div class="metric-box"><div class="metric-val">{fa.get("dimension","—")}</div><div class="metric-lbl">Dimensions</div></div>
        <div class="metric-box"><div class="metric-val">{fa.get("hit_rate_pct","—")}%</div><div class="metric-lbl">Hit Rate</div></div>
        <div class="metric-box"><div class="metric-val">{fa.get("avg_latency_ms","—")}ms</div><div class="metric-lbl">Avg Latency</div></div>
      </div>
      {'<table class="tbl"><thead><tr><th>Query</th><th>Expected (any)</th><th>Top-3 Results</th><th>Score</th><th>Latency</th><th>Hit</th></tr></thead><tbody>' + fa_rows + '</tbody></table>' if fa_rows else '<p class="muted">FAISS index not loaded.</p>'}"""

    # ── 6. LLM ────────────────────────────────────────────────────────────
    ll = comps.get("llm", {})
    ll_cases = ll.get("cases", [])
    ll_rows = "".join(
        f'<tr><td>{c["question"][:60]}</td>'
        f'<td>{c["disease"]} / age {c["age"]}</td>'
        f'<td>{c["checks_passed"]}</td>'
        f'<td>{"OK" if c["has_disclaimer"] else "FAIL"}</td>'
        f'<td>{c["latency_ms"]}ms</td></tr>'
        for c in ll_cases
    )
    ll_body = f"""
      <div class="metric-row">
        <div class="metric-box"><div class="metric-val">{ll.get("keyword_score_pct","—")}%</div><div class="metric-lbl">Keyword Score</div></div>
        <div class="metric-box"><div class="metric-val">{ll.get("disclaimer_rate_pct","—")}%</div><div class="metric-lbl">Disclaimer Rate</div></div>
      </div>
      {'<table class="tbl"><thead><tr><th>Question</th><th>Context</th><th>Keywords</th><th>Disclaimer</th><th>Latency</th></tr></thead><tbody>' + ll_rows + '</tbody></table>' if ll_rows else '<p class="muted">LLM evaluation skipped — run without --no-llm to test.</p>'}"""

    # ── 5. OCR ────────────────────────────────────────────────────────────
    oc = comps.get("ocr", {})
    pp = oc.get("preprocessing", {})
    ta = oc.get("text_analysis", {})
    oc_eng = ", ".join(oc.get("engines_available", [])) or "none detected"
    oc_case_rows = "".join(
        f'<tr class="{"pass" if c.get("hit") else "fail"}">'
        f'<td>{c["label"]}</td>'
        f'<td title="{c["input_text"]}">{c["input_text"][:50]}...</td>'
        f'<td>{", ".join(c["expected"])}</td>'
        f'<td>{", ".join(c["found"]) or "(none)"}</td>'
        f'<td>{c["latency_ms"]}ms</td>'
        f'<td>{"OK" if c.get("hit") else "FAIL"}</td></tr>'
        for c in ta.get("cases", [])
    )
    oc_body = f"""
      <div class="metric-row">
        <div class="metric-box"><div class="metric-val" style="font-size:14px">{oc_eng}</div><div class="metric-lbl">Engines</div></div>
        <div class="metric-box"><div class="metric-val">{ta.get("hit_rate_pct","—")}%</div><div class="metric-lbl">Text Hit Rate</div></div>
        <div class="metric-box"><div class="metric-val">{pp.get("score_pct","—")}%</div><div class="metric-lbl">Preprocessing</div></div>
        <div class="metric-box"><div class="metric-val">{ta.get("avg_latency_ms","—")}ms</div><div class="metric-lbl">Avg Latency</div></div>
      </div>
      <div class="metric-row" style="flex-wrap:wrap;gap:8px;font-size:13px">
        <span style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:4px 10px">Abbreviation stripping: {pp.get("abbreviation_stripping","—")}</span>
        <span style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:4px 10px">Safe tokens intact: {pp.get("safe_tokens_intact","—")}</span>
        <span style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:4px 10px">Scoring logic: {"OK" if pp.get("scoring_logic_correct") else "FAIL"}</span>
        <span style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:4px 10px">Text cleaning: {pp.get("text_cleaning_passed","—")}</span>
      </div>
      {'<table class="tbl"><thead><tr><th>Case</th><th>Input text</th><th>Expected</th><th>Found</th><th>Latency</th><th>Result</th></tr></thead><tbody>' + oc_case_rows + '</tbody></table>' if oc_case_rows else '<p class="muted">OCR evaluation not run.</p>'}"""

    # ── 8. Pytest ─────────────────────────────────────────────────────────
    py = comps.get("pytest", {})
    py_body = f"""
      <div class="metric-row">
        <div class="metric-box"><div class="metric-val">{py.get("passed","—")}</div><div class="metric-lbl">Tests Passed</div></div>
        <div class="metric-box"><div class="metric-val">{py.get("failed","—")}</div><div class="metric-lbl">Tests Failed</div></div>
        <div class="metric-box"><div class="metric-val">{py.get("score_pct","—")}%</div><div class="metric-lbl">Pass Rate</div></div>
      </div>
      <pre class="code-block">{py.get("raw_output","")}</pre>"""

    # ── Overall summary ───────────────────────────────────────────────────
    scored = [v for v in comps.values()
              if v.get("status") not in ("SKIP",) and "score_pct" in v and v.get("score_pct", 0) > 0]
    overall = round(sum(v["score_pct"] for v in scored) / len(scored), 1) if scored else 0.0

    summary_rows = "".join(
        f'<tr><td><strong>{k.replace("_"," ").title()}</strong></td>'
        f'<td style="color:{_color(v.get("score_pct",0))};font-weight:800">'
        f'{v.get("score_pct","—")}%</td>'
        f'<td>{_grade(v.get("score_pct",0)) if v.get("status")!="SKIP" else "—"}</td>'
        f'<td>{_status_badge(v.get("status","—"))}</td></tr>'
        for k, v in comps.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MediAssist Bot — Evaluation Report</title>
<style>
  :root{{
    --bg:#f1f5f9;--card:#fff;--text:#0f172a;--muted:#64748b;
    --border:#e2e8f0;--accent:#2563eb;--radius:16px;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:Inter,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);padding:32px}}
  .page-header{{max-width:1100px;margin:0 auto 28px;display:flex;justify-content:space-between;align-items:flex-end}}
  .page-title{{font-size:28px;font-weight:800;letter-spacing:-.03em}}
  .page-sub{{color:var(--muted);font-size:14px;margin-top:4px}}
  .overall-badge{{background:var(--accent);color:#fff;border-radius:12px;padding:14px 24px;text-align:center}}
  .overall-score{{font-size:36px;font-weight:900;line-height:1}}
  .overall-label{{font-size:12px;opacity:.8;letter-spacing:.06em;text-transform:uppercase}}
  .grid{{max-width:1100px;margin:0 auto;display:grid;gap:20px}}
  .card{{background:var(--card);border-radius:var(--radius);border:1px solid var(--border);overflow:hidden}}
  .card-head{{display:flex;justify-content:space-between;align-items:center;padding:16px 20px;border-bottom:1px solid var(--border);background:#f8fafc}}
  .card-title{{font-size:16px;font-weight:800}}
  .card-elapsed{{color:var(--muted);font-size:12px;margin-left:8px}}
  .card-body{{padding:18px 20px}}
  .score-ring{{
    width:52px;height:52px;border-radius:50%;
    background:conic-gradient(var(--clr,#22c55e) calc(var(--pct,0)*1%),#e2e8f0 0);
    display:flex;align-items:center;justify-content:center;position:relative
  }}
  .score-ring::before{{content:'';position:absolute;inset:6px;border-radius:50%;background:#fff}}
  .score-ring span{{position:relative;font-weight:800;font-size:12px;z-index:1}}
  .metric-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}}
  .metric-box{{flex:1;min-width:100px;background:#f8fafc;border:1px solid var(--border);border-radius:12px;padding:12px 16px;text-align:center}}
  .metric-val{{font-size:22px;font-weight:800;color:var(--accent)}}
  .metric-lbl{{font-size:12px;color:var(--muted);margin-top:2px}}
  .tbl{{width:100%;border-collapse:collapse;font-size:13px}}
  .tbl th{{background:#f1f5f9;text-align:left;padding:8px 12px;font-weight:700;border-bottom:2px solid var(--border)}}
  .tbl td{{padding:7px 12px;border-bottom:1px solid var(--border)}}
  .tbl tr.pass td{{background:#f0fdf4}}
  .tbl tr.fail td{{background:#fef2f2}}
  .bar-track{{background:#e2e8f0;border-radius:999px;height:8px;width:140px}}
  .bar-fill{{height:8px;border-radius:999px}}
  .muted{{color:var(--muted);font-size:14px}}
  .code-block{{background:#0f172a;color:#e2e8f0;border-radius:12px;padding:14px;font-size:12px;
    overflow:auto;max-height:200px;white-space:pre-wrap;word-break:break-all}}
  summary-table table{{width:100%}}
</style>
</head>
<body>
<div class="page-header">
  <div>
    <div class="page-title">MediAssist Bot — Evaluation Report</div>
    <div class="page-sub">Generated {ts} · Llama 3.1 8B via Groq · 6 diseases · FAISS + SciSpaCy</div>
  </div>
  <div class="overall-badge">
    <div class="overall-score">{overall}%</div>
    <div class="overall-label">Overall Score</div>
  </div>
</div>

<div class="grid">
  <!-- Summary -->
  <div class="card">
    <div class="card-head">
      <span class="card-title">Component Summary</span>
      <span class="muted">{len(comps)} components evaluated</span>
    </div>
    <div class="card-body">
      <table class="tbl" style="margin:0">
        <thead><tr><th>Component</th><th>Score</th><th>Grade</th><th>Status</th></tr></thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </div>
  </div>

  {section("1 · Dataset Quality", "dataset", ds_body)}
  {section("2 · Drug Lookup Accuracy", "drug_lookup", lu_body)}
  {section("3 · NER Drug Extraction", "ner", ner_body)}
  {section("4 · Safety Layer", "safety", sa_body)}
  {section("5 · OCR Pipeline", "ocr", oc_body)}
  {section("6 · FAISS Semantic Search", "faiss", fa_body)}
  {section("7 · LLM Response Quality", "llm", ll_body)}
  {section("8 · Automated Test Suite", "pytest", py_body)}
</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"\n  HTML report -> {out_path}")


def generate_markdown(report: Dict[str, Any], out_path: Path) -> None:
    c  = report["components"]
    ts = report["generated_at"]

    scored = [v for v in c.values()
              if v.get("status") not in ("SKIP",) and v.get("score_pct", 0) > 0]
    overall = round(sum(v["score_pct"] for v in scored) / len(scored), 1) if scored else 0.0

    def badge(pct: float, st: str) -> str:
        g = _grade(pct) if st != "SKIP" else "SKIP"
        icons = {"EXCELLENT": "[+]", "GOOD": "[~]", "FAIR": "[!]",
                 "POOR": "[X]", "SKIP": "[-]"}
        return f"{icons.get(g,'[-]')} **{pct}%** — {g}"

    lines = [
        "# MediAssist Bot — Evaluation Report",
        "",
        f"> Generated: {ts}",
        "",
        f"## Overall Score: **{overall}%**",
        "",
        "| Component | Score | Grade | Status |",
        "|---|---|---|---|",
    ]
    for k, v in c.items():
        sc = v.get("score_pct", 0.0)
        st = v.get("status", "—")
        gr = _grade(sc) if st != "SKIP" else "—"
        lines.append(f"| {k.replace('_',' ').title()} | {sc}% | {gr} | {st} |")

    lines += [
        "",
        "## Component Details",
        "",
        f"### Dataset Quality",
        f"- Total drugs: **{c.get('dataset',{}).get('total_drugs','—')}**",
        f"- Avg field coverage: **{c.get('dataset',{}).get('average_coverage_pct','—')}%**",
        f"- Duplicates: **{c.get('dataset',{}).get('duplicate_records','—')}**",
        "",
        f"### Drug Lookup",
        f"- Accuracy: **{c.get('drug_lookup',{}).get('accuracy_pct','—')}%** ({c.get('drug_lookup',{}).get('passed','—')}/{c.get('drug_lookup',{}).get('total','—')} passed)",
        f"- Avg latency: **{c.get('drug_lookup',{}).get('avg_latency_ms','—')} ms**",
        "",
        f"### NER Drug Extraction",
        f"- Precision: **{c.get('ner',{}).get('precision_pct','—')}%**",
        f"- Recall: **{c.get('ner',{}).get('recall_pct','—')}%**",
        f"- F1 Score: **{c.get('ner',{}).get('f1_pct','—')}%**",
        "",
        f"### Safety Layer",
        f"- Emergency recall: **{c.get('safety',{}).get('emergency_recall_pct','—')}%**",
        f"- Safe-phrase precision: **{c.get('safety',{}).get('safe_precision_pct','—')}%**",
        "",
        f"### OCR Pipeline",
        f"- Engines available: **{', '.join(c.get('ocr',{}).get('engines_available', [])) or 'none'}**",
        f"- Text analysis hit rate: **{c.get('ocr',{}).get('text_analysis',{}).get('hit_rate_pct','—')}%**",
        f"- Preprocessing score: **{c.get('ocr',{}).get('preprocessing',{}).get('score_pct','—')}%**",
        f"- Avg latency: **{c.get('ocr',{}).get('text_analysis',{}).get('avg_latency_ms','—')} ms**",
        "",
        f"### FAISS Semantic Search",
        f"- Vectors indexed: **{c.get('faiss',{}).get('vector_count','—')}**",
        f"- Semantic hit rate: **{c.get('faiss',{}).get('hit_rate_pct','—')}%**",
        f"- Avg query latency: **{c.get('faiss',{}).get('avg_latency_ms','—')} ms**",
        "",
        f"### Automated Tests (pytest)",
        f"- Passed: **{c.get('pytest',{}).get('passed','—')}/{c.get('pytest',{}).get('total','—')}**",
        f"- Pass rate: **{c.get('pytest',{}).get('score_pct','—')}%**",
        "",
        "---",
        "*MediAssist Bot — AI-powered medication assistant for chronic disease patients*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Markdown report -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MediAssist Bot Evaluation Suite")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM API calls")
    parser.add_argument("--output", default=str(ROOT / "reports"),
                        help="Output directory (default: reports/)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MediAssist Bot — Evaluation Suite")
    print(f"  {_now()}")
    print("=" * 60)

    total_t0 = time.time()

    report: Dict[str, Any] = {
        "generated_at": _now(),
        "components": {
            "dataset":    eval_dataset(),
            "drug_lookup": eval_drug_lookup(),
            "ner":        eval_ner(),
            "safety":     eval_safety(),
            "conversation_memory": eval_conversation_memory(),
            "chat_engine": eval_chat_engine(),
            "rag":        eval_rag(),
            "activity_metrics": eval_activity_metrics(),
            "ocr":        eval_ocr(),
            "faiss":      eval_faiss(),
            "llm":        eval_llm(skip=args.no_llm),
            "pytest":     eval_pytest(),
        },
        "total_elapsed_s": round(time.time() - total_t0, 1),
    }

    print("\n[>] Generating reports …")

    json_path = out_dir / "evaluation_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  JSON report   -> {json_path}")

    generate_html(report, out_dir / "evaluation_report.html")
    generate_markdown(report, out_dir / "evaluation_summary.md")

    # ── Final summary ──────────────────────────────────────────────────────
    comps = report["components"]
    scored = [v for v in comps.values()
              if v.get("status") not in ("SKIP",) and v.get("score_pct", 0) > 0]
    overall = round(sum(v["score_pct"] for v in scored) / len(scored), 1) if scored else 0.0

    print("\n" + "=" * 60)
    print(f"  OVERALL SCORE : {overall}%  ({_grade(overall)})")
    print(f"  Total time    : {report['total_elapsed_s']}s")
    print("=" * 60)
    print("\nOpen reports/evaluation_report.html in a browser for the full visual report.")


if __name__ == "__main__":
    main()
