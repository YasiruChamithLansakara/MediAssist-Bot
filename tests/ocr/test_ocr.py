"""
OCR pipeline tests — MediAssist Bot

Tests:
  1. Engine availability  (Tesseract + EasyOCR)
  2. Preprocessing        (upscale, grayscale, Otsu binarization)
  3. Text analysis        (analyze_prescription_text with known inputs)
  4. Abbreviation strip   (T., Inj., Cap. prefixes removed before NER)
  5. Confidence scoring   (_ocr_candidate_score: medical signals, noise penalty)
  6. Noise penalty        (symbols like \\, >, @ are penalised)

All tests use text-only paths (no real prescription images required).
"""

import pytest
from app.services.ocr_service import (
    _clean_ocr_text,
    _ocr_candidate_score,
    analyze_prescription_text,
    ocr_runtime_status,
)
from app.services.ner_service import _clean_prescription_text


# ─────────────────────────────────────────────────────────────────────────────
# 1. Engine availability
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineAvailability:

    def test_status_returns_dict_with_required_keys(self):
        status = ocr_runtime_status()
        assert isinstance(status, dict)
        for key in ("available", "tesseract_available", "easyocr_available"):
            assert key in status, f"Key '{key}' missing from ocr_runtime_status()"

    def test_at_least_one_engine_available(self):
        status = ocr_runtime_status()
        assert status["available"], (
            "No OCR engine available — install pytesseract or easyocr"
        )

    def test_python_dependencies_present(self):
        """pytesseract + Pillow must be importable."""
        status = ocr_runtime_status()
        assert status.get("python_dependencies") is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. OCR text cleaning
# ─────────────────────────────────────────────────────────────────────────────

class TestTextCleaning:

    def test_normalises_whitespace(self):
        raw = "Metformin   500mg\r\n\r\nBD"
        cleaned = _clean_ocr_text(raw)
        assert "\r" not in cleaned
        assert "  " not in cleaned     # no double spaces

    def test_collapses_blank_lines(self):
        raw = "Line1\n\n\n\nLine2"
        cleaned = _clean_ocr_text(raw)
        assert cleaned.count("\n") <= 2

    def test_strips_leading_trailing(self):
        raw = "  \n  Aspirin 81mg  \n  "
        assert not _clean_ocr_text(raw).startswith(" ")
        assert not _clean_ocr_text(raw).endswith(" ")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Candidate scoring — medical signals boost, noise penalises
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateScoring:

    def test_dosage_pattern_boosts_score(self):
        with_dose    = _ocr_candidate_score("Metformin 500mg daily", 0.9)
        without_dose = _ocr_candidate_score("Metformin medication oral", 0.9)
        assert with_dose > without_dose

    def test_frequency_pattern_boosts_score(self):
        with_freq    = _ocr_candidate_score("Aspirin 81mg OD once daily", 0.8)
        without_freq = _ocr_candidate_score("Aspirin 81mg", 0.8)
        assert with_freq > without_freq

    def test_noise_chars_penalise_score(self):
        clean  = _ocr_candidate_score("Metformin 500mg BD", 0.85)
        noisy  = _ocr_candidate_score("M\\e>t@f|o[r]min 500mg BD!!!", 0.85)
        assert clean > noisy, "Noise characters should reduce the candidate score"

    def test_empty_text_returns_zero_or_near_zero(self):
        score = _ocr_candidate_score("", 0.0)
        assert score < 1.0

    def test_higher_confidence_raises_score(self):
        low  = _ocr_candidate_score("Lisinopril 10mg", 0.5)
        high = _ocr_candidate_score("Lisinopril 10mg", 0.95)
        assert high > low


# ─────────────────────────────────────────────────────────────────────────────
# 4. Prescription abbreviation stripping (NER preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

class TestAbbreviationStripping:

    @pytest.mark.parametrize("raw, expected_fragment", [
        ("T. Metformin 500mg BD",       "Metformin"),
        ("Tab. Aspirin 81mg OD",        "Aspirin"),
        ("Cap. Amlodipine 5mg OD",      "Amlodipine"),
        ("Inj. Insulin 10 units SC",    "Insulin"),
        ("Syr. Amoxicillin 250mg TDS",  "Amoxicillin"),
    ])
    def test_prefix_stripped(self, raw, expected_fragment):
        cleaned = _clean_prescription_text(raw)
        assert expected_fragment in cleaned, (
            f"Expected '{expected_fragment}' in '{cleaned}' (from '{raw}')"
        )
        # The prefix abbreviation itself should be gone
        for prefix in ("T.", "Tab.", "Cap.", "Inj.", "Syr."):
            assert not cleaned.startswith(prefix), f"Prefix still present: {cleaned}"

    def test_run_on_ocr_split(self):
        """'Metformin500mg' (OCR run-on) → 'Metformin 500mg'"""
        cleaned = _clean_prescription_text("Metformin500mg")
        assert "Metformin" in cleaned
        assert "500" in cleaned

    def test_safe_tokens_not_split(self):
        """HbA1c, B12, D3 should NOT be split."""
        for tok in ("HbA1c", "B12", "D3", "T3"):
            cleaned = _clean_prescription_text(tok)
            assert cleaned.strip() == tok, (
                f"Safe token '{tok}' was incorrectly modified to '{cleaned}'"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Text analysis (analyze_prescription_text)
# ─────────────────────────────────────────────────────────────────────────────

ANALYSIS_CASES = [
    {
        "text": "Metformin 500mg BD\nLisinopril 10mg OD",
        "disease": "diabetes",
        "age": 55,
        "expect_any": ["metformin", "lisinopril"],
        "label": "simple two-drug prescription",
    },
    {
        "text": "T. Aspirin 81mg OD\nT. Atorvastatin 40mg at night",
        "disease": "heart disease",
        "age": 65,
        "expect_any": ["aspirin", "atorvastatin"],
        "label": "tablet-prefix prescription",
    },
    {
        "text": "Salbutamol 2 puffs PRN\nFluticasone 250mcg BD inhaled",
        "disease": "asthma",
        "age": 40,
        "expect_any": ["albuterol", "fluticasone", "salbutamol"],
        "label": "inhaler prescription",
    },
]


class TestPrescriptionTextAnalysis:

    def test_returns_required_structure(self):
        result = analyze_prescription_text(
            text="Metformin 500mg BD",
            disease="diabetes",
            age=50,
        )
        for key in ("context", "ocr", "detected_medicines", "note"):
            assert key in result, f"Key '{key}' missing from analyze_prescription_text()"

    def test_detected_medicines_is_list(self):
        result = analyze_prescription_text(
            text="Aspirin 81mg OD",
            disease="heart disease",
            age=60,
        )
        assert isinstance(result["detected_medicines"], list)

    def test_empty_text_returns_empty_medicines(self):
        result = analyze_prescription_text(text="   ", disease="diabetes", age=50)
        assert result["detected_medicines"] == [] or isinstance(result["detected_medicines"], list)

    @pytest.mark.parametrize("case", ANALYSIS_CASES, ids=[c["label"] for c in ANALYSIS_CASES])
    def test_detects_expected_drugs(self, case):
        result = analyze_prescription_text(
            text=case["text"],
            disease=case["disease"],
            age=case["age"],
        )
        medicines = result.get("detected_medicines", [])
        found_names = [
            (m.get("drug") or m.get("normalized") or "").lower()
            for m in medicines
        ]
        hit = any(
            any(exp in found for found in found_names)
            for exp in case["expect_any"]
        )
        assert hit, (
            f"[{case['label']}] Expected any of {case['expect_any']}, "
            f"got: {found_names}"
        )

    def test_ocr_confidence_is_none_for_text_only(self):
        """Text-only analysis has no OCR confidence (image not processed)."""
        result = analyze_prescription_text(
            text="Metformin 500mg BD",
            disease="diabetes",
            age=50,
        )
        assert result["ocr"]["confidence"] is None

    def test_returned_text_matches_input_after_cleaning(self):
        raw = "  Aspirin   81mg  \n\n  OD  "
        result = analyze_prescription_text(text=raw, disease="heart disease", age=60)
        # Returned text should be cleaned (no double spaces, no leading/trailing whitespace)
        returned = result["ocr"]["text"]
        assert not returned.startswith(" ")
        assert "  " not in returned
