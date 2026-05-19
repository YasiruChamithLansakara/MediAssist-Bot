from __future__ import annotations

import os
import re
import shutil
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract

# NEW: updated NER integration (SciSpaCy)
from app.services.ner_service import extract_medication_entities

# OPTIONAL: future FAISS hook (safe import)
try:
    from app.services.rag_service import RAGService
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False


# =====================================================
# CONFIGURATION
# =====================================================
class OCRDependencyError(RuntimeError):
    pass


class OCRImageError(ValueError):
    pass


# =====================================================
# TESSERACT CONFIG
# =====================================================
def _configure_tesseract():
    cmd = os.getenv("TESSERACT_CMD", "").strip()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


# =====================================================
# IMAGE PREPROCESSING (IMPROVED)
# =====================================================
def _open_image(image_bytes: bytes):
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise OCRImageError("Invalid image file")


def _preprocess(image):
    width, height = image.size

    # upscale small images
    if max(width, height) < 1500:
        scale = 2.5
        image = image.resize((int(width * scale), int(height * scale)))

    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.4)

    return image.filter(ImageFilter.SHARPEN)


# =====================================================
# OCR CORE ENGINE
# =====================================================
def _run_ocr(image) -> Tuple[str, float]:
    _configure_tesseract()

    text = pytesseract.image_to_string(image)

    # confidence extraction
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confs = [float(c) for c in data.get("conf", []) if c != "-1"]
        confidence = sum(confs) / len(confs) / 100 if confs else 0.0
    except Exception:
        confidence = 0.0

    return text, round(confidence, 3)


# =====================================================
# CLEAN TEXT
# =====================================================
def _clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# =====================================================
# NER + MEDICAL EXTRACTION (NEW PIPELINE)
# =====================================================
def _extract_medical_entities(text: str, disease: str, age: int):
    return extract_medication_entities(
        text=text,
        disease=disease,
        age=age,
        max_entities=8,
    )


# =====================================================
# OPTIONAL FAISS ENRICHMENT
# =====================================================
def _faiss_context(query: str):
    if not RAG_AVAILABLE:
        return []

    try:
        rag = RAGService()
        return rag.retrieve_context(query, top_k=2)
    except Exception:
        return []


# =====================================================
# MAIN OCR FUNCTION
# =====================================================
def ocr_prescription_image(
    *,
    image_bytes: bytes,
    filename: str,
    disease: str,
    age: int,
    request_id: str = "",
) -> Dict[str, Any]:

    image = _preprocess(_open_image(image_bytes))

    text, confidence = _run_ocr(image)
    cleaned_text = _clean(text)

    # STEP 1: NER extraction (SciSpaCy)
    entities = _extract_medical_entities(cleaned_text, disease, age)

    # STEP 2: FAISS enrichment (optional)
    faiss_results = []
    for e in entities:
        faiss_results.extend(_faiss_context(e.get("text", "")))

    # STEP 3: format output
    result = {
        "context": {
            "disease": disease,
            "age": age,
        },
        "file": {
            "name": filename,
        },
        "ocr": {
            "text": cleaned_text,
            "confidence": confidence,
        },
        "ner": {
            "entities": entities,
        },
        "faiss": {
            "matches": faiss_results,
        },
        "pipeline": {
            "ocr": True,
            "ner": True,
            "faiss": RAG_AVAILABLE,
        },
        "note": "OCR + NER + FAISS pipeline (educational medical system)",
    }

    if request_id:
        result["request_id"] = request_id

    return result


# =====================================================
# STATUS CHECK
# =====================================================
def ocr_runtime_status():
    return {
        "tesseract": shutil.which("tesseract") is not None,
        "ner": True,
        "faiss": RAG_AVAILABLE,
    }