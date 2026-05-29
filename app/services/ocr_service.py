# Improve by Yasiru
from __future__ import annotations

import os
import re
import shutil
import threading
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from app.services.ner_service import DOSAGE_RE, FREQUENCY_RE, ROUTE_RE
from app.services.ner_service import extract_medical_entities

try:
    import pytesseract
    from PIL import Image, UnidentifiedImageError
except Exception:  # pragma: no cover
    pytesseract = None
    Image = None

    class UnidentifiedImageError(Exception):
        pass

try:
    import easyocr  # type: ignore[import]
except Exception:  # pragma: no cover
    easyocr = None


class OCRDependencyError(RuntimeError):
    pass


class OCRImageError(ValueError):
    pass


_easyocr_reader = None
_easyocr_warmup_done = False


def _configure_tesseract() -> None:
    """
    On Windows, set TESSERACT_CMD to the full tesseract.exe path.
    Example: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
    """
    if pytesseract is None:
        return
    cmd = os.getenv("TESSERACT_CMD", "").strip()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def _easyocr_available() -> bool:
    return easyocr is not None and Image is not None


def _get_easyocr_reader():
    global _easyocr_reader
    if not _easyocr_available():
        return None
    if _easyocr_reader is not None:
        return _easyocr_reader
    try:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception:
        _easyocr_reader = None
    return _easyocr_reader


def warmup_easyocr() -> None:
    """
    Pre-load the EasyOCR model in a background thread at startup.

    Without this, the first prescription scan triggers a 20-30s model load.
    Call this once from the FastAPI lifespan startup handler.
    """
    global _easyocr_warmup_done
    if _easyocr_warmup_done or not _easyocr_available():
        return

    def _load():
        global _easyocr_warmup_done
        import logging
        log = logging.getLogger("mediassist.ocr")
        log.info("EasyOCR: warming up model in background …")
        reader = _get_easyocr_reader()
        _easyocr_warmup_done = True
        if reader is not None:
            log.info("EasyOCR: model ready")
        else:
            log.warning("EasyOCR: model failed to load — Tesseract-only fallback")

    threading.Thread(target=_load, daemon=True, name="easyocr-warmup").start()


def _clean_ocr_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _open_image(image_bytes: bytes):
    if Image is None:
        raise OCRDependencyError("OCR image dependency is not installed. Install Pillow.")
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise OCRImageError("Uploaded file is not a readable image.") from exc


def _otsu_threshold(image) -> int:
    """Compute Otsu's binarisation threshold using only numpy (no scikit-image)."""
    try:
        import numpy as np
        arr = np.array(image, dtype=np.int32).flatten()
        hist = np.bincount(arr, minlength=256).astype(np.float64)
        total = arr.size
        sum_total = float(np.dot(np.arange(256), hist))
        best, weight_bg, sum_bg = 0.0, 0.0, 0.0
        best_thresh = 127
        for t in range(256):
            weight_bg += hist[t]
            weight_fg = total - weight_bg
            if weight_bg == 0 or weight_fg == 0:
                continue
            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var > best:
                best = var
                best_thresh = t
        return int(best_thresh)
    except Exception:
        return 127  # safe fallback


def _preprocess_base(image):
    """
    Shared first stage: upscale + grayscale + auto-contrast.
    Returned image is suitable for EasyOCR directly.
    """
    if Image is None:
        return image

    from PIL import ImageFilter, ImageOps

    width, height = image.size
    largest_side = max(width, height)
    if largest_side and largest_side < 1800:
        scale = min(3.0, 1800 / largest_side)
        image = image.resize(
            (int(width * scale), int(height * scale)),
            resample=Image.Resampling.LANCZOS,
        )

    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image, cutoff=1)
    return image


def _preprocess_for_tesseract(image):
    """
    Tesseract-specific extra steps on top of the base image:
    Otsu binarisation + sharpen gives clean black-on-white text
    that maximises Tesseract accuracy.
    """
    if Image is None:
        return image

    from PIL import ImageFilter

    thresh = _otsu_threshold(image)
    image = image.point(lambda px: 255 if px > thresh else 0, "L")
    return image.filter(ImageFilter.SHARPEN)


def _preprocess_image(image):
    """Full pipeline for Tesseract (backward-compat wrapper)."""
    return _preprocess_for_tesseract(_preprocess_base(image))


def ocr_runtime_status() -> Dict[str, Any]:
    cmd = os.getenv("TESSERACT_CMD", "").strip()
    executable = cmd or shutil.which("tesseract") or ""
    tesseract_ok = pytesseract is not None and Image is not None and bool(executable)
    easyocr_ready = _easyocr_reader is not None  # only True after warmup or first use
    configured = tesseract_ok or easyocr_ready
    return {
        "available": configured,          # key the frontend checks
        "configured": configured,         # backward-compat alias
        "python_dependencies": pytesseract is not None and Image is not None,
        "tesseract_available": tesseract_ok,
        "easyocr_available": easyocr_ready,
        "tesseract_cmd": executable or None,
    }


_TESS_CONFIG = "--psm 6 --oem 3"  # uniform block of text, LSTM engine


def _ocr_single_image(image) -> Tuple[str, Optional[float]]:
    try:
        text = pytesseract.image_to_string(image, config=_TESS_CONFIG)
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRDependencyError(
            "Tesseract is not installed or not in PATH. Install it or set TESSERACT_CMD."
        ) from exc

    confidence = None
    try:
        data = pytesseract.image_to_data(image, config=_TESS_CONFIG, output_type=pytesseract.Output.DICT)
        # Only count words Tesseract is reasonably sure about (conf >= 40).
        # Low-confidence entries are usually noise/artifacts and skew the average down.
        values: List[float] = []
        for raw in data.get("conf", []):
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                continue
            if parsed >= 40:
                values.append(parsed)
        if values:
            confidence = round((sum(values) / len(values)) / 100.0, 4)
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRDependencyError(
            "Tesseract is not installed or not in PATH. Install it or set TESSERACT_CMD."
        ) from exc
    except Exception:
        confidence = None

    return _clean_ocr_text(text), confidence


def _easyocr_single_image(image) -> Tuple[str, Optional[float]]:
    reader = _get_easyocr_reader()
    if reader is None:
        return "", None

    try:
        results = reader.readtext(image, detail=1, paragraph=False)
    except Exception:
        return "", None

    lines: List[str] = []
    confidences: List[float] = []
    for item in results or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = str(item[1] or "").strip()
        if text:
            lines.append(text)
        try:
            conf = float(item[2])
        except (TypeError, ValueError):
            continue
        if conf >= 0:
            confidences.append(conf)

    text = _clean_ocr_text("\n".join(lines))
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else None
    return text, confidence


def _ocr_candidate_score(text: str, confidence: Optional[float]) -> float:
    medication_signal = 0
    medication_signal += len(DOSAGE_RE.findall(text or "")) * 4
    medication_signal += len(FREQUENCY_RE.findall(text or "")) * 2
    medication_signal += len(ROUTE_RE.findall(text or "")) * 2

    useful_words = len(re.findall(r"[A-Za-z]{4,}", text or ""))
    line_bonus = sum(
        1
        for line in (text or "").splitlines()
        if DOSAGE_RE.search(line) or FREQUENCY_RE.search(line) or ROUTE_RE.search(line)
    )
    return float(confidence or 0.0) + medication_signal + line_bonus + min(useful_words / 30.0, 3.0)


def _ocr_with_confidence(image_bytes: bytes) -> Tuple[str, Optional[float], Optional[str]]:
    """
    Hybrid OCR pipeline:

    Tesseract: angle sweep at 0°, ±6°, ±10° on the binarized image.
               Binarization (Otsu) maximises Tesseract accuracy on printed text.
    EasyOCR:   single pass on the grayscale-only image (no binarization).
               EasyOCR's CNN is trained on natural images; binarization hurts it.

    The best candidate is chosen by medication-signal score (dosage/frequency/route
    keyword density + engine confidence), not raw confidence alone.
    """
    if pytesseract is None and not _easyocr_available():
        raise OCRDependencyError(
            "OCR dependency is not installed. Install pytesseract or easyocr."
        )

    _configure_tesseract()
    raw = _open_image(image_bytes)
    base_image = _preprocess_base(raw)                       # grayscale + autocontrast
    tess_image = _preprocess_for_tesseract(base_image)       # + Otsu binarize + sharpen

    candidates: List[Tuple[float, str, Optional[float], Optional[str]]] = []

    # ── Tesseract angle sweep (binarized image) ───────────────────────────
    if pytesseract is not None:
        for angle in (0, -10, 10, -6, 6):
            candidate_image = (
                tess_image
                if angle == 0
                else tess_image.rotate(
                    angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                    fillcolor=255,
                )
            )
            text, confidence = _ocr_single_image(candidate_image)
            candidates.append(
                (_ocr_candidate_score(text, confidence), text, confidence, "tesseract")
            )

    # ── EasyOCR — grayscale image, no binarization ────────────────────────
    if _easyocr_available():
        easy_text, easy_confidence = _easyocr_single_image(base_image)
        if easy_text:
            candidates.append(
                (
                    _ocr_candidate_score(easy_text, easy_confidence),
                    easy_text,
                    easy_confidence,
                    "easyocr",
                )
            )

    if not candidates:
        raise OCRDependencyError(
            "OCR runtime is unavailable. Install pytesseract or easyocr."
        )

    _, best_text, best_confidence, best_engine = max(candidates, key=lambda item: item[0])
    return best_text, best_confidence, best_engine


def _detected_medicines(text: str, *, disease: str, age: int) -> List[Dict[str, Any]]:
    """
    Extract medicines from OCR text via the NER pipeline.

    Medicines are included even when no dosage/frequency/route pattern is
    detected — many real prescriptions list only the drug name and strength
    without structured frequency text that our regexes recognise.
    """
    ner_result = extract_medical_entities(text, disease=disease, age=age, max_drugs=8, min_confidence=0.80)
    entities = ner_result.get("drugs", [])
    medicines: List[Dict[str, Any]] = []
    for entity in entities:
        medicines.append(
            {
                "query": entity.get("candidate") or entity.get("text"),
                "text": entity.get("text"),
                "drug": entity.get("drug_name") or entity.get("drug") or entity.get("text"),
                "normalized": entity.get("normalized") or entity.get("drug_name") or entity.get("text"),
                "confidence": entity.get("confidence"),
                "best_score": entity.get("best_score"),
                "dosage": entity.get("dosage"),
                "frequency": entity.get("frequency"),
                "route": entity.get("route"),
                "best_match": entity.get("best_match"),
            }
        )
    return medicines


def analyze_prescription_text(
    *,
    text: str,
    disease: str,
    age: int,
    request_id: str = "",
) -> Dict[str, Any]:
    cleaned = _clean_ocr_text(text)
    detected = _detected_medicines(cleaned, disease=disease, age=age)

    out: Dict[str, Any] = {
        "context": {"disease": disease, "age": age},
        "ocr": {"text": cleaned, "confidence": None},
        "detected_medicines": detected,
        "note": (
            "Educational demo only. Corrected OCR text can still contain mistakes; "
            "verify medicine names and instructions with a pharmacist or doctor."
        ),
    }
    if request_id:
        out["request_id"] = request_id
    return out


def ocr_prescription_image(
    *,
    image_bytes: bytes,
    filename: str,
    disease: str,
    age: int,
    request_id: str = "",
) -> Dict[str, Any]:
    """
    OCR endpoint logic:
    - run Tesseract OCR
    - return extracted text
    - detect likely medicines via the lightweight medication NER layer
    """
    text, confidence, engine = _ocr_with_confidence(image_bytes)
    detected = _detected_medicines(text, disease=disease, age=age)

    out: Dict[str, Any] = {
        "context": {"disease": disease, "age": age},
        "file": {"name": filename},
        "ocr": {"text": text, "confidence": confidence, "engine": engine},
        "detected_medicines": detected,
        "preprocessing": {
            "enabled": True,
            "method": "pil_upscale_grayscale_autocontrast_sharpen_angle_sweep_hybrid_tesseract_easyocr",
            "engine": engine,
        },
        "note": "Educational demo only. OCR may be inaccurate. Verify with a pharmacist or doctor.",
    }
    if request_id:
        out["request_id"] = request_id
    return out
