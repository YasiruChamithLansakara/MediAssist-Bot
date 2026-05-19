from __future__ import annotations

import os
import re
import shutil
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from app.services.medication_ner import DOSAGE_RE, FREQUENCY_RE, ROUTE_RE
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
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None


class OCRDependencyError(RuntimeError):
    pass


class OCRImageError(ValueError):
    pass


_easyocr_reader = None


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
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    except Exception:
        _easyocr_reader = None
    return _easyocr_reader


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


def _preprocess_image(image):
    """
    Simple PIL-only preprocessing for demo reliability:
    - upscale small images
    - grayscale
    - auto contrast
    - light sharpening
    """
    if Image is None:
        return image

    from PIL import ImageEnhance, ImageFilter, ImageOps

    width, height = image.size
    largest_side = max(width, height)
    if largest_side and largest_side < 1800:
        scale = min(3.0, 1800 / largest_side)
        image = image.resize((int(width * scale), int(height * scale)))

    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.35)
    return image.filter(ImageFilter.SHARPEN)


def ocr_runtime_status() -> Dict[str, Any]:
    cmd = os.getenv("TESSERACT_CMD", "").strip()
    executable = cmd or shutil.which("tesseract") or ""
    easyocr_ready = _get_easyocr_reader() is not None
    return {
        "python_dependencies": pytesseract is not None and Image is not None,
        "easyocr_available": easyocr_ready,
        "tesseract_cmd": executable or None,
        "configured": bool((pytesseract is not None and Image is not None and executable) or easyocr_ready),
    }


def _ocr_single_image(image) -> Tuple[str, Optional[float]]:
    try:
        text = pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRDependencyError(
            "Tesseract is not installed or not in PATH. Install it or set TESSERACT_CMD."
        ) from exc

    confidence = None
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        values: List[float] = []
        for raw in data.get("conf", []):
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
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


def _ocr_with_confidence(image_bytes: bytes) -> Tuple[str, Optional[float]]:
    """
    Returns (text, avg_confidence 0..1 or None).

    A small angle sweep helps with camera captures where the prescription is
    tilted. The chosen candidate is the one with the strongest medication-like
    OCR signal, not only the highest raw Tesseract confidence.
    """
    if pytesseract is None and not _easyocr_available():
        raise OCRDependencyError("OCR dependency is not installed. Install pytesseract or easyocr.")

    image = _preprocess_image(_open_image(image_bytes))

    candidates: List[Tuple[float, str, Optional[float]]] = []
    _configure_tesseract()
    for angle in (0, -10, 10, -6, 6):
        candidate_image = image
        if angle:
            candidate_image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
        if pytesseract is not None:
            text, confidence = _ocr_single_image(candidate_image)
            candidates.append((_ocr_candidate_score(text, confidence), text, confidence))

        easy_text, easy_confidence = _easyocr_single_image(candidate_image)
        if easy_text:
            candidates.append((_ocr_candidate_score(easy_text, easy_confidence), easy_text, easy_confidence))

    if not candidates:
        raise OCRDependencyError("OCR runtime is unavailable. Install pytesseract or easyocr.")

    _, best_text, best_confidence = max(candidates, key=lambda item: item[0])
    return best_text, best_confidence


def _detected_medicines(text: str, *, disease: str, age: int) -> List[Dict[str, Any]]:
    ner_result = extract_medical_entities(text, disease=disease, age=age, max_drugs=8, min_confidence=0.80)
    entities = ner_result.get("drugs", [])
    medicines: List[Dict[str, Any]] = []
    for entity in entities:
        if not (entity.get("dosage") or entity.get("frequency") or entity.get("route")):
            continue
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
    text, confidence = _ocr_with_confidence(image_bytes)
    detected = _detected_medicines(text, disease=disease, age=age)

    out: Dict[str, Any] = {
        "context": {"disease": disease, "age": age},
        "file": {"name": filename},
        "ocr": {"text": text, "confidence": confidence},
        "detected_medicines": detected,
        "preprocessing": {
            "enabled": True,
            "method": "pil_upscale_grayscale_autocontrast_sharpen_angle_sweep_hybrid_tesseract_easyocr",
        },
        "note": "Educational demo only. OCR may be inaccurate. Verify with a pharmacist or doctor.",
    }
    if request_id:
        out["request_id"] = request_id
    return out
