from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.drug_lookup import init_store, normalize_text
from app.services.ocr_service import OCRDependencyError, ocr_prescription_image, ocr_runtime_status

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Pillow is required for OCR validation: {exc}") from exc


DEFAULT_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ocr_validation"

PRESCRIPTION_LINES = [
    "MediAssist Clinic",
    "Patient: Demo Patient    Age: 45",
    "Diagnosis: Diabetes and hypertension",
    "Rx",
    "Metformin 500 mg oral twice daily",
    "Atorvastatin 20 mg oral once daily",
    "Paracetamol 500 mg oral as needed",
    "Review after 2 weeks",
]

EXPECTED_DRUGS = ["metformin", "atorvastatin calcium", "acetaminophen"]


def _font(candidates: List[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _as_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_prescription(*, handwritten: bool = False) -> Image.Image:
    image = Image.new("RGB", (1180, 820), "#fbfbf7")
    draw = ImageDraw.Draw(image)

    title_font = _font([r"C:\Windows\Fonts\arialbd.ttf"], 44)
    body_font = _font(
        [
            r"C:\Windows\Fonts\segoepr.ttf",
            r"C:\Windows\Fonts\segoesc.ttf",
            r"C:\Windows\Fonts\LHANDW.TTF",
            r"C:\Windows\Fonts\arial.ttf",
        ]
        if handwritten
        else [r"C:\Windows\Fonts\arial.ttf"],
        38 if handwritten else 34,
    )
    small_font = _font([r"C:\Windows\Fonts\arial.ttf"], 24)

    draw.rectangle((38, 38, 1142, 782), outline="#d8d8d8", width=3)
    draw.text((70, 62), PRESCRIPTION_LINES[0], fill="#111111", font=title_font)
    draw.line((70, 124, 1110, 124), fill="#cfcfcf", width=2)

    y = 158
    for index, line in enumerate(PRESCRIPTION_LINES[1:]):
        x = 78 + (index % 2) * 6 if handwritten else 78
        draw.text((x, y), line, fill="#151515", font=body_font)
        y += 72 if handwritten else 66

    draw.text((760, 720), "Dr. Demo", fill="#222222", font=small_font)
    draw.line((740, 712, 1080, 712), fill="#444444", width=2)
    return image


def _printed_clean() -> Image.Image:
    return _draw_prescription(handwritten=False)


def _handwritten_like() -> Image.Image:
    image = _draw_prescription(handwritten=True)
    return image.rotate(-1.5, resample=Image.Resampling.BICUBIC, fillcolor="#fbfbf7")


def _blurry() -> Image.Image:
    return _printed_clean().filter(ImageFilter.GaussianBlur(radius=2.2))


def _rotated() -> Image.Image:
    return _printed_clean().rotate(9, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="#f6f3ec")


def _low_light() -> Image.Image:
    image = _printed_clean()
    image = ImageEnhance.Brightness(image).enhance(0.52)
    image = ImageEnhance.Contrast(image).enhance(0.72)
    overlay = Image.new("RGB", image.size, "#302c24")
    return Image.blend(image, overlay, 0.18)


def _low_resolution() -> Image.Image:
    image = _printed_clean()
    small = image.resize((590, 410), Image.Resampling.BICUBIC)
    return small.resize(image.size, Image.Resampling.BICUBIC)


CASES = {
    "printed_clean": _printed_clean,
    "handwritten_like": _handwritten_like,
    "blurry": _blurry,
    "rotated": _rotated,
    "low_light": _low_light,
    "low_resolution": _low_resolution,
}


def _detected_names(result: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for item in result.get("detected_medicines") or []:
        name = str(item.get("drug") or item.get("normalized") or "").strip()
        key = normalize_text(name)
        if key and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def _score_case(result: Dict[str, Any]) -> Dict[str, Any]:
    detected = {normalize_text(name) for name in _detected_names(result)}
    expected = {normalize_text(name) for name in EXPECTED_DRUGS}
    found = sorted(detected & expected)
    missed = sorted(expected - detected)
    extra = sorted(detected - expected)
    recall = len(found) / len(expected) if expected else 0.0

    return {
        "found_expected": found,
        "missed_expected": missed,
        "extra_detected": extra,
        "medicine_recall": round(recall, 4),
        "passed": recall >= (2 / 3),
    }


def _summarize_detection(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query": item.get("query"),
        "drug": item.get("drug"),
        "confidence": item.get("confidence"),
        "best_score": item.get("best_score"),
        "dosage": item.get("dosage"),
        "frequency": item.get("frequency"),
        "route": item.get("route"),
        "text": item.get("text"),
    }


def _check_remote_meta(api_base: Optional[str]) -> Optional[Dict[str, Any]]:
    if not api_base:
        return None

    base = api_base.rstrip("/")
    if not base.endswith("/api"):
        base = f"{base}/api"

    try:
        with urlopen(f"{base}/meta", timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"error": str(exc), "api_base": base}


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# MediAssist OCR Validation Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Runtime",
        "",
        f"- Local OCR configured: `{report['local_ocr_runtime'].get('configured')}`",
        f"- Tesseract command: `{report['local_ocr_runtime'].get('tesseract_cmd')}`",
        f"- Railway/API check: `{report.get('remote_meta') or 'not provided'}`",
        "",
        "## Results",
        "",
        "| Case | OCR Confidence | Medicine Recall | Found | Missed | Pass |",
        "|---|---:|---:|---|---|---|",
    ]

    for case in report["cases"]:
        confidence = case.get("ocr_confidence")
        confidence_text = "-" if confidence is None else f"{confidence:.2f}"
        score = case["score"]
        lines.append(
            "| {name} | {confidence} | {recall:.2f} | {found} | {missed} | {passed} |".format(
                name=case["name"],
                confidence=confidence_text,
                recall=score["medicine_recall"],
                found=", ".join(score["found_expected"]) or "-",
                missed=", ".join(score["missed_expected"]) or "-",
                passed="yes" if score["passed"] else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Printed prescriptions are expected to perform best.",
            "- Handwritten-like validation uses installed Windows script fonts, not real doctor handwriting.",
            "- Blurry, rotated, low-light, and low-resolution cases test robustness of preprocessing.",
            "- Low recall cases should be manually corrected in the frontend OCR text editor, then re-analyzed.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(*, output_dir: Path, api_base: Optional[str]) -> Dict[str, Any]:
    if not os.getenv("TESSERACT_CMD") and Path(DEFAULT_TESSERACT).exists():
        os.environ["TESSERACT_CMD"] = DEFAULT_TESSERACT

    init_store()
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "expected_drugs": EXPECTED_DRUGS,
        "local_ocr_runtime": ocr_runtime_status(),
        "remote_meta": _check_remote_meta(api_base),
        "cases": [],
    }

    for name, factory in CASES.items():
        image = factory()
        image_path = output_dir / f"{name}.png"
        image.save(image_path)

        case: Dict[str, Any] = {"name": name, "image": str(image_path.relative_to(PROJECT_ROOT))}
        try:
            result = ocr_prescription_image(
                image_bytes=_as_png_bytes(image),
                filename=image_path.name,
                disease="diabetes",
                age=45,
            )
            case["ocr_text"] = result.get("ocr", {}).get("text", "")
            case["ocr_confidence"] = result.get("ocr", {}).get("confidence")
            case["detected_medicines"] = [
                _summarize_detection(item) for item in result.get("detected_medicines") or []
            ]
            case["score"] = _score_case(result)
        except OCRDependencyError as exc:
            case["error"] = str(exc)
            case["score"] = {
                "found_expected": [],
                "missed_expected": EXPECTED_DRUGS,
                "extra_detected": [],
                "medicine_recall": 0.0,
                "passed": False,
            }

        report["cases"].append(case)

    report_path = output_dir / "ocr_validation_report.json"
    markdown_path = output_dir / "ocr_validation_report.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, markdown_path)

    report["report_path"] = str(report_path.relative_to(PROJECT_ROOT))
    report["markdown_path"] = str(markdown_path.relative_to(PROJECT_ROOT))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the MediAssist OCR pipeline with generated prescription images.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Directory where generated images and reports are written.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("RAILWAY_API_BASE") or os.getenv("MEDIASSIST_API_BASE"),
        help="Optional deployed API base URL for Railway /api/meta verification.",
    )
    args = parser.parse_args()

    report = run_validation(output_dir=Path(args.output_dir), api_base=args.api_base)
    print(json.dumps(
        {
            "generated_at": report["generated_at"],
            "local_ocr_runtime": report["local_ocr_runtime"],
            "report_path": report["report_path"],
            "markdown_path": report["markdown_path"],
            "cases": [
                {
                    "name": case["name"],
                    "ocr_confidence": case.get("ocr_confidence"),
                    "medicine_recall": case["score"]["medicine_recall"],
                    "found": case["score"]["found_expected"],
                    "missed": case["score"]["missed_expected"],
                    "passed": case["score"]["passed"],
                }
                for case in report["cases"]
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
