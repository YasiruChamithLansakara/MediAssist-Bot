# OCR Validation Summary

Generated on: 2026-04-25

## What Was Validated

The MediAssist prescription OCR pipeline was validated with generated prescription images that cover the required demo categories:

- Printed prescription
- Handwritten-like prescription using Windows script fonts
- Blurry prescription
- Rotated prescription
- Low-light prescription
- Low-resolution prescription

The validation uses the real local Tesseract installation:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Local OCR runtime status:

```text
python_dependencies=True
tesseract_cmd=C:\Program Files\Tesseract-OCR\tesseract.exe
configured=True
```

## Latest Validation Result

Report files:

```text
outputs/ocr_validation/20260425_234128/ocr_validation_report.json
outputs/ocr_validation/20260425_234128/ocr_validation_report.md
```

Expected medicines:

```text
metformin
atorvastatin calcium
acetaminophen
```

| Case             | OCR Confidence | Medicine Recall | Result |
| ---------------- | -------------: | --------------: | -----: |
| printed_clean    |         0.9547 |          1.0000 |   pass |
| handwritten_like |         0.9389 |          1.0000 |   pass |
| blurry           |         0.9350 |          0.6667 |   pass |
| rotated          |         0.9567 |          1.0000 |   pass |
| low_light        |         0.9539 |          1.0000 |   pass |
| low_resolution   |         0.9392 |          1.0000 |   pass |

## Improvements Made During Validation

- Added a repeatable OCR validation script: `scripts/validate_ocr_pipeline.py`
- Added OCR image preprocessing:
  - upscale small images
  - grayscale conversion
  - auto contrast
  - contrast boost
  - sharpening
- Added angle-sweep OCR fallback for tilted prescription images.
- Filtered prescription detections so non-medication lines are less likely to become false positives.
- Confirmed local Tesseract OCR is installed and available to the backend.

## Known Limitations

- The handwritten validation is handwritten-like, not real doctor handwriting.
- Blurry images may still require manual correction in the frontend OCR text editor.
- Railway OCR has not been verified from a deployed URL yet. To verify Railway after deployment, run:

```powershell
$env:RAILWAY_API_BASE="https://your-railway-backend-url"
.\venv\Scripts\python.exe scripts\validate_ocr_pipeline.py
```

The script will call `/api/meta` on the deployed backend and include the remote OCR runtime result in the validation report.
