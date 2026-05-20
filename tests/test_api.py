import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.services.ocr_service import OCRDependencyError


@pytest.fixture(scope="session")
def client():
    # Using context manager ensures lifespan runs reliably
    with TestClient(app) as c:
        yield c


def assert_error_shape(data: dict):
    assert "error" in data
    assert isinstance(data["error"], dict)
    assert "code" in data["error"]
    assert "message" in data["error"]


def test_health_ok_api(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_meta_ok(client: TestClient):
    r = client.get("/api/meta")
    assert r.status_code == 200
    data = r.json()
    assert "supported_diseases" in data
    assert "age_range" in data
    assert data["features"]["lightweight_ner"] is True
    assert "ocr_runtime" in data


def test_lookup_requires_params(client: TestClient):
    r = client.get("/api/lookup?drug=acetaminophen")
    assert r.status_code == 422
    data = r.json()
    assert_error_shape(data)
    assert data["error"]["code"] == "validation_error"


def test_lookup_rejects_whitespace_drug(client: TestClient):
    r = client.get("/api/lookup?drug=%20%20%20&disease=diabetes&age=30")
    assert r.status_code == 422
    data = r.json()
    assert_error_shape(data)


def test_lookup_invalid_disease(client: TestClient):
    r = client.get("/api/lookup?drug=acetaminophen&disease=cancer&age=30")
    assert r.status_code == 422
    data = r.json()
    assert_error_shape(data)
    assert data["error"]["code"] == "unsupported_disease"
    assert "supported_diseases" in (data["error"].get("details") or {})


@pytest.mark.parametrize("age", [0, -1, 121])
def test_lookup_invalid_age(client: TestClient, age: int):
    r = client.get(f"/api/lookup?drug=acetaminophen&disease=diabetes&age={age}")
    assert r.status_code == 422
    data = r.json()
    assert_error_shape(data)
    assert data["error"]["code"] == "validation_error"


def test_lookup_ok_returns_context_and_supported_and_tailored(client: TestClient):
    r = client.get("/api/lookup?drug=acetaminophen&disease=diabetes&age=30")
    assert r.status_code == 200
    data = r.json()

    assert "status" in data
    assert "best_match" in data
    assert "matches" in data

    assert "context" in data
    assert data["context"]["disease"] == "diabetes"
    assert data["context"]["age"] == 30

    assert "supported_diseases" in data
    assert isinstance(data["supported_diseases"], list)
    assert "diabetes" in data["supported_diseases"]

    # tailored only if best_match exists
    if data.get("best_match") is not None:
        assert "tailored" in data
        assert "age_group" in data["tailored"]
        assert "notes" in data["tailored"]
        assert "snippets" in data["tailored"]


def test_chat_extracts_drug_from_message(client: TestClient):
    r = client.post(
        "/api/chat",
        json={
            "message": "Is paracetamol safe for hypertension at age 60?",
            "disease": "hypertension",
            "age": 60,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["context"]["disease"] == "hypertension"
    assert data["intent"] == "safety"
    assert data["detected_entities"]
    assert data["matched_drugs"]
    assert any((m.get("best_match") or {}).get("generic_name_clean") == "acetaminophen" for m in data["matched_drugs"])
    assert "answer" in data and "acetaminophen" in data["answer"].lower()
    assert "safety_notice" in data


def test_chat_accepts_explicit_drug(client: TestClient):
    r = client.post(
        "/api/chat",
        json={
            "message": "What dosage section should I check?",
            "drug": "metformin",
            "disease": "diabetes",
            "age": 45,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "dosage"
    assert data["matched_drugs"]
    assert data["matched_drugs"][0]["best_match"] is not None


def test_prescription_rejects_non_image_upload(client: TestClient):
    r = client.post(
        "/api/prescription",
        data={"disease": "diabetes", "age": "45"},
        files={"file": ("note.txt", b"metformin 500 mg", "text/plain")},
    )
    assert r.status_code == 415
    data = r.json()
    assert_error_shape(data)
    assert data["error"]["code"] == "unsupported_file_type"


def test_prescription_analyze_text_detects_medicine_and_dosage(client: TestClient):
    r = client.post(
        "/api/prescription/analyze-text",
        json={
            "text": "Metformin 500 mg oral twice daily\nParacetamol 500 mg prn",
            "disease": "diabetes",
            "age": 45,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["context"] == {"disease": "diabetes", "age": 45}
    assert data["ocr"]["confidence"] is None
    assert data["detected_medicines"]
    names = {str(item.get("drug") or "").lower() for item in data["detected_medicines"]}
    assert "metformin" in names
    metformin = next(item for item in data["detected_medicines"] if str(item.get("drug")).lower() == "metformin")
    assert metformin["dosage"].lower() == "500 mg"
    assert metformin["route"].lower() == "oral"


def test_prescription_analyze_text_rejects_empty_text(client: TestClient):
    r = client.post(
        "/api/prescription/analyze-text",
        json={"text": "   ", "disease": "diabetes", "age": 45},
    )
    assert r.status_code == 422
    data = r.json()
    assert_error_shape(data)


def test_prescription_reports_ocr_dependency_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def fake_ocr_prescription_image(**kwargs):
        raise OCRDependencyError("Tesseract is not installed.")

    monkeypatch.setattr(main_module, "ocr_prescription_image", fake_ocr_prescription_image)

    r = client.post(
        "/api/prescription",
        data={"disease": "diabetes", "age": "45"},
        files={"file": ("prescription.png", b"not really an image", "image/png")},
    )
    assert r.status_code == 503
    data = r.json()
    assert_error_shape(data)
    assert data["error"]["code"] == "ocr_unavailable"
