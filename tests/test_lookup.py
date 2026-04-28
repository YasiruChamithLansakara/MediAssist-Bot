import pytest

from app.services.drug_lookup import init_store, lookup_drug, SUPPORTED_DISEASES


@pytest.fixture(scope="session", autouse=True)
def _store_loaded():
    init_store()


def assert_schema(r: dict):
    for k in (
        "query",
        "normalized",
        "status",
        "confidence",
        "best_score",
        "best_match",
        "matches",
        "suggestions",
        "message",
        "match_type",
        "resolution_path",
    ):
        assert k in r, f"Missing key: {k}"

    assert isinstance(r["query"], str)
    assert isinstance(r["normalized"], str)

    assert r["status"] in ("ok", "low_confidence", "no_match")
    assert isinstance(r["confidence"], (int, float))
    assert 0.0 <= float(r["confidence"]) <= 1.0

    assert isinstance(r["best_score"], (int, float))
    assert 0.0 <= float(r["best_score"]) <= 100.0

    assert isinstance(r["matches"], list)
    assert isinstance(r["suggestions"], list)
    assert isinstance(r["resolution_path"], list)
    assert len(r["resolution_path"]) >= 1

    if r["status"] == "no_match":
        assert r["best_match"] is None
        assert r["matches"] == []
    else:
        assert r["best_match"] is not None
        assert len(r["matches"]) >= 1

    assert all(isinstance(s, str) for s in r["suggestions"])
    sugg_norm = [s.strip().lower() for s in r["suggestions"] if s.strip()]
    assert len(sugg_norm) == len(set(sugg_norm)), "Suggestions contain duplicates"


def assert_match_shape(m: dict):
    for k in (
        "drug_id",
        "generic_name",
        "generic_name_clean",
        "brand_names",
        "drug_class",
        "route",
        "indications",
        "dosage_and_administration",
        "warnings",
        "contraindications",
        "sources",
        "last_updated",
        "score",
        "match",
    ):
        assert k in m, f"Match missing key: {k}"

    assert isinstance(m["score"], (int, float))
    assert 0.0 <= float(m["score"]) <= 100.0


def assert_context_fields(r: dict, disease: str, age: int):
    assert "context" in r
    assert "supported_diseases" in r

    assert r["context"]["disease"] == disease
    assert r["context"]["age"] == age

    assert isinstance(r["supported_diseases"], list)
    assert set(r["supported_diseases"]) == set(SUPPORTED_DISEASES)

    # ✅ new field name: "tailored" (only when best_match exists)
    if r["best_match"] is not None:
        assert "tailored" in r
        t = r["tailored"]
        assert t["disease"] == disease
        assert t["age"] == age
        assert "age_group" in t
        assert "notes" in t
        assert "snippets" in t
        assert isinstance(t["notes"], list)
        assert isinstance(t["snippets"], dict)


def test_exact_acetaminophen_with_context():
    r = lookup_drug("acetaminophen", disease="diabetes", age=30)
    assert_schema(r)
    assert_context_fields(r, "diabetes", 30)
    assert r["status"] in ("ok", "low_confidence")
    assert r["best_match"] is not None
    assert r["best_score"] >= 80
    assert_match_shape(r["best_match"])


def test_alias_paracetamol_with_context():
    r = lookup_drug("paracetamol", disease="hypertension", age=40)
    assert_schema(r)
    assert_context_fields(r, "hypertension", 40)
    assert r["status"] in ("ok", "low_confidence")
    assert r["normalized"] == "acetaminophen"
    assert r["best_match"] is not None
    assert r["best_score"] >= 80
    assert_match_shape(r["best_match"])


def test_brand_panadol_with_context():
    r = lookup_drug("Panadol", disease="asthma", age=25)
    assert_schema(r)
    assert_context_fields(r, "asthma", 25)
    assert r["status"] in ("ok", "low_confidence")
    assert r["normalized"] == "acetaminophen"
    assert r["best_match"] is not None
    assert r["best_score"] >= 80
    assert_match_shape(r["best_match"])


def test_dosage_paracetamol_500mg_with_context():
    r = lookup_drug("paracetamol 500mg", disease="heart disease", age=65)
    assert_schema(r)
    assert_context_fields(r, "heart disease", 65)
    assert r["status"] in ("ok", "low_confidence")
    assert r["normalized"] == "acetaminophen"
    assert r["best_match"] is not None
    assert_match_shape(r["best_match"])


@pytest.mark.parametrize("q", ["paracetmol", "paracetemol", "paracetam0l"])
def test_typo_family(q):
    r = lookup_drug(q, disease="arthritis", age=55)
    assert_schema(r)
    assert "context" in r
    assert r["context"]["disease"] == "arthritis"
    assert r["context"]["age"] == 55
    assert r["status"] in ("ok", "low_confidence", "no_match")
    if r["status"] != "no_match":
        assert_match_shape(r["best_match"])


def test_gibberish_no_match_still_returns_context():
    r = lookup_drug("zzzzzz", disease="diabetes", age=30)
    assert_schema(r)
    assert "context" in r
    assert r["context"]["disease"] == "diabetes"
    assert r["context"]["age"] == 30
    assert r["status"] == "no_match"
    assert r["best_match"] is None
    assert r["matches"] == []


def test_empty_or_whitespace_query_behaviour():
    r = lookup_drug("   ", disease="diabetes", age=30)
    assert_schema(r)
    assert r["status"] == "no_match"