"""Tests for medical safety layer."""

import pytest

from app.services.medical_safety import (
    MedicalSafetyGuard,
    detect_emergency_symptoms,
    detect_forbidden_patterns,
    check_safe_communication,
    get_safety_notice,
)


class TestEmergencyDetection:
    """Test emergency symptom detection."""

    def test_breathing_difficulty_detected(self):
        text = "I'm having difficulty breathing and can't catch my breath"
        is_emergency, symptoms = detect_emergency_symptoms(text)
        assert is_emergency is True
        assert "breathing" in symptoms

    def test_chest_pain_detected(self):
        text = "I have severe chest pain and pressure in my chest"
        is_emergency, symptoms = detect_emergency_symptoms(text)
        assert is_emergency is True
        assert "chest" in symptoms

    def test_severe_allergy_detected(self):
        text = "I'm experiencing anaphylaxis and my throat is closing"
        is_emergency, symptoms = detect_emergency_symptoms(text)
        assert is_emergency is True
        assert "severe_allergy" in symptoms

    def test_loss_of_consciousness_detected(self):
        text = "I fainted and then was confused"
        is_emergency, symptoms = detect_emergency_symptoms(text)
        assert is_emergency is True
        assert len(symptoms) > 0

    def test_seizure_detected(self):
        text = "The patient is experiencing a seizure with convulsions"
        is_emergency, symptoms = detect_emergency_symptoms(text)
        assert is_emergency is True
        assert "seizure" in symptoms

    def test_no_emergency_normal_question(self):
        text = "Is acetaminophen safe for my hypertension?"
        is_emergency, symptoms = detect_emergency_symptoms(text)
        assert is_emergency is False
        assert len(symptoms) == 0

    def test_empty_text_no_emergency(self):
        is_emergency, symptoms = detect_emergency_symptoms("")
        assert is_emergency is False
        assert len(symptoms) == 0


class TestForbiddenPatterns:
    """Test detection of direct medical recommendations."""

    def test_safe_to_take_pattern_detected(self):
        text = "You can safely take aspirin"
        has_forbidden, patterns = detect_forbidden_patterns(text)
        assert has_forbidden is True
        assert "safe_to_take" in patterns

    def test_direct_recommendation_detected(self):
        text = "You should take this medicine"
        has_forbidden, patterns = detect_forbidden_patterns(text)
        assert has_forbidden is True

    def test_no_forbidden_patterns_in_safe_text(self):
        text = "Please consult your pharmacist about this medication"
        has_forbidden, patterns = detect_forbidden_patterns(text)
        assert has_forbidden is False

    def test_empty_text_no_forbidden(self):
        has_forbidden, patterns = detect_forbidden_patterns("")
        assert has_forbidden is False


class TestSafePatterns:
    """Test detection of safe communication patterns."""

    def test_consult_pharmacist_detected(self):
        text = "Please consult a pharmacist about your medications"
        has_safe = check_safe_communication(text)
        assert has_safe is True

    def test_consult_doctor_detected(self):
        text = "Discuss this with your doctor before taking it"
        has_safe = check_safe_communication(text)
        assert has_safe is True

    def test_ask_doctor_detected(self):
        text = "Ask your doctor if this is right for you"
        has_safe = check_safe_communication(text)
        assert has_safe is True

    def test_no_safe_patterns_in_unsafe_text(self):
        text = "This medicine is definitely safe for you"
        has_safe = check_safe_communication(text)
        assert has_safe is False


class TestMedicalSafetyGuard:
    """Test the MedicalSafetyGuard class."""

    def test_emergency_warning_generated(self):
        symptoms = ["breathing", "chest"]
        warning = MedicalSafetyGuard.build_emergency_warning(symptoms)
        assert "EMERGENCY" in warning
        assert "911" in warning or "emergency services" in warning
        assert len(warning) > 0

    def test_empty_symptoms_no_warning(self):
        warning = MedicalSafetyGuard.build_emergency_warning([])
        assert warning == ""

    def test_safety_notice_is_strong(self):
        notice = MedicalSafetyGuard.enhance_safety_notice()
        assert "NOT medical advice" in notice or "NOT" in notice
        assert "pharmacist" in notice.lower() or "doctor" in notice.lower()
        assert len(notice) > 100

    def test_answer_enhanced_with_footer(self):
        answer = "Aspirin is used for pain relief"
        enhanced = MedicalSafetyGuard.enhance_answer_with_safety(answer, "diabetes", 45)
        assert "---" in enhanced
        assert len(enhanced) > len(answer)
        assert "pharmacist" in enhanced.lower() or "doctor" in enhanced.lower()

    def test_validate_user_intent_normal(self):
        result = MedicalSafetyGuard.validate_user_intent(
            "Is metformin safe for diabetes?",
            "diabetes",
            50
        )
        assert result["is_emergency"] is False
        assert result["recommended_action"] == "proceed_normal"

    def test_validate_user_intent_emergency(self):
        result = MedicalSafetyGuard.validate_user_intent(
            "I'm having chest pain",
            "diabetes",
            50
        )
        assert result["is_emergency"] is True
        assert result["recommended_action"] == "emergency_alert"

    def test_validate_user_intent_forbidden(self):
        result = MedicalSafetyGuard.validate_user_intent(
            "You should take this medicine",
            "diabetes",
            50
        )
        assert result["has_forbidden_patterns"] is True

    def test_validate_user_intent_with_safe_patterns(self):
        result = MedicalSafetyGuard.validate_user_intent(
            "Please consult your pharmacist about this",
            "diabetes",
            50
        )
        # Should still not have forbidden if it includes safe patterns
        assert result["has_safe_patterns"] is True


class TestSafetyNotice:
    """Test safety notice generation."""

    def test_get_safety_notice_not_empty(self):
        notice = get_safety_notice()
        assert len(notice) > 0
        assert isinstance(notice, str)

    def test_safety_notice_includes_key_terms(self):
        notice = get_safety_notice()
        key_terms = ["disclaimer", "NOT", "pharmacist", "doctor", "emergency"]
        found_terms = [term for term in key_terms if term.lower() in notice.lower()]
        # Should have at least 3 of these key terms
        assert len(found_terms) >= 3
