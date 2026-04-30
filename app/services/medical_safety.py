"""
Medical Safety Layer: Emergency detection, disclaimers, and safe communication patterns.

This module enforces strict safety boundaries:
- Detects emergency symptoms and urgent warnings
- Prevents direct medical recommendations
- Ensures proper disclaimer language
- Guides users to healthcare professionals
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Emergency symptoms that require immediate care
EMERGENCY_SYMPTOMS = {
    "breathing": [
        "difficulty breathing", "shortness of breath", "can't breathe", "trouble breathing",
        "gasping", "wheezing", "breathing difficulty", "dyspnea", "respiratory distress",
        "can not breathe", "unable to breathe"
    ],
    "chest": [
        "chest pain", "chest pressure", "chest tightness", "heart pain", "cardiac pain",
        "sternum pain", "chest discomfort", "chest pressure", "crushing chest"
    ],
    "severe_allergy": [
        "anaphylaxis", "anaphylactic", "severe allergic reaction", "severe allergy",
        "throat closing", "throat swelling", "tongue swelling", "severe itching",
        "severe hives", "severe rash", "histamine shock"
    ],
    "consciousness": [
        "fainting", "fainted", "blacked out", "losing consciousness", "passing out",
        "dizzy", "dizziness", "confusion", "confused", "disorientation", "unresponsive",
        "loss of consciousness", "syncope", "altered mental status"
    ],
    "seizure": [
        "seizure", "seizing", "convulsion", "convulsing", "muscle spasm", "jerking",
        "tremor", "spasming", "epileptic"
    ],
    "severe_bleeding": [
        "severe bleeding", "uncontrolled bleeding", "hemorrhage", "heavy bleeding",
        "bleeding won't stop", "major bleeding", "blood loss"
    ],
    "severe_pain": [
        "severe pain", "intense pain", "excruciating pain", "unbearable pain",
        "worst pain", "can't tolerate pain"
    ],
    "poisoning": [
        "poisoning", "overdose", "toxic", "toxicity", "poisoned", "ingestion",
        "swallowed", "accidental ingestion"
    ],
}

# Anti-patterns: phrases that suggest direct recommendations (forbidden)
FORBIDDEN_PATTERNS = {
    "safe_to_take": [
        r"(?:you\s+)?(?:can|should|will|is\s+)?(?:safely\s+)?(?:take|use|consume|drink)\s+",
        r"(?:it\'s?|it\s+is)\s+(?:safe|okay|ok)\s+(?:to\s+)?",
        r"(?:is|are)\s+(?:safe|okay|fine)\s+for\s+you",
    ],
    "direct_recommendation": [
        r"(?:you\s+)?(?:should|must|need to|have to|must)\s+(?:take|use|do|try)\s+",
        r"(?:recommend|suggest|advise)\s+(?:you|your\s+)?(?:take|use|do)\s+",
        r"(?:the\s+)?best\s+(?:option|choice|medicine|drug)\s+(?:is|for\s+you)",
    ],
    "diagnostic": [
        r"(?:you\s+)?(?:have|suffer from|are)\s+(?:[a-z\s]+)\s+disease",
        r"(?:you\s+)?(?:likely|probably|definitely)\s+(?:have|suffer from)\s+",
        r"(?:your\s+)?(?:diagnosis|condition|illness)\s+is\s+",
    ],
}

# Safe alternative phrases (good patterns)
SAFE_PATTERNS = [
    "consult a pharmacist",
    "consult a doctor",
    "speak with your doctor",
    "talk to your healthcare provider",
    "confirm with a healthcare professional",
    "verify with your pharmacist",
    "check with a pharmacist",
    "ask your doctor",
    "discuss with your physician",
    "discuss with your doctor",
    "discuss this with your",
]


def detect_emergency_symptoms(text: str) -> Tuple[bool, List[str]]:
    """
    Detects if text mentions emergency symptoms requiring immediate care.

    Returns:
        (has_emergency, detected_categories)
    """
    if not text:
        return False, []

    text_lower = text.lower()
    detected = []

    for category, symptoms in EMERGENCY_SYMPTOMS.items():
        for symptom in symptoms:
            if symptom in text_lower:
                detected.append(category)
                break

    has_emergency = len(detected) > 0
    return has_emergency, list(set(detected))  # unique categories


def detect_forbidden_patterns(text: str) -> Tuple[bool, List[str]]:
    """
    Detects if text contains direct medical recommendations (forbidden).

    Returns:
        (has_forbidden, detected_pattern_types)
    """
    if not text:
        return False, []

    detected = []

    for pattern_type, patterns in FORBIDDEN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(pattern_type)
                break

    has_forbidden = len(detected) > 0
    return has_forbidden, list(set(detected))  # unique types


def check_safe_communication(text: str) -> bool:
    """
    Checks if text includes safe communication patterns (good practice).

    Returns True if at least one safe pattern is found.
    """
    if not text:
        return False

    text_lower = text.lower()
    
    # Check for safe patterns (more flexible matching)
    safe_keywords = [
        "consult", "ask", "discuss", "speak with", "talk to",
        "confirm with", "verify with", "check with", "tell",
        "pharmacist", "doctor", "physician", "healthcare provider"
    ]
    
    # Look for at least one safe keyword
    for keyword in safe_keywords:
        if keyword in text_lower:
            return True

    return False


class MedicalSafetyGuard:
    """
    Enforces medical safety boundaries on chat responses and user inputs.
    """

    @staticmethod
    def build_emergency_warning(symptoms: List[str]) -> str:
        """
        Builds urgent warning message for detected emergency symptoms.
        """
        if not symptoms:
            return ""

        warning = (
            "🚨 **EMERGENCY ALERT** 🚨\n\n"
            "You described urgent symptoms that require **immediate medical attention**:\n"
        )

        for symptom in symptoms:
            if symptom == "breathing":
                warning += "• **Breathing trouble** — Call emergency services now.\n"
            elif symptom == "chest":
                warning += "• **Chest pain/pressure** — This is a sign of serious cardiac emergency. Call 911/emergency.\n"
            elif symptom == "severe_allergy":
                warning += "• **Severe allergic reaction** — Use EpiPen if available and call 911.\n"
            elif symptom == "consciousness":
                warning += "• **Loss of consciousness/severe confusion** — Seek emergency care immediately.\n"
            elif symptom == "seizure":
                warning += "• **Seizure activity** — Protect the person and call emergency services.\n"
            elif symptom == "severe_bleeding":
                warning += "• **Uncontrolled bleeding** — Apply pressure and call 911.\n"
            elif symptom == "severe_pain":
                warning += "• **Severe or unbearable pain** — Get emergency medical evaluation.\n"
            elif symptom == "poisoning":
                warning += "• **Possible poisoning/overdose** — Call Poison Control or 911 immediately.\n"

        warning += (
            "\n**DO NOT wait.** "
            "Stop reading this app and seek emergency care now by calling your local emergency number "
            "(911 in the USA, 112 in EU, 999 in UK)."
        )

        return warning

    @staticmethod
    def enhance_safety_notice() -> str:
        """
        Returns enhanced, stronger safety notice with all required disclaimers.
        """
        return (
            "⚠️ **IMPORTANT DISCLAIMER** ⚠️\n\n"
            "**This is NOT medical advice.** I am an educational tool only and cannot:\n"
            "• Diagnose diseases or conditions\n"
            "• Replace a doctor's or pharmacist's judgment\n"
            "• Recommend whether a medicine is safe for YOU specifically\n"
            "• Change, modify, or suggest stopping any prescription\n"
            "• Provide personalized medical treatment plans\n\n"
            "**Always** consult your pharmacist or doctor before:\n"
            "• Starting any new medicine\n"
            "• Stopping a prescribed medicine\n"
            "• Changing dosage or frequency\n"
            "• Taking multiple medicines together\n\n"
            "**For any concerns**, ask:\n"
            "1. Your prescribing doctor\n"
            "2. Your pharmacist (they review all your medicines)\n"
            "3. A poison control center (for overdose concerns)\n"
            "4. Emergency services (for urgent symptoms)"
        )

    @staticmethod
    def enhance_answer_with_safety(answer: str, disease: str, age: int) -> str:
        """
        Enhances an answer by adding safety footers and pharmacist consultation prompts.
        """
        if not answer:
            return ""

        # Add a strong closing that emphasizes consultation
        footer = (
            "\n\n---\n\n"
            "✅ **NEXT STEP**: \n"
            f"Compare this information with your prescription label. "
            "Discuss any questions with your pharmacist or doctor. "
            "They have your full medical history and can advise on your specific situation.\n\n"
            "This information is based on publicly available drug data and is not specific to your health."
        )

        return answer.rstrip() + footer

    @staticmethod
    def validate_user_intent(message: str, disease: str, age: int) -> Dict[str, any]:
        """
        Validates user intent and returns safety analysis.

        Returns dict with:
        - is_emergency: bool
        - emergency_symptoms: List[str]
        - has_forbidden_patterns: bool
        - forbidden_patterns: List[str]
        - has_safe_patterns: bool
        - recommended_action: str
        """
        is_emergency, symptoms = detect_emergency_symptoms(message)
        has_forbidden, forbidden_types = detect_forbidden_patterns(message)
        has_safe = check_safe_communication(message)

        recommended_action = "proceed_normal"
        if is_emergency:
            recommended_action = "emergency_alert"
        elif has_forbidden:
            recommended_action = "rephrase_requested"

        return {
            "is_emergency": is_emergency,
            "emergency_symptoms": symptoms,
            "has_forbidden_patterns": has_forbidden,
            "forbidden_patterns": forbidden_types,
            "has_safe_patterns": has_safe,
            "recommended_action": recommended_action,
            "disease": disease,
            "age": age,
        }


# Global helper functions for easy import
def get_safety_notice() -> str:
    """Get the enhanced safety notice."""
    return MedicalSafetyGuard.enhance_safety_notice()


def validate_input(message: str, disease: str, age: int) -> Dict[str, any]:
    """Validate user input for safety."""
    return MedicalSafetyGuard.validate_user_intent(message, disease, age)


def get_emergency_warning(symptoms: List[str]) -> str:
    """Get emergency warning for symptoms."""
    return MedicalSafetyGuard.build_emergency_warning(symptoms)
