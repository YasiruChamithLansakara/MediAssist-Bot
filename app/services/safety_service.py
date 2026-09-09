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
from typing import Any, Dict, List, Tuple

# Emergency symptoms that require immediate care
EMERGENCY_SYMPTOMS = {
    "breathing": [
        "difficulty breathing", "shortness of breath", "can't breathe", "trouble breathing",
        "gasping", "wheezing", "breathing difficulty", "dyspnea", "respiratory distress",
        "can not breathe", "cannot breathe", "unable to breathe", "struggling to breathe"
    ],
    "chest": [
        "chest pain", "chest pressure", "chest tightness", "heart pain", "cardiac pain",
        "sternum pain", "chest discomfort", "chest pressure", "crushing chest"
    ],
    "severe_allergy": [
        "anaphylaxis", "anaphylactic", "severe allergic reaction", "severe allergy",
        "throat closing", "throat is closing", "throat closing up",
        "throat swelling", "throat is swelling", "tongue swelling",
        "tongue is swelling", "face is swelling", "severe itching",
        "severe hives", "severe rash", "histamine shock"
    ],
    "consciousness": [
        "fainting", "fainted", "blacked out", "losing consciousness", "passing out",
        "unresponsive", "loss of consciousness", "syncope", "altered mental status",
        "sudden confusion", "sudden disorientation", "cannot wake",
        # "dizzy"/"confused" removed — too common/benign to reliably flag as emergency
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


# ---------------------------------------------------------------------------
# EMERGENCY CONTEXT FILTERS
# ---------------------------------------------------------------------------
# Plain substring matching turned every mention of a symptom into a red alert.
# "What are the signs of a metformin overdose?" is a legitimate educational
# question about a drug label, and answering it with "call 911 and stop
# reading this app" is both wrong and erodes trust in the alerts that matter.
#
# Three contexts suppress the alert. They are checked against the words
# immediately AROUND the matched symptom, not the whole message, so
# "I had a seizure last year, and now I have chest pain" still fires.

# Asking what a symptom means, rather than reporting it.
_INFORMATIONAL_RE = re.compile(
    r"\b(?:what (?:are|is|happens)|signs? of|symptoms? of|side ?effects? of|"
    r"warning signs?|risk of|how (?:do|would) i know|"
    # "can it cause", "can this medicine cause", "does warfarin cause"
    r"(?:can|could|does|do|would|will|might)\b(?:\s+[\w'-]+){0,3}"
    r"\s+(?:cause|causes|trigger|triggers|lead to|result in)|"
    r"is .{0,25} a (?:sign|symptom)|"
    r"tell me about|information (?:on|about)|read about|learn about)\b",
    re.IGNORECASE,
)

# Clause boundaries. A past-tense marker in a NEIGHBOURING clause must not
# suppress a symptom in the current one: in "I had a seizure last year, and
# now I have chest pain" the chest pain is happening now.
_CLAUSE_SPLIT_RE = re.compile(
    r"[,;.!?]|\band now\b|\bbut\b|\bhowever\b|\bcurrently\b|\btoday\b",
    re.IGNORECASE,
)

# The symptom is denied.
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|denies|denied|free of|absence of|"
    r"haven'?t had|hasn'?t had|didn'?t have|don'?t have|doesn'?t have)\b",
    re.IGNORECASE,
)

# The symptom happened in the past, or belongs to someone's history.
_PAST_RE = re.compile(
    r"\b(?:last (?:year|month|week)|years? ago|months? ago|weeks? ago|"
    r"used to|history of|previously|in the past|as a child|when i was|"
    r"since then|recovered from)\b",
    re.IGNORECASE,
)

# How many characters either side of the match count as "context".
_CONTEXT_WINDOW = 60


def _suppressed_by_context(text_lower: str, symptom: str) -> bool:
    """True when the words around `symptom` show it is not a live emergency."""
    start = text_lower.find(symptom)
    if start == -1:
        return False

    # Clip both sides to the clause the symptom actually sits in.
    left_raw = text_lower[max(0, start - _CONTEXT_WINDOW):start]
    right_raw = text_lower[start + len(symptom): start + len(symptom) + _CONTEXT_WINDOW]
    left = _CLAUSE_SPLIT_RE.split(left_raw)[-1]
    right = _CLAUSE_SPLIT_RE.split(right_raw)[0]
    window = f"{left} {right}"

    # An informational framing almost always precedes the symptom
    # ("what are the signs of an overdose"), so only the left side counts.
    if _INFORMATIONAL_RE.search(left):
        return True
    if _NEGATION_RE.search(left):
        return True
    if _PAST_RE.search(window):
        return True
    return False


def detect_emergency_symptoms(text: str) -> Tuple[bool, List[str]]:
    """
    Detect emergency symptoms that require immediate care.

    A symptom only counts when it is being *reported as happening*. Questions
    about what a symptom means, denials, and past-tense history are excluded —
    see the context filters above.

    Returns:
        (has_emergency, detected_categories)
    """
    if not text:
        return False, []

    text_lower = text.lower()
    detected = []

    for category, symptoms in EMERGENCY_SYMPTOMS.items():
        for symptom in symptoms:
            if symptom in text_lower and not _suppressed_by_context(text_lower, symptom):
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
    def validate_user_intent(message: str, disease: str, age: int) -> Dict[str, Any]:
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


def validate_input(message: str, disease: str, age: int) -> Dict[str, Any]:
    """Validate user input for safety."""
    return MedicalSafetyGuard.validate_user_intent(message, disease, age)


def get_emergency_warning(symptoms: List[str]) -> str:
    """Get emergency warning for symptoms."""
    return MedicalSafetyGuard.build_emergency_warning(symptoms)
