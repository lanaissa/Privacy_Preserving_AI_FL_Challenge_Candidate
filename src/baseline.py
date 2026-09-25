"""Deliberately simple starter baseline.

Applicants are expected to replace or substantially improve this code. It does
not implement federated learning or a privacy extension.
"""
from __future__ import annotations

import re
from typing import Any

DIAGNOSIS_KEYWORDS = {
    "atrial_fibrillation": ("atrial fibrillation",),
    "heart_failure": ("heart failure",),
    "hypertension": ("hypertension",),
    "type_2_diabetes": ("type 2 diabetes", "type 2 dm"),
    "chronic_kidney_disease": ("chronic kidney disease",),
    "coronary_artery_disease": ("coronary artery disease",),
    "acute_coronary_syndrome": ("acute coronary syndrome",),
    "pneumonia": ("pneumonia",),
    "copd": ("chronic obstructive pulmonary disease",),
}
MEDICATIONS = (
    "apixaban",
    "rivaroxaban",
    "warfarin",
    "metoprolol",
    "bisoprolol",
    "furosemide",
    "ramipril",
    "amlodipine",
    "metformin",
    "insulin",
    "atorvastatin",
    "aspirin",
    "clopidogrel",
    "amiodarone",
    "digoxin",
    "azithromycin",
)


def _first_number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def extract_clinical_data(note: str) -> dict[str, Any]:
    lower = note.lower()
    diagnoses = [
        canonical
        for canonical, keywords in DIAGNOSIS_KEYWORDS.items()
        if any(keyword in lower for keyword in keywords)
    ]
    medications = [medication for medication in MEDICATIONS if medication in lower]

    heart_rate = _first_number(r"(?:\bhr\b|pulse|ventricular rate)\s*[=:]?\s*(\d{2,3})", note)
    systolic_bp = _first_number(r"(?:\bbp\b|blood pressure)\s*[=:]?\s*(\d{2,3})(?:\s*/|\s+over\s+)", note)
    creatinine = _first_number(r"(?:serum\s+)?creatinine\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*mg/dl", note)
    hemoglobin = _first_number(r"(?:hemoglobin|\bhb\b)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*g/dl", note)
    lvef = _first_number(r"(?:lvef|\bef\b|ejection fraction(?: approximately)?)\s*[=:]?\s*(\d{2})", note)

    if any(token in lower for token in ("never smoked", "non-smoker", "no tobacco use")):
        smoking = "never"
    elif any(token in lower for token in ("former smoker", "ex-smoker", "stopped smoking")):
        smoking = "former"
    elif any(token in lower for token in ("current smoker", "actively smokes", "ongoing tobacco")):
        smoking = "current"
    else:
        smoking = None

    if any(token in lower for token in ("nkda", "no known drug allergies", "no medication allergy")):
        allergy = "none"
    elif "penicillin" in lower:
        allergy = "penicillin"
    elif any(token in lower for token in ("nsaid", "ibuprofen", "non-steroidal")):
        allergy = "nsaid"
    elif "contrast" in lower:
        allergy = "iodinated_contrast"
    else:
        allergy = None

    return {
        "diagnoses": sorted(diagnoses),
        "medications": sorted(medications),
        "heart_rate_bpm": int(heart_rate) if heart_rate is not None else None,
        "systolic_bp_mmhg": int(systolic_bp) if systolic_bp is not None else None,
        "creatinine_mg_dl": creatinine,
        "hemoglobin_g_dl": hemoglobin,
        "lvef_percent": int(lvef) if lvef is not None else None,
        "smoking_status": smoking,
        "allergy": allergy,
    }
