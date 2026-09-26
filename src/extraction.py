"""Structured extraction of clinical fields from notes (starter logic, not yet improved)."""
from __future__ import annotations

import re
from typing import Any

# Every wording that means each canonical diagnosis: the ones seen in the notes plus
# common abbreviations, spellings and German terms (Berlin) for unseen notes.
# Entries are regex fragments matched as whole words. Entries written in capitals
# are abbreviations and must appear in capitals ("CAP" = pneumonia, "cap" = capsule).
DIAGNOSIS_SYNONYMS = {
    "atrial_fibrillation": [
        "atrial fibrillation", "AF", "afib", "a-fib", "vorhofflimmern", "a fib"
    ],
    "heart_failure": [
        "heart failure", "cardiac failure", "LV failure", "left ventricular failure",
        "HFrEF", "HFpEF", "HFmrEF", "CHF", "CCF", "herzinsuffizienz", "HF", "heart failure with reduced ejection fraction",
         "heart failure with mildly reduced ejection fraction", "heart failure with preserved ejection fraction",
         "congestive heart failure", "congestive cardiac failure"
    ],
    "hypertension": [
        # "pulmonary/portal hypertension" are different diseases.
        r"(?<!pulmonary )(?<!portal )(?<!intracranial )hypertension", "HTN", "high blood pressure",
        "hypertensive disease", "arterielle hypertonie", "hypertonie",
    ],
    "type_2_diabetes": [
        r"type (?:2|ii) diabetes", r"diabetes mellitus,? type (?:2|ii)", "T2DM", "DM2", "DM",
        "NIDDM", r"diabetes mellitus typ 2",
        # Plain "diabetes mellitus" is labelled type 2 in all 15 public notes, but not type 1 or gestational.
        r"(?<!type 1 )(?<!type i )(?<!gestational )diabetes mellitus",
    ],
    "chronic_kidney_disease": [
        "chronic kidney disease", "CKD", r"chronic renal (?:disease|dysfunction|failure|insufficiency|impairment)",
        "chronische niereninsuffizienz",
    ],
    "coronary_artery_disease": [
        "coronary artery disease", "coronary heart disease", "coronary disease", "CAD", "IHD",
        r"isch(?:a)?emic heart disease", "chronic coronary syndrome", "KHK", "koronare herzkrankheit",
    ],
    "acute_coronary_syndrome": [
        "acute coronary syndrome", "ACS", "NSTEMI", "STEMI", "NSTE-ACS", "unstable angina",
        "acute myocardial infarction", "AMI", "ACS"
    ],
    "pneumonia": [
        "pneumonia", "bronchopneumonia", "CAP", "infective consolidation", "pneumonie",
    ],
    "copd": [
        "COPD", "COAD", r"chronic obstructive (?:pulmonary|airways?|lung) disease",
        "chronic airway obstruction", "obstructive lung disease",
    ],
}

# Generic names plus brands, short forms and spelling variants (British "frusemide",
# German "furosemid"). Brands must map to the generic name (DATA_DICTIONARY.md).
MEDICATION_SYNONYMS = {
    "apixaban": ["apixaban", "eliquis", "APX"],
    "rivaroxaban": ["rivaroxaban", "xarelto"],
    "warfarin": ["warfarin", "coumadin", "marevan", "jantoven"],
    "metoprolol": ["metoprolol", "lopressor", "betaloc", "toprol"],
    "bisoprolol": ["bisoprolol", "concor", "cardicor"],
    "furosemide": ["furosemide", "frusemide", "furosemid", "lasix"],
    "ramipril": ["ramipril", "tritace", "altace"],
    "amlodipine": ["amlodipine", "amlodipin", "norvasc", "amlong", "istin"],
    "metformin": ["metformin", "glucophage", "glycomet"],
    "insulin": ["insulin", "lantus", "levemir", "tresiba", "toujeo", "novorapid", "humalog", "humulin"],
    "atorvastatin": ["atorvastatin", "atorva", "lipitor", "sortis"],
    "aspirin": ["aspirin", "acetylsalicylic acid", "ASA", "ASS", "ecosprin", "disprin"],
    "clopidogrel": ["clopidogrel", "plavix", "clopilet"],
    "amiodarone": ["amiodarone", "amiodaron", "cordarone", "pacerone"],
    "digoxin": ["digoxin", "lanoxin"],
    "azithromycin": ["azithromycin", "azithro", "zithromax", "azee", "azithral"],
}


def _compile_synonyms(synonyms: dict[str, list[str]]) -> dict[str, list[re.Pattern[str]]]:
    """Turn each wording into a whole-word pattern; all-caps entries stay case-sensitive."""
    compiled = {}
    for canonical, wordings in synonyms.items():
        patterns = []
        for wording in wordings:
            flags = 0 if wording.isupper() else re.IGNORECASE
            # (?<!\w) / (?!\w): whole words only, so "AF" never matches inside "after"
            # but "CKD-3" still matches "CKD".
            patterns.append(re.compile(rf"(?<!\w)(?:{wording})(?!\w)", flags))
        compiled[canonical] = patterns
    return compiled


DIAGNOSIS_PATTERNS = _compile_synonyms(DIAGNOSIS_SYNONYMS)
MEDICATION_PATTERNS = _compile_synonyms(MEDICATION_SYNONYMS)


# --- Negation: mentions that are not an active diagnosis or a current drug -------

# A clause ends at a sentence end, semicolon, pipe, line break, bracket, "?" or "!".
_CLAUSE_END = re.compile(r"\.(?=\s|$)|[;|\n\[\]?!]")
# Cues BEFORE the term, in the same clause: negated, ruled out, uncertain, a relative's
# condition, or a drug the patient is not taking ("The patient denies a history of COPD").
_NEGATION_BEFORE = re.compile(
    r"(?i)\b(?:no|not|denies|denied|without|negative for|free of|absence of|ruled out|rule out|r/o|"
    r"excluded|never had|possible|suspected|query|family history|fhx|mother|father|brother|sister|"
    r"sibling|parents?|grandmother|grandfather|aunt|uncle|relatives?|stopped|discontinued|held|"
    r"allergic to|allergy to|intolerant (?:of|to)|previously on|no longer on|declined|refused)\b"
)
# Cues AFTER the term, in the same clause ("AF was considered but not confirmed",
# "Apixaban was discussed but was not started", "diabetes in her mother").
_NEGATION_AFTER = re.compile(
    r"(?i)\b(?:was|were) (?:considered|discussed|suspected|ruled out|excluded|stopped|discontinued|held)\b"
    r"|\bnot (?:confirmed|started|present|found)\b|\bruled out\b|\bunlikely\b"
    r"|\b(?:stopped|discontinued|ceased|withheld)\b"
    r"|\bin (?:his|her|the|a) (?:mother|father|brother|sister|sibling|parents?|relatives?)\b"
)
# Words that end the reach of an earlier cue: in "no chest pain but known AF", AF is active.
_SCOPE_END = re.compile(r"(?i)\b(?:but|however|although|except|apart from|aside from)\b")


def _clause_bounds(note: str, start: int, end: int) -> tuple[int, int]:
    """Start and end of the clause containing note[start:end]."""
    clause_start = max((m.end() for m in _CLAUSE_END.finditer(note, 0, start)), default=0)
    next_break = _CLAUSE_END.search(note, end)
    return clause_start, (next_break.start() if next_break else len(note))


def _is_negated(note: str, start: int, end: int) -> bool:
    """True if the mention at note[start:end] is negated, uncertain, a relative's, or not current."""
    clause_start, clause_end = _clause_bounds(note, start, end)

    before = note[clause_start:start]
    scope_ends = list(_SCOPE_END.finditer(before))
    if scope_ends:
        before = before[scope_ends[-1].end() :]
    return bool(_NEGATION_BEFORE.search(before) or _NEGATION_AFTER.search(note, end, clause_end))


def _find_terms(note: str, patterns: dict[str, list[re.Pattern[str]]]) -> list[str]:
    """Canonical names with at least one mention that is not negated, sorted."""
    return sorted(
        canonical
        for canonical, wordings in patterns.items()
        if any(not _is_negated(note, *m.span()) for p in wordings for m in p.finditer(note))
    )


# --- Lab values with unit conversion ---------------------------------------------

_NUMBER = r"(\d+(?:[.,]\d+)?)"  # group 1: the value, with a decimal point or comma
# Words allowed between the lab name and its value: "creatinine (serum): 88", "Hb was 12.1".
_LAB_GAP = r"(?:\s*\((?:serum|blood|s)\))?\s*(?:level\s*)?(?:[:=]|\bwas\b|\bis\b|\bof\b|\bat\b)?\s*"
# Group 2: the unit, if written. µ may be the micro sign, the Greek letter mu, or a plain "u".
CREATININE_PATTERN = re.compile(
    rf"(?i)\b(?:creatinine|creat|s?cr|kreatinin)\b(?!\s*clearance){_LAB_GAP}{_NUMBER}"
    r"\s*(mg\s*/\s*dl|mg\s*%|[µμu]mol\s*/\s*l|mmol\s*/\s*l)?"
)
# (?!\s*a1c): "Hb A1c" / "HbA1c" is a diabetes test, not hemoglobin.
HEMOGLOBIN_PATTERN = re.compile(
    rf"(?i)\b(?:ha?emoglobin|h[äa]moglobin|hgb|hb)\b(?!\s*a1c){_LAB_GAP}{_NUMBER}"
    r"\s*(g\s*/\s*dl|g\s*%|g\s*/\s*l|mmol\s*/\s*l)?"
)
# Multiply by these to reach the standard unit (factors from DATA_DICTIONARY.md;
# hemoglobin mmol/L is the standard conversion 1 mmol/L = 1.611 g/dL).
# The first unit in each dict is the standard one.
CREATININE_TO_MG_DL = {"mg/dl": 1.0, "mg%": 1.0, "µmol/l": 1 / 88.4, "mmol/l": 1000 / 88.4}
HEMOGLOBIN_TO_G_DL = {"g/dl": 1.0, "g%": 1.0, "g/l": 0.1, "mmol/l": 1.611}
# Values outside these ranges (in the standard unit) are not realistic for a patient.
CREATININE_RANGE = (0.1, 20.0)
HEMOGLOBIN_RANGE = (3.0, 25.0)


def _normalize_unit(unit: str | None) -> str | None:
    """'µmol / L', 'umol/L', 'μmol/l' -> 'µmol/l'."""
    if not unit:
        return None
    unit = re.sub(r"\s+", "", unit.lower())
    return re.sub(r"^[μu](?=mol)", "µ", unit)


def _lab_value(
    note: str, pattern: re.Pattern[str], factors: dict[str, float], plausible: tuple[float, float]
) -> float | None:
    """First value of a lab test, converted to the standard unit; None if not in the note."""
    match = pattern.search(note)
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    unit = _normalize_unit(match.group(2))
    # Use the written unit first. If there is no unit, or it gives an impossible value
    # ("creatinine 88" can only be µmol/L), try the other units in order.
    candidates = ([unit] if unit in factors else []) + [u for u in factors if u != unit]
    for candidate in candidates:
        converted = value * factors[candidate]
        if plausible[0] <= converted <= plausible[1]:
            return round(converted, 2)
    return None


# --- Smoking status --------------------------------------------------------------

# Checked in this order inside each clause, because "non-smoker" contains "smoker"
# and "stopped smoking" contains "smoking".
SMOKING_PATTERNS = [
    ("never", re.compile(
        r"(?i)\bnever[- ]?(?:smoked|a smoker|smoker)\b|\bnon[- ]?smoker\b|\bno (?:history of )?(?:tobacco|smoking)\b"
        r"|\bno smoking history\b|\bdenies (?:any )?(?:smoking|tobacco)\b|\bdo(?:es)? not smoke\b|\bnichtraucher"
    )),
    ("former", re.compile(
        r"(?i)\b(?:former|ex|previous|past)[- ]?smoker\b|\b(?:quit|stopped|gave up|ceased) smoking\b"
        r"|\bsmoked until\b|\bused to smoke\b|\b(?:former|previous|past) tobacco\b|\bex-?raucher"
    )),
    ("current", re.compile(
        r"(?i)\b(?:current|active)(?:ly)? smok(?:er|es|ing)\b|\bactively smokes\b|\bstill smok(?:es|ing)\b"
        r"|\bsmokes\b|\b(?:ongoing|current) tobacco\b|\bcigarettes? (?:per|a) day\b|\bbidis?\b|\braucher(?:in)?\b"
    )),
]
# Clauses about someone else's smoking are skipped ("father smokes", "passive smoking").
_OTHER_PERSON = re.compile(
    r"(?i)\b(?:mother|father|husband|wife|partner|spouse|son|daughter|brother|sister|parents?|family|passive|second[- ]hand)\b"
)


def _smoking_status(note: str) -> str | None:
    """never / former / current from the first clause that states it; None if not stated."""
    for clause in _CLAUSE_END.split(note):
        if _OTHER_PERSON.search(clause):
            continue
        for status, pattern in SMOKING_PATTERNS:
            if pattern.search(clause):
                return status
    return None


# --- Allergy ---------------------------------------------------------------------

# Allergens for the three canonical allergy classes. Penicillin-class antibiotics
# (amoxicillin, ...) count as penicillin. Aspirin is left out on purpose: it is in
# most medication lists here, and aspirin intolerance is not always an NSAID allergy.
# "\w*" lets German compound words match too ("Penicillinallergie", "Kontrastmittelallergie").
ALLERGEN_SYNONYMS = {
    "penicillin": [r"penicillin\w*", "PCN", "amoxicillin", "ampicillin", "flucloxacillin"],
    "nsaid": ["NSAIDs?", r"non-?steroidal anti-?inflammatory(?: drugs?)?", "ibuprofen", "diclofenac", "naproxen"],
    "iodinated_contrast": ["iodinated contrast", "contrast", "radiocontrast", "iodine", r"kontrastmittel\w*"],
}
ALLERGEN_PATTERNS = _compile_synonyms(ALLERGEN_SYNONYMS)
# An allergen only counts if its clause talks about an allergy or a reaction
# ("Rx: aspirin" and "contrast CT" are not allergies; "ibuprofen-associated urticaria" is).
_ALLERGY_CONTEXT = re.compile(
    r"(?i)allerg|intoleran|hypersensitiv|reaction|rash|urticaria|hives|anaphyla|angioedema|\bcauses\b|-associated|unvertr"
)
_ALLERGY_NEGATION = re.compile(r"(?i)\b(?:no|not|denies|without|negative for)\b")
NO_ALLERGY = re.compile(
    r"(?i)\bnkdas?\b|\bnka\b|\bno (?:known )?(?:drug |medication )?allerg|\ballerg(?:y|ies)\s*[:=]?\s*(?:none|nil|no)\b"
    r"|\bdenies (?:any )?allergies\b|\bkeine (?:bekannten )?allergien"
)


def _allergy(note: str) -> str | None:
    """The first allergen stated as an allergy; "none" if no allergies are stated; None if not mentioned."""
    found = []
    for canonical, patterns in ALLERGEN_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(note):
                clause_start, clause_end = _clause_bounds(note, *match.span())
                negated = _ALLERGY_NEGATION.search(note, clause_start, match.start())
                if _ALLERGY_CONTEXT.search(note, clause_start, clause_end) and not negated:
                    found.append((match.start(), canonical))
    if found:
        return min(found)[1]
    if NO_ALLERGY.search(note):
        return "none"
    return None


def _first_number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def extract_clinical_data(note: str) -> dict[str, Any]:
    diagnoses = _find_terms(note, DIAGNOSIS_PATTERNS)
    medications = _find_terms(note, MEDICATION_PATTERNS)

    heart_rate = _first_number(r"(?:\bhr\b|pulse|ventricular rate)\s*[=:]?\s*(\d{2,3})", note)
    systolic_bp = _first_number(r"(?:\bbp\b|blood pressure)\s*[=:]?\s*(\d{2,3})(?:\s*/|\s+over\s+)", note)
    creatinine = _lab_value(note, CREATININE_PATTERN, CREATININE_TO_MG_DL, CREATININE_RANGE)
    hemoglobin = _lab_value(note, HEMOGLOBIN_PATTERN, HEMOGLOBIN_TO_G_DL, HEMOGLOBIN_RANGE)
    lvef = _first_number(r"(?:lvef|\bef\b|ejection fraction(?: approximately)?)\s*[=:]?\s*(\d{2})", note)

    smoking = _smoking_status(note)
    allergy = _allergy(note)

    return {
        "diagnoses": diagnoses,
        "medications": medications,
        "heart_rate_bpm": int(heart_rate) if heart_rate is not None else None,
        "systolic_bp_mmhg": int(systolic_bp) if systolic_bp is not None else None,
        "creatinine_mg_dl": creatinine,
        "hemoglobin_g_dl": hemoglobin,
        "lvef_percent": int(lvef) if lvef is not None else None,
        "smoking_status": smoking,
        "allergy": allergy,
    }
