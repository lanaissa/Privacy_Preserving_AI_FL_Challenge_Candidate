import json
from pathlib import Path

import pytest

from src.extraction import extract_clinical_data

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOLERANCE = {
    "heart_rate_bpm": 3,
    "systolic_bp_mmhg": 5,
    "creatinine_mg_dl": 0.10,
    "hemoglobin_g_dl": 0.30,
    "lvef_percent": 3,
}


def diagnoses(note: str) -> list[str]:
    return extract_clinical_data(note)["diagnoses"]


def medications(note: str) -> list[str]:
    return extract_clinical_data(note)["medications"]


# --- Output shape ------------------------------------------------------------------

# An empty note still returns all 9 fields: empty lists and null values.
def test_empty_note_returns_all_fields_empty() -> None:
    assert extract_clinical_data("") == {
        "diagnoses": [],
        "medications": [],
        "heart_rate_bpm": None,
        "systolic_bp_mmhg": None,
        "creatinine_mg_dl": None,
        "hemoglobin_g_dl": None,
        "lvef_percent": None,
        "smoking_status": None,
        "allergy": None,
    }


# One full note per hospital, as written in the training data: every field must be right.
@pytest.mark.parametrize(
    "note, expected",
    [
        (
            "Diagnoses: COPD; stable ischemic heart disease; high blood pressure; type 2 diabetes mellitus.\n"
            "On assessment: HR 88 bpm; blood pressure 142 over 88. Laboratory: creatinine 0,91 mg/dL. "
            "Echocardiography: LVEF 59%.\nMedication list: Aspirin 100 mg daily; Atorvastatin 40 mg nightly; "
            "metformin; Ramipril 5 mg daily.\nAllergies: no medication allergy documented. "
            "Smoking status: stopped smoking several years ago. No evidence of Parkinson disease.",
            {
                "diagnoses": ["copd", "coronary_artery_disease", "hypertension", "type_2_diabetes"],
                "medications": ["aspirin", "atorvastatin", "metformin", "ramipril"],
                "heart_rate_bpm": 88, "systolic_bp_mmhg": 142, "creatinine_mg_dl": 0.91,
                "hemoglobin_g_dl": None, "lvef_percent": 59, "smoking_status": "former", "allergy": "none",
            },
        ),
        (
            "Known diagnoses: HFrEF; paroxysmal AF.\nObs—BP 130/86 mmHg; pulse 92/min. Laboratory: "
            "creatinine 1.08 mg/dL; Hb 130 g/L. Echocardiography: LVEF 35%.\n"
            "Medicines at discharge: Tab Lasix 40 mg OD; Eliquis 5 mg b.i.d.; Tab bisoprolol 5 mg OD.\n"
            "Drug allergy: allergic to penicillin. Tobacco history: non-smoker.",
            {
                "diagnoses": ["atrial_fibrillation", "heart_failure"],
                "medications": ["apixaban", "bisoprolol", "furosemide"],
                "heart_rate_bpm": 92, "systolic_bp_mmhg": 130, "creatinine_mg_dl": 1.08,
                "hemoglobin_g_dl": 13.0, "lvef_percent": 35, "smoking_status": "never", "allergy": "penicillin",
            },
        ),
        (
            "Dx: infective consolidation. Rx: azithro 500. Clinical data: pulse 87/min; BP 159/84 mmHg. "
            "Laboratory: serum creatinine 59 µmol/L; hemoglobin 12.3 g/dL. Echocardiography: LVEF 58%. "
            "Allergy: allergic to iodinated contrast. Smoking: ongoing tobacco use.",
            {
                "diagnoses": ["pneumonia"],
                "medications": ["azithromycin"],
                "heart_rate_bpm": 87, "systolic_bp_mmhg": 159, "creatinine_mg_dl": 0.67,
                "hemoglobin_g_dl": 12.3, "lvef_percent": 58, "smoking_status": "current",
                "allergy": "iodinated_contrast",
            },
        ),
    ],
    ids=["berlin", "chennai", "hyderabad"],
)
def test_hospital_note_formats(note: str, expected: dict) -> None:
    assert extract_clinical_data(note) == expected


# --- Diagnosis synonyms ------------------------------------------------------------

# Different wordings of the same illness map to one canonical name (seen in the data, plus unseen ones).
@pytest.mark.parametrize(
    "wording, canonical",
    [
        ("HTN", "hypertension"),
        ("arterial hypertension", "hypertension"),
        ("hypertensive disease", "hypertension"),
        ("arterielle Hypertonie", "hypertension"),
        ("T2DM", "type_2_diabetes"),
        ("DM2", "type_2_diabetes"),
        ("type II diabetes", "type_2_diabetes"),
        ("diabetes mellitus", "type_2_diabetes"),
        ("rapid AF", "atrial_fibrillation"),
        ("Vorhofflimmern", "atrial_fibrillation"),
        ("afib", "atrial_fibrillation"),
        ("congestive cardiac failure", "heart_failure"),
        ("LV failure", "heart_failure"),
        ("HFpEF", "heart_failure"),
        ("CKD stage III", "chronic_kidney_disease"),
        ("chronic renal dysfunction", "chronic_kidney_disease"),
        ("IHD", "coronary_artery_disease"),
        ("ischaemic heart disease", "coronary_artery_disease"),
        ("NSTEMI", "acute_coronary_syndrome"),
        ("unstable angina", "acute_coronary_syndrome"),
        ("community-acquired pneumonia", "pneumonia"),
        ("bronchopneumonia", "pneumonia"),
        ("CAP", "pneumonia"),
        ("COAD", "copd"),
        ("chronic obstructive airway disease", "copd"),
    ],
)
def test_diagnosis_synonyms(wording: str, canonical: str) -> None:
    assert diagnoses(f"Dx: {wording}.") == [canonical]


# Long-term narrowed arteries (CAD) and a sudden event (ACS) are separate labels.
def test_stable_vs_unstable_coronary_disease() -> None:
    assert diagnoses("Dx: stable ischemic heart disease.") == ["coronary_artery_disease"]
    assert diagnoses("Dx: unstable angina.") == ["acute_coronary_syndrome"]


# Look-alike diseases that must NOT be mapped.
@pytest.mark.parametrize(
    "note", ["Pulmonary hypertension.", "Type 1 diabetes mellitus.", "Gestational diabetes mellitus."]
)
def test_different_diseases_are_not_mapped(note: str) -> None:
    assert diagnoses(note) == []


# Abbreviations only match as whole words, and only in capitals.
@pytest.mark.parametrize(
    "note",
    [
        "Reviewed after rounds.",  # "af" inside "after"
        "Cap amoxicillin 500 mg.",  # "Cap" = capsule, not "CAP" = pneumonia
        "Seen by Dr. Afzal.",  # "Af" inside a name
        "Caffeine intake high.",
        "ADMISSION NOTE - STAFF SAFETY REVIEW - DECADE",  # all-caps words hiding "DM", "AF", "CAD"
    ],
)
def test_abbreviations_need_whole_word_in_capitals(note: str) -> None:
    assert diagnoses(note) == []


# "CKD-3" still counts as CKD: a hyphen after the abbreviation is fine.
def test_abbreviation_followed_by_hyphen() -> None:
    assert diagnoses("Dx: CKD-3; DM2.") == ["chronic_kidney_disease", "type_2_diabetes"]


# --- Medication synonyms -----------------------------------------------------------

# Brands, short forms and spelling variants map to the generic name.
@pytest.mark.parametrize(
    "wording, canonical",
    [
        ("Lasix", "furosemide"),
        ("frusemide", "furosemide"),
        ("Eliquis", "apixaban"),
        ("APX 5 mg BID", "apixaban"),
        ("Xarelto", "rivaroxaban"),
        ("Ecosprin", "aspirin"),
        ("ASA", "aspirin"),
        ("acetylsalicylic acid", "aspirin"),
        ("azithro", "azithromycin"),
        ("insulin glargine", "insulin"),
        ("metoprolol succinate", "metoprolol"),
        ("Coumadin", "warfarin"),
        ("Amlodipin", "amlodipine"),
    ],
)
def test_medication_synonyms(wording: str, canonical: str) -> None:
    assert medications(f"Rx: {wording}.") == [canonical]


# --- Negation: mentioned but not active / not current -------------------------------

# The three distractor sentences from the public data.
@pytest.mark.parametrize(
    "note",
    [
        "The patient denies a history of COPD.",
        "Atrial fibrillation was considered but not confirmed.",
        "Apixaban was discussed but was not started.",
    ],
)
def test_distractor_sentences_from_data(note: str) -> None:
    result = extract_clinical_data(note)
    assert result["diagnoses"] == [] and result["medications"] == []


# Negated, ruled out, uncertain or a relative's condition: not an active diagnosis.
@pytest.mark.parametrize(
    "note",
    [
        "No evidence of heart failure.",
        "Pneumonia ruled out on CXR.",
        "Suspected pneumonia.",
        "Family history of CAD in father.",
        "Diabetes mellitus in her mother.",
    ],
)
def test_negated_diagnoses(note: str) -> None:
    assert diagnoses(note) == []


# A cue only reaches its own clause, and "but" ends it.
@pytest.mark.parametrize(
    "note, expected",
    [
        ("No chest pain but known AF.", ["atrial_fibrillation"]),
        ("Dx: AF. Echo: no heart failure.", ["atrial_fibrillation"]),
        ("Dx: HTN, CKD-3. No AF.", ["chronic_kidney_disease", "hypertension"]),
        ("Known AF with no recent admissions.", ["atrial_fibrillation"]),  # "no" comes after AF
    ],
)
def test_negation_scope(note: str, expected: list[str]) -> None:
    assert diagnoses(note) == expected


# Stopped, avoided or allergy drugs are not current medications.
@pytest.mark.parametrize(
    "note, expected",
    [
        ("Warfarin stopped due to bleeding; started apixaban.", ["apixaban"]),
        ("Rx: metformin; no insulin.", ["metformin"]),
        ("Metformin discontinued. Rx: insulin.", ["insulin"]),
        ("Allergic to aspirin.", []),
    ],
)
def test_medications_not_current(note: str, expected: list[str]) -> None:
    assert medications(note) == expected


# --- Lab values and unit conversion -------------------------------------------------

# Creatinine in any unit ends up in mg/dL.
@pytest.mark.parametrize(
    "note, expected",
    [
        ("creatinine 1.21 mg/dL", 1.21),
        ("creatinine 1,21 mg/dL", 1.21),  # decimal comma (Berlin)
        ("serum creatinine 59 µmol/L", 0.67),  # micro sign (Hyderabad)
        ("creatinine 79 μmol/L", 0.89),  # Greek mu: looks the same, different character
        ("creatinine 79 umol/L", 0.89),  # plain "u"
        ("creatinine was 106 µmol/L", 1.2),
        ("sCr 1.4 mg/dl", 1.4),
        ("Kreatinin 0,9 mg/dl", 0.9),
    ],
)
def test_creatinine_units(note: str, expected: float) -> None:
    assert extract_clinical_data(note)["creatinine_mg_dl"] == pytest.approx(expected, abs=0.01)


# Hemoglobin in any unit ends up in g/dL.
@pytest.mark.parametrize(
    "note, expected",
    [
        ("hemoglobin 12.3 g/dL", 12.3),
        ("Hb 133 g/L", 13.3),  # Chennai
        ("Hb 12,6 g/dL", 12.6),
        ("Hgb 9.8 g/dl", 9.8),
        ("Hämoglobin 8.5 mmol/L", 13.69),  # German labs
    ],
)
def test_hemoglobin_units(note: str, expected: float) -> None:
    assert extract_clinical_data(note)["hemoglobin_g_dl"] == pytest.approx(expected, abs=0.01)


# No unit written: the unit is inferred from the size of the number.
def test_missing_unit_is_inferred_from_value() -> None:
    assert extract_clinical_data("creatinine 88")["creatinine_mg_dl"] == pytest.approx(1.0, abs=0.01)
    assert extract_clinical_data("creatinine 1.3")["creatinine_mg_dl"] == 1.3
    assert extract_clinical_data("Hb 120")["hemoglobin_g_dl"] == 12.0


# An impossible unit is treated as a typo (1.2 µmol/L would be 0.014 mg/dL).
def test_impossible_unit_is_treated_as_typo() -> None:
    assert extract_clinical_data("creatinine 1.2 µmol/L")["creatinine_mg_dl"] == 1.2


# Look-alike tests that are not the lab we want.
def test_other_lab_tests_are_ignored() -> None:
    assert extract_clinical_data("creatinine clearance 45 mL/min; creatinine 1.1 mg/dL")["creatinine_mg_dl"] == 1.1
    assert extract_clinical_data("HbA1c 7.2 %; Hb 13.1 g/dL")["hemoglobin_g_dl"] == 13.1
    assert extract_clinical_data("Hb A1c 6.8%")["hemoglobin_g_dl"] is None


# A lab that is mentioned without a value gives null, never a guess.
def test_lab_without_value_is_null() -> None:
    assert extract_clinical_data("hemoglobin not measured")["hemoglobin_g_dl"] is None


# --- Smoking -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "note, expected",
    [
        ("Smoking: never smoked.", "never"),
        ("Tobacco history: no tobacco use.", "never"),
        ("Lifelong non-smoker.", "never"),
        ("Does not smoke.", "never"),
        ("Nichtraucher.", "never"),
        ("Smoking status: ex-smoker.", "former"),
        ("Quit smoking in 2015.", "former"),
        ("Gave up smoking 10 years ago.", "former"),
        ("Ex-Raucher seit 2010.", "former"),
        ("Smoking: ongoing tobacco use.", "current"),
        ("Currently smokes.", "current"),
        ("Smokes 10 cigarettes a day.", "current"),
        ("20 bidis per day.", "current"),
    ],
)
def test_smoking_wordings(note: str, expected: str) -> None:
    assert extract_clinical_data(note)["smoking_status"] == expected


# Someone else's smoking does not count.
@pytest.mark.parametrize(
    "note, expected",
    [
        ("Smoking: never smoked. Father smokes.", "never"),
        ("Husband is a current smoker; patient never smoked.", "never"),
        ("Exposed to passive smoking.", None),
    ],
)
def test_smoking_of_other_people_is_ignored(note: str, expected: str | None) -> None:
    assert extract_clinical_data(note)["smoking_status"] == expected


# --- Allergy -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "note, expected",
    [
        ("Allergy: allergic to penicillin.", "penicillin"),
        ("Allergy: penicillin causes rash.", "penicillin"),
        ("Allergic to amoxicillin (rash).", "penicillin"),  # a penicillin-class antibiotic
        ("PCN allergy.", "penicillin"),
        ("Allergies: NSAID allergy.", "nsaid"),
        ("ibuprofen-associated urticaria; current smoker.", "nsaid"),  # a reaction, no word "allergy"
        ("Diclofenac causes hives.", "nsaid"),
        ("Allergies: contrast medium reaction.", "iodinated_contrast"),
        ("Kontrastmittelallergie.", "iodinated_contrast"),  # German compound word
        ("Penicillinallergie.", "penicillin"),
    ],
)
def test_allergens(note: str, expected: str) -> None:
    assert extract_clinical_data(note)["allergy"] == expected


# Every way of saying "no allergies".
@pytest.mark.parametrize(
    "note",
    [
        "Allergies: NKDA.",
        "Allergy: no known drug allergies.",
        "Allergies: no medication allergy documented.",
        "Allergies: none.",
        "Allergy: nil.",
        "NKA.",
        "Denies any allergies.",
        "Keine bekannten Allergien.",
    ],
)
def test_no_allergy_wordings(note: str) -> None:
    assert extract_clinical_data(note)["allergy"] == "none"


# A drug or contrast mentioned outside an allergy statement is not an allergy.
@pytest.mark.parametrize(
    "note, expected",
    [
        ("Penicillin V 500 mg QID. No known drug allergies.", "none"),
        ("CT with contrast performed. Allergies: NKDA.", "none"),
        ("Meds: ibuprofen PRN.", None),
        ("Not allergic to penicillin. Allergy: ibuprofen-associated urticaria.", "nsaid"),
    ],
)
def test_allergen_needs_allergy_context(note: str, expected: str | None) -> None:
    assert extract_clinical_data(note)["allergy"] == expected


# Deliberate choice: an aspirin allergy is not mapped to NSAID (see REPORT.md).
def test_aspirin_allergy_is_not_mapped() -> None:
    assert extract_clinical_data("Allergic to aspirin.")["allergy"] is None


# --- Public data regression check ---------------------------------------------------

def _public_notes() -> list[tuple[str, str, dict]]:
    notes = [
        (r["case_id"], r["note_text"], r["labels"]["extracted_clinical_data"])
        for r in map(json.loads, open(DATA_DIR / "train.jsonl"))
    ]
    inputs = {r["case_id"]: r["note_text"] for r in map(json.loads, open(DATA_DIR / "validation_inputs.jsonl"))}
    for truth in map(json.loads, open(DATA_DIR / "validation_ground_truth.jsonl")):
        notes.append((truth["case_id"], inputs[truth["case_id"]], truth["extracted_clinical_data"]))
    return notes


# All 150 public notes: every field correct (numbers within the evaluator's tolerance).
def test_public_data_is_fully_extracted() -> None:
    for case_id, note, truth in _public_notes():
        result = extract_clinical_data(note)
        for field, value in truth.items():
            if field in TOLERANCE and value is not None and result[field] is not None:
                assert abs(result[field] - value) <= TOLERANCE[field], (case_id, field)
            else:
                assert result[field] == value, (case_id, field)


# --- Known limitations (expected to fail; they pass once fixed) ----------------------

# A cue after a term belongs to the nearest term, but the rule applies it to the whole clause.
@pytest.mark.xfail(strict=True, reason="an after-cue ('ruled out') also negates earlier terms in the clause")
def test_after_cue_of_a_later_term() -> None:
    assert diagnoses("AF, but CKD was ruled out.") == ["atrial_fibrillation"]


# A past, resolved illness is not active, but the word "resolved" is not recognised.
@pytest.mark.xfail(strict=True, reason="resolved past illnesses are not detected")
def test_resolved_illness() -> None:
    assert diagnoses("History of pneumonia, resolved.") == []


# Negation stated in the next sentence is not linked back.
@pytest.mark.xfail(strict=True, reason="negation in a following sentence is not linked back")
def test_negation_in_next_sentence() -> None:
    assert diagnoses("Echo was checked for heart failure. None found.") == []


# Only the first lab value is used, even when a later one is the current value.
@pytest.mark.xfail(strict=True, reason="only the first value of a lab test is read")
def test_latest_lab_value() -> None:
    note = "Admission creatinine 2.1 mg/dL; discharge creatinine 1.3 mg/dL."
    assert extract_clinical_data(note)["creatinine_mg_dl"] == 1.3
