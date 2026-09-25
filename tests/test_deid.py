import json
from pathlib import Path

import pytest

from src.deid import detect_pii, render_deidentified

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def found(note: str) -> list[tuple[str, str]]:
    """Return (text, label) for every detected span, in order."""
    return [(note[s["start"] : s["end"]], s["label"]) for s in detect_pii(note)]


def masked(note: str) -> str:
    return render_deidentified(note, detect_pii(note))


# --- Each hospital's header format -------------------------------------------

# One note header per hospital, as written in the training data.
# Every PII value in it must be found with the right label.
@pytest.mark.parametrize(
    "note, expected",
    [
        (
            "Patient: Tobias Berger | DOB: 25.05.1960 | Case ID: B-904427\n"
            "Address: Birkenpfad 37, 13353 Berlin | Telephone: +49 30 0000 2893",
            [
                ("Tobias Berger", "PATIENT_NAME"),
                ("25.05.1960", "DATE_OF_BIRTH"),
                ("B-904427", "PATIENT_ID"),
                ("Birkenpfad 37, 13353 Berlin", "ADDRESS"),
                ("+49 30 0000 2893", "PHONE_NUMBER"),
            ],
        ),
        (
            "Name: Naveen Iyer | Age/Sex: 57/M | UHID: CHN-8371950\n"
            "DOB: 25-May-1969 | Admission: 28-Jun-2026\n"
            "Residence: Flat 5-A, 37 Palm Avenue, Chennai 600018 | Mobile: +91 00000 50327",
            [
                ("Naveen Iyer", "PATIENT_NAME"),
                ("CHN-8371950", "PATIENT_ID"),
                ("25-May-1969", "DATE_OF_BIRTH"),
                ("28-Jun-2026", "ENCOUNTER_DATE"),
                ("Flat 5-A, 37 Palm Avenue, Chennai 600018", "ADDRESS"),
                ("+91 00000 50327", "PHONE_NUMBER"),
            ],
        ),
        (
            "MRN: HYD371017 | Patient: Aarohi Qureshi | DOB: 16/08/1976\n"
            "Visit: 14/05/2025 | Contact: +91 00000 23780 | Address: Flat 13-A, 63 Hill Crest, Hyderabad 500034",
            [
                ("HYD371017", "PATIENT_ID"),
                ("Aarohi Qureshi", "PATIENT_NAME"),
                ("16/08/1976", "DATE_OF_BIRTH"),
                ("14/05/2025", "ENCOUNTER_DATE"),
                ("+91 00000 23780", "PHONE_NUMBER"),
                ("Flat 13-A, 63 Hill Crest, Hyderabad 500034", "ADDRESS"),
            ],
        ),
    ],
    ids=["berlin", "chennai", "hyderabad"],
)
def test_hospital_header_formats(note: str, expected: list[tuple[str, str]]) -> None:
    assert found(note) == expected


# Header lines with no "Patient:" word: the name is recognised by what follows it.
@pytest.mark.parametrize(
    "note, name",
    [
        ("B-406380 // 14.02.2026\nClara Scholz, born 23.06.1976", "Clara Scholz"),  # ", born"
        ("NODE NOTE\nKiran Naidu / HYD994420 / born 22/04/1960", "Kiran Naidu"),  # "/ ID"
        ("Seen today: Clara Scholz, 56 y, stable.", "Clara Scholz"),  # ", 56 y" (age)
    ],
)
def test_patient_name_found_by_cue_after_it(note: str, name: str) -> None:
    assert (name, "PATIENT_NAME") in found(note)


# Every clinician sign-off wording seen in the data.
@pytest.mark.parametrize(
    "note",
    [
        "Treating clinician: Deepa Naidu\nDx: HTN.",
        "Signed: Deepa Naidu (deepa.naidu@synthetic-hospital.example).",
        "Responsible doctor Deepa Naidu, email deepa.naidu@synthetic-hospital.example",
        "Consultant: Dr Deepa Naidu\nKnown diagnoses: HTN.",
        "Reviewed by Deepa Naidu; deepa.naidu@synthetic-hospital.example",
        "Attending physician: Dr. Deepa Naidu\nDiagnoses: COPD.",
        "Electronically signed by Dr. Deepa Naidu; contact deepa.naidu@synthetic-clinic.example.",
        "Author: Deepa Naidu <deepa.naidu@synthetic-clinic.example>",
    ],
)
def test_clinician_trigger_wordings(note: str) -> None:
    assert ("Deepa Naidu", "CLINICIAN_NAME") in found(note)


# --- Names -------------------------------------------------------------------

# "Dr." / "Dr" is a title, not part of the name, so it must stay visible.
@pytest.mark.parametrize(
    "note, expected",
    [
        ("Attending physician: Dr. Emil Brandt", "Attending physician: Dr. [CLINICIAN_NAME]"),
        ("Consultant: Dr Emil Brandt", "Consultant: Dr [CLINICIAN_NAME]"),
    ],
)
def test_title_is_not_part_of_name(note: str, expected: str) -> None:
    assert masked(note) == expected


# A name after "Dr." in an unfamiliar sentence is still a clinician.
def test_title_marks_clinician_in_any_sentence() -> None:
    assert ("Emil Brandt", "CLINICIAN_NAME") in found("Discussed with Dr. Emil Brandt today.")


# A patient who is a doctor: "Patient:" beats the "Dr." title, for every mention.
def test_doctor_who_is_the_patient_keeps_patient_label() -> None:
    note = "Patient: Dr. Anna Keller | DOB: 01.02.1970. Dr. Anna Keller was advised rest."
    labels = {label for text, label in found(note) if text == "Anna Keller"}
    assert labels == {"PATIENT_NAME"}


# A name that appears twice is masked both times (the ground truth labels each mention).
def test_repeated_name_is_masked_every_time() -> None:
    note = "Treating clinician: Deepa Naidu\nDx: HTN.\nSigned: Deepa Naidu."
    assert found(note).count(("Deepa Naidu", "CLINICIAN_NAME")) == 2


# Clinician emails are "firstname.lastname@...", so the matching name is found
# even with no trigger word before it.
def test_clinician_found_from_email() -> None:
    note = "Follow-up arranged with Deepa Naidu (deepa.naidu@synthetic-hospital.example)."
    assert ("Deepa Naidu", "CLINICIAN_NAME") in found(note)


# German umlauts: "krueger" in the email matches "Krüger" in the text.
def test_email_match_handles_umlauts() -> None:
    note = "Follow-up with Philipp Krüger <philipp.krueger@synthetic-clinic.example>."
    assert ("Philipp Krüger", "CLINICIAN_NAME") in found(note)


# If the email belongs to a name already known to be the patient, it stays a patient.
def test_email_does_not_override_patient_label() -> None:
    note = "Patient: Anna Keller | DOB: 01.02.1970. Contact anna.keller@mail.example"
    assert ("Anna Keller", "PATIENT_NAME") in found(note)


# Name shapes not in the training data: hyphen, apostrophe, particles, initials, capitals, 3 words.
@pytest.mark.parametrize(
    "note, name",
    [
        ("Patient: Anne-Marie Schulz | DOB: 01.02.1970", "Anne-Marie Schulz"),
        ("Patient: Liam O'Brien | DOB: 01.02.1970", "Liam O'Brien"),
        ("Patient: Jan van der Berg | DOB: 01.02.1970", "Jan van der Berg"),
        ("Signed: Dr. R. Menon", "R. Menon"),
        ("PATIENT: CLARA SCHOLZ | DOB: 01.02.1970", "CLARA SCHOLZ"),
        ("Name: Venkata Ramana Rao | Age/Sex: 60/M", "Venkata Ramana Rao"),
    ],
)
def test_unusual_name_shapes(note: str, name: str) -> None:
    assert name in [text for text, label in found(note) if label.endswith("_NAME")]


# --- Dates, IDs, phones, addresses in unseen formats --------------------------

# Dates written in other formats are still found.
@pytest.mark.parametrize("date", ["14 May 2025", "May 14, 2025", "2025-05-14", "1.2.2025", "3rd March 1960"])
def test_other_date_formats(date: str) -> None:
    assert (date, "ENCOUNTER_DATE") in found(f"Visit: {date}.")


# Birth date vs visit date: decided by the nearest cue word, not by distance.
def test_birth_and_visit_dates_side_by_side() -> None:
    note = "DOB: 16/08/1976\nVisit: 14/05/2025"
    assert found(note) == [("16/08/1976", "DATE_OF_BIRTH"), ("14/05/2025", "ENCOUNTER_DATE")]


# An ID format we have never seen is found when it follows an ID keyword.
def test_unseen_id_format_after_keyword() -> None:
    assert ("XY-99887766", "PATIENT_ID") in found("MRN: XY-99887766 | Patient: Clara Scholz")


# An international number ("+49 ...") is a phone even without the word "phone".
def test_international_phone_without_keyword() -> None:
    assert ("+49 30 0000 7311", "PHONE_NUMBER") in found("Results to +49 30 0000 7311.")


# An address split over two lines is masked completely (nothing left on line 2).
def test_address_across_two_lines() -> None:
    note = "Address: Flat 13-A, 63 Hill Crest,\nHyderabad 500034"
    assert masked(note) == "Address: [ADDRESS]"


# A street address with no "Address:" word is recognised by its street shape.
def test_street_address_without_trigger() -> None:
    assert masked("Lives in Birkenweg 12, 10115 Berlin with family.") == "Lives in [ADDRESS] with family."


# An address without a postal code is masked up to the next separator.
def test_address_without_postcode() -> None:
    assert masked("Address: 12 Baker Street | Tel") == "Address: [ADDRESS] | Tel"


# --- Clinical content must stay visible ----------------------------------------

# Drug brands and diagnoses look like names ("Tab Lasix") but must not be masked.
@pytest.mark.parametrize(
    "note",
    [
        "Rx: Tab Eliquis, Tab Lasix, Tab Xarelto.",
        "Diagnoses: Heart Failure, Atrial Fibrillation.",
        "Pt Tab Lasix 40 mg.",  # a patient trigger followed by a drug, not a name
        "Dx: Parkinson disease, Crohn Disease, Graves disease.",
        "creatinine 1,14 mg/dL; Hb 12,6 g/dL; BP 159/84; HR 88 bpm; LVEF 59%.",
    ],
)
def test_clinical_terms_are_not_masked(note: str) -> None:
    assert detect_pii(note) == []


# --- Output rules --------------------------------------------------------------

# An empty note gives no spans and an empty masked text.
def test_empty_note() -> None:
    assert detect_pii("") == []
    assert render_deidentified("", []) == ""


# Placeholders already in the text are not masked again.
def test_existing_placeholders_are_left_alone() -> None:
    assert detect_pii("Patient: [PATIENT_NAME] | DOB: [DATE_OF_BIRTH]") == []


# Windows line endings (\r\n) don't break detection.
def test_windows_line_endings() -> None:
    note = "Patient: Clara Scholz\r\nDOB: 01.02.1970"
    assert found(note) == [("Clara Scholz", "PATIENT_NAME"), ("01.02.1970", "DATE_OF_BIRTH")]


def _public_notes() -> list[tuple[str, list[dict]]]:
    notes = [(r["note_text"], r["labels"]["pii_entities"]) for r in map(json.loads, open(DATA_DIR / "train.jsonl"))]
    inputs = {r["case_id"]: r["note_text"] for r in map(json.loads, open(DATA_DIR / "validation_inputs.jsonl"))}
    for truth in map(json.loads, open(DATA_DIR / "validation_ground_truth.jsonl")):
        notes.append((inputs[truth["case_id"]], truth["pii_entities"]))
    return notes


# On all 150 public notes: spans never overlap, stay inside the note, and are sorted.
# Overlaps matter because the evaluator drops them.
def test_spans_are_sorted_non_overlapping_and_in_bounds() -> None:
    for note, _ in _public_notes():
        spans = detect_pii(note)
        for span in spans:
            assert 0 <= span["start"] < span["end"] <= len(note)
        for left, right in zip(spans, spans[1:]):
            assert left["end"] <= right["start"]


# The masked text is exactly what the evaluator rebuilds from our spans.
def test_masked_text_matches_spans() -> None:
    note = "Patient: Aarohi Qureshi | DOB: 16/08/1976"
    assert masked(note) == "Patient: [PATIENT_NAME] | DOB: [DATE_OF_BIRTH]"


# Regression check: every labelled span in the 150 public notes is found exactly.
def test_public_data_is_fully_detected() -> None:
    for note, truth in _public_notes():
        expected = {(e["start"], e["end"], e["label"]) for e in truth}
        predicted = {(s["start"], s["end"], s["label"]) for s in detect_pii(note)}
        assert predicted == expected, note[:60]


# --- Known limitations (expected to fail; they pass once fixed) ----------------

# A one-word name is not accepted: a single capital word could be anything.
@pytest.mark.xfail(strict=True, reason="names need at least two words")
def test_single_word_name() -> None:
    assert ("Madonna", "PATIENT_NAME") in found("Patient: Madonna | DOB: 01.02.1970")


# A name with no trigger word and no cue after it has no context to confirm it.
@pytest.mark.xfail(strict=True, reason="no context around the name")
def test_name_without_any_context() -> None:
    assert ("Clara Scholz", "PATIENT_NAME") in found("Seen today: Clara Scholz is stable.")


# A patient who is a doctor, with only "Dr." and no role word, is labelled clinician.
# The name is still masked; only the label is wrong.
@pytest.mark.xfail(strict=True, reason="a title with no role word defaults to clinician")
def test_doctor_patient_without_role_word() -> None:
    assert ("Anna Keller", "PATIENT_NAME") in found("Dr. Anna Keller was admitted with chest pain.")


# A relative's name has no label among the 8, so it is not masked.
@pytest.mark.xfail(strict=True, reason="no label for relatives' names")
def test_relative_name() -> None:
    assert "Anna Scholz" not in masked("Patient: Clara Scholz | DOB: 01.02.1970. Wife Anna Scholz present.")
