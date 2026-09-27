"""Rule-based de-identification of clinical notes.

Detection runs in layers: fixed-shape entities (email, ID, date, phone) first,
then addresses, then names. Name rules accept a capitalised token sequence only
when context supports it (a trigger before it, a cue after it, a title, a
matching clinician email, or an earlier confirmed mention), and a clinical-vocabulary
stoplist blocks drug and diagnosis words from being redacted as names.
"""
from __future__ import annotations

import re
from typing import Any

# A detected entity: {"start": int, "end": int, "label": str}, offsets as in note[start:end].
Span = dict[str, Any]

# --- Name building blocks -----------------------------------------------------

# Upper/lowercase letters including accented ones (Krüger, Müller, Hélène).
_UP = "A-ZÀ-ÖØ-Þ"
_LO = "a-zß-öø-ÿ"
# One capitalised name word: Clara, O'Brien, D'Souza, McDonald, Anne-Marie.
_WORD = rf"(?:[{_UP}]['’])?[{_UP}][{_LO}]+(?:[{_UP}][{_LO}]+)?(?:-[{_UP}][{_LO}]+)?"
# One all-caps name word: CLARA, SCHOLZ.
_CAPS_WORD = rf"[{_UP}]{{2,}}(?:-[{_UP}]{{2,}})?"
# An initial: "R." in "R. Menon".
_INITIAL = rf"[{_UP}]\."
# Lowercase joining words inside surnames: "Jan van der Berg".
_PARTICLE = r"(?:van|von|de|der|den|del|da|di|du|la|le|bin|al|ten|ter)"


def _name_pattern(word: str) -> str:
    # 2-4 tokens; literal spaces (not \s) so a name never runs across a line break.
    # At most 3 particles in a row ("van der", "de la"), which also keeps matching fast.
    token = rf"(?:{word}|{_INITIAL})"
    return rf"{token}(?: (?:{_PARTICLE} ){{0,3}}{token}){{1,3}}"


# Normal-case names, used everywhere (patients and clinicians).
NAME = _name_pattern(_WORD)
# ALL-CAPS names are only accepted after an explicit trigger; elsewhere they are
# indistinguishable from headers such as "SYNTHETIC EMR EXPORT" or abbreviations.
NAME_ANY_CASE = _name_pattern(rf"(?:{_WORD}|{_CAPS_WORD})")
# Titles before a name. Matched outside the name so they are not masked.
_TITLE = r"(?:(?:Dr|Prof|Mr|Mrs|Ms|Miss)\.?\s+)"
# Finds titles at the start of a matched text, so they can be trimmed off.
_TITLE_PREFIX = re.compile(rf"^{_TITLE}+")

# --- Fixed-shape entities -----------------------------------------------------

# Any email address.
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# ID formats known from the data and the starter code (B-406380, CHN-8371950, HYD371017, ...).
_KNOWN_ID = r"(?:B-\d{6}|CHN-\d{7}|HYD\d{6}|BER/\d{4}/\d{2}|UHID/\d{5}/\d{2}|MRN-\d{2}-\d{5})"
KNOWN_ID_PATTERN = re.compile(rf"\b{_KNOWN_ID}\b")
# Unseen ID formats: any digit-bearing token after an ID keyword ("MRN: XY-99887766").
# Group 1 is the ID itself, without the keyword.
KEYWORD_ID_PATTERN = re.compile(
    r"(?i:\b(?:MRN|UHID|PID|case\s+id|patient\s+id|hospital\s+(?:no|number)|record\s+(?:no|number))\b\.?\s*[:#]?\s*)"
    r"([A-Z0-9][A-Z0-9/-]*\d[A-Z0-9/-]*)"
)

# Month names, short or long, any case: Mar, March, MAR.
_MONTH = (
    r"(?i:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
# Dates in four forms. All need three parts, so lab values (12,6) and BP (159/84) never match.
DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,2}[./-]\d{1,2}[./-](?:\d{4}|\d{2})"  # 14.02.2026, 16/08/1976, 1-2-25
    r"|\d{4}-\d{2}-\d{2}"  # 2025-05-14
    rf"|\d{{1,2}}(?:st|nd|rd|th)?[ -]{_MONTH}\.?,?[ -]\d{{4}}"  # 14 May 2025, 22-Apr-2025
    rf"|{_MONTH}\.? \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}"  # May 14, 2025
    r")\b"
)
# Words that mark a date as a birth date, or as a visit/admission date.
DOB_CUE = re.compile(r"(?i:\bdob\b|d\.o\.b|\bborn\b|\bbirth|\bgeb\.)")
ENCOUNTER_CUE = re.compile(r"(?i:\bvisit|\bseen\b|\badmi(?:ssion|tted)|\bdoa\b|\bencounter|\bdischarge|\bdate of service)")

# A phone-shaped number: optional "+country code", then groups of digits.
PHONE_PATTERN = re.compile(r"(?:\+\d{1,3}[ -]?)?(?:\(?\d{2,5}\)?[ /-]?){2,4}\d{3,8}")
# Words that mark a number as a phone.
PHONE_CUE = re.compile(r"(?i:phone|\btel|mobile|\bmob\b|contact|\bph\b|\bcell|\bfax|\bcall)")

# --- Addresses ----------------------------------------------------------------

# A city name of one or two words: Berlin, New Delhi.
_CITY = rf"[{_UP}][{_LO}]+(?: [{_UP}][{_LO}]+)?"
# Addresses contain commas, so they end on the postal code: "10785 Berlin", "Chennai 600018".
_POSTCODE_END = rf"(?:\b\d{{5}} {_CITY}|{_CITY}[ -]+\d{{6}}\b)"
_ADDRESS_BODY = r"(?:[^\n|;]|(?<=,)\n)"  # a line break is allowed only straight after a comma
# Words that introduce an address.
_ADDRESS_TRIGGER = r"(?i:\b(?:address|residence|resident of|home|lives at|from)\b\s*:?\s*)"
# Trigger, then the shortest text (3-120 chars) that ends in a postal code. Group 1 is the address.
ADDRESS_WITH_POSTCODE = re.compile(rf"{_ADDRESS_TRIGGER}({_ADDRESS_BODY}{{3,120}}?{_POSTCODE_END})")
# No trigger: a recognisable street shape, optionally followed by postcode and city.
# German: "Birkenweg 12". English/Indian: "Flat 5-A, 37 Gandhi Nagar", "12 Baker Street".
STREET_ADDRESS = re.compile(
    rf"\b(?:"
    rf"[{_UP}][{_LO}]+(?:straße|strasse|str\.|weg|allee|ring|pfad|platz|damm|gasse|ufer) \d{{1,4}}[a-z]?"
    rf"|(?:(?:Flat|Apt\.?|Apartment|Unit|House|Plot|Door) (?:No\.? ?)?[\w-]+, )?\d{{1,4}}[A-Za-z]?,? "
    rf"(?:[{_UP}][{_LO}]+ ){{1,3}}(?:Road|Rd\.|Street|St\.|Avenue|Ave\.|Lane|Nagar|Marg|Colony|Layout|Crescent|Drive)"
    rf")(?:,? {_POSTCODE_END})?"
)
# Last resort when there is no postal code: "Address:" up to the next | ; line break or sentence end.
ADDRESS_FALLBACK = re.compile(r"(?i:\b(?:address|residence|home)\s*:\s*)([^\n|;]{3,120}?)(?=\s*(?:[|;\n]|\.\s|$))")

# --- Names --------------------------------------------------------------------
# In every name pattern, group 1 is the name alone (no trigger word, no title).

# Patient word before the name: "Patient: Clara Scholz", "Name: ...", "Pt ...", "Mrs. ...".
PATIENT_TRIGGER_NAME = re.compile(
    rf"(?i:\b(?:patient\s+name|patient|pt|name|mr|mrs|ms|miss)\b\.?\s*:?\s*){_TITLE}?({NAME_ANY_CASE})"
)
# Clinician word before the name: "Signed: ...", "Reviewed by ...", "Attending physician: Dr. ...".
# Longer phrases come first so "electronically signed by" wins over "signed".
CLINICIAN_TRIGGER_NAME = re.compile(
    r"(?i:\b(?:treating\s+clinician|attending\s+physician|responsible\s+doctor|electronically\s+signed\s+by|"
    r"signed\s+by|reviewed\s+by|seen\s+by|dictated\s+by|prepared\s+by|referred\s+by|consultant|attending|"
    r"physician|clinician|doctor|surgeon|author|signed)\b\s*:?\s*)"
    rf"{_TITLE}?({NAME_ANY_CASE})"
)
# "Dr." / "Prof." marks a clinician in any template.
CLINICIAN_TITLE_NAME = re.compile(rf"\b(?:Dr|Prof)\.?\s+({NAME})")
# An ID right after a name: a known format, or a short code plus 5+ digits.
_ID_AFTER = rf"(?:{_KNOWN_ID}|[A-Z]{{1,5}}[-/]?\d{{5,}})"
# Header lines with no trigger: "Clara Scholz, born ...", "Kiran Naidu / HYD994420", "Clara Scholz, 49 y".
# The (?=...) part must follow the name but is not included in the match.
PATIENT_CUE_NAME = re.compile(
    rf"(?<![\w.])({NAME})(?="
    rf"\s*[|(,/]?\s*(?i:dob\b|d\.o\.b|born\b)"
    rf"|\s*[/|]\s*{_ID_AFTER}"
    rf"|,\s*\d{{1,3}}\s*(?i:y|yrs?|years?|yo)\b"
    rf")"
)
# Any name-shaped text, used only to compare against email names.
ANY_NAME = re.compile(rf"(?<![\w.]){NAME}")

# Words that are never part of a person's name here. Built from the canonical
# vocabulary in DATA_DICTIONARY.md plus the only capitalised non-PII phrases in
# the training notes ("Tab Lasix", "Tab Xarelto", "Tab Eliquis").
NON_NAME_WORDS = frozenset(
    """
    apixaban rivaroxaban warfarin metoprolol bisoprolol furosemide ramipril amlodipine metformin
    insulin atorvastatin aspirin clopidogrel amiodarone digoxin azithromycin
    eliquis xarelto coumadin marevan lopressor betaloc concor lasix tritace norvasc amlong glucophage
    lantus lipitor ecosprin plavix clopilet cordarone lanoxin zithromax azee
    tab tabs tablet cap caps capsule inj syr mg
    atrial fibrillation heart failure hypertension diabetes mellitus kidney renal disease chronic
    coronary artery acute syndrome pneumonia pulmonary obstructive infarction angina stroke sepsis
    htn dm af ckd cad acs copd hf nkda ecg
    hospital clinic synthetic node emr export record note summary discharge department ward unit
    team service cardiology medicine diagnosis diagnoses medication medications allergy allergies
    history assessment plan laboratory echocardiography smoking status known clinical
    """.split()
)

NAME_LABELS = ("PATIENT_NAME", "CLINICIAN_NAME")


def _add_span(spans: list[Span], start: int, end: int, label: str) -> bool:
    """Add a span unless it is empty or overlaps an existing one (first match wins)."""
    if start < 0 or end <= start:
        return False
    for span in spans:
        if start < span["end"] and end > span["start"]:
            return False
    spans.append({"start": start, "end": end, "label": label})
    return True


def _add_name(
    spans: list[Span], note: str, start: int, end: int, label: str, confirmed: dict[str, str] | None = None
) -> None:
    """Add a name span after trimming titles and rejecting clinical words."""
    text = note[start:end]
    # Drop a leading "Dr." / "Mrs." so only the name is masked.
    title = _TITLE_PREFIX.match(text)
    if title:
        start += title.end()
        text = note[start:end]
    # A name already found with a role keeps that role ("Patient: Dr. Anna Keller").
    if confirmed:
        label = confirmed.get(text, label)
    # Reject one-word matches and anything containing a drug/diagnosis/section word.
    tokens = [token.strip(".").lower() for token in text.split()]
    if len(tokens) < 2 or any(token in NON_NAME_WORDS for token in tokens):
        return
    if not re.search(f"[{_LO}]{{2}}|[{_UP}]{{2}}", text):  # initials only, e.g. "A. B."
        return
    _add_span(spans, start, end, label)


def _normalize_name(text: str) -> str:
    """Lowercase letters only, umlauts spelled out: "Philipp Krüger" -> "philippkrueger"."""
    text = text.lower()
    for source, target in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(source, target)
    return re.sub(r"[^a-z]", "", text)


def _date_label(context: str) -> str:
    """Birth date only if a birth cue is present and closer to the date than any encounter cue."""
    dob = [match.end() for match in DOB_CUE.finditer(context)]
    encounter = [match.end() for match in ENCOUNTER_CUE.finditer(context)]
    if dob and (not encounter or dob[-1] > encounter[-1]):
        return "DATE_OF_BIRTH"
    return "ENCOUNTER_DATE"


def _detect_fixed_shape(note: str, spans: list[Span]) -> None:
    """Emails, IDs, dates and phones: entities recognisable by their shape."""
    for match in EMAIL_PATTERN.finditer(note):
        _add_span(spans, *match.span(), "EMAIL")

    for match in KNOWN_ID_PATTERN.finditer(note):
        _add_span(spans, *match.span(), "PATIENT_ID")
    for match in KEYWORD_ID_PATTERN.finditer(note):
        start, end = match.span(1)
        # Trim a trailing "/" or "-" that belongs to the separator, not the ID.
        end -= len(match.group(1)) - len(match.group(1).rstrip("/-"))
        _add_span(spans, start, end, "PATIENT_ID")

    previous_date_end = 0
    for match in DATE_PATTERN.finditer(note):
        # Only look back to the previous date, so "DOB: x / Visit: y" can't label y as a birth date.
        context = note[max(previous_date_end, match.start() - 30) : match.start()]
        _add_span(spans, *match.span(), _date_label(context))
        previous_date_end = match.end()

    for match in PHONE_PATTERN.finditer(note):
        # Real phone numbers have 8-15 digits; this skips short codes and long IDs.
        digits = sum(ch.isdigit() for ch in match.group(0))
        if not 8 <= digits <= 15:
            continue
        context = note[max(0, match.start() - 20) : match.start()]
        # An international "+CC ..." number is a phone in any template; otherwise require a cue word.
        if match.group(0).startswith("+") or PHONE_CUE.search(context):
            _add_span(spans, *match.span(), "PHONE_NUMBER")


def _detect_addresses(note: str, spans: list[Span]) -> None:
    """Addresses, most reliable pattern first."""
    for match in ADDRESS_WITH_POSTCODE.finditer(note):
        _add_span(spans, *match.span(1), "ADDRESS")
    for match in STREET_ADDRESS.finditer(note):
        _add_span(spans, *match.span(), "ADDRESS")
    for match in ADDRESS_FALLBACK.finditer(note):
        _add_span(spans, *match.span(1), "ADDRESS")


def _detect_names(note: str, spans: list[Span]) -> None:
    """Patient and clinician names, strongest evidence first."""
    # Explicit role context runs first, so "Patient: Dr. Anna Keller" stays a patient.
    for pattern, label in (
        (CLINICIAN_TRIGGER_NAME, "CLINICIAN_NAME"),
        (PATIENT_TRIGGER_NAME, "PATIENT_NAME"),
        (PATIENT_CUE_NAME, "PATIENT_NAME"),
    ):
        for match in pattern.finditer(note):
            _add_name(spans, note, *match.span(1), label)

    # Weaker evidence (a title, a matching email) defaults to clinician but keeps
    # the role of a name already confirmed above.
    confirmed = {note[s["start"] : s["end"]]: s["label"] for s in spans if s["label"] in NAME_LABELS}
    for match in CLINICIAN_TITLE_NAME.finditer(note):
        _add_name(spans, note, *match.span(1), "CLINICIAN_NAME", confirmed)

    # Emails are clinician contacts ("firstname.lastname@...", see DATA_DICTIONARY.md).
    email_names = {
        _normalize_name(match.group(0).split("@")[0])
        for match in EMAIL_PATTERN.finditer(note)
    }
    for match in ANY_NAME.finditer(note):
        # Try suffixes so a leading word ("Consultant Rakesh Menon") doesn't block the match.
        for word in re.finditer(r"\S+", match.group(0)):
            if _normalize_name(match.group(0)[word.start() :]) in email_names:
                _add_name(spans, note, match.start() + word.start(), match.end(), "CLINICIAN_NAME", confirmed)
                break

    # A confirmed name is labelled everywhere else it appears in the note.
    found = {(note[s["start"] : s["end"]], s["label"]) for s in spans if s["label"] in NAME_LABELS}
    for text, label in found:
        # (?<!\w) / (?!\w): whole words only, so "Anna" never matches inside "Annabel".
        for match in re.finditer(rf"(?<!\w){re.escape(text)}(?!\w)", note):
            _add_span(spans, *match.span(), label)


def detect_pii(note: str) -> list[Span]:
    """Return non-overlapping PII spans sorted by position."""
    spans: list[Span] = []
    # Order matters: overlapping matches are rejected, so the most certain detectors run first.
    _detect_fixed_shape(note, spans)
    _detect_addresses(note, spans)
    _detect_names(note, spans)
    return sorted(spans, key=lambda span: (span["start"], span["end"]))


def render_deidentified(note: str, spans: list[Span]) -> str:
    """Replace each span with its [LABEL] placeholder."""
    output = note
    # Replace from the end so earlier offsets stay valid as the text length changes.
    for span in reversed(spans):
        output = output[: span["start"]] + f"[{span['label']}]" + output[span["end"] :]
    return output
