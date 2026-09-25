# REPORT

## 1. Executive summary

_To be completed._

## 2. System architecture

_To be completed._

## 3. De-identification

In this step, the goal is to find the personal information (PII) in each clinical note and replace it with a placeholder such as `[PATIENT_NAME]`, while keeping the clinical content readable. The starter code already detected 5 of the 8 labels: patient ID, date of birth, encounter date, phone number and email (the emails in this dataset belong to clinicians, not patients). It had no code at all for patient names, clinician names and addresses, so on the validation set it missed all 105 of those spans and leaked 45.7% of the PII characters.

### Starting point

First, I pushed the challenge files to GitHub as they were, updated Python to 3.11 (the version used in the Dockerfile), created a virtual environment and installed the requirements. Then I ran `make evaluate` on the starter to confirm the setup works and to get a baseline:

| Metric (validation, 30 cases) | Starter | After my changes |
|---|---:|---:|
| De-identification score | 0.7279 | **1.0000** |
| PII characters leaked | 45.7% | **0%** |
| Structured extraction score | 0.8797 | 0.8797 (not changed yet) |
| Readmission prediction score | 0.7727 | 0.7727 (not changed yet) |
| Automated points | 31.84 / 40 | 35.92 / 40 |

### Method: rule-based detection

I chose a rule-based approach (regex patterns plus context rules) instead of a trained model. The notes follow a small number of templates per hospital, there are only 120 labelled training notes, the evaluation runs without GPU or network access, and the README favours a simple, well-tested solution over an opaque one. I moved the de-identification code from `src/baseline.py` into its own module, `src/deid.py`.

Each rule answers two questions: what does the value look like (its shape), and what words around it confirm it is PII (its context). Emails, IDs, dates and phone numbers have a fixed shape, so the shape is almost enough. Names and addresses don't have a fixed format, which is the main challenge: they need several patterns and they need context. The other challenge is that the rules should not only fit the public data. The data card says the hidden test set contains new formatting variants, so every rule has a general version on top of the dataset-specific one.

**Names.** First, I created a general name pattern: 2 to 4 capitalised words, including accented letters (Krüger), hyphens (Anne-Marie), apostrophes (O'Brien), inner capitals (McDonald), particles (van der Berg), initials (R. Menon) and all-caps names. Titles such as Dr., Mr. or Mrs. are matched outside the name, so they stay visible after masking, as in the ground truth. Then I separated patients from clinicians by context:

- *Patients:* a word before the name (`Patient:`, `Name:`, `Pt`, `Mrs.`), or, when there is no such word, a cue after the name: a date of birth or "born", a patient ID, or an age ("Clara Scholz, 49 y"). This covers header lines like `Kiran Naidu / HYD994420 / born 22/04/1960` that have no "patient" word at all.
- *Clinicians:* all 8 sign-off wordings found in the data (`Treating clinician:`, `Signed:`, `Reviewed by`, `Author:`, ...) plus similar ones that could appear in unseen notes (`seen by`, `dictated by`, `prepared by`, `referred by`, `physician`, `surgeon`). A "Dr." or "Prof." title marks a clinician in any sentence. A name that matches a clinician email (`deepa.naidu@...` → Deepa Naidu) is also a clinician, which catches sign-offs that have no trigger word.
- *A patient who is a doctor:* "Patient: Dr. Anna Keller" would be labelled as a clinician if the title rule ran first. So the rules with an explicit role (Patient:, born, Signed:) run first, and the title and email rules keep the label of a name already found. Later mentions of the same person keep the patient label too.
- *Repeated names:* once a name is found, every other mention of it in the note is masked with the same label.

**Addresses.** Addresses contain commas, so instead of stopping at punctuation, the main rule starts after a trigger word (`Address:`, `Residence`, `Home:`, `from`) and ends at the postal code: 5 digits plus city for Berlin (`10785 Berlin`), city plus 6 digits for India (`Chennai 600018`). An address can continue on a second line after a comma. For unseen templates there is a second rule for street shapes without a trigger (`Birkenweg 12`, `12 Baker Street`, `Flat 5-A, 37 Gandhi Nagar`), and a last fallback that masks everything after `Address:` up to the next separator when there is no postal code.

**Extending the 5 existing labels.** IDs: besides the known formats, any code containing a digit after an ID keyword (`MRN: XY-99887766`). Dates: numeric dates with 1 or 2 digit day and month, ISO dates, and written months (`14 May 2025`, `May 14, 2025`). Birth date vs. encounter date: the starter looked for "dob" in the previous 20 characters, which only worked because of the note layout. Now a date is a birth date only when a birth word is closer to it than any visit word (`Visit`, `seen`, `Admission`, `DOA`), looking back no further than the previous date. Phones: any international number starting with `+` is a phone even without a word like "phone".

### Order of the rules and overlapping spans

The evaluator ignores overlapping spans, so a new span is rejected if it overlaps one already found. Because of this the order matters: fixed-shape entities run first (email, ID, date, phone), then addresses, then names, and inside names the strongest evidence runs first.

### Over-redaction controls

Masking clinical content by mistake would hide useful information. To avoid it:

- A name must be capitalised and must have context (a trigger, a cue, a title or a matching email). A capitalised phrase alone is never masked.
- Any name candidate that contains a drug, dosage, diagnosis or section word is rejected. I built this word list from the canonical vocabulary in `DATA_DICTIONARY.md` and from the data: the only capitalised non-PII phrases in the 120 training notes are `Tab Lasix`, `Tab Xarelto` and `Tab Eliquis`, which have exactly the shape of a first and last name.
- Diseases named after people (Parkinson, Graves) are safe without being on the list, because "Parkinson disease" is not a two-word capitalised name. I left surnames like Wilson off the list on purpose, since they are also real patient names.
- Dates need three parts, so lab values (`12,6`) and blood pressure (`159/84`) are never read as dates.

### Results and error analysis

On the public data, every labelled span is found exactly, with no false detections, for every label and every hospital. I checked hospitals separately to see if the rules were fitted to one site's format:

| Hospital | Train (120 notes) | Validation (30 notes) |
|---|---|---|
| Berlin | 370 / 370 | 88 / 88 |
| Chennai | 312 / 312 | 80 / 80 |
| Hyderabad | 339 / 339 | 87 / 87 |

Since the public data is templated, a perfect score here doesn't prove the rules work on the hidden set. To test unseen formats, I wrote about 30 made-up edge cases. With the first version, 9 of them leaked PII (apostrophe and particle names, all-caps names, initials, written-out dates, a phone without a keyword, an unseen ID format, and an address split over two lines). After extending the rules, the remaining failures are the limitations listed in section 8.

One error I found through the per-label scores: widening the date-of-birth look-back window from 20 to 25 characters labelled 49 Hyderabad visit dates as birth dates, because in `DOB: 16/08/1976 / Visit: 14/05/2025` the word "DOB" was now within reach of the second date. This showed that a fixed distance is fragile, and led to the "closest cue wins" rule above.

### Tests

The starter only had one smoke test for de-identification. `tests/test_deid.py` adds tests for each hospital's header format, every clinician wording, the title staying out of the name, the doctor-patient case, repeated names, names from emails, unusual name shapes, other date/ID/phone/address formats, clinical terms that must not be masked, empty notes, no overlapping spans on all 150 public notes, and a full check against the public labels. The known limitations are written as tests marked "expected to fail" (`xfail`, strict), so they stay visible, and pytest warns if one starts passing so the marker can be removed.

### Extending to scanned documents and images

The current data is text only, but the same detector can be reused for other formats:

- **Scanned documents and PDFs:** run OCR, which returns the text plus a bounding box for each word. Run `detect_pii` on the OCR text, map each character span back to the boxes of the words it covers, and black out those regions of the image. OCR mistakes (for example "0" read as "O") would lower recall, so the patterns would need to tolerate them.
- **Medical images:** PII is often in the file metadata (for example DICOM header fields such as patient name and birth date), which can be removed field by field, and in text burned into the image, which can go through the same OCR path.
- **Checking the result:** after masking, running OCR again on the output image should find no PII.

## 4. Structured extraction and standardization

_To be completed._

## 5. Federated-learning experiment

_To be completed._

## 6. Privacy extension and threat model

_To be completed._

## 7. Reproducibility and testing

- Python 3.11 in a virtual environment, matching the Dockerfile.
- `make test` runs all tests (currently 53 passed, 4 expected failures for known de-identification limitations).
- `make evaluate` rebuilds `outputs/validation_predictions.jsonl` and `outputs/validation_report.json`. The `outputs/` folder is in `.gitignore`, so evaluation results are regenerated rather than committed.
- The de-identification rules are deterministic: the same note always gives the same spans.

_To be completed for the other components._

## 8. Limitations and next steps

**De-identification, limitations of my implementation:**

- Single-word names (`Patient: Madonna`) are not detected, because a single capitalised word could be anything.
- A name with no trigger word and no cue around it (`Seen today: Clara Scholz is stable`) is not detected.
- A patient who is a doctor, mentioned only with "Dr." and no role word (`Dr. Anna Keller was admitted with chest pain`), is labelled as a clinician. The name is still masked, only the label is wrong. Some further text analysis could handle this.
- A relative's name (`Wife Anna Scholz`) is not masked, because none of the 8 labels covers relatives.
- The rules depend on trigger words and formats. A hidden note written very differently could still leak PII. A next step would be a fallback layer that finds names without context, for example a list of first names or a small pretrained named-entity model, filtered by the same clinical word list.

**De-identification, limitations of the benchmark:**

- The notes are generated from templates, so a perfect score on the public data says little about real clinical notes, which are longer and much more varied.

_To be completed for the other components._
