# Intelligent Document Anomaly & Inconsistency Detection System

An end-to-end system that ingests business documents (invoices, contracts,
applications, reports), extracts structured information from them, and
flags suspicious inconsistencies — mismatched dates, arithmetic errors,
duplicate submissions, unusual amounts, conflicting statements, missing
fields, look-alike vendor names, and deviations from historical patterns —
along with a plain-English explanation for every flag.

Built as an internship project to demonstrate: **OCR, NLP/feature
extraction, feature engineering, unsupervised ML (anomaly detection),
embeddings/similarity, and explainable AI**, combined into one working
pipeline rather than treated as separate exercises.

---

## 1. Why this design

Real anomaly detection in documents needs two complementary strategies:

1. **Rule-based checks** for problems that are best expressed as explicit
   logic — a due date before an invoice date is *always* wrong, regardless
   of what a statistical model thinks. Rules are also inherently explainable.
2. **Unsupervised ML** for problems that only show up as an unusual
   *combination* of otherwise-plausible values — nothing individually looks
   wrong, but the overall pattern doesn't match history. This is what an
   Isolation Forest is good at, and what pure rules would miss.

Neither approach alone is enough — rules miss the “huh, that’s weird but not
technically illegal” cases, and ML models alone are black boxes that are
noisy and hard to trust. The system runs both and merges their outputs into
one report.

---

## 2. Architecture

```
                 ┌───────────────┐
  raw file  ───► │  ingestion.py │  OCR / PDF text layer / plain text
 (.pdf/.png/.txt)└──────┬────────┘
                        ▼
                ┌────────────────────┐
                │ field_extraction.py│  regex/NLP field parsing
                │  (invoice #, dates,│  → structured field dict
                │  amounts, vendor…) │
                └──────┬─────────────┘
                        ▼
        ┌───────────────────────────────┐
        │   feature_engineering.py      │  builds interpretable numeric
        │  (vendor baselines, z-scores, │  features + loads historical
        │   date deltas, text stats)    │  vendor profiles (CSV)
        └──────┬─────────────────┬──────┘
               ▼                 ▼
     ┌──────────────────┐  ┌───────────────────┐
     │  rule_checks.py   │  │  anomaly_model.py  │  Isolation Forest,
     │ (explicit logic:  │  │ (unsupervised ML,  │  trained on historical
     │  dates, math,     │  │  trained on history)│  ("normal") documents
     │  dup IDs, missing │  └─────────┬──────────┘
     │  fields, vendor   │            │ + permutation-based
     │  spoofing, etc.)  │            │   feature attribution
     └─────────┬─────────┘            │
               │                      │
               │        ┌─────────────┘
               ▼        ▼
        ┌────────────────────┐        ┌───────────────────┐
        │ explainability.py  │◄───────┤   similarity.py    │
        │  merges rule flags,│        │ TF-IDF "embeddings"│
        │  ML attributions,  │        │ + cosine similarity│
        │  duplicate matches │        │ across the batch   │
        │  → risk level +    │        └───────────────────┘
        │  ranked reasons    │
        └─────────┬──────────┘
                   ▼
         DocumentReport (JSON / CLI / Flask UI)
```

`src/pipeline.py`'s `DocumentAnomalyPipeline` class is the one object that
wires all of the above together; `main.py` (CLI) and `app.py` (Flask web
demo) are both just thin front-ends over it.

---

## 3. Concept → module mapping

| Concept | Where it lives | What it does |
|---|---|---|
| **OCR** | `src/ingestion.py` | Grayscale + denoise + adaptive-threshold preprocessing, then Tesseract OCR for images and scanned PDF pages; `pdfplumber` for native PDF text layers. |
| **NLP / information extraction** | `src/field_extraction.py` | Regex-based field extraction (invoice #, PO #, dates, amounts, vendor, payment terms) plus flexible date parsing (`dateutil`) and detection of key phrases ("paid in full", "balance due", "draft"/"final"). |
| **Feature engineering** | `src/feature_engineering.py` | Turns raw fields + historical vendor statistics into an interpretable numeric vector: amount z-score vs. vendor history, days between invoice/due date, deviation from typical payment terms, weekend-dating, digit density, etc. |
| **Unsupervised ML / anomaly detection** | `src/anomaly_model.py` | Isolation Forest trained only on historical ("normal") documents; scores new documents by how well they fit learned patterns — catches anomalies no single rule would catch. |
| **Embeddings** | `src/similarity.py` | TF-IDF vectorization ("embedding") of document text + cosine similarity to catch duplicate/near-duplicate submissions within a batch. Interface is embedding-model-agnostic — see §6. |
| **Explainable AI** | `src/explainability.py` | Combines rule flags + a **permutation-based feature attribution** for the ML model (a dependency-free stand-in for SHAP: each feature is neutralized to the historical median and the resulting drop in anomaly score becomes its "contribution") into one ranked, human-readable list of reasons — never a bare score. |

---

## 4. Project layout

```
doc_anomaly_project/
├── README.md
├── requirements.txt
├── generate_sample_data.py     # builds the synthetic dataset (already run)
├── main.py                     # CLI: batch-analyze a folder of documents
├── app.py                      # Flask web demo
├── templates/index.html        # demo UI
├── src/
│   ├── ingestion.py
│   ├── field_extraction.py
│   ├── feature_engineering.py
│   ├── rule_checks.py
│   ├── similarity.py
│   ├── anomaly_model.py
│   ├── explainability.py
│   └── pipeline.py
├── data/
│   ├── historical_invoices.csv       # 250 synthetic "normal" invoices (training data)
│   ├── sample_documents/             # 18 test documents, most normal, several anomalous
│   └── sample_documents_index.csv    # ground-truth notes for each sample (for grading/demo)
├── tests/
│   └── test_pipeline.py        # 11 unit/integration tests, one per anomaly type
└── outputs/                    # generated reports land here
```

---

## 5. Running it

```bash
pip install -r requirements.txt
# Tesseract OCR engine must also be installed system-side for image OCR,
# e.g. `sudo apt-get install tesseract-ocr` on Ubuntu.

# 1. (Already done once, but re-runnable) regenerate the synthetic dataset:
python generate_sample_data.py

# 2. Run the CLI over the sample batch:
python main.py
#   -> prints a risk-ranked summary + reasons to the terminal
#   -> writes outputs/report.json with the full structured report

# 3. Or launch the interactive web demo:
python app.py
#   -> open http://127.0.0.1:5000, pick a sample or upload your own
#      .txt / .pdf / .png / .jpg document

# 4. Run the test suite:
python -m unittest tests/test_pipeline.py -v
```

### Example CLI output

```
[ HIGH ] doc_10_duplicate_invoice_number.txt — 2 issue(s) flagged
    - (high) Total amount is 11.2 standard deviations higher than this vendor's historical average...
    - (high) Invoice number 'INV-10149' already exists in historical records — possible duplicate submission...
[ HIGH ] doc_12_amount_outlier.txt — 2 issue(s) flagged
    - (high) Total amount is 12.5 standard deviations higher than this vendor's historical average...
    - (low) Subtotal ($2,600.00) and total are suspiciously round numbers...
[ LOW  ] doc_02_normal.txt — 0 issue(s) flagged
```

Every one of the 12 injected anomaly types in `data/sample_documents/` is
correctly identified (see `data/sample_documents_index.csv` for the
ground-truth list and `tests/test_pipeline.py` for automated verification).

---

## 6. Design notes & future improvements

- **Embeddings**: TF-IDF is used instead of a transformer embedding model
  (e.g. `sentence-transformers`, OpenAI/Anthropic embeddings API) so the
  project has zero heavyweight/network dependencies. `src/similarity.py`
  exposes a small, swappable interface (`embed()` / `pairwise_similarity()`)
  specifically so it can be upgraded to dense semantic embeddings without
  touching any other module — this would meaningfully improve detection of
  *paraphrased* duplicate/conflicting content (not just near-identical text).
- **Explainability without SHAP**: SHAP/LIME aren't assumed to be available
  in every environment, so `anomaly_model.py` implements a permutation-based
  attribution method that's faithful to the actual Isolation Forest (not a
  separate surrogate model). Swapping in `shap.TreeExplainer` is a drop-in
  option if the library is available.
- **Field extraction**: currently regex-based for portability. A production
  system would likely add a spaCy/transformer NER model or an LLM-based
  extractor for messier, non-templated documents (e.g. free-form contracts),
  behind the same `extract_fields(text) -> dict` contract.
- **Conflicting statements**: currently keyword/phrase-based (e.g. "paid in
  full" + "balance due"). A stronger version would use sentence embeddings +
  a natural-language-inference (NLI) model to catch semantically
  contradictory sentences that don't share exact keywords.
- **Feedback loop**: flagged documents that a human reviewer confirms as
  false positives/negatives could be fed back to retrain the Isolation
  Forest or tune `contamination`/thresholds — turning this into a
  semi-supervised system over time.
- **Scale**: for large volumes, `historical_invoices.csv` would become a
  proper database with incremental model retraining rather than a
  fit-on-startup CSV load.

---

## 7. Dataset

Since this is a project (not a client with real financial documents), all
data is **synthetically generated** by `generate_sample_data.py`:

- `historical_invoices.csv`: 250 plausible invoices across 10 vendors with
  realistic amount distributions and payment terms, used purely to establish
  "normal" patterns.
- `data/sample_documents/`: 18 new documents to analyze — 8 clean/normal and
  10 with a deliberately injected, labeled anomaly (see
  `sample_documents_index.csv`), so the system's output can be checked
  against known ground truth.
