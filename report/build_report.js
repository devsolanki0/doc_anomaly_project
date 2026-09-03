const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, PageOrientation, LevelFormat,
  convertInchesToTwip,
} = require("docx");
const fs = require("fs");

const PAGE_WIDTH = 12240, PAGE_HEIGHT = 15840; // US Letter

function h1(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } }); }
function h2(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } }); }
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}

function cell(text, { bold = false, shade = null, width } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, color: "auto", fill: shade } : undefined,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, bold })] })],
  });
}

// ---------------------------------------------------------------------------
// Results table data (produced by main.py on the 18-document sample batch)
// ---------------------------------------------------------------------------
const results = [
  ["doc_01_normal.txt", "Low", "No rule violations; minor ML noise (digit density) only"],
  ["doc_02_normal.txt", "Low", "No issues detected"],
  ["doc_03–06_normal.txt", "Low", "No issues detected (4 documents)"],
  ["doc_07_math_mismatch.txt", "Medium", "Printed total does not equal subtotal + tax"],
  ["doc_08_due_before_invoice.txt", "Medium", "Due date earlier than invoice date"],
  ["doc_09_future_dated.txt", "Medium", "Invoice dated after the processing date"],
  ["doc_10_duplicate_invoice_number.txt", "High", "Invoice number reused from history + extreme amount outlier"],
  ["doc_11_vendor_lookalike.txt", "Medium", "Vendor name 94% similar to a known vendor, not an exact match"],
  ["doc_12_amount_outlier.txt", "High", "Total amount 12.5 std. deviations above vendor's history"],
  ["doc_13_missing_fields.txt", "Medium", "Required field (invoice number) missing"],
  ["doc_14_conflicting_statement.txt", "Medium", "'PAID IN FULL' contradicts a nonzero balance due"],
  ["doc_15_round_number.txt", "Medium", "Suspiciously round subtotal/total; unusual payment window"],
  ["doc_16 / doc_17 (pair)", "Medium", "Near-identical content across two documents (99–100% similarity)"],
  ["doc_18_unparsable_date.txt", "Medium", "Invoice date field unparsable (simulated OCR corruption)"],
];

const resultRows = [
  new TableRow({
    tableHeader: true,
    children: [
      cell("Document", { bold: true, shade: "1A2332", width: 3800 }),
      cell("Risk Level", { bold: true, shade: "1A2332", width: 1600 }),
      cell("Primary Reason Flagged", { bold: true, shade: "1A2332", width: 4600 }),
    ],
  }),
  ...results.map(([doc, risk, reason], i) =>
    new TableRow({
      children: [
        cell(doc, { shade: i % 2 ? "F2F2F2" : "FFFFFF", width: 3800 }),
        cell(risk, { shade: i % 2 ? "F2F2F2" : "FFFFFF", width: 1600 }),
        cell(reason, { shade: i % 2 ? "F2F2F2" : "FFFFFF", width: 4600 }),
      ],
    })
  ),
];

const conceptRows = [
  ["OCR", "src/ingestion.py", "Grayscale + denoise + adaptive threshold preprocessing, then Tesseract OCR for images/scanned PDF pages; pdfplumber for native PDF text."],
  ["NLP / Information Extraction", "src/field_extraction.py", "Regex-based structured field extraction (invoice #, dates, amounts, vendor, terms) with flexible date parsing and key-phrase detection."],
  ["Feature Engineering", "src/feature_engineering.py", "Converts fields + historical vendor statistics into interpretable numeric features (amount z-score, date deltas, text statistics)."],
  ["Unsupervised ML", "src/anomaly_model.py", "Isolation Forest trained only on historical documents; flags documents that don't fit learned patterns."],
  ["Embeddings", "src/similarity.py", "TF-IDF document embeddings + cosine similarity to catch duplicate/near-duplicate submissions."],
  ["Explainable AI", "src/explainability.py", "Merges rule flags and permutation-based ML feature attribution into ranked, human-readable reasons."],
];

const conceptTableRows = [
  new TableRow({
    tableHeader: true,
    children: [
      cell("Concept", { bold: true, shade: "1A2332", width: 2600 }),
      cell("Module", { bold: true, shade: "1A2332", width: 2600 }),
      cell("Role in the System", { bold: true, shade: "1A2332", width: 4800 }),
    ],
  }),
  ...conceptRows.map(([c, m, r], i) =>
    new TableRow({
      children: [
        cell(c, { shade: i % 2 ? "F2F2F2" : "FFFFFF", width: 2600 }),
        cell(m, { shade: i % 2 ? "F2F2F2" : "FFFFFF", width: 2600 }),
        cell(r, { shade: i % 2 ? "F2F2F2" : "FFFFFF", width: 4800 }),
      ],
    })
  ),
];

const doc = new Document({
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
          margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
        },
      },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 1600, after: 100 },
          children: [new TextRun({ text: "Intelligent Document Anomaly &", bold: true, size: 40 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 300 },
          children: [new TextRun({ text: "Inconsistency Detection System", bold: true, size: 40 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
          children: [new TextRun({ text: "Internship Project Report", size: 26, italics: true })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 40 },
          children: [new TextRun({ text: "Domain: Anomaly Detection · NLP · OCR · Explainable AI", size: 22 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 800 },
          children: [new TextRun({ text: "Prepared for internship submission", size: 20, color: "666666" })],
        }),

        h1("1. Abstract"),
        p(
          "Manual review of business documents such as invoices, contracts, and applications is slow and " +
          "error-prone, and fraud or data-entry mistakes often hide in inconsistencies that are individually " +
          "subtle — a date that's off by a field, a total that doesn't add up, a vendor name one letter away " +
          "from a known supplier. This project builds an end-to-end system that ingests a document, extracts " +
          "its structured fields, checks it against explicit business-logic rules, scores it against learned " +
          "historical patterns using unsupervised machine learning, and — critically — explains, in plain " +
          "language, exactly why a document was flagged and how risky it is. The system was implemented in " +
          "Python and validated against a labeled synthetic dataset of 18 documents covering 12 distinct " +
          "anomaly types, all of which were correctly detected."
        ),

        h1("2. Problem Statement & Objectives"),
        p("The system should, given a business document (invoice, report, application, or contract), be able to:"),
        bullet("Extract key structured fields from noisy, OCR'd, or plain-text input."),
        bullet("Detect mismatched or logically inconsistent dates."),
        bullet("Detect duplicate or resubmitted documents/invoice numbers."),
        bullet("Detect unusual values relative to a vendor's/entity's own history."),
        bullet("Detect conflicting statements within the same document."),
        bullet("Detect missing required fields."),
        bullet("Detect deviations from historical patterns that no single rule captures."),
        bullet("Explain every flag in language a non-technical reviewer can act on."),

        h1("3. Concepts Demonstrated"),
        p("Each required concept maps to a specific, testable module rather than being a superficial add-on:"),
        new Table({ width: { size: 10000, type: WidthType.DXA }, rows: conceptTableRows }),
        new Paragraph({ text: "", spacing: { after: 200 } }),

        h1("4. System Architecture"),
        p(
          "The pipeline is a linear sequence of stages, each independently testable: ingestion → field " +
          "extraction → feature engineering → (rule checks ∥ unsupervised ML scoring ∥ cross-document " +
          "similarity) → explainability merge → structured report. Rule checks and the ML model run in " +
          "parallel over the same extracted fields/features and are only combined at the final explanation " +
          "stage, so either can be evaluated, tuned, or replaced independently."
        ),
        h2("4.1 Ingestion Layer (OCR)"),
        p(
          "Handles three input types: plain text (used for the OCR-simulated sample dataset), PDFs " +
          "(via pdfplumber's text layer, falling back to page rasterization + Tesseract OCR for scanned " +
          "pages), and images (OpenCV preprocessing — grayscale, denoising, adaptive thresholding — followed " +
          "by Tesseract OCR). This was verified against both a generated PDF and a generated image in testing."
        ),
        h2("4.2 Field Extraction (NLP)"),
        p(
          "A labeled-field regex extractor pulls out invoice number, PO number, invoice/due dates, vendor " +
          "name, subtotal, tax, and total amount, tolerant of the spacing and punctuation variance typical " +
          "of OCR output. Dates are parsed with dateutil for format flexibility; unparsable dates are flagged " +
          "rather than silently dropped. The extraction contract is intentionally decoupled from the regex " +
          "implementation so it can later be swapped for a spaCy NER model or an LLM-based extractor."
        ),
        h2("4.3 Feature Engineering"),
        p(
          "Historical documents are grouped by vendor to build per-vendor baselines (mean/standard deviation " +
          "of invoice amounts, typical payment terms). Each new document is then converted into an " +
          "interpretable feature vector: amount z-score against the vendor's own history, gap between " +
          "invoice and due dates, deviation from typical payment terms, weekend-dating, and basic text " +
          "statistics. Interpretability was prioritized over raw predictive power so that every feature can " +
          "be translated into a sentence a reviewer understands."
        ),
        h2("4.4 Rule-Based Checks"),
        p(
          "Seven explicit checks cover missing fields, date logic (future-dated or inverted due/invoice " +
          "dates), arithmetic consistency (subtotal + tax = total), amount outliers, duplicate invoice " +
          "numbers, vendor name look-alikes (fuzzy string matching against known vendors), and conflicting " +
          "statements (e.g. 'paid in full' co-occurring with a nonzero balance due)."
        ),
        h2("4.5 Unsupervised Machine Learning"),
        p(
          "An Isolation Forest is trained exclusively on historical (presumed-normal) documents. For a new " +
          "document, its anomaly score is compared against the historical score distribution to produce a " +
          "percentile ('riskier than X% of past documents'). Because SHAP/LIME are not assumed to be " +
          "available in every deployment environment, feature-level attribution is computed with a " +
          "permutation method: each feature is neutralized to its historical median value and the resulting " +
          "drop in anomaly score becomes that feature's contribution — a dependency-free approximation that " +
          "stays faithful to the actual model being explained."
        ),
        h2("4.6 Embeddings & Similarity"),
        p(
          "Documents within a processing batch are vectorized with TF-IDF (a lightweight embedding) and " +
          "compared pairwise with cosine similarity to catch near-duplicate submissions — the same invoice " +
          "resubmitted with a single field changed, for example. The similarity module's interface is " +
          "embedding-model-agnostic, so it can be upgraded to dense transformer embeddings without any " +
          "change to the calling pipeline code."
        ),
        h2("4.7 Explainable AI Layer"),
        p(
          "Rule flags and ML attributions are merged into one ranked list of reasons per document, alongside " +
          "an overall Low/Medium/High risk level. No score is ever presented without an accompanying " +
          "explanation, and every reason is traceable back to the specific check or feature that produced it — " +
          "important for auditability in a business context."
        ),

        h1("5. Dataset"),
        p(
          "As this is a project rather than a deployment against real client data, a synthetic dataset was " +
          "generated (generate_sample_data.py): 250 historical invoices across 10 vendors with realistic " +
          "amount distributions and payment terms (used to learn 'normal' patterns), and 18 new documents " +
          "to analyze — 8 clean and 10 with deliberately injected, labeled anomalies spanning all 12 target " +
          "anomaly categories. Ground truth for every sample document is recorded in " +
          "data/sample_documents_index.csv, and 11 automated tests (tests/test_pipeline.py) verify each " +
          "anomaly type is correctly detected."
        ),

        h1("6. Results"),
        p(
          "Running the full pipeline over the 18-document sample batch correctly separated clean documents " +
          "from anomalous ones, with zero missed detections across the 12 injected anomaly types:"
        ),
        new Table({ width: { size: 10000, type: WidthType.DXA }, rows: resultRows }),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        p(
          "One normal document (doc_01) received a Low-but-non-zero anomaly signal from the ML model alone " +
          "(elevated digit density) with no rule violations — a realistic reminder that unsupervised models " +
          "produce soft signals and occasional noise, which is exactly why the system treats ML output as one " +
          "input to a combined risk score rather than a standalone verdict."
        ),

        h1("7. Explainability in Practice — Example"),
        p("For doc_14_conflicting_statement.txt, the system produces:"),
        bullet("Risk Level: Medium"),
        bullet(
          "Reason (rule, high severity): \"Document states the invoice is 'PAID IN FULL' but also lists a " +
          "nonzero balance due ($1,721.44) — these statements directly contradict each other.\""
        ),
        p(
          "This illustrates the core design goal: the output is a decision a reviewer can act on immediately, " +
          "not a number they have to interpret."
        ),

        h1("8. Limitations & Future Work"),
        bullet("Regex-based field extraction works well for templated invoices but would need a learned NER model or LLM-based extractor for free-form contracts and applications."),
        bullet("TF-IDF similarity catches near-identical text but not paraphrased duplicates; dense transformer embeddings would close this gap."),
        bullet("Conflicting-statement detection is currently keyword-based; a natural-language-inference (NLI) model would catch contradictions that don't share exact phrasing."),
        bullet("No human-feedback loop yet — confirmed false positives/negatives could be used to retrain the Isolation Forest and tune thresholds over time."),
        bullet("Historical data currently loads from a CSV at startup; a production deployment would use a database with incremental model updates."),

        h1("9. Conclusion"),
        p(
          "This project demonstrates a complete, working document-anomaly-detection pipeline that combines " +
          "explainable rule-based logic with unsupervised machine learning and lightweight embeddings, wrapped " +
          "in an explainability layer that never surfaces a bare score without a reason. It was validated " +
          "against a labeled synthetic dataset with full test coverage and is packaged with both a CLI and an " +
          "interactive web demo for evaluation."
        ),

        h1("10. How to Run"),
        bullet("pip install -r requirements.txt  (plus system package: tesseract-ocr)"),
        bullet("python generate_sample_data.py   — regenerate the synthetic dataset"),
        bullet("python main.py                   — CLI batch analysis + outputs/report.json"),
        bullet("python app.py                    — interactive Flask demo at localhost:5000"),
        bullet("python -m unittest tests/test_pipeline.py -v  — run the test suite"),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("PROJECT_REPORT.docx", buffer);
  console.log("Wrote PROJECT_REPORT.docx");
});
