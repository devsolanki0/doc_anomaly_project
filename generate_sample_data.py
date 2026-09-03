"""
generate_sample_data.py
------------------------
Creates a synthetic dataset for the Document Anomaly Detection project:

1. data/historical_invoices.csv
   A "clean" history of ~250 past invoices across 10 vendors, used to learn
   normal patterns (typical amounts, payment terms, frequency, etc.)

2. data/sample_documents/*.txt
   ~18 "new" invoices formatted as plain text (this stands in for OCR/PDF
   text-extraction output). Most are normal. Several have deliberately
   injected anomalies so the pipeline has something real to catch:
     - arithmetic mismatch (subtotal + tax != total)
     - due date before invoice date
     - future-dated invoice
     - duplicate invoice number (reused from history)
     - vendor name look-alike / spoof ("Acme Supplies Co." vs "Acme Suppiles Co.")
     - amount wildly outside a vendor's historical range
     - missing required fields (no PO number / no invoice number)
     - conflicting statements ("PAID IN FULL" + a nonzero balance due)
     - near-duplicate of another document in the same batch

Run:
    python generate_sample_data.py
"""

import os
import csv
import random
from datetime import date, timedelta

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "sample_documents")
os.makedirs(DOCS_DIR, exist_ok=True)

VENDORS = [
    ("Acme Supplies Co.", "NET30", 1500, 300),
    ("Bluepeak Logistics", "NET15", 4200, 900),
    ("Crestline Manufacturing", "NET45", 12500, 2500),
    ("Delta Office Solutions", "NET30", 800, 150),
    ("Everwood Consulting", "NET15", 6000, 1200),
    ("Falcon IT Services", "NET30", 3000, 600),
    ("Granite Facilities Mgmt", "NET30", 2200, 400),
    ("Harbor Freight Partners", "NET60", 9000, 1800),
    ("Ironclad Security Inc.", "NET30", 2600, 500),
    ("Juniper Marketing Group", "NET15", 1800, 350),
]

TAX_RATE = 0.08


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def money(x: float) -> str:
    return f"{x:,.2f}"


# ---------------------------------------------------------------------------
# 1. Historical invoices (normal data used to learn what "typical" looks like)
# ---------------------------------------------------------------------------
history_rows = []
inv_counter = 10000
hist_start = date(2024, 1, 1)
hist_end = date(2025, 12, 31)

for _ in range(250):
    vendor, terms, mean_amt, std_amt = random.choice(VENDORS)
    inv_counter += 1
    invoice_no = f"INV-{inv_counter}"
    invoice_dt = random_date(hist_start, hist_end)
    net_days = int(terms.replace("NET", ""))
    due_dt = invoice_dt + timedelta(days=net_days)
    subtotal = max(50, random.gauss(mean_amt, std_amt))
    tax = subtotal * TAX_RATE
    total = subtotal + tax
    po_number = f"PO-{random.randint(20000, 29999)}"

    history_rows.append({
        "invoice_number": invoice_no,
        "vendor_name": vendor,
        "invoice_date": invoice_dt.isoformat(),
        "due_date": due_dt.isoformat(),
        "payment_terms": terms,
        "po_number": po_number,
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "total_amount": round(total, 2),
    })

history_rows.sort(key=lambda r: r["invoice_date"])
hist_path = os.path.join(DATA_DIR, "historical_invoices.csv")
with open(hist_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
    writer.writeheader()
    writer.writerows(history_rows)

print(f"Wrote {len(history_rows)} historical invoices -> {hist_path}")


# ---------------------------------------------------------------------------
# 2. Helper to render an invoice dict as plain text (simulated OCR output)
# ---------------------------------------------------------------------------
def render_invoice_text(fields: dict, extra_lines=None, statement=None) -> str:
    lines = [
        f"{fields.get('vendor_name', '')}",
        "123 Commerce Way, Springfield",
        "-----------------------------------------",
        "INVOICE",
        f"Invoice Number: {fields.get('invoice_number', '')}",
        f"Invoice Date: {fields.get('invoice_date', '')}",
        f"Due Date: {fields.get('due_date', '')}",
        f"PO Number: {fields.get('po_number', '')}",
        f"Payment Terms: {fields.get('payment_terms', '')}",
        "-----------------------------------------",
        "Description                Qty     Amount",
        "Professional Services        1     " + money(fields.get("subtotal", 0)),
        "-----------------------------------------",
        f"Subtotal: ${money(fields.get('subtotal', 0))}",
        f"Tax (8%): ${money(fields.get('tax', 0))}",
        f"Total Amount Due: ${money(fields.get('total_amount', 0))}",
    ]
    if statement:
        lines.append(statement)
    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines)


NEW_START = date(2026, 6, 1)
NEW_END = date(2026, 8, 28)
sample_index = []  # (filename, description of injected anomaly or "normal")


def new_normal_invoice(n):
    vendor, terms, mean_amt, std_amt = random.choice(VENDORS)
    invoice_no = f"INV-{20000 + n}"
    invoice_dt = random_date(NEW_START, NEW_END)
    net_days = int(terms.replace("NET", ""))
    due_dt = invoice_dt + timedelta(days=net_days)
    subtotal = max(50, random.gauss(mean_amt, std_amt))
    tax = round(subtotal * TAX_RATE, 2)
    total = round(subtotal + tax, 2)
    fields = {
        "vendor_name": vendor,
        "invoice_number": invoice_no,
        "invoice_date": invoice_dt.isoformat(),
        "due_date": due_dt.isoformat(),
        "po_number": f"PO-{random.randint(30000, 39999)}",
        "payment_terms": terms,
        "subtotal": round(subtotal, 2),
        "tax": tax,
        "total_amount": total,
    }
    return fields


docs = []

# --- 6 clean, normal invoices -------------------------------------------------
for i in range(6):
    f = new_normal_invoice(i)
    docs.append((f"doc_{i+1:02d}_normal.txt", render_invoice_text(f), "normal"))

# --- Arithmetic mismatch: subtotal + tax != total -----------------------------
f = new_normal_invoice(100)
f["total_amount"] = round(f["subtotal"] + f["tax"] + 250.00, 2)  # inflated total
docs.append(("doc_07_math_mismatch.txt", render_invoice_text(f),
             "Arithmetic mismatch: printed total does not equal subtotal + tax"))

# --- Due date before invoice date --------------------------------------------
f = new_normal_invoice(101)
f["due_date"] = (date.fromisoformat(f["invoice_date"]) - timedelta(days=10)).isoformat()
docs.append(("doc_08_due_before_invoice.txt", render_invoice_text(f),
             "Due date is earlier than the invoice date"))

# --- Future-dated invoice ------------------------------------------------------
f = new_normal_invoice(102)
f["invoice_date"] = (NEW_END + timedelta(days=60)).isoformat()
f["due_date"] = (date.fromisoformat(f["invoice_date"]) + timedelta(days=30)).isoformat()
docs.append(("doc_09_future_dated.txt", render_invoice_text(f),
             "Invoice is dated in the future relative to processing date"))

# --- Duplicate invoice number (reused from historical records) ---------------
dup_hist = random.choice(history_rows)
f = new_normal_invoice(103)
f["invoice_number"] = dup_hist["invoice_number"]
f["vendor_name"] = dup_hist["vendor_name"]
docs.append(("doc_10_duplicate_invoice_number.txt", render_invoice_text(f),
             "Invoice number already exists in historical records (possible duplicate/resubmission)"))

# --- Vendor name look-alike / spoof -------------------------------------------
f = new_normal_invoice(104)
f["vendor_name"] = "Acme Suppiles Co."  # subtle misspelling of "Acme Supplies Co."
docs.append(("doc_11_vendor_lookalike.txt", render_invoice_text(f),
             "Vendor name closely resembles a known vendor but does not match exactly (possible impersonation)"))

# --- Amount far outside vendor's historical range -----------------------------
vendor, terms, mean_amt, std_amt = VENDORS[3]  # Delta Office Solutions, usually small invoices
f = new_normal_invoice(105)
f["vendor_name"] = vendor
f["payment_terms"] = terms
f["subtotal"] = round(mean_amt + std_amt * 12, 2)  # extreme outlier
f["tax"] = round(f["subtotal"] * TAX_RATE, 2)
f["total_amount"] = round(f["subtotal"] + f["tax"], 2)
docs.append(("doc_12_amount_outlier.txt", render_invoice_text(f),
             "Invoice amount is far outside this vendor's normal historical range"))

# --- Missing required fields ---------------------------------------------------
f = new_normal_invoice(106)
f["po_number"] = ""
f["invoice_number"] = ""
docs.append(("doc_13_missing_fields.txt", render_invoice_text(f),
             "Missing required fields: invoice number and PO number"))

# --- Conflicting statements: "PAID IN FULL" + nonzero balance due -------------
f = new_normal_invoice(107)
docs.append(("doc_14_conflicting_statement.txt",
             render_invoice_text(f, statement=f"Status: PAID IN FULL. Remaining Balance Due: ${money(f['total_amount'])}"),
             "Conflicting statements: marked 'PAID IN FULL' but a nonzero balance is also listed"))

# --- Weekend/holiday-dated + suspiciously round total (secondary heuristic) ---
f = new_normal_invoice(108)
f["subtotal"] = 5000.00
f["tax"] = 400.00
f["total_amount"] = 5400.00
docs.append(("doc_15_round_number.txt", render_invoice_text(f),
             "Suspiciously round subtotal/total amounts (common in fabricated invoices)"))

# --- Near-duplicate document (content copied from doc_01 with tiny edits) ----
base_fields = new_normal_invoice(109)
docs.append(("doc_16_original_for_duplicate.txt", render_invoice_text(base_fields), "normal"))
near_dup_fields = dict(base_fields)
near_dup_fields["invoice_number"] = base_fields["invoice_number"] + "-A"  # only ID changed
docs.append(("doc_17_near_duplicate.txt", render_invoice_text(near_dup_fields),
             "Near-duplicate of another document in this batch (nearly identical content, different ID)"))

# --- Currency/date field that fails to parse (garbled OCR-like text) ---------
f = new_normal_invoice(110)
text = render_invoice_text(f)
text = text.replace(f["invoice_date"], "3O/13/2O26")  # OCR confusion: letter O for digit 0
docs.append(("doc_18_unparsable_date.txt", text,
             "Date field is malformed / unparsable (likely OCR error), preventing date-logic checks"))

for fname, content, note in docs:
    with open(os.path.join(DOCS_DIR, fname), "w") as fh:
        fh.write(content)
    sample_index.append((fname, note))

index_path = os.path.join(DATA_DIR, "sample_documents_index.csv")
with open(index_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "ground_truth_note"])
    writer.writerows(sample_index)

print(f"Wrote {len(docs)} sample documents -> {DOCS_DIR}")
print(f"Wrote ground-truth index -> {index_path}")
