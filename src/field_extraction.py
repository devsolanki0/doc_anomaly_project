"""
field_extraction.py
--------------------
Lightweight information-extraction ("NLP") layer that turns raw OCR/PDF text
into a structured dict of fields. Uses labeled-field regexes (robust to the
kind of key-value formatting found on most invoices/contracts/applications)
plus dateutil for flexible date parsing.

This is intentionally dependency-light (no spaCy/transformers needed) so it
runs anywhere, but the extraction contract (`extract_fields(text) -> dict`)
is designed so it can be swapped for a spaCy NER pipeline or an LLM-based
extractor later without touching the rest of the system.
"""

import re
from dateutil import parser as dateparser

# Each field maps to a list of candidate regex patterns (first match wins).
# Patterns are deliberately tolerant of spacing/punctuation variance seen in
# real OCR output.
FIELD_PATTERNS = {
    "invoice_number": [
        r"invoice\s*(?:number|no\.?|#)[ \t]*[:\-]?[ \t]*([A-Za-z0-9\-\/]+)",
    ],
    "po_number": [
        r"(?:po|p\.o\.)\s*(?:number|no\.?|#)?[ \t]*[:\-]?[ \t]*([A-Za-z0-9\-\/]+)",
    ],
    "invoice_date": [
        r"invoice\s*date\s*[:\-]?\s*([A-Za-z0-9,\/\-\s]+?)(?:\n|$)",
    ],
    "due_date": [
        r"due\s*date\s*[:\-]?\s*([A-Za-z0-9,\/\-\s]+?)(?:\n|$)",
    ],
    "payment_terms": [
        r"payment\s*terms\s*[:\-]?\s*([A-Za-z0-9\s]+?)(?:\n|$)",
    ],
    "subtotal": [
        r"sub\s*-?\s*total\s*[:\-]?\s*\$?\s*([\d,]+\.\d{2})",
    ],
    "tax": [
        r"tax(?:\s*\(\d+%\))?\s*[:\-]?\s*\$?\s*([\d,]+\.\d{2})",
    ],
    "total_amount": [
        r"total\s*amount\s*due\s*[:\-]?\s*\$?\s*([\d,]+\.\d{2})",
        r"(?<!sub)(?<!sub-)\btotal\b(?!\s*\(\d)\s*(?:amount)?\s*(?:due)?\s*[:\-]?\s*\$?\s*([\d,]+\.\d{2})",
    ],
}

VENDOR_NAME_LINE = 0  # first non-empty line of the document is treated as vendor/letterhead


def _clean_money(raw: str):
    try:
        return round(float(raw.replace(",", "").strip()), 2)
    except (ValueError, AttributeError):
        return None


def _clean_date(raw: str):
    """Parse a loosely-formatted date string. Returns (iso_string, parse_ok)."""
    if not raw:
        return None, False
    raw = raw.strip()
    try:
        dt = dateparser.parse(raw, fuzzy=True)
        if dt is None:
            return raw, False
        return dt.date().isoformat(), True
    except (ValueError, OverflowError):
        return raw, False


def extract_vendor_name(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[VENDOR_NAME_LINE] if lines else ""


def extract_fields(text: str) -> dict:
    """Extract structured fields from raw document text.

    Returns a dict containing both the parsed field values and metadata
    flags (e.g. `invoice_date_parse_ok`) that downstream rule checks use to
    detect malformed/OCR-garbled fields.
    """
    fields = {"raw_text": text, "vendor_name": extract_vendor_name(text)}

    for field, patterns in FIELD_PATTERNS.items():
        value = None
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                break
        fields[f"{field}_raw"] = value

    # Post-process dates
    for date_field in ("invoice_date", "due_date"):
        raw = fields.get(f"{date_field}_raw")
        parsed, ok = _clean_date(raw)
        fields[date_field] = parsed
        fields[f"{date_field}_parse_ok"] = ok

    # Post-process money fields
    for money_field in ("subtotal", "tax", "total_amount"):
        fields[money_field] = _clean_money(fields.get(f"{money_field}_raw"))

    fields["invoice_number"] = fields.get("invoice_number_raw") or None
    fields["po_number"] = fields.get("po_number_raw") or None
    fields["payment_terms"] = fields.get("payment_terms_raw") or None

    # Detect free-text statements that often indicate payment/contract status
    # (used later for conflicting-statement checks).
    lower = text.lower()
    fields["mentions_paid_in_full"] = "paid in full" in lower
    fields["mentions_balance_due"] = bool(re.search(r"balance\s*due", lower))
    fields["mentions_draft"] = "draft" in lower
    fields["mentions_final"] = "final" in lower

    return fields
