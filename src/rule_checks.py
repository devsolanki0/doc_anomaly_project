"""
rule_checks.py
---------------
Deterministic, explainable rule checks. These catch the anomaly types that
are best expressed as explicit logic rather than learned statistically:
mismatched dates, arithmetic errors, duplicate IDs, missing fields,
conflicting statements, and look-alike vendor names.

Each check returns zero or more Flag objects. A Flag is intentionally a
plain dataclass (not a free-text string) so explainability.py and any UI
can consistently render severity, category, and message.
"""

import difflib
from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Flag:
    category: str          # e.g. "date_logic", "arithmetic", "duplicate"
    severity: str           # "low" | "medium" | "high"
    message: str            # human-readable explanation
    evidence: dict = field(default_factory=dict)


REQUIRED_FIELDS = ["invoice_number", "invoice_date", "total_amount", "vendor_name"]
AMOUNT_TOLERANCE = 0.02       # allowed rounding error for arithmetic checks
VENDOR_SIMILARITY_LOW = 0.80  # below this, names are considered unrelated
VENDOR_SIMILARITY_HIGH = 0.98  # at/above this, treated as the same name


def _parse_date(value, ok_flag):
    if not ok_flag or not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except (ValueError, TypeError):
        return None


def check_missing_fields(fields: dict) -> list:
    flags = []
    missing = [f for f in REQUIRED_FIELDS if not fields.get(f)]
    if missing:
        flags.append(Flag(
            category="missing_fields",
            severity="high" if len(missing) > 1 else "medium",
            message=f"Missing required field(s): {', '.join(missing)}.",
            evidence={"missing_fields": missing},
        ))
    return flags


def check_date_logic(fields: dict, reference_date: date = None) -> list:
    flags = []
    reference_date = reference_date or date.today()

    if fields.get("invoice_date") and not fields.get("invoice_date_parse_ok"):
        flags.append(Flag(
            category="date_format",
            severity="medium",
            message=f"Invoice date field ('{fields.get('invoice_date_raw')}') could not be parsed "
                     f"as a valid date — likely an OCR error or malformed input.",
            evidence={"raw_value": fields.get("invoice_date_raw")},
        ))
    if fields.get("due_date") and not fields.get("due_date_parse_ok"):
        flags.append(Flag(
            category="date_format",
            severity="low",
            message=f"Due date field ('{fields.get('due_date_raw')}') could not be parsed as a valid date.",
            evidence={"raw_value": fields.get("due_date_raw")},
        ))

    inv_date = _parse_date(fields.get("invoice_date"), fields.get("invoice_date_parse_ok"))
    due_date = _parse_date(fields.get("due_date"), fields.get("due_date_parse_ok"))

    if inv_date and due_date and due_date < inv_date:
        flags.append(Flag(
            category="date_logic",
            severity="high",
            message=f"Due date ({due_date.isoformat()}) is earlier than the invoice date "
                     f"({inv_date.isoformat()}) — dates are logically inconsistent.",
            evidence={"invoice_date": inv_date.isoformat(), "due_date": due_date.isoformat()},
        ))

    if inv_date and inv_date > reference_date:
        flags.append(Flag(
            category="date_logic",
            severity="high",
            message=f"Invoice is dated {inv_date.isoformat()}, which is in the future "
                     f"relative to today ({reference_date.isoformat()}).",
            evidence={"invoice_date": inv_date.isoformat(), "reference_date": reference_date.isoformat()},
        ))

    return flags


def check_arithmetic(fields: dict) -> list:
    flags = []
    subtotal = fields.get("subtotal")
    tax = fields.get("tax")
    total = fields.get("total_amount")
    if subtotal is not None and tax is not None and total is not None:
        expected = round(subtotal + tax, 2)
        if abs(expected - total) > AMOUNT_TOLERANCE:
            flags.append(Flag(
                category="arithmetic",
                severity="high",
                message=f"Printed total (${total:,.2f}) does not equal subtotal + tax "
                         f"(${subtotal:,.2f} + ${tax:,.2f} = ${expected:,.2f}); "
                         f"difference of ${abs(expected - total):,.2f}.",
                evidence={"subtotal": subtotal, "tax": tax, "total_amount": total, "expected_total": expected},
            ))
    if subtotal and round(subtotal, 0) == subtotal and subtotal >= 1000 and \
            (fields.get("total_amount") and round(fields["total_amount"], 0) == fields["total_amount"]):
        flags.append(Flag(
            category="round_numbers",
            severity="low",
            message=f"Subtotal (${subtotal:,.2f}) and total are suspiciously round numbers; "
                     f"genuine itemized invoices rarely land on exact round figures.",
            evidence={"subtotal": subtotal},
        ))
    return flags


def check_amount_outlier(fields: dict, feature_dict: dict, zscore_threshold: float = 3.0) -> list:
    flags = []
    z = feature_dict.get("amount_zscore_vs_vendor", 0.0)
    if abs(z) >= zscore_threshold:
        direction = "higher" if z > 0 else "lower"
        flags.append(Flag(
            category="amount_outlier",
            severity="high" if abs(z) >= 5 else "medium",
            message=f"Total amount is {abs(z):.1f} standard deviations {direction} than this "
                     f"vendor's historical average — a significant deviation from past invoices.",
            evidence={"zscore": z},
        ))
    return flags


def check_duplicate_invoice_number(fields: dict, known_invoice_numbers: set) -> list:
    flags = []
    inv_no = fields.get("invoice_number")
    if inv_no and inv_no in known_invoice_numbers:
        flags.append(Flag(
            category="duplicate",
            severity="high",
            message=f"Invoice number '{inv_no}' already exists in historical records — "
                     f"possible duplicate submission or resubmitted invoice.",
            evidence={"invoice_number": inv_no},
        ))
    return flags


def check_vendor_lookalike(fields: dict, known_vendors: list) -> list:
    """Flags vendor names that are suspiciously close to (but not exactly)
    a known vendor — a classic vendor-impersonation / fraud pattern.
    """
    flags = []
    name = fields.get("vendor_name")
    if not name or not known_vendors:
        return flags
    if name in known_vendors:
        return flags  # exact match against known vendor is fine

    best_match, best_score = None, 0.0
    for known in known_vendors:
        score = difflib.SequenceMatcher(None, name.lower(), known.lower()).ratio()
        if score > best_score:
            best_match, best_score = known, score

    if VENDOR_SIMILARITY_LOW <= best_score < VENDOR_SIMILARITY_HIGH:
        flags.append(Flag(
            category="vendor_lookalike",
            severity="high",
            message=f"Vendor name '{name}' is {best_score*100:.0f}% similar to known vendor "
                     f"'{best_match}' but does not match exactly — possible vendor impersonation "
                     f"or a typo introduced during data entry/OCR.",
            evidence={"vendor_name": name, "closest_known_vendor": best_match, "similarity": best_score},
        ))
    return flags


def check_conflicting_statements(fields: dict) -> list:
    flags = []
    total = fields.get("total_amount")
    if fields.get("mentions_paid_in_full") and fields.get("mentions_balance_due") and total and total > 0:
        flags.append(Flag(
            category="conflicting_statement",
            severity="high",
            message=f"Document states the invoice is 'PAID IN FULL' but also lists a nonzero "
                     f"balance due (${total:,.2f}) — these statements directly contradict each other.",
            evidence={"total_amount": total},
        ))
    if fields.get("mentions_draft") and fields.get("mentions_final"):
        flags.append(Flag(
            category="conflicting_statement",
            severity="medium",
            message="Document contains both 'draft' and 'final' language, which is contradictory "
                     "for a single document version.",
            evidence={},
        ))
    return flags


def run_all_rule_checks(fields: dict, feature_dict: dict, known_invoice_numbers: set,
                         known_vendors: list, reference_date: date = None) -> list:
    flags = []
    flags += check_missing_fields(fields)
    flags += check_date_logic(fields, reference_date=reference_date)
    flags += check_arithmetic(fields)
    flags += check_amount_outlier(fields, feature_dict)
    flags += check_duplicate_invoice_number(fields, known_invoice_numbers)
    flags += check_vendor_lookalike(fields, known_vendors)
    flags += check_conflicting_statements(fields)
    return flags
