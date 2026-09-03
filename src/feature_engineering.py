"""
feature_engineering.py
-----------------------
Converts structured fields (from field_extraction.py) plus vendor-level
historical statistics (from historical_invoices.csv) into a numeric feature
vector suitable for the unsupervised anomaly model, and computes the
per-vendor baselines used both by the ML model and by explicit rule checks.

Design note: features are built to be *interpretable* (each one maps to a
plain-English idea like "how unusual is this amount for this vendor") so
that explainability.py can turn a raised feature back into a sentence a
non-technical reviewer can understand.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "subtotal",
    "tax_ratio",
    "amount_zscore_vs_vendor",
    "days_invoice_to_due",
    "days_terms_deviation",
    "is_weekend_invoice_date",
    "vendor_seen_before",
    "text_length",
    "digit_density",
    "days_since_latest_history",
]


def load_history(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["invoice_date", "due_date"])
    return df


def build_vendor_profiles(history: pd.DataFrame) -> dict:
    """Per-vendor baseline statistics learned from historical documents.

    These profiles are the "historical pattern" that new documents are
    compared against, both for rule-based checks (e.g. amount outlier) and
    as inputs to the unsupervised model.
    """
    profiles = {}
    for vendor, grp in history.groupby("vendor_name"):
        terms_mode = grp["payment_terms"].mode()
        profiles[vendor] = {
            "amount_mean": grp["total_amount"].mean(),
            "amount_std": max(grp["total_amount"].std(), 1e-6),
            "typical_terms": terms_mode.iloc[0] if not terms_mode.empty else None,
            "typical_net_days": _terms_to_days(terms_mode.iloc[0]) if not terms_mode.empty else None,
            "invoice_count": len(grp),
            "known_invoice_numbers": set(grp["invoice_number"]),
        }
    profiles["_global"] = {
        "amount_mean": history["total_amount"].mean(),
        "amount_std": max(history["total_amount"].std(), 1e-6),
        "latest_date": history["invoice_date"].max(),
    }
    return profiles


def _terms_to_days(terms: str):
    if not terms or not isinstance(terms, str):
        return None
    digits = "".join(ch for ch in terms if ch.isdigit())
    return int(digits) if digits else None


def _safe_date(value):
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.date() if isinstance(value, datetime) else value
    if hasattr(value, "date") and callable(getattr(value, "date")):
        # handles pandas.Timestamp and similar datetime-like objects
        try:
            return value.date()
        except TypeError:
            pass
    try:
        return datetime.fromisoformat(value).date()
    except (ValueError, TypeError):
        return None


def build_feature_vector(fields: dict, profiles: dict, reference_date: date = None) -> dict:
    """Build the interpretable feature dict for one document.

    Missing/unparsable inputs degrade gracefully to neutral values (0) rather
    than raising, since a document with malformed fields is exactly the kind
    of thing we still need to be able to score and flag.
    """
    reference_date = reference_date or date.today()
    vendor = fields.get("vendor_name")
    profile = profiles.get(vendor, profiles["_global"])
    global_profile = profiles["_global"]

    total = fields.get("total_amount") or 0.0
    subtotal = fields.get("subtotal") or 0.0
    tax = fields.get("tax") or 0.0

    amount_mean = profile.get("amount_mean", global_profile["amount_mean"])
    amount_std = profile.get("amount_std", global_profile["amount_std"])
    amount_zscore = (total - amount_mean) / amount_std if amount_std else 0.0

    inv_date = _safe_date(fields.get("invoice_date")) if fields.get("invoice_date_parse_ok") else None
    due_date = _safe_date(fields.get("due_date")) if fields.get("due_date_parse_ok") else None

    days_invoice_to_due = (due_date - inv_date).days if (inv_date and due_date) else 0

    typical_net_days = profile.get("typical_net_days")
    days_terms_deviation = (
        abs(days_invoice_to_due - typical_net_days)
        if (typical_net_days is not None and (inv_date and due_date))
        else 0
    )

    is_weekend = 1.0 if (inv_date and inv_date.weekday() >= 5) else 0.0
    vendor_seen_before = 1.0 if vendor in profiles else 0.0

    raw_text = fields.get("raw_text", "") or ""
    text_length = len(raw_text)
    digit_density = (sum(c.isdigit() for c in raw_text) / text_length) if text_length else 0.0

    latest_hist_date = _safe_date(global_profile.get("latest_date"))
    days_since_latest_history = (
        (inv_date - latest_hist_date).days if (inv_date and latest_hist_date) else 0
    )

    feature_dict = {
        "subtotal": subtotal,
        "tax_ratio": (tax / subtotal) if subtotal else 0.0,
        "amount_zscore_vs_vendor": amount_zscore,
        "days_invoice_to_due": days_invoice_to_due,
        "days_terms_deviation": days_terms_deviation,
        "is_weekend_invoice_date": is_weekend,
        "vendor_seen_before": vendor_seen_before,
        "text_length": text_length,
        "digit_density": digit_density,
        "days_since_latest_history": days_since_latest_history,
    }
    return feature_dict


def features_to_vector(feature_dict: dict) -> np.ndarray:
    return np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=float)


def build_history_feature_matrix(history: pd.DataFrame, profiles: dict) -> np.ndarray:
    """Builds the training matrix for the unsupervised model from the
    historical ("normal") invoices, so the Isolation Forest learns what
    typical documents look like.
    """
    rows = []
    for _, row in history.iterrows():
        fields = {
            "vendor_name": row["vendor_name"],
            "total_amount": row["total_amount"],
            "subtotal": row["subtotal"],
            "tax": row["tax"],
            "invoice_date": row["invoice_date"].date().isoformat(),
            "invoice_date_parse_ok": True,
            "due_date": row["due_date"].date().isoformat(),
            "due_date_parse_ok": True,
            "raw_text": f"{row['vendor_name']} {row['invoice_number']} {row['total_amount']}",
        }
        fd = build_feature_vector(fields, profiles, reference_date=row["invoice_date"].date())
        rows.append(features_to_vector(fd))
    return np.vstack(rows)
