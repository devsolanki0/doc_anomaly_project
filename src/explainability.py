"""
explainability.py
-------------------
Turns raw signals (rule Flags + Isolation Forest score/attribution +
near-duplicate matches) into a single structured, human-readable report per
document: an overall risk level, a ranked list of plain-English reasons, and
the underlying evidence for audit purposes.

This is the "explainable AI" layer: nothing here presents a bare numeric
score to the reviewer without also saying *why*.
"""

from dataclasses import dataclass, field

# Friendlier, non-technical phrasing for the model-driven (as opposed to
# rule-driven) feature attributions.
FEATURE_EXPLANATIONS = {
    "subtotal": "The subtotal amount is unusual compared to typical documents.",
    "tax_ratio": "The ratio of tax to subtotal is unusual.",
    "amount_zscore_vs_vendor": "The total amount deviates sharply from this vendor's historical pattern.",
    "days_invoice_to_due": "The gap between invoice date and due date is unusual.",
    "days_terms_deviation": "The payment window doesn't match this vendor's usual payment terms.",
    "is_weekend_invoice_date": "The invoice is dated on a weekend, which is unusual for this document type.",
    "vendor_seen_before": "This vendor has little or no prior transaction history.",
    "text_length": "The overall length/structure of the document is atypical.",
    "digit_density": "The proportion of numeric content in the document is atypical.",
    "days_since_latest_history": "The invoice date falls far outside the range of historical records on file.",
}

SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 3}
ML_CONTRIBUTION_THRESHOLD = 0.05  # ignore near-zero attributions as noise


@dataclass
class DocumentReport:
    doc_id: str
    fields: dict
    rule_flags: list
    ml_score: float
    ml_percentile: float
    ml_top_features: list
    near_duplicates: list = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        rule_weight = sum(SEVERITY_WEIGHT[f.severity] for f in self.rule_flags)
        ml_weight = 2 if self.ml_percentile >= 95 else (1 if self.ml_percentile >= 85 else 0)
        dup_weight = 2 if self.near_duplicates else 0
        total = rule_weight + ml_weight + dup_weight
        if total == 0:
            return "Low"
        if total <= 2:
            return "Low"
        if total <= 5:
            return "Medium"
        return "High"

    @property
    def reasons(self) -> list:
        reasons = []
        for f in sorted(self.rule_flags, key=lambda x: SEVERITY_WEIGHT[x.severity], reverse=True):
            reasons.append({"source": "rule", "category": f.category, "severity": f.severity, "message": f.message})

        for name, value, contribution in self.ml_top_features:
            if contribution > ML_CONTRIBUTION_THRESHOLD:
                explanation = FEATURE_EXPLANATIONS.get(name, f"Feature '{name}' is unusual.")
                reasons.append({
                    "source": "ml_model",
                    "category": f"pattern_deviation:{name}",
                    "severity": "medium" if contribution < 0.3 else "high",
                    "message": f"{explanation} (observed value: {value:.2f}, "
                               f"contribution to anomaly score: {contribution:.2f})",
                })

        for a, b, score in self.near_duplicates:
            other = b if a == self.doc_id else a
            reasons.append({
                "source": "similarity",
                "category": "near_duplicate",
                "severity": "high",
                "message": f"Content is {score*100:.0f}% similar to another document in this batch "
                           f"('{other}') — possible duplicate submission.",
            })

        if not reasons:
            reasons.append({
                "source": "summary",
                "category": "clean",
                "severity": "low",
                "message": "No inconsistencies detected; document is consistent with historical patterns.",
            })
        return reasons

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "risk_level": self.risk_level,
            "ml_anomaly_score": round(self.ml_score, 4),
            "ml_anomaly_percentile": round(self.ml_percentile, 1),
            "extracted_fields": {
                k: v for k, v in self.fields.items()
                if k in ("vendor_name", "invoice_number", "invoice_date", "due_date",
                         "po_number", "payment_terms", "subtotal", "tax", "total_amount")
            },
            "reasons": self.reasons,
        }

    def summary_line(self) -> str:
        n = len([r for r in self.reasons if r["category"] != "clean"])
        return f"[{self.risk_level.upper():^6}] {self.doc_id} — {n} issue(s) flagged"
