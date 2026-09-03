"""
pipeline.py
------------
End-to-end orchestrator that wires together every stage of the system:

    raw file
      -> ingestion.extract_text()            (OCR / PDF / text)
      -> field_extraction.extract_fields()   (NLP / regex field parsing)
      -> feature_engineering.build_feature_vector()
      -> rule_checks.run_all_rule_checks()   (explainable rule engine)
      -> anomaly_model.AnomalyModel          (unsupervised ML on features)
      -> similarity.find_near_duplicates()   (cross-document embeddings)
      -> explainability.DocumentReport       (final human-readable report)

`DocumentAnomalyPipeline` is the single class a caller (CLI, Flask app,
notebook, etc.) needs to interact with.
"""

import os
from datetime import date

import numpy as np

from . import ingestion, field_extraction, feature_engineering as fe
from . import rule_checks, similarity
from .anomaly_model import AnomalyModel
from .explainability import DocumentReport


class DocumentAnomalyPipeline:
    def __init__(self, history_csv_path: str, contamination: float = 0.05):
        self.history = fe.load_history(history_csv_path)
        self.profiles = fe.build_vendor_profiles(self.history)
        self.known_invoice_numbers = set(self.history["invoice_number"])
        self.known_vendors = list(self.profiles.keys() - {"_global"})

        X_hist = fe.build_history_feature_matrix(self.history, self.profiles)
        self.model = AnomalyModel(contamination=contamination).fit(X_hist)
        self._reference_scores = np.array([
            self.model.score(row) for row in X_hist
        ])

    # -- single-document processing ------------------------------------
    def process_text(self, doc_id: str, text: str, reference_date: date = None) -> DocumentReport:
        fields = field_extraction.extract_fields(text)
        feature_dict = fe.build_feature_vector(fields, self.profiles, reference_date=reference_date)
        x = fe.features_to_vector(feature_dict)

        flags = rule_checks.run_all_rule_checks(
            fields, feature_dict,
            known_invoice_numbers=self.known_invoice_numbers,
            known_vendors=self.known_vendors,
            reference_date=reference_date,
        )

        ml_score = self.model.score(x)
        ml_percentile = self.model.anomaly_percentile(ml_score, self._reference_scores)
        ml_top_features = self.model.explain(x, top_k=3)

        return DocumentReport(
            doc_id=doc_id,
            fields=fields,
            rule_flags=flags,
            ml_score=ml_score,
            ml_percentile=ml_percentile,
            ml_top_features=ml_top_features,
        )

    def process_file(self, path: str, reference_date: date = None) -> DocumentReport:
        text = ingestion.extract_text(path)
        doc_id = os.path.basename(path)
        return self.process_text(doc_id, text, reference_date=reference_date)

    # -- batch processing (adds cross-document near-duplicate detection) --
    def process_batch(self, paths: list, reference_date: date = None) -> list:
        texts, doc_ids = [], []
        reports = []
        for path in paths:
            report = self.process_file(path, reference_date=reference_date)
            reports.append(report)
            doc_ids.append(report.doc_id)
            texts.append(report.fields.get("raw_text", ""))

        dup_pairs = similarity.find_near_duplicates(doc_ids, texts)
        for report in reports:
            report.near_duplicates = [p for p in dup_pairs if report.doc_id in (p[0], p[1])]

        return reports
