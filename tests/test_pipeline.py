"""
Basic unit/integration tests for the anomaly detection pipeline.

Run:
    python -m pytest tests/ -v
or, without pytest:
    python tests/test_pipeline.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import DocumentAnomalyPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_CSV = os.path.join(BASE_DIR, "data", "historical_invoices.csv")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "sample_documents")


class TestDocumentAnomalyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = DocumentAnomalyPipeline(HISTORY_CSV)

    def _categories(self, report):
        return {r["category"] for r in report.reasons}

    def test_normal_document_is_low_risk(self):
        report = self.pipeline.process_file(os.path.join(SAMPLES_DIR, "doc_02_normal.txt"))
        self.assertEqual(report.risk_level, "Low")

    def test_arithmetic_mismatch_detected(self):
        report = self.pipeline.process_file(os.path.join(SAMPLES_DIR, "doc_07_math_mismatch.txt"))
        self.assertIn("arithmetic", self._categories(report))
        self.assertNotEqual(report.risk_level, "Low")

    def test_due_before_invoice_date_detected(self):
        report = self.pipeline.process_file(os.path.join(SAMPLES_DIR, "doc_08_due_before_invoice.txt"))
        self.assertIn("date_logic", self._categories(report))

    def test_future_dated_invoice_detected(self):
        report = self.pipeline.process_file(os.path.join(SAMPLES_DIR, "doc_09_future_dated.txt"))
        self.assertIn("date_logic", self._categories(report))

    def test_duplicate_invoice_number_detected(self):
        report = self.pipeline.process_file(os.path.join(SAMPLES_DIR, "doc_10_duplicate_invoice_number.txt"))
        self.assertIn("duplicate", self._categories(report))
        self.assertEqual(report.risk_level, "High")

    def test_vendor_lookalike_detected(self):
        report = self.pipeline.process_file(os.path.join(SAMPLES_DIR, "doc_11_vendor_lookalike.txt"))
        self.assertIn("vendor_lookalike", self._categories(report))

    def test_amount_outlier_detected(self):
        report = self.pipeline.process_file(os.path.join(SAMPLES_DIR, "doc_12_amount_outlier.txt"))
        self.assertIn("amount_outlier", self._categories(report))
        self.assertEqual(report.risk_level, "High")

    def test_missing_fields_detected(self):
        report = self.pipeline.process_file(os.path.join(SAMPLES_DIR, "doc_13_missing_fields.txt"))
        self.assertIn("missing_fields", self._categories(report))

    def test_conflicting_statement_detected(self):
        report = self.pipeline.process_file(os.path.join(SAMPLES_DIR, "doc_14_conflicting_statement.txt"))
        self.assertIn("conflicting_statement", self._categories(report))

    def test_unparsable_date_detected(self):
        report = self.pipeline.process_file(os.path.join(SAMPLES_DIR, "doc_18_unparsable_date.txt"))
        self.assertIn("date_format", self._categories(report))

    def test_near_duplicate_detected_in_batch(self):
        paths = [
            os.path.join(SAMPLES_DIR, "doc_16_original_for_duplicate.txt"),
            os.path.join(SAMPLES_DIR, "doc_17_near_duplicate.txt"),
        ]
        reports = self.pipeline.process_batch(paths)
        for report in reports:
            self.assertTrue(len(report.near_duplicates) >= 1)
            self.assertIn("near_duplicate", self._categories(report))


if __name__ == "__main__":
    unittest.main()
