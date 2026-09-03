"""
main.py
--------
CLI entry point.

Usage:
    python main.py
    python main.py --docs data/sample_documents --history data/historical_invoices.csv --out outputs/report.json

Prints a one-line summary per document plus full reasons for anything above
"Low" risk, and writes a full JSON report for every document to --out.
"""

import argparse
import glob
import json
import os

from src.pipeline import DocumentAnomalyPipeline


def main():
    parser = argparse.ArgumentParser(description="Intelligent Document Anomaly Detection")
    parser.add_argument("--docs", default="data/sample_documents", help="Folder of documents to analyze")
    parser.add_argument("--history", default="data/historical_invoices.csv", help="CSV of historical invoices")
    parser.add_argument("--out", default="outputs/report.json", help="Path to write the full JSON report")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.docs, "*")))
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print(f"No documents found in {args.docs}")
        return

    print(f"Loading historical patterns from {args.history} ...")
    pipeline = DocumentAnomalyPipeline(args.history)

    print(f"Analyzing {len(paths)} document(s) ...\n")
    reports = pipeline.process_batch(paths)

    results = []
    for report in sorted(reports, key=lambda r: r.risk_level, reverse=True):
        print(report.summary_line())
        if report.risk_level != "Low":
            for reason in report.reasons:
                print(f"    - ({reason['severity']}) {reason['message']}")
        results.append(report.to_dict())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
