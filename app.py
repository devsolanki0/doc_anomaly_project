"""
app.py
-------
Minimal Flask demo UI: upload a document (.txt/.pdf/.png/.jpg), or pick one
of the bundled sample documents, and see the anomaly report rendered in the
browser with risk level, reasons, and extracted fields.

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import glob
import os

from flask import Flask, render_template, request

from src.pipeline import DocumentAnomalyPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "outputs", "uploads")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "sample_documents")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
pipeline = DocumentAnomalyPipeline(os.path.join(BASE_DIR, "data", "historical_invoices.csv"))


def _sample_names():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(SAMPLES_DIR, "*.txt")))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", samples=_sample_names(), report=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("document")
    sample_choice = request.form.get("sample")

    if file and file.filename:
        path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(path)
    elif sample_choice:
        path = os.path.join(SAMPLES_DIR, sample_choice)
    else:
        return render_template("index.html", samples=_sample_names(), report=None,
                                error="Please choose a sample or upload a file.")

    report = pipeline.process_file(path)
    return render_template("index.html", samples=_sample_names(), report=report.to_dict(),
                            analyzed_name=os.path.basename(path))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
