"""
ingestion.py
------------
Document ingestion layer: turns a raw file (PDF, image, or plain text) into
raw text, regardless of source format. This is the "OCR" stage of the
pipeline.

Supported inputs:
    .txt            -> read directly (used for our simulated OCR-output samples)
    .pdf            -> text-layer extraction via pdfplumber, falling back to
                       OCR (pytesseract) per-page if a page has no extractable
                       text (i.e. it's a scanned image)
    .png/.jpg/.jpeg/.tiff/.bmp -> image preprocessing + Tesseract OCR

Image preprocessing (grayscale, denoising, adaptive thresholding) is applied
before OCR since real-world scanned business documents are rarely clean, and
OCR accuracy drops sharply on noisy/skewed input.
"""

import os

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Classic OCR preprocessing pipeline: grayscale -> denoise -> threshold.

    Business documents scanned on office equipment often have uneven
    lighting and light noise; adaptive thresholding handles both far better
    than a single global threshold.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=15,
    )
    return thresh


def ocr_image_file(path: str) -> str:
    """Run OCR on a single image file and return extracted text."""
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed; cannot OCR images.")
    image = cv2.imread(path)
    if image is None:
        # Fall back to PIL for formats OpenCV struggles with
        pil_img = Image.open(path).convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    processed = _preprocess_for_ocr(image)
    text = pytesseract.image_to_string(processed)
    return text


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF, using the text layer where present and
    falling back to OCR (rendering the page to an image) for scanned pages.
    """
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed; cannot read PDFs.")

    all_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                all_text.append(page_text)
            elif pytesseract is not None:
                # Scanned page with no text layer -> rasterize and OCR it
                pil_img = page.to_image(resolution=300).original
                image = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
                processed = _preprocess_for_ocr(image)
                all_text.append(pytesseract.image_to_string(processed))
    return "\n".join(all_text)


def extract_text(path: str) -> str:
    """Unified entry point: dispatches to the right extractor by file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in IMAGE_EXTENSIONS:
        return ocr_image_file(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
