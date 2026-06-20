import io
import re

import fitz  # PyMuPDF
import PyPDF2
import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter


import sys

if sys.platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


MIN_TEXT_THRESHOLD = 50


def clean_text(text):
    """Remove common watermark/boilerplate noise picked up from scraped PDFs
    (e.g. Studocu downloads), and collapse excess blank lines."""
    noise_patterns = [
        r"messages\.\w+",       
        r"lOMoARcPSD\|?\d*",    
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text)

    text = re.sub(r"\n\s*\n+", "\n", text)  
    return text.strip()


def render_page_to_image(fitz_page, dpi=200):
    """Render a single PyMuPDF page object to a PIL Image for OCR."""
    pix = fitz_page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def extract_text_from_pdf(pdf_path):
    """
    Per-page extraction that captures text from BOTH sources when needed:

      1. Native text via PyPDF2 (fast, always attempted).
      2. OCR via Tesseract — triggered whenever the page contains ANY
         embedded image (detected via PyMuPDF's page.get_images()),
         regardless of how much native text is also present.

    Why: a page can have a full paragraph of real text AND a diagram with
    labels baked into the image (e.g. "item_key", "branch_id"). A purely
    threshold-based check would skip OCR on that page because the native
    text alone clears the bar — silently losing everything inside the
    diagram. Checking for embedded images instead catches this case.

    As a safety net, pages with NO images but suspiciously little native
    text (e.g. blank/corrupted pages) still fall back to OCR via the
    MIN_TEXT_THRESHOLD check.
    """
    text = ""
    reader = PyPDF2.PdfReader(pdf_path)
    doc = fitz.open(pdf_path)

    for i, page in enumerate(reader.pages):
        native_text = page.extract_text() or ""
        fitz_page = doc[i]
        has_images = len(fitz_page.get_images()) > 0

        needs_ocr = has_images or (len(native_text.strip()) < MIN_TEXT_THRESHOLD)

        if needs_ocr:
            reason = "contains embedded image(s)" if has_images else "low native text"
            print(f"[INFO] Page {i + 1}: {reason} — running OCR...")
            page_image = render_page_to_image(fitz_page)
            ocr_text = pytesseract.image_to_string(page_image)

            # Combine both sources so we don't lose paragraph text OR
            # diagram text — they often aren't redundant with each other.
            combined = native_text.strip() + "\n" + ocr_text.strip()
            text += combined + "\n"
        else:
            print(f"[INFO] Page {i + 1}: {len(native_text.strip())} chars native text, no images — skipping OCR.")
            text += native_text + "\n"

    doc.close()
    return text


def extract_text_from_pdfs(pdf_paths):
    """Process multiple PDFs, clean each, and combine into one text blob."""
    all_text = ""
    for path in pdf_paths:
        raw_text = extract_text_from_pdf(path)
        all_text += clean_text(raw_text) + "\n"
    return all_text


def chunk_text(text):
    """Split combined text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)