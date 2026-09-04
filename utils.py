import os
import re

import fitz  # PyMuPDF
import nltk
import pytesseract
from PIL import Image
from docx import Document
from nltk.tokenize import word_tokenize


# ---------------------------------
# NLTK SETUP
# ---------------------------------

def setup_nltk():
    resources = [
        ("punkt", "tokenizers/punkt")
    ]

    for package, resource_path in resources:
        try:
            nltk.data.find(resource_path)

        except LookupError:
            nltk.download(package, quiet=True)


setup_nltk()


# ---------------------------------
# EXTRACT TEXT FROM FILE
# ---------------------------------

def extract_text(file_path):
    """
    Extract text from PDF, DOCX, or TXT files.

    OCR is used as a fallback for scanned PDFs.
    """

    if not file_path or not os.path.exists(file_path):
        print("⚠️ File not found.")
        return ""

    text = ""

    file_extension = os.path.splitext(
        file_path
    )[1].lower()

    try:

        # ---------------- PDF ----------------

        if file_extension == ".pdf":

            with fitz.open(file_path) as document:

                # Normal text extraction
                for page in document:

                    page_text = page.get_text()

                    if page_text:
                        text += page_text + " "

                print(
                    "Initial extracted text length:",
                    len(text)
                )

                # OCR fallback for scanned PDFs
                if not text.strip():

                    print(
                        "⚠️ No selectable text found. "
                        "Using OCR fallback..."
                    )

                    for page in document:

                        pixmap = page.get_pixmap(
                            matrix=fitz.Matrix(2, 2)
                        )

                        image_mode = (
                            "RGBA"
                            if pixmap.alpha
                            else "RGB"
                        )

                        image = Image.frombytes(
                            image_mode,
                            [pixmap.width, pixmap.height],
                            pixmap.samples
                        )

                        image = image.convert("L")

                        ocr_text = (
                            pytesseract.image_to_string(
                                image
                            )
                        )

                        text += ocr_text + " "

        # ---------------- DOCX ----------------

        elif file_extension == ".docx":

            document = Document(file_path)

            paragraphs = [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]

            text = " ".join(paragraphs)

        # ---------------- TXT ----------------

        elif file_extension == ".txt":

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:

                text = file.read()

        else:

            print(
                "⚠️ Unsupported file format:",
                file_extension
            )

            return ""

    except Exception as error:

        print(
            "❌ Text extraction error:",
            error
        )

        return ""

    return text.lower().strip()


# ---------------------------------
# PREPROCESS TEXT
# ---------------------------------

def preprocess(text):
    """
    Clean and tokenize extracted text
    before it is used by the ML pipeline.
    """

    if not isinstance(text, str):

        return ""

    text = text.lower().strip()

    if not text:

        return ""

    # Remove special characters
    # while preserving alphabetic words
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Tokenize
    tokens = word_tokenize(text)

    # Remove very short tokens
    tokens = [
        token
        for token in tokens
        if len(token) > 2
    ]

    processed_text = " ".join(tokens)

    return processed_text
