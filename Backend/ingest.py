"""
PDF ingestion: turns a PDF into a list of page-level chunks the extractor can
work on. Kept deliberately dumb -- one chunk per page, plus any tables found
on that page rendered as pipe-separated text so numbers in tables aren't
lost inside merged paragraphs.

Why page-level chunks and not fixed-token windows: financial/report PDFs
usually keep a fact and its context (a table + the sentence introducing it)
on the same page, and it makes the "page number" evidence citation trivial
and exact instead of approximate.
"""

from dataclasses import dataclass
from typing import List

import pymupdf
import pdfplumber


@dataclass
class PageChunk:
    page_number: int  # 1-indexed, matches what a human would cite
    text: str
    tables_text: str  # tables on this page, rendered as simple text


def extract_pages(pdf_path: str) -> List[PageChunk]:
    chunks: List[PageChunk] = []

    # Text via PyMuPDF (fast, good layout handling)
    doc = pymupdf.open(pdf_path)
    page_texts = [page.get_text("text") for page in doc]
    doc.close()

    # Tables via pdfplumber (better structured-table extraction)
    page_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            rendered = []

            for t in tables:
                for row in t:
                    cells = [c if c else "" for c in row]
                    rendered.append(" | ".join(cells))

            page_tables.append("\n".join(rendered))

    n = max(len(page_texts), len(page_tables))

    for i in range(n):
        text = page_texts[i] if i < len(page_texts) else ""
        tables_text = page_tables[i] if i < len(page_tables) else ""

        chunks.append(
            PageChunk(
                page_number=i + 1,
                text=text,
                tables_text=tables_text,
            )
        )

    return chunks


def page_count(pdf_path: str) -> int:
    doc = pymupdf.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n