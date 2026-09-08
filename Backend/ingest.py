"""
PDF ingestion: turns a PDF into page-level chunks. One chunk per page, plus
any tables on that page rendered as pipe-separated text so numbers in
tables aren't lost inside merged paragraphs.
"""
from dataclasses import dataclass
from typing import List
import pymupdf
import pdfplumber


@dataclass
class PageChunk:
    page_number: int  # 1-indexed
    text: str
    tables_text: str


def extract_pages(pdf_path: str) -> List[PageChunk]:
    chunks: List[PageChunk] = []

    doc = pymupdf.open(pdf_path)
    page_texts = [page.get_text("text") for page in doc]
    doc.close()

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
        chunks.append(PageChunk(page_number=i + 1, text=text, tables_text=tables_text))

    return chunks


def page_count(pdf_path: str) -> int:
    doc = pymupdf.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n
