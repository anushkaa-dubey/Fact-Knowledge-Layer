"""
Post-extraction noise filter.

Removes document-navigation metadata such as table-of-contents
entries and page references while keeping genuine numerical facts.
"""

import re
from typing import Optional


PAGE_REFERENCE_KEYWORDS = (
    "page",
    "pg.",
    "starting page",
    "ending page",
    "section number",
    "chapter number",
    "table of contents",
    "index",
)


def is_probable_toc_entry(
    fact: dict,
    doc_page_count: Optional[int] = None
) -> bool:

    metric = (fact.get("metric") or "").lower().strip()
    value = str(
        fact.get("value") or ""
    ).replace(",", "").strip()

    quote = (fact.get("quote") or "").strip()

    # A TOC page reference is normally just a small integer.
    value_is_small_int = bool(
        re.fullmatch(r"\d{1,4}", value)
    )

    if not value_is_small_int:
        return False

    # ---------------------------------------------------------
    # Rule 1: Explicit page/navigation metrics
    # ---------------------------------------------------------

    mentions_page = any(
        keyword in metric
        for keyword in PAGE_REFERENCE_KEYWORDS
    )

    if mentions_page:
        return True

    # ---------------------------------------------------------
    # Rule 2: Very strict TOC pattern
    #
    # Examples:
    #   "Management Discussion and Analysis .... 520"
    #   "Capitalisation Statement | 519"
    #
    # Do NOT treat normal sentences ending in a number as TOC.
    # ---------------------------------------------------------

    toc_shape = re.fullmatch(
        r"[A-Za-z][A-Za-z0-9 &,\-'/()]{2,80}"
        r"(?:\.{2,}|\s{2,}|\|)\s*"
        r"\d{1,4}",
        quote
    )

    if toc_shape:
        return True

    return False