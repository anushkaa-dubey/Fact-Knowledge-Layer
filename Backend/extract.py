"""
Fact extraction: turns one page of raw PDF text into a list of atomic,
evidence-grounded facts using Groq.
"""

import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
from .ingest import PageChunk
from . import db


MODEL = os.environ.get("FACTLAYER_MODEL", "llama-3.3-70b-versatile")

_client: Optional[Groq] = None


def get_client() -> Groq:
    global _client

    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Set it before running the server."
            )

        _client = Groq(api_key=api_key)

    return _client


EXTRACTION_SYSTEM_PROMPT = """You extract atomic, checkable facts from a single page of a document
(company filing, financial report, government report, or similar).

A "fact" is a specific, checkable claim -- typically a number, a status, or a stated relationship
(e.g. a metric with a value, a person's role, an event with a date). Skip vague or purely narrative
sentences ("the company performed well") that carry no checkable content.

For every fact you find, report:
- subject: the entity the fact is about (a company, a country, a person, a division -- whatever fits)
- metric: what is being measured or claimed, in your own concise words
- value: the number or stated value (as a string, keep original formatting e.g. "7,831")
- unit: the unit if any (e.g. "INR crore", "%", "USD million"). Use null if not applicable.
- period: the time period or as-of date this fact applies to, if stated
- scope: any qualifying scope. Use null if none.
- quote: the exact sentence or table row that supports this fact.
- confidence: your confidence (0.0-1.0) that you extracted this correctly.

Only extract facts that are explicitly stated or directly readable from a table on this page.
Do not infer, calculate, or bring in outside knowledge.

Respond ONLY with a JSON array of fact objects.
No preamble, no markdown fences.

If there are no extractable facts on this page, respond with an empty array: []
"""


def extract_facts_from_page(
    chunk: PageChunk,
    doc_id: int
) -> List[dict]:

    combined_text = chunk.text

    if chunk.tables_text.strip():
        combined_text += (
            "\n\n[TABLES ON THIS PAGE]\n"
            + chunk.tables_text
        )

    if not combined_text.strip():
        return []

    client = get_client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4000,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": EXTRACTION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": combined_text[:12000]
                }
            ],
        )

        raw = response.choices[0].message.content.strip()

        raw = _strip_code_fences(raw)

        data = json.loads(raw)

        # Groq JSON mode returns valid JSON.
        # Accept either a direct list or {"facts": [...]}
        if isinstance(data, list):
            facts = data
        elif isinstance(data, dict) and "facts" in data:
            facts = data["facts"]
        else:
            raise ValueError("Expected a JSON array or a facts object")

    except Exception as e:

        db.insert_failure(
            doc_id=doc_id,
            stage="extraction",
            detail=(
                f"page {chunk.page_number}: "
                f"{type(e).__name__}: {e}"
            ),
            raw_snippet=combined_text[:500],
        )

        return []

    for fact in facts:
        fact["page"] = chunk.page_number

    return facts


def _strip_code_fences(text: str) -> str:

    text = text.strip()

    if text.startswith("```"):

        lines = text.split("\n")
        lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()