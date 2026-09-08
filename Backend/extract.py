"""
Fact extraction: turns one page of raw PDF text into a list of atomic,
evidence-grounded facts, using Groq (OpenAI-compatible chat.completions API).

Design choice: no fixed schema. We don't assume the document is about
"revenue" or "GDP" -- the model reports each fact's own
subject/metric/period/scope, because the same code has to work on a
company prospectus AND a macroeconomic report without being told which.
This is what makes the system generalize to unseen PDFs.
"""
import json
import os
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq

from .ingest import PageChunk
from . import db

load_dotenv()

MODEL = os.environ.get("FACTLAYER_MODEL", "openai/gpt-oss-120b")

_client: Optional[Groq] = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Put it in a .env file or export it: "
                "export GROQ_API_KEY=gsk_..."
            )
        _client = Groq(api_key=api_key)
    return _client


EXTRACTION_SYSTEM_PROMPT = """You extract atomic, checkable facts from a single page of a document \
(company filing, financial report, government report, or similar).

A "fact" is a specific, checkable claim -- typically a number, a status, or a stated relationship \
(e.g. a metric with a value, a person's role, an event with a date). Skip vague or purely narrative \
sentences ("the company performed well") that carry no checkable content.

Do NOT extract table-of-contents, index, or navigational entries -- e.g. "Section Name ... page 12" or a \
heading immediately followed by a bare page number. These describe the document's own layout, not a checkable \
fact about the subject matter, and must be skipped even if they appear inside a table.

For every fact you find, report:
- subject: the entity the fact is about (a company, a country, a person, a division -- whatever fits)
- metric: what is being measured or claimed, in your own concise words
- value: the number or stated value (as a string, keep original formatting e.g. "7,831")
- unit: the unit if any (e.g. "INR crore", "%", "USD million"). Use null if not applicable.
- period: the time period or as-of date this fact applies to, if stated (e.g. "FY24", "as of 31 March 2024"). \
Use null if not stated.
- scope: any qualifying scope (e.g. "consolidated", "standalone", "excluding one-time items"). Use null if none.
- quote: the exact sentence or table row (verbatim, as it appears on the page) that supports this fact. Keep it \
short -- one sentence or one table row, not a paragraph.
- confidence: your confidence (0.0-1.0) that you extracted this correctly and it's genuinely a checkable fact.

Only extract facts explicitly stated or directly readable from a table on this page. Do not infer, calculate, \
or bring in outside knowledge. If a table is malformed or ambiguous, still extract what you can and lower the \
confidence score rather than skipping it silently.

Respond ONLY with a JSON object of this exact shape, no preamble, no markdown fences:
{"facts": [ { "subject": ..., "metric": ..., "value": ..., "unit": ..., "period": ..., "scope": ..., \
"quote": ..., "confidence": ... }, ... ]}

If there are no extractable facts on this page, respond with {"facts": []}
"""


def extract_facts_from_page(chunk: PageChunk, doc_id: int) -> List[dict]:
    """Calls Groq once per page. Returns a list of fact dicts (including
    'page'). Failures are logged to the failures table instead of raising,
    so one bad page doesn't kill the whole document."""

    combined_text = chunk.text
    if chunk.tables_text.strip():
        combined_text += "\n\n[TABLES ON THIS PAGE]\n" + chunk.tables_text

    if not combined_text.strip():
        return []
    
    if not any(ch.isdigit() for ch in combined_text):
        return []


    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4000,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": combined_text[:12000]},
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = _strip_code_fences(raw)
        data = json.loads(raw)
        facts = data.get("facts", []) if isinstance(data, dict) else data
        if not isinstance(facts, list):
            raise ValueError("Expected a 'facts' array")
    except Exception as e:
        db.insert_failure(
            doc_id=doc_id,
            stage="extraction",
            detail=f"page {chunk.page_number}: {type(e).__name__}: {e}",
            raw_snippet=combined_text[:500],
        )
        return []

    for f in facts:
        f["page"] = chunk.page_number
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
