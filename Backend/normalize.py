"""
Normalization: makes facts comparable across documents written by different
authors in different styles.

1. Unit/value normalization uses plain rules (regex), not an LLM --
   converting "crore"/"lakh"/"million"/"%" to a common base number is
   deterministic arithmetic; an LLM is the wrong tool for that.
2. Canonicalizing subject/metric/period names needs semantic judgment, so
   that part uses Groq, batched across facts. Batches are capped at
   BATCH_SIZE so a large document's response can't get truncated, and IDs
   are matched as strings on both sides -- a common silent-failure source
   is the model returning "1" instead of 1, which fails a dict lookup with
   no exception at all.
"""
import json
import re
import os
from typing import List, Optional, Tuple

from . import db
from .extract import get_client, MODEL, _strip_code_fences

UNIT_MULTIPLIERS = {
    "crore": 1e7, "cr": 1e7,
    "lakh": 1e5, "lac": 1e5,
    "million": 1e6, "mn": 1e6,
    "bn": 1e9, "billion": 1e9,
    "thousand": 1e3,
    "%": 1, "percent": 1,
}

CURRENCY_PATTERN = re.compile(r"(₹|inr|rs\.?|usd|\$)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[-+]?[\d,]*\.?\d+")


def normalize_value_unit(value: str, unit: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    """Best-effort deterministic parse of value+unit into (float, base_unit).
    Returns (None, None) if it can't be parsed confidently."""
    if value is None:
        return None, None

    raw = str(value).strip()
    num_match = NUMBER_PATTERN.search(raw.replace(",", ""))
    if not num_match:
        return None, None
    try:
        num = float(num_match.group())
    except ValueError:
        return None, None

    unit_text = (unit or "").lower()
    currency = None
    cur_match = CURRENCY_PATTERN.search(unit_text) or CURRENCY_PATTERN.search(raw)
    if cur_match:
        currency = "INR" if cur_match.group().lower() in ("₹", "inr", "rs", "rs.") else "USD"

    multiplier = 1.0
    base_unit = currency or ""
    for keyword, mult in UNIT_MULTIPLIERS.items():
        if keyword in unit_text:
            multiplier = mult
            if keyword == "%":
                base_unit = "%"
            break

    normalized_value = num * multiplier
    normalized_unit = (base_unit + "_absolute").strip("_") if base_unit else "absolute"
    return normalized_value, normalized_unit


CANONICALIZE_SYSTEM_PROMPT = """You will be given a JSON list of facts extracted from one or more documents. \
Each has an id, subject, metric, and period as originally written.

Your job: assign each fact a canonical_subject, canonical_metric, and canonical_period so that facts which \
refer to the SAME real-world entity/metric/period get the EXACT SAME canonical string, even if worded \
differently in the original. Examples of things that should collapse to the same canonical form:
- "Delhivery Limited", "Delhivery", "the Company" (when clearly the same company) -> one canonical_subject
- "revenue from operations", "total revenue", "operating revenue" (same line item) -> one canonical_metric. \
But do NOT merge genuinely different metrics -- "revenue from operations" and "net profit" must stay distinct.
- "FY24", "financial year ended 31 March 2024", "year ended March 2024" -> one canonical_period. But do NOT \
merge different periods -- FY23 and FY24 must stay distinct. When in doubt, keep separate rather than merge, \
since a false merge causes a false contradiction later.

Respond ONLY with a JSON object of this exact shape, no preamble, no markdown fences:
{"results": [ {"id": <id>, "canonical_subject": "...", "canonical_metric": "...", "canonical_period": "..."} ]}
"""

BATCH_SIZE = 40  # keeps each call's JSON response comfortably under the token limit


def _canonicalize_batch(batch: List[dict]) -> None:
    payload = [
        {"id": f["id"], "subject": f["subject"], "metric": f["metric"], "period": f["period"]}
        for f in batch
    ]

    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=8000,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CANONICALIZE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = _strip_code_fences(raw)
        data = json.loads(raw)
        results = data.get("results", []) if isinstance(data, dict) else data
    except Exception as e:
        for f in batch:
            db.insert_failure(doc_id=None, stage="normalization",
                               detail=f"{type(e).__name__}: {e}", raw_snippet=str(payload)[:500])
        return

    # Match by STRING id on both sides -- a type mismatch here (model
    # returns "1" instead of 1) fails a dict lookup silently, no exception,
    # which is the most likely reason canonical_* stayed null before.
    by_id = {str(f["id"]): f for f in batch}
    matched_ids = set()

    for r in results:
        fact = by_id.get(str(r.get("id")))
        if not fact:
            continue
        matched_ids.add(str(r.get("id")))
        normalized_value, normalized_unit = normalize_value_unit(fact["value"], fact["unit"])
        db.update_fact_normalization(
            fact_id=fact["id"],
            canonical_subject=r.get("canonical_subject"),
            canonical_metric=r.get("canonical_metric"),
            canonical_period=r.get("canonical_period"),
            normalized_value=normalized_value,
            normalized_unit=normalized_unit,
        )

    unmatched = len(batch) - len(matched_ids)
    if unmatched > 0:
        db.insert_failure(
            doc_id=None, stage="normalization",
            detail=f"{unmatched}/{len(batch)} facts had no matching id in the model's "
                   f"response (dropped or id mismatch) and were left null.",
            raw_snippet=str(payload)[:500],
        )


def canonicalize_facts(facts: List[dict]) -> None:
    """Batched LLM call(s) to assign canonical names, split into fixed-size
    batches so a large document's response can't get truncated."""
    if not facts:
        return
    for i in range(0, len(facts), BATCH_SIZE):
        _canonicalize_batch(facts[i:i + BATCH_SIZE])
