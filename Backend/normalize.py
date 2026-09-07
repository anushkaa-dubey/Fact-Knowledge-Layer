"""
Normalization: makes facts comparable across documents written by different
authors in different styles.

Two different mechanisms are used deliberately:

1. Unit/value normalization is done with plain rules (regex), not an LLM.
   Converting "crore" / "lakh" / "million" / "billion" / "%" to a common
   base number is a deterministic operation -- using an LLM for arithmetic
   is slower, costs money, and is the kind of thing LLMs still get wrong on
   edge cases. Rules are the right tool here.

2. Canonicalizing subject/metric names ("Delhivery Limited" vs "the Company"
   vs "Delhivery"; "revenue from operations" vs "total revenue") genuinely
   needs semantic judgment, so that part uses an LLM, batched across many
   facts at once to keep it cheap and to let the model see the full set of
   names it needs to reconcile in one pass.
"""

import json
import re
import os
from typing import List, Optional, Tuple

from . import db
from .extract import get_client, MODEL, _strip_code_fences


UNIT_MULTIPLIERS = {
    "crore": 1e7,
    "cr": 1e7,
    "lakh": 1e5,
    "lac": 1e5,
    "million": 1e6,
    "mn": 1e6,
    "bn": 1e9,
    "billion": 1e9,
    "thousand": 1e3,
    "%": 1,
    "percent": 1,
}


CURRENCY_PATTERN = re.compile(r"(₹|inr|rs\.?|usd|\$)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[-+]?[\d,]*\.?\d+")


def normalize_value_unit(
    value: str,
    unit: Optional[str]
) -> Tuple[Optional[float], Optional[str]]:
    """Best-effort deterministic parse of value+unit into (float, base_unit).

    Returns (None, None) if it can't be parsed confidently.
    Callers should keep the original value/unit for display either way;
    this is only used for numeric comparison during reasoning.
    """

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

    cur_match = (
        CURRENCY_PATTERN.search(unit_text)
        or CURRENCY_PATTERN.search(raw)
    )

    if cur_match:
        currency = (
            "INR"
            if cur_match.group().lower() in ("₹", "inr", "rs", "rs.")
            else "USD"
        )

    multiplier = 1.0
    base_unit = currency or ""

    for keyword, mult in UNIT_MULTIPLIERS.items():

        if keyword in unit_text:
            multiplier = mult

            if keyword == "%":
                base_unit = "%"
            elif not base_unit:
                base_unit = ""

            break

    normalized_value = num * multiplier

    normalized_unit = (
        (base_unit + "_absolute").strip("_")
        if base_unit
        else "absolute"
    )

    return normalized_value, normalized_unit


CANONICALIZE_SYSTEM_PROMPT = """You will be given a JSON list of facts extracted from one or more documents.
Each has an id, subject, metric, and period as originally written.

Your job: assign each fact a canonical_subject, canonical_metric, and canonical_period so that facts which
refer to the SAME real-world entity/metric/period get the EXACT SAME canonical string, even if worded
differently in the original.

Examples of things that should collapse to the same canonical form:

- "Delhivery Limited", "Delhivery", "the Company" (when clearly referring to the same company)
  -> one canonical_subject

- "revenue from operations", "total revenue", "operating revenue" (when they clearly mean the same line item)
  -> one canonical_metric

But do NOT merge genuinely different metrics.
For example, "revenue from operations" and "net profit" must stay distinct even though both are financial figures.

- "FY24", "financial year ended 31 March 2024", "year ended March 2024"
  -> one canonical_period

But do NOT merge different periods.
FY23 and FY24 must stay distinct.

When in doubt, keep them separate rather than merging, since a false merge causes a false contradiction later.

Respond ONLY with a JSON object containing a "results" key.

The value of "results" must be an array of objects in this exact form:

{
  "results": [
    {
      "id": <id>,
      "canonical_subject": "...",
      "canonical_metric": "...",
      "canonical_period": "..."
    }
  ]
}

No preamble, no markdown fences.
"""


def canonicalize_facts(facts: List[dict]) -> None:
    """Batched LLM call to assign canonical names, then writes results back
    to the DB.

    Facts should all belong to the same reasoning scope (e.g. all facts
    currently in the system) so the model can see everything it needs
    to reconcile at once.
    """

    if not facts:
        return

    payload = [
        {
            "id": f["id"],
            "subject": f["subject"],
            "metric": f["metric"],
            "period": f["period"],
        }
        for f in facts
    ]

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
                    "content": CANONICALIZE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(payload),
                },
            ],
        )

        raw = response.choices[0].message.content.strip()

        raw = _strip_code_fences(raw)

        data = json.loads(raw)

        if isinstance(data, dict):
            results = data.get("results", [])
        else:
            results = data

    except Exception as e:

        for f in facts:
            db.insert_failure(
                doc_id=None,
                stage="normalization",
                detail=f"{type(e).__name__}: {e}",
                raw_snippet=str(payload)[:500],
            )

        return

    by_id = {
        f["id"]: f
        for f in facts
    }

    for r in results:

        fact = by_id.get(r.get("id"))

        if not fact:
            continue

        normalized_value, normalized_unit = normalize_value_unit(
            fact["value"],
            fact["unit"],
        )

        db.update_fact_normalization(
            fact_id=fact["id"],
            canonical_subject=r.get("canonical_subject"),
            canonical_metric=r.get("canonical_metric"),
            canonical_period=r.get("canonical_period"),
            normalized_value=normalized_value,
            normalized_unit=normalized_unit,
        )