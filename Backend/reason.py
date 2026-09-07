"""
Cross-document reasoning:
finds candidate fact pairs that plausibly refer to the same real-world thing,
then asks the LLM to judge how they relate.

Matching uses TF-IDF + cosine similarity locally.
Only pairs above the similarity threshold and from different documents
are sent to the LLM.
"""

import json
import itertools
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import db
from .extract import get_client, MODEL, _strip_code_fences


SIMILARITY_THRESHOLD = 0.35


REASONING_SYSTEM_PROMPT = """You are given two facts extracted from two different documents, each with its supporting quote. Decide how they relate.

Respond with a JSON object:
{
  "type": "corroborates" | "contradicts" | "reconciled" | "unrelated",
  "explanation": "<2-3 sentences, plain language, referencing the specific evidence>",
  "confidence": <0.0-1.0>
}

Guidance:

- "corroborates": both facts state the same real-world value/status for the same subject, metric, and period.

- "contradicts": both facts claim to describe the same subject, metric, and period, but state different values/statuses, AND the difference cannot be explained from the given context.

- "reconciled": the facts look contradictory at first glance but are explainable by a stated difference in time period, scope, methodology, or units. Explain exactly what resolves the difference.

- "unrelated": the facts are not actually comparable. Do not force a verdict.

Be conservative. Only call something a contradiction when it cannot reasonably be reconciled from the provided context.

Respond ONLY with the JSON object, no preamble, no markdown fences.
"""


def find_candidate_pairs(
    facts: List[dict],
) -> List[Tuple[dict, dict, float]]:
    """
    Returns (fact_a, fact_b, similarity) for cross-document
    pairs above the similarity threshold.
    """

    texts = []

    for f in facts:
        text = " ".join(
            filter(
                None,
                [
                    f.get("canonical_subject") or f.get("subject"),
                    f.get("canonical_metric") or f.get("metric"),
                ],
            )
        )

        texts.append(text or "")

    if len(facts) < 2 or not any(texts):
        return []

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    sims = cosine_similarity(matrix)

    candidates = []

    for i, j in itertools.combinations(range(len(facts)), 2):

        # Only compare facts from different documents
        if facts[i]["doc_id"] == facts[j]["doc_id"]:
            continue

        score = sims[i][j]

        if score >= SIMILARITY_THRESHOLD:
            candidates.append(
                (
                    facts[i],
                    facts[j],
                    float(score),
                )
            )

    candidates.sort(key=lambda c: -c[2])

    return candidates


def reason_about_pair(fact_a: dict, fact_b: dict) -> dict:

    payload = {
        "fact_a": {
            "subject": fact_a.get("subject"),
            "metric": fact_a.get("metric"),
            "value": fact_a.get("value"),
            "unit": fact_a.get("unit"),
            "period": fact_a.get("period"),
            "scope": fact_a.get("scope"),
            "quote": fact_a.get("quote"),
        },
        "fact_b": {
            "subject": fact_b.get("subject"),
            "metric": fact_b.get("metric"),
            "value": fact_b.get("value"),
            "unit": fact_b.get("unit"),
            "period": fact_b.get("period"),
            "scope": fact_b.get("scope"),
            "quote": fact_b.get("quote"),
        },
    }

    client = get_client()

    try:
        # GROQ API
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": REASONING_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(payload),
                },
            ],
        )

        raw = response.choices[0].message.content.strip()

        raw = _strip_code_fences(raw)

        result = json.loads(raw)

        return result

    except Exception as e:

        db.insert_failure(
            doc_id=None,
            stage="reasoning",
            detail=f"{type(e).__name__}: {e}",
            raw_snippet=str(payload)[:500],
        )

        return {
            "type": "unrelated",
            "explanation": f"reasoning call failed: {e}",
            "confidence": 0.0,
        }


def run_cross_document_reasoning(
    min_confidence: float = 0.0,
) -> int:
    """
    Runs matching + reasoning over all facts currently in the DB
    and stores relationships.

    Returns number of relationships created.
    """

    facts = db.get_all_facts()

    candidates = find_candidate_pairs(facts)

    created = 0

    for fact_a, fact_b, sim_score in candidates:

        result = reason_about_pair(
            fact_a,
            fact_b,
        )

        rel_type = result.get(
            "type",
            "unrelated",
        )

        if rel_type == "unrelated":
            continue

        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )

        if confidence < min_confidence:
            continue

        db.insert_relationship(
            fact_id_a=fact_a["id"],
            fact_id_b=fact_b["id"],
            rel_type=rel_type,
            explanation=result.get(
                "explanation",
                "",
            ),
            confidence=confidence,
        )

        created += 1

    return created