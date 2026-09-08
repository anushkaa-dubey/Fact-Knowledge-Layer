"""
Cross-document reasoning: finds candidate fact pairs that plausibly refer to
the same real-world thing, then asks Groq to judge how they relate.

Matching uses TF-IDF + cosine similarity locally (free, no embeddings API
needed) over "canonical_subject canonical_metric" text. Only pairs above the
threshold and from DIFFERENT documents go to the LLM, keeping the number of
(comparatively expensive) reasoning calls small even as the corpus grows.
"""
import json
import itertools
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import db
from .extract import get_client, MODEL, _strip_code_fences

SIMILARITY_THRESHOLD = 0.35

REASONING_SYSTEM_PROMPT = """You are given two facts extracted from two different documents, each with its \
supporting quote. Decide how they relate.

Respond with a JSON object:
{"type": "corroborates" | "contradicts" | "reconciled" | "unrelated",
 "explanation": "<2-3 sentences, plain language, referencing the specific evidence>",
 "confidence": <0.0-1.0>}

Guidance:
- "corroborates": both facts state the same real-world value/status for the same subject, metric, and period \
(wording may differ).
- "contradicts": both facts describe the same subject, metric, and period, but state different values/statuses, \
and you cannot explain the difference from what's stated -- a genuine or likely conflict worth flagging.
- "reconciled": looks contradictory at first glance but is explainable by a stated difference in time period, \
scope (consolidated vs standalone), methodology, or units -- explain exactly which of these resolves it.
- "unrelated": not actually comparable (different subjects/metrics despite similar wording) -- say so plainly.

Be conservative: only call something a "contradiction" if you cannot reconcile it from the given context. If \
unsure, say so in the explanation and lower confidence rather than guessing.

Respond ONLY with the JSON object, no preamble, no markdown fences.
"""


def find_candidate_pairs(facts: List[dict]) -> List[Tuple[dict, dict, float]]:
    texts = []
    for f in facts:
        text = " ".join(filter(None, [
            f.get("canonical_subject") or f.get("subject"),
            f.get("canonical_metric") or f.get("metric"),
        ]))
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
        if facts[i]["doc_id"] == facts[j]["doc_id"]:
            continue
        score = sims[i][j]
        if score >= SIMILARITY_THRESHOLD:
            candidates.append((facts[i], facts[j], float(score)))

    candidates.sort(key=lambda c: -c[2])
    return candidates


def reason_about_pair(fact_a: dict, fact_b: dict) -> dict:
    payload = {
        "fact_a": {k: fact_a.get(k) for k in ("subject", "metric", "value", "unit", "period", "scope", "quote")},
        "fact_b": {k: fact_b.get(k) for k in ("subject", "metric", "value", "unit", "period", "scope", "quote")},
    }
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = _strip_code_fences(raw)
        return json.loads(raw)
    except Exception as e:
        db.insert_failure(doc_id=None, stage="reasoning",
                           detail=f"{type(e).__name__}: {e}", raw_snippet=str(payload)[:500])
        return {"type": "unrelated", "explanation": f"reasoning call failed: {e}", "confidence": 0.0}


def run_cross_document_reasoning(min_confidence: float = 0.0) -> int:
    """Runs matching + reasoning over ALL facts currently in the DB. Skips
    'unrelated' verdicts. Returns number of relationships created."""
    facts = db.get_all_facts()
    candidates = find_candidate_pairs(facts)

    created = 0
    for fact_a, fact_b, sim_score in candidates:
        result = reason_about_pair(fact_a, fact_b)
        rel_type = result.get("type", "unrelated")
        if rel_type == "unrelated":
            continue
        confidence = float(result.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        db.insert_relationship(
            fact_id_a=fact_a["id"], fact_id_b=fact_b["id"],
            rel_type=rel_type, explanation=result.get("explanation", ""), confidence=confidence,
        )
        created += 1
    return created
