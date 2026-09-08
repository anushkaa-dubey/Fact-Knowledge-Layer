# Fact Knowledge Layer

A system that ingests PDFs, extracts atomic facts grounded in exact source
evidence, and reasons about how facts relate across documents — corroborating,
contradicting, or reconcilable through context (time period, scope, units).

## Setup and Run Instructions

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your `GROQ_API_KEY` to `.env`, then run:

```bash
uvicorn Backend.main:app --reload
```

- Web UI: **http://localhost:8000**
- Swagger API docs: **http://localhost:8000/docs**

To upload a PDF: use the UI's upload button, or `POST /documents` in Swagger.
Facts appear under `GET /documents/{id}/facts`; cross-document relationships
under `GET /relationships`; pipeline issues under `GET /failures`.

## Approach

Pipeline: **PDF → Ingest → Extract → Filter → Normalize → Reason → Store**
<img width="509" height="329" alt="image" src="https://github.com/user-attachments/assets/9a0868f8-c5e1-4939-a70e-d831ec949ee4" />


**Ingestion** — PyMuPDF extracts page-level text; pdfplumber extracts table
content separately (financial/report PDFs carry most numeric facts in
tables, which plain text extraction tends to mangle). Page numbers are
preserved throughout for exact evidence citation.

**Fact extraction** — a Groq LLM call per page extracts atomic, checkable
facts with no fixed schema (no hardcoded "revenue" or "GDP" fields) — the
model reports each fact's own subject, metric, value, unit, period, and
scope, because the same code has to work on a company prospectus and a
macroeconomic report without being told which. Every fact keeps its exact
supporting quote and page number, satisfying the "grounded in evidence"
requirement structurally rather than as an afterthought.

**Filtering** — a deterministic post-extraction filter (`Backend/filters.py`)
removes document-navigation noise, such as table-of-contents rows
("Management's Discussion and Analysis ... page 520") that structurally
resemble facts but are metadata about the document's own layout, not
claims about the subject matter. See "Failure Handling" below — this was
found as a real bug during development, not a theoretical concern.

**Normalization** — units/values (crore, lakh, million, %) are converted to
a common base with deterministic regex rules, since unit arithmetic is not
something an LLM should be trusted to do reliably. Subject/metric/period
names are canonicalized by a batched LLM call (e.g. "Delhivery Limited" and
"the Company" collapse to one canonical subject) since that step genuinely
needs semantic judgment.

**Cross-document reasoning** — TF-IDF + cosine similarity over canonical
subject/metric text finds candidate fact pairs across *different*
documents cheaply, without an embeddings API. Only pairs above a similarity
threshold go to the LLM, which classifies each as `corroborates` /
`contradicts` / `reconciled` / `unrelated`, with a plain-language
explanation and confidence score.

**Storage** — SQLite holds documents, facts (with evidence + normalized
fields), relationships, and a `failures` table logging every extraction,
normalization, or reasoning issue — so the system's mistakes are visible
in the UI, not hidden.

## Demonstrated Required Cases

**Status: partially captured.** The pipeline is designed to produce all
four required cases end-to-end (see "Approach" above for exactly how each
one is generated), but a full multi-document run to capture clean
corroborate/contradict/reconciled examples was blocked by Groq API
token/rate limits during testing — canonicalization and reasoning calls
started failing mid-run once enough pages had been processed, which is
visible directly in this project's own `failures` table (that table exists
specifically to surface this kind of issue rather than hide it).

**What is confirmed working end-to-end**, independent of any rate limit:
- **Case 4 (extraction/reasoning failure)** is fully captured and fixed:
  table-of-contents rows (e.g. *"Management's Discussion and Analysis →
  page 520"*) were being extracted as facts and two such entries were
  wrongly marked `corroborates` by the reasoning step. Root cause and fix
  are detailed in "Failure Handling" below, and the fix
  (`Backend/filters.py`) was verified against the exact failure case plus
  real facts (revenue figures, GDP growth, pin codes covered) with no
  false positives.
- Fact extraction with grounded evidence (subject/metric/value/quote/page)
  works correctly on real starter documents.
- The reasoning classification logic (corroborates/contradicts/reconciled)
  runs correctly when given fact pairs — confirmed on the ToC case above,
  which the model correctly reasoned about once the pair reached it.

**What is not yet captured**: a clean corroborates, a genuine contradicts,
and a reconciled example from real financial/macro facts across two
documents, blocked by rate limiting before completing a full run. See
"How to Fix" below — the fix is understood, just not yet re-run at
submission time.

## How to Fix the Token/Rate-Limit Issue

- **Add retry with exponential backoff** around every Groq call in
  `extract.py`, `normalize.py`, and `reason.py` (currently a single
  attempt; a 429 goes straight to the `failures` table instead of
  retrying after a short wait).
- **Reduce call volume**: the extraction skip-page heuristic (skip pages
  with no digits) and thread-pooled page extraction both cut the number
  and wall-clock time of calls, but total call count could be reduced
  further by batching multiple pages into one extraction call instead of
  one call per page, at the cost of slightly less precise page attribution.
- **Respect Groq's actual rate limit headers** — read the
  `retry-after`/rate-limit headers Groq returns and pace requests
  accordingly, instead of firing the thread pool at a fixed worker count.
- **Cache normalization results** — canonical names for the same
  subject/metric string don't need to be re-derived by the LLM every run;
  a persisted cache keyed on the raw string would cut canonicalization
  calls to near-zero after the first pass.
- **Upgrade Groq tier or add a provider fallback** (e.g. fall back to a
  different model/provider on repeated 429s) for a production version.

## Engineering Trade-offs

**LLM + deterministic rules, not LLM-only** — flexible LLM extraction for
open-ended fact discovery, deterministic rules for anything predictable
(unit conversion, navigation filtering). Reduces dependence on getting
every prompt exactly right, and the navigation filter keeps working even
after switching LLM providers (see below).

**TF-IDF candidate matching, not embeddings** — free, runs locally, no
extra API key. Weaker on pure paraphrase with little shared vocabulary
(e.g. "logistics expense" vs "cost of doing business") than real
embeddings would be — documented as a known limitation, and isolated to
`reason.py` if it needs upgrading later.

**Page-level chunking, not fixed-token windows** — keeps a fact and its
context (e.g. a table + the sentence introducing it) together, and makes
the page-number evidence citation exact rather than approximate.

**Claude → Groq mid-project** — originally built against the Claude API;
switched to Groq (`openai/gpt-oss-120b`) for cost reasons. The JSON-object
response contract stayed the same, so the change was isolated to the three
LLM-calling modules (`extract.py`, `normalize.py`, `reason.py`) — the
trade-off being Groq's free-tier rate limits, discussed above.

## Limitations and Next Steps

- **Rate limiting** (see "How to Fix" above) — the most immediate blocker
  right now; addressing it is the next concrete step before this can
  reliably run against larger multi-document batches.
- **Semantic matching** is TF-IDF, not embeddings — will miss facts that
  are heavily reworded with little shared vocabulary. Swapping in an
  embeddings API is an isolated change to `reason.py`.
- **Synchronous processing** — a large PDF blocks the upload request until
  fully processed. Page extraction is parallelized with a thread pool for
  speed, but a background job queue would make this fully non-blocking.
- **Reasoning scale** — TF-IDF pre-filtering keeps LLM reasoning calls
  proportional to plausible matches, not all pairs, but a much larger
  corpus would benefit from incremental indexing so adding one new
  document doesn't require rescanning the whole set.
- **Normalization consistency** — canonicalization is LLM-driven per batch
  with no persisted cache yet; a stored entity/metric registry would make
  canonical names stable across runs and cut repeat LLM calls (also helps
  the rate-limit issue above).
- **Filter heuristics** — the table-of-contents filter is rule-based and
  tuned against the failure case actually found in this dataset; a
  differently-formatted document could still slip facts past it, which is
  itself the kind of thing the required "failure case" is meant to
  surface honestly rather than hide.

## Failure Handling

During development with the Delhivery prospectus, the extraction layer
exposed a failure mode where table-of-contents entries containing page
numbers were interpreted as facts (e.g. *"Management's Discussion and
Analysis → page 520"*), and two such entries from different documents
were then marked `corroborates` by the reasoning step purely because they
shared the same heading and page number.

**Root cause:** structurally, a ToC row ("Heading ... 520") looks
identical to a real subject–value fact to an LLM extracting page by page
with no awareness that a given page is a table of contents.

**Fix:** added `Backend/filters.py`, a deterministic rule-based filter run
after extraction, independent of which LLM produced the fact — catches
facts whose metric references a page/section locator, or whose quote has
the literal shape of a ToC row (heading + dot leaders + bare number).
Verified against the exact failure case above plus real facts (revenue,
GDP growth, pin codes covered) to confirm no false positives. This
approach also makes the system less dependent on a particular LLM
provider's exact behavior.

## Tech Stack

Python · FastAPI · SQLite · PyMuPDF · pdfplumber · scikit-learn · Groq

## Project Structure

```text
Superjoin/
├── Backend/
│   ├── __init__.py
│   ├── db.py
│   ├── ingest.py
│   ├── extract.py
│   ├── filters.py
│   ├── normalize.py
│   ├── reason.py
│   └── main.py
├── frontend/
│   └── index.html
├── starter-datasets/
├── .env.example
├── requirements.txt
└── README.md
```

## Additional Notes

The system makes no assumptions specific to the provided starter documents
and accepts new PDFs through the same upload interface with no hardcoded
facts, filenames, or schemas. The rate-limit issue affected volume of
testing, not the design of the pipeline itself — the underlying extraction,
filtering, normalization, and reasoning logic each work independently, as
shown by the Case 4 example captured in full.
