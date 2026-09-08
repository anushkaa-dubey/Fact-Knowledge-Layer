# Fact Knowledge Layer

A system that ingests PDFs, extracts atomic facts, grounds each in exact
source evidence, and reasons about how facts relate across documents
(corroborate / contradict / reconcilable via context).

## Setup and Run Instructions

```bash
pip install -r requirements.txt
cp .env.example .env   # then put your real GROQ_API_KEY in .env
uvicorn Backend.main:app --reload
```

Open http://localhost:8000 for the UI, or http://localhost:8000/docs for
the Swagger API explorer.

## Approach

Pipeline: ingest (PyMuPDF + pdfplumber, page-level chunks) -> extract
(LLM call per page, schema-light JSON facts with quote+page evidence) ->
filter (deterministic post-filter drops table-of-contents/navigational
noise) -> normalize (regex-based unit conversion + batched LLM
canonicalization of subject/metric/period names) -> reason (TF-IDF
candidate matching across documents, then an LLM call per candidate pair
to classify corroborates/contradicts/reconciled with an explanation).

LLM: Groq (`openai/gpt-oss-120b`) via the OpenAI-compatible
chat.completions API, used for extraction, canonicalization, and
relationship reasoning. Originally built against the Claude API; switched
to Groq for cost reasons -- the JSON-object response contract stayed the
same, so the switch was isolated to the three LLM-calling modules.

## Limitations and Next Steps

- Candidate matching uses TF-IDF, not real embeddings -- catches
  reworded-but-similar facts but will miss pure paraphrase with little
  shared vocabulary (e.g. "logistics expense" vs "cost of doing
  business"). Swapping in an embeddings API is an isolated change to
  `reason.py`.
- Upload is processed synchronously; a background job queue would make
  large PDFs and multi-document uploads non-blocking.
- Table-of-contents/navigational entries were initially mis-extracted as
  facts (e.g. "MD&A -> page 520") -- this is documented as our Case 4
  (extraction failure) in the demo. Fixed with `Backend/filters.py`, a
  deterministic post-filter that catches these independent of which LLM
  produced them.

## Additional Notes

Starter dataset PDFs live under `starter-datasets/` (not committed with
API keys). The system makes no assumptions specific to those documents --
verified by running the same pipeline unmodified against them.
