# Antigravity Mission Prompt — Fact Knowledge Layer

Paste everything below into Antigravity as one mission. It's written as a spec so
the agent can plan, scaffold, and self-verify rather than needing you to
micromanage each file.

---

## MISSION

Build a **Fact Knowledge Layer**: a system that ingests PDFs, extracts atomic
facts, grounds each fact in the exact source evidence, and reasons about how
facts across documents relate (corroborate / contradict / reconcilable via
context). This must generalize to PDFs it has never seen — do not hardcode
facts, filenames, schemas, or document-specific rules.

Build it as a working prototype, not a polished product. Clarity of reasoning
matters more than UI polish.

## TECH STACK (use this unless you have a strong reason not to)

- Backend: Python + FastAPI
- PDF parsing: PyMuPDF (text + page numbers) and pdfplumber (tables)
- LLM: Claude API (Anthropic) for extraction, normalization, and reasoning —
  use structured/tool-based JSON outputs, not free-text parsing
- Embeddings: any available embedding API (OpenAI, Voyage, or local
  sentence-transformers if no API key) for candidate-matching facts across
  documents before doing expensive pairwise LLM reasoning
- DB: SQLite for the prototype (schema below); make it trivial to swap for
  Postgres later
- Frontend: a single-page React app (or plain HTML if faster) — file upload,
  a fact browser with expandable evidence/quotes, and a relationships view
  filterable by corroborates / contradicts / reconciled
- Keep all API keys in a `.env` file, never committed. Add a `.env.example`.

## PIPELINE

### 1. Ingest
- Accept PDF upload via API endpoint.
- Extract text per page, preserving page numbers.
- Extract tables separately (pdfplumber) since financial/statistical PDFs
  carry most of the important numeric facts in tables, not prose.
- Chunk by page/section, not fixed token windows.

### 2. Fact extraction (schema-light, LLM-driven)
- For each chunk, prompt Claude to extract atomic, checkable facts. Do NOT
  hardcode field names for a specific domain (e.g. don't assume "revenue" is
  a field) — instead have the model infer per-fact:
  - subject (entity the fact is about)
  - metric / claim
  - value
  - unit
  - time period / as-of date
  - scope (e.g. consolidated vs standalone, national vs state)
  - a verbatim short quote it was extracted from
  - source document id + page number
  - a confidence score
- Store every fact with its exact quote and page reference — this is the
  "grounded in evidence" requirement and is non-negotiable.

### 3. Normalization
- Canonicalize entity names, dates/fiscal periods, and units (e.g. ₹ crore
  vs ₹ lakh vs $ million; "FY24" vs "year ended March 31, 2024") so facts
  become comparable across documents written by different authors.
- This step is what prevents false contradictions caused by wording, not
  substance — treat it as a first-class part of the reasoning, not a
  throwaway cleanup step.

### 4. Cross-document reasoning
- Use embeddings to find candidate fact pairs/groups across documents that
  plausibly refer to the same subject+metric (avoid O(n²) LLM calls on
  everything).
- For each candidate group, prompt Claude with all the facts' quotes and ask
  it to classify the relationship as corroborates / contradicts /
  reconcilable-via-context, and require a plain-language explanation citing
  which specific evidence drove the judgment.
- Persist relationships with their explanation and confidence score.

## DATA MODEL (SQLite, keep it this simple)

```sql
Document(id, filename, upload_date)
Fact(id, doc_id, page, subject, metric, value, unit, period, scope, quote, confidence)
Relationship(id, fact_id_a, fact_id_b, type, explanation, confidence)
```

## API / UI REQUIREMENTS

- `POST /documents` — upload a PDF, triggers the pipeline, returns doc id
- `GET /documents/{id}/facts` — list extracted facts with evidence
- `GET /relationships?type=contradicts|corroborates|reconciled` — browse
  cross-document relationships with explanations
- A minimal frontend that lets a non-technical reviewer: upload a PDF, browse
  facts, click into evidence (quote + page), and browse relationships with
  the system's explanation visible.

## STARTER DATASET

Two folders of real PDFs are provided at `starter-datasets/`:
- `delhivery/` — a 2022 prospectus, an FY24 annual report, and a Q4 FY24
  earnings presentation. Good for corporate/financial cross-checks
  (revenue figures, director/leadership status across filing dates).
- `india-macroeconomy/` — the Economic Survey 2024-25, the RBI Annual
  Report 2024-25, and the IMF Article IV report. Good for macro figures
  (GDP growth, inflation) reported differently by different institutions
  and vintages — deliberately curated to contain real
  agreement/disagreement/context-dependent cases.

Use these to build and to source your four demo cases below. But the system
itself must not reference these documents by name or structure anywhere in
the code — it must work on an arbitrary new PDF.

## REQUIRED: FOUR DEMONSTRATED CASES

Build test/demo scripts (or a documented walkthrough) that surface, with
evidence and the system's own explanation for each:

1. A fact corroborated across two or more of the provided documents, even
   though phrased differently.
2. A genuine or likely contradiction between two facts.
3. An apparent contradiction that the system correctly explains away via
   context (different time period, scope, or units).
4. An extraction or reasoning failure — something the pipeline actually
   gets wrong on this dataset. Document what failed and either fix it or
   write up how you would fix it.

## OPTIONAL — GENERALIZATION TEST (do this near the end, not first)

To prove the system isn't overfit to the starter PDFs, find 1-2 additional
public PDFs of the same *kind* (e.g. another company's annual report, or
another macroeconomic report) and run them through the same pipeline
unmodified. Good sources: Kaggle datasets tagged "annual report," "10-K
filings," or "financial statements" (search kaggle.com/datasets), or any
public company/government report PDF. Do not write dataset-specific code to
handle these — if the pipeline needs new rules for a new PDF, that's a
finding to report in Limitations, not something to hardcode around.

## BROWNIE POINTS (attempt only after the core loop works end-to-end)

Pick one or two, don't try all of them:
- Handle large PDFs (100+ pages) without major slowdown — stream/parallelize
  chunk extraction.
- Support many PDFs in the same knowledge layer at once.
- Let the fact schema evolve dynamically as new kinds of facts appear,
  rather than assuming a fixed set of fields.
- Support adding a new document incrementally without recomputing
  relationships for the entire existing corpus — only recompute what's
  newly comparable.

## DELIVERABLES

- A GitHub repo with clean, meaningful commit history (not one giant commit).
- `README.md` with these exact sections: Setup and Run Instructions, Video
  Demo (link, ≤3 min, showing a PDF being processed and all four required
  cases), Approach (architecture, decisions, trade-offs, which AI tools were
  used and how), Limitations and Next Steps, Additional Notes.
- No credentials committed. Include `.env.example`. If any part depends on a
  paid API, include enough sample output/screenshots in the repo for
  reviewers to evaluate without needing your account.

## WORKING STYLE

Work incrementally: get PDF → text → single extracted fact working first and
show me the output before building normalization or cross-document
reasoning. Then get one clean corroboration example working end-to-end
before attempting the contradiction/reconciliation logic. Flag any step
where you had to guess at a design decision instead of silently picking one.