"""
FastAPI app. Three doors in, as discussed:
  POST /documents            -- upload a PDF, runs the full pipeline
  GET  /documents/{id}/facts -- see extracted facts + evidence for one doc
  GET  /relationships        -- see cross-document relationships
Plus a couple of convenience endpoints (list docs, list failures) that make
the "extraction/reasoning failure" required case easy to demo.
"""
import os
import shutil
import tempfile
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional

from . import db
from .ingest import extract_pages, page_count
from .extract import extract_facts_from_page
from .normalize import canonicalize_facts
from .reason import run_cross_document_reasoning

app = FastAPI(title="Fact Knowledge Layer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    db.init_db()


def process_document(doc_id: int, pdf_path: str):
    """Runs ingestion -> extraction -> normalization -> cross-doc reasoning
    for one newly uploaded document, then re-runs reasoning across the WHOLE
    corpus (cheap thanks to TF-IDF pre-filtering) so new facts get compared
    against everything already in the system."""
    try:
        chunks = extract_pages(pdf_path)
        db.update_document_status(doc_id, "processing", page_count=len(chunks))

        for chunk in chunks:
            facts = extract_facts_from_page(chunk, doc_id)
            for f in facts:
                f["id"] = db.insert_fact(doc_id, f)

        doc_facts = db.get_facts_for_doc(doc_id)
        canonicalize_facts(doc_facts)

        run_cross_document_reasoning()

        db.update_document_status(doc_id, "done")
    except Exception as e:
        db.insert_failure(doc_id=doc_id, stage="pipeline", detail=f"{type(e).__name__}: {e}")
        db.update_document_status(doc_id, "failed")


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    doc_id = db.insert_document(file.filename, datetime.now(timezone.utc).isoformat())

    # Synchronous for simplicity/predictability in this prototype -- see
    # README "Limitations" for the note on making this a background job.
    process_document(doc_id, tmp_path)
    os.unlink(tmp_path)

    return db.get_document(doc_id)


@app.get("/documents")
def list_documents():
    return db.list_documents()


@app.get("/documents/{doc_id}/facts")
def get_document_facts(doc_id: int):
    if not db.get_document(doc_id):
        raise HTTPException(404, "Document not found")
    return db.get_facts_for_doc(doc_id)


@app.get("/relationships")
def get_relationships(type: Optional[str] = Query(default=None,
                       description="corroborates | contradicts | reconciled")):
    return db.get_relationships(rel_type=type)


@app.get("/failures")
def get_failures():
    """Surfaces extraction/normalization/reasoning failures -- this is the
    endpoint to point to for the required 'failure case' demo."""
    return db.get_failures()


@app.post("/reasoning/rerun")
def rerun_reasoning():
    """Manually re-trigger cross-document reasoning over everything in the
    DB. Useful after uploading several documents in a row."""
    count = run_cross_document_reasoning()
    return {"relationships_created": count}


# Serve the minimal frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")