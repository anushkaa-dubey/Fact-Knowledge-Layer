"""
FastAPI app. Three doors in:
  POST /documents            -- upload a PDF, runs the full pipeline
  GET  /documents/{id}/facts -- see extracted facts + evidence for one doc
  GET  /relationships        -- see cross-document relationships

Plus convenience endpoints for listing documents, filtering failures,
re-running reasoning, resetting data, and viewing summary counts.
"""

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import db
from .ingest import extract_pages
from .extract import extract_facts_from_page
from .normalize import canonicalize_facts
from .reason import run_cross_document_reasoning
from .filters import is_probable_toc_entry


MAX_WORKERS = int(
    os.environ.get("FACTLAYER_EXTRACT_WORKERS", "3")
)


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
    """
    Ingestion -> parallel extraction -> TOC filtering
    -> normalization -> cross-document reasoning.
    """

    try:
        # -----------------------------
        # Ingestion
        # -----------------------------

        chunks = extract_pages(pdf_path)
        doc_page_count = len(chunks)

        db.update_document_status(
            doc_id,
            "processing",
            page_count=doc_page_count
        )

        # -----------------------------
        # Parallel page extraction
        # -----------------------------

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            futures = {
                executor.submit(
                    extract_facts_from_page,
                    chunk,
                    doc_id
                ): chunk
                for chunk in chunks
            }

            for future in as_completed(futures):

                facts = future.result()

                for fact in facts:

                    # -----------------------------
                    # Filter probable TOC entries
                    # -----------------------------

                    if is_probable_toc_entry(
                        fact,
                        doc_page_count=doc_page_count
                    ):
                        db.insert_failure(
                            doc_id=doc_id,
                            stage="extraction_filtered",
                            detail=(
                                "Discarded probable "
                                "table-of-contents/navigational entry"
                            ),
                            raw_snippet=fact.get("quote", ""),
                        )
                        continue

                    # -----------------------------
                    # Store fact
                    # -----------------------------

                    fact["id"] = db.insert_fact(
                        doc_id,
                        fact
                    )

        # -----------------------------
        # Normalization
        # -----------------------------

        doc_facts = db.get_facts_for_doc(doc_id)

        canonicalize_facts(doc_facts)

        # -----------------------------
        # Cross-document reasoning
        # -----------------------------

        run_cross_document_reasoning()

        # -----------------------------
        # Done
        # -----------------------------

        db.update_document_status(
            doc_id,
            "done"
        )

    except Exception as e:

        db.insert_failure(
            doc_id=doc_id,
            stage="pipeline",
            detail=(
                f"{type(e).__name__}: {e}"
            )
        )

        db.update_document_status(
            doc_id,
            "failed"
        )


# ============================================================
# Document endpoints
# ============================================================

@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            400,
            "Only PDF files are supported."
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        shutil.copyfileobj(
            file.file,
            tmp
        )

        tmp_path = tmp.name

    doc_id = db.insert_document(
        file.filename,
        datetime.now(timezone.utc).isoformat()
    )

    try:
        process_document(
            doc_id,
            tmp_path
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return db.get_document(doc_id)


@app.get("/documents")
def list_documents():
    return db.list_documents()


@app.get("/documents/{doc_id}/facts")
def get_document_facts(
    doc_id: int
):
    if not db.get_document(doc_id):
        raise HTTPException(
            404,
            "Document not found"
        )

    return db.get_facts_for_doc(doc_id)


# ============================================================
# Relationship endpoints
# ============================================================

@app.get("/relationships")
def get_relationships(
    type: Optional[str] = Query(
        default=None,
        description=(
            "corroborates | contradicts | reconciled"
        )
    )
):
    return db.get_relationships(
        rel_type=type
    )


@app.post("/reasoning/rerun")
def rerun_reasoning():
    """
    Re-trigger cross-document reasoning over all facts.
    """

    count = run_cross_document_reasoning()

    return {
        "relationships_created": count
    }


# ============================================================
# Failure endpoints
# ============================================================

@app.get("/failures")
def get_failures(
    stage: Optional[str] = Query(
        default=None,
        description=(
            "extraction | extraction_filtered | "
            "normalization | reasoning | pipeline"
        )
    )
):
    return db.get_failures(
        stage=stage
    )


# ============================================================
# Admin
# ============================================================

@app.post("/admin/reset")
def reset_all():
    """
    Wipes all documents, facts, relationships,
    and failures.
    """

    db.reset_all()

    return {
        "status": "reset"
    }


# ============================================================
# Summary
# ============================================================

@app.get("/summary")
def summary():
    """
    Quick counts for the dashboard.
    """

    return {
        "documents": len(
            db.list_documents()
        ),
        "facts": len(
            db.get_all_facts()
        ),
        "relationships": len(
            db.get_relationships()
        ),
        "failures": len(
            db.get_failures()
        ),
    }


# ============================================================
# Frontend
# ============================================================

frontend_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "frontend"
)

if os.path.isdir(frontend_dir):

    app.mount(
        "/",
        StaticFiles(
            directory=frontend_dir,
            html=True
        ),
        name="frontend"
    )