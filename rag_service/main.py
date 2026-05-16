"""HTTP wrapper around the epsilon_rag library.

On startup the lifespan handler warms every model (Docling, Surya, formulas,
figures, embeddings, reranker) so the first request doesn't pay model-load
latency. The DB connection pool is opened then and drained on shutdown.

Routes:
  POST /ingest                          — run the full ingest pipeline
  POST /query                           — hybrid retrieval (no LLM)
  POST /channels                        — create a RAG channel (admin)
  GET  /channels/{id}/reports           — list reports + chunk counts (admin)
  DELETE /reports/{id}                  — delete a report and its chunks (admin)
  GET  /health                          — liveness probe

The admin routes are unauthenticated at this layer — only the platform
backend reaches rag_service over the docker network, and it enforces the
admin gate at its own /admin/* endpoints.
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

# epsilon_rag is on PYTHONPATH (set in Dockerfile.rag) so these imports
# resolve directly against /app/epsilon_rag.
from core.db import close_pool, get_conn, init_pool
from models.schema import IngestOptions
from pipeline import embeddings, figures, formulas, layout, ocr, registry, reranker
from pipeline.ingest import run_ingest_from_path
from pipeline.retrieval import hybrid_query

from schemas import (
    CreateChannelRequest,
    CreateChannelResponse,
    DeleteReportResponse,
    IngestRequest,
    IngestResponse,
    ListReportsResponse,
    QueryRequest,
    QueryResponse,
    ReportSummary,
    ResultMetadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rag_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool()
    started = time.perf_counter()
    for name, fn in [
        ("layout",     layout.init),
        ("ocr",        ocr.init),
        ("formulas",   formulas.init),
        ("figures",    figures.init),
        ("embeddings", embeddings.init),
        ("reranker",   reranker.init),
    ]:
        t = time.perf_counter()
        try:
            fn()
            logger.info("warmup: %s ready in %.1fs", name, time.perf_counter() - t)
        except Exception as exc:
            # A failed model load degrades the corresponding stage but the
            # service still answers requests — surface the failure as a
            # warning rather than crashing the container.
            logger.exception("warmup: %s failed — service will degrade: %s", name, exc)
    logger.info("warmup: total %.1fs", time.perf_counter() - started)
    yield
    close_pool()


app = FastAPI(title="Epsilon RAG Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Upsert a report row and run the full extract → chunk → embed → persist
    pipeline against the file at req.file_path (on the shared uploads volume)."""
    if not os.path.isfile(req.file_path):
        raise HTTPException(status_code=400, detail=f"File not found at path: {req.file_path}")

    try:
        report_id = registry.upsert_report(
            channel_id=req.rag_channel_id,
            filename=req.filename,
            title=req.title or req.filename,
            metadata=req.metadata or {},
        )
        logger.info("ingest: upserted report %s for channel %s", report_id, req.rag_channel_id)
    except Exception as exc:
        logger.exception("ingest: failed to upsert report: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to create report record: {exc}")

    try:
        stats = run_ingest_from_path(
            req.rag_channel_id,
            report_id,
            req.file_path,
            IngestOptions(),
        )
    except Exception as exc:
        logger.exception("ingest: pipeline failed for report %s: %s", report_id, exc)
        raise HTTPException(status_code=500, detail=f"Ingest pipeline error: {exc}")

    return IngestResponse(
        success=True,
        report_id=report_id,
        pages_processed=stats["pages_processed"],
        chunks_inserted=stats["chunks_inserted"],
        pages_total=stats["pages_total"],
        extractor=stats["extractor"],
        embed_model=stats["embed_model"],
        total_latency_ms=stats["total_latency_ms"],
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run hybrid retrieval and return per-chunk metadata (no content body).

    Neighbour-expansion rows are filtered out; only the primary ranked hits
    are returned, ordered by rerank score.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        results = hybrid_query(
            req.question,
            channel_id=req.rag_channel_id,
            top_k=req.top_k,
        )
    except Exception as exc:
        logger.exception("query: retrieval failed for channel %s: %s", req.rag_channel_id, exc)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    primary = [r for r in results if not r.get("is_neighbour")]
    metadata = [
        ResultMetadata(
            report_id=r["report_id"],
            page_number=r["page_number"],
            chunk_index=r["chunk_index"],
            rerank_score=r.get("rerank_score"),
            rrf_score=r.get("rrf_score"),
            section_title=(r.get("metadata") or {}).get("section_title", ""),
        )
        for r in primary
    ]

    return QueryResponse(
        success=True,
        result_count=len(metadata),
        results_metadata=metadata,
    )


# ── Admin endpoints ──────────────────────────────────────────────────────────

@app.post("/channels", response_model=CreateChannelResponse)
def create_channel(req: CreateChannelRequest):
    """Create a row in the RAG schema's public.channels table."""
    try:
        channel_id = registry.create_channel(
            name=req.name,
            description=req.description,
            metadata=req.metadata or {},
        )
    except Exception as exc:
        logger.exception("create_channel: failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to create channel: {exc}")
    return CreateChannelResponse(success=True, channel_id=channel_id)


@app.get("/channels/{channel_id}/reports", response_model=ListReportsResponse)
def list_reports(channel_id: str):
    """List every report in a channel, enriched with page and chunk counts
    pulled from report_chunks in a single round-trip."""
    try:
        reports = registry.list_reports(channel_id)
    except Exception as exc:
        logger.exception("list_reports: failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {exc}")

    summaries: list[ReportSummary] = []
    if not reports:
        return ListReportsResponse(success=True, reports=summaries)

    report_ids = [r["id"] for r in reports]
    try:
        with get_conn() as conn:
            cur = conn.execute(
                """
                SELECT "ReportId",
                       COUNT(DISTINCT "PageNumber") AS page_count,
                       COUNT(*)                     AS chunk_count
                FROM report_chunks
                WHERE "ReportId" = ANY(%s)
                GROUP BY "ReportId"
                """,
                [report_ids],
            )
            counts = {str(row[0]): (int(row[1]), int(row[2])) for row in cur.fetchall()}
    except Exception as exc:
        logger.exception("list_reports: chunk-count query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to count chunks: {exc}")

    for r in reports:
        page_count, chunk_count = counts.get(r["id"], (0, 0))
        summaries.append(ReportSummary(
            report_id=r["id"],
            filename=r["filename"],
            title=r["title"],
            page_count=page_count,
            chunk_count=chunk_count,
            created_at=r.get("created_at"),
        ))

    return ListReportsResponse(success=True, reports=summaries)


@app.delete("/reports/{report_id}", response_model=DeleteReportResponse)
def delete_report(report_id: str):
    """Delete a report and all of its chunks. The chunks would cascade via FK,
    but the explicit DELETE gives us a rowcount to report back."""
    try:
        with get_conn() as conn:
            cur = conn.execute(
                'DELETE FROM report_chunks WHERE "ReportId" = %s',
                [report_id],
            )
            deleted_chunks = cur.rowcount or 0
            cur = conn.execute(
                'DELETE FROM reports WHERE "Id" = %s',
                [report_id],
            )
            if (cur.rowcount or 0) == 0:
                raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
            conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("delete_report: failed for %s: %s", report_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {exc}")
    return DeleteReportResponse(success=True, deleted_chunks=deleted_chunks)
