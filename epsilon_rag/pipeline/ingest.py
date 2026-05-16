"""Full ingest pipeline — extract → chunk → embed → persist.

`run_ingest_from_path(report_id, pdf_path, options)` is the single entry
point: it reads PDF bytes off disk and walks them through the four
stages, returning a stats dict. Used by [main.py](../main.py).

All stages run synchronously; no event loop, no background workers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core import storage
from core.config import settings
from core.db import get_conn
from models.schema import ExtractOptions, IngestOptions
from pipeline import chunker, embeddings
from pipeline.errors import InvalidPdfError
from pipeline.orchestrator import process_document


def _sanitize_filename(name: str) -> str:
    """Make a filename safe for use as an S3 object key fragment.

    S3 itself allows almost anything in keys but spaces / control chars /
    non-ASCII make the MinIO console + downstream CLIs miserable. Keep
    alphanumerics + dot + dash + underscore; collapse everything else to '_'.
    """
    keep = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return keep.strip("_") or "document.pdf"



logger = logging.getLogger(__name__)

# Retry policy applied to every I/O / GPU stage.
# 3 retries → 4 total attempts; delays between consecutive attempts (seconds).
_RETRY_DELAYS: tuple[int, ...] = (2, 4, 6)


def _retry(fn, stage: str, report_id: str):
    """Run fn() up to 4 times (1 + 3 retries) with 2 / 4 / 6 s back-off.

    Logs a WARNING on each failed attempt and re-raises the last exception
    once all attempts are exhausted so the caller can mark the job Failed.
    """
    last_exc: Exception = RuntimeError("unreachable")
    total = len(_RETRY_DELAYS) + 1  # 4
    for attempt in range(1, total + 1):
        try:
            return fn()
        except InvalidPdfError as exc:
            # Deterministic input failure — the bytes will not change
            # between retries, so re-running burns ~40 s of GPU for
            # nothing. Fail fast so the job marks Failed quickly.
            logger.error(
                "[ingest %s] stage=%s aborting (non-retryable): %s",
                report_id, stage, exc,
            )
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == total:
                logger.error(
                    "[ingest %s] stage=%s failed after %d attempts: %s",
                    report_id, stage, total, exc,
                )
                raise
            delay = _RETRY_DELAYS[attempt - 1]
            logger.warning(
                "[ingest %s] stage=%s attempt %d/%d failed: %s — retrying in %ds",
                report_id, stage, attempt, total, exc, delay,
            )
            time.sleep(delay)
    raise last_exc  # unreachable but satisfies type-checkers


def run_ingest_from_path(
    channel_id: str,
    report_id: str,
    pdf_path: str | Path,
    options: IngestOptions,
) -> dict:
    """Read a local PDF and walk it through the full pipeline.

    `channel_id` and `report_id` must already exist in the channels /
    reports tables — the CLI handles creation via pipeline.registry
    before getting here. ChannelId is denormalised onto every chunk
    so retrieval can filter through the HNSW indexes without a JOIN.

    Returns a stats dict (caller prints or persists it). Raises on any
    unrecoverable error so the CLI can surface a non-zero exit code.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    pdf_bytes = path.read_bytes()
    logger.info(
        "[ingest %s] loaded %.2f MB from %s (channel=%s)",
        report_id, len(pdf_bytes) / 1024 / 1024, path, channel_id,
    )

    overall_started = time.perf_counter()

    # ── Stage 0: Object storage (MinIO) — upload raw PDF ─────────────────────
    # Done early so the source survives even if a later stage fails. Re-runs
    # are idempotent: the object key is deterministic in (channel_id,
    # report_id, filename) so a retry overwrites in place rather than
    # accumulating orphans. Markdown + metadata follow at stages 1.5 and 5.
    minio_cfg = settings.minio_config()
    _retry(
        lambda: storage.ensure_bucket(minio_cfg),
        "minio_ensure_bucket", report_id,
    )
    safe_name = _sanitize_filename(path.name)
    raw_key = storage.object_key(
        minio_cfg, channel_id, "raw", report_id, safe_name,
    )
    _retry(
        lambda: storage.safe_put(
            minio_cfg, raw_key, pdf_bytes, "application/pdf",
        ),
        "minio_upload_raw", report_id,
    )
    logger.info(
        "[ingest %s] uploaded raw PDF → s3://%s/%s",
        report_id, minio_cfg.bucket, raw_key,
    )

    # ── Stage 1: Extract ──────────────────────────────────────────────────────
    extract_options = ExtractOptions(
        extract_tables=options.extract_tables,
        extract_figures=options.extract_figures,
        extract_formulas=options.extract_formulas,
        ocr_fallback=options.ocr_fallback,
        figure_captioning=options.figure_captioning,
    )
    extract_response = _retry(
        lambda: process_document(pdf_bytes, extract_options, None),
        "extract", report_id,
    )
    pages_total   = extract_response.document_metadata.page_count
    extractor_tag = extract_response.extractor
    languages     = extract_response.document_metadata.detected_languages
    language      = languages[0] if languages else "mixed"
    logger.info(
        "[ingest %s] extracted %d pages (extractor=%s)",
        report_id, pages_total, extractor_tag,
    )

    # ── Stage 1.5: Upload markdown to MinIO ──────────────────────────────────
    # Markdown comes from Docling's own `export_to_markdown()` on the same
    # parsed tree the layout stage used — no second Docling pass. Empty
    # markdown is fine to upload (e.g. all-image PDFs); keeps the bucket
    # layout uniform so downstream consumers can list `markdown/*.md`
    # without "is this report missing?" branching.
    markdown_text = extract_response.markdown or ""
    # Encode once and reuse the byte length in the metadata payload below.
    markdown_blob = markdown_text.encode("utf-8")
    markdown_key  = storage.object_key(
        minio_cfg, channel_id, "markdown", f"{report_id}.md",
    )
    _retry(
        lambda: storage.safe_put(
            minio_cfg,
            markdown_key,
            markdown_blob,
            "text/markdown; charset=utf-8",
        ),
        "minio_upload_markdown", report_id,
    )
    logger.info(
        "[ingest %s] uploaded markdown (%d chars, %d bytes) → s3://%s/%s",
        report_id, len(markdown_text), len(markdown_blob),
        minio_cfg.bucket, markdown_key,
    )

    # ── Stage 2: Chunk ────────────────────────────────────────────────────────
    # Pure in-memory — deterministic, no I/O; retry not applicable.
    pending: list[dict] = []
    for page in extract_response.pages:
        page_num = page.page_number
        if not page_num:
            continue
        # Thread bbox through to the chunker — it returns `bboxes` per
        # chunk, which we persist into chunk metadata so the retrieval
        # layer can produce citation regions on the source page.
        block_dicts = [
            {
                "type":          b.type,
                "content":       b.content,
                "reading_order": b.reading_order,
                "bbox":          b.bbox,
            }
            for b in page.blocks
        ]
        if block_dicts:
            for idx, ch in enumerate(chunker.chunk_blocks_with_meta(block_dicts)):
                pending.append({
                    "page_number":   page_num,
                    "chunk_index":   idx,
                    "content":       ch["content"],
                    "section_title": ch["section_title"],
                    "block_types":   ch["block_types"],
                    "bboxes":        ch.get("bboxes") or [],
                })
        elif (page.content or "").strip():
            for idx, piece in enumerate(chunker.chunk_text(page.content)):
                pending.append({
                    "page_number":   page_num,
                    "chunk_index":   idx,
                    "content":       piece,
                    "section_title": "",
                    "block_types":   ["text"],
                    "bboxes":        [],
                })
    logger.info("[ingest %s] chunked → %d chunks", report_id, len(pending))

    # ── Stage 3: Embed (dense + sparse, hybrid) ──────────────────────────────
    dense_vectors: list[list[float]] = []
    sparse_vectors: list[dict[int, float]] = []
    if pending:
        if not embeddings.is_ready():
            raise RuntimeError(
                "embedding model is not loaded — pipeline cannot embed chunks"
            )
        dense_vectors, sparse_vectors = _retry(
            lambda: embeddings.embed([p["content"] for p in pending], "passage"),
            "embed", report_id,
        )
        if len(dense_vectors) != len(pending) or len(sparse_vectors) != len(pending):
            raise RuntimeError(
                f"embedder returned dense={len(dense_vectors)} sparse="
                f"{len(sparse_vectors)} vectors for {len(pending)} chunks"
            )
    logger.info(
        "[ingest %s] embedded %d chunks (model=%s, dim_dense=%d, dim_sparse=%d)",
        report_id, len(dense_vectors), settings.embed_model_id,
        len(dense_vectors[0]) if dense_vectors else 0,
        settings.embed_sparse_dim,
    )

    # ── Stage 4: Persist ─────────────────────────────────────────────────────
    from pgvector.psycopg import SparseVector

    def _persist() -> int:
        with get_conn() as conn:
            cur = conn.execute(
                'DELETE FROM report_chunks WHERE "ReportId" = %s',
                [report_id],
            )
            deleted = cur.rowcount or 0
            for i, p in enumerate(pending):
                dense_vec = np.array(dense_vectors[i], dtype=np.float32)
                # pgvector's SparseVector accepts a {idx: val} dict +
                # explicit dim; this format aligns 1:1 with what bge-m3's
                # lexical head emits (after the int-coercion we did in
                # pipeline/embeddings.py).
                sparse_vec = SparseVector(
                    sparse_vectors[i], settings.embed_sparse_dim,
                )
                metadata = {
                    "section_title": p["section_title"],
                    "block_types":   p["block_types"],
                    # `bboxes` is a list of 4-tuples (x_min, y_min, x_max, y_max)
                    # on the rendered page (top-left origin, pixel-space at
                    # the orchestrator's _RENDER_DPI = 150). Lets the UI
                    # / answer layer draw citation regions on the source.
                    "bboxes":        p.get("bboxes") or [],
                    "language":      language,
                    "extractor":     extractor_tag,
                }
                conn.execute(
                    """
                    INSERT INTO report_chunks
                        ("ChannelId", "ReportId", "PageNumber", "ChunkIndex",
                         "Content", embedding, sparse_embedding, metadata,
                         "CreatedAt")
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
                    """,
                    [
                        channel_id,
                        report_id,
                        p["page_number"],
                        p["chunk_index"],
                        p["content"],
                        dense_vec,
                        sparse_vec,
                        json.dumps(metadata),
                    ],
                )
            conn.commit()
        return deleted

    deleted = _retry(_persist, "persist", report_id)

    pages_with_chunks = len({p["page_number"] for p in pending})
    total_ms = int((time.perf_counter() - overall_started) * 1000)

    logger.info(
        "[ingest %s] persisted: deleted=%d inserted=%d pages=%d/%d "
        "extractor=%s latency=%dms",
        report_id, deleted, len(pending),
        pages_with_chunks, pages_total, extractor_tag, total_ms,
    )

    # ── Stage 5: Upload metadata JSON to MinIO ───────────────────────────────
    # Written last so it reflects the actual success state (chunks inserted,
    # latency, content hash). Consumers can use this as the source of truth
    # for "did this report complete?" without round-tripping to Postgres.
    #
    # The first eight fields match the upstream EsplionRAG ingestion
    # pipeline's metadata shape verbatim, so any tooling that already
    # consumes that JSON keeps working. Everything below `use_ocr` is
    # operational signal our richer pipeline produces for free.
    metadata_payload = {
        # ── Reference shape ──
        "document_id":     report_id,
        "channel_id":      channel_id,
        "source_filename": safe_name,
        "content_sha256":  hashlib.sha256(pdf_bytes).hexdigest(),
        "ingested_at":     datetime.now(timezone.utc).isoformat(),
        "markdown_bytes":  len(markdown_blob),
        "original_bytes":  len(pdf_bytes),
        "use_ocr":         options.ocr_fallback,
        # ── Operational extras (doc-processor only) ──
        "extractor":          extractor_tag,
        "language":           language,
        "detected_languages": languages,
        "pages_total":        pages_total,
        "pages_with_chunks":  pages_with_chunks,
        "chunks_inserted":    len(pending),
        "embed_model":        settings.embed_model_id,
        "embed_dim":          (len(dense_vectors[0]) if dense_vectors else 1024),
        "embed_sparse_dim":   settings.embed_sparse_dim,
        "total_latency_ms":   total_ms,
        "minio": {
            "bucket":       minio_cfg.bucket,
            "raw_key":      raw_key,
            "markdown_key": markdown_key,
        },
    }
    metadata_key = storage.object_key(
        minio_cfg, channel_id, "metadata", f"{report_id}.json",
    )
    _retry(
        lambda: storage.put_json(minio_cfg, metadata_key, metadata_payload),
        "minio_upload_metadata", report_id,
    )
    logger.info(
        "[ingest %s] uploaded metadata → s3://%s/%s",
        report_id, minio_cfg.bucket, metadata_key,
    )

    return {
        "channel_id":       channel_id,
        "report_id":        report_id,
        "pages_processed":  pages_with_chunks,
        "chunks_inserted":  len(pending),
        "pages_total":      pages_total,
        "extractor":        extractor_tag,
        "embed_model":      settings.embed_model_id,
        "embed_dim":        (len(dense_vectors[0]) if dense_vectors else 1024),
        "embed_sparse_dim": settings.embed_sparse_dim,
        "total_latency_ms": total_ms,
        "minio_bucket":     minio_cfg.bucket,
        "minio_raw_key":    raw_key,
        "minio_markdown_key": markdown_key,
        "minio_metadata_key": metadata_key,
    }
