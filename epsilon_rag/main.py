"""CLI entrypoint — ingest one local PDF (or a folder of PDFs) into a channel.

Loads all pipeline models, reads each PDF off disk, runs extract → chunk
→ embed → persist, and prints stats per file. Every ingest is scoped to
a channel: the channel must already exist (create one with
`python channels.py create "<name>"`), and the resulting chunks carry
both the channel UUID and a report UUID so retrieval can filter at the
SQL level.

Usage
=====
    # Single PDF
    python main.py path/to/document.pdf --channel-id <uuid>
    python main.py path/to/document.pdf --channel-id <uuid> --title "Q4 Financials"
    python main.py path/to/document.pdf --channel-id <uuid> \\
                                        --meta author=Smith --meta year=2024

    # Batch: ingest every PDF in a folder
    python main.py path/to/folder --channel-id <uuid>
    python main.py path/to/folder --channel-id <uuid> --recursive

Stages
======
  1. Warm up Docling, RapidOCR, formula OCR, figures, bge-m3 embedder
     (dense + sparse), bge-reranker-v2-m3 reranker. Loaded ONCE per
     CLI invocation, even in batch mode.
  2. Open the PostgreSQL connection pool.
  3. For each PDF: resolve / upsert the report row, then run
     [pipeline.ingest.run_ingest_from_path](pipeline/ingest.py).
  4. Drain the pool and print stats.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

from core.config import settings
from core.db import close_pool, init_pool
from models.schema import IngestOptions
from pipeline import embeddings, figures, formulas, layout, ocr, reranker, registry
from pipeline.ingest import run_ingest_from_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")


# ── Warmup ──────────────────────────────────────────────────────────────────

def _warmup() -> None:
    """Load every model up-front so the first ingest call doesn't pay
    model-load latency. A failed load is logged and the corresponding
    pipeline stage degrades — the ingest will surface that as an error
    when it reaches the stage."""
    started = time.perf_counter()
    for name, fn in [
        ("docling",     layout.init),
        ("rapidocr",    ocr.init),
        ("pix2tex",     formulas.init),
        ("figures",     figures.init),
        ("embeddings",  embeddings.init),
        ("reranker",    reranker.init),
    ]:
        stage_started = time.perf_counter()
        try:
            fn()
            logger.info(
                "warmup: %s ready in %.1fs",
                name, time.perf_counter() - stage_started,
            )
        except Exception as exc:
            logger.exception("warmup: %s failed to initialise: %s", name, exc)
    logger.info("warmup: total %.1fs", time.perf_counter() - started)


# ── CLI plumbing ────────────────────────────────────────────────────────────

def _parse_meta(items: list[str]) -> dict[str, str]:
    """Parse `--meta key=value` repeats into a flat dict.

    Values stay as strings — the jsonb column happily takes them, and
    keeping it stringly-typed avoids surprises around int / float / date
    coercion. Callers who want richer types can edit the row directly.
    """
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"--meta value must be key=value, got: {raw}")
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise SystemExit(f"--meta key must be non-empty: {raw}")
        out[key] = value
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a local PDF (or a folder of PDFs) into a channel.",
    )
    parser.add_argument(
        "path", type=Path,
        help="Path to a PDF file or a folder containing PDF files.",
    )
    parser.add_argument(
        "--channel-id", type=str, required=True,
        help="UUID of the channel these PDFs belong to. Create one first "
             "with `python channels.py create \"<name>\"`.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="When `path` is a folder, walk subdirectories too. Ignored "
             "for a single-file path.",
    )
    parser.add_argument(
        "--title", type=str, default="",
        help="Title for the report row. Only honoured for single-file "
             "ingests; in batch mode each report's title defaults to its "
             "filename so they stay distinguishable.",
    )
    parser.add_argument(
        "--meta", action="append", default=[],
        metavar="KEY=VALUE",
        help="Free-form metadata key=value pair applied to every report "
             "row ingested in this run. Repeat for multiple entries.",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="In batch mode, abort the run on the first failed PDF. By "
             "default failures are logged and the next file is attempted.",
    )
    parser.add_argument(
        "--no-tables", action="store_true",
        help="Skip table extraction (table regions emit no block).",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure regions entirely.",
    )
    parser.add_argument(
        "--no-formulas", action="store_true",
        help="Skip formula extraction (pix2tex is disabled anyway).",
    )
    parser.add_argument(
        "--no-ocr", action="store_true",
        help="Disable RapidOCR fallback on regions without a usable text layer.",
    )
    parser.add_argument(
        "--no-captions", action="store_true",
        help="Skip BLIP figure captioning (figures still get [Figure] "
             "stubs + OCR'd legends).",
    )
    return parser.parse_args(argv)


def _validate_uuid(value: str, label: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise SystemExit(f"{label} is not a valid UUID: {exc}")


def _print_stats(stats: dict, label: str = "") -> None:
    """One-screen summary of what just got persisted. `label` (when set)
    appears in the header — used in batch mode to disambiguate output."""
    print()
    print("─" * 60)
    if label:
        print(f"  {label}")
        print("─" * 60)
    print(f"  Channel ID       {stats['channel_id']}")
    print(f"  Report ID        {stats['report_id']}")
    print(f"  Pages processed  {stats['pages_processed']} / {stats['pages_total']}")
    print(f"  Chunks inserted  {stats['chunks_inserted']}")
    print(f"  Extractor        {stats['extractor']}")
    print(f"  Embed model      {stats['embed_model']}")
    print(f"  Embed dim        {stats['embed_dim']} (dense) + {stats['embed_sparse_dim']} (sparse)")
    print(f"  Total latency    {stats['total_latency_ms']} ms")
    print("─" * 60)


def _resolve_pdfs(path: Path, recursive: bool) -> list[Path]:
    """Return the list of PDFs to ingest for `path`. Single file → one-
    element list; folder → every .pdf inside (recursive if requested)."""
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise SystemExit(f"error: not a PDF: {path}")
        return [path]
    if path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        # Case-insensitive: collect .pdf and .PDF, dedupe, sort. Filename
        # case can differ between OSes, and dropping mixed-case extensions
        # silently would surprise users on Linux.
        files = sorted({
            *path.glob(pattern),
            *path.glob(pattern.replace(".pdf", ".PDF")),
        })
        if not files:
            raise SystemExit(f"error: no PDFs found in {path}")
        return list(files)
    raise SystemExit(f"error: not a file or directory: {path}")


def _ingest_one(
    pdf: Path,
    channel_id: str,
    title: str,
    metadata: dict[str, str],
    options: IngestOptions,
) -> dict:
    """Resolve / upsert the report row, then run the pipeline. Pulled
    out of main() so the batch loop can call it per file with shared
    warmup + DB pool."""
    report_id = registry.upsert_report(
        channel_id=channel_id,
        filename=pdf.name,
        title=title or pdf.stem,
        metadata=metadata,
    )
    logger.info("ingesting %s as report_id=%s", pdf, report_id)
    return run_ingest_from_path(channel_id, report_id, pdf, options)


# ── Entry point ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    channel_id = _validate_uuid(args.channel_id, "--channel-id")
    metadata = _parse_meta(args.meta)

    options = IngestOptions(
        extract_tables=not args.no_tables,
        extract_figures=not args.no_figures,
        extract_formulas=not args.no_formulas,
        ocr_fallback=not args.no_ocr,
        # BLIP-based figure captioning is on by default (see
        # pipeline/figures.py). Pass --no-captions to skip it; the
        # orchestrator's area gate already skips tiny figures.
        figure_captioning=not args.no_captions,
    )

    if not settings.database_url:
        print(
            "error: DATABASE_URL is not configured — set it in .env or the "
            "environment before running the CLI.",
            file=sys.stderr,
        )
        return 2

    try:
        pdfs = _resolve_pdfs(args.path, args.recursive)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    is_batch = len(pdfs) > 1
    # In batch mode every report needs a distinct title — using args.title
    # for all of them would lose the per-PDF distinction. Honour it only
    # in single-file mode.
    title_for = (lambda _p: args.title) if not is_batch else (lambda p: p.stem)

    _warmup()
    init_pool()

    overall_started = time.perf_counter()
    succeeded: list[Path] = []
    failed: list[tuple[Path, str]] = []

    try:
        channel = registry.get_channel(channel_id)
        if channel is None:
            print(
                f"error: channel {channel_id} does not exist. "
                "Create one with `python channels.py create \"<name>\"`.",
                file=sys.stderr,
            )
            return 2
        logger.info(
            "resolved channel %s (%s) — %d PDF(s) to ingest",
            channel["name"], channel_id, len(pdfs),
        )

        for idx, pdf in enumerate(pdfs, start=1):
            prefix = f"[{idx}/{len(pdfs)}] {pdf.name}" if is_batch else pdf.name
            try:
                stats = _ingest_one(
                    pdf, channel_id, title_for(pdf), metadata, options,
                )
            except FileNotFoundError as exc:
                logger.error("%s: %s", prefix, exc)
                failed.append((pdf, str(exc)))
                if args.stop_on_error:
                    break
                continue
            except Exception as exc:
                logger.exception("%s: ingest failed: %s", prefix, exc)
                failed.append((pdf, str(exc)))
                if args.stop_on_error:
                    break
                continue
            succeeded.append(pdf)
            _print_stats(stats, label=prefix if is_batch else "")
    finally:
        close_pool()

    if is_batch:
        elapsed = time.perf_counter() - overall_started
        print()
        print("═" * 60)
        print(f"  Batch summary — {elapsed:.1f}s total")
        print(f"  Succeeded: {len(succeeded)} / {len(pdfs)}")
        if failed:
            print(f"  Failed:    {len(failed)}")
            for pdf, err in failed:
                print(f"    · {pdf.name} — {err}")
        print("═" * 60)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
