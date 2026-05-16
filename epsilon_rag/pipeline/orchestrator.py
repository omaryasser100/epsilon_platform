"""End-to-end page processing — wires Docling, OCR, captioning, and formula
extraction into a single ExtractResponse.

Stage order
===========
1. Decode the PNG and run Docling (layout + reading order + table cells +
   text-layer extraction where present).
2. For each region, decide what to do based on its kind:
       text   → keep Docling's text; if too short, OCR the crop as a fallback.
       table  → render TableFormer cells as GFM markdown.
       figure → caption with Florence-2; OCR labels separately so axis text
                is searchable even when the caption is generic.
       formula → pix2tex on the crop → wrapped LaTeX.
       heading / footer → keep text; never OCR (would just add noise).
3. Concatenate block content in reading order to form the chunk-ready
   `content` string. Tables / figures / formulas appear inline so chunking
   can't strand them away from their surrounding context.
4. Compute page-level metadata (section_title, page_type, language) from
   the structured blocks.

Failure isolation
=================
A single bad block (corrupted image, model OOM) must never fail the whole
page — it gets logged and skipped. Empty pages return a valid response with
content="" so the ai-service can decide whether to retain them.
"""
from __future__ import annotations

import io
import logging
import time
from collections import Counter

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from core.config import settings
from models.schema import (
    Block,
    BoundingBox,
    DocumentMetadata,
    ExtractDocumentResponse,
    ExtractOptions,
    ExtractResponse,
    FigureData,
    FormulaData,
    PageMetadata,
    TableData,
)
from pipeline import arabic_normalize, figures, formulas, layout, ocr, tables

logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────────────

def process_page(
    png_bytes: bytes,
    page_number: int,
    options: ExtractOptions | None = None,
) -> ExtractResponse:
    """Run the full pipeline on a single rendered PDF page.

    The function never raises on individual stage failures; it always returns
    a populated ExtractResponse so the ai-service caller has a response to
    surface or fall back from.
    """
    started = time.perf_counter()
    options = options or ExtractOptions()

    page_img = _decode_png(png_bytes)
    if page_img is None:
        logger.warning("orchestrator: page %d failed to decode", page_number)
        return _empty_response(page_number, started)

    regions = layout.analyze_page(png_bytes)
    if not regions:
        logger.info("orchestrator: page %d had no detected regions", page_number)
        return _empty_response(page_number, started)

    # Soft cap — pathological PDFs can produce hundreds of micro-regions; we
    # process up to N then truncate. Reading order is preserved by the slice.
    if len(regions) > settings.max_blocks_per_page:
        logger.warning(
            "orchestrator: page %d had %d regions, truncating to %d",
            page_number, len(regions), settings.max_blocks_per_page,
        )
        regions = regions[: settings.max_blocks_per_page]

    blocks: list[Block] = []
    for region in regions:
        try:
            block = _process_region(region, page_img, options)
        except Exception as exc:
            logger.warning(
                "orchestrator: region kind=%s order=%d failed: %s",
                region.kind, region.reading_order, exc,
            )
            continue
        if block is not None:
            blocks.append(block)

    content = _build_content(blocks)
    metadata = _build_metadata(blocks)

    return ExtractResponse(
        content=content,
        metadata=metadata,
        blocks=blocks,
        page_number=page_number,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


# ── Region-shape helpers (used to gate expensive GPU calls) ────────────────

def _region_area_ratio(region, page_img: np.ndarray | None) -> float:
    """Region pixel area as a fraction of the page's pixel area.
    Returns 0.0 when the page wasn't rendered (we can't gate further;
    callers should treat 0.0 as "small" if their threshold is positive)."""
    if page_img is None:
        return 0.0
    page_h, page_w = page_img.shape[:2]
    page_area = page_h * page_w
    if page_area <= 0:
        return 0.0
    x_min, y_min, x_max, y_max = region.bbox
    region_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    return region_area / page_area


def _region_worth_ocring(region, page_img: np.ndarray | None) -> bool:
    """Cheap pre-flight before invoking Surya OCR on a region crop.

    Two gates, both ~microseconds:
      1. Area: skip regions smaller than `ocr_min_region_area_ratio` of
         the page. Tiny regions are almost always decorations, not text.
      2. Variance: skip crops with std-dev below `ocr_min_pixel_stddev`.
         A solid-color rectangle has near-zero variance and OCR will
         take 2-3 s only to return "" — a clear waste.

    Skipping these regions on a 100-page report cuts a typical 5-10 min
    OCR-fallback budget by ~30-50%.
    """
    if page_img is None:
        return False
    if _region_area_ratio(region, page_img) < settings.ocr_min_region_area_ratio:
        return False
    x_min, y_min, x_max, y_max = (int(round(v)) for v in region.bbox)
    page_h, page_w = page_img.shape[:2]
    x_min = max(0, min(page_w, x_min))
    x_max = max(0, min(page_w, x_max))
    y_min = max(0, min(page_h, y_min))
    y_max = max(0, min(page_h, y_max))
    if x_max - x_min < 4 or y_max - y_min < 4:
        return False
    crop = page_img[y_min:y_max, x_min:x_max]
    return float(crop.std()) >= settings.ocr_min_pixel_stddev


# ── Per-region dispatch ─────────────────────────────────────────────────────

def _process_region(
    region: layout.LayoutRegion,
    page_img: np.ndarray,
    options: ExtractOptions,
) -> Block | None:
    """Convert one LayoutRegion into a populated Block, or None to drop it."""
    bbox = BoundingBox(
        x_min=region.bbox[0], y_min=region.bbox[1],
        x_max=region.bbox[2], y_max=region.bbox[3],
    )
    base = {
        "type": region.kind,
        "bbox": bbox,
        "reading_order": region.reading_order,
    }

    if region.kind in ("text", "heading", "footer"):
        text = region.text
        # Fall back to OCR only when Docling produced nothing usable AND the
        # caller hasn't disabled OCR. Headings / footers are usually short
        # by design — only OCR them if they're effectively empty.
        #
        # Additional gating beyond ocr_fallback_min_chars (added 2026-05-08
        # to cut bulk-ingest wallclock): skip the OCR call when the region
        # is too small to plausibly contain text, or when the crop is
        # effectively a solid color (decorative border, page margin). Each
        # skipped call saves ~50-300 ms of Surya OCR work.
        if (options.ocr_fallback
                and len(text) < settings.ocr_fallback_min_chars
                and _region_worth_ocring(region, page_img)):
            ocr_text = ocr.ocr_crop(page_img, region.bbox)
            if ocr_text:
                text = ocr_text
        if not text:
            return None
        # Repair PDFs that encode Arabic as visual-order glyph forms.
        # No-op for already-correct text — NFKC is idempotent and the
        # reverse only fires above the Arabic-density threshold.
        text = arabic_normalize.normalize(text)
        return Block(content=text, **base)

    if region.kind == "table" and options.extract_tables:
        if not region.table_data:
            return None
        # Normalize each cell BEFORE building markdown — keeps the table
        # structure intact while fixing any Arabic glyph-form cells.
        normalized_rows = [
            [arabic_normalize.normalize(cell) for cell in row]
            for row in region.table_data
        ]
        markdown = tables.grid_to_markdown(normalized_rows)
        if not markdown:
            return None
        return Block(
            content=markdown,
            table=TableData(
                rows=normalized_rows,
                markdown=markdown,
                n_rows=len(normalized_rows),
                n_cols=len(normalized_rows[0]) if normalized_rows else 0,
            ),
            **base,
        )

    if region.kind == "figure" and options.extract_figures:
        caption = ""
        # Florence-2 captioning is the single most expensive per-region
        # operation (~5-10 s on L4). Only run it on figures that are big
        # enough to plausibly carry information; tiny figures (logos,
        # decorative ornaments) get a "[Figure]" stub.
        if (options.figure_captioning
                and _region_area_ratio(region, page_img)
                    >= settings.figure_caption_min_area_ratio):
            caption = figures.caption_crop(page_img, region.bbox)
        # Pull any axis labels / legend text from the figure crop.
        # This is independent of captioning — even when Florence-2 is off
        # we still want searchable text for legends. Same area-and-content
        # gating as the text-block fallback so we don't OCR tiny logos.
        if _region_worth_ocring(region, page_img):
            extracted = ocr.ocr_crop(page_img, region.bbox)
        else:
            extracted = ""
        # Florence-2 captions are English so NFKC is a no-op; OCR'd legends
        # may be Arabic glyphs and need the same fix as text blocks.
        extracted = arabic_normalize.normalize(extracted)
        # Compose chunk-friendly inline marker so retrieval can recover
        # figure context even after embedding.
        if caption and extracted:
            content = f"[Figure: {caption}]\n{extracted}"
        elif caption:
            content = f"[Figure: {caption}]"
        elif extracted:
            content = f"[Figure]\n{extracted}"
        else:
            content = "[Figure]"
        return Block(
            content=content,
            figure=FigureData(caption=caption, extracted_text=extracted),
            **base,
        )

    if region.kind == "formula" and options.extract_formulas:
        latex = formulas.to_latex_crop(page_img, region.bbox)
        if not latex:
            return None
        if len(latex) > settings.max_formula_chars:
            latex = latex[: settings.max_formula_chars]
        return Block(
            content=f"$$ {latex} $$",
            formula=FormulaData(latex=latex),
            **base,
        )

    # Unknown / disabled kind — skip.
    return None


# ── Page-level synthesis ────────────────────────────────────────────────────

def _build_content(blocks: list[Block]) -> str:
    """Join block content into a single chunk-ready string.

    Reading order is already on each block; we trust it so the output flows
    naturally even on multi-column or RTL pages.
    """
    parts = [b.content for b in blocks if b.content]
    return "\n\n".join(parts).strip()


def _build_metadata(blocks: list[Block]) -> PageMetadata:
    """Derive page-level metadata from the blocks.

    section_title — first heading block, if any.
    page_type     — heuristic from block-type distribution.
    language      — dominant script in the text content (Arabic vs Latin).
    """
    section_title = next(
        (b.content for b in blocks if b.type == "heading" and b.content),
        "",
    )

    page_type = _classify_page_type(blocks)

    language = _detect_language(_build_content(blocks))

    tables_data   = [b.table for b in blocks if b.table]
    figures_data  = [b.figure for b in blocks if b.figure]
    formulas_data = [b.formula for b in blocks if b.formula]

    reading_order = [b.reading_order for b in blocks]

    return PageMetadata(
        section_title=section_title.strip()[:300],   # cap to avoid pathological titles
        page_type=page_type,
        language=language,
        tables=tables_data,
        figures=figures_data,
        formulas=formulas_data,
        reading_order=reading_order,
    )


def _classify_page_type(blocks: list[Block]) -> str:
    """Pick a single page_type label from the block-type distribution.

    The categories match ai-service's existing schema so swapping the
    extractor doesn't change downstream filtering.
    """
    if not blocks:
        return "empty"

    counts = Counter(b.type for b in blocks)
    n_total = sum(counts.values())

    # Specialised pages first.
    if any(b.type == "heading" and "table of contents" in (b.content or "").lower()
           for b in blocks):
        return "toc"

    if counts.get("table", 0) and counts["table"] / n_total > 0.5:
        return "table"
    if counts.get("figure", 0) and counts["figure"] / n_total > 0.5:
        return "chart"

    # Cover pages tend to be sparse with a dominant heading and very little
    # body text. ~3 blocks total, mostly headings → cover.
    if n_total <= 3 and counts.get("heading", 0) >= 1 and counts.get("text", 0) <= 1:
        return "cover"

    if counts.get("text", 0) == n_total:
        return "text"
    return "mixed"


def _detect_language(text: str) -> str:
    """Cheap language classifier — counts Arabic vs Latin characters.

    Doesn't need to be perfect (it's metadata for retrieval filtering) and
    we deliberately skip a heavyweight language-detection library to keep
    the cold-start surface small.
    """
    if not text:
        return "mixed"

    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = arabic + latin
    if total == 0:
        return "mixed"

    ar_ratio = arabic / total
    if ar_ratio > 0.7:
        return "ar"
    if ar_ratio < 0.1:
        return "en"
    return "mixed"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _decode_png(png_bytes: bytes) -> np.ndarray | None:
    try:
        return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    except Exception as exc:
        logger.warning("orchestrator: failed to decode page PNG: %s", exc)
        return None


def _empty_response(page_number: int, started: float) -> ExtractResponse:
    return ExtractResponse(
        content="",
        metadata=PageMetadata(),
        blocks=[],
        page_number=page_number,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


# ── Full-PDF processing ─────────────────────────────────────────────────────
# This is the preferred entry point when the caller has the original PDF.
# Docling sees the real text layer (no rasterisation loss for digital PDFs),
# resolves reading order across pages, and amortises layout-model overhead
# over the whole document.
#
# Pages are rendered to numpy arrays *only when needed* — i.e. when a page
# contains figures, formulas, or text regions Docling couldn't extract.
# Pure-text pages with a clean text layer skip rasterisation entirely, which
# is the headline win of this path over the per-page PNG mode.

# Page rendering settings — match ai-service's existing PDF-to-PNG DPI so
# pixel coordinates from Docling align with what the per-page entry point
# uses for OCR / caption crops.
_RENDER_DPI = 150
_RENDER_MATRIX = fitz.Matrix(_RENDER_DPI / 72, _RENDER_DPI / 72)


def process_document(
    pdf_bytes: bytes,
    options: ExtractOptions | None = None,
    page_range: tuple[int, int] | None = None,
) -> ExtractDocumentResponse:
    """Run the full pipeline on a complete PDF.

    The function never raises on individual page failures; pages that error
    out come back with empty content so callers can choose whether to retain
    or skip them. An entirely unparseable PDF returns an empty `pages` list
    plus zeroed document_metadata.
    """
    started = time.perf_counter()
    options = options or ExtractOptions()

    layout_started = time.perf_counter()
    doc_layout = layout.analyze_document(pdf_bytes)
    pages_regions = doc_layout.regions_by_page
    document_markdown = doc_layout.markdown
    layout_ms = int((time.perf_counter() - layout_started) * 1000)
    logger.info(
        "orchestrator: layout done in %d ms — %d pages with regions, markdown=%d chars",
        layout_ms, len(pages_regions), len(document_markdown),
    )
    if page_range is not None:
        lo, hi = page_range
        pages_regions = {
            p: r for p, r in pages_regions.items() if lo <= p <= hi
        }

    if not pages_regions:
        return _empty_document_response(started)

    # PyMuPDF document handle for lazy per-page rasterisation.
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[ExtractResponse] = []
    pages_to_process = sorted(pages_regions.keys())
    total_pages = len(pages_to_process)
    # Progress log every N pages OR every 30 s — whichever comes first.
    # Without this the post-layout per-page loop (OCR fallback + Florence-2
    # captions on figures) looks like a silent 5-15 min stall.
    _PROGRESS_EVERY_N_PAGES = 10
    _PROGRESS_EVERY_SECONDS = 30.0
    last_progress_at = time.perf_counter()
    # Cumulative time spent in each per-region stage — surfaced once at the
    # end so we can tell which model is the bottleneck without reading every
    # per-page log line.
    page_loop_started = time.perf_counter()
    try:
        for idx, page_no in enumerate(pages_to_process, start=1):
            page_started = time.perf_counter()
            regions = pages_regions[page_no]

            if len(regions) > settings.max_blocks_per_page:
                logger.warning(
                    "orchestrator: page %d had %d regions, truncating to %d",
                    page_no, len(regions), settings.max_blocks_per_page,
                )
                regions = regions[: settings.max_blocks_per_page]

            page_img = _render_page_if_needed(
                pdf_doc, page_no, regions, options,
            )

            blocks: list[Block] = []
            for region in regions:
                try:
                    block = _process_region(region, page_img, options)
                except Exception as exc:
                    logger.warning(
                        "orchestrator: page=%d region kind=%s order=%d failed: %s",
                        page_no, region.kind, region.reading_order, exc,
                    )
                    continue
                if block is not None:
                    blocks.append(block)

            page_latency_ms = int((time.perf_counter() - page_started) * 1000)
            pages.append(ExtractResponse(
                content=_build_content(blocks),
                metadata=_build_metadata(blocks),
                blocks=blocks,
                page_number=page_no,
                latency_ms=page_latency_ms,
            ))

            # Slow-page warning so a single bad page (lots of OCR fallback,
            # heavy Florence-2 captioning) is visible in logs even when
            # progress reporting hasn't ticked yet.
            if page_latency_ms > 30_000:
                logger.warning(
                    "orchestrator: page=%d slow — %d ms (regions=%d)",
                    page_no, page_latency_ms, len(regions),
                )

            now = time.perf_counter()
            since_last = now - last_progress_at
            if (idx % _PROGRESS_EVERY_N_PAGES == 0) or since_last >= _PROGRESS_EVERY_SECONDS:
                elapsed = now - page_loop_started
                rate = idx / elapsed if elapsed > 0 else 0.0
                remaining = (total_pages - idx) / rate if rate > 0 else float("inf")
                logger.info(
                    "orchestrator: progress %d/%d pages (%.1fs elapsed, ~%.0fs remaining, %.2f pages/s)",
                    idx, total_pages, elapsed, remaining, rate,
                )
                last_progress_at = now
    finally:
        pdf_doc.close()
    page_loop_ms = int((time.perf_counter() - page_loop_started) * 1000)
    logger.info(
        "orchestrator: per-page loop done — %d/%d pages in %d ms (%.1f s)",
        len(pages), total_pages, page_loop_ms, page_loop_ms / 1000,
    )

    document_metadata = DocumentMetadata(
        page_count=len(pages),
        detected_languages=sorted({p.metadata.language for p in pages}),
        has_tables  = any(p.metadata.tables  for p in pages),
        has_figures = any(p.metadata.figures for p in pages),
        has_formulas= any(p.metadata.formulas for p in pages),
    )

    return ExtractDocumentResponse(
        pages=pages,
        document_metadata=document_metadata,
        total_latency_ms=int((time.perf_counter() - started) * 1000),
        markdown=document_markdown,
    )


# ── Helpers (full-PDF) ──────────────────────────────────────────────────────

def _render_page_if_needed(
    pdf_doc,
    page_no: int,
    regions: list,
    options: ExtractOptions,
) -> np.ndarray | None:
    """Render the page only if any region needs pixel access.

    Decision rule:
      • Always render when the page has figures or formulas (the captioner
        and pix2tex both need the crop).
      • Render when OCR fallback is enabled AND there's a text/heading/footer
        region whose text came back empty (Docling lost the text layer for
        that block — usually a scanned region).
      • Otherwise skip — the page is fully resolvable from Docling's output,
        and rasterising would just burn CPU.

    This is the headline efficiency win of full-PDF mode for digital PDFs.
    """
    needs_image = False
    for r in regions:
        if r.kind in ("figure", "formula"):
            needs_image = True
            break
        if (
            r.kind in ("text", "heading", "footer")
            and options.ocr_fallback
            and len(r.text) < settings.ocr_fallback_min_chars
        ):
            needs_image = True
            break

    if not needs_image:
        return None

    zero_based = page_no - 1
    if zero_based < 0 or zero_based >= len(pdf_doc):
        logger.warning(
            "orchestrator: page %d out of range (PDF has %d pages)",
            page_no, len(pdf_doc),
        )
        return None

    try:
        pix = pdf_doc[zero_based].get_pixmap(
            matrix=_RENDER_MATRIX, colorspace=fitz.csRGB,
        )
        # pix.samples is a flat RGB byte buffer; reshape to (H, W, 3).
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3,
        )
    except Exception as exc:
        logger.warning(
            "orchestrator: failed to render page %d: %s", page_no, exc,
        )
        return None


def _empty_document_response(started: float) -> ExtractDocumentResponse:
    return ExtractDocumentResponse(
        pages=[],
        document_metadata=DocumentMetadata(
            page_count=0,
            detected_languages=[],
            has_tables=False,
            has_figures=False,
            has_formulas=False,
        ),
        total_latency_ms=int((time.perf_counter() - started) * 1000),
    )
