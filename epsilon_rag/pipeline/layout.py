"""Docling wrapper — layout detection + reading order + table structure.

Docling is the spine of this pipeline. A single `convert()` call gives us:
    • layout regions    (DocLayNet-trained model: text / table / figure / formula)
    • reading order     (deterministic ordering across multi-column layouts)
    • table structure   (TableFormer cell extraction)
    • text extraction   (PyMuPDF for digital PDFs, no OCR involvement)

We do NOT use Docling for OCR — its built-in OCR backend is slower and
doesn't have an Arabic-tuned model. We post-process Docling's output
and run RapidOCR in `pipeline.ocr` on regions that came back empty.

Two entry points
================
`analyze_page(png_bytes)`     — single rendered page (kept for fallback /
                                small-image inputs).
`analyze_document(pdf_bytes)` — full PDF, preferred path. Avoids rasterising
                                the text layer, gives reading order across
                                page breaks, and runs Docling once per
                                document instead of once per page.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Iterable

import fitz  # PyMuPDF
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import CoordOrigin

from core.config import settings
from pipeline.errors import InvalidPdfError


# Must stay in sync with orchestrator._RENDER_DPI — see _resolve_bbox below
# for why. Duplicated rather than imported to avoid a layout↔orchestrator
# import cycle; if either side ever changes its DPI, change both.
_RENDER_DPI = 150
_POINTS_TO_PIXELS = _RENDER_DPI / 72.0

logger = logging.getLogger(__name__)

# Minimum plausible PDF size — anything below this is either empty or a
# truncated upload. Real PDFs are >1 KB; 32 B is just a sanity floor.
_MIN_PDF_BYTES = 32


# ── Output dataclasses ──────────────────────────────────────────────────────
# We don't return Docling's native types because they're an unstable API
# surface — pinning to ours protects the orchestrator from breaking on
# Docling minor version bumps.

@dataclass
class LayoutRegion:
    """A single detected region on the page.

    bbox is in **top-left pixel space at _RENDER_DPI** so the
    orchestrator can slice its rasterised `page_img` numpy array
    directly. See `_resolve_bbox` for the unit / origin normalisation
    that enforces this contract.
    """
    kind: str                  # "text" | "table" | "figure" | "formula" | "heading" | "footer"
    bbox: tuple[float, float, float, float]   # (x_min, y_min, x_max, y_max), pixel space @ _RENDER_DPI
    text: str                  # raw text Docling extracted; "" if Docling has no text layer here
    reading_order: int         # 0-based ordinal within its page
    table_data: list[list[str]] | None = None  # row-major, only for kind="table"


@dataclass
class DocumentLayout:
    """Whole-PDF Docling output. Returned by `analyze_document()`.

    Bundling regions + markdown means callers don't have to run Docling
    twice to get both — the markdown is exported off the same parsed
    document tree that produced the regions.
    """
    regions_by_page: dict[int, list[LayoutRegion]]   # {page_no: [regions...]}
    markdown: str                                    # Docling's `export_to_markdown()` for the whole doc


# Mapping from Docling item class names to the small fixed kind vocabulary
# the orchestrator uses. Centralised so both entry points stay consistent.
_KIND_MAP = {
    "TextItem":          "text",
    "SectionHeaderItem": "heading",
    "TitleItem":         "heading",
    "TableItem":         "table",
    "PictureItem":       "figure",
    "FormulaItem":       "formula",
    "ListItem":          "text",
    "CodeItem":          "text",
    "FootnoteItem":      "footer",
}


# ── Module state ────────────────────────────────────────────────────────────
# Docling's converter is heavy to instantiate (loads layout + table models).
# Build once at process startup; reuse across requests.
_converter: DocumentConverter | None = None


def _build_converter() -> DocumentConverter:
    """Configure Docling for image-as-page input.

    do_ocr=False because we run our own Arabic-aware OCR downstream;
    do_table_structure=True turns on TableFormer (cell-level table extraction).

    accelerator_options pins Docling's layout + table models to the same
    CUDA device the rest of the pipeline uses. Without this Docling auto-
    detects the device and has been observed to fall back to CPU silently
    in some image variants — which inflates layout from ~150 ms/page to
    ~1.5 s/page. Pinning explicitly removes that variance.
    """
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False
    pipeline_opts.do_table_structure = True
    pipeline_opts.table_structure_options.do_cell_matching = True

    # Pin accelerator to CUDA so Docling's layout + table models run on
    # the GPU instead of silently falling back to CPU on some image
    # variants. num_threads=4 is for CPU-side preprocessing (PDF →
    # raster) and matches the Cloud Run --cpu=4 we deploy with.
    device = (
        AcceleratorDevice.CUDA
        if settings.device.lower() == "cuda"
        else AcceleratorDevice.CPU
    )
    pipeline_opts.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=device,
    )
    logger.info(
        "docling: pinned accelerator device=%s num_threads=4", device,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
        }
    )


def init() -> None:
    """Eagerly build the Docling converter so the first request doesn't
    pay layout-model initialisation latency on top of cold-start."""
    global _converter
    if _converter is None:
        logger.info("docling: initialising converter")
        _converter = _build_converter()
        logger.info("docling: ready")


def is_ready() -> bool:
    return _converter is not None


# ── Public API: per-page (rendered PNG) ─────────────────────────────────────

def analyze_page(png_bytes: bytes) -> list[LayoutRegion]:
    """Run Docling on a single page rendered as PNG.

    The PNG is wrapped in a synthetic single-page PDF so Docling's document
    API can ingest it. Returns regions in reading order; an empty list means
    Docling produced no detected blocks (very blank or weird PDF).

    Prefer `analyze_document()` when you have the original PDF — it avoids
    the synthetic-PDF wrapping AND gives Docling access to the real PDF text
    layer (no rasterisation loss for digital PDFs).
    """
    pdf_bytes = _png_to_single_page_pdf(png_bytes)
    doc_layout = analyze_document(pdf_bytes)
    if not doc_layout.regions_by_page:
        return []
    # The synthetic PDF has exactly one page. Whichever index Docling gives
    # us, just return the only bucket.
    return next(iter(doc_layout.regions_by_page.values()))


# ── Public API: full PDF ────────────────────────────────────────────────────

def analyze_document(pdf_bytes: bytes) -> DocumentLayout:
    """Run Docling on a full PDF.

    Returns a `DocumentLayout` with:
      • `regions_by_page` — `{page_no: [LayoutRegion, ...]}` with 1-based
        page numbers. Each page's regions are ordered by Docling's global
        reading-order traversal (so multi-column layouts and Arabic
        right-to-left columns flow correctly within their page).
      • `markdown` — Docling's `export_to_markdown()` for the whole
        document, captured off the same parsed tree so we don't pay for
        a second conversion. Empty string if Docling produced no document.

    An unparseable PDF returns an empty layout (empty dict + empty markdown)
    — callers should treat that the same as "no content extracted" rather
    than a hard error, so a single bad document doesn't crash batch ingest.
    """
    if _converter is None:
        init()

    # Pre-validate before handing to Docling. Docling's own failure mode
    # is a generic RuntimeError("... is not valid.") that doesn't tell you
    # WHY (empty? encrypted? wrong format?). Catching the obvious cases
    # here gives ai_jobs.error_message useful detail.
    if len(pdf_bytes) < _MIN_PDF_BYTES:
        raise InvalidPdfError(
            "PDF too small (likely empty or truncated upload)",
            size=len(pdf_bytes), header=pdf_bytes[:8],
        )
    if not pdf_bytes.startswith(b"%PDF-"):
        raise InvalidPdfError(
            "Not a PDF (missing %PDF- header — wrong content-type or HTML error page?)",
            size=len(pdf_bytes), header=pdf_bytes[:8],
        )

    from docling.datamodel.base_models import DocumentStream

    stream = DocumentStream(name="document.pdf", stream=io.BytesIO(pdf_bytes))
    try:
        result = _converter.convert(stream)
    except RuntimeError as exc:
        # Docling's PDF backend (PyPdfium2) flips in_doc.valid=False on
        # encrypted, corrupted, or otherwise-unreadable PDFs and surfaces
        # it as a plain RuntimeError. Re-raise as our typed error so the
        # retry layer knows not to retry.
        if "is not valid" in str(exc):
            raise InvalidPdfError(
                f"Docling rejected PDF (likely encrypted or corrupted): {exc}",
                size=len(pdf_bytes), header=pdf_bytes[:8],
            ) from exc
        raise

    if result.document is None:
        logger.warning("docling: convert() returned no document")
        return DocumentLayout(regions_by_page={}, markdown="")

    regions_by_page = _bucket_regions_by_page(result.document)

    # `export_to_markdown()` walks the same parsed tree; cheap relative to
    # the layout / table-structure model passes. Belt-and-braces fallback
    # to empty string keeps ingest robust to docling-version surprises.
    try:
        markdown = result.document.export_to_markdown() or ""
    except Exception as exc:
        logger.warning("docling: export_to_markdown() failed: %s", exc)
        markdown = ""

    return DocumentLayout(regions_by_page=regions_by_page, markdown=markdown)


# ── Region extraction (shared) ──────────────────────────────────────────────

def _bucket_regions_by_page(doc) -> dict[int, list[LayoutRegion]]:
    """Walk Docling's items, classify them, and bucket by page number.

    Docling's `iterate_items()` yields items in global reading order. We
    keep a per-page counter so each region carries a stable ordinal within
    its own page — the orchestrator joins on that to flow content
    page-by-page even when callers process pages independently.

    `page_heights_pt` is pre-built so `_resolve_bbox` can convert any
    BOTTOMLEFT-origin Docling bbox to TOPLEFT pixel coordinates (the
    contract `LayoutRegion.bbox` has always claimed but the previous
    implementation drifted from — see _resolve_bbox docstring).
    """
    by_page: dict[int, list[LayoutRegion]] = {}
    page_orders: dict[int, int] = {}

    # Pre-build {page_no: page_height_pt} once per document so we don't
    # touch doc.pages on every item. Docling pages dict keys are 1-based
    # ints matching prov.page_no on items.
    page_heights_pt: dict[int, float] = {}
    try:
        for page_no, page in (doc.pages or {}).items():
            size = getattr(page, "size", None)
            if size is not None:
                page_heights_pt[int(page_no)] = float(size.height)
    except Exception as exc:
        # Don't fail the whole document over a malformed pages dict —
        # _resolve_bbox falls back to no-op origin handling when the
        # height is missing.
        logger.warning("docling: failed to read page heights: %s", exc)

    for item, _level in doc.iterate_items():
        cls_name = type(item).__name__
        kind = _KIND_MAP.get(cls_name)
        if kind is None:
            continue   # unknown / paginated container — skip rather than guess

        page_no = _resolve_page_no(item)
        if page_no is None:
            continue

        page_height_pt = page_heights_pt.get(page_no)
        bbox = _resolve_bbox(item, page_height_pt)
        text = _resolve_text(item, kind)
        table_data = _resolve_table(item) if kind == "table" else None

        order_idx = page_orders.get(page_no, 0)
        page_orders[page_no] = order_idx + 1

        by_page.setdefault(page_no, []).append(LayoutRegion(
            kind=kind,
            bbox=bbox,
            text=text,
            reading_order=order_idx,
            table_data=table_data,
        ))
    return by_page


# ── Helpers ─────────────────────────────────────────────────────────────────

def _png_to_single_page_pdf(png_bytes: bytes) -> bytes:
    """Wrap a PNG into a one-page PDF so Docling can ingest it.

    Used only by the per-page entry point. PyMuPDF handles this in-memory
    — no temp files. The wrapping is lossy for digital PDFs (because we
    started from a 150-dpi raster, not the source PDF), which is exactly
    why `analyze_document()` is the preferred path.
    """
    img_doc = fitz.open(stream=png_bytes, filetype="png")
    pix = img_doc[0].get_pixmap()
    width, height = pix.width, pix.height

    pdf = fitz.open()
    page = pdf.new_page(width=width, height=height)
    page.insert_image(fitz.Rect(0, 0, width, height), stream=png_bytes)
    pdf_bytes = pdf.tobytes()
    pdf.close()
    img_doc.close()
    return pdf_bytes


def _resolve_page_no(item) -> int | None:
    """Read the 1-based page number off a Docling item's provenance."""
    try:
        prov = item.prov[0] if getattr(item, "prov", None) else None
        if prov is not None and getattr(prov, "page_no", None) is not None:
            return int(prov.page_no)
    except Exception:
        return None
    return None


def _resolve_bbox(
    item,
    page_height_pt: float | None = None,
) -> tuple[float, float, float, float]:
    """Best-effort bbox extraction in **top-left pixel coordinates at
    _RENDER_DPI**, matching what the orchestrator slices `page_img` with.

    Two mismatches the previous implementation didn't handle:

    1. **Unit**: Docling reports bboxes in PDF points (72 / inch). The
       orchestrator rasterises pages at _RENDER_DPI=150 and indexes the
       numpy pixel array directly. A whole-page A4 bbox of (0,0,595,842)
       sliced into a (~1754, 1240, 3) ndarray picks up only the top-left
       ~48% — the rest of the figure (and any OCR text on it) is
       silently dropped.

    2. **Y-axis origin**: Docling bboxes may carry CoordOrigin.BOTTOMLEFT
       (PDF native). In that case `b.t` (top) numerically *exceeds*
       `b.b` (bottom) because y grows upward from the page bottom.
       `page_img[t:b]` becomes a zero-row slice; the downstream OCR /
       caption model receives a 0-height image, returns "" without
       raising, and the orchestrator emits a "[Figure]" stub.

    Both bugs disappear when we (a) call Docling's own
    `to_top_left_origin(page_height_pt)` for BOTTOMLEFT bboxes, (b)
    scale points → pixels by _POINTS_TO_PIXELS, (c) defensively reorder
    so x0 ≤ x1 and y0 ≤ y1 (some PDFs emit transposed metadata).

    `page_height_pt` is required to invert BOTTOMLEFT origins; when it
    isn't available (very rare) we skip the origin conversion and
    accept that BOTTOMLEFT bboxes from those pages may still slice
    empty. The crop guard in orchestrator/_process_region's callees
    catches that path.
    """
    try:
        prov = item.prov[0] if getattr(item, "prov", None) else None
        if not (prov and getattr(prov, "bbox", None)):
            return (0.0, 0.0, 0.0, 0.0)

        b = prov.bbox

        # Origin normalisation — only Docling knows its own conventions,
        # so prefer its own conversion over hand-rolling y = H - y.
        origin = getattr(b, "coord_origin", None)
        if origin == CoordOrigin.BOTTOMLEFT and page_height_pt is not None:
            try:
                b = b.to_top_left_origin(page_height_pt)
            except Exception as exc:
                logger.warning(
                    "docling: to_top_left_origin failed (page_height=%s): %s",
                    page_height_pt, exc,
                )

        # Point → pixel.
        x0 = float(b.l) * _POINTS_TO_PIXELS
        y0 = float(b.t) * _POINTS_TO_PIXELS
        x1 = float(b.r) * _POINTS_TO_PIXELS
        y1 = float(b.b) * _POINTS_TO_PIXELS

        # Defensive reorder. Mostly catches odd PDFs where the producer
        # wrote a transposed bbox, but also rescues any residual origin
        # surprise we didn't normalise above.
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        return (x0, y0, x1, y1)
    except Exception as exc:
        logger.warning("docling: bbox resolution failed: %s", exc)
        return (0.0, 0.0, 0.0, 0.0)


def _resolve_text(item, kind: str) -> str:
    """Return the textual content for the item, or '' if Docling didn't
    extract any. For tables we deliberately return '' — the orchestrator
    formats them via `pipeline.tables`."""
    if kind == "table":
        return ""
    text = getattr(item, "text", None) or ""
    return text.strip()


def _resolve_table(item) -> list[list[str]] | None:
    """Pull TableFormer's cell grid into a row-major 2-D list of strings.

    Docling exposes a `data` field on TableItem with row/col indices on each
    cell. We bucket them by row, sort each row by column index, and emit the
    raw text. Merged cells repeat their content into each spanned slot — that
    keeps downstream Markdown rendering trivial at the cost of slight
    duplication, which is the right trade-off for a chunk-and-embed
    consumer."""
    try:
        table_data = item.data
        if table_data is None:
            return None

        cells = list(getattr(table_data, "table_cells", []) or [])
        if not cells:
            return None

        n_rows = max(c.end_row_offset_idx for c in cells)
        n_cols = max(c.end_col_offset_idx for c in cells)
        grid: list[list[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]

        for c in cells:
            text = (getattr(c, "text", "") or "").strip()
            for r in range(c.start_row_offset_idx, c.end_row_offset_idx):
                for col in range(c.start_col_offset_idx, c.end_col_offset_idx):
                    if 0 <= r < n_rows and 0 <= col < n_cols:
                        grid[r][col] = text
        return grid
    except Exception as exc:
        logger.warning("docling: failed to extract table cells: %s", exc)
        return None
