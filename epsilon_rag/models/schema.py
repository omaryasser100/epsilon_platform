"""Pydantic models the pipeline returns from its public functions.

Used by:
  - [pipeline/orchestrator.py](../pipeline/orchestrator.py) — returns
    `ExtractResponse` (per page) and `ExtractDocumentResponse` (whole PDF).
  - [pipeline/ingest.py](../pipeline/ingest.py) — accepts `IngestOptions` /
    `ExtractOptions`.
  - [main.py](../main.py) — builds an `IngestOptions` from CLI flags.

These models describe the shape the pipeline passes around internally —
chunk-ready text, block annotations, per-page and document-level
metadata, and the option toggles every stage reads.
"""
from typing import Literal

from pydantic import BaseModel, Field


# ── Pipeline options ────────────────────────────────────────────────────────

class ExtractOptions(BaseModel):
    """Per-call toggles for the extraction stages. Defaults turn everything
    on; callers turn off what they don't want to pay latency for."""
    extract_tables: bool   = True
    extract_figures: bool  = True
    extract_formulas: bool = True
    ocr_fallback: bool     = True   # run OCR on regions without a text layer
    figure_captioning: bool = True  # currently a no-op (see pipeline/figures.py)


class IngestOptions(BaseModel):
    """Toggles for `run_ingest_from_path` / `run_ingest` — same flags as
    `ExtractOptions`. Kept separate so the ingest entry points and the
    pure-extract path can evolve independently."""
    extract_tables: bool    = True
    extract_figures: bool   = True
    extract_formulas: bool  = True
    ocr_fallback: bool      = True
    figure_captioning: bool = True


# ── Structured blocks (orchestrator output) ─────────────────────────────────

BlockType = Literal["text", "heading", "table", "figure", "formula", "footer"]


class BoundingBox(BaseModel):
    """Pixel-space bbox in (x_min, y_min, x_max, y_max) order, top-left origin."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class TableData(BaseModel):
    """Structured cell data for tabular regions."""
    rows: list[list[str]]   # row-major, Markdown-friendly strings
    markdown: str           # GFM table — what gets embedded into `content`
    n_rows: int
    n_cols: int


class FigureData(BaseModel):
    """Caption + (optional) extracted text from a figure region."""
    caption: str            # currently always "" — captioning is disabled
    extracted_text: str     # OCR'd labels/legends from the figure; "" if none


class FormulaData(BaseModel):
    """LaTeX form of a recognised formula image."""
    latex: str              # raw LaTeX string, no $$ delimiters


class Block(BaseModel):
    """A single layout-aware block on the page."""
    type: BlockType
    content: str            # text representation suitable for chunking / embedding
    bbox: BoundingBox | None = None
    reading_order: int      # global ordinal within the page; 0 = first
    table: TableData | None = None
    figure: FigureData | None = None
    formula: FormulaData | None = None


# ── Page-level metadata ─────────────────────────────────────────────────────

PageType = Literal["cover", "toc", "text", "table", "chart", "mixed", "empty"]
Language = Literal["ar", "en", "mixed"]


class PageMetadata(BaseModel):
    """Page-level metadata — section_title / page_type / language plus
    structured extras for tables, figures, and formulas."""
    section_title: str = ""
    page_type: PageType = "mixed"
    language: Language  = "mixed"

    tables:  list[TableData]  = Field(default_factory=list)
    figures: list[FigureData] = Field(default_factory=list)
    formulas: list[FormulaData] = Field(default_factory=list)
    reading_order: list[int]  = Field(default_factory=list)


class ExtractResponse(BaseModel):
    """One page's extraction result.

    `content` is the canonical chunk-ready string the embedder consumes.
    Tables are inlined as GFM markdown, figures as `[Figure: caption]`,
    formulas as `$$ … $$` so all extracted information survives chunking.
    """
    content: str
    metadata: PageMetadata
    blocks: list[Block]                 # full block list, in reading order
    extractor: str = "doc-processor-v1"
    page_number: int
    latency_ms: int


# ── Document-level extraction ───────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Cross-page summary for routing / filtering."""
    page_count: int
    detected_languages: list[Language]   # union of per-page languages
    has_tables:   bool
    has_figures:  bool
    has_formulas: bool


class ExtractDocumentResponse(BaseModel):
    """Full-PDF extraction result. `pages` preserves the per-page
    `ExtractResponse` shape so callers can stream / index them uniformly.

    `markdown` is Docling's own `export_to_markdown()` output for the whole
    document — captured on the same Docling pass that produced the layout
    regions, so there's no second conversion. The ingest path uploads this
    to MinIO at `<bucket>/<prefix>/<channel_id>/markdown/<report_id>.md`.
    """
    pages: list[ExtractResponse]
    document_metadata: DocumentMetadata
    extractor: str = "doc-processor-v1"
    total_latency_ms: int
    markdown: str = ""
