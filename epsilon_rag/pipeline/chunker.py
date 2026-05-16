"""Document-level chunking.

Packs the document's structured blocks (across every page, in global
reading order) into chunk-shaped strings the embedder consumes,
threading section titles and per-bbox source pages through so the
retrieval layer can produce citation-friendly results.

Why document-level (not per-page)
==================================
Pages are a typesetter's layout artifact, not a semantic boundary.
Per-page chunking forced a `flush()` at every page break, which:
  • split lists / paragraphs / sections that wrapped across a page;
  • left orphan one-line chunks when a heading landed at the bottom of
    a page and its body started on the next;
  • dropped section_title context for chunks whose parent heading lived
    on a previous page.
Walking the whole document in one pass fixes all three. Each chunk's
`page_number` is the page of its FIRST contributing block (the
"primary" citation page), and the `bboxes` list carries `{"page", "bbox"}`
entries so multi-page chunks still cite every source region.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_CHARS = 2000   # ~500 tokens
DEFAULT_OVERLAP_CHARS = 200  # ~50 tokens of context bleed
# Light overlap — only enough to catch a sentence that lands within ~50
# tokens of a chunk boundary. Trades full-sentence guarantees for fewer
# near-duplicate hits at retrieval time (which is also why MMR is no
# longer wired in the query path; with low overlap, duplicate-pruning
# isn't earning its latency). If a sentence longer than ~150 tokens
# straddles a boundary, retrieval may see only half — accept that risk
# for sharper ranking. Re-ingest required for chunks already in the DB
# to pick up the new size.

# Block types that must never be split mid-content. Their `content` is
# already chunk-friendly (GFM markdown for tables, "[Figure: caption]\n
# extracted text" for figures, "$$ ... $$" for formulas) and splitting
# would break either the markup or the semantic unit.
_ATOMIC_BLOCK_TYPES = frozenset({"table", "figure", "formula"})

# Separator priority: largest semantic unit first. Arabic punctuation (؟ ۔)
# included alongside Latin so sentence-aware splits work for both scripts.
_SEPARATORS = [
    "\n\n",   # paragraph break
    "\n",     # line break
    ". ",     # English sentence end
    "؟ ",     # Arabic question mark
    "۔ ",     # Urdu / Arabic full stop
    "! ",
    "? ",
    "؛ ",     # Arabic semicolon
    "; ",
    " ",      # word break
    "",       # hard char cut (last resort)
]

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_CHUNK_CHARS,
    chunk_overlap=DEFAULT_OVERLAP_CHARS,
    separators=_SEPARATORS,
    keep_separator=True,
    length_function=len,
)


def chunk_text(text: str) -> list[str]:
    """Split `text` into ~500-token chunks with ~50-token overlap.

    Empty / whitespace-only input → []. Single short text → [text].
    """
    text = (text or "").strip()
    if not text:
        return []
    return _splitter.split_text(text)


# ── Structure-aware chunker ─────────────────────────────────────────────────


def chunk_blocks(blocks: list[dict]) -> list[str]:
    """Pack a page's structured blocks into chunk strings (text only).

    Convenience wrapper around `chunk_blocks_with_meta` for callers that
    don't need per-chunk metadata.
    """
    return [c["content"] for c in chunk_blocks_with_meta(blocks)]


def chunk_blocks_with_meta(blocks: list[dict]) -> list[dict]:
    """Pack blocks into chunks and return per-chunk metadata.

    Each block dict may carry `type`, `content`, `reading_order`, `bbox`,
    and `page_number`. The caller is expected to provide blocks in
    global reading order with a monotonic `reading_order` across the
    whole document — see [ingest.py](../pipeline/ingest.py) for how
    that's built.

    Each returned dict is
    `{content, block_types, section_title, page_number, bboxes}`:
      • `page_number` is the page of the chunk's FIRST contributing
        block (the "primary" citation page; multi-page chunks may pull
        bboxes from later pages too).
      • `bboxes` is a list of `{"page": int|None, "bbox": [x_min, y_min,
        x_max, y_max]}` dicts — one per contributing block — and is
        empty when the input blocks had no bbox field.
    """
    if not blocks:
        return []

    # Sort by reading_order so multi-column or RTL layouts stay coherent.
    # When the caller is ingest.py, reading_order is already globally
    # monotonic across pages, so this is effectively a no-op stabiliser
    # rather than a true reorder.
    sorted_blocks = sorted(
        (b for b in blocks if (b.get("content") or "").strip()),
        key=lambda b: int(b.get("reading_order") or 0),
    )
    if not sorted_blocks:
        return []

    chunks: list[dict] = []
    current_text = ""
    current_types: list[str] = []
    current_bboxes: list[dict] = []
    current_section = ""
    current_page: int | None = None   # primary page = first block's page
    last_section = ""  # heading-as-section for chunks AFTER the heading too

    def flush() -> None:
        nonlocal current_text, current_types, current_bboxes, current_page
        if current_text.strip():
            chunks.append({
                "content":       current_text.strip(),
                "block_types":   list(current_types),
                "section_title": current_section,
                "page_number":   current_page,
                "bboxes":        list(current_bboxes),
            })
        current_text = ""
        current_types = []
        current_bboxes = []
        current_page = None

    def _bbox_entry(bbox, page) -> dict | None:
        """Wrap a 4-tuple bbox + its source page into the storage shape.
        Returns None when bbox is None so callers can `if entry:` without
        emitting `{"page": …, "bbox": None}` dicts."""
        if not bbox:
            return None
        return {"page": page, "bbox": list(bbox)}

    for block in sorted_blocks:
        btype = block.get("type") or "text"
        text = (block.get("content") or "").strip()
        if not text:
            continue
        bbox = _coerce_bbox(block.get("bbox"))
        page = block.get("page_number")
        bbox_entry = _bbox_entry(bbox, page)

        # Atomic blocks: emit any pending chunk, then emit this block as
        # its own chunk regardless of size. Splitting a table mid-row or
        # a formula mid-LaTeX is worse than an oversized chunk.
        if btype in _ATOMIC_BLOCK_TYPES:
            flush()
            chunks.append({
                "content":       text,
                "block_types":   [btype],
                "section_title": last_section,
                "page_number":   page,
                "bboxes":        [bbox_entry] if bbox_entry else [],
            })
            continue

        # Headings start a new chunk and become the running section title.
        #
        # Exception: when the pending chunk is heading-only (no body has
        # been appended yet), fold the new heading into it instead of
        # flushing. Kills the one-line chunks that used to happen at
        # consecutive-heading transitions like
        # "Program Curriculum" → "1. Session 1: …", AND now (since
        # chunking is document-level) handles the cross-page variant
        # where the parent heading sits at the end of page N and the
        # subheading + body starts on page N+1.
        if btype == "heading":
            if current_types == ["heading"] and current_text.strip():
                merged_text = current_text.strip() + " — " + text
                last_section = merged_text[:300]
                current_section = last_section
                current_text = merged_text
                # current_types stays ["heading"]; current_page stays the
                # FIRST heading's page (don't move forward — the chunk
                # logically starts where the parent heading appears).
                if bbox_entry:
                    current_bboxes.append(bbox_entry)
            else:
                flush()
                last_section = text[:300]
                current_section = last_section
                current_text = text
                current_types = ["heading"]
                current_bboxes = [bbox_entry] if bbox_entry else []
                current_page = page
            continue

        # Plain text / footer block. If it alone exceeds the chunk size,
        # split it with the recursive splitter and emit each piece as
        # its own chunk. The bbox propagates to every piece because the
        # whole block is the source region for each piece.
        if len(text) > DEFAULT_CHUNK_CHARS:
            flush()
            for piece in _splitter.split_text(text):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append({
                    "content":       piece,
                    "block_types":   [btype],
                    "section_title": last_section,
                    "page_number":   page,
                    "bboxes":        [bbox_entry] if bbox_entry else [],
                })
            continue

        # Greedy pack: append to the current chunk if it fits, else flush
        # and start a new one with this block.
        joined = (current_text + "\n\n" + text) if current_text else text
        if len(joined) <= DEFAULT_CHUNK_CHARS:
            current_text = joined
            current_types.append(btype)
            if bbox_entry:
                current_bboxes.append(bbox_entry)
            if not current_section:
                current_section = last_section
            if current_page is None:
                current_page = page
        else:
            flush()
            current_text = text
            current_types = [btype]
            current_bboxes = [bbox_entry] if bbox_entry else []
            current_section = last_section
            current_page = page

    flush()
    return chunks


# ── Helpers ─────────────────────────────────────────────────────────────────

def _coerce_bbox(raw) -> tuple[float, float, float, float] | None:
    """Accept the BoundingBox pydantic model, a 4-tuple/list, or None
    and return a plain 4-tuple. Robust to None / malformed input so a
    legacy block dict without bbox doesn't blow up the chunker."""
    if raw is None:
        return None
    if isinstance(raw, (tuple, list)) and len(raw) == 4:
        return tuple(float(v) for v in raw)  # type: ignore[return-value]
    # Pydantic BoundingBox — duck-type on the four attribute names so
    # we don't have to import the schema here.
    try:
        return (
            float(raw.x_min), float(raw.y_min),
            float(raw.x_max), float(raw.y_max),
        )
    except AttributeError:
        return None
