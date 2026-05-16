"""Sub-page chunking.

Packs a page's structured blocks into chunk-shaped strings the embedder
consumes, threading section titles and source bboxes through so the
retrieval layer can produce citation-friendly results.
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

    Each block dict may carry `type`, `content`, `reading_order`, and
    (optionally) `bbox` — when bbox is present it's threaded onto every
    chunk that consumed (any portion of) that block, so the retrieval
    layer can render citations like "page 5, top-right".

    Each returned dict is `{content, block_types, section_title, bboxes}`.
    `bboxes` is a list of (x_min, y_min, x_max, y_max) tuples — one per
    contributing block — and is empty when the input blocks had no bbox
    field (e.g. legacy callers).
    """
    if not blocks:
        return []

    # Sort by reading_order so multi-column or RTL layouts stay coherent.
    sorted_blocks = sorted(
        (b for b in blocks if (b.get("content") or "").strip()),
        key=lambda b: int(b.get("reading_order") or 0),
    )
    if not sorted_blocks:
        return []

    chunks: list[dict] = []
    current_text = ""
    current_types: list[str] = []
    current_bboxes: list[tuple[float, float, float, float]] = []
    current_section = ""
    last_section = ""  # heading-as-section for chunks AFTER the heading too

    def flush() -> None:
        nonlocal current_text, current_types, current_bboxes
        if current_text.strip():
            chunks.append({
                "content":       current_text.strip(),
                "block_types":   list(current_types),
                "section_title": current_section,
                "bboxes":        list(current_bboxes),
            })
        current_text = ""
        current_types = []
        current_bboxes = []

    for block in sorted_blocks:
        btype = block.get("type") or "text"
        text = (block.get("content") or "").strip()
        if not text:
            continue
        bbox = _coerce_bbox(block.get("bbox"))

        # Atomic blocks: emit any pending chunk, then emit this block as
        # its own chunk regardless of size. Splitting a table mid-row or
        # a formula mid-LaTeX is worse than an oversized chunk.
        if btype in _ATOMIC_BLOCK_TYPES:
            flush()
            chunks.append({
                "content":       text,
                "block_types":   [btype],
                "section_title": last_section,
                "bboxes":        [bbox] if bbox else [],
            })
            continue

        # Headings start a new chunk and become the running section title.
        if btype == "heading":
            flush()
            last_section = text[:300]
            current_section = last_section
            current_text = text
            current_types = ["heading"]
            current_bboxes = [bbox] if bbox else []
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
                    "bboxes":        [bbox] if bbox else [],
                })
            continue

        # Greedy pack: append to the current chunk if it fits, else flush
        # and start a new one with this block.
        joined = (current_text + "\n\n" + text) if current_text else text
        if len(joined) <= DEFAULT_CHUNK_CHARS:
            current_text = joined
            current_types.append(btype)
            if bbox:
                current_bboxes.append(bbox)
            if not current_section:
                current_section = last_section
        else:
            flush()
            current_text = text
            current_types = [btype]
            current_bboxes = [bbox] if bbox else []
            current_section = last_section

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
