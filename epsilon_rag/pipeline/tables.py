"""Table grid → GitHub-Flavored Markdown.

Why GFM markdown specifically:
    1. The text passes through embedding / LLM stages downstream — markdown
       tables are the most LLM-friendly tabular format (most chat-tuned
       models reliably parse them). CSV survives, but the LLM has to
       remember column structure across rows.
    2. We can drop the markdown directly into the chunk content; it survives
       chunking because the LangChain RecursiveCharacterTextSplitter falls
       back to character splits before splitting inside a markdown row.
    3. Re-display in the frontend is one library call (any markdown
       renderer) instead of bespoke table layout code.

Cells are escaped so a stray `|` inside a value can't shift columns.
"""
from __future__ import annotations

from core.config import settings


def grid_to_markdown(grid: list[list[str]]) -> str:
    """Render a row-major 2-D string grid as a GFM markdown table.

    The first row is treated as the header. If the table has only one row,
    we still emit a separator line so consumers can parse it without
    special-casing.
    """
    if not grid or not grid[0]:
        return ""

    n_cols = max(len(r) for r in grid)
    rows = [_pad(r, n_cols) for r in grid]

    header = rows[0]
    body   = rows[1:] if len(rows) > 1 else []

    lines: list[str] = []
    lines.append("| " + " | ".join(_escape_cell(c) for c in header) + " |")
    lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    for row in body:
        lines.append("| " + " | ".join(_escape_cell(c) for c in row) + " |")

    md = "\n".join(lines)
    if len(md) > settings.max_table_chars:
        # Truncate large tables rather than blowing up downstream chunkers.
        md = md[: settings.max_table_chars] + "\n| … truncated … |"
    return md


def _pad(row: list[str], n: int) -> list[str]:
    if len(row) >= n:
        return row[:n]
    return list(row) + [""] * (n - len(row))


def _escape_cell(s: str) -> str:
    """GFM cell escaping: pipe and newline are the only structural chars to
    worry about. Backslashes left as-is — markdown renderers handle them."""
    if not s:
        return ""
    return s.replace("\n", " ").replace("|", r"\|").strip()
