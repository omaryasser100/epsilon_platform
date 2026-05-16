"""Arabic text normalization for the doc-processor pipeline.

What this module does (current)
===============================
`normalize(text)` runs Unicode NFKC compatibility decomposition. That folds
Arabic Presentation Forms (U+FB50–FDFF and U+FE70–FEFF) — the ligature /
contextual-shape codepoints some older PDFs embed — into standard Arabic
codepoints (U+0600–06FF) that downstream tokenisers / embedders expect.

That's the whole job. NFKC is idempotent: already-normalised text passes
through unchanged.

Why no character-order reversal anymore
=======================================
A previous version of this module additionally reversed any line whose
"Arabic letter ratio" exceeded 0.6, on the theory that PyMuPDF returns
Arabic glyphs in visual (left-to-right) order and we needed to flip them
back to logical reading order.

That theory was true for ~2014-era glyph-encoded PDFs *without* a ToUnicode
CMap. It is NOT true for modern, properly-tagged PDFs: PyMuPDF ≥ 1.18 and
Docling's text extractor both honour bidi correctly and return Arabic in
logical order out of the box. Applying the reversal on top of already-
correct text destroys it — every "موافق" comes out as "قفاوم", retrieval
silently breaks across every Arabic chunk in the corpus, and the agent
has no way to recover from the damage.

The reversal was removed on 2026-05-11 after this exact symptom hit a real
report (a3df9adc-aa47-4f05-a686-fc7fa35b9228). The fix was a one-line
change and proved out the next ingest.

What to do if you DO find a visual-order PDF
============================================
If a specific document genuinely comes out of extraction in visual order,
do not reintroduce the blanket reversal — that's how we got into this
mess in the first place. The right places to fix it are:

  1. Docling pipeline options — there's a bidi handling flag worth setting
     explicitly even if it defaults correctly.
  2. PyMuPDF flags — `TEXTFLAGS_TEXT | TEXT_PRESERVE_LIGATURES` plus a
     `bidirectional=True` analog (check the version's options).
  3. As a last resort, a per-PDF override (e.g. a column on `reports` that
     marks a specific upload as "extract-in-visual-order", and reversal
     only fires when that column is true).

Reversing post-hoc, by line-level heuristic, is not invertible and not safe.
"""
from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    """Normalise Arabic text extracted from a PDF.

    Currently runs NFKC compatibility decomposition only — see the module
    docstring for why the previous line-reversal pass was removed.

    Empty / whitespace-only input returns it unchanged.
    """
    if not text:
        return text
    # Per-line so behaviour is independent of how the caller batched its
    # input. Cheap — NFKC over a few KB is microseconds.
    return "\n".join(
        unicodedata.normalize("NFKC", line)
        for line in text.split("\n")
    )
