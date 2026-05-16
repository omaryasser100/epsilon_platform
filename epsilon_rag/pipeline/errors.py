"""Custom exception types raised by the ingest pipeline.

Lives in its own module so layout.py and ingest.py can both import it
without circular dependencies.
"""
from __future__ import annotations


class InvalidPdfError(Exception):
    """The input PDF can't be parsed by Docling.

    Carries diagnostic context (size + first 8 bytes hex) so the failure
    message in ai_jobs.error_message and Cloud Logging tells you whether
    the upload was empty, truncated, encrypted, or simply not a PDF.

    Treated as non-retryable by ingest._retry — the bytes are deterministic,
    so re-running 4× wastes ~40 s of GPU on a doomed input.
    """

    def __init__(self, reason: str, *, size: int, header: bytes):
        header_preview = header[:8].hex() if header else ""
        super().__init__(f"{reason} (size={size}B, header_hex={header_preview})")
        self.reason = reason
        self.size = size
        self.header = header
