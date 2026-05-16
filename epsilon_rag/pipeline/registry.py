"""Channel + report bookkeeping helpers.

These functions own the upstream of report_chunks: every chunk row points
at a `reports.Id`, and every report points at a `channels.Id`. The CLI
entry points (channels.py, main.py) and the query layer go through this
module rather than writing raw SQL inline so the SQL surface stays small
and consistent.

The functions are deliberately thin (one round-trip each, no caching) so
they're trivial to reason about and there's no stale-state to flush at
process exit. The DB pool in core/db.py is the only state in play.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from core.db import get_conn

logger = logging.getLogger(__name__)


# ── Channels ────────────────────────────────────────────────────────────────

def create_channel(
    name: str,
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Insert a new channel and return its UUID.

    Names are uniqued by the channels.Name UNIQUE constraint, so two
    callers can't create the same logical channel twice — the second
    INSERT raises and the caller gets a 23505 unique-violation.
    """
    if not name.strip():
        raise ValueError("channel name must be non-empty")
    meta_json = json.dumps(metadata or {})
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO channels ("Name", "Description", metadata)
            VALUES (%s, %s, %s::jsonb)
            RETURNING "Id"
            """,
            [name.strip(), description, meta_json],
        )
        row = cur.fetchone()
        conn.commit()
    if row is None:
        raise RuntimeError("INSERT … RETURNING produced no row")
    return str(row[0])


def get_channel(channel_id: str) -> dict[str, Any] | None:
    """Look up a channel by UUID. Returns None when no such row exists."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT "Id", "Name", "Description", metadata, "CreatedAt"
            FROM channels
            WHERE "Id" = %s
            """,
            [channel_id],
        )
        row = cur.fetchone()
    return _channel_row(row) if row else None


def get_channel_by_name(name: str) -> dict[str, Any] | None:
    """Look up a channel by Name. Convenience for CLI scripts that prefer
    a human-readable identifier; the canonical identifier is still the UUID."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT "Id", "Name", "Description", metadata, "CreatedAt"
            FROM channels
            WHERE "Name" = %s
            """,
            [name.strip()],
        )
        row = cur.fetchone()
    return _channel_row(row) if row else None


def list_channels() -> list[dict[str, Any]]:
    """All channels, oldest first. Used by `channels.py list`."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT "Id", "Name", "Description", metadata, "CreatedAt"
            FROM channels
            ORDER BY "CreatedAt" ASC
            """,
        )
        rows = cur.fetchall()
    return [_channel_row(r) for r in rows]


# ── Reports ─────────────────────────────────────────────────────────────────

def upsert_report(
    channel_id: str,
    filename: str,
    title: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Find-or-create the reports row for (channel_id, filename).

    On conflict, `Title` / `metadata` are overwritten with whatever the
    caller passed — re-running an ingest with new metadata replaces the
    old values rather than silently keeping them. If the caller doesn't
    want to clobber, they should pass the existing values back.

    Returns the report UUID — that's what gets stamped onto every chunk
    the ingest pipeline writes for this PDF.
    """
    if not filename.strip():
        raise ValueError("filename must be non-empty")
    meta_json = json.dumps(metadata or {})
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO reports ("ChannelId", "Filename", "Title", metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT ("ChannelId", "Filename") DO UPDATE
                SET "Title" = EXCLUDED."Title",
                    metadata = EXCLUDED.metadata
            RETURNING "Id"
            """,
            [channel_id, filename.strip(), title, meta_json],
        )
        row = cur.fetchone()
        conn.commit()
    if row is None:
        raise RuntimeError("UPSERT … RETURNING produced no row")
    return str(row[0])


def get_report(report_id: str) -> dict[str, Any] | None:
    """Look up a report by UUID. Returns None when not found."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT "Id", "ChannelId", "Filename", "Title", metadata, "CreatedAt"
            FROM reports
            WHERE "Id" = %s
            """,
            [report_id],
        )
        row = cur.fetchone()
    return _report_row(row) if row else None


def list_reports(channel_id: str) -> list[dict[str, Any]]:
    """All reports in a channel, oldest first. Used by `channels.py show`."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT "Id", "ChannelId", "Filename", "Title", metadata, "CreatedAt"
            FROM reports
            WHERE "ChannelId" = %s
            ORDER BY "CreatedAt" ASC
            """,
            [channel_id],
        )
        rows = cur.fetchall()
    return [_report_row(r) for r in rows]


# ── Row → dict adapters ─────────────────────────────────────────────────────

def _channel_row(row: tuple) -> dict[str, Any]:
    return {
        "id":          str(row[0]),
        "name":        row[1],
        "description": row[2],
        "metadata":    row[3] or {},
        "created_at":  row[4].isoformat() if row[4] else None,
    }


def _report_row(row: tuple) -> dict[str, Any]:
    return {
        "id":          str(row[0]),
        "channel_id":  str(row[1]),
        "filename":    row[2],
        "title":       row[3],
        "metadata":    row[4] or {},
        "created_at":  row[5].isoformat() if row[5] else None,
    }
