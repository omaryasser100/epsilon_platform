"""Admin endpoints — channel CRUD, RAG-channel linking, report management.

All routes here require the caller's JWT to carry the `admin_panel` feature.
Channel/report operations that touch the RAG schema are proxied to rag_service
so the platform backend never talks to pgvector directly.
"""
import json
import logging
import os
from typing import Any, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from db import engine
from jwt_auth import get_current_user

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag_service:8001")
RAG_TIMEOUT = 30

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Gate dependency: rejects callers without the admin_panel feature."""
    if "admin_panel" not in (user.get("authorized_features") or []):
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


# ── Schemas ──────────────────────────────────────────────────────────────────

class ChannelOverview(BaseModel):
    channelid: int
    name: str
    rag_channel_id: Optional[str] = None
    metadata: dict[str, Any] = {}
    authorized_features: list[str] = []


class ChannelsResponse(BaseModel):
    success: bool
    channels: list[ChannelOverview]


class CreateChannelRequest(BaseModel):
    name: str
    description: str = ""
    authorized_features: list[str] = []


class CreateChannelResponse(BaseModel):
    success: bool
    channelid: int
    name: str


class CreateRagChannelRequest(BaseModel):
    name: Optional[str] = None        # defaults to control.channel.name
    description: str = ""


class CreateRagChannelResponse(BaseModel):
    success: bool
    channelid: int
    rag_channel_id: str
    name: str


class ReportSummary(BaseModel):
    report_id: str
    filename: str
    title: str
    page_count: int
    chunk_count: int
    created_at: Optional[str] = None


class ReportsResponse(BaseModel):
    success: bool
    reports: list[ReportSummary]


class DeleteResponse(BaseModel):
    success: bool
    deleted_chunks: int


# ── Routes ───────────────────────────────────────────────────────────────────

@router.get("/channels", response_model=ChannelsResponse)
def list_channels(_user: dict = Depends(require_admin)):
    """Return every control.channel row with its RAG link state and features."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT channelid, name, rag_channel_id, metadata, authorized_features
            FROM control.channel
            ORDER BY channelid
        """)).mappings().fetchall()

    channels = [
        ChannelOverview(
            channelid=r["channelid"],
            name=r["name"],
            rag_channel_id=str(r["rag_channel_id"]) if r["rag_channel_id"] else None,
            metadata=r["metadata"] or {},
            authorized_features=r["authorized_features"] or [],
        )
        for r in rows
    ]
    return ChannelsResponse(success=True, channels=channels)


@router.post("/channels", response_model=CreateChannelResponse)
def create_channel(req: CreateChannelRequest, _user: dict = Depends(require_admin)):
    """Insert a new tenant row into control.channel. RAG linking is a
    separate step via POST /admin/channels/{channelid}/rag-channel."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Channel name cannot be empty.")

    metadata: dict[str, Any] = {}
    if req.description.strip():
        metadata["description"] = req.description.strip()

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    INSERT INTO control.channel (name, metadata, authorized_features)
                    VALUES (:name, CAST(:metadata AS jsonb), CAST(:features AS jsonb))
                    RETURNING channelid
                """),
                {
                    "name": name,
                    "metadata": _json_dumps(metadata),
                    "features": _json_dumps(req.authorized_features or []),
                },
            ).fetchone()
            conn.commit()
    except Exception as exc:
        # UniqueViolation on name → 409; other errors → 500.
        msg = str(exc)
        if "duplicate key" in msg.lower() or "unique constraint" in msg.lower():
            raise HTTPException(status_code=409, detail=f"Channel '{name}' already exists.")
        logger.exception("create_channel: failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to create channel: {exc}")

    return CreateChannelResponse(success=True, channelid=int(row[0]), name=name)


@router.post("/channels/{channelid}/rag-channel", response_model=CreateRagChannelResponse)
def create_and_link_rag_channel(
    channelid: int,
    req: CreateRagChannelRequest,
    _user: dict = Depends(require_admin),
):
    """Create a RAG channel via rag_service, then save its UUID onto the
    matching control.channel row. Fails if the channel is already linked."""
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT channelid, name, rag_channel_id, metadata
                FROM control.channel
                WHERE channelid = :cid
            """),
            {"cid": channelid},
        ).mappings().fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Channel {channelid} not found.")
    if row["rag_channel_id"] is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Channel {channelid} is already linked to RAG channel {row['rag_channel_id']}.",
        )

    rag_name = (req.name or row["name"]).strip()
    if not rag_name:
        raise HTTPException(status_code=400, detail="Channel name cannot be empty.")

    try:
        resp = requests.post(
            f"{RAG_SERVICE_URL}/channels",
            json={
                "name": rag_name,
                "description": req.description,
                "metadata": row["metadata"] or {},
            },
            timeout=RAG_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        logger.error("create_rag_channel: rag_service error: %s", detail)
        raise HTTPException(status_code=502, detail=f"RAG service error: {detail}")
    except Exception as exc:
        logger.exception("create_rag_channel: rag_service call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"RAG service unreachable: {exc}")

    rag_channel_id = data["channel_id"]

    with engine.connect() as conn:
        conn.execute(
            text("""
                UPDATE control.channel
                SET rag_channel_id = :rid
                WHERE channelid = :cid
            """),
            {"rid": rag_channel_id, "cid": channelid},
        )
        conn.commit()

    return CreateRagChannelResponse(
        success=True,
        channelid=channelid,
        rag_channel_id=rag_channel_id,
        name=rag_name,
    )


@router.get("/channels/{channelid}/reports", response_model=ReportsResponse)
def list_channel_reports(channelid: int, _user: dict = Depends(require_admin)):
    """List ingested reports for the given control channel. Returns an empty
    list when the channel is not yet linked to a RAG channel."""
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT rag_channel_id
                FROM control.channel
                WHERE channelid = :cid
            """),
            {"cid": channelid},
        ).mappings().fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Channel {channelid} not found.")
    if row["rag_channel_id"] is None:
        return ReportsResponse(success=True, reports=[])

    try:
        resp = requests.get(
            f"{RAG_SERVICE_URL}/channels/{row['rag_channel_id']}/reports",
            timeout=RAG_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.exception("list_channel_reports: rag_service call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"RAG service error: {exc}")

    reports = [ReportSummary(**r) for r in data.get("reports", [])]
    return ReportsResponse(success=True, reports=reports)


@router.delete("/reports/{report_id}", response_model=DeleteResponse)
def delete_report(report_id: str, _user: dict = Depends(require_admin)):
    """Delete a report and all of its chunks. Cascades through the RAG schema's
    foreign keys; returns the number of removed chunk rows."""
    try:
        resp = requests.delete(
            f"{RAG_SERVICE_URL}/reports/{report_id}",
            timeout=RAG_TIMEOUT,
        )
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found.")
        resp.raise_for_status()
        data = resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("delete_report: rag_service call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"RAG service error: {exc}")

    return DeleteResponse(success=True, deleted_chunks=data.get("deleted_chunks", 0))


# ── Internals ────────────────────────────────────────────────────────────────

def _json_dumps(value: Any) -> str:
    """Stable JSON encoding for the JSONB cast parameters above."""
    return json.dumps(value)
