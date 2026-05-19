"""Action router for the chatbot and ingestion flows.

The frontend POSTs {action, payload} to /orchestrate; this function decides
what the action means (currently "query" and "ingest"), enforces tenant
isolation, and proxies to rag_service over HTTP.
"""
import logging
import os

import requests

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag_service:8001")

logger = logging.getLogger(__name__)


def orchestrator(user_session: dict, action: str, payload: dict | None = None):
    if payload is None:
        payload = {}

    if not user_session:
        return {
            "success": False,
            "path": None,
            "message": "You are not logged in.",
        }

    if action == "query":
        return _query(user_session, payload)

    if action == "ingest":
        return _ingest(user_session, payload)

    return {
        "success": False,
        "path": None,
        "message": f"Unknown action: {action}",
    }


def _query(user_session: dict, payload: dict) -> dict:
    """Forward a question to rag_service scoped to the caller's RAG channel."""
    if not user_session.get("rag_channel_id"):
        return {
            "success": False,
            "path": "query",
            "message": "This channel does not have a RAG channel ID yet.",
        }

    question = payload.get("question", "").strip()
    if not question:
        return {
            "success": False,
            "path": "query",
            "message": "Question cannot be empty.",
        }

    try:
        resp = requests.post(
            f"{RAG_SERVICE_URL}/query",
            json={
                "rag_channel_id": user_session["rag_channel_id"],
                "question": question,
                # Forwarded to Langfuse as trace tags. `username` over
                # `userid` because the UI is more useful with the human
                # name; the numeric id is in metadata. `session_id` is
                # the JWT session — one login → one Langfuse session.
                "user_id":    user_session.get("username"),
                "session_id": user_session.get("session_id"),
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.exception("orchestrator: query failed: %s", exc)
        return {
            "success": False,
            "path": "query",
            "message": f"RAG service error: {exc}",
        }

    return {
        "success": True,
        "path": "query",
        "message": f"Retrieved {data['result_count']} result(s).",
        "result_count": data["result_count"],
        "results_metadata": data["results_metadata"],
    }


def _ingest(user_session: dict, payload: dict) -> dict:
    """Run an ingest against the supplied RAG channel. Admin-only."""
    is_admin = "admin_panel" in (user_session.get("authorized_features") or [])
    if not is_admin:
        return {
            "success": False,
            "path": "ingest",
            "message": "Only admins can run ingestion.",
        }

    rag_channel_id = payload.get("rag_channel_id")
    file_path = payload.get("file_path")
    filename = payload.get("filename")

    if not rag_channel_id or not file_path or not filename:
        return {
            "success": False,
            "path": "ingest",
            "message": "rag_channel_id, file_path, and filename are required.",
        }

    try:
        resp = requests.post(
            f"{RAG_SERVICE_URL}/ingest",
            json={
                "rag_channel_id": rag_channel_id,
                "file_path": file_path,
                "filename": filename,
                "title": payload.get("title", ""),
                "metadata": payload.get("metadata", {}),
                # Trace tags — see _query for rationale.
                "user_id":    user_session.get("username"),
                "session_id": user_session.get("session_id"),
            },
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.exception("orchestrator: ingest failed: %s", exc)
        return {
            "success": False,
            "path": "ingest",
            "message": f"RAG service error: {exc}",
        }

    return {
        "success": True,
        "path": "ingest",
        "message": f"Ingested {data['pages_processed']} pages, {data['chunks_inserted']} chunks.",
        "report_id": data["report_id"],
        "pages_processed": data["pages_processed"],
        "pages_total": data["pages_total"],
        "chunks_inserted": data["chunks_inserted"],
        "total_latency_ms": data["total_latency_ms"],
    }
