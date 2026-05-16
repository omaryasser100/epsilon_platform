"""Request/response models for the rag_service HTTP API."""
from typing import Any, Optional
from pydantic import BaseModel


# ── Ingest ───────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    rag_channel_id: str
    file_path: str
    filename: str
    title: str = ""
    metadata: Optional[dict[str, Any]] = None


class IngestResponse(BaseModel):
    success: bool
    report_id: str
    pages_processed: int
    chunks_inserted: int
    pages_total: int
    extractor: str
    embed_model: str
    total_latency_ms: int


# ── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    rag_channel_id: str
    question: str
    top_k: int = 10


class ResultMetadata(BaseModel):
    report_id: str
    page_number: int
    chunk_index: int
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None
    section_title: str = ""


class QueryResponse(BaseModel):
    success: bool
    result_count: int
    results_metadata: list[ResultMetadata]


# ── Admin ────────────────────────────────────────────────────────────────────

class CreateChannelRequest(BaseModel):
    name: str
    description: str = ""
    metadata: Optional[dict[str, Any]] = None


class CreateChannelResponse(BaseModel):
    success: bool
    channel_id: str


class ReportSummary(BaseModel):
    report_id: str
    filename: str
    title: str
    page_count: int
    chunk_count: int
    created_at: Optional[str] = None


class ListReportsResponse(BaseModel):
    success: bool
    reports: list[ReportSummary]


class DeleteReportResponse(BaseModel):
    success: bool
    deleted_chunks: int
