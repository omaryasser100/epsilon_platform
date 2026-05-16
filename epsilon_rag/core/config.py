"""Doc-processor settings — env-driven so the same image can run different
environments with different model variants and thresholds without rebuilds."""
from dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class MinIOConfig:
    """Pass-around handle for the MinIO Python SDK. Built from `Settings`
    via `settings.minio_config()` — matches the shape used in the
    reference EsplionRAG ingestion pipeline so core.storage's signatures
    line up 1:1 with what the rest of the org already uses.
    """
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False
    prefix: str = "channels"


# Maps .NET / Npgsql connection-string keys to libpq keys so the doc-processor
# can accept either a libpq URI or a .NET-style connection string in
# DATABASE_URL — useful when the same secret is shared with a .NET API.
_DOTNET_TO_LIBPQ = {
    "host":        "host",
    "server":      "host",
    "port":        "port",
    "database":    "dbname",
    "username":    "user",
    "user id":     "user",
    "password":    "password",
    "ssl mode":    "sslmode",
    "sslmode":     "sslmode",
}


def _normalize_database_url(value: str) -> str:
    """Accept a libpq URI (postgres://...), libpq keyword string, or a .NET
    Npgsql connection string (Host=...;Database=...;Username=...;Password=...).
    .NET strings are converted to libpq keyword format that psycopg3 understands.
    """
    v = value.strip()
    if not v:
        return v
    if v.startswith(("postgres://", "postgresql://")):
        return v
    if "=" in v and ";" in v:   # .NET semicolon-separated format
        parts = []
        for chunk in v.split(";"):
            if "=" not in chunk:
                continue
            k, _, val = chunk.partition("=")
            key = _DOTNET_TO_LIBPQ.get(k.strip().lower())
            if not key:
                continue
            val = val.strip()
            if any(c in val for c in " '\\"):
                val = "'" + val.replace("\\", "\\\\").replace("'", "\\'") + "'"
            parts.append(f"{key}={val}")
        return " ".join(parts)
    return v


class Settings(BaseSettings):
    # ── Models ───────────────────────────────────────────────────────────────
    # BLIP — figure captioning. Salesforce/blip-image-captioning-base is
    # ~990 MB and works on transformers ≥ 4.40, which is the range
    # docling 2.93 also supports. Override via env if you want BLIP-2
    # or a different VLM (the loader uses AutoProcessor / BlipFor-
    # ConditionalGeneration, so any BLIP-family model works as-is).
    figures_model_id: str = "Salesforce/blip-image-captioning-base"
    figure_caption_max_tokens: int = 64   # one-sentence captions
    figure_caption_beams: int = 3         # 3 beams trades ~30ms for noticeably better wording

    # Formula OCR — VisionEncoderDecoderModel from HF. Default is
    # breezedeus/pix2text-mfr (math formula recognition head only,
    # ~340 MB). Swap for any HF model that ships an AutoProcessor +
    # VisionEncoderDecoderModel pair if you need broader support.
    formulas_model_id: str = "breezedeus/pix2text-mfr"
    formulas_max_tokens: int = 384        # long enough for multi-line equations
    formulas_beams: int = 1               # greedy by default; raise for accuracy at ~2× latency

    # Surya OCR language hints — CSV. 'ar,en' covers Arabic + English.
    # Adding a language is just an env var change (no image rebuild)
    # because Surya's recognition model is multilingual; the codes
    # here gate script-routing behaviour, not separate model downloads.
    ocr_languages: str = "ar,en"

    # Multilingual sentence-transformer for /v1/embed.
    # BAAI/bge-m3 — 1024 dims, ~2.5 GB VRAM, top-of-class on Arabic retrieval
    # (MIRACL ~75 vs e5-base ~51) and strong cross-lingual AR↔EN. Unlike E5
    # it does NOT require "query: " / "passage: " prefixes — raw text is
    # what it was trained on. Schema column is vector(1024) accordingly.
    embed_model_id: str = "BAAI/bge-m3"
    embed_max_seq_length: int = 8192       # bge-m3 supports up to 8k tokens; chunks are well under
    embed_batch_size: int = 16             # bge-m3 is bigger than e5; smaller batch keeps VRAM safe

    # bge-m3's sparse / lexical head emits weights over the XLM-RoBERTa
    # vocabulary (250 002 tokens). This dim must match the
    # report_chunks.sparse_embedding sparsevec(...) column exactly —
    # mismatches throw at INSERT.
    embed_sparse_dim: int = 250002

    # Cross-encoder reranker for /v1/rerank. Trained as a sibling to bge-m3
    # so query and chunk vectors live in compatible representation spaces;
    # joint scoring is meaningfully better than re-scoring with cosine.
    # ~568M params, ~2.5 GB VRAM in fp16 alongside bge-m3 + Docling + Surya
    # — total VRAM stays comfortably under 8 GB so a 12 GB consumer card
    # is enough headroom.
    rerank_model_id: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_seq_length: int = 1024      # 512 query + 512 passage is plenty for chunk-level rerank
    rerank_batch_size: int = 8             # bge-reranker-v2-m3 is heavier than the embedder; smaller batch

    # Hardware / inference toggles.
    device: str = "cuda"                         # "cuda" | "cpu" — set "cpu" for local dev
    fp16: bool  = True                           # half-precision for VLM speed

    # ── Pipeline behaviour ───────────────────────────────────────────────────
    # When Docling produces text directly (digital PDF), we trust it. When the
    # text is empty / suspiciously short, we run OCR on the region as fallback.
    ocr_fallback_min_chars: int = 20             # below this length → OCR

    # Skip OCR fallback on regions that are too small to contain meaningful
    # text or are visually empty (solid-color rectangles, decorative borders).
    # Each skipped region saves ~50-300 ms of Surya OCR work; a typical
    # bulk ingest has 5-10 such regions per page.
    ocr_min_region_area_ratio: float = 0.005     # < 0.5% of page area → skip
    ocr_min_pixel_stddev: float = 8.0            # solid-color crops → skip

    # Vestigial: captioning is disabled, so this threshold has no effect on
    # the runtime path. Kept so the env contract doesn't break, and so the
    # orchestrator's area gate stays a one-line flip the day a captioner
    # is re-introduced.
    figure_caption_min_area_ratio: float = 0.02  # < 2% of page area → no caption

    # Cap how many figures / tables / formulas we process per page so a
    # malformed PDF can't blow up GPU memory or runtime.
    max_blocks_per_page: int = 200
    max_table_chars: int     = 50000
    max_formula_chars: int   = 5000

    # ── Database (used by the ingest pipeline to persist chunks) ────────────────
    # Default points at the local pgvector/pgvector:pg16 Docker container
    # (`rag-pgvec`) with the schema in `init.sql` (report_chunks + ai_jobs).
    # Override via the DATABASE_URL env var (libpq URI or .NET-style
    # connection string). Set to "" to start in extract-only mode (no DB
    # pool initialised).
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_db"

    # ── Object storage (MinIO / S3-compatible) ───────────────────────────────
    # Consumed by core/storage.py via settings.minio_config(). Bucket and
    # the per-channel raw/markdown/metadata prefixes are created lazily on
    # first ingest by storage.ensure_bucket() — no `mc` bootstrap needed.
    # Defaults point at the local `rag-minio` Docker container.
    minio_endpoint: str       = "localhost:9000"
    minio_access_key: str     = "minioadmin"
    minio_secret_key: str     = "minioadmin"
    minio_bucket: str         = "epsilon-rag"
    minio_bucket_prefix: str  = "channels"
    minio_use_ssl: bool       = False

    # extra="ignore" — .env also carries POSTGRES_* / MINIO_ROOT_* keys for
    # docker-compose.yml; those are not Settings fields.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context) -> None:
        self.database_url = _normalize_database_url(self.database_url)

    def minio_config(self) -> MinIOConfig:
        return MinIOConfig(
            endpoint=self.minio_endpoint,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            bucket=self.minio_bucket,
            secure=self.minio_use_ssl,
            prefix=self.minio_bucket_prefix.strip().strip("/"),
        )


settings = Settings()  # type: ignore[call-arg]
