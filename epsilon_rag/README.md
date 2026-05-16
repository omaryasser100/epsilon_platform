# Esplion RAG

A local, multi-tenant RAG ingest + query pipeline for Arabic and English PDFs.
Channel-scoped, hybrid retrieval (dense + sparse), cross-encoder reranking,
runs entirely off your own GPU + Postgres.

```
                 PDF                                  question
                  │                                       │
        ┌─────────▼─────────┐                  ┌──────────▼─────────┐
        │ Docling layout    │                  │ bge-m3 embed       │
        │ RapidOCR (PP-OCR) │                  │  (dense + sparse)  │
        │ BLIP captions     │                  │ pgvector hybrid    │
        │ pix2text formulas │                  │  retrieval         │
        │ bge-m3 embed      │                  │ weighted RRF       │
        │  (dense + sparse) │                  │ bge-reranker-v2-m3 │
        └─────────┬─────────┘                  └──────────┬─────────┘
                  │                                       │
                  └────────────► Postgres ◄───────────────┘
                                + pgvector
                          channels → reports → chunks
```

## Stack

| Stage | Model / Tool |
|---|---|
| Layout / tables / reading order | Docling 2.93 (RT-DETRv2 + TableFormer) |
| OCR (Arabic + English) | RapidOCR (PP-OCRv3, ONNX runtime) |
| Figure captioning | `Salesforce/blip-image-captioning-base` |
| Formula → LaTeX | `breezedeus/pix2text-mfr` via HF `VisionEncoderDecoderModel` |
| Embeddings (hybrid) | `BAAI/bge-m3` via FlagEmbedding — 1024-dim dense + sparse over XLM-RoBERTa vocab |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder |
| Storage | PostgreSQL 16 + pgvector 0.8 (HNSW indexes on both vectors) |

Total VRAM warm: ~7 GB ingest, ~5 GB query. Fits a 12 GB consumer card.

## Quick start

```powershell
# 1. Spin up Postgres + pgvector + MinIO + schema migrate.
#    The `epsilon-rag` bucket is created lazily on first ingest;
#    objects land under <bucket>/<prefix>/<channel_id>/{raw,markdown,metadata}/.
cp .env.example .env
docker compose up -d
# MinIO console: http://localhost:9001  (login = MINIO_ROOT_USER/PASSWORD)

# 2. Install deps (host GPU path) — skip if you only use the `app` service.
pip install -r requirements.txt
# migrate already ran in Docker; re-run on the host if needed:
python scripts/migrate.py

# 3. Create a channel (company / tenant)
python channels.py create "Acme Corp" --desc "Q4 2024 reports"
# → prints UUID

# 4. Ingest PDFs into that channel
python main.py path/to/file.pdf --channel-id <uuid>
python main.py path/to/folder --channel-id <uuid> --recursive       # batch
python main.py file.pdf --channel-id <uuid> --meta author=Smith --meta year=2024

# 5. Query
python query.py "what were Q4 revenues?" --channel-id <uuid>
python query.py "..." --channel-id <uuid> --top-k 5 --alpha 0.7
python query.py "..." --channel-id <uuid> --prf-beta 0.3            # Rocchio expansion
python query.py "..." --channel-id <uuid> --neighbours 1            # ±1 chunks for context
python query.py "..." --channel-id <uuid> --json                    # machine-readable
```

### Docker GPU CLI (optional)

Builds `Dockerfile.gpu` on first run (large image with pre-cached HF weights).
Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```powershell
# Put PDFs in ./data (or set PDF_HOST_DIR in .env)
docker compose --profile gpu run --rm --gpus all app channels.py create "Acme Corp"
docker compose --profile gpu run --rm --gpus all app main.py /data/report.pdf --channel-id <uuid>
docker compose --profile gpu run --rm --gpus all app query.py "what were Q4 revenues?" --channel-id <uuid>
```

Inside compose, `DATABASE_URL` and `MINIO_ENDPOINT` point at the `postgres` and
`minio` service hostnames. On the host, `.env` keeps `localhost` for both.

## Layout

```
esplion-rag/
├── docker-compose.yml       # Postgres + MinIO + migrate (+ optional GPU app)
├── .env.example             # copy to .env, edit
├── requirements.txt
├── Dockerfile.gpu           # optional containerised CLI (ingest/query)
├── Dockerfile.migrate       # slim one-shot schema bootstrap image
├── migrations/              # numbered SQL, applied by scripts/migrate.py
├── scripts/migrate.py       # idempotent migration runner
├── core/
│   ├── config.py            # env-driven settings
│   └── db.py                # psycopg3 pool
├── models/schema.py         # pydantic models the pipeline passes around
├── pipeline/
│   ├── layout.py            # Docling wrapper
│   ├── ocr.py               # RapidOCR (PP-OCRv3 ONNX)
│   ├── tables.py            # cells → GFM markdown
│   ├── figures.py           # BLIP captioning
│   ├── formulas.py          # HF VisionEncoderDecoder
│   ├── arabic_normalize.py  # NFKC for Arabic
│   ├── chunker.py           # block-aware + recursive splitter
│   ├── embeddings.py        # bge-m3 (dense + sparse, one pass)
│   ├── reranker.py          # cross-encoder
│   ├── ingest.py            # end-to-end ingest
│   ├── retrieval.py         # hybrid SQL + RRF + rerank + neighbours
│   └── registry.py          # channels + reports CRUD
├── main.py                  # ingest CLI
├── query.py                 # query CLI
└── channels.py              # channel admin CLI
```

## Multi-tenant guarantees

Channel isolation is enforced **at the SQL level**:
- Every chunk row has a `NOT NULL` `ChannelId` with cascade delete.
- Every retrieval CTE starts with `WHERE "ChannelId" = $1`.
- `hybrid_query()` raises if `channel_id` isn't passed.
- Deleting a channel cascades to all reports + chunks for that tenant.

## Retrieval knobs

| Flag | Default | What it does |
|---|---|---|
| `--top-k` | 10 | Final result count |
| `--alpha` | 0.5 | Dense weight in RRF (0=sparse, 1=dense) |
| `--prf-beta` | 0.0 | Rocchio query expansion strength (0=off) |
| `--neighbours` | 1 | ±N adjacent chunks per result for context (0 disables) |
| `--no-rerank` | off | Skip cross-encoder for ~150 ms speedup |
| `--report-id` | (none) | Narrow to one PDF within the channel |

## License

Internal / private use. Pin your own license here when you're ready.
