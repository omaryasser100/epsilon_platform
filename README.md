# Epsilon Platform

Multi-tenant RAG chatbot platform. A Gradio frontend talks to a FastAPI
backend, which dispatches retrieval and ingestion to a GPU-backed RAG
service that wraps the `epsilon_rag` pipeline. PostgreSQL (with `pgvector`)
holds both the control plane (tenants, users, sessions) and the RAG corpus
(channels, reports, chunks). MinIO stores the raw PDFs, extracted
markdown, and per-ingest metadata blobs.

---

## Quick start

The whole stack runs from one `docker compose up`. Two run targets are
supported: a local workstation, and a Lightning AI Studio.

### Run locally

```bash
docker compose up -d --build
```

When all five containers are healthy (`docker compose ps`), open
<http://localhost:7860> in your browser.

The first run downloads ~5 GB of HuggingFace weights into `rag_service`'s
container — watch progress with `docker compose logs -f rag_service` and
wait for `warmup: total Xs`. After that, restarts warm up in seconds
because the models are cached.

### Run on Lightning AI

```bash
docker compose up -d --build
```

The simplest way to reach the Gradio UI from Lightning AI is to enable
Gradio's public-tunnel mode:

1. Append one line to `.env`:
   ```
   GRADIO_SHARE=true
   ```
2. Rebuild the frontend container:
   ```bash
   docker compose up -d --build frontend
   ```
3. Tail the frontend logs and copy the `https://*.gradio.live` URL it
   prints:
   ```bash
   docker compose logs -f frontend
   ```
4. Open that URL in any browser — it's valid for 72 hours.

If you'd rather not use the public tunnel, the default launch mode mounts
Gradio inside a FastAPI app that strips `X-Frame-Options`, so Lightning
AI's built-in port preview (`Ports → 7860 → Open`) renders the UI without
the iframe being blocked.

---

## Seeded users

These accounts are inserted by `db/init_control.sql` on first boot of a
clean `postgres_data` volume. Passwords are plaintext for the demo only.

| Username         | Password | Channel    | Authorized features                                          | Role / use it to test                             |
|------------------|----------|------------|--------------------------------------------------------------|---------------------------------------------------|
| `admin`          | `1234`   | Epsilon AI | `chatbot`, `document_search`, `voice_agent`, **`admin_panel`** | The **only admin** in the system. All four tabs. |
| `omar`           | `1111`   | Epsilon AI | `chatbot`, `document_search`                                 | Regular Epsilon-AI user. Chat tab only.           |
| `aman_admin`     | `1234`   | Aman       | `chatbot`, `document_search`, `tables_search`                | Regular Aman user (name is legacy; no admin).     |
| `aman_user`      | `1111`   | Aman       | `chatbot`, `document_search`                                 | Regular Aman user.                                |
| `swedy_admin`    | `1234`   | Swedy      | `chatbot`, `document_search`, `tables_search`, `voice_agent` | Regular Swedy user (name is legacy; no admin).    |
| `swedy_engineer` | `1111`   | Swedy      | `chatbot`, `document_search`, `tables_search`                | Regular Swedy user.                               |

**How the gate works:** the JWT issued at login carries `authorized_features`.
The Gradio frontend shows the admin tabs only when the JWT contains
`admin_panel`. The backend's `/admin/*` routes enforce the same check
server-side, so even a tampered token can't reach those endpoints.

To regenerate seed (e.g. after editing the SQL):

```bash
docker compose down -v
docker compose up -d --build
```

---

## Architecture at a glance

| Service             | Port (host) | Role                                              |
|---------------------|-------------|---------------------------------------------------|
| `frontend`          | 7860        | Gradio UI (login, chat, admin)                    |
| `backend`           | 8000        | FastAPI — JWT auth, orchestrator, admin routes    |
| `rag_service`       | 8001 (internal) | FastAPI — HTTP wrapper over `epsilon_rag`     |
| `postgres`          | 5433        | `pgvector/pg16` — control schema + RAG tables     |
| `minio`             | 9000 / 9001 | Object store for raw PDFs, markdown, metadata     |
| `langfuse_web`      | 3000        | Langfuse UI — query / ingest traces               |
| `langfuse_worker`   | internal    | Background event processor for Langfuse           |
| `langfuse_postgres` | internal    | Langfuse metadata store (separate from app DB)    |
| `clickhouse`        | internal    | Trace event store backing Langfuse                |
| `redis`             | internal    | Queue + cache for Langfuse                        |

Data flow for a chat query:

```
Gradio  ─Bearer JWT─▶  backend  ─HTTP─▶  rag_service  ──▶  Postgres (chunks + vectors)
```

Data flow for an ingest:

```
Gradio  ─PDF upload─▶  backend  ─writes─▶  shared /uploads volume
                                            │
                                            ▼
                                       rag_service  ─pipeline─▶  Postgres + MinIO
```

The pipeline (`epsilon_rag/`) is the read-only RAG engine: Docling layout
→ RapidOCR (PP-OCRv3) → BLIP figure captions → pix2tex formula recognition →
`bge-m3` hybrid embeddings (dense + sparse) → `pgvector` persistence,
with weighted RRF + `bge-reranker-v2-m3` cross-encoder at query time.

---

## Walkthrough — end-to-end example

After the stack is up and you can see the login screen, this exercise
takes a fresh tenant from creation to a working query.

### 1. Log in as the admin

```
Username: admin
Password: 1234
```

You'll see four tabs: **Chat**, **Channels**, **Reports**, **Ingest PDF**.

### 2. Create a new channel

In the **Channels** tab, scroll to **Add a new channel**:

- **Channel name:** `Acme`
- **Description:** `Demo tenant for ACME Corp`
- **Authorized features:** check `chatbot`, `document_search`
- Click **Add channel** — the overview table refreshes and `Acme` appears
  with `— not linked —` in the RAG column.

### 3. Link a RAG channel to it

Scroll to **Create & link a RAG channel**:

- **Unlinked control channel:** pick `Acme`
- Leave name and description blank
- Click **Create & link** — the overview row for `Acme` now shows a UUID.

### 4. Ingest a PDF

Switch to the **Ingest PDF** tab:

- **Channel:** `Acme` (only linked channels appear here)
- **Upload PDF:** pick something small (5–10 pages for your first run)
- Click **Ingest document**

The status box shows `Ingestion complete. Report ID: ..., Pages: N/N,
Chunks: M, Latency: ... ms` when the pipeline finishes. Watch
`docker compose logs -f rag_service` if you want to see the per-stage
progress.

### 5. Verify in the Reports tab

In the **Reports** tab, pick `Acme` from the dropdown and click **Load
reports**. The table shows your uploaded PDF with its page and chunk
counts.

### 6. Add a user for the new tenant (optional)

The seed only ships users for Epsilon AI, Aman, and Swedy. To add a
non-admin user for `Acme`, exec into Postgres:

```bash
docker compose exec postgres psql -U postgres -d epsilon -c \
  "INSERT INTO control.users (name, username, password, channelid,
     authorized_features, authorized_data)
   VALUES ('Acme User', 'acme_user', '1111',
     (SELECT channelid FROM control.channel WHERE name='Acme'),
     '[\"chatbot\", \"document_search\"]', '{}');"
```

### 7. Query as a normal user

Log out from the admin session. Log back in as the user you just added
(or as `omar` / `1111` if you want to test against Epsilon AI):

```
Username: acme_user
Password: 1111
```

Only the **Chat** tab is visible (no admin features). Ask any question
about the PDF you ingested. The reply lists the top-K matching chunks
with page numbers and rerank scores — there's no LLM in this phase, so
no synthesised prose. Chunk metadata is the deliverable.

---

## Useful URLs and commands

| URL or command                                  | What it is                       |
|-------------------------------------------------|----------------------------------|
| <http://localhost:7860>                         | Gradio UI                        |
| <http://localhost:8000/health>                  | Backend liveness                 |
| <http://localhost:9001>                         | MinIO console (`minioadmin`/`minioadmin`) |
| `docker compose ps`                             | Container status                 |
| `docker compose logs -f <service>`              | Tail one service                 |
| `docker compose down -v`                        | Stop **and wipe** the DB volume  |

`docker compose down -v` is the easiest way to reset the seed if you've
been hacking on the SQL — the init scripts only run when the
`postgres_data` volume is empty.

---

## Observability — Langfuse

Every `/query` and `/ingest` call against `rag_service` opens a Langfuse
trace with per-stage child spans (embed → hybrid SQL → rerank → neighbour
expansion for queries; raw-upload → extract → chunk → embed → persist →
metadata-upload for ingests). Each span carries its own latency and the
parent trace carries the input, output, and tenant ID, so a slow query
in production can be opened in the UI and the offending stage is the
slowest bar in the timeline.

**User + session attribution.** The backend forwards the caller's
`username` and JWT `session_id` to `rag_service` on every request. In
Langfuse this means:

- The **Users** tab shows one row per `username` with their query count,
  latency p50/p95, and a drill-down into every trace they produced.
- The **Sessions** tab groups all traces from a single login under one
  session ID, so a multi-turn chat session reads as a single timeline.
- The **Traces** tab is filterable by user, session, or tag (`query` vs
  `ingest`) for quick triage.

### First-time setup

The Langfuse stack boots empty — there's no project, no API keys. Two
minutes of bootstrap:

1. **Start the stack.** Tracing is off until keys are set, so the rag
   service still answers requests with `LANGFUSE_PUBLIC_KEY` blank.
   ```bash
   docker compose up -d --build
   ```
2. **Open the Langfuse UI** at <http://localhost:3000>, create the
   first admin account (it's local-only — credentials don't leave your
   machine), then create an organisation and a project.
3. **Copy the keys.** Project → Settings → API Keys → "Create new API
   key". Paste the **Public** and **Secret** values into `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```
4. **Recreate rag_service** so it picks up the new keys. Use
   `up -d`, not `restart` — `docker compose restart` reuses the existing
   container with its original env, so a fresh `.env` would be ignored:
   ```bash
   docker compose up -d rag_service
   ```
   Verify the keys made it into the container:
   ```bash
   docker compose exec rag_service printenv LANGFUSE_PUBLIC_KEY
   ```
   The startup log should now show `langfuse: client ready (host=...)`
   instead of `tracing disabled`.
5. **Make a query** from the Gradio UI. A new trace appears in the
   Langfuse UI within a few seconds.

### Production hardening

The placeholders in `.env` (`LANGFUSE_NEXTAUTH_SECRET`,
`LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY`) are demo-grade. Before
running this somewhere other than your laptop, regenerate them:

```bash
openssl rand -hex 32   # → LANGFUSE_ENCRYPTION_KEY (64 hex chars)
openssl rand -base64 32 # → LANGFUSE_SALT and LANGFUSE_NEXTAUTH_SECRET
```

Rotating `LANGFUSE_ENCRYPTION_KEY` after data exists requires running
Langfuse's key-rotation flow; do it before first boot if you can.

---

## What's intentionally not in this phase

- **No LLM answer generation.** Chat returns chunk metadata, not prose.
- **Plaintext passwords** in the user seed.
- **No per-token session revocation** — logout writes `ended_at` to the
  session row but the JWT keeps working until it expires.
- **Voice agent** is a feature-flag label only; there is no implementation.
