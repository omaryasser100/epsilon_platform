"""Hybrid retrieval — query text → top-K reranked chunks.

Pipeline
========
1. Embed the query with bge-m3 (dense + sparse, single forward pass).

2. Optional Rocchio pseudo-relevance feedback (PRF). When enabled, the
   first stage runs a cheap retrieval, blends the centroid of the top-N
   hits back into the dense query vector
       expanded = (1 - beta) * original + beta * mean(top_N_vecs)
   then re-runs retrieval with the expanded vector. No LLM in the loop;
   just one extra round-trip. Effective on short queries where the
   user's wording is sparse compared to the chunk vocabulary.

3. Pull two candidate lists from Postgres in one round-trip:
     • dense:  ORDER BY embedding <=> $dense_vec        LIMIT N
     • sparse: ORDER BY sparse_embedding <#> $sparse_vec LIMIT N
   `<=>` is cosine distance, `<#>` is negative inner product on sparsevec
   (both smaller = closer; both use the HNSW indexes).

4. Fuse the two rankings with weighted Reciprocal Rank Fusion (RRF):
       rrf(d) = alpha * 1/(k + dense_rank(d))
              + (1-alpha) * 1/(k + sparse_rank(d))
   alpha = 0.5 is plain RRF. Raise it when dense should win (most
   conceptual queries); lower it for keyword-heavy queries where the
   lexical channel is more trustworthy.

5. Optional cross-encoder rerank on the fused top pool. With chunk
   overlap dropped to ~50 tokens, near-duplicates are rare enough that
   an MMR diversity pass between RRF and rerank wasn't earning its
   latency, so it's no longer wired here.

6. Optional neighbour expansion — for each survivor, also return its
   ±n adjacent chunks (same report, chunk_index ± 1…n) so the answer
   layer has surrounding context, not isolated fragments.

Returns
=======
A list of dicts:
    {
      "id":            uuid,
      "channel_id":    uuid,
      "report_id":     uuid,
      "page_number":   int,
      "chunk_index":   int,
      "content":       str,
      "metadata":      dict,           # bboxes, section_title, …
      "rrf_score":     float,
      "rerank_score":  float | None,   # None when rerank=False
      "is_neighbour":  bool,           # True for ±n expansion rows
      "neighbour_of":  str | None,     # parent chunk Id when is_neighbour
    }
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pgvector.psycopg import SparseVector

from core.config import settings
from core.db import get_conn
from pipeline import embeddings, reranker

logger = logging.getLogger(__name__)


# ── Default tuning ──────────────────────────────────────────────────────────

# How many candidates each retrieval channel pulls before fusion.
_DEFAULT_CANDIDATES = 50

# RRF "k" parameter — softens the contribution of low-rank items.
_DEFAULT_RRF_K = 60

# Dense / sparse weight in the fused score. 0.5 = balanced RRF.
_DEFAULT_ALPHA = 0.5

# Rocchio PRF blend factor. 0.0 = no expansion (skip), ~0.3 = mild
# expansion. >0.5 starts drifting away from the user's intent.
_DEFAULT_PRF_BETA = 0.0   # off by default
_DEFAULT_PRF_TOPN = 5     # centroid pool when PRF is on


# ── SQL ─────────────────────────────────────────────────────────────────────
# Two CTEs in one round-trip. Every query is channel-scoped — multi-
# tenant data isolation is a SQL invariant, not a UI convention. The
# optional report_filter narrows to a single PDF within the channel.
#
# Returns per-row (ranks + the dense embedding) so the Python side can
# do weighted RRF and optional Rocchio PRF without a second round-trip.
_HYBRID_SQL = """
WITH dense_hits AS (
    SELECT "Id",
           embedding,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %(dense)s) AS rank
    FROM report_chunks
    WHERE "ChannelId" = %(channel_id)s
      {report_filter}
    ORDER BY embedding <=> %(dense)s
    LIMIT %(candidates)s
),
sparse_hits AS (
    SELECT "Id",
           ROW_NUMBER() OVER (ORDER BY sparse_embedding <#> %(sparse)s) AS rank
    FROM report_chunks
    WHERE "ChannelId" = %(channel_id)s
      {report_filter}
    ORDER BY sparse_embedding <#> %(sparse)s
    LIMIT %(candidates)s
),
fused AS (
    SELECT "Id" AS id,
           MAX(dense_rank)  AS dense_rank,
           MAX(sparse_rank) AS sparse_rank
    FROM (
        SELECT "Id", rank AS dense_rank, NULL::bigint AS sparse_rank FROM dense_hits
        UNION ALL
        SELECT "Id", NULL::bigint AS dense_rank, rank AS sparse_rank FROM sparse_hits
    ) combined
    GROUP BY "Id"
)
SELECT rc."Id",
       rc."ChannelId",
       rc."ReportId",
       rc."PageNumber",
       rc."ChunkIndex",
       rc."Content",
       rc.metadata,
       rc.embedding,
       fused.dense_rank,
       fused.sparse_rank
FROM fused
JOIN report_chunks rc ON rc."Id" = fused.id
"""

# Neighbour chunks (chunk_index ± n) for a given report. Fetched in one
# round-trip after retrieval ranks the parents. ChannelId is in the
# WHERE clause defensively so a bad parent row can't leak into the wrong
# tenant's data.
_NEIGHBOURS_SQL = """
SELECT "Id", "ChannelId", "ReportId", "PageNumber", "ChunkIndex",
       "Content", metadata
FROM report_chunks
WHERE "ChannelId" = %(channel_id)s
  AND "ReportId"  = %(report_id)s
  AND "ChunkIndex" BETWEEN %(lo)s AND %(hi)s
ORDER BY "ChunkIndex" ASC
"""


# ── Public API ──────────────────────────────────────────────────────────────

def hybrid_query(
    text: str,
    channel_id: str,
    *,
    top_k: int = 10,
    candidates: int = _DEFAULT_CANDIDATES,
    rrf_k: int = _DEFAULT_RRF_K,
    alpha: float = _DEFAULT_ALPHA,
    prf_beta: float = _DEFAULT_PRF_BETA,
    prf_topn: int = _DEFAULT_PRF_TOPN,
    neighbours: int = 1,
    report_id: str | None = None,
    rerank: bool = True,
) -> list[dict[str, Any]]:
    """Run hybrid retrieval inside one channel and return the top-K.

    Args:
        text:        natural-language query.
        channel_id:  UUID of the channel. Required — multi-tenant
                     isolation is enforced at the SQL level.
        top_k:       final result count.
        candidates:  per-channel candidate pool before fusion. ≥ top_k.
        rrf_k:       RRF softening constant.
        alpha:       weight on dense in the fused score
                     (0.0 = sparse only, 1.0 = dense only).
        prf_beta:    Rocchio expansion strength (0.0 disables PRF).
        prf_topn:    centroid pool size when PRF is on.
        neighbours:  ±n adjacent chunks to include per result (0 = off).
        report_id:   optional narrow-scope filter to one PDF.
        rerank:      cross-encoder rerank the fused pool.

    Returns:
        Up to `top_k` chunk dicts (+ neighbour rows if `neighbours > 0`).
    """
    if not text or not text.strip():
        return []
    if not channel_id or not channel_id.strip():
        raise ValueError("channel_id is required — queries must be channel-scoped")
    if not embeddings.is_ready():
        raise RuntimeError("embedding model is not loaded — call embeddings.init() first")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not (0.0 <= prf_beta < 1.0):
        raise ValueError(f"prf_beta must be in [0, 1), got {prf_beta}")
    if neighbours < 0:
        raise ValueError(f"neighbours must be ≥ 0, got {neighbours}")

    # Give the second-stage rerank a wider pool than top_k so its
    # re-ordering has room to bubble up something the first stage
    # ranked 30th. 3× is the rule of thumb.
    fused_limit = max(top_k, candidates)
    if rerank:
        fused_limit = max(fused_limit, top_k * 3)

    dense_list, sparse_list = embeddings.embed([text], kind="query")
    if not dense_list or not sparse_list:
        raise RuntimeError("query embedding returned empty result")

    dense_vec = np.array(dense_list[0], dtype=np.float32)
    sparse_vec = SparseVector(sparse_list[0], settings.embed_sparse_dim)

    # ── Optional Rocchio PRF ────────────────────────────────────────
    # First pass with the original vector, blend in the centroid of the
    # top-N hits, then redo the SQL with the expanded dense vector.
    if prf_beta > 0.0:
        prf_rows = _run_hybrid_sql(
            dense_vec, sparse_vec, candidates, channel_id, report_id,
        )
        if prf_rows:
            centroid = _centroid([row["_embedding"] for row in prf_rows[:prf_topn]])
            if centroid is not None:
                dense_vec = _renormalise(
                    (1.0 - prf_beta) * dense_vec + prf_beta * centroid
                )
                logger.info(
                    "retrieval: PRF expanded query vector "
                    "(beta=%.2f, top_n=%d)", prf_beta, prf_topn,
                )

    rows = _run_hybrid_sql(
        dense_vec, sparse_vec, candidates, channel_id, report_id,
    )
    if not rows:
        return []

    # ── Weighted RRF over the union ─────────────────────────────────
    for row in rows:
        dense_rank  = row.pop("_dense_rank")
        sparse_rank = row.pop("_sparse_rank")
        row["rrf_score"] = _weighted_rrf(dense_rank, sparse_rank, alpha, rrf_k)
    rows.sort(key=lambda r: r["rrf_score"], reverse=True)
    rows = rows[:fused_limit]

    # ── Optional cross-encoder rerank ────────────────────────────────
    if rerank:
        rows = _rerank(text, rows, top_k=top_k)
    else:
        rows = rows[:top_k]
        for row in rows:
            row["rerank_score"] = None

    # ── Optional neighbour expansion ─────────────────────────────────
    if neighbours > 0:
        rows = _attach_neighbours(rows, channel_id, neighbours)

    # Strip the internal _embedding before returning. PRF needs it
    # during the first pass; everything after that just carries it.
    for row in rows:
        row.pop("_embedding", None)
    return rows


# ── SQL execution ───────────────────────────────────────────────────────────

def _run_hybrid_sql(
    dense_vec: np.ndarray,
    sparse_vec: SparseVector,
    candidates: int,
    channel_id: str,
    report_id: str | None,
) -> list[dict[str, Any]]:
    """Execute the hybrid retrieval CTE and return rows enriched with the
    per-channel ranks + raw embedding (used by PRF when enabled)."""
    if report_id is not None:
        sql = _HYBRID_SQL.format(report_filter='AND "ReportId" = %(report_id)s')
    else:
        sql = _HYBRID_SQL.format(report_filter="")

    params: dict[str, Any] = {
        "dense":        dense_vec,
        "sparse":       sparse_vec,
        "candidates":   candidates,
        "channel_id":   channel_id,
    }
    if report_id is not None:
        params["report_id"] = report_id

    with get_conn() as conn:
        cur = conn.execute(sql, params)
        cols = [desc[0] for desc in cur.description]
        raw = cur.fetchall()

    out: list[dict[str, Any]] = []
    for row in raw:
        d = dict(zip(cols, row))
        out.append({
            "id":           str(d["Id"]),
            "channel_id":   str(d["ChannelId"]),
            "report_id":    str(d["ReportId"]),
            "page_number":  int(d["PageNumber"]),
            "chunk_index":  int(d["ChunkIndex"]),
            "content":      d["Content"],
            "metadata":     d.get("metadata") or {},
            # Internal-only — stripped before returning to the caller.
            "_embedding":   np.asarray(d["embedding"], dtype=np.float32),
            "_dense_rank":  d.get("dense_rank"),
            "_sparse_rank": d.get("sparse_rank"),
        })
    return out


# ── Fusion + diversity helpers ──────────────────────────────────────────────

def _weighted_rrf(
    dense_rank: int | None,
    sparse_rank: int | None,
    alpha: float,
    rrf_k: int,
) -> float:
    """Weighted RRF score. A missing rank contributes 0 (NULL from the
    UNION when only one channel surfaced that ID)."""
    dense_part  = alpha       * (1.0 / (rrf_k + dense_rank))  if dense_rank  else 0.0
    sparse_part = (1.0-alpha) * (1.0 / (rrf_k + sparse_rank)) if sparse_rank else 0.0
    return dense_part + sparse_part


def _centroid(vecs: list[np.ndarray]) -> np.ndarray | None:
    """Mean of a non-empty list of vectors. Returns None when the input
    is empty — caller skips PRF in that case."""
    if not vecs:
        return None
    return np.mean(np.stack(vecs), axis=0)


def _renormalise(vec: np.ndarray) -> np.ndarray:
    """Re-L2-normalise after PRF blending so cosine similarity stays the
    intended metric. bge-m3 emits unit vectors; the blended result
    isn't unit-length anymore."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec
    return (vec / norm).astype(np.float32)


# ── Reranker ────────────────────────────────────────────────────────────────

def _rerank(
    text: str,
    rows: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Score (query, content) pairs with the cross-encoder and re-order.

    Falls back to the incoming order when the reranker isn't loaded or
    returns nothing, so a degraded reranker doesn't blank the response.
    """
    if not reranker.is_ready():
        logger.warning(
            "retrieval: reranker not ready — returning fused order without rerank"
        )
        for row in rows[:top_k]:
            row["rerank_score"] = None
        return rows[:top_k]

    passages = [row["content"] for row in rows]
    ranked = reranker.rerank(text, passages, top_k=top_k)
    if not ranked:
        logger.warning("retrieval: reranker returned empty list — falling back")
        for row in rows[:top_k]:
            row["rerank_score"] = None
        return rows[:top_k]

    out: list[dict[str, Any]] = []
    for orig_idx, score in ranked:
        row = dict(rows[orig_idx])
        row["rerank_score"] = float(score)
        out.append(row)
    return out


# ── Neighbour expansion ─────────────────────────────────────────────────────

def _attach_neighbours(
    rows: list[dict[str, Any]],
    channel_id: str,
    n: int,
) -> list[dict[str, Any]]:
    """Fetch chunk_index ± n for every result row, dedupe against the
    original results, and slot them in next to their parent. Parents
    retain their order; neighbours sort by chunk_index inside each group.

    Uses one round-trip per (report_id) bucket — typically just a handful
    of bucket queries even when top_k is large.
    """
    if not rows or n <= 0:
        return rows

    # Group result rows by report_id; mark each as the "anchor" so we
    # don't refetch its own row as a neighbour.
    by_report: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row.setdefault("is_neighbour", False)
        row.setdefault("neighbour_of", None)
        by_report.setdefault(row["report_id"], []).append(row)

    final: list[dict[str, Any]] = []
    with get_conn() as conn:
        for report_id, parents in by_report.items():
            anchor_indices = {p["chunk_index"] for p in parents}
            lo = max(0, min(p["chunk_index"] for p in parents) - n)
            hi = max(p["chunk_index"] for p in parents) + n
            cur = conn.execute(
                _NEIGHBOURS_SQL,
                {"channel_id": channel_id,
                 "report_id":  report_id,
                 "lo": lo, "hi": hi},
            )
            cols = [desc[0] for desc in cur.description]
            window = [dict(zip(cols, r)) for r in cur.fetchall()]

            # Map chunk_index → row for fast lookup.
            by_idx = {int(r["ChunkIndex"]): r for r in window}

            # Interleave: for each parent, prepend N before and append N
            # after, but skip rows that are themselves anchors (don't
            # duplicate). Keeps original parent order.
            for parent in parents:
                center = parent["chunk_index"]
                # Before — ascending idx ending right before the parent.
                for off in range(n, 0, -1):
                    cand = by_idx.get(center - off)
                    if cand and int(cand["ChunkIndex"]) not in anchor_indices:
                        final.append(_neighbour_dict(cand, parent["id"]))
                final.append(parent)
                # After — ascending idx starting right after the parent.
                for off in range(1, n + 1):
                    cand = by_idx.get(center + off)
                    if cand and int(cand["ChunkIndex"]) not in anchor_indices:
                        final.append(_neighbour_dict(cand, parent["id"]))

    return final


def _neighbour_dict(row: dict, parent_id: str) -> dict[str, Any]:
    """Wrap a neighbour-window row as a public result dict. Score
    fields stay None because neighbours weren't ranked individually."""
    return {
        "id":           str(row["Id"]),
        "channel_id":   str(row["ChannelId"]),
        "report_id":    str(row["ReportId"]),
        "page_number":  int(row["PageNumber"]),
        "chunk_index":  int(row["ChunkIndex"]),
        "content":      row["Content"],
        "metadata":     row.get("metadata") or {},
        "rrf_score":    None,
        "rerank_score": None,
        "is_neighbour": True,
        "neighbour_of": parent_id,
    }
