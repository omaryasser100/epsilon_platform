"""CLI entrypoint — hybrid-retrieve the top-K chunks for a query.

Loads only the two models needed at query time (the bge-m3 hybrid
embedder and the bge-reranker-v2-m3 cross-encoder), opens the DB pool,
runs the retrieval pipeline, and prints the top hits.

Every query is scoped to a channel: the SQL CTE enforces
`WHERE "ChannelId" = $1`, so two companies' data is hard-isolated by
the database, not just by application logic.

Usage
=====
    python query.py "your question" --channel-id <uuid>
    python query.py "..." --channel-id <uuid> --top-k 5
    python query.py "..." --channel-id <uuid> --report-id <uuid>   # one PDF
    python query.py "..." --channel-id <uuid> --no-rerank          # raw RRF
    python query.py "..." --channel-id <uuid> --candidates 100     # wider first stage
    python query.py "..." --channel-id <uuid> --json               # machine-readable

Stages
======
  1. Warm up bge-m3 (dense + sparse heads) and bge-reranker-v2-m3.
     Docling / OCR / pix2tex / figures stay cold — not needed for query.
  2. Open the PostgreSQL connection pool.
  3. Run [pipeline.retrieval.hybrid_query](pipeline/retrieval.py):
     embed → channel-scoped dense+sparse retrieval → weighted RRF
     fusion → optional cross-encoder rerank → optional neighbour
     expansion.
  4. Drain the pool and print results.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from textwrap import shorten

from core.config import settings
from core.db import close_pool, init_pool
from pipeline import embeddings, registry, reranker
from pipeline.retrieval import hybrid_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("query")


# ── Warmup ──────────────────────────────────────────────────────────────────

def _warmup() -> None:
    """Load only the models the query path needs — embedder + reranker."""
    started = time.perf_counter()
    for name, fn in [
        ("embeddings",  embeddings.init),
        ("reranker",    reranker.init),
    ]:
        stage_started = time.perf_counter()
        try:
            fn()
            logger.info(
                "warmup: %s ready in %.1fs",
                name, time.perf_counter() - stage_started,
            )
        except Exception as exc:
            logger.exception("warmup: %s failed to initialise: %s", name, exc)
    logger.info("warmup: total %.1fs", time.perf_counter() - started)


# ── CLI plumbing ────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid-retrieve chunks for a query against the local rag_db.",
    )
    parser.add_argument("query", type=str, help="The user question / search query.")
    parser.add_argument(
        "--channel-id", type=str, required=True,
        help="UUID of the channel to search. Required — retrieval is "
             "always channel-scoped.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of results to return after reranking (default: 10).",
    )
    parser.add_argument(
        "--candidates", type=int, default=50,
        help="Per-channel candidate pool before RRF fusion (default: 50).",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=60,
        help="RRF softening constant (default: 60).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight on the dense channel in the RRF score (0.0 = sparse "
             "only, 1.0 = dense only, 0.5 = balanced; default 0.5).",
    )
    parser.add_argument(
        "--prf-beta", type=float, default=0.0,
        help="Rocchio pseudo-relevance feedback strength (0.0 = off, "
             "~0.3 = mild expansion; >0.5 starts drifting from the user's "
             "intent). Adds one round-trip when > 0.",
    )
    parser.add_argument(
        "--prf-topn", type=int, default=5,
        help="Centroid pool size when --prf-beta > 0 (default 5).",
    )
    parser.add_argument(
        "--neighbours", "--neighbors", type=int, default=1, dest="neighbours",
        help="Also return ±N adjacent chunks per result for context "
             "(default 1; pass 0 to disable).",
    )
    parser.add_argument(
        "--report-id", type=str, default=None,
        help="Narrow retrieval to a single report (PDF) within the channel.",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Skip the cross-encoder; return raw fused order.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON instead of the human-readable preview format.",
    )
    return parser.parse_args(argv)


def _validate_uuid(value: str, label: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise SystemExit(f"{label} is not a valid UUID: {exc}")


# ── Output formatting ───────────────────────────────────────────────────────

def _print_human(results: list[dict], query_text: str) -> None:
    """Print results in a grep-friendly preview format. Long content is
    shortened to one line; full content lives in the JSON output mode.

    Neighbour rows (added by --neighbours) render indented under their
    parent, with a `~` marker and no score so they're easy to skim past.
    """
    if not results:
        print(f'\nNo results for: "{query_text}"\n')
        return

    primary_rows = [r for r in results if not r.get("is_neighbour")]
    has_rerank = any(r.get("rerank_score") is not None for r in primary_rows)

    print()
    print("─" * 72)
    print(f'  Query: "{query_text}"')
    print(f"  {len(primary_rows)} hits ({len(results)} rows incl. neighbours)  "
          f"hybrid + RRF{' + rerank' if has_rerank else ''}")
    print("─" * 72)

    rank = 0
    for row in results:
        section = (row.get("metadata") or {}).get("section_title") or ""
        section_suffix = f"  «{section}»" if section else ""
        preview = " ".join(row["content"].split())

        if row.get("is_neighbour"):
            print(
                f"       ~ neighbour  page={row['page_number']}  "
                f"chunk={row['chunk_index']}{section_suffix}"
            )
            print(f"         {shorten(preview, width=200, placeholder=' …')}")
            continue

        rank += 1
        if row.get("rerank_score") is not None:
            score_label = f"score={row['rerank_score']:.3f}"
        elif row.get("rrf_score") is not None:
            score_label = f"rrf={row['rrf_score']:.4f}"
        else:
            score_label = "—"
        print(
            f"\n  [{rank:>2}] {score_label}  "
            f"report={row['report_id'][:8]}…  "
            f"page={row['page_number']}  "
            f"chunk={row['chunk_index']}{section_suffix}"
        )
        print(f"       {shorten(preview, width=220, placeholder=' …')}")
    print()


def _print_json(results: list[dict]) -> None:
    """Machine-readable output for piping into other tools."""
    print(json.dumps(results, ensure_ascii=False, indent=2))


# ── Entry point ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.query.strip():
        print("error: query cannot be empty", file=sys.stderr)
        return 2

    channel_id = _validate_uuid(args.channel_id, "--channel-id")
    report_id = _validate_uuid(args.report_id, "--report-id") if args.report_id else None

    if not settings.database_url:
        print(
            "error: DATABASE_URL is not configured — set it in .env or the "
            "environment before running the CLI.",
            file=sys.stderr,
        )
        return 2

    _warmup()
    init_pool()

    try:
        # Sanity-check the channel exists before paying for embedding —
        # a typo on the UUID would otherwise silently return zero
        # results, which is indistinguishable from "no chunks match".
        if registry.get_channel(channel_id) is None:
            print(
                f"error: channel {channel_id} does not exist.",
                file=sys.stderr,
            )
            return 2

        results = hybrid_query(
            args.query,
            channel_id=channel_id,
            top_k=args.top_k,
            candidates=args.candidates,
            rrf_k=args.rrf_k,
            alpha=args.alpha,
            prf_beta=args.prf_beta,
            prf_topn=args.prf_topn,
            neighbours=args.neighbours,
            report_id=report_id,
            rerank=not args.no_rerank,
        )
    except Exception as exc:
        logger.exception("query failed: %s", exc)
        return 1
    finally:
        close_pool()

    if args.json:
        _print_json(results)
    else:
        _print_human(results, args.query)
    return 0


if __name__ == "__main__":
    sys.exit(main())
