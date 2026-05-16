"""Cross-encoder reranker — (query, passage) → relevance score on GPU.

Why this exists
===============
After hybrid retrieval (dense + BM25 RRF) returns ~20 candidate chunks, a
cross-encoder reranker scores each (query, chunk) pair *jointly*. Bi-encoders
like bge-m3 score query and passage independently and combine via cosine,
which loses fine-grained relevance signals. Cross-encoders see both texts at
once and can tell apart "almost relevant" from "actually answers the question".

Model contract
==============
- BAAI/bge-reranker-v2-m3 — multilingual (incl. Arabic), trained as a sibling
  to bge-m3. Same checkpoint family, ~568M params, ~2.5 GB VRAM.
- Input: (query: str, passage: str) pairs.
- Output: relevance score (typically [-10, 10] for the raw model — we apply
  sigmoid to map to [0, 1] so callers can compare across reranker swaps
  without retuning thresholds).

Failure model
=============
init() flips `_init_attempted` before doing any work. If the load fails,
is_ready() returns False and `rerank()` returns an empty list — callers
fall back to the original retrieval order. The reranker is a quality
booster, not a hard dependency.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from core.config import settings

logger = logging.getLogger(__name__)


# ── Module state ────────────────────────────────────────────────────────────

_model: Optional["CrossEncoder"] = None  # type: ignore[name-defined]
_device: Optional[str] = None
_init_attempted: bool = False


def init() -> None:
    """Load the cross-encoder onto the configured device. Idempotent."""
    global _model, _device, _init_attempted
    if _init_attempted:
        return
    _init_attempted = True

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        logger.exception("reranker: sentence-transformers not installed: %s", exc)
        return

    _device = settings.device if torch.cuda.is_available() else "cpu"
    logger.info(
        "reranker: loading model=%s device=%s max_seq=%d",
        settings.rerank_model_id, _device, settings.rerank_max_seq_length,
    )

    try:
        # CrossEncoder is the right wrapper for bge-reranker-v2-m3 (a
        # CausalLM under the hood, but we use it in pair-scoring mode).
        # automodel_args forwards fp16 weights to the GPU on load — saves a
        # second .half() call afterward.
        kwargs: dict = {
            "max_length": settings.rerank_max_seq_length,
            "device": _device,
        }
        if settings.fp16 and _device == "cuda":
            kwargs["automodel_args"] = {"torch_dtype": torch.float16}

        model = CrossEncoder(settings.rerank_model_id, **kwargs)
        _model = model
        logger.info("reranker: ready")
    except Exception as exc:
        logger.exception("reranker: failed to load model: %s", exc)


def is_ready() -> bool:
    return _init_attempted and _model is not None


# ── Public API ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def rerank(
    query: str,
    passages: list[str],
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Score each (query, passage) pair and return the top-k by score.

    Args:
        query: the user's question.
        passages: list of candidate passages (chunk content strings).
        top_k:   number of items to keep. Defaults to len(passages).

    Returns:
        List of (original_index, score) tuples in descending score order,
        truncated to top_k. Score is sigmoid-normalised to [0, 1] so it
        compares across reranker model swaps without retuning thresholds.
        Returns an empty list when the model failed to load.
    """
    if not is_ready():
        logger.warning("reranker: rerank() called before init succeeded")
        return []
    if not passages:
        return []

    keep = top_k if top_k is not None else len(passages)
    pairs = [(query, p or "") for p in passages]

    try:
        # batch_size controls VRAM pressure at scoring time. bge-reranker-v2-m3
        # is fp16 ~2.5 GB; batch=8 gives ~30-50 ms total for 20 candidates.
        scores = _model.predict(  # type: ignore[union-attr]
            pairs,
            batch_size=settings.rerank_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning(
            "reranker: OOM at batch=%d: %s", settings.rerank_batch_size, exc,
        )
        torch.cuda.empty_cache()
        return []
    except Exception as exc:
        logger.exception("reranker: predict failed: %s", exc)
        return []

    # Sigmoid mapping → [0, 1]. Stable form to avoid overflow on extreme
    # logits. Numpy-compatible; works whether predict returned ndarray or list.
    import numpy as np
    raw = np.asarray(scores, dtype=np.float64)
    norm = 1.0 / (1.0 + np.exp(-raw))

    ranked = sorted(
        ((i, float(norm[i])) for i in range(len(pairs))),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:keep]
