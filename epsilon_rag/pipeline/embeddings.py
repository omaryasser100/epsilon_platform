"""bge-m3 hybrid embedder — emits both the dense head and the lexical
(sparse) head in a single forward pass.

Model contract
==============
- Dense: 1024-dim L2-normalised float vectors (matches
  `report_chunks.embedding vector(1024)`).
- Sparse: dict[int, float] keyed by XLM-RoBERTa token_id with the
  learned lexical weight (matches `report_chunks.sparse_embedding
  sparsevec(250002)`).
- Multilingual, Arabic-first. Identical input format for queries and
  passages — no E5-style prefixes. The `kind` parameter is accepted for
  call-site stability but is a no-op for bge-m3.

Why FlagEmbedding (not sentence-transformers)
=============================================
sentence-transformers' `SentenceTransformer.encode()` exposes only the
dense head. The sparse and ColBERT heads of bge-m3 live behind extra
linear layers that are only wired up by `BGEM3FlagModel` from the
`FlagEmbedding` library. We need the sparse head, so we use that loader
here. The cross-encoder reranker in `pipeline/reranker.py` still uses
sentence-transformers — that's a separate model with no sparse output.

Failure model
=============
init() flips `_init_attempted` before doing any work; if the load
fails, `is_ready()` stays False and `embed()` returns empty lists.
Callers (`pipeline/ingest.py`) treat empty as a hard failure since
embeddings are non-optional — this is different from figure
captioning, which can degrade silently.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import torch

from core.config import settings

logger = logging.getLogger(__name__)


# ── Module state ────────────────────────────────────────────────────────────

_model: Optional["BGEM3FlagModel"] = None  # type: ignore[name-defined]
_device: Optional[str] = None
_init_attempted: bool = False


# bge-m3 doesn't use task prefixes — same input format for queries and
# passages. The `kind` parameter is kept on the public API so callers
# can stay stable across future embedder swaps (e.g. an E5-family model
# that needs prefixes).
EmbedKind = Literal["query", "passage"]


def init() -> None:
    """Load the BGEM3FlagModel onto the configured device. Idempotent."""
    global _model, _device, _init_attempted
    if _init_attempted:
        return
    _init_attempted = True

    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        logger.exception("embeddings: FlagEmbedding not installed: %s", exc)
        return

    _device = settings.device if torch.cuda.is_available() else "cpu"
    use_fp16 = bool(settings.fp16 and _device == "cuda")
    logger.info(
        "embeddings: loading model=%s device=%s fp16=%s",
        settings.embed_model_id, _device, use_fp16,
    )

    try:
        _model = BGEM3FlagModel(
            settings.embed_model_id,
            use_fp16=use_fp16,
            device=_device,
        )
        logger.info("embeddings: ready (dense + sparse heads)")
    except Exception as exc:
        logger.exception("embeddings: failed to load model: %s", exc)


def is_ready() -> bool:
    return _init_attempted and _model is not None


# ── Public API ──────────────────────────────────────────────────────────────

def embed(
    texts: list[str],
    kind: EmbedKind = "passage",
) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Run both heads of bge-m3 over `texts` in one pass.

    Args:
        texts: list of UTF-8 strings, possibly empty. Empty strings still
               produce a (zero) dense vector and an empty sparse dict so
               the output lists always have the same length as the input.
        kind:  reserved for parity with prefix-using embedders; no-op here.

    Returns:
        Tuple `(dense_vectors, sparse_vectors)`:
          - `dense_vectors`: list of 1024-dim Python lists (JSON-friendly)
          - `sparse_vectors`: list of `{token_id: weight}` dicts; keys
            are ints suitable for pgvector's `SparseVector` constructor.

        Both lists are empty if the model failed to load — caller should
        surface that as an error.
    """
    if not is_ready():
        logger.warning("embeddings: embed() called before init succeeded")
        return [], []
    if not texts:
        return [], []

    inputs = [t or "" for t in texts]

    try:
        # return_dense + return_sparse runs the encoder once and forwards
        # the pooled embedding through both heads. ColBERT vecs are off
        # because we're not storing them (and they cost extra latency).
        output = _model.encode(  # type: ignore[union-attr]
            inputs,
            batch_size=settings.embed_batch_size,
            max_length=settings.embed_max_seq_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning("embeddings: OOM at batch=%d: %s", settings.embed_batch_size, exc)
        torch.cuda.empty_cache()
        return [], []
    except Exception as exc:
        logger.exception("embeddings: encode failed: %s", exc)
        return [], []

    dense_arr = output.get("dense_vecs")
    lexical = output.get("lexical_weights") or []

    if isinstance(dense_arr, np.ndarray):
        dense_list = dense_arr.astype(np.float32).tolist()
    else:
        dense_list = [list(v) for v in (dense_arr or [])]

    # FlagEmbedding returns lexical_weights as list[dict[str, float]] —
    # the keys are stringified token_ids. pgvector's SparseVector wants
    # int indices, so coerce eagerly.
    sparse_list = [
        {int(k): float(v) for k, v in (lex or {}).items()}
        for lex in lexical
    ]

    return dense_list, sparse_list
