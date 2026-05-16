"""Formula image → LaTeX via a HuggingFace VisionEncoderDecoderModel.

Why not pix2tex
===============
pix2tex 0.1.4 (the package the original code targeted) pins
`x-transformers==0.15.0`, which segfaults at import when combined with
`transformers >= 4.42`. We need transformers 4.50 for docling's
RT-DETRv2 layout model, so pix2tex isn't installable here.

This module loads a generic VisionEncoderDecoderModel from HF
(`settings.formulas_model_id`) instead — no extra top-level package,
no x-transformers, no dep conflict. Default model is
`breezedeus/pix2text-mfr` (math formula recognition only, ~340 MB),
which is what's left after stripping the page-level OCR head that the
full pix2text pipeline ships.

Failure model
=============
A failed load or a model that turns out to be incompatible with the
current transformers version doesn't crash the service — init() logs
the exception, is_ready() stays False, and to_latex / to_latex_crop
return "". The orchestrator treats empty LaTeX the same as a
formula-region that produced nothing: it drops the block. So formula
extraction degrades from "rich math" to "skipped region" rather than
from "rich math" to "pipeline crash".
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Optional

import numpy as np
import torch
from PIL import Image

from core.config import settings

logger = logging.getLogger(__name__)


# ── Module state ────────────────────────────────────────────────────────────

_processor = None         # type: ignore[var-annotated]
_model = None             # type: ignore[var-annotated]
_device: Optional[str] = None
_init_attempted: bool = False
_lock = threading.Lock()


def init() -> None:
    """Load the formula-OCR model onto the configured device. Idempotent."""
    global _processor, _model, _device, _init_attempted
    with _lock:
        if _init_attempted:
            return
        _init_attempted = True

        try:
            from transformers import (
                AutoProcessor,
                VisionEncoderDecoderModel,
            )
        except ImportError as exc:
            logger.exception("formulas: transformers not installed: %s", exc)
            return

        _device = settings.device if torch.cuda.is_available() else "cpu"
        model_id = settings.formulas_model_id
        logger.info("formulas: loading model=%s device=%s", model_id, _device)

        try:
            _processor = AutoProcessor.from_pretrained(model_id)
            model = VisionEncoderDecoderModel.from_pretrained(model_id)
            if settings.fp16 and _device == "cuda":
                model = model.half()
            model = model.to(_device)
            model.eval()
            _model = model
            logger.info("formulas: ready")
        except Exception as exc:
            # Common causes: model id not found on HF, network unreachable
            # at first run, processor/model architecture mismatch with
            # transformers version. None are fatal — the orchestrator just
            # skips formula regions.
            logger.warning(
                "formulas: failed to load model=%s (%s) — "
                "formula regions will be skipped",
                model_id, exc,
            )


def is_ready() -> bool:
    return _init_attempted and _model is not None and _processor is not None


# ── Public API ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def to_latex(img: bytes | np.ndarray | Image.Image) -> str:
    """Convert a formula image into a LaTeX string. Returns "" on any
    failure — empty is a safe degraded value the orchestrator handles."""
    if not is_ready():
        return ""

    pil = _coerce_to_pil(img)
    if pil is None:
        return ""

    try:
        inputs = _processor(images=pil, return_tensors="pt").to(_device)
        if settings.fp16 and _device == "cuda":
            inputs = {
                k: (v.half() if v.dtype == torch.float32 else v)
                for k, v in inputs.items()
            }
        out = _model.generate(  # type: ignore[union-attr]
            **inputs,
            max_new_tokens=settings.formulas_max_tokens,
            num_beams=settings.formulas_beams,
        )
        text = _processor.batch_decode(out, skip_special_tokens=True)
        latex = (text[0] if text else "").strip()
        # Some formula-OCR models emit the LaTeX without delimiters, others
        # wrap in $..$. Strip outer single-dollar delimiters so the
        # orchestrator's `$$ ... $$` wrap doesn't double-up.
        if latex.startswith("$") and latex.endswith("$"):
            latex = latex.strip("$").strip()
        return latex
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning("formulas: OOM: %s", exc)
        torch.cuda.empty_cache()
        return ""
    except Exception as exc:
        logger.warning("formulas: inference failed: %s", exc)
        return ""


def to_latex_crop(
    page_img: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> str:
    """Crop a region off the page and run formula OCR. Same fail-soft
    contract as to_latex()."""
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    h, w = page_img.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = page_img[y0:y1, x0:x1]
    return to_latex(crop)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _coerce_to_pil(img: bytes | np.ndarray | Image.Image) -> Image.Image | None:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        try:
            return Image.fromarray(img).convert("RGB")
        except Exception as exc:
            logger.warning("formulas: failed to convert ndarray: %s", exc)
            return None
    if isinstance(img, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(img)).convert("RGB")
        except Exception as exc:
            logger.warning("formulas: failed to decode bytes: %s", exc)
            return None
    return None
