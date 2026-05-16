"""Figure captioning — BLIP-base via transformers.

Model choice
============
`Salesforce/blip-image-captioning-base` is the smallest VLM that
- works cleanly on transformers ≥ 4.40 (no `trust_remote_code`),
- ships ~990 MB of weights (cold-start friendly),
- produces a one-line caption that's good enough to make figure regions
  searchable, which is the whole point of captioning here.

We tried Florence-2 (transformers config breaks at ≥ 4.50) and managed
Vertex Gemini Vision (cloud dependency the user wanted gone). BLIP is
the deliberate "do what works" pick. Swap to BLIP-2 or Qwen2-VL by
overriding `figures_model_id` in .env — the code is generic over any
HF AutoProcessor / AutoModelForVision2Seq pair.

Failure model
=============
init() flips `_init_attempted` before doing any work; if the load
fails, `is_ready()` stays False and caption*/caption_crop return "".
The orchestrator treats empty captions as a no-op (a "[Figure]" stub +
OCR'd legend text still gets embedded), so a missing model degrades
gracefully rather than failing the page.
"""
from __future__ import annotations

import io
import logging
import threading
import time
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
    """Load BLIP onto the configured device. Idempotent."""
    global _processor, _model, _device, _init_attempted
    with _lock:
        if _init_attempted:
            return
        _init_attempted = True

        try:
            from transformers import (
                AutoProcessor,
                BlipForConditionalGeneration,
            )
        except ImportError as exc:
            logger.exception("figures: transformers not installed: %s", exc)
            return

        _device = settings.device if torch.cuda.is_available() else "cpu"
        model_id = settings.figures_model_id
        logger.info("figures: loading model=%s device=%s", model_id, _device)

        try:
            _processor = AutoProcessor.from_pretrained(model_id)
            model = BlipForConditionalGeneration.from_pretrained(model_id)
            if settings.fp16 and _device == "cuda":
                # BLIP base is small enough that fp16 isn't strictly needed
                # for memory, but it gives a ~1.5× speedup on inference.
                model = model.half()
            model = model.to(_device)
            model.eval()
            _model = model
            logger.info("figures: ready")
        except Exception as exc:
            logger.exception("figures: failed to load model: %s", exc)


def is_ready() -> bool:
    return _init_attempted and _model is not None and _processor is not None


# ── Public API ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def caption(img: bytes | np.ndarray | Image.Image) -> str:
    """Return a one-line caption for a figure region.

    Returns "" on any failure (decode, OOM, model error) — the
    orchestrator falls back to a "[Figure]" stub + OCR'd legends in
    that case, so empty is a safe-degraded value, not a hard error.

    Observability: every successful call logs dims + elapsed ms + an
    80-char preview. Each silent-empty path logs a distinct WARNING
    so geometry / model regressions are visible in one log line.
    """
    if not is_ready():
        return ""

    pil = _coerce_to_pil(img)
    if pil is None:
        return ""

    started = time.perf_counter()
    img_w, img_h = pil.size
    logger.info(
        "figures: captioning image=%dx%d model=%s",
        img_w, img_h, settings.figures_model_id,
    )

    try:
        inputs = _processor(images=pil, return_tensors="pt").to(_device)
        if settings.fp16 and _device == "cuda":
            # processor returns float32 even when the model is fp16;
            # cast image tensors explicitly to keep the matmul on fp16.
            inputs = {
                k: (v.half() if v.dtype == torch.float32 else v)
                for k, v in inputs.items()
            }
        out = _model.generate(  # type: ignore[union-attr]
            **inputs,
            max_new_tokens=settings.figure_caption_max_tokens,
            num_beams=settings.figure_caption_beams,
        )
        decoded = _processor.batch_decode(out, skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning(
            "figures: OOM at image=%dx%d: %s", img_w, img_h, exc,
        )
        torch.cuda.empty_cache()
        return ""
    except Exception as exc:
        logger.warning(
            "figures: caption failed at image=%dx%d: %s", img_w, img_h, exc,
        )
        return ""

    text = (decoded[0] if decoded else "").strip()
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if not text:
        # BLIP produced something but it stripped to empty (whitespace-
        # only). Usually means the crop was visually featureless — but
        # also catches model regressions, so log distinctly.
        logger.warning(
            "figures: caption decoded to empty after strip "
            "(image=%dx%d, elapsed=%dms)",
            img_w, img_h, elapsed_ms,
        )
        return ""

    preview = text[:80]
    logger.info(
        "figures: ok — %d chars, %d ms — preview: %s",
        len(text), elapsed_ms, preview,
    )
    return text


def caption_crop(
    page_img: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> str:
    """Crop the figure region off the page and run captioning. Same
    fail-soft contract as `caption()`.

    Logs the input bbox and the clamped crop dimensions so geometry
    bugs (wrong-unit, wrong-origin) surface in logs rather than
    silently producing 0-size crops that BLIP then captions as empty.
    """
    if page_img is None or page_img.size == 0:
        return ""

    page_h, page_w = page_img.shape[:2]
    raw = tuple(float(v) for v in bbox)
    x0, y0, x1, y1 = (int(round(v)) for v in raw)
    x0, x1 = max(0, x0), min(page_w, x1)
    y0, y1 = max(0, y0), min(page_h, y1)
    crop_w, crop_h = x1 - x0, y1 - y0
    logger.info(
        "figures: caption_crop cropping bbox=%s page=%dx%d → crop=%dx%d",
        raw, page_w, page_h, max(0, crop_w), max(0, crop_h),
    )

    if x1 <= x0 or y1 <= y0:
        logger.warning(
            "figures: caption_crop bbox clamped to empty "
            "(raw=%s page=%dx%d) — upstream bbox / origin / unit mismatch?",
            raw, page_w, page_h,
        )
        return ""

    crop = page_img[y0:y1, x0:x1]
    return caption(crop)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _coerce_to_pil(img: bytes | np.ndarray | Image.Image) -> Image.Image | None:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        try:
            return Image.fromarray(img).convert("RGB")
        except Exception as exc:
            logger.warning("figures: failed to convert ndarray: %s", exc)
            return None
    if isinstance(img, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(img)).convert("RGB")
        except Exception as exc:
            logger.warning("figures: failed to decode bytes: %s", exc)
            return None
    return None
