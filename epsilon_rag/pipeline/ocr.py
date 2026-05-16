"""Surya OCR wrapper — Arabic + English text recognition.

OCR runs only on regions where Docling's text-layer extraction came
back empty (or suspiciously short). Two reasons:
    1. OCR is the slowest step per region (~50-300 ms each on GPU).
    2. Native PDF text is always more accurate than OCR'd text — never
       overwrite real text with an OCR guess.

Why Surya (and not EasyOCR / PaddleOCR / Tesseract)
====================================================
Surya is a transformer-based OCR from Datalab (the team behind Marker
and Texify). For Arabic-first content it beats the alternatives in
the dimensions we actually care about:

  - **Cursive ligatures**: Arabic letters change shape based on their
    position in a word. EasyOCR (CRNN+CTC) was trained on Latin first
    and bolts on multilingual support; it struggles with the
    contextual forms common in scanned reports. Surya was trained
    multilingual from the start.

  - **RTL bidi**: Surya returns text lines in logical (reading) order
    for RTL scripts. EasyOCR's `paragraph=True` mode is a heuristic
    that occasionally orders Arabic columns left-to-right by accident.

  - **PyTorch-native, transformers-only**: drops in alongside the
    embedder and reranker without a second deep-learning runtime.
    PaddleOCR would have meant pulling in paddlepaddle-gpu and
    juggling a separate CUDA pinning matrix.

  - **HF cache**: weights download into HF_HOME alongside bge-m3
    etc., so Dockerfile pre-pulls are one cache to warm.

Public API
==========
`init()`, `is_ready()`, `ocr_image()`, `ocr_crop()` keep their
signatures so the orchestrator and warmup loop don't need to know
which engine is wired in.
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
# Surya's RecognitionPredictor + DetectionPredictor each hold a transformer
# and a processor. Both are heavy to construct (downloads + load). We build
# them once at startup and reuse across pages.

_det_predictor = None     # type: ignore[var-annotated]
_rec_predictor = None     # type: ignore[var-annotated]
_languages: list[str] = []
_init_attempted: bool = False
_lock = threading.Lock()


def init() -> None:
    """Load Surya's detection + recognition predictors. Idempotent.

    The language list controls which RTL/script behaviour Surya enables
    per page. It comes from settings.ocr_languages (CSV, default 'ar,en')
    so adding e.g. Farsi only takes an env var change, no rebuild.
    """
    global _det_predictor, _rec_predictor, _languages, _init_attempted
    with _lock:
        if _init_attempted:
            return
        _init_attempted = True

        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            logger.exception("surya: package not installed: %s", exc)
            return

        _languages = [
            s.strip() for s in settings.ocr_languages.split(",") if s.strip()
        ] or ["en"]
        device = settings.device if torch.cuda.is_available() else "cpu"
        logger.info(
            "surya: initialising languages=%s device=%s", _languages, device,
        )

        try:
            # The predictors honour the TORCH_DEVICE env var when they
            # construct their internal models; setting it here keeps the
            # rest of the codebase device-agnostic.
            import os
            os.environ.setdefault("TORCH_DEVICE", device)
            _det_predictor = DetectionPredictor()
            _rec_predictor = RecognitionPredictor()
            logger.info("surya: ready")
        except Exception as exc:
            # Failed loads land us with is_ready() == False; the
            # orchestrator's OCR-fallback path will skip silently and the
            # extracted block just goes empty rather than the page failing.
            logger.exception("surya: failed to load predictors: %s", exc)


def is_ready() -> bool:
    return _det_predictor is not None and _rec_predictor is not None


# ── Public API ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def ocr_image(img: bytes | np.ndarray | Image.Image) -> str:
    """Run OCR on an image and return concatenated text in reading order.

    Accepts bytes (any PIL-supported format), a numpy array (H, W, C),
    or a PIL Image. Returns "" on any failure — never raises, so the
    orchestrator's fallback path stays predictable.

    For RTL scripts the returned text is in logical (reading) order, so
    a downstream tokenizer / embedder sees Arabic the same way the
    author wrote it, not as visual-order glyphs.

    Observability: every successful call logs dims + line count + char
    count + elapsed ms + an 80-char preview. The three silent-failure
    modes that bit us before (clamped-empty crop, predictor returned
    no results, lines detected but all empty after strip) each log a
    distinct WARNING so the next regression is one log line away.
    """
    if not is_ready():
        return ""

    pil = _coerce_to_pil(img)
    if pil is None:
        return ""

    started = time.perf_counter()
    img_w, img_h = pil.size
    logger.info(
        "surya: recognizing image=%dx%d langs=%s", img_w, img_h, _languages,
    )

    try:
        # Surya's recognition takes the detection predictor as an arg so
        # it knows where to find line boxes. Languages are passed per-
        # image as a list of lists; we use the same set for every image
        # since the language hint is just a script-routing signal, not a
        # filter.
        predictions = _rec_predictor(
            [pil], [_languages], _det_predictor,
        )
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning("surya: OOM at image=%dx%d: %s", img_w, img_h, exc)
        torch.cuda.empty_cache()
        return ""
    except Exception as exc:
        logger.warning("surya: recognition failed at image=%dx%d: %s",
                       img_w, img_h, exc)
        return ""

    if not predictions:
        logger.warning("surya: predictor returned no results for image=%dx%d",
                       img_w, img_h)
        return ""

    # One image in → one OCRResult out. text_lines is already ordered
    # top-to-bottom (and right-to-left within RTL rows), so joining on
    # newlines preserves reading order.
    page = predictions[0]
    raw_lines = list(getattr(page, "text_lines", None) or [])
    lines = []
    for line in raw_lines:
        text = (getattr(line, "text", None) or "").strip()
        if text:
            lines.append(text)

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if not lines:
        # Distinct from "no predictions": detector saw N candidate
        # lines but every one stripped to empty. Usually means tiny
        # crop or a model confidence floor wiped them.
        logger.warning(
            "surya: %d line(s) detected but all empty after strip "
            "(image=%dx%d, elapsed=%dms)",
            len(raw_lines), img_w, img_h, elapsed_ms,
        )
        return ""

    out = "\n".join(lines)
    preview = out.replace("\n", " ⏎ ")[:80]
    logger.info(
        "surya: ok — %d line(s), %d chars, %d ms — preview: %s",
        len(lines), len(out), elapsed_ms, preview,
    )
    return out


def ocr_crop(
    page_img: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> str:
    """OCR a sub-region of a page image. Used to fill in text for layout
    regions where Docling didn't return any (scanned PDFs, image-only pages).

    Logs the input bbox and the clamped crop dimensions so geometry
    bugs (wrong-unit, wrong-origin) are visible in logs instead of
    silently producing 0-size crops.
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
        "surya: ocr_crop cropping bbox=%s page=%dx%d → crop=%dx%d",
        raw, page_w, page_h, max(0, crop_w), max(0, crop_h),
    )

    if x1 <= x0 or y1 <= y0:
        # The bbox-fix in layout._resolve_bbox should make this branch
        # rare. When it still fires, it means upstream geometry is off
        # — log loudly so the source is obvious.
        logger.warning(
            "surya: ocr_crop bbox clamped to empty "
            "(raw=%s page=%dx%d) — upstream bbox / origin / unit mismatch?",
            raw, page_w, page_h,
        )
        return ""

    crop = page_img[y0:y1, x0:x1]
    return ocr_image(crop)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _coerce_to_pil(img: bytes | np.ndarray | Image.Image) -> Image.Image | None:
    """Normalise the various accepted input forms to a PIL RGB image —
    Surya's predictors take PIL Images directly."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim not in (2, 3):
            return None
        try:
            return Image.fromarray(img).convert("RGB")
        except Exception as exc:
            logger.warning("surya: failed to convert ndarray: %s", exc)
            return None
    if isinstance(img, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(img)).convert("RGB")
        except Exception as exc:
            logger.warning("surya: failed to decode image bytes: %s", exc)
            return None
    return None
