"""RapidOCR wrapper — Arabic + English text recognition via ONNX runtime.

OCR runs only on regions where Docling's text-layer extraction came
back empty (or suspiciously short). Two reasons:
    1. OCR is the slowest step per region (~50-300 ms each on GPU).
    2. Native PDF text is always more accurate than OCR'd text — never
       overwrite real text with an OCR guess.

Why RapidOCR (and not Surya / EasyOCR / PaddleOCR-Paddle / Tesseract)
====================================================================
RapidOCR ships the PaddleOCR PP-OCR family models exported to ONNX,
with a thin Python wrapper that runs them on onnxruntime instead of
PaddlePaddle. Three properties matter for our setup:

  - **CRNN + CTC, not autoregressive**. The recognition head has a
    "blank" output token, so given a non-text image (e.g. the
    decorative photo Docling sometimes mis-classifies as a figure
    region) it can output blank-blank-blank... and produce empty
    text naturally. Surya (transformer encoder-decoder, autoregressive)
    can't do that — its decoder is forced to emit *some* token, which
    is how we ended up with hallucinated repetitions of high-frequency
    Arabic words flooding every chunk on document-template pages.

  - **Native confidence scores**. RapidOCR returns a per-line
    confidence in the result tuple. Hallucinated lines on garbage
    crops score noticeably lower than real text (~0.2-0.5 vs 0.9+),
    so a simple threshold filter (`settings.ocr_min_confidence`) wipes
    them out without a separate "is this gibberish" classifier.

  - **onnxruntime-gpu, no paddlepaddle**. The PP-OCRv3 Arabic weights
    are state-of-the-art on cursive Arabic ligatures, but we get them
    without dragging a second deep-learning runtime (PaddlePaddle) into
    the image. Pure-ONNX runs alongside PyTorch on the same CUDA device
    without conflict.

Model files
===========
Three files are required (det + rec + dict). Resolution order at
init() time:

  1. Explicit override paths from settings.ocr_{det,rec,cls}_model_path
     and settings.ocr_rec_keys_path.
  2. `huggingface_hub.hf_hub_download` from `SWHL/RapidOCR` (the
     canonical, RapidOCR-author-maintained repo of PP-OCR ONNX
     conversions) into HF_HOME.

The Dockerfile pre-downloads them during the image build so the first
real ingest doesn't stall on network I/O.

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
from pathlib import Path

import numpy as np
from PIL import Image

from core.config import settings

logger = logging.getLogger(__name__)


# ── Module state ────────────────────────────────────────────────────────────
# RapidOCR's engine bundles a detector + recognizer + (optional) angle
# classifier, each as an ONNX session. Heavy to construct; build once.

_engine = None             # type: ignore[var-annotated]
_init_attempted: bool = False
_lock = threading.Lock()


# Canonical HF source for the Arabic-tuned PP-OCRv3 ONNX models. These
# files were converted from PaddleOCR's official .pdmodel / .pdiparams
# releases by the RapidOCR maintainer. Override via OCR_*_PATH env vars
# if you have a local mirror or want a different model variant.
_HF_REPO = "SWHL/RapidOCR"
_HF_FILES = {
    "det":  "PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.onnx",
    "rec":  "PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.onnx",
    "keys": "PP-OCRv3/multilingual/arabic_dict.txt",
    # Orientation classifier — optional. The PP-OCR mobile cls model is
    # ~1.5 MB and useful when scanned pages have 90/180/270° rotations.
    # For our PDFs (page rotation already normalised by Docling) it's a
    # no-op most of the time, but harmless to keep enabled.
    "cls":  "PP-OCRv3/ch_ppocr_mobile_v2.0_cls_infer.onnx",
}


def _resolve_model_paths() -> dict[str, str] | None:
    """Return {det, rec, keys, cls} → absolute filesystem paths, or None
    if any required file is missing and cannot be fetched.

    Resolution order per file:
      1. Explicit override path from settings (set the file directly).
      2. `huggingface_hub.hf_hub_download` from SWHL/RapidOCR into HF_HOME.

    cls is optional — if unavailable the engine still runs without
    orientation classification.
    """
    overrides = {
        "det":  settings.ocr_det_model_path,
        "rec":  settings.ocr_rec_model_path,
        "keys": settings.ocr_rec_keys_path,
        "cls":  settings.ocr_cls_model_path,
    }
    paths: dict[str, str] = {}

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None  # type: ignore[assignment]

    for key, override in overrides.items():
        if override and Path(override).is_file():
            paths[key] = override
            continue
        if hf_hub_download is None:
            if key == "cls":
                continue
            logger.error(
                "rapidocr: huggingface_hub not installed and OCR_%s_PATH not set",
                key.upper(),
            )
            return None
        try:
            paths[key] = hf_hub_download(
                repo_id=_HF_REPO, filename=_HF_FILES[key],
            )
        except Exception as exc:
            if key == "cls":
                logger.info(
                    "rapidocr: cls model unavailable (%s) — running without orientation classifier",
                    exc,
                )
                continue
            logger.error(
                "rapidocr: failed to fetch %s from %s/%s: %s",
                key, _HF_REPO, _HF_FILES[key], exc,
            )
            return None
    return paths


def init() -> None:
    """Load RapidOCR engine + ONNX models. Idempotent.

    On first call: downloads Arabic recognizer + multilingual detector
    ONNX models from Hugging Face (SWHL/RapidOCR) into HF_HOME if not
    already cached, then instantiates a RapidOCR session bound to them.
    """
    global _engine, _init_attempted
    with _lock:
        if _init_attempted:
            return
        _init_attempted = True

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            logger.exception("rapidocr: package not installed: %s", exc)
            return

        paths = _resolve_model_paths()
        if paths is None:
            logger.error("rapidocr: unable to resolve model paths — OCR disabled")
            return

        use_cuda = settings.ocr_use_cuda
        logger.info(
            "rapidocr: initialising det=%s rec=%s use_cuda=%s",
            Path(paths["det"]).name, Path(paths["rec"]).name, use_cuda,
        )

        try:
            kwargs: dict = {
                "det_model_path": paths["det"],
                "rec_model_path": paths["rec"],
                "rec_keys_path":  paths["keys"],
                "det_use_cuda":   use_cuda,
                "rec_use_cuda":   use_cuda,
                "cls_use_cuda":   use_cuda,
            }
            if "cls" in paths:
                kwargs["cls_model_path"] = paths["cls"]
                kwargs["use_cls"] = True
            else:
                kwargs["use_cls"] = False
            _engine = RapidOCR(**kwargs)
            logger.info("rapidocr: ready")
        except Exception as exc:
            # Failed loads land us with is_ready() == False; the
            # orchestrator's OCR-fallback path will skip silently and the
            # extracted block just goes empty rather than the page failing.
            logger.exception("rapidocr: failed to load engine: %s", exc)


def is_ready() -> bool:
    return _engine is not None


# ── Public API ──────────────────────────────────────────────────────────────

def ocr_image(img: bytes | np.ndarray | Image.Image) -> str:
    """Run OCR on an image and return concatenated text in reading order.

    Accepts bytes (any PIL-supported format), a numpy array (H, W, C),
    or a PIL Image. Returns "" on any failure — never raises, so the
    orchestrator's fallback path stays predictable.

    Line ordering: RapidOCR returns detection boxes top-to-bottom by
    centroid; within each line, characters are in script-native order
    (Arabic logical order for AR text, LTR for Latin). Matches what
    Surya returned, so the orchestrator contract is unchanged.

    Confidence filter: lines below `settings.ocr_min_confidence` are
    dropped. This is the main defence against the
    OCR-hallucinating-on-non-text-crops failure mode — well-calibrated
    PP-OCRv3 confidences make a simple threshold reliable.

    Observability: every successful call logs dims + lines kept + lines
    dropped + char count + elapsed ms + an 80-char preview. The
    "detector found nothing" path logs distinctly from "lines detected
    but all below threshold" so a regression on either is one log line
    away.
    """
    if not is_ready():
        return ""

    arr = _coerce_to_ndarray(img)
    if arr is None:
        return ""

    started = time.perf_counter()
    img_h, img_w = arr.shape[:2]
    logger.info("rapidocr: recognizing image=%dx%d", img_w, img_h)

    try:
        result, _engine_elapsed = _engine(arr)
    except Exception as exc:
        logger.warning("rapidocr: recognition failed at image=%dx%d: %s",
                       img_w, img_h, exc)
        return ""

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # `result` is None when the detector returned no boxes, or a list
    # of [bbox, text, score] entries. The "None" path is the natural
    # "this isn't text" signal a CTC OCR gives us — exactly what Surya
    # couldn't express because its autoregressive decoder always emits
    # *some* token.
    if not result:
        logger.info(
            "rapidocr: no text detected (image=%dx%d, elapsed=%dms)",
            img_w, img_h, elapsed_ms,
        )
        return ""

    min_conf = settings.ocr_min_confidence
    kept: list[str] = []
    dropped_low_conf = 0
    for entry in result:
        # Entry shape is [bbox, text, score] in rapidocr-onnxruntime 1.3.x.
        # Be defensive about layout in case a future version adds fields.
        if not entry or len(entry) < 3:
            continue
        text = (entry[1] or "").strip()
        try:
            score = float(entry[2])
        except (TypeError, ValueError):
            score = 0.0
        if not text:
            continue
        if score < min_conf:
            dropped_low_conf += 1
            continue
        kept.append(text)

    if not kept:
        logger.info(
            "rapidocr: %d line(s) detected, all below conf=%.2f "
            "(image=%dx%d, elapsed=%dms)",
            len(result), min_conf, img_w, img_h, elapsed_ms,
        )
        return ""

    out = "\n".join(kept)
    preview = out.replace("\n", " ⏎ ")[:80]
    logger.info(
        "rapidocr: ok — %d line(s) kept (%d dropped <%.2f conf), "
        "%d chars, %d ms — preview: %s",
        len(kept), dropped_low_conf, min_conf, len(out), elapsed_ms, preview,
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
        "rapidocr: ocr_crop cropping bbox=%s page=%dx%d → crop=%dx%d",
        raw, page_w, page_h, max(0, crop_w), max(0, crop_h),
    )

    if x1 <= x0 or y1 <= y0:
        # The bbox-fix in layout._resolve_bbox should make this branch
        # rare. When it still fires, it means upstream geometry is off
        # — log loudly so the source is obvious.
        logger.warning(
            "rapidocr: ocr_crop bbox clamped to empty "
            "(raw=%s page=%dx%d) — upstream bbox / origin / unit mismatch?",
            raw, page_w, page_h,
        )
        return ""

    crop = page_img[y0:y1, x0:x1]
    return ocr_image(crop)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _coerce_to_ndarray(img: bytes | np.ndarray | Image.Image) -> np.ndarray | None:
    """Normalise the various accepted input forms to a HxWx3 RGB numpy
    array — RapidOCR's engine takes a numpy array directly. PIL → np
    via `np.array(img.convert("RGB"))`."""
    if isinstance(img, np.ndarray):
        if img.ndim not in (2, 3):
            return None
        return img
    if isinstance(img, Image.Image):
        try:
            return np.array(img.convert("RGB"))
        except Exception as exc:
            logger.warning("rapidocr: failed to convert PIL image: %s", exc)
            return None
    if isinstance(img, (bytes, bytearray)):
        try:
            return np.array(Image.open(io.BytesIO(img)).convert("RGB"))
        except Exception as exc:
            logger.warning("rapidocr: failed to decode image bytes: %s", exc)
            return None
    return None
