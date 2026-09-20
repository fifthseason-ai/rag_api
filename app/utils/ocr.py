"""Bounded local OCR for PDF pages that carry no text layer.

WHY THIS EXISTS
---------------
A scanned PDF is a photograph of a page: pypdf extracts zero characters from it, so
before this the honest-but-useless answer was 422 "no extractable text". Richard
approved local-first OCR with a controlled escalation to the already-approved AWS
route (FS-CONTINUE-R3 / 01-APPROVED-DECISIONS, "OCR"): read native text first, OCR
only the pages that need it, and when local OCR is not good enough say so in a
machine-readable way so Core can escalate. This module is the rag_api half of that,
and ONLY that -- it never calls an external provider, never chooses a strategy and
never owns an escalation policy.

WHAT WAS MEASURED BEFORE IT WAS WRITTEN (deployed lite image, network disabled)
------------------------------------------------------------------------------
* PyPDFLoader(extract_images=True) -- and even images_parser=RapidOCRBlobParser() --
  returns **0 chars** on a genuine scan. The engine was never the problem; the
  langchain image wiring simply does not deliver. So the page-to-bitmap step is done
  here explicitly rather than by flipping PDF_EXTRACT_IMAGES.
* rapidocr-onnxruntime==1.4.4 ships in BOTH requirements files, initialises in
  ~0.35 s from bundled models with no network, and reads a 3.7 MP page in ~0.9 s
  (first page ~2.5 s, warm-up). Through the REAL ROUTE in a 4 GB container it is
  ~1.9 s/page: the isolated engine figure is not the deployed one, and the bounds are
  sized on the deployed measurement rather than on the engine benchmark.
* End to end, total request time is CAPPED by the time budget at ~62 s whatever the
  document size -- a 120-page scan takes the same ~62 s as a 50-page one and reports
  the remaining 87 pages as not attempted. Peak RSS tops out around 1.0 GB.
* Character recall against known ground truth on upright scans: **0.96-1.00**.
  (An earlier word-level metric said 0.63 -- that metric was wrong, not the engine:
  it scored "quarterlycapacity" as two misses when every character was recovered.)
* Honouring the page's own /Rotate lifts rotated scans to **0.99**. Whole-page rotation
  with NO /Rotate signal stays weak and is exactly what the insufficient/escalation
  outcome is for -- rapidocr's use_angle_cls does not help there (measured: it
  classifies text-line flips, not page rotation).
  The weak figures are fixture-dependent and should be read as a RANGE, not a constant:
  0.00-0.50 for sideways and 0.31-0.43 for upside-down across the fixtures measured
  here and independently by review. What is stable is the SHAPE -- a large share of the
  page is lost while the engine's confidence stays high (0.89-0.96) -- and that is the
  claim the design rests on, not any single number.
* Peak RSS rises to ~764 MB for a 12-page scan and ~1.0 GB at 33.7 MP/page, and
  quality is flat from 3.7 MP to 33.7 MP -- which is why oversized images are
  downscaled to a pixel cap rather than OCR'd whole, and why the caps exist at all.

Every default below is that measurement, not a guess.
"""

import io
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from app.config import (
    logger,
    PDF_OCR_ENABLED,
    PDF_OCR_MAX_IMAGES_PER_PAGE,
    PDF_OCR_MAX_PAGES,
    PDF_OCR_LOW_CONFIDENCE_BELOW,
    PDF_OCR_MAX_PIXELS,
    PDF_OCR_MIN_CHARS,
    PDF_OCR_SIDEWAYS_BOX_RATIO,
    PDF_OCR_TIME_BUDGET_SECONDS,
)

#: Reported in the receipt so an operator can tell which engine produced stored text.
ENGINE_NAME = "rapidocr-onnxruntime"

_engine = None
_engine_lock = threading.Lock()
_engine_failed: Optional[str] = None


class OcrCancelled(Exception):
    """The caller asked us to stop. Raised between pages so a cancelled request does
    not leave a worker thread OCR-ing the rest of a long document for nothing."""


def get_engine():
    """Build the OCR engine once per process, lazily.

    Lazily because the import and model load cost ~0.35 s and ~50 MB, and the vast
    majority of uploads (native PDFs, workbooks, decks, text) never need it. Once,
    because paying that per page would dominate the per-page cost we measured.

    Returns None -- never raises -- when the engine is unavailable, so a missing or
    broken OCR install degrades to exactly the pre-OCR behaviour (an honest empty
    verdict) instead of turning every scanned upload into a 5xx.
    """
    global _engine, _engine_failed
    if _engine is not None or _engine_failed is not None:
        return _engine
    with _engine_lock:
        if _engine is not None or _engine_failed is not None:
            return _engine
        try:
            from rapidocr_onnxruntime import RapidOCR

            started = time.monotonic()
            _engine = RapidOCR()
            logger.info(
                "local OCR engine ready (%s) in %.2fs",
                ENGINE_NAME,
                time.monotonic() - started,
            )
        except Exception as error:
            _engine_failed = f"{type(error).__name__}: {error}"
            logger.warning(
                "local OCR unavailable, scanned pages will be reported as "
                "insufficient rather than read: %s",
                _engine_failed,
            )
    return _engine


def reset_engine_for_tests() -> None:
    """Drop the cached engine/failure so a test can exercise both branches."""
    global _engine, _engine_failed
    with _engine_lock:
        _engine = None
        _engine_failed = None


@dataclass
class OcrBudget:
    """One document's worth of allowance, shared across its pages.

    Bounds page expansion, compute, memory and duration together -- a document is
    refused further OCR when ANY of them is exhausted, and the reason is recorded so
    the receipt can say which bound stopped the work rather than silently truncating.
    """

    # `default_factory`, not a plain default: a bare default is evaluated once when the
    # class is defined, which froze the configured limits into the dataclass and made
    # them unreachable afterwards -- so a test could not exercise a bound, and neither
    # could anything else that reconfigures at runtime. Read them when a budget is made.
    max_pages: int = field(default_factory=lambda: PDF_OCR_MAX_PAGES)
    max_images_per_page: int = field(default_factory=lambda: PDF_OCR_MAX_IMAGES_PER_PAGE)
    max_pixels: int = field(default_factory=lambda: PDF_OCR_MAX_PIXELS)
    time_budget_seconds: float = field(default_factory=lambda: PDF_OCR_TIME_BUDGET_SECONDS)
    should_stop: Optional[Callable[[], bool]] = None

    pages_started: int = 0
    stopped_reason: Optional[str] = None
    _deadline: Optional[float] = field(default=None, repr=False)

    def start(self) -> None:
        if self._deadline is None:
            self._deadline = time.monotonic() + self.time_budget_seconds

    def check_cancelled(self) -> None:
        """Raise if the caller has gone away. Called between pages AND before each
        image, so the longest we can overrun a cancellation is one image."""
        if self.should_stop is not None and self.should_stop():
            self.stopped_reason = "cancelled"
            raise OcrCancelled("OCR cancelled by caller")

    def may_start_page(self) -> bool:
        """True while there is allowance left for another page. Records the first
        bound that ran out; later pages inherit that same reason."""
        self.check_cancelled()
        self.start()
        if self.pages_started >= self.max_pages:
            self.stopped_reason = self.stopped_reason or "page_limit"
            return False
        if time.monotonic() >= self._deadline:
            self.stopped_reason = self.stopped_reason or "time_limit"
            return False
        self.pages_started += 1
        return True

    def seconds_left(self) -> float:
        self.start()
        return max(0.0, self._deadline - time.monotonic())


@dataclass
class PageOcrResult:
    """What OCR made of one page. `sufficient` is the typed signal Core escalates on."""

    text: str = ""
    confidence: float = 0.0
    images_seen: int = 0
    images_read: int = 0
    duration_seconds: float = 0.0
    #: True only when OCR ran on this page. A page skipped because a bound was already
    #: exhausted is NOT attempted -- Core called that distinction load-bearing, since
    #: "we read it and got nothing" and "we never looked" mean different things.
    attempted: bool = False
    reason: str = "not_attempted"
    #: Fraction of detected text boxes that are taller than they are wide. Measured:
    #: 0.00 on every upright page in the corpus and 1.00 on a sideways one, so it is a
    #: geometric FACT that a page is rotated -- not a quality score.
    tall_box_ratio: float = 0.0
    lines: int = 0
    #: 'downscaled' when an oversized image was reduced to the pixel cap before OCR.
    notes: List[str] = field(default_factory=list)


def _page_images(page, budget: "OcrBudget") -> List[bytes]:
    """The raster images embedded in a page, bounded in count.

    Deliberately embedded images and NOT a rendered page bitmap: a genuine scan IS
    an embedded image, this needs no renderer, and it measured BETTER than rendering
    on rotated pages. A page drawn as vector art with no text layer therefore yields
    nothing here -- a real limitation, reported as insufficient rather than papered
    over, because the approved AWS route handles it and adding a second renderer
    dependency here would not be "bounded".
    """
    try:
        images = list(page.images)
    except Exception as error:
        logger.info("no readable embedded images on page: %s", error)
        return []
    out = []
    for image in images[: budget.max_images_per_page]:
        budget.check_cancelled()
        try:
            out.append(image.data)
        except Exception as error:
            # One undecodable image must not lose the rest of the page.
            logger.info("skipping an undecodable embedded image: %s", error)
    return out


def _to_array(data: bytes, rotation: int, budget: "OcrBudget", notes: List[str]):
    """Decode one embedded image into an RGB array the engine can read.

    Honours the page's own /Rotate: measured, that alone lifts rotated scans from
    a weak reading to 0.99 character recall. PDF /Rotate is degrees CLOCKWISE at display
    time and PIL rotates counter-clockwise, so the correction is rotate(-rotation).
    (My first fixture had this backwards and made a correct correction look broken --
    the sign is measured, not reasoned.)
    """
    import numpy as np
    from PIL import Image

    # `Image.open` reads the header only, so the declared size is known BEFORE any
    # pixels are decoded. Review found the first version converting to RGB first and
    # capping afterwards, which bounded the engine's work but not the decode: an image
    # between the cap and PIL's own ~89 MP default would materialise whole before being
    # shrunk. The cap is applied to the decode itself now.
    #
    # Measured on a 60 MP JPEG with a 16 MP cap, peak RSS in a fresh process:
    # +180 MB through the draft path against +229 MB decoding whole and resizing after.
    # Real and worth having -- but a 22% saving, not an elimination, because the capped
    # image and its array still have to exist. Said that way because the first attempt to
    # measure this used tracemalloc, which sees only Python allocations and reported the
    # two strategies as identical; PIL's decode buffer is a C allocation.
    image = Image.open(io.BytesIO(data))
    pixels = image.width * image.height
    oversized = budget.max_pixels > 0 and pixels > budget.max_pixels

    if oversized:
        # Quality is flat from 3.7 MP to 33.7 MP but memory is not, so cap the pixels
        # rather than the quality.
        ratio = (budget.max_pixels / pixels) ** 0.5
        target = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
        # For JPEG -- which is what a scanner produces -- `draft` decodes at reduced
        # resolution in the DCT domain, so the full-size bitmap never exists. It is a
        # no-op for other formats, which then fall back to decode-then-resize below.
        try:
            image.draft("RGB", target)
        except Exception:
            pass
        notes.append("downscaled")

    if image.mode != "RGB":
        image = image.convert("RGB")
    if oversized and image.width * image.height > budget.max_pixels:
        # `draft` only lands on power-of-two-ish steps, and does nothing at all for
        # non-JPEG, so finish the job exactly.
        ratio = (budget.max_pixels / (image.width * image.height)) ** 0.5
        image = image.resize(
            (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
        )
    if rotation:
        image = image.rotate(-rotation, expand=True)
    return np.array(image)


def is_low_confidence(text: str, confidence: float) -> bool:
    """Flag text the ENGINE itself does not vouch for. A report, not a verdict.

    Core owns the judgement of whether a document is well enough covered to call
    ingested ("if you find yourself inventing a confidence score to make that call,
    that is the boundary and it is mine"), so this deliberately does NOT decide
    whether the text is stored -- extracted characters are never thrown away here.
    It only decides whether the page is REPORTED as weak, which is what puts it in
    `escalation.locators` for Core to act on.

    The number itself is not invented: rapidocr returns a per-line confidence and
    this is their mean. The thresholds are where we stop calling that good, and they
    are reporting thresholds, which is why they are named for reporting.

    Measured on real fixtures: clean upright scans come back at 0.95-0.97 with
    hundreds of characters; a page the engine cannot read comes back at 0.00 with
    none. Nothing in the corpus landed between, so these bounds catch degenerate
    cases rather than routinely arbitrating quality.
    """
    stripped = text.strip()
    if not stripped:
        return False  # no text at all is a different outcome, reported separately
    return len(stripped) < PDF_OCR_MIN_CHARS or confidence < PDF_OCR_LOW_CONFIDENCE_BELOW


def _join(texts: List[str]) -> str:
    """Join the per-image text of one page without inventing separators inside words."""
    return re.sub(r"[ \t]+", " ", "\n".join(t for t in texts if t and t.strip())).strip()


def ocr_page(page, budget: "OcrBudget") -> PageOcrResult:
    """Read one page's embedded images. Never raises except OcrCancelled.

    Any other failure is reported as an insufficient page, because a page this
    service could not read is a fact about the extraction -- not grounds for telling
    the uploader their file is broken, and not grounds for a 5xx.
    """
    result = PageOcrResult()
    if not PDF_OCR_ENABLED:
        result.reason = "disabled"
        return result
    if not budget.may_start_page():
        result.reason = budget.stopped_reason or "budget_exhausted"
        return result

    engine = get_engine()
    if engine is None:
        result.reason = "engine_unavailable"
        return result

    started = time.monotonic()
    try:
        rotation = int(getattr(page, "rotation", 0) or 0) % 360
    except Exception:
        rotation = 0

    images = _page_images(page, budget)
    result.images_seen = len(images)
    if not images:
        result.reason = "no_page_image"
        result.duration_seconds = time.monotonic() - started
        return result

    texts: List[str] = []
    confidences: List[float] = []
    tall = 0
    boxes = 0
    for data in images:
        budget.check_cancelled()
        if budget.seconds_left() <= 0:
            budget.stopped_reason = budget.stopped_reason or "time_limit"
            break
        try:
            array = _to_array(data, rotation, budget, result.notes)
            lines, _ = engine(array)
        except OcrCancelled:
            raise
        except Exception as error:
            logger.info("OCR failed on one page image: %s", error)
            continue
        # `attempted` is set HERE, not before the image loop: a page with nothing to
        # read was never looked at by the engine, whatever we intended. Review caught it
        # being set up front, which counted a blank page as attempted and made
        # `pages_attempted` mean something other than what its docstring says.
        result.attempted = True
        result.images_read += 1
        if not lines:
            continue
        texts.append(" ".join(str(line[1]) for line in lines))
        confidences.extend(float(line[2]) for line in lines)
        tall += sum(1 for line in lines if _is_tall(line[0]))
        boxes += len(lines)

    result.text = _join(texts)
    result.confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    result.lines = boxes
    result.tall_box_ratio = (tall / boxes) if boxes else 0.0
    result.duration_seconds = time.monotonic() - started
    if not result.text.strip():
        result.reason = "ocr_no_text"
    elif result.tall_box_ratio >= PDF_OCR_SIDEWAYS_BOX_RATIO:
        # The page is sideways and the PDF gave no /Rotate to say so. Measured, this
        # costs most of the page's characters while mean confidence stays at 0.89-0.96 --
        # which is exactly why confidence alone must not be the only signal.
        result.reason = "ocr_orientation_suspect"
    elif is_low_confidence(result.text, result.confidence):
        result.reason = "ocr_low_confidence"
    else:
        result.reason = "ocr"
    return result


def _is_tall(box) -> bool:
    """True when a detected text box is taller than it is wide, i.e. the line runs
    vertically -- what a sideways page looks like to the detector."""
    try:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        return (max(ys) - min(ys)) > (max(xs) - min(xs))
    except Exception:
        return False
