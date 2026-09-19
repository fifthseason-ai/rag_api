"""A bound on how much work ONE native PDF may cost before the service stops reading it.

THE EXPOSURE THIS ADDRESSES. Nothing in the chain protects this service from a large native PDF:
Core's extractor gate does not cover every format, the edge permits 25 MB, and rag_api itself has
no limit at all -- only `EMBEDDING_MAX_QUEUE_SIZE`, which bounds embedding, not extraction. Past
the point where every timeout in the chain has expired, the worker keeps parsing and allocating
for a response no caller is still waiting for, and if it is killed for memory the honest error
contract (a retryable 503, an actionable 400, a `Reference:<hex>` in the log) produces nothing at
all: the uploader gets a dropped connection.

The scanned path already has such a bound (`OcrBudget`). This is the same idea for the pages the
native text layer produces, and it is deliberately a SEPARATE object: the OCR budget exists to cap
work this service CHOOSES to do on a page, and this one caps how much of the document it reads at
all. Folding them together would make one limit silently move the other.

TWO THINGS THIS DELIBERATELY DOES NOT DO.

1. **It invents no production threshold.** Both bounds default to 0 = OFF, so behaviour is
   byte-identical to before until an operator sets them. A safe number depends on facts this lane
   cannot read -- the task's memory limit, the load balancer's idle timeout, the ingress body cap --
   and a number chosen without them would be a prediction dressed as a measurement.
2. **It never turns a truncation into an empty document or a fake success.** A stopped extraction
   keeps every page it did read, marks WHICH bound stopped it, and the receipt reports `partial`.
   A configured limit below one page is raised to one page for the same reason: zero pages would be
   reported as an empty document, which is a different (and false) statement about the file.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from app.config import PDF_EXTRACT_MAX_PAGES, PDF_EXTRACT_TIME_BUDGET_SECONDS

#: Metadata keys the loader stamps on the last page it read when a bound stopped it. The receipt
#: reads them; they are a CONTRACT SURFACE the same way `text_source` is.
STOPPED_KEY = "extraction_stopped"
NOT_ATTEMPTED_KEY = "extraction_pages_not_attempted"
ATTEMPTED_KEY = "extraction_pages_read"

UNLIMITED = 0


@dataclass
class ExtractionBudget:
    """One document's allowance for READING pages, shared across its pages.

    Values are read when a budget is made, not when the class is defined: a default evaluated at
    class-definition time froze the configured limits and made them unreachable to anything that
    reconfigures at runtime -- including a test that wants to exercise the bound. That mistake was
    made once already in `OcrBudget` and corrected there.
    """

    max_pages: int = field(default_factory=lambda: PDF_EXTRACT_MAX_PAGES)
    time_budget_seconds: float = field(
        default_factory=lambda: PDF_EXTRACT_TIME_BUDGET_SECONDS
    )

    pages_read: int = 0
    stopped_reason: Optional[str] = None
    _deadline: Optional[float] = field(default=None, repr=False)

    @property
    def enabled(self) -> bool:
        """False only when NEITHER bound is configured -- in which case this object changes nothing.

        Any non-zero value counts as configured, including a negative one. An earlier version
        tested `> 0`, which meant `PDF_EXTRACT_MAX_PAGES=-5` silently disabled the bound entirely:
        a control indistinguishable from its absence, and worse, one that looks configured in the
        environment. A nonsense value is clamped below and reported, never ignored.
        """
        return self.max_pages != UNLIMITED or self.time_budget_seconds != UNLIMITED

    @property
    def effective_max_pages(self) -> int:
        """A configured limit below one page still reads one page.

        Zero pages would make a bounded document indistinguishable from a document with no
        extractable text, which is the 422 refusal path — a different and false statement about
        the file. A misconfiguration should cost coverage, never honesty.
        """
        if self.max_pages == UNLIMITED:
            return UNLIMITED
        return max(1, self.max_pages)

    def start(self) -> None:
        if self._deadline is None and self.time_budget_seconds != UNLIMITED:
            # A negative budget is a typo, not an instruction to read backwards: it becomes an
            # immediate deadline, which the first-page guarantee below still keeps honest.
            self._deadline = time.monotonic() + max(0.0, self.time_budget_seconds)

    def may_read_page(self) -> bool:
        """True while there is allowance left for another page.

        Records the FIRST bound that ran out; later pages inherit that reason rather than
        overwriting it, so the receipt names what actually stopped the work.
        """
        if not self.enabled:
            return True
        self.start()
        if self.pages_read == 0:
            # ONE PAGE IS ALWAYS READ, whichever bound is tight -- including a time budget that
            # was already spent before the first page. Returning zero pages would make a bounded
            # document indistinguishable from a document with no extractable text, which is the
            # 422 refusal path: a different and false statement about the file. The tests caught
            # this: the page-limit clamp was there, the time-limit case was not.
            self.pages_read = 1
            return True
        limit = self.effective_max_pages
        if limit > UNLIMITED and self.pages_read >= limit:
            self.stopped_reason = self.stopped_reason or "page_limit"
            return False
        if self._deadline is not None and time.monotonic() >= self._deadline:
            self.stopped_reason = self.stopped_reason or "time_limit"
            return False
        self.pages_read += 1
        return True
