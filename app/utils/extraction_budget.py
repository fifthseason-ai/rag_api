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
2. **A stopped extraction is never a fake success.** It keeps every page it did read, marks WHICH
   bound stopped it, and the receipt reports `partial`. A configured limit below one page is raised
   to one page for the same reason: zero pages would make a bounded document indistinguishable from
   one with no text at all.

   **CORRECTED after review, because the original wording promised more than the code delivers.**
   This said a truncation is *never* turned into an empty document. It can be: if the pages inside
   the bound happen to carry no text -- a title page, a cover sheet, a blank scan in front of nine
   readable pages -- then `units_extracted` is 0, and the empty-extraction guard refuses the upload
   with 422 exactly as it would for a genuinely empty file. **No page floor can prevent that**, only
   a bound high enough to reach the text. What IS guaranteed: the receipt carries `extraction_bound`
   so the cause is never ambiguous, and the refusal message names the bound instead of blaming the
   file. That is a real operational consequence of setting a low page limit and it belongs in the
   rollout decision, not in a docstring that promises it away.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from app.config import PDF_EXTRACT_MAX_PAGES, PDF_EXTRACT_TIME_BUDGET_SECONDS

#: Metadata keys the loader stamps on the last page it read when a bound stopped it, and which the
#: receipt turns into `extraction_bound`. No consumer outside this repo reads them yet.
STOPPED_KEY = "extraction_stopped"
#: NOT "not attempted". One page beyond the bound IS pulled from the producer and then discarded --
#: the generator holds a page back so it can stamp the last one it keeps -- so a count of pages the
#: reader never opened would be off by one. This counts pages ABSENT FROM THIS RECEIPT, which is
#: both true and the thing a caller actually needs to know.
NOT_INCLUDED_KEY = "extraction_pages_not_included"
ATTEMPTED_KEY = "extraction_pages_read"

UNLIMITED = 0


@dataclass
class ExtractionBudget:
    """One document's allowance for READING pages, shared across its pages.

    Values are read when a budget is MADE, not when the class is defined: a default evaluated at
    class-definition time froze the configured limits into the signature. That mistake was made
    once already in `OcrBudget` and corrected there.

    Scope of that, stated precisely after review, because the earlier wording overclaimed: the
    factory closes over THIS module's globals, which are bound at import from the environment.
    Reassigning `app.config.PDF_EXTRACT_MAX_PAGES` at runtime therefore does NOT reach it -- a test
    that wants to exercise the bound passes the limits to the constructor (as every test here does)
    or patches this module's own names. Production is unaffected: the environment is read at import
    and never changes afterwards.
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
