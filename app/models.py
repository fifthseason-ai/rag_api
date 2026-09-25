# app/models.py
import hashlib
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Tuple


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict


class DocumentModel(BaseModel):
    page_content: str
    metadata: Optional[dict] = {}

    def generate_digest(self):
        hash_obj = hashlib.md5(self.page_content.encode())
        return hash_obj.hexdigest()


class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str


class QueryDocument(BaseModel):
    """One document as `/query` ALREADY returns it -- a description, not a redesign.

    Every field here was read off the wire before this model existed
    (`evidence/.../joint-run/F-QC1-WIRE-BEFORE.json`), and the model is deliberately no
    narrower than what was measured.

    `metadata` IS AN OPEN DICT AND MUST STAY ONE. Its keys differ per format: a spreadsheet
    carries `page_name`/`page_number`, a PDF `page`/`page_label`/`total_pages`, a presentation
    `slide_number`/`slide_title`, a CSV `row`. Twenty-seven distinct keys across five formats.
    FastAPI validates and re-serialises through a response_model, so typing this dict would
    DELETE every key not named -- including the locators a citation is built from -- and the
    change would read as additive while removing the field the consumer depends on.

    `id` and `type` are declared for the same reason: LangChain's Document serialises them, and
    omitting them here would drop them from the response.

    IT ALSO VALIDATES, WHICH IS A CHANGE. Declaring a response_model does not only describe the
    response -- FastAPI raises `ResponseValidationError` (a 500) where the route previously
    serialised whatever it was handed. Independent review demonstrated it with a `None` score,
    which used to come back as `200` with `score: null`.

    That is NOT reachable today: `_hybrid_or_dense_search` calls `round(score, 4)`, which raises on a
    non-numeric score before any response is built, and `page_content` is always a `str`. The
    upstream guard is pinned by a test rather than left as a comment, because this paragraph
    going stale is exactly how a 500 appears later with nothing pointing at the cause.
    """

    id: Optional[str] = None
    metadata: dict = {}
    page_content: str
    type: Optional[str] = None

    #: What the pair's score MEANS, and which way is better (`app/services/score_kind.py`;
    #: P06-5 contract addendum 2). Top-level, never inside `metadata`: metadata is the chunk's
    #: stored data, which consumers forward and persist; this describes the RESPONSE. `None`
    #: (sent as null) is UNKNOWN -- a result built outside the declaring pipeline -- and must be
    #: marked by the consumer, never guessed from the value's range.
    score_kind: Optional[str] = None
    score_direction: Optional[str] = None

    #: Whether this chunk carries a stored citation link, a QUARANTINED one, or none (CARD-P2-01 S1 C):
    #: 'present' | 'quarantined' | 'none'. Top-level for the same reason as `score_kind`: it
    #: DESCRIBES the stored metadata, it is not stored data. Derived from the marker the stored-link
    #: backfill writes (CARD-F-EMBED-LINK-BACKFILL); see `document_routes._link_state`. `None` (sent
    #: as null) only on a document built outside the shaping seam -- UNKNOWN, never "no link".
    #: 'present' says a link is STORED, not that it is governed: Core's consumer-side governed-URL
    #: gate stays load-bearing. When 'quarantined', `metadata.quarantined_link` is the REDACTED
    #: object {scheme, host, refusal_reason, sha256} -- never the raw refused URL (decision (2));
    #: `_redacted_metadata` strips a raw string on emit, so the raw never leaves the process.
    link_state: Optional[str] = None


#: What `/query` returns: a list of (document, score) pairs. The pair is a two-element JSON array,
#: not an object -- that is the existing wire and this does not change it. The score is NOT always
#: a similarity: a dense pgvector search returns a cosine DISTANCE (lower is better) and the other
#: modes return higher-is-better numbers. The document's `score_kind` / `score_direction` say which.
QueryHit = Tuple[QueryDocument, float]


class QueryRequestBody(BaseModel):
    query: str
    file_id: str
    k: int = 4
    entity_id: Optional[str] = None


class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"


class QueryByEntityBody(BaseModel):
    query: str
    k: int = 4
    args: Optional[dict] = None


class QueryMultipleBody(BaseModel):
    query: str
    file_ids: List[str]
    k: int = 4


class FileSummary(BaseModel):
    file_id: str
    summary: str
    chunk_count: int


class DocumentOwnerType(Enum):
    AGENT = "AGENT"
    KNOWLEDGE = "KNOWLEDGE"


class DocumentOriginType(Enum):
    # Parity with Core packages/data-provider/src/tempo.ts DocumentOriginType.
    # Core declares { ORGANIC, SHAREPOINT, ONEDRIVE, GDRIVE, GMAIL } and tags
    # embeds/deletes per provider; rag_api must accept those origins so per-origin
    # vector cleanup works. BOX is the forward-add for the KI-02 inc-B Box adapter
    # (Core switches Box to BOX only after this rag_api change is DEPLOYED).
    # Additive only; string values equal member names, matching Core exactly.
    ORGANIC = "ORGANIC"
    SHAREPOINT = "SHAREPOINT"
    ONEDRIVE = "ONEDRIVE"
    GDRIVE = "GDRIVE"
    GMAIL = "GMAIL"
    BOX = "BOX"


class DeleteDocumentsBody(BaseModel):
    entity_id: Optional[str] = None
    file_ids: List[str] = []
    document_origin_type: Optional[DocumentOriginType] = None
    subscription_id: Optional[str] = None
    #: Delete only the rows a particular producer wrote (FILES-01): `native` for text
    #: taken from the document's own text layer, `ocr` for text a machine read off a
    #: page image. Omitted, nothing changes and every matching row is deleted.
    #:
    #: This exists so an escalation can SUPERSEDE rather than supplement: Core embeds
    #: the better text first, then deletes only the superseded local-OCR rows. The
    #: other ordering -- delete, then embed -- can leave a document with ZERO rows if
    #: the second step fails, which is the one outcome an escalation must never be able
    #: to produce.
    text_source: Optional[str] = None
