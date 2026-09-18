# app/models.py
import hashlib
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List


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
