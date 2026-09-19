# app/routes/document_routes.py
import os
import errno
import uuid
from pathlib import Path
import hashlib
import traceback
import aiofiles
import aiofiles.os
from shutil import copyfileobj
from typing import List, Iterable, Optional, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Request,
    UploadFile,
    HTTPException,
    File,
    Form,
    Body,
    Query,
    status,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
import threading

if TYPE_CHECKING:
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from langchain_community.vectorstores.pgvector import PGVector as PgVector

from app.build_info import build_summary
from app.config import (
    logger,
    vector_store,
    llm,
    RAG_UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_QUEUE_SIZE,
    VECTOR_DB_TYPE,
    VectorDBType,
    HYBRID_SEARCH_ENABLED,
    RERANK_ENABLED,
    RERANK_CANDIDATES,
    RERANK_TOP_N, SUM_UP_KNOWLEDGE_FILES,
)
from app.constants import ERROR_MESSAGES
from app.models import (
    StoreDocument,
    QueryRequestBody,
    QueryByEntityBody,
    DocumentResponse,
    QueryMultipleBody,
    DeleteDocumentsBody,
    DocumentOriginType,
    DocumentOwnerType,
)
from app.services.summarization import summarize_files
from app.services.summary_store import (
    upsert_file_summary,
    get_summaries_by_user,
    delete_summaries_by_file_ids,
)
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.services.hybrid_search import keyword_search, reciprocal_rank_fusion
from app.services.reranker import rerank
from app.utils.document_loader import (
    get_loader,
    clean_text,
    process_documents,
    cleanup_temp_encoding_file,
    DocumentVerdictError,
)
from app.utils.extraction_budget import (
    ATTEMPTED_KEY,
    NOT_ATTEMPTED_KEY,
    STOPPED_KEY,
    ExtractionBudget,
)
from app.utils.ocr import ENGINE_NAME as OCR_ENGINE_NAME, OcrBudget
from app.utils.health import is_health_ok

router = APIRouter()


def calculate_num_batches(total: int, batch_size: int) -> int:
    """Calculate the number of batches needed to process total items."""
    if batch_size <= 0:
        return 1
    return (total + batch_size - 1) // batch_size


def get_user_id(request: Request, entity_id: str = None) -> str:
    """Extract user ID from request or entity_id."""
    if not hasattr(request.state, "user"):
        return entity_id if entity_id else "public"
    else:
        return entity_id if entity_id else request.state.user.get("id")


# --- Entitlement enforcement (D-KSPT-1) -------------------------------------
# Authority is the signed token's entitlement, attached to request.state by
# app.middleware. Caller-supplied ids (path/body/form) are only ever FILTERS: a
# request is allowed for an id iff that id is within the token's entitlement and
# the route's action is authorized. We never fall back to caller ids or widen.


def _require_action(request: Request, action: str) -> dict:
    """Return the entitlement after asserting the route's action is authorized.

    The middleware fails closed, so a protected route always has an entitlement;
    a missing one (or an action not granted) is forbidden. Never falls open.
    """
    ent = getattr(request.state, "entitlement", None)
    if ent is None:
        raise HTTPException(status_code=403, detail="Missing entitlement")
    if action not in ent["actions"]:
        raise HTTPException(
            status_code=403, detail=f"Action '{action}' not authorized"
        )
    return ent


def _require_entity(request: Request, action: str, entity_id: Optional[str]) -> dict:
    """Assert the action is authorized AND the given entity id is within the token
    entitlement. Returns the entitlement."""
    ent = _require_action(request, action)
    if entity_id is None or str(entity_id) not in ent["entity_ids"]:
        raise HTTPException(
            status_code=403, detail="Not authorized for the requested entity"
        )
    return ent


async def save_upload_file_async(file: UploadFile, temp_file_path: str) -> None:
    """Save uploaded file asynchronously."""
    try:
        async with aiofiles.open(temp_file_path, "wb") as temp_file:
            chunk_size = 64 * 1024  # 64 KB
            while content := await file.read(chunk_size):
                await temp_file.write(content)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        # KI-02 SP-01.13 -- a save failure is OUR storage (our temp directory), so it is a service fault:
        # 503, no str(e), no temp path. describe_failure logs the exception and traceback under a
        # reference the caller is given. The path is still in the log line above for the operator.
        status_code, message = describe_failure(e, getattr(file, "filename", None))
        raise HTTPException(status_code=status_code, detail=message)


def save_upload_file_sync(file: UploadFile, temp_file_path: str) -> None:
    """Save uploaded file synchronously."""
    try:
        with open(temp_file_path, "wb") as temp_file:
            copyfileobj(file.file, temp_file)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        # KI-02 SP-01.13 -- see save_upload_file_async.
        status_code, message = describe_failure(e, getattr(file, "filename", None))
        raise HTTPException(status_code=status_code, detail=message)


def validate_file_path(base_dir: str, file_path: str) -> Optional[str]:
    """Validate that file_path resolves within base_dir. Returns resolved absolute path or None."""
    if not file_path or not file_path.strip():
        return None
    try:
        allowed = Path(base_dir).resolve()
        requested = Path(os.path.join(base_dir, file_path)).resolve()
        requested.relative_to(allowed)
        return str(requested)
    except (ValueError, RuntimeError, TypeError, OSError):
        return None


def _make_unique_temp_path(user_id: str, filename: str) -> Optional[str]:
    """Build a unique temp file path under RAG_UPLOAD_DIR/{user_id}/ to prevent
    concurrent upload collisions. Returns a validated absolute path, or None if
    the raw filename would escape RAG_UPLOAD_DIR (path traversal rejection)."""
    # Validate the raw filename to reject traversal attempts
    if validate_file_path(RAG_UPLOAD_DIR, os.path.join(user_id, filename)) is None:
        return None
    # unique_name is stem + "_" + [0-9a-f]{32} + suffix — no path separators,
    # so it cannot escape the directory validated above.
    p = Path(filename)
    unique_name = f"{p.stem}_{uuid.uuid4().hex}{p.suffix}"
    return str(Path(RAG_UPLOAD_DIR, user_id, unique_name).resolve())


# ── KI-02 SP-01.10 — honest failure attribution ───────────────────────────────────────────────────
#
# SP-01.5 fixed the production Excel failure by pinning `msoffcrypto-tool`. That fixed the INSTANCE.
# The CLASS was this: a dependency missing from OUR image surfaced as
#
#     400 "Error during file processing: No module named 'msoffcrypto'"
#
# — our fault, reported to the uploader as their file's fault, over a status code that means "your
# request is bad, do not retry", so Core's listener recorded those files as failed instead of retrying
# them once the image was fixed. Reproduced against this seam for a missing dependency, an
# out-of-memory, and a parser error that echoed our internal temp path back to the caller.
#
# These types are unambiguously OURS. Nothing about the uploaded bytes can cause them:
#   * ImportError        — includes ModuleNotFoundError: our image is missing a package.
#   * MemoryError        — our process, our limits.
#   * OSError            — includes ConnectionError, TimeoutError, PermissionError, FileNotFoundError
#                          and "no space left on device". The file we would be reading is one WE just
#                          wrote to OUR temp directory, so an OS-level failure on it is our storage,
#                          never the uploader's content.
#   * RecursionError     — our parser, not their document.
# `asyncio.TimeoutError` is listed separately because on Python 3.10 it is NOT an OSError subclass.
#
# Everything else is left at its existing status ON PURPOSE. There is no evidence for a different code,
# and changing it would silently change retry behaviour for every consumer. What changes is that the
# caller stops receiving our exception text and is told plainly that the cause is not established.
_SERVICE_FAULT_TYPES = (
    ImportError,
    MemoryError,
    OSError,
    RecursionError,
    asyncio.TimeoutError,
)


# SP-01.10b, after independent review checked the class hierarchy instead of assuming it. The typed set
# above was both too narrow and too wide.
#
# TOO NARROW, and this is the part that mattered: psycopg2.OperationalError,
# sqlalchemy.exc.OperationalError, httpx.ConnectError, redis.ConnectionError and botocore's connection
# errors are NOT OSError subclasses. A vector-DB or embeddings-API outage therefore landed on the
# non-retryable 400 path -- the exact production harm this correction exists to close, left open for its
# most likely cause. Exceptions from these libraries are always about REACHING something; none of them
# parses a document, so none can be provoked by the uploaded bytes.
#
# Detection is by MODULE, not by type, on purpose: this file must not import a driver the image may not
# carry, and a driver upgrade that renames a class must not silently reopen the hole.
_SERVICE_FAULT_MODULE_ROOTS = frozenset(
    {
        "psycopg2",
        "psycopg",
        "asyncpg",
        "sqlalchemy",
        "redis",
        "httpx",
        "httpcore",
        "urllib3",
        "requests",
        "aiohttp",
        "botocore",
        "boto3",
        # config.py supports these embeddings providers too; production uses bedrock (botocore).
        "openai",
        "google",
        "ollama",
    }
)

# TOO WIDE: PIL.UnidentifiedImageError IS an OSError and is a statement about CONTENT -- a document
# parser reaches it through embedded media. Excusing it as retryable would retry, forever, a file that
# can never work. Checked BEFORE the typed set so inheritance cannot override the specific knowledge.
_CONTENT_FAULT_MODULE_ROOTS = frozenset({"PIL"})


# SP-01.10c, after re-review found the module-root set TOO BROAD. A DB driver raises BOTH kinds of
# error: an outage (OperationalError -- ours, transient) and a complaint about the VALUE being written
# (DataError -- permanent, and in this service that value is text extracted from the uploaded file: a NUL
# byte, an invalid UTF-8 sequence, an over-length field). Module root cannot tell them apart, so 10b
# promoted DataError to a retryable 503 and would retry, forever, a file that can never work -- the PIL
# bug-shape arriving through psycopg2. Denied by CLASS NAME because every DB driver spells these the
# same way (PEP 249) and this file imports none of them.
_CONTENT_FAULT_TYPE_NAMES = frozenset(
    {
        "DataError",
        "IntegrityError",
        "UnidentifiedImageError",
    }
)

# Bounded so a self-referential chain cannot hang a request.
_CAUSE_CHAIN_LIMIT = 10


def _causes(error: BaseException):
    """The exception and its __cause__/__context__ chain, each link once, bounded.

    langchain wraps provider calls in tenacity, so a real outage can arrive as a RetryError whose cause
    is the httpx/botocore error that actually happened. Reading only the top exception's module missed
    every one of those.
    """
    seen, queue, out = set(), [error], []
    while queue and len(out) < _CAUSE_CHAIN_LIMIT:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        out.append(current)
        # `raise X from None` sets __suppress_context__: the author has said explicitly that whatever was
        # being handled is INCIDENTAL. Python's own traceback machinery hides a suppressed context, and
        # this must mirror it -- otherwise a genuine transient outage raised while some unrelated content
        # error happened to be in flight is marked permanent, which is the laundering problem running in
        # reverse and costs a file that would have worked. An explicit __cause__ is never suppressed.
        queue.append(current.__cause__)
        if not getattr(current, "__suppress_context__", False):
            queue.append(current.__context__)
    return out


def _is_name_too_long(error: BaseException) -> bool:
    """Whether any link in the chain is an ENAMETOOLONG. SP-01.15: the temp path is built from the user's
    filename, so a name too long for the filesystem is a permanent CONTENT fault -- it fails identically
    forever, so it must never be a retryable 503. Whether the length is the user's filename or our temp
    directory cannot be told from the errno, but 503 "retry, it's us" is wrong either way; the actionable
    half a caller can act on is the filename, so the message names it."""
    return any(
        isinstance(link, OSError) and link.errno == errno.ENAMETOOLONG
        for link in _causes(error)
    )


def _is_pandoc_missing(error: BaseException) -> bool:
    """Whether this failure is pandoc not being installed on the server.

    FILES-01 F2 -- this is a REGRESSION REPAIR, not a new policy. `/text` and `/local/embed` each carry
    an explicit `"No pandoc was found" in str(e)` branch answering with
    `ERROR_MESSAGES.PANDOC_NOT_INSTALLED`. Those branches sit on the OUTER handler, but the failure they
    target is raised inside `loader.lazy_load()`, and SP-01.10 now converts that into an `HTTPException`
    at the `load_file_content` seam -- which the outer handlers re-raise untouched. So the branches
    stopped being reached and the one actionable answer on those handlers was silently replaced by "the
    cause is not established". Measured, not inferred: a route test drives a pandoc-less loader and reads
    the response.

    It belongs HERE because `describe_failure` is now the single classifier every intake path flows
    through; putting it back on the outer handlers would restore a branch that can no longer execute.

    Status stays 400, exactly as it was before the regression. A missing server package is infrastructure
    but it is not TRANSIENT: no amount of retrying installs pandoc, and a 503 would tell Core's listener
    to retry forever a file that cannot work until an operator acts -- the same harm the DataError and
    ENAMETOOLONG corrections exist to prevent. The message carries the operator action instead.

    HOW IT MATCHES, and why not the obvious way. An independent review broke the first version of this,
    which was `"No pandoc was found" in str(link)` over the whole chain. That substring is
    CALLER-INFLUENCEABLE: a save-path `OSError` carries the temp path in its message, and that path is
    built by `_make_unique_temp_path` from the UPLOADER'S FILENAME. A file named
    `No pandoc was found.txt` therefore made a genuine, retryable storage outage answer 400 "install
    pandoc" -- turning a 503 into a permanent do-not-retry for every route sharing this classifier. That
    is exactly the harm the DataError and ENAMETOOLONG corrections exist to prevent, re-opened by me.

    So the match is pinned to what the library actually does, verified in the shipped image rather than
    assumed: `pypandoc/__init__.py:802` raises `OSError("No pandoc was found: either install pandoc ...")`
    -- the phrase is the START of the message. A user-controlled filename can only ever reach an OSError
    message through the `[Errno N] strerror: 'path'` form, where it is never at position 0, so
    `startswith` closes the injection. `isinstance(OSError)` narrows it further, and no code in `app/`
    constructs a single-argument OSError from user input (checked).

    NOTE for anyone tempted by the reviewer's other suggestion -- moving this check to run only when
    `is_service_fault` is False. It looks safer and would SILENTLY BREAK THE REPAIR: pypandoc raises an
    OSError, OSError is a service-fault type, so the real case would never reach the branch and the
    actionable message would be lost again. Discriminate by type and position, not by order.
    """
    return any(
        isinstance(link, OSError) and str(link).startswith("No pandoc was found")
        for link in _causes(error)
    )


def is_service_fault(error: BaseException) -> bool:
    """Whether this failure is OURS. Fail-safe direction: when unsure, say no and do not exonerate
    ourselves -- but never accuse the file either (see `describe_failure`).

    CONTENT WINS OVER THE WHOLE CHAIN, and is checked first. A permanent content fault wrapped in a
    retry must not be laundered into a transient one; that would retry forever a file that can never
    work, which is the opposite harm to the one this correction exists to fix but no less wrong.
    """
    if _is_name_too_long(error):
        return False
    chain = _causes(error)
    for link in chain:
        root = (type(link).__module__ or "").split(".")[0]
        if root in _CONTENT_FAULT_MODULE_ROOTS or type(link).__name__ in _CONTENT_FAULT_TYPE_NAMES:
            return False
    for link in chain:
        root = (type(link).__module__ or "").split(".")[0]
        if isinstance(link, _SERVICE_FAULT_TYPES) or root in _SERVICE_FAULT_MODULE_ROOTS:
            return True
    return False


def describe_failure(error: BaseException, filename: str) -> tuple:
    """Turn a non-verdict failure into `(status_code, caller_message)` and log the real detail.

    The caller gets a sentence and a reference. The operator gets the exception, its type and the
    traceback under that same reference. Withholding internals is only acceptable because the reference
    makes them findable — a test asserts the reference actually reaches the log.

    `detail` is deliberately a plain STRING on these paths. Core's direct-upload consumer interpolates
    `detail` straight into the user's toast (`crud.js:341`), so an object renders there as
    `[object Object]` — a live defect this lane found and recorded. The machine-readable half is the
    STATUS CODE, which every consumer already reads; the human half is the sentence. Neither needs the
    other to be fixed first.
    """
    reference = uuid.uuid4().hex[:12]
    name = filename or "the uploaded file"
    if _is_pandoc_missing(error):
        # FILES-01 F2 -- restore the actionable answer the outer handlers can no longer produce.
        # An operator can fix this; "the cause is not established" told nobody anything.
        logger.error(
            "File processing failed [reference=%s] [file=%s] [attribution=%s] [type=%s]: %s\nTraceback: %s",
            reference,
            name,
            "service:pandoc_not_installed",
            type(error).__name__,
            error,
            traceback.format_exc(),
        )
        return (
            status.HTTP_400_BAD_REQUEST,
            f"{ERROR_MESSAGES.PANDOC_NOT_INSTALLED} Reference: {reference}.",
        )
    if _is_name_too_long(error):
        # SP-01.15 -- we know exactly what is wrong here, so say it instead of "cause not established".
        logger.error(
            "File processing failed [reference=%s] [file=%s] [attribution=%s] [type=%s]: %s\nTraceback: %s",
            reference,
            name,
            "content:name_too_long",
            type(error).__name__,
            error,
            traceback.format_exc(),
        )
        return (
            status.HTTP_400_BAD_REQUEST,
            f"'{name}' could not be saved because the file name is too long. Shorten it and upload "
            f"again. Reference: {reference}.",
        )
    service = is_service_fault(error)
    if service:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        message = (
            f"'{name}' could not be processed because of a problem on our side, "
            f"not with your file. It is safe to try again. Reference: {reference}."
        )
    else:
        status_code = status.HTTP_400_BAD_REQUEST
        message = (
            f"'{name}' could not be read. The cause is not established \u2014 it may be the file "
            f"or this service. Reference: {reference}."
        )
    logger.error(
        "File processing failed [reference=%s] [file=%s] [attribution=%s] [type=%s]: %s\nTraceback: %s",
        reference,
        name,
        "service" if service else "undetermined",
        type(error).__name__,
        error,
        traceback.format_exc(),
    )
    return status_code, message


async def load_file_content(
    filename: str, content_type: str, file_path: str, executor
) -> tuple:
    """Load file content using appropriate loader.

    A loader that reaches a terminal verdict about the file (KI-02 SP-01.5:
    password-protected, damaged container) raises `DocumentVerdictError`, which
    is translated here into a 422 carrying BOTH the machine-readable verdict and
    a sentence the uploader can act on. This is the single seam every embed/text
    route loads through, and each of those routes re-raises `HTTPException`
    unchanged, so the honest answer reaches the caller instead of collapsing into
    the generic "Error during file processing: ..." 400. The verdict is raised
    during loading, before any `add_documents` call, so nothing is stored.
    """
    loader = None
    # Bounded local OCR for scanned PDF pages (FILES-01). `stop` is the cancellation
    # half: `run_in_executor` cancels the FUTURE when the caller goes away, but the
    # worker THREAD keeps running -- so without this a disconnected client leaves a
    # 50-page OCR burning CPU for nobody. The loader checks this flag between pages.
    #
    # There is deliberately NO `except OcrCancelled` handler here. The flag is set only
    # inside the `except asyncio.CancelledError` below, which re-raises immediately, so
    # by the time the worker thread raises `OcrCancelled` nobody is awaiting that future
    # and the exception is discarded -- which is the correct outcome, because a cancelled
    # request has no caller left to answer. The first version answered 503 there; review
    # showed it was unreachable, and an uncovered handler that implies a tested path is
    # worse than no handler (the same call made for the dead verdict guard in #29).
    stop = threading.Event()
    try:
        loader, known_type, file_ext = get_loader(
            filename,
            content_type,
            file_path,
            ocr_budget=OcrBudget(should_stop=stop.is_set),
            # Unconfigured by default: `ExtractionBudget` with both bounds at 0 is a no-op,
            # so this call costs and behaves exactly as before until an operator sets a limit.
            extraction_budget=ExtractionBudget(),
        )
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(executor, lambda: list(loader.lazy_load()))
        except asyncio.CancelledError:
            stop.set()
            raise
        return data, known_type, file_ext
    except DocumentVerdictError as verdict_error:
        logger.warning(
            "Terminal verdict for %s [verdict=%s]: %s",
            filename,
            verdict_error.verdict,
            verdict_error,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": f"'{filename or 'uploaded file'}': {verdict_error}",
                "extraction": {
                    "status": "unsupported",
                    "verdict": verdict_error.verdict,
                    "locator_kind": "none",
                    "units_total": 0,
                    "units_extracted": 0,
                    "units_empty": 0,
                    "units_image_only": 0,
                    "empty_locators": [],
                    "reasons": [
                        {"locator": None, "reason": verdict_error.verdict}
                    ],
                },
            },
        ) from verdict_error
    except HTTPException:
        # Already an honest, deliberate answer (e.g. raised by `get_loader`). Never re-wrap it.
        raise
    except Exception as error:
        # KI-02 SP-01.10 — anything that is not a terminal verdict about the FILE. The uploader is never
        # told their file is bad on the strength of an exception we have not classified, and never
        # receives `str(error)`.
        status_code, message = describe_failure(error, filename)
        raise HTTPException(status_code=status_code, detail=message) from error
    finally:
        # Clean up temporary UTF-8 file if it was created for encoding conversion
        if loader is not None:
            cleanup_temp_encoding_file(loader)


def extract_text_from_documents(documents: List[Document], file_ext: str) -> str:
    """Extract text content from loaded documents."""
    text_content = ""
    if documents:
        for doc in documents:
            if hasattr(doc, "page_content"):
                # Clean text if it's a PDF
                if file_ext == "pdf":
                    text_content += clean_text(doc.page_content) + "\n"
                else:
                    text_content += doc.page_content + "\n"

    # Remove trailing newline
    return text_content.rstrip("\n")


async def cleanup_temp_file_async(file_path: str) -> None:
    """Clean up temporary file asynchronously."""
    try:
        await aiofiles.os.remove(file_path)
    except Exception as e:
        logger.error(
            "Failed to remove temporary file | Path: %s | Error: %s | Traceback: %s",
            file_path,
            str(e),
            traceback.format_exc(),
        )


@router.get("/ids")
async def get_all_ids(request: Request):
    _require_action(request, "read")
    try:
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.get_all_ids(executor=request.app.state.thread_pool)
        else:
            ids = vector_store.get_all_ids()

        return list(set(ids))
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_all_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to get all IDs | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    try:
        if await is_health_ok():
            # `build` is additive: `status` keeps its exact existing value and meaning, so a
            # caller that only reads `status` is unaffected. It is here because this is the route
            # that actually answers in the deployed app, and a health check that cannot say WHICH
            # build is healthy leaves the only question a deployment receipt needs unanswerable.
            return {"status": "UP", "build": build_summary()}
        else:
            logger.error("Health check failed")
            return {"status": "DOWN"}, 503
    except Exception as e:
        logger.error(
            "Error during health check | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        return {"status": "DOWN", "error": str(e)}, 503


@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents_by_ids(request: Request, ids: list[str] = Query(...)):
    ent = _require_action(request, "read")
    try:
        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.get_documents_by_ids(
                ids, executor=request.app.state.thread_pool
            )
        else:
            documents = vector_store.get_documents_by_ids(ids)

        # Entitlement filter (D-KSPT-1): a document is only visible if its
        # owning entity (user_id) is within the token entitlement. Ids outside
        # the entitlement are treated as not found and never disclosed.
        documents = [
            d for d in documents if d.metadata.get("user_id") in ent["entity_ids"]
        ]
        authorized_ids = {d.metadata.get("file_id") for d in documents}
        if not all(id in authorized_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given IDs"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_documents_by_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error getting documents by IDs | IDs: %s | Error: %s | Traceback: %s",
            ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


#: The producers a delete may be narrowed to. These are the exact values the PDF loader
#: writes into `text_source` on every stored chunk; they are a CONTRACT SURFACE that Core
#: reads at retrieval time, so renaming one silently breaks a consumer.
_DELETABLE_TEXT_SOURCES = frozenset({"native", "ocr"})


@router.delete("/documents")
async def delete_documents(
    body: DeleteDocumentsBody,
    request: Request,
):
    document_ids = body.file_ids
    user_id = body.entity_id
    document_origin_type = body.document_origin_type
    subscription_id = body.subscription_id
    # FILES-01: when set, only the rows written by this producer are removed and the
    # file itself survives. Validated against a closed set rather than passed through,
    # because every filter in the delete path NARROWS: a value that reached the SQL
    # unmatched would delete nothing, but a value that got DROPPED on the way would
    # delete the whole file. A typo must be a refusal, never a wider delete.
    text_source = body.text_source
    if text_source is not None and text_source not in _DELETABLE_TEXT_SOURCES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": (
                    f"Unknown text_source '{text_source}'. Expected one of: "
                    f"{', '.join(sorted(_DELETABLE_TEXT_SOURCES))}. Nothing was deleted."
                )
            },
        )

    # Entitlement (D-KSPT-1): delete is scoped to an authorized entity. The
    # caller-supplied entity_id is only a filter; it must be within the token
    # entitlement and the token must grant the delete action. We never delete by
    # file id alone (which would cross entities).
    ent = _require_action(request, "delete")
    if user_id is None or str(user_id) not in ent["entity_ids"]:
        raise HTTPException(
            status_code=403, detail="Not authorized for the requested entity"
        )

    try:
        origin_type_value = document_origin_type.value if document_origin_type else None
        logger.info(
            "[delete_documents] request [user_id=%s][file_ids=%s][document_origin_type=%s][subscription_id=%s]",
            user_id, document_ids, origin_type_value, subscription_id,
        )
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                document_ids,
                user_id=user_id,
                document_origin_type=origin_type_value,
                subscription_id=subscription_id,
                executor=request.app.state.thread_pool,
            )
            await vector_store.delete(
                ids=document_ids,
                user_id=user_id,
                document_origin_type=origin_type_value,
                subscription_id=subscription_id,
                text_source=text_source,
                executor=request.app.state.thread_pool,
            )
        else:
            existing_ids = vector_store.get_filtered_ids(document_ids)
            if text_source is not None:
                # Only the pgvector store can narrow a delete by producer. Refusing is
                # the only honest answer for the others: silently deleting the whole
                # file would destroy exactly the text the caller asked to keep.
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail={
                        "message": (
                            "Deleting by text_source is only supported on the pgvector "
                            "store. Nothing was deleted."
                        )
                    },
                )
            vector_store.delete(ids=document_ids)

        if document_ids:
            if not all(id in existing_ids for id in document_ids):
                raise HTTPException(status_code=404, detail="One or more IDs not found")
        else:
            document_ids = list(set(existing_ids))

        logger.info(
            "[delete_documents] matched [existing_ids=%d][to_delete=%d]",
            len(existing_ids), len(document_ids),
        )

        # Delete cached summaries for the removed files. This runs for a text_source
        # delete too, and deliberately: a summary built from text that has just been
        # superseded is stale, and a stale summary presented as current is worse than
        # no summary. It is regenerated on next use.
        if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
            try:
                await delete_summaries_by_file_ids(document_ids, user_id=user_id)
            except Exception as summary_err:
                logger.warning(
                    "Failed to delete summaries for file_ids %s: %s",
                    document_ids,
                    summary_err,
                )

        file_count = len(document_ids)
        if text_source is not None:
            # NOT "deleted successfully": the file still exists and still has its other
            # rows. Saying otherwise would be the same shape of dishonest success this
            # service has spent the whole lane removing.
            return {
                "message": (
                    f"Removed the '{text_source}' rows for {file_count} "
                    f"file{'s' if file_count > 1 else ''}. The file"
                    f"{'s' if file_count > 1 else ''} and any rows from other sources "
                    f"remain."
                ),
                "text_source": text_source,
            }
        return {
            "message": f"Documents for {file_count} file{'s' if file_count > 1 else ''} deleted successfully"
        }
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in delete_documents | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to delete documents | IDs: %s | Error: %s | Traceback: %s",
            document_ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


# The embedding function is wrapped with the Redis cache in app.config, so
# embed_query transparently caches results.
def get_cached_query_embedding(query: str):
    return vector_store.embedding_function.embed_query(query)


def _to_langchain_filter(filters: dict) -> dict:
    """Convert a plain {key: value} filter into LangChain PGVector syntax.

    Scalar values become {"$eq": value}; list/tuple values become {"$in": [...]}.
    None values are dropped.
    """
    lc_filter: dict = {}
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            lc_filter[key] = {"$in": list(value)}
        else:
            lc_filter[key] = {"$eq": value}
    return lc_filter


async def _hybrid_or_dense_search(
    request: Request,
    query: str,
    embedding,
    k: int,
    filters: dict,
):
    """Retrieve chunks for a query, using hybrid (dense + keyword) search when
    enabled and falling back to dense-only otherwise or on keyword failure.

    `filters` is a plain {key: value} dict (e.g. {"file_id": ..., "user_id": ...});
    list values are supported (matched with IN/ANY semantics). Returns a list of
    (Document, score) tuples, ordered best-first.
    """
    lc_filter = _to_langchain_filter(filters)

    if not isinstance(vector_store, AsyncPgVector):
        logger.info(
            "[retrieve] mode=dense-only (sync vector store) | k=%d | filters=%s | query=%r",
            k, filters, query,
        )
        results = vector_store.similarity_search_with_score_by_vector(
            embedding, k=k, filter=lc_filter
        )
        logger.info("[retrieve] dense-only returned %d documents", len(results))
        return results

    dense_coro = vector_store.asimilarity_search_with_score_by_vector(
        embedding,
        k=k,
        filter=lc_filter,
        executor=request.app.state.thread_pool,
    )

    if not HYBRID_SEARCH_ENABLED:
        logger.info(
            "[retrieve] mode=dense-only (HYBRID_SEARCH_ENABLED=false) | k=%d | filters=%s | query=%r",
            k, filters, query,
        )
        results = await dense_coro
        logger.info("[retrieve] dense-only returned %d documents", len(results))
        return results

    logger.info(
        "[retrieve] mode=hybrid (dense + keyword) | k=%d | filters=%s | query=%r",
        k, filters, query,
    )

    # Run dense + keyword search concurrently. return_exceptions lets us salvage
    # the dense results if the keyword path fails (e.g. the document_tsv column /
    # GIN index from tempo migration 10081 has not been applied yet).
    dense_results, keyword_results = await asyncio.gather(
        dense_coro,
        keyword_search(query, k=k, filters=filters),
        return_exceptions=True,
    )

    if isinstance(dense_results, Exception):
        raise dense_results
    if isinstance(keyword_results, Exception):
        logger.warning(
            "[retrieve] keyword search failed; falling back to dense-only "
            "(dense=%d documents): %s",
            len(dense_results), keyword_results,
        )
        return dense_results

    # Intermediate results, before fusion.
    logger.info(
        "[retrieve] intermediate: dense=%d documents, keyword=%d documents",
        len(dense_results), len(keyword_results),
    )
    logger.debug(
        "[retrieve] dense scores (distance, lower=better): %s",
        [round(score, 4) for _doc, score in dense_results],
    )
    logger.debug(
        "[retrieve] keyword scores (ts_rank_cd, higher=better): %s",
        [round(score, 4) for _doc, score in keyword_results],
    )

    fused = reciprocal_rank_fusion([dense_results, keyword_results], k=k)
    logger.info(
        "[retrieve] fused (RRF) -> %d documents returned (from %d dense + %d keyword)",
        len(fused), len(dense_results), len(keyword_results),
    )
    return fused


def _cohere_rerank_enabled(args: Optional[dict]) -> bool:
    """Per-request rerank gate (VI-535).

    Reranking requires the deployment capability (RERANK_ENABLED). When callers pass
    a `cohere` flag in args (knowledge queries), it overrides on a per-request basis;
    when absent (e.g. file_id queries), the deployment default applies.
    """
    if not RERANK_ENABLED:
        return False
    if args is not None and "cohere" in args:
        return bool(args["cohere"])
    return True


async def _retrieve_documents(
    request: Request,
    query: str,
    embedding,
    k: int,
    filters: dict,
    args: Optional[dict] = None,
):
    """Full retrieval pipeline: hybrid/dense search + optional cross-encoder rerank.

    When reranking is available (VI-438) and enabled for the request (VI-535), a
    larger candidate pool is fetched from hybrid search (RERANK_CANDIDATES) and
    reranked against the question down to min(k, RERANK_TOP_N). Otherwise the
    hybrid/dense top-k is returned directly.
    Returns a list of (Document, score) tuples, ordered best-first.
    """
    if not _cohere_rerank_enabled(args):
        return await _hybrid_or_dense_search(request, query, embedding, k, filters)

    fetch_k = max(RERANK_CANDIDATES, k)
    top_n = min(k, RERANK_TOP_N)
    logger.info(
        "[retrieve] reranking enabled | fetching %d candidates, returning top %d",
        fetch_k, top_n,
    )

    candidates = await _hybrid_or_dense_search(request, query, embedding, fetch_k, filters)
    return await rerank(query, candidates, top_n=top_n)


@router.post("/query")
async def query_embeddings_by_file_id(
    body: QueryRequestBody,
    request: Request,
):
    # Entitlement (D-KSPT-1): authority is the token entitlement. A caller-supplied
    # entity_id is only a filter and must be within the entitlement. Retrieval is
    # constrained to the authorized entity set, and results are defensively
    # re-filtered so a document owned by an unauthorized entity is never returned.
    ent = _require_action(request, "read")
    if body.entity_id is not None and str(body.entity_id) not in ent["entity_ids"]:
        raise HTTPException(
            status_code=403, detail="Not authorized for the requested entity"
        )
    user_filter = [body.entity_id] if body.entity_id else list(ent["entity_ids"])

    authorized_documents = []

    try:
        embedding = get_cached_query_embedding(body.query)

        documents = await _retrieve_documents(
            request,
            body.query,
            embedding,
            body.k,
            {"file_id": body.file_id, "user_id": user_filter},
        )

        if not documents:
            return authorized_documents

        authorized_documents = [
            (doc, score)
            for (doc, score) in documents
            if doc.metadata.get("user_id") in ent["entity_ids"]
        ]
        if len(authorized_documents) != len(documents):
            logger.warning(
                "[query] filtered %d unauthorized document(s) out of %d for file_id=%s",
                len(documents) - len(authorized_documents),
                len(documents),
                body.file_id,
            )

        return authorized_documents

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_id | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query embeddings | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/{entity_id}")
async def query_embeddings_by_entity_id(
    entity_id: str,
    body: QueryByEntityBody,
    request: Request,
):
    # Entitlement (D-KSPT-1): the path entity_id is a filter; it must be within the
    # token entitlement and read must be granted.
    _require_entity(request, "read", entity_id)
    logger.info(
        "[query_embeddings_by_entity_id] request [entity_id=%s][query=%r][k=%d][args=%s]",
        entity_id, body.query, body.k, body.args
    )
    try:
        embedding = get_cached_query_embedding(body.query)

        documents = await _retrieve_documents(
            request,
            body.query,
            embedding,
            body.k,
            {"user_id": entity_id},
            body.args,
        )

        logger.info(
            "[query_embeddings_by_entity_id] results [entity_id=%s][documents_found=%d]",
            entity_id, len(documents),
        )

        if not documents:
            return []

        return documents

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_entity_id | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query by entity | Entity ID: %s | Query: %s | Error: %s | Traceback: %s",
            entity_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


async def _process_documents_async_pipeline(
    documents: List[Document],
    file_id: str,
    vector_store: "AsyncPgVector",
    executor: "ThreadPoolExecutor",
) -> List[str]:
    """
    Process documents using async producer-consumer pattern for batched embedding and insertion.

    Args:
        documents: List of Document objects to process
        file_id: Unique identifier for the file being processed
        vector_store: AsyncPgVector instance for document storage
        executor: ThreadPoolExecutor for concurrent operations

    Returns:
        List of document IDs that were successfully inserted
    """
    total_chunks = len(documents)
    if total_chunks == 0:
        return []

    # Create queues for producer-consumer pattern
    # embedding_queue is bounded to limit document data held in memory.
    # results_queue is unbounded — it holds only small UUID lists, and the
    # drain loop runs after gather(), so bounding it would deadlock when
    # num_batches > maxsize.
    embedding_queue = asyncio.Queue(maxsize=EMBEDDING_MAX_QUEUE_SIZE)
    results_queue = asyncio.Queue()
    all_ids = []

    num_batches = calculate_num_batches(total_chunks, EMBEDDING_BATCH_SIZE)

    logger.info(
        "Starting async pipeline for file %s: %d chunks with %d batch size",
        file_id,
        total_chunks,
        EMBEDDING_BATCH_SIZE,
    )

    async def batch_producer():
        """Produce document batches and put them in the queue."""
        try:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * EMBEDDING_BATCH_SIZE
                end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, total_chunks)
                batch_documents = documents[start_idx:end_idx]
                batch_ids = [file_id] * len(batch_documents)

                logger.info(
                    "Generating embeddings for batch %d/%d: chunks %d-%d",
                    batch_idx + 1,
                    num_batches,
                    start_idx,
                    end_idx - 1,
                )

                # Put batch in queue for processing
                await embedding_queue.put(
                    (batch_documents, batch_ids, batch_idx + 1, num_batches)
                )
        except Exception as e:
            logger.error("Error in batch producer: %s", e)
            raise
        finally:
            # Always signal end of production
            await embedding_queue.put(None)

    async def embedding_consumer():
        """Consume batches from queue, embed and insert into database."""
        try:
            while True:
                item = await embedding_queue.get()
                if item is None:  # End signal
                    embedding_queue.task_done()
                    break

                batch_documents, batch_ids, batch_num, total_batches = item

                logger.info(
                    "Inserting batch %d/%d into database (%d chunks)",
                    batch_num,
                    total_batches,
                    len(batch_documents),
                )

                try:
                    # Insert batch into database
                    batch_result_ids = await vector_store.aadd_documents(
                        batch_documents, ids=batch_ids, executor=executor
                    )
                    await results_queue.put(batch_result_ids)
                except Exception as e:
                    logger.error(
                        "Error processing batch %d/%d: %s", batch_num, total_batches, e
                    )
                    await results_queue.put(e)  # Put exception object
                finally:
                    embedding_queue.task_done()

        except Exception as e:
            logger.error("Fatal error in embedding consumer: %s", e)
            await results_queue.put(e)
            raise

    producer_task = None
    consumer_task = None

    try:
        # Start producer and consumer concurrently
        producer_task = asyncio.create_task(batch_producer())
        consumer_task = asyncio.create_task(embedding_consumer())

        # Wait for both to complete
        await asyncio.gather(producer_task, consumer_task, return_exceptions=False)

        # Collect results from all batches
        for _ in range(num_batches):
            result = await results_queue.get()
            if isinstance(result, Exception):
                raise result
            all_ids.extend(result)

        logger.info(
            "Async pipeline completed for file %s: %d embeddings created",
            file_id,
            len(all_ids),
        )

        return all_ids

    except Exception as e:
        logger.error("Pipeline failed for file %s: %s", file_id, e)
        if consumer_task is not None or producer_task is not None:
            # if one of the tasks is still running, cancel it
            if consumer_task is not None and not consumer_task.done():
                consumer_task.cancel()
            if producer_task is not None and not producer_task.done():
                producer_task.cancel()

            # Await cancelled tasks to ensure proper cleanup
            if consumer_task is None:
                await asyncio.gather(producer_task, return_exceptions=True)
            elif producer_task is None:
                await asyncio.gather(consumer_task, return_exceptions=True)
            else:
                await asyncio.gather(
                    consumer_task, producer_task, return_exceptions=True
                )

        # Attempt rollback only if we inserted something
        if all_ids:
            try:
                logger.warning("Performing rollback of file %s", file_id)
                await vector_store.delete(ids=[file_id], executor=executor)
                logger.info("Rollback completed for file %s", file_id)
            except Exception as cleanup_error:
                logger.error("Rollback failed for file %s: %s", file_id, cleanup_error)

        # Re-raise the original error
        raise


async def _process_documents_batched_sync(
    documents: List[Document],
    file_id: str,
    vector_store: "PgVector",
    executor: "ThreadPoolExecutor",
) -> List[str]:
    """
    Process documents in batches using synchronous vector store operations.

    Args:
        documents: List of Document objects to process
        file_id: Unique identifier for the file being processed
        vector_store: Synchronous PgVector instance for document storage
        executor: ThreadPoolExecutor for running sync operations

    Returns:
        List of document IDs that were successfully inserted
    """
    total_chunks = len(documents)
    if total_chunks == 0:
        return []

    all_ids = []
    num_batches = calculate_num_batches(total_chunks, EMBEDDING_BATCH_SIZE)

    logger.info(
        "Processing file %s with sync batching: %d batches of %d chunks each",
        file_id,
        num_batches,
        EMBEDDING_BATCH_SIZE,
    )

    loop = asyncio.get_running_loop()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * EMBEDDING_BATCH_SIZE
        end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, total_chunks)
        batch_documents = documents[start_idx:end_idx]
        batch_ids = [file_id] * len(batch_documents)

        logger.info(
            "Processing batch %d/%d: chunks %d-%d (%d chunks)",
            batch_idx + 1,
            num_batches,
            start_idx,
            end_idx - 1,
            len(batch_documents),
        )

        try:
            # Wrap sync call in executor to avoid blocking the event loop
            batch_result_ids = await loop.run_in_executor(
                executor,
                lambda docs=batch_documents, ids=batch_ids: vector_store.add_documents(
                    documents=docs, ids=ids
                ),
            )
            all_ids.extend(batch_result_ids)

        except Exception as batch_error:
            logger.error("Batch %d failed: %s", batch_idx + 1, batch_error)

            # Rollback entire file from vector store
            if (
                all_ids
            ):  # any batch succeeded (i.e., any chunks for this file were inserted)
                logger.warning("Rolling back file %s due to batch failure", file_id)
                try:
                    await loop.run_in_executor(
                        executor, lambda: vector_store.delete(ids=[file_id])
                    )
                    logger.info("Rollback completed for file %s", file_id)
                except Exception as rollback_error:
                    logger.error(
                        "Rollback failed for file %s: %s", file_id, rollback_error
                    )

            raise batch_error

    return all_ids


def generate_digest(page_content: str) -> str:
    return hashlib.md5(page_content.encode("utf-8", "ignore")).hexdigest()


def _prepare_documents_sync(
    data: Iterable[Document],
    file_id: str,
    user_id: str,
    clean_content: bool,
    document_origin_type: str = DocumentOriginType.ORGANIC.value,
    filename: str = None,
    link: str = None,
    subscription_id: str = None,
    tenant_id: str = None,
) -> List[Document]:
    """
    Synchronous document preparation - runs in executor to avoid blocking event loop.
    Handles text splitting, cleaning, and metadata preparation.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(data)

    # If `clean_content` is True, clean the page_content of each document (remove null bytes)
    if clean_content:
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

    # Preparing documents with page content and metadata for insertion.
    return [
        Document(
            page_content=doc.page_content,
            metadata={
                "file_id": file_id,
                "user_id": user_id,
                "digest": generate_digest(doc.page_content),
                "document_origin_type": document_origin_type,
                # Tenant tag (D-KSPT-1): stored on every embed path so a later
                # increment can filter by tenant. rag_api has no tenant column
                # today, so this lives in cmetadata.
                **({"tenant_id": tenant_id} if tenant_id else {}),
                **({"filename": filename} if filename else {}),
                **({"link": link} if link else {}),
                **({"subscription_id": subscription_id} if subscription_id else {}),
                **(doc.metadata or {}),
            },
        )
        for doc in documents
    ]


# Per-unit locator metadata keys, in detection precedence. Each loader emits at
# most ONE of these families (PDF -> `page`, PPTX -> `slide_number`, XLSX
# elements -> `page_name`), so the precedence only guards a defensive
# mixed-metadata edge; a format with no per-unit locator (DOCX/MD/TXT/CSV) folds
# into a single `none` unit.
_UNIT_LOCATOR_KEYS = (
    ("page", "page"),           # PDF: 0-indexed page (SafePyPDFLoader / pypdf)
    ("slide", "slide_number"),  # PPTX: 1-indexed true slide index (SlidePowerPointLoader)
    ("sheet", "page_name"),     # XLSX: sheet name (UnstructuredExcelLoader mode="elements")
)


#: How one page's local-OCR outcome becomes the receipt's unit reason (FILES-01).
#:
#: `no_page_image` and `disabled` deliberately map to nothing, so those pages keep the
#: plain `empty` they reported before OCR existed: a page with no embedded image is
#: indistinguishable HERE from a blank page, and claiming it needs escalation would be
#: inventing a fact. Everything else names what went wrong specifically.
#:
#: `ocr_no_text` and `page_limit`/`time_limit`/`cancelled` map to DIFFERENT reasons on
#: purpose. "We read the page and got nothing" and "we never looked at the page" are
#: the pair Core called out as the one that gets collapsed by accident.
_OCR_REASON_TO_UNIT_REASON = {
    "ocr_no_text": "ocr_no_text",
    "ocr_low_confidence": "ocr_low_confidence",
    "ocr_orientation_suspect": "ocr_orientation_suspect",
    "engine_unavailable": "ocr_unavailable",
    "page_limit": "ocr_not_attempted",
    "time_limit": "ocr_not_attempted",
    "cancelled": "ocr_not_attempted",
    "budget_exhausted": "ocr_not_attempted",
}

#: Outcomes where a better reader might succeed -- the ONLY ones that set
#: `escalation.recommended`. `ocr_low_confidence` belongs here even though such a page
#: DID yield stored text: it is the "nonempty text is not success" case.
_ESCALATABLE_OCR_REASONS = frozenset(
    {"ocr_no_text", "ocr_low_confidence", "ocr_orientation_suspect", "engine_unavailable",
     "page_limit", "time_limit", "cancelled", "budget_exhausted"}
)

#: Outcomes where OCR DID produce stored text that should not read as clean success.
#: Deliberately NOT used for any per-reason count: each weak page is counted under its own
#: reason exactly once, so the buckets stay addable.
#:
#: Review caught the first version counting an orientation-suspect page under BOTH
#: `pages_low_confidence` and `pages_orientation_suspect` -- one weak page, two increments.
#: The label was false as well as duplicated: a sideways page comes back at HIGH
#: confidence (~0.96 measured), which is the entire reason the geometric signal exists, so
#: reporting it as low confidence contradicted the data and this module's own docstring.
_OCR_WEAK_REASONS = frozenset({"ocr_low_confidence", "ocr_orientation_suspect"})

#: Page-level outcomes that mean a BOUND stopped the work rather than the page itself
#: being unreadable. Surfaced as `ocr.stopped_reason` so a truncated run is never
#: mistaken for a complete one.
_OCR_STOP_REASONS = frozenset({"page_limit", "time_limit", "cancelled", "budget_exhausted"})


def _extraction_receipt(data: Iterable[Document]) -> dict:
    """Build the additive extraction receipt for the /embed response (KI-02 WP-G1).

    Reports, per page/slide/sheet UNIT, whether real text was extracted — derived
    ONLY from loader signals that already exist (empty `page_content` on a scanned
    PDF page / image-only slide, the PPTX `image_only` marker, the per-slide/page/
    sheet locator metadata), NEVER success-by-default. A unit counts as EXTRACTED
    only when its content survives `clean_text(...).strip()` — the exact same
    normalization the empty guard uses and the pipeline persists — so
    `units_extracted` equals the units that actually contribute stored chunks
    (empty/image-only units carry empty content and split to zero chunks). A
    document with some extracted and some empty units is therefore reported
    `partial`, and `partial` can never be reported as a plain success.

    Shape:
      status:         'complete' (every unit extracted) | 'partial' (>=1 extracted
                      AND >=1 empty/image-only unit) | 'empty' (0 extracted units;
                      this is the 422 path).
                      ALSO forced to 'partial' when `escalation.recommended` is true,
                      so a document with pages that still need a better reader can
                      never read as `complete` on the field consumers already check --
                      including when every page yielded some text the engine does not
                      vouch for. Nonempty text is not success.
      locator_kind:   'page' | 'slide' | 'sheet' | 'none'
      units_total / units_extracted / units_empty / units_image_only
      empty_locators: sorted locators (page ints / slide ints / sheet names) of
                      every unit that yielded NO extractable text (locator-bearing
                      units only)
      reasons:        [{locator, reason: 'image_only' | 'empty'}] per non-extracted
                      locator-bearing unit
      ocr:            PRESENT ONLY when local OCR ran on at least one unit (FILES-01) —
                      {engine,
                       pages_attempted,      pages the engine actually looked at
                       pages_recovered,      pages whose stored text came from OCR
                       pages_no_text,        attempted, engine returned nothing
                       pages_not_attempted,  skipped because a bound was already spent
                       pages_low_confidence, recovered but the engine does not vouch
                       mean_confidence,      over recovered pages, or null
                       stopped_reason}       'page_limit'|'time_limit'|'cancelled'|null
                      `pages_attempted` and `pages_not_attempted` are deliberately
                      separate: "read it and got nothing" and "never looked" are
                      different facts about coverage.
      escalation:     PRESENT with `ocr`. The typed outcome for a controlled fallback
                      to the approved AWS document route — {recommended, reason,
                      locators}. rag_api states what it could not read well; it never
                      calls, chooses or pays for the fallback and owns no policy about
                      whether escalating is worth it.
      extraction_bound: PRESENT ONLY when a configured read bound stopped the loader
                      before the end of the document (FILES-01) —
                      {stopped_reason: 'page_limit' | 'time_limit',
                       pages_read: int,
                       pages_not_attempted: int | null   (null = the file's own page
                                                          count could not be established)}
                      `status` is forced to `partial` whenever this is present: a
                      stopped read is never a finished one. The pages that were never
                      opened have no locators, so they cannot appear in
                      `empty_locators` — this block is the only place their absence is
                      visible.
      text_sources:   Per-page provenance grouped by producer —
                      {'native': [...], 'ocr': [...], 'none': [...]} — so a citation
                      can say WHICH pages came from the text layer and which from OCR.
      formulas:       PRESENT ONLY for formats that report a formula scan
                      (spreadsheets; KI-02 SP-01.5) —
                      {scan: 'complete' | 'unavailable',
                       uncached_cells_total: int,
                       uncached: [{locator, cells: [...]}]}
                      A formula cell whose result was never cached in the file
                      extracts with its NUMBER MISSING (the label survives, the
                      value does not). We never compute a substitute, so this
                      block is how the caller learns the extraction is short of
                      values; `scan: 'unavailable'` means the workbook could not
                      be re-read to check (never a silent zero).

    Honesty note: PDF has no image-only signal at the loader (a scanned page and a
    truly blank page are both empty `page_content`), so PDF empty pages are
    reported `empty`, never `image_only`; only PPTX marks `image_only`.
    """
    docs = list(data)

    locator_kind = "none"
    meta_key: Optional[str] = None
    for kind, key in _UNIT_LOCATOR_KEYS:
        if any((getattr(d, "metadata", None) or {}).get(key) is not None for d in docs):
            locator_kind, meta_key = kind, key
            break

    # Group Documents into units. With no locator family every Document folds into
    # ONE logical unit (honest: the loader exposes no sub-locator to cite).
    units: dict = {}
    order: list = []
    # Spreadsheet-only, additive (KI-02 SP-01.5): a formula whose result was never
    # cached in the file extracts as a MISSING number, not a wrong one, so the
    # receipt must say so rather than let a "Total" row arrive silently blank.
    formula_scan: Optional[str] = None
    uncached_cells: dict = {}
    #: Set when a configured read bound stopped the extraction early (FILES-01). Absent means the
    #: whole document was read -- never "we did not check".
    extraction_stop: Optional[dict] = None
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        loc = meta.get(meta_key) if meta_key is not None else None
        if loc not in units:
            units[loc] = {
                "content": False,
                "image_only": False,
                "ocr": None,
                "source": None,
                "attempted": False,
            }
            order.append(loc)
        pc = getattr(d, "page_content", None)
        if pc and clean_text(pc).strip():
            units[loc]["content"] = True
        if meta.get("image_only") is True:
            units[loc]["image_only"] = True
        if meta.get("ocr_reason") is not None:
            units[loc]["ocr"] = meta["ocr_reason"]
            if meta.get("ocr_attempted"):
                units[loc]["attempted"] = True
            if meta.get("ocr_confidence") is not None:
                units[loc]["ocr_confidence"] = meta["ocr_confidence"]
        if meta.get("text_source") is not None:
            units[loc]["source"] = meta["text_source"]
        if meta.get(STOPPED_KEY) is not None:
            # Scanned across ALL documents rather than read off the last one: which page
            # carries the marker is an implementation detail of the loader, and a receipt
            # that depended on it would go quietly wrong the day that changed.
            extraction_stop = {
                "stopped_reason": meta[STOPPED_KEY],
                "pages_read": meta.get(ATTEMPTED_KEY),
                "pages_not_attempted": meta.get(NOT_ATTEMPTED_KEY),
            }
        if meta.get("formula_scan") is not None:
            formula_scan = meta["formula_scan"]
        cells = meta.get("formula_uncached_cells")
        if cells:
            # Same unit reported twice (elements mode emits several Documents per
            # sheet) carries the same cell list; keep one copy, not a duplicate.
            uncached_cells.setdefault(loc, list(cells))

    units_total = len(units)
    extracted = [loc for loc in order if units[loc]["content"]]
    image_only = [
        loc for loc in order
        if not units[loc]["content"] and units[loc]["image_only"]
    ]
    empty = [
        loc for loc in order
        if not units[loc]["content"] and not units[loc]["image_only"]
    ]

    units_extracted = len(extracted)
    if units_extracted == 0:
        status_str = "empty"
    elif units_extracted == units_total:
        status_str = "complete"
    else:
        status_str = "partial"

    def _sort_key(loc):
        return (0, loc) if isinstance(loc, (int, float)) else (1, str(loc))

    non_extracted = sorted(image_only + empty, key=_sort_key)
    reason_by_loc = {loc: "image_only" for loc in image_only}
    reason_by_loc.update({loc: "empty" for loc in empty})

    # --- local OCR outcome, and the typed escalation signal Core acts on -------------
    #
    # Every value here is derived from what the loader actually did to each page; none
    # of it decides anything about cost or providers. A page whose OCR fell short keeps
    # the honest `empty` family it already had, but gains a SPECIFIC reason, because
    # "this page is a scan we could not read well enough" and "this page is blank" are
    # different facts and only one of them is worth escalating.
    for loc in non_extracted:
        mapped = _OCR_REASON_TO_UNIT_REASON.get(units[loc]["ocr"])
        if mapped:
            reason_by_loc[loc] = mapped

    receipt = {
        "status": status_str,
        "locator_kind": locator_kind,
        "units_total": units_total,
        "units_extracted": units_extracted,
        "units_empty": len(empty),
        "units_image_only": len(image_only),
        # Only locator-bearing units are listed; a `none`-kind empty unit has no
        # locator to point at (its emptiness is conveyed by status/units_empty).
        "empty_locators": [loc for loc in non_extracted if loc is not None],
        "reasons": [
            {"locator": loc, "reason": reason_by_loc[loc]}
            for loc in non_extracted
            if loc is not None
        ],
    }

    # Present ONLY when OCR actually ran on at least one unit, so every receipt for a
    # native PDF, a workbook or a deck keeps its exact previous shape.
    ocr_locs = [loc for loc in order if units[loc]["ocr"] is not None]
    if ocr_locs:
        recovered = [loc for loc in ocr_locs if units[loc]["source"] == "ocr"]
        weak = [loc for loc in recovered if units[loc]["ocr"] == "ocr_low_confidence"]
        suspect = [loc for loc in recovered if units[loc]["ocr"] == "ocr_orientation_suspect"]
        attempted = [loc for loc in ocr_locs if units[loc]["attempted"]]
        not_attempted = [
            loc for loc in ocr_locs
            if _OCR_REASON_TO_UNIT_REASON.get(units[loc]["ocr"]) == "ocr_not_attempted"
        ]
        no_text = [loc for loc in ocr_locs if units[loc]["ocr"] == "ocr_no_text"]
        confidences = [
            units[loc]["ocr_confidence"]
            for loc in recovered
            if units[loc].get("ocr_confidence") is not None
        ]
        escalate = sorted(
            (loc for loc in ocr_locs if units[loc]["ocr"] in _ESCALATABLE_OCR_REASONS),
            key=_sort_key,
        )
        stopped = next(
            (units[loc]["ocr"] for loc in ocr_locs if units[loc]["ocr"] in _OCR_STOP_REASONS),
            None,
        )
        receipt["ocr"] = {
            "engine": OCR_ENGINE_NAME,
            # COVERAGE, in three separable counts. `pages_attempted` counts only pages
            # the engine actually looked at; a page skipped because a bound was already
            # spent is in `pages_not_attempted`, never folded into "attempted, nothing".
            "pages_attempted": len(attempted),
            "pages_recovered": len(recovered),
            "pages_no_text": len(no_text),
            "pages_not_attempted": len(not_attempted),
            # Recovered but weak. Separated from `pages_recovered` so nonempty text can
            # never read as clean success.
            "pages_low_confidence": len(weak),
            # Pages whose detected text runs vertically: read, but almost certainly
            # sideways, so most of the page was missed. Counted separately AND
            # exclusively -- it is a DIFFERENT fact from low confidence, because these
            # pages come back confident and wrong. A page is never in both buckets.
            "pages_orientation_suspect": len(suspect),
            "mean_confidence": (
                round(sum(confidences) / len(confidences), 4) if confidences else None
            ),
            # Which bound ended the work, or null when the whole document was read.
            "stopped_reason": stopped,
        }
        # The typed outcome Core escalates on. rag_api states WHAT it could not read
        # well; it does not call, choose or pay for the fallback, and it owns no policy
        # about whether escalating is worth it.
        receipt["escalation"] = {
            "recommended": bool(escalate),
            "reason": (
                _OCR_REASON_TO_UNIT_REASON.get(units[escalate[0]]["ocr"], units[escalate[0]]["ocr"])
                if escalate
                else None
            ),
            "locators": [loc for loc in escalate if loc is not None],
        }
        # THE LOAD-BEARING ONE. A document with pages still needing a better reader must
        # never present as `complete` on the field consumers already read -- including
        # the case where every page yielded SOME text but the engine does not vouch for
        # it. Nonempty text is not success. `empty` is left alone: that is the 422 path
        # and is already the strongest possible statement of failure.
        if escalate and receipt["status"] == "complete":
            receipt["status"] = "partial"

    # --- a read that was STOPPED is never a read that FINISHED ----------------------
    #
    # The same rule the escalation signal enforces, for the other way a document can be
    # short: a configured bound stopped the loader before the end of the file. Every page
    # already read is kept and stored -- this is not a refusal -- but `complete` would be a
    # false statement about coverage on the exact field consumers gate on, and the pages
    # that were never opened have no locators to appear in `empty_locators`, so without
    # this block a truncated document could look flawless.
    if extraction_stop is not None:
        receipt["extraction_bound"] = {
            "stopped_reason": extraction_stop["stopped_reason"],
            "pages_read": extraction_stop["pages_read"],
            # None means the file's own page count could not be established. Reported as
            # unknown rather than zero: "we did not open any more" and "there were no more"
            # are different claims.
            "pages_not_attempted": extraction_stop["pages_not_attempted"],
        }
        if receipt["status"] == "complete":
            receipt["status"] = "partial"

    # PER-PAGE PROVENANCE: which engine produced which page's text. Core needs this at
    # page granularity, not as one document-level label, because a citation has to be
    # able to say that page 3 came from OCR and page 4 from the native text layer.
    # Grouped rather than one object per page so a long document stays compact.
    if ocr_locs or any(units[loc]["source"] for loc in order):
        sources: dict = {}
        for loc in order:
            source = units[loc]["source"]
            if source is None:
                source = "native" if units[loc]["content"] else "none"
            sources.setdefault(source, []).append(loc)
        receipt["text_sources"] = {
            name: sorted((l for l in locs if l is not None), key=_sort_key)
            for name, locs in sorted(sources.items())
        }

    # Present ONLY for formats that report a formula scan (spreadsheets today), so
    # every existing receipt keeps its exact shape.
    if formula_scan is not None:
        receipt["formulas"] = {
            "scan": formula_scan,
            "uncached_cells_total": sum(len(c) for c in uncached_cells.values()),
            "uncached": [
                {"locator": loc, "cells": uncached_cells[loc]}
                for loc in order
                if loc in uncached_cells
            ],
        }

    return receipt


def _assert_extractable_content(
    data: Iterable[Document], filename: Optional[str]
) -> dict:
    """Empty-extraction guard + extraction receipt (KI-02 WP-C / WP-G1).

    Extraction that yields no Document, or only whitespace, must never be stored
    as a successful embed: a corrupt/scanned/empty file would otherwise return
    HTTP 200 with zero vector rows, indistinguishable from a real ingest. Raising
    here — before any `add_documents` call — guarantees no vector rows are written
    for an empty extraction. Called by every embed route.

    `data` must be a materialized sequence (all embed paths pass
    `list(loader.lazy_load())`), so this scan does not consume a one-shot
    iterator.

    Emptiness is decided from the extraction receipt: `units_extracted == 0` iff
    no unit has content that survives `clean_text(...).strip()` — the SAME
    normalization the pipeline persists (`_prepare_documents_sync` runs
    `clean_text` on the PDF path, and `clean_text` strips NUL and invalid UTF-8).
    `str.strip()` alone leaves NUL bytes and lone surrogates intact, so a page
    that is only NUL / invalid-UTF8 would pass a raw-strip guard and then be
    cleaned to '' and embedded as an empty chunk — "empty extraction counting as
    success", the one invariant this guard exists to enforce. This is byte-for-
    byte the same predicate as the prior `any(... clean_text ...)` guard.

    Returns the extraction receipt so the caller can attach it to the SUCCESS
    response (additive; existing fields byte-preserved). On an empty extraction
    the receipt (status 'empty') is embedded in the 422 body alongside the
    original human-readable message.
    """
    receipt = _extraction_receipt(data)
    if receipt["units_extracted"] == 0:
        name = filename or "uploaded file"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": (
                    f"No extractable text found in '{name}'. The file may be empty, "
                    f"image-only/scanned, corrupted, or password-protected. Nothing "
                    f"was stored."
                ),
                "extraction": receipt,
            },
        )
    return receipt


async def store_data_in_vector_db(
    data: Iterable[Document],
    file_id: str,
    user_id: str = "",
    clean_content: bool = False,
    executor=None,
    document_origin_type: str = DocumentOriginType.ORGANIC.value,
    filename: str = None,
    link: str = None,
    subscription_id: str = None,
    tenant_id: str = None,
) -> dict:
    # Run document preparation in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(
        executor,
        _prepare_documents_sync,
        data,
        file_id,
        user_id,
        clean_content,
        document_origin_type,
        filename,
        link,
        subscription_id,
        tenant_id,
    )

    try:
        if EMBEDDING_BATCH_SIZE <= 0:
            # synchronously embed the file and insert into vector store in one go
            if isinstance(vector_store, AsyncPgVector):
                ids = await vector_store.aadd_documents(
                    docs, ids=[file_id] * len(docs), executor=executor
                )
            else:
                ids = vector_store.add_documents(docs, ids=[file_id] * len(docs))
        else:
            # asynchronously embed the file and insert into vector store as it is embedding
            # to lessen memory impact and speed up slightly as the majority of the document
            # is inserted into db by the time it is fully embedded

            if isinstance(vector_store, AsyncPgVector):
                ids = await _process_documents_async_pipeline(
                    docs, file_id, vector_store, executor
                )
            else:
                # Fallback to batched processing for sync vector stores
                ids = await _process_documents_batched_sync(
                    docs, file_id, vector_store, executor
                )

        return {"message": "Documents added successfully", "ids": ids, "docs": docs}

    except Exception as e:
        logger.error(
            "Failed to store data in vector DB | File ID: %s | User ID: %s | Error: %s | Traceback: %s",
            file_id,
            user_id,
            str(e),
            traceback.format_exc(),
        )
        # FILES-01 F3 -- PROPAGATE, so the caller can attribute the failure.
        #
        # This used to return {"message": "An error occurred...", "error": str(e)}. Both that and the
        # success value are truthy dicts, so the four callers each invented their own way to read it
        # and only one was right:
        #     /summarize     `if not result or "error" in result:`  correct
        #     /local/embed   `if result:`                           ALWAYS true -> 200 {"status": true}
        #     /embed-upload  `if not result:`                       NEVER true  -> 200 {"status": true}
        #     /embed         has an `if "error" in result` check, but sets response_message = the RAW
        #                    exception and falls through to the 200 success return
        # A vector-store outage was therefore reported to the uploader and to Core as a successful
        # ingest with zero rows written -- and on /embed it also handed the caller str(e).
        #
        # My first fix returned None. That killed the fake success, but an independent review showed it
        # traded one harm for another: swallowing the exception left every store failure indistinguishable,
        # so an outage and a PERMANENT content fault both became a flat 500. `describe_failure` classifies
        # a psycopg2/sqlalchemy DataError as content (`_CONTENT_FAULT_TYPE_NAMES`) precisely because the
        # value being written is text extracted from the upload -- a NUL byte in a non-PDF extraction
        # reaches pgvector unmodified, since `clean_text` only runs when clean_content is True. Under the
        # None fix that permanent fault was answered 5xx, so a listener keying retry on 5xx would retry
        # forever a file that can never store. I introduced that; this corrects it.
        #
        # Raising is also the fix that cannot be misread: there is no sentinel value for a caller to
        # interpret, so `if result:` / `if not result:` / `"error" in result` are all moot. Every caller
        # already has `except Exception -> describe_failure`, the same seam the loader path uses, so an
        # outage becomes a retryable 503 that exonerates the file and a content fault stays a permanent
        # 400. The exception and traceback are logged here, where they happen, before it leaves.
        raise


@router.post("/local/embed")
async def embed_local_file(
    document: StoreDocument, request: Request, entity_id: str = None
):
    file_path = validate_file_path(RAG_UPLOAD_DIR, document.filepath)

    # Check if the file exists and if it is within the allowed upload directory
    if file_path is None or not os.path.exists(file_path):
        logger.warning("Path validation failed for local embed: %s", document.filepath)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.FILE_NOT_FOUND,
        )

    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")

    # Entitlement (D-KSPT-1): embedding writes under the resolved entity; that
    # entity must be within the token entitlement and write must be granted.
    ent = _require_entity(request, "write", user_id)
    tenant_id = ent["tenant_id"]

    try:
        # Loads through the shared `load_file_content`, like every other embed route,
        # instead of repeating get_loader + lazy_load here. That is what gives this
        # route the terminal verdicts too (KI-02 SP-01.5): the inline copy left an
        # encrypted or damaged workbook reported as a generic 400 here while /embed
        # answered honestly. The helper also owns the temp-encoding-file cleanup, so
        # the local `loader`/`finally` pair below is no longer needed.
        data, known_type, file_ext = await load_file_content(
            document.filename,
            document.file_content_type,
            file_path,
            request.app.state.thread_pool,
        )

        # Empty-extraction guard (KI-02 WP-C): never store an empty extraction.
        # Returns the additive extraction receipt (KI-02 WP-G1) for the response.
        extraction_receipt = _assert_extractable_content(data, document.filename)

        result = await store_data_in_vector_db(
            data,
            document.file_id,
            user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
            filename=document.filename,
            tenant_id=tenant_id,
        )

        if result:
            return {
                "status": True,
                "file_id": document.file_id,
                "filename": document.filename,
                "known_type": known_type,
                "extraction": extraction_receipt,
            }
        else:
            # Defensive only: `store_data_in_vector_db` now raises rather than returning a falsy
            # sentinel, so a store failure cannot reach here. Kept as a guard, and given the same
            # message its sibling routes use instead of the generic "Something went wrong :/" --
            # a string this lane's own tests forbid on the attributed path.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_local_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        # FILES-01 F3 (re-review NOTE-2) -- the last raw `str(e)` on the intake surface.
        #
        # This handler carried BOTH of the defects already fixed on /text:
        #   * `ERROR_MESSAGES.DEFAULT(e)` interpolates the exception verbatim
        #     (`constants.py`: f"Something went wrong :/\n{err}"), so any failure reaching here
        #     handed the caller our internal text -- and at 400, telling Core's listener that a
        #     fault which may well be ours is the uploader's file and must not be retried.
        #   * the `"No pandoc was found" in str(e)` branch was DEAD (the loader failure it targets
        #     is converted to an HTTPException upstream and re-raised by the handler above) AND was
        #     the caller-influenceable substring match that an independent review broke on /text: a
        #     save-path OSError embeds our temp path, which is built from the uploader's filename.
        #
        # Both are now the shared, reviewed `describe_failure`: pandoc keeps its actionable operator
        # message via a match pinned to type and position, a service fault becomes a retryable 503
        # that exonerates the file, and anything unclassified keeps 400 without the raw text. The
        # exception and traceback stay in the log under the reference the caller is given.
        logger.error(
            "Error in embed_local_file | File: %s | Error: %s | Traceback: %s",
            document.filename,
            str(e),
            traceback.format_exc(),
        )
        status_code, message = describe_failure(e, document.filename)
        raise HTTPException(status_code=status_code, detail=message) from e


async def _generate_summary_background(
    file_id: str,
    user_id: str,
    docs: List[Document],
    executor,
    llm_instance,
) -> None:
    """Generate and persist a file summary in the background (non-blocking)."""
    try:
        if SUM_UP_KNOWLEDGE_FILES is False:
            logger.info(f"Summary turned off [user_id=%s][file_id=%s]", user_id, file_id)
            return

        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(
            executor,
            summarize_files,
            llm_instance,
            {file_id: docs},
        )
        if summary:
            s = summary[0]
            await upsert_file_summary(file_id, user_id, s["summary"], s["chunk_count"])
            logger.info(
                "Background summary completed for file %s (%d chunks)",
                file_id,
                s["chunk_count"],
            )
    except Exception as e:
        logger.warning(
            "Background summary failed for file %s: %s (embeddings preserved)",
            file_id,
            e,
        )


@router.post("/embed")
async def embed_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
    document_owner_type: Optional[DocumentOwnerType] = Form(DocumentOwnerType.AGENT),
    document_origin_type: DocumentOriginType = Form(DocumentOriginType.ORGANIC),
    link: Optional[str] = Form(None),
    subscription_id: Optional[str] = Form(None)
):
    response_status = True
    response_message = "File processed successfully."
    known_type = None
    extraction_receipt = None

    user_id = get_user_id(request, entity_id)
    logger.info(
        "[embed_file] request [file_id=%s][filename=%s][user_id=%s][owner_type=%s][origin_type=%s][subscription_id=%s]",
        file_id, file.filename, user_id, document_owner_type, document_origin_type, subscription_id,
    )
    validated_file_path = _make_unique_temp_path(user_id, file.filename)

    if validated_file_path is None:
        logger.warning("Path validation failed for embed: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    # Entitlement (D-KSPT-1): embedding writes under the resolved entity; that
    # entity must be within the token entitlement and write must be granted.
    ent = _require_entity(request, "write", user_id)
    tenant_id = ent["tenant_id"]

    try:
        os.makedirs(os.path.dirname(validated_file_path), exist_ok=True)
        await save_upload_file_async(file, validated_file_path)
        data, known_type, file_ext = await load_file_content(
            file.filename,
            file.content_type,
            validated_file_path,
            request.app.state.thread_pool,
        )

        # Empty-extraction guard (KI-02 WP-C): never store an empty extraction.
        # Returns the additive extraction receipt (KI-02 WP-G1) for the response.
        extraction_receipt = _assert_extractable_content(data, file.filename)

        result = await store_data_in_vector_db(
            data=data,
            file_id=file_id,
            user_id=user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
            document_origin_type=document_origin_type.value,
            filename=file.filename,
            link=link,
            subscription_id=subscription_id,
            tenant_id=tenant_id,
        )

        if not result:
            response_status = False
            response_message = "Failed to process/store the file data."
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )

        logger.info(
            "[embed_file] stored [file_id=%s][chunks=%d]",
            file_id, len(result.get("docs", [])),
        )

        if "error" in result:
            response_status = False
            response_message = "Failed to process/store the file data."
            if isinstance(result["error"], str):
                response_message = result["error"]
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An unspecified error occurred.",
                )

        # Schedule summary generation as a background task (non-blocking)
        if (
            document_owner_type == DocumentOwnerType.KNOWLEDGE
            and llm is not None
            and VECTOR_DB_TYPE == VectorDBType.PGVECTOR
            and "docs" in result
        ):
            background_tasks.add_task(
                _generate_summary_background,
                file_id,
                user_id,
                result["docs"],
                request.app.state.thread_pool,
                llm,
            )

    except HTTPException as http_exc:
        response_status = False
        response_message = f"HTTP Exception: {http_exc.detail}"
        logger.error(
            "HTTP Exception in embed_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        # KI-02 SP-01.10 — same attribution the loader seam uses, for failures that happen OUTSIDE
        # loading (vector-store writes, storage). A connection error to the vector DB is our fault and
        # is ours -- and SP-01.10b is what actually makes that true: psycopg2 and sqlalchemy
        # errors are not OSError subclasses, so until module-based detection existed this comment
        # described an intention the code did not implement. The raw exception stays in the log.
        response_status = False
        status_code, response_message = describe_failure(e, getattr(file, "filename", None))
        raise HTTPException(status_code=status_code, detail=response_message)
    finally:
        await cleanup_temp_file_async(validated_file_path)

    return {
        "status": response_status,
        "message": response_message,
        "file_id": file_id,
        "filename": file.filename,
        "known_type": known_type,
        "extraction": extraction_receipt,
    }


@router.get("/documents/{id}/context")
async def load_document_context(request: Request, id: str):
    ent = _require_action(request, "read")
    ids = [id]
    try:
        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.get_documents_by_ids(
                ids, executor=request.app.state.thread_pool
            )
        else:
            documents = vector_store.get_documents_by_ids(ids)

        # Entitlement filter (D-KSPT-1): only the owning entity's document is
        # visible. An id outside the entitlement is not found and not disclosed.
        documents = [
            d for d in documents if d.metadata.get("user_id") in ent["entity_ids"]
        ]
        if not documents:
            raise HTTPException(
                status_code=404, detail="The specified file_id was not found"
            )

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No document found for the given ID"
            )

        return process_documents(documents)
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in load_document_context | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error loading document context | Document ID: %s | Error: %s | Traceback: %s",
            id,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.post("/embed-upload")
async def embed_file_upload(
    request: Request,
    file_id: str = Form(...),
    uploaded_file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    user_id = get_user_id(request, entity_id)

    validated_temp_file_path = _make_unique_temp_path(user_id, uploaded_file.filename)

    if validated_temp_file_path is None:
        logger.warning(
            "Path validation failed for embed-upload: %s", uploaded_file.filename
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    # Entitlement (D-KSPT-1): write under the resolved entity, which must be
    # within the token entitlement and have write granted.
    ent = _require_entity(request, "write", user_id)
    tenant_id = ent["tenant_id"]
    extraction_receipt = None

    try:
        os.makedirs(os.path.dirname(validated_temp_file_path), exist_ok=True)
        await save_upload_file_async(uploaded_file, validated_temp_file_path)
        data, known_type, file_ext = await load_file_content(
            uploaded_file.filename,
            uploaded_file.content_type,
            validated_temp_file_path,
            request.app.state.thread_pool,
        )

        # Empty-extraction guard (KI-02 WP-C): never store an empty extraction.
        # Returns the additive extraction receipt (KI-02 WP-G1) for the response.
        extraction_receipt = _assert_extractable_content(data, uploaded_file.filename)

        result = await store_data_in_vector_db(
            data,
            file_id,
            user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
            filename=uploaded_file.filename,
            tenant_id=tenant_id,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_file_upload | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        # KI-02 SP-01.10 — see `describe_failure`. Attribution by status code, sentence to the caller,
        # exception and traceback to the log under a shared reference.
        status_code, message = describe_failure(e, uploaded_file.filename)
        raise HTTPException(status_code=status_code, detail=message)
    finally:
        await cleanup_temp_file_async(validated_temp_file_path)

    return {
        "status": True,
        "message": "File processed successfully.",
        "file_id": file_id,
        "filename": uploaded_file.filename,
        "known_type": known_type,
        "extraction": extraction_receipt,
    }


@router.post("/query_multiple")
async def query_embeddings_by_file_ids(request: Request, body: QueryMultipleBody):
    # Entitlement (D-KSPT-1): file ids are only filters. Constrain retrieval to
    # the authorized entity set and defensively drop any document owned by an
    # entity outside the entitlement.
    ent = _require_action(request, "read")
    try:
        # Get the embedding of the query text
        embedding = get_cached_query_embedding(body.query)

        filters = {"file_id": body.file_ids, "user_id": list(ent["entity_ids"])}

        # Perform hybrid (or dense-only) search filtered by the file_ids in metadata
        documents = await _retrieve_documents(
            request,
            body.query,
            embedding,
            body.k,
            filters,
        )

        documents = [
            (doc, score)
            for (doc, score) in documents
            if doc.metadata.get("user_id") in ent["entity_ids"]
        ]

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given query"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query multiple embeddings | File IDs: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_ids,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text")
async def extract_text_from_file(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    """
    Extract text content from an uploaded file without creating embeddings.
    Returns the raw text content for text parsing purposes.
    """
    user_id = get_user_id(request, entity_id)
    validated_temp_file_path = _make_unique_temp_path(user_id, file.filename)

    if validated_temp_file_path is None:
        logger.warning("Path validation failed for text extraction: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    # Entitlement (D-KSPT-1): /text only extracts and returns text from the
    # uploaded bytes; it writes no vectors, so it is a READ operation (Core mints
    # act=['read'] for it). The resolved entity must be within the entitlement.
    _require_entity(request, "read", user_id)

    try:
        os.makedirs(os.path.dirname(validated_temp_file_path), exist_ok=True)
        await save_upload_file_async(file, validated_temp_file_path)
        data, known_type, file_ext = await load_file_content(
            file.filename,
            file.content_type,
            validated_temp_file_path,
            request.app.state.thread_pool,
        )

        # Extract text content from loaded documents
        text_content = extract_text_from_documents(data, file_ext)

        return {
            "text": text_content,
            "file_id": file_id,
            "filename": file.filename,
            "known_type": known_type,
        }

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in extract_text_from_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        # FILES-01 F2 -- the last caller-facing `str(e)` on the /text intake path. (NOT the last in the
        # file: /local/embed's else-branch still returns ERROR_MESSAGES.DEFAULT(e), which interpolates
        # str(e) verbatim. That route is outside this increment; recorded, not silently absorbed.)
        #
        # `save_upload_file_async` (SP-01.13) and `load_file_content` (SP-01.10) now raise
        # `HTTPException` and are re-raised untouched above. Two calls in the `try` are behind neither
        # seam: `os.makedirs`, whose `PermissionError`/`OSError` carries OUR temp directory in `str(e)`,
        # and `extract_text_from_documents`. Both landed here and were echoed to the caller verbatim --
        # proven at route level, with an injected marker that reached the response body.
        #
        # Worse than the disclosure: every one of them was answered 400, telling the caller their file is
        # bad and telling Core's listener not to retry, for faults that are ours. `describe_failure`
        # attributes it instead -- a service fault becomes a retryable 503 that exonerates the file, a
        # missing pandoc keeps its actionable operator message, and anything unclassified keeps its 400
        # but says the cause is not established rather than guessing. The exception and traceback stay in
        # the log, under the reference the caller is given.
        #
        # The `"No pandoc was found"` branch that stood here is gone deliberately, not dropped: it could
        # no longer execute (the loader failure it targets is converted to an `HTTPException` upstream),
        # and the answer it produced now comes from `describe_failure` where it is reachable again.
        logger.error(
            "Error during text extraction | File: %s | Error: %s | Traceback: %s",
            file.filename,
            str(e),
            traceback.format_exc(),
        )
        status_code, message = describe_failure(e, file.filename)
        raise HTTPException(status_code=status_code, detail=message) from e
    finally:
        await cleanup_temp_file_async(validated_temp_file_path)


@router.post("/summarize/{entity_id}")
async def summarize_entity_files(
    request: Request,
    entity_id: str,
    file_id: str = Form(...),
    knowledge_id: str = Form(...),
):
    """Retrieve all documents for an entity, group by file_id, summarize each file,
    and embed the combined summary into the vector store under the given file_id."""
    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider not configured. Set LLM_PROVIDER env var.",
        )

    user_id = get_user_id(request, entity_id)
    # Entitlement (D-KSPT-1): summarization reads the entity's documents and
    # writes the summary under knowledge_id. Both the source entity and the
    # destination knowledge id must be within the token entitlement, and write
    # must be granted.
    ent = _require_entity(request, "write", user_id)
    if str(knowledge_id) not in ent["entity_ids"]:
        raise HTTPException(
            status_code=403, detail="Not authorized for the requested knowledge id"
        )
    tenant_id = ent["tenant_id"]
    logger.info(
        "[summarize_entity_files] request [entity_id=%s][user_id=%s][file_id=%s][knowledge_id=%s]",
        entity_id, user_id, file_id, knowledge_id,
    )

    try:
        # Try to load cached summaries first (PGVector only)
        cached_summaries = []
        cached_file_ids = set()
        if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
            try:
                cached_summaries = await get_summaries_by_user(user_id)
                cached_file_ids = {s["file_id"] for s in cached_summaries}
            except Exception as cache_err:
                logger.warning("Failed to load cached summaries: %s", cache_err)

        if isinstance(vector_store, AsyncPgVector):
            grouped_docs = await vector_store.get_documents_grouped_by_file_id(
                user_id=user_id, executor=request.app.state.thread_pool
            )
        else:
            grouped_docs = vector_store.get_documents_grouped_by_file_id(
                user_id=user_id
            )

        if not grouped_docs:
            logger.info("[summarize_entity_files] no documents found [entity_id=%s]", entity_id)
            return {
                "entity_id": entity_id,
                "file_count": 0,
                "summaries": [],
            }

        logger.info(
            "[summarize_entity_files] documents loaded [entity_id=%s][total_files=%d][cached=%d][to_compute=%d]",
            entity_id, len(grouped_docs), len(cached_file_ids),
            sum(1 for fid in grouped_docs if fid not in cached_file_ids),
        )

        # Build results: use cached summaries where available, compute on-the-fly for the rest
        summaries = []
        files_to_summarize = {}

        for fid, docs in grouped_docs.items():
            if fid in cached_file_ids:
                cached = next(s for s in cached_summaries if s["file_id"] == fid)
                summaries.append({
                    "file_id": fid,
                    "summary": cached["summary"],
                    "chunk_count": cached["chunk_count"],
                })
            else:
                files_to_summarize[fid] = docs

        # Compute on-the-fly summaries for files without cache
        if files_to_summarize:
            loop = asyncio.get_running_loop()
            on_the_fly = await loop.run_in_executor(
                request.app.state.thread_pool,
                summarize_files,
                llm,
                files_to_summarize,
            )
            summaries.extend(on_the_fly)

            # Persist newly computed summaries to DB for future requests
            if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
                for s in on_the_fly:
                    try:
                        await upsert_file_summary(
                            s["file_id"], user_id, s["summary"], s["chunk_count"]
                        )
                    except Exception as persist_err:
                        logger.warning(
                            "Failed to persist summary for file %s: %s",
                            s["file_id"],
                            persist_err,
                        )

        # Embed the combined summary text into the vector store under the provided file_id
        combined_summary_text = "\n\n".join(
            s["summary"] for s in summaries if s.get("summary")
        )
        if combined_summary_text:
            summary_documents = [Document(
                page_content=combined_summary_text,
                metadata={"source": entity_id},
            )]
            result = await store_data_in_vector_db(
                data=summary_documents,
                file_id=file_id,
                user_id=knowledge_id,
                clean_content=False,
                executor=request.app.state.thread_pool,
                tenant_id=tenant_id,
            )
            if not result or "error" in result:
                error_detail = result.get("error", "Unknown error") if result else "No result"
                logger.error(
                    "Failed to embed summary for entity %s | file_id: %s | Error: %s",
                    entity_id,
                    file_id,
                    error_detail,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Summary generated but embedding failed: {error_detail}",
                )

        return {
            "entity_id": entity_id,
            "file_id": file_id,
            "file_count": len(grouped_docs),
            "summaries": summaries,
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(
            "Error summarizing files for entity %s | Error: %s | Traceback: %s",
            entity_id,
            str(e),
            traceback.format_exc(),
        )
        # FILES-01 F3 -- this handler became REACHABLE for store failures when
        # `store_data_in_vector_db` started propagating instead of swallowing, so leaving `str(e)` here
        # would have turned a fix into a new leak on a route that previously never saw those exceptions.
        # Same reviewed contract as every other intake path: attribution by status code, a sentence plus
        # a reference to the caller, the exception and traceback to the log.
        #
        # `None`, not `file_id`: this route has no filename in scope, and passing the id put an
        # internal identifier where the sentence says "filename" -- which would have told a user to
        # "shorten the file name" of something that is not a name they chose. `describe_failure`
        # falls back to "the uploaded file": vaguer, but not wrong.
        status_code, message = describe_failure(e, None)
        raise HTTPException(status_code=status_code, detail=message) from e
