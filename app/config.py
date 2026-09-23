# app/config.py
import os
import json
import boto3
import logging
import urllib.parse
from enum import Enum
from typing import Optional
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.vector_store.factory import get_vector_store

load_dotenv(find_dotenv())


class VectorDBType(Enum):
    PGVECTOR = "pgvector"
    ATLAS_MONGO = "atlas-mongo"


class EmbeddingsProvider(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    HUGGINGFACETEI = "huggingfacetei"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    GOOGLE_GENAI = "google_genai"
    GOOGLE_VERTEXAI = "vertexai"


def get_env_variable(
    var_name: str, default_value: str = None, required: bool = False
) -> str:
    value = os.getenv(var_name)
    if value is None:
        if default_value is None and required:
            raise ValueError(f"Environment variable '{var_name}' not found.")
        return default_value
    return value


RAG_HOST = os.getenv("RAG_HOST", "0.0.0.0")
RAG_PORT = int(os.getenv("RAG_PORT", 8000))

RAG_UPLOAD_DIR = get_env_variable("RAG_UPLOAD_DIR", "./uploads/")
if not os.path.exists(RAG_UPLOAD_DIR):
    os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)

VECTOR_DB_TYPE = VectorDBType(
    get_env_variable("VECTOR_DB_TYPE", VectorDBType.PGVECTOR.value)
)
POSTGRES_USE_UNIX_SOCKET = (
    get_env_variable("POSTGRES_USE_UNIX_SOCKET", "False").lower() == "true"
)
POSTGRES_DB = get_env_variable("POSTGRES_DB", "mydatabase")
POSTGRES_USER = get_env_variable("POSTGRES_USER", "myuser")
POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD", "mypassword")
DB_HOST = get_env_variable("DB_HOST", "db")
DB_PORT = get_env_variable("DB_PORT", "5432")
COLLECTION_NAME = get_env_variable("COLLECTION_NAME", "testcollection")
ATLAS_MONGO_DB_URI = get_env_variable(
    "ATLAS_MONGO_DB_URI", "mongodb://127.0.0.1:27018/LibreChat"
)
ATLAS_SEARCH_INDEX = get_env_variable("ATLAS_SEARCH_INDEX", "vector_index")
MONGO_VECTOR_COLLECTION = get_env_variable(
    "MONGO_VECTOR_COLLECTION", None
)  # Deprecated, backwards compatability
CHUNK_SIZE = int(get_env_variable("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(get_env_variable("CHUNK_OVERLAP", "100"))

# Batch processing configuration for memory-constrained environments.
# When EMBEDDING_BATCH_SIZE > 0, documents are processed in batches to reduce
# peak memory usage. This is useful for Kubernetes pods with memory limits.
#
# Trade-offs:
# - Smaller batch size = lower memory, more DB round trips
# - Larger batch size = higher memory, fewer DB round trips
# - 0 = disable batching, process all at once
#
# Default of 500 is conservative and works well for most embedding providers.
# Increase to 750 for higher throughput at the cost of higher peak memory.
EMBEDDING_BATCH_SIZE = int(get_env_variable("EMBEDDING_BATCH_SIZE", "500"))

# Maximum number of batches to buffer in memory during async processing.
# Higher values allow more parallelism but use more memory.
EMBEDDING_MAX_QUEUE_SIZE = int(get_env_variable("EMBEDDING_MAX_QUEUE_SIZE", "3"))

env_value = get_env_variable("PDF_EXTRACT_IMAGES", "False").lower()
PDF_EXTRACT_IMAGES = True if env_value == "true" else False

# --- Local-first OCR for scanned PDF pages (FILES-01, FS-CONTINUE-R3) ---
#
# A scanned PDF has no text layer, so pypdf extracts nothing from it. Richard approved
# reading native text first and then applying BOUNDED local OCR to the pages that need
# it, with escalation to the already-approved AWS route when local OCR is not good
# enough. Nothing here calls an external provider or spends money.
#
# PDF_EXTRACT_IMAGES above is a DIFFERENT, pre-existing switch (langchain's own image
# wiring) and is NOT the OCR control: measured in the shipped image it returns 0
# characters on a genuine scan even with an OCR image parser attached.
#
# Every default below is a measured number, not a guess (deployed lite image, network
# disabled) -- see app/utils/ocr.py for the measurement note behind each one.
PDF_OCR_ENABLED = get_env_variable("PDF_OCR_ENABLED", "True").lower() in (
    "true",
    "1",
    "yes",
    "y",
    "t",
)
# A BACKSTOP, not the bound that actually binds. The engine alone reads a page in ~0.9 s,
# but through the real route in a 4 GB container it is ~1.9 s/page, so the time budget
# below runs out first -- measured at 32 of 50 pages, and 33 of 120.
# (An earlier version of this comment justified 50 as "~45 s of OCR" from the 0.9 s
# figure. That was the isolated engine measurement used to PREDICT the deployed path
# instead of measuring it, and the deployed path is twice as slow. The defaults were
# safe either way, because the tighter bound wins; the reasoning was not.)
PDF_OCR_MAX_PAGES = int(get_env_variable("PDF_OCR_MAX_PAGES", "50"))
# THE effective bound. Half of Core's 120 s /embed client timeout, leaving the other half
# for parsing, chunking and embedding. Measured: total request time is capped at ~62 s
# whatever the document size -- a 120-page scan takes the same ~62 s as a 50-page one and
# reports the remaining 87 pages as not attempted. A document needing more OCR than this
# is reported partial with stopped_reason=time_limit, never silently truncated.
PDF_OCR_TIME_BUDGET_SECONDS = float(get_env_variable("PDF_OCR_TIME_BUDGET_SECONDS", "60"))
# How long a write waits for another write of the SAME file_id to finish (FILES-01 F02)
# before giving up with nothing stored. Core's /embed client timeout -- the same 120 s the
# OCR budget above is derived from: waiting longer than the caller would is pointless, and
# a caller that leaves sooner ends the wait itself.
FILE_WRITE_LOCK_WAIT_SECONDS = float(get_env_variable("FILE_WRITE_LOCK_WAIT_SECONDS", "120"))
# Bounds page expansion: a page carrying dozens of small images is a figure-heavy page,
# not a scan, and OCR-ing all of them buys nothing.
PDF_OCR_MAX_IMAGES_PER_PAGE = int(get_env_variable("PDF_OCR_MAX_IMAGES_PER_PAGE", "8"))
# Character recall is flat from 3.7 MP to 33.7 MP but peak RSS is not (764 MB -> 1.0 GB),
# so an oversized image is downscaled to this cap before OCR. 16 MP is ~2x a 300 dpi A4
# scan: comfortably above the resolution where quality stops improving.
PDF_OCR_MAX_PIXELS = int(get_env_variable("PDF_OCR_MAX_PIXELS", "16000000"))
# REPORTING thresholds, not discard thresholds. Text below them is still extracted and
# still stored -- the page is simply reported as weak so Core can escalate it, because
# judging whether a document is well enough covered is Core's call, not this service's.
# Measured: a good upright scan returns hundreds of characters at confidence 0.95-0.97,
# while a page the engine cannot read returns 0 characters at confidence 0.00, and
# nothing in the fixture corpus landed between.
PDF_OCR_MIN_CHARS = int(get_env_variable("PDF_OCR_MIN_CHARS", "24"))
PDF_OCR_LOW_CONFIDENCE_BELOW = float(
    get_env_variable("PDF_OCR_LOW_CONFIDENCE_BELOW", "0.5")
)
# A page is reported as probably sideways when this share of its detected text boxes
# are taller than they are wide. Measured on the fixture corpus: 0.00 on every upright
# page, 1.00 on a sideways one -- so 0.6 sits in a very wide empty margin rather than
# being tuned. This matters because a sideways scan loses ~60% of its characters while
# mean confidence stays HIGH (0.89-0.96 across fixtures): confidence alone cannot see it.
PDF_OCR_SIDEWAYS_BOX_RATIO = float(get_env_variable("PDF_OCR_SIDEWAYS_BOX_RATIO", "0.6"))

# --- Mixed text/image page OCR (FILES-01 F05) ---
#
# A page WITH a text layer may ALSO carry an image holding text the layer lacks (a scanned
# body under a typed header). #66 disclosed that gap (image_ocr_coverage=not_attempted);
# this reads it: when ON, a native page's embedded image is OCR'd through the SAME local
# adapter and budget as a scanned page, and any text it yields that the layer does not
# already contain is stored as a sibling chunk citing the same page. Coverage then flips to
# `attempted`.
#
# DEFAULT OFF, on purpose. With it off, behaviour is byte-identical to before (native pages
# are never OCR'd) and the honest disclosure from #66 stands. Turning it ON costs ~1.3 s per
# text-layer page that carries an image (measured, F05 audit) -- a cost/latency choice that
# belongs to the operator, not a default this service imposes. It never calls a paid or
# network provider; the paid fallback stays held.
PDF_OCR_MIXED_PAGE = get_env_variable("PDF_OCR_MIXED_PAGE", "False").lower() in (
    "true",
    "1",
    "yes",
    "y",
    "t",
)

# --- Bounded NATIVE PDF extraction (FILES-01) ---
#
# THE MECHANISM IS HERE; THE NUMBERS ARE THE OPERATOR'S. Both bounds default to 0 = OFF, so with
# no configuration this service reads a native PDF exactly as it did before. That is deliberate:
# a safe value depends on facts this lane cannot read -- the ECS task's memory limit, the load
# balancer's idle timeout, and the ingress body cap actually in force -- and a number chosen
# without them would be a prediction dressed as a measurement. Twice already in this lane a bound
# justified from the wrong measurement had to be corrected.
#
# What the bounds protect against, measured in the shipped lite image: an unbounded native PDF
# keeps parsing and allocating long after every timeout in the chain has expired, for a response
# no caller is still waiting for -- and a worker killed for memory produces none of the honest
# failure contract (retryable 503, actionable 400, log reference), just a dropped connection.
# See evidence FILES-01 native-pdf-capacity for the measured curve at the size the edge permits.
#
# When a bound stops the work, every page already read is KEPT and the receipt reports `partial`
# with the bound that stopped it. A truncated extraction is never reported as complete, and never
# as empty.
PDF_EXTRACT_MAX_PAGES = int(get_env_variable("PDF_EXTRACT_MAX_PAGES", "0"))
PDF_EXTRACT_TIME_BUDGET_SECONDS = float(
    get_env_variable("PDF_EXTRACT_TIME_BUDGET_SECONDS", "0")
)

# --- Hybrid retrieval (VI-436) ---
# Enable BM25/keyword full-text search alongside the dense vector search and
# fuse the two result sets. When disabled, retrieval behaves exactly as before
# (dense-only). Failures in the keyword path fall back to dense-only at runtime.
HYBRID_SEARCH_ENABLED = get_env_variable("HYBRID_SEARCH_ENABLED", "True").lower() in (
    "true",
    "1",
    "yes",
    "y",
    "t",
)
# Postgres text-search configuration used to build the tsquery at query time.
# MUST match the configuration used to build the document_tsv generated column
# in the DB migration (tempo migration 10081 uses 'english').
FTS_CONFIG = get_env_variable("RAG_FTS_CONFIG", "english")
# Reciprocal Rank Fusion constant (k). Larger values flatten the contribution
# of top ranks; 60 is the commonly used default.
RRF_K = int(get_env_variable("RRF_K", "60"))

# --- Reranking: Cohere Rerank 3.5 via AWS Bedrock (VI-438) ---
# After hybrid retrieval, rerank the candidate pool against the question and keep
# the best ones. Best-effort: on failure it falls back to the pre-rerank order.
# AWS credentials/region are reused from the Bedrock setup.
# NOTE: Cohere Rerank 3.5 availability is REGION-SPECIFIC, so probe it rather than assume it.
# This comment used to assert that us-east-1 does NOT host the model. That is measurably wrong
# here: cohere.rerank-v3-5:0 SUCCEEDED in us-east-1 on a real 18eca4c build, 2026-09-23T21:47Z
# (RV-118 note 1). That measurement is ONE account on ONE date, so it does not license the
# opposite absolute either -- hence "probe", not "us-east-1 works". A wrong guess costs recall,
# not correctness: if the call fails, rerank degrades to the pre-rerank order.
RERANK_ENABLED = get_env_variable("RERANK_ENABLED", "True").lower() in (
    "true",
    "1",
    "yes",
    "y",
    "t",
)
RERANK_MODEL = get_env_variable("RERANK_MODEL", "cohere.rerank-v3-5:0")
RERANK_AWS_REGION = get_env_variable("RERANK_AWS_REGION", "us-east-1")
# Candidate pool fetched from hybrid search before reranking (30-50).
RERANK_CANDIDATES = int(get_env_variable("RERANK_CANDIDATES", "40"))
# Max chunks returned after reranking; effective count is min(requested k, this).
RERANK_TOP_N = int(get_env_variable("RERANK_TOP_N", "10"))

if POSTGRES_USE_UNIX_SOCKET:
    connection_suffix = f"{urllib.parse.quote_plus(POSTGRES_USER)}:{urllib.parse.quote_plus(POSTGRES_PASSWORD)}@/{urllib.parse.quote_plus(POSTGRES_DB)}?host={urllib.parse.quote_plus(DB_HOST)}"
else:
    connection_suffix = f"{urllib.parse.quote_plus(POSTGRES_USER)}:{urllib.parse.quote_plus(POSTGRES_PASSWORD)}@{DB_HOST}:{DB_PORT}/{urllib.parse.quote_plus(POSTGRES_DB)}"

CONNECTION_STRING = f"postgresql+psycopg2://{connection_suffix}"
DSN = f"postgresql://{connection_suffix}"

## Logging

HTTP_RES = "http_res"
HTTP_REQ = "http_req"

logger = logging.getLogger()

debug_mode = os.getenv("DEBUG_RAG_API", "False").lower() in (
    "true",
    "1",
    "yes",
    "y",
    "t",
)
console_json = get_env_variable("CONSOLE_JSON", "False").lower() == "true"

if debug_mode:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

if console_json:

    class JsonFormatter(logging.Formatter):
        def __init__(self):
            super(JsonFormatter, self).__init__()

        def format(self, record):
            json_record = {}

            json_record["message"] = record.getMessage()

            if HTTP_REQ in record.__dict__:
                json_record[HTTP_REQ] = record.__dict__[HTTP_REQ]

            if HTTP_RES in record.__dict__:
                json_record[HTTP_RES] = record.__dict__[HTTP_RES]

            if record.levelno == logging.ERROR and record.exc_info:
                json_record["exception"] = self.formatException(record.exc_info)

            timestamp = datetime.fromtimestamp(record.created)
            json_record["timestamp"] = timestamp.isoformat()

            # add level
            json_record["level"] = record.levelname
            json_record["filename"] = record.filename
            json_record["lineno"] = record.lineno
            json_record["funcName"] = record.funcName
            json_record["module"] = record.module
            json_record["threadName"] = record.threadName

            return json.dumps(json_record)

    formatter = JsonFormatter()
else:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

handler = logging.StreamHandler()  # or logging.FileHandler("app.log")
handler.setFormatter(formatter)
logger.addHandler(handler)


class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        logger_method = logger.info

        if str(request.url).endswith("/health"):
            logger_method = logger.debug

        logger_method(
            f"Request {request.method} {request.url} - {response.status_code}",
            extra={
                HTTP_REQ: {"method": request.method, "url": str(request.url)},
                HTTP_RES: {"status_code": response.status_code},
            },
        )

        return response


logging.getLogger("uvicorn.access").disabled = True

## Credentials

OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY", "")
RAG_OPENAI_API_KEY = get_env_variable("RAG_OPENAI_API_KEY", OPENAI_API_KEY)
RAG_OPENAI_BASEURL = get_env_variable("RAG_OPENAI_BASEURL", None)
RAG_OPENAI_PROXY = get_env_variable("RAG_OPENAI_PROXY", None)
AZURE_OPENAI_API_KEY = get_env_variable("AZURE_OPENAI_API_KEY", "")
RAG_AZURE_OPENAI_API_VERSION = get_env_variable("RAG_AZURE_OPENAI_API_VERSION", None)
RAG_AZURE_OPENAI_API_KEY = get_env_variable(
    "RAG_AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY
)
AZURE_OPENAI_ENDPOINT = get_env_variable("AZURE_OPENAI_ENDPOINT", "")
RAG_AZURE_OPENAI_ENDPOINT = get_env_variable(
    "RAG_AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT
).rstrip("/")
HF_TOKEN = get_env_variable("HF_TOKEN", "")
OLLAMA_BASE_URL = get_env_variable("OLLAMA_BASE_URL", "http://ollama:11434")
AWS_ACCESS_KEY_ID = get_env_variable("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = get_env_variable("AWS_SECRET_ACCESS_KEY", "")
GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY", "")
GOOGLE_KEY = get_env_variable("GOOGLE_KEY", GOOGLE_API_KEY)
RAG_GOOGLE_API_KEY = get_env_variable("RAG_GOOGLE_API_KEY", GOOGLE_KEY)
AWS_SESSION_TOKEN = get_env_variable("AWS_SESSION_TOKEN", "")
AWS_DEFAULT_REGION = get_env_variable("AWS_DEFAULT_REGION", "us-east-1")
GOOGLE_APPLICATION_CREDENTIALS = get_env_variable("GOOGLE_APPLICATION_CREDENTIALS", "")
env_value = get_env_variable("RAG_CHECK_EMBEDDING_CTX_LENGTH", "True").lower()
RAG_CHECK_EMBEDDING_CTX_LENGTH = True if env_value == "true" else False

## Authentication / entitlement configuration (D-KSPT-1)
# The signing secret is read live per-request in app.middleware; this module-level
# value drives the startup guard. Fail closed UNCONDITIONALLY: without JWT_SECRET
# the service refuses to start. There is no opt-out.
JWT_SECRET = os.getenv("JWT_SECRET")


def require_auth_config() -> None:
    """Refuse to start when authentication cannot be enforced (D-KSPT-1).

    Called from the app lifespan. If ``JWT_SECRET`` is unset, raise so the process
    never comes up in an auth-less ("open") state. Missing JWT configuration must
    fail closed unconditionally — there is no bypass.
    """
    if not os.getenv("JWT_SECRET"):
        raise RuntimeError(
            "JWT_SECRET is not set; refusing to start because protected routes "
            "could not be authenticated. Set JWT_SECRET."
        )

## Embeddings


def init_embeddings(provider, model):
    if provider == EmbeddingsProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            api_key=RAG_OPENAI_API_KEY,
            openai_api_base=RAG_OPENAI_BASEURL,
            openai_proxy=RAG_OPENAI_PROXY,
            chunk_size=EMBEDDINGS_CHUNK_SIZE,
            check_embedding_ctx_length=RAG_CHECK_EMBEDDING_CTX_LENGTH,
        )
    elif provider == EmbeddingsProvider.AZURE:
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=model,
            api_key=RAG_AZURE_OPENAI_API_KEY,
            azure_endpoint=RAG_AZURE_OPENAI_ENDPOINT,
            api_version=RAG_AZURE_OPENAI_API_VERSION,
            chunk_size=EMBEDDINGS_CHUNK_SIZE,
            check_embedding_ctx_length=RAG_CHECK_EMBEDDING_CTX_LENGTH,
        )
    elif provider == EmbeddingsProvider.HUGGINGFACE:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model, encode_kwargs={"normalize_embeddings": True}
        )
    elif provider == EmbeddingsProvider.HUGGINGFACETEI:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        return HuggingFaceEndpointEmbeddings(model=model)
    elif provider == EmbeddingsProvider.OLLAMA:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)
    elif provider == EmbeddingsProvider.GOOGLE_GENAI:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=RAG_GOOGLE_API_KEY or None,
        )
    elif provider == EmbeddingsProvider.GOOGLE_VERTEXAI:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=RAG_GOOGLE_API_KEY or None,
            vertexai=True,
            project=get_env_variable("GOOGLE_CLOUD_PROJECT", None),
            location=get_env_variable("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    elif provider == EmbeddingsProvider.BEDROCK:
        from langchain_aws import BedrockEmbeddings

        session_kwargs = {
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "region_name": AWS_DEFAULT_REGION,
        }

        if AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN

        session = boto3.Session(**session_kwargs)
        return BedrockEmbeddings(
            client=session.client("bedrock-runtime"),
            model_id=model,
            region_name=AWS_DEFAULT_REGION,
        )
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")


# Operator-approved embeddings providers (D-KSPT-2; ported from RATB-01 5d9fe48).
# rag_api cannot distinguish client / second-tenant documents from others, and
# Richard's ruling is that OpenAI embeddings are NOT authorized for client or
# second-tenant documents — Bedrock Titan is the explicitly approved provider.
# So a syntactically-valid provider is not enough: it must also appear in this
# allow-list. Default is `bedrock` only; an operator may widen it explicitly via
# RAG_APPROVED_EMBEDDINGS_PROVIDERS (comma-separated). OpenAI is therefore never
# used unless an operator approves it deliberately — never a silent fallback.
def _approved_embeddings_providers() -> list:
    return [
        p.strip().lower()
        for p in get_env_variable(
            "RAG_APPROVED_EMBEDDINGS_PROVIDERS", "bedrock"
        ).split(",")
        if p.strip()
    ]


RAG_APPROVED_EMBEDDINGS_PROVIDERS = _approved_embeddings_providers()


def resolve_embeddings_provider(
    value: Optional[str], approved: Optional[list] = None
) -> EmbeddingsProvider:
    """Resolve the configured embeddings provider, failing closed (D-KSPT-2).

    There is NO default. A missing or unknown ``EMBEDDINGS_PROVIDER`` raises so
    that documents are never embedded (or queried) through an unintended provider.
    Additionally the provider must be in the operator-approved allow-list
    (``approved``; defaults to ``RAG_APPROVED_EMBEDDINGS_PROVIDERS``, i.e.
    ``bedrock`` unless widened) — a valid-but-unapproved provider (e.g. an
    explicitly-set ``openai``) also fails closed. Production uses ``bedrock``.
    """
    accepted = ", ".join(p.value for p in EmbeddingsProvider)
    if value is None or str(value).strip() == "":
        raise ValueError(
            "EMBEDDINGS_PROVIDER is required and has no default. Set it explicitly "
            f"to one of: {accepted}. Production uses 'bedrock' (Amazon Titan)."
        )
    provider_raw = str(value).strip().lower()
    try:
        provider = EmbeddingsProvider(provider_raw)
    except ValueError:
        raise ValueError(
            f"Unknown EMBEDDINGS_PROVIDER {value!r}. Accepted values: {accepted}."
        )
    approved_set = approved if approved is not None else RAG_APPROVED_EMBEDDINGS_PROVIDERS
    if provider_raw not in approved_set:
        raise ValueError(
            f"EMBEDDINGS_PROVIDER '{provider_raw}' is not in the approved set "
            f"{approved_set}. Approve it explicitly via "
            f"RAG_APPROVED_EMBEDDINGS_PROVIDERS; rag_api refuses to start with an "
            f"unapproved embeddings provider. OpenAI embeddings are not authorized "
            f"for client or second-tenant documents."
        )
    return provider


EMBEDDINGS_PROVIDER = resolve_embeddings_provider(os.getenv("EMBEDDINGS_PROVIDER"))

if EMBEDDINGS_PROVIDER == EmbeddingsProvider.OPENAI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-small")
    # 1000 is the default chunk size for OpenAI, but this causes API rate limits to be hit
    EMBEDDINGS_CHUNK_SIZE = get_env_variable("EMBEDDINGS_CHUNK_SIZE", 200)
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.AZURE:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-small")
    # 2048 is the default (and maximum) chunk size for Azure, but this often causes unexpected 429 errors
    EMBEDDINGS_CHUNK_SIZE = get_env_variable("EMBEDDINGS_CHUNK_SIZE", 200)
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.HUGGINGFACE:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.HUGGINGFACETEI:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "http://huggingfacetei:3000"
    )
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.GOOGLE_VERTEXAI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "gemini-embedding-001")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.OLLAMA:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "nomic-embed-text")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.GOOGLE_GENAI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "gemini-embedding-001")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.BEDROCK:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "amazon.titan-embed-text-v1"
    )
else:
    raise ValueError(f"Unsupported embeddings provider: {EMBEDDINGS_PROVIDER}")

embeddings = init_embeddings(EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL)

logger.info(f"Initialized embeddings of type: {type(embeddings)}")

## Embedding cache
# Caches embedding vectors in Redis so repeated text (query embedding in the
# endpoints and document embedding during ingestion) skips the embedding-provider
# round trip. Local and production differ only in connection settings:
# REDIS_HOST points at the local Redis container by default and at the AWS
# ElastiCache endpoint in production. The cache wraps the Embeddings instance, so
# it applies everywhere embeddings are produced. Keys are namespaced by the
# embeddings model so switching models never returns stale vectors.
from app.services.cache import RedisCache, CachingEmbeddings

REDIS_HOST = get_env_variable("REDIS_HOST", "localhost")
REDIS_PORT = int(get_env_variable("REDIS_PORT", "6379"))
# Time-to-live in seconds for cached embeddings; 0 disables expiry.
REDIS_TTL = int(get_env_variable("REDIS_TTL", "0"))
REDIS_KEY_PREFIX = get_env_variable("REDIS_KEY_PREFIX", "rag:emb:")
REDIS_SSL = get_env_variable("REDIS_SSL", "False").lower() in (
    "true",
    "1",
    "yes",
    "y",
    "t",
)
REDIS_PASSWORD = get_env_variable("REDIS_PASSWORD", None)

embedding_cache = RedisCache(
    host=REDIS_HOST,
    port=REDIS_PORT,
    ttl=REDIS_TTL,
    key_prefix=REDIS_KEY_PREFIX,
    ssl=REDIS_SSL,
    password=REDIS_PASSWORD,
)
embeddings = CachingEmbeddings(
    embeddings, embedding_cache, namespace=EMBEDDINGS_MODEL
)

SUM_UP_KNOWLEDGE_FILES = get_env_variable("SUM_UP_KNOWLEDGE_FILES", "False").lower() in (
    "true",
    "1",
    "yes",
    "y",
    "t",
)

# FILES-01 F04 (worker resource protection): the summarizer runs an LLM call
# (`summarize_files` -> `.invoke`) inside a thread-pool worker. Unbounded, a slow or hung
# model call holds one of the pool's (max 8) threads and blocks its request indefinitely;
# enough of them wedge the service. This bounds the summary REQUEST. Default 60 s, matching
# the OCR time budget's derivation from Core's 120 s client timeout (half the window).
# Note the limit this cannot cross: it frees the awaiting request, not the worker thread --
# `wait_for` cancels the future, the thread runs to completion (same as the OCR path). A
# request timeout on the LLM CLIENT is the operator's deeper control and is not set here,
# because it would change behaviour for every LLM consumer, not just the summarizer.
SUMMARY_TIMEOUT_SECONDS = float(get_env_variable("SUMMARY_TIMEOUT_SECONDS", "60"))

# Vector store
if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
    vector_store = get_vector_store(
        connection_string=CONNECTION_STRING,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="async",
    )
elif VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
    # Backward compatability check
    if MONGO_VECTOR_COLLECTION:
        logger.info(
            f"DEPRECATED: Please remove env var MONGO_VECTOR_COLLECTION and instead use COLLECTION_NAME and ATLAS_SEARCH_INDEX. You can set both as same, but not neccessary. See README for more information."
        )
        ATLAS_SEARCH_INDEX = MONGO_VECTOR_COLLECTION
        COLLECTION_NAME = MONGO_VECTOR_COLLECTION
    vector_store = get_vector_store(
        connection_string=ATLAS_MONGO_DB_URI,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="atlas-mongo",
        search_index=ATLAS_SEARCH_INDEX,
    )
else:
    raise ValueError(f"Unsupported vector store type: {VECTOR_DB_TYPE}")

retriever = vector_store.as_retriever()

## LLM

LLM_PROVIDER = get_env_variable("LLM_PROVIDER", "bedrock")
LLM_MODEL = get_env_variable(
    "LLM_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
LLM_TEMPERATURE = float(get_env_variable("LLM_TEMPERATURE", "0"))


def init_llm(provider, model, temperature):
    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        session_kwargs = {
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "region_name": AWS_DEFAULT_REGION,
        }

        if AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN

        session = boto3.Session(**session_kwargs)
        return ChatBedrockConverse(
            client=session.client("bedrock-runtime"),
            model=model,
            temperature=temperature,
            region_name=AWS_DEFAULT_REGION,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=OLLAMA_BASE_URL,
        )
    else:
        return None


llm = init_llm(LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE)

if llm:
    logger.info(f"Initialized LLM: provider={LLM_PROVIDER}, model={LLM_MODEL}")
else:
    logger.info("No LLM provider configured (LLM_PROVIDER not set)")

known_source_ext = [
    "go",
    "py",
    "java",
    "sh",
    "bat",
    "ps1",
    "cmd",
    "js",
    "ts",
    "css",
    "cpp",
    "hpp",
    "h",
    "c",
    "cs",
    "sql",
    "log",
    "ini",
    "pl",
    "pm",
    "r",
    "dart",
    "dockerfile",
    "env",
    "php",
    "hs",
    "hsc",
    "lua",
    "nginxconf",
    "conf",
    "m",
    "mm",
    "plsql",
    "perl",
    "rb",
    "rs",
    "db2",
    "scala",
    "bash",
    "swift",
    "vue",
    "svelte",
    "yml",
    "yaml",
    "eml",
    "ex",
    "exs",
    "erl",
    "tsx",
    "jsx",
    "lhs",
]
