"""Which build is answering — the one question a running service could not be asked.

WHY THIS EXISTS. An ECS task definition names a source revision, but the thing that serves
traffic is an IMAGE. Nothing in this repo used to connect the two: `deploy/push.sh` builds the
operator's working tree rather than a git ref, neither Dockerfile carried a label, the image tag
defaults to `latest`, and the workflows that build images push *dev* images to ghcr, not the
production registry. So a deployed image's source revision existed only in the transient stdout of
one operator run, and `/health` answered `{"status": "ok"}` — no version, nothing to compare.

WHAT THIS DOES AND DOES NOT ESTABLISH.

- It stamps images built AFTER it ships. **It cannot identify an image built before it existed**
  — task definition 24 included. Nothing here should ever be read as evidence about that image.
- A stamp is only as honest as the tree it was built from, so the *dirtiness* of that tree is
  carried too: an image built from uncommitted edits says so, rather than naming a revision whose
  contents it does not actually contain.
- Absence is reported as `unknown`, never guessed. A developer's `docker build .` with no
  arguments still works and simply says `unknown` — which is the truth about it.

WHERE THE VALUES COME FROM. `deploy/push.sh` derives them from the build tree and passes them as
build args; the Dockerfiles turn them into `ENV` (read here) and OCI `LABEL`s (readable on the
image without running it). The two paths are deliberate: the label answers "what is this image?"
for anyone who can read the registry, and the header answers it for anyone who can only reach the
wire.

These values are read from the environment at CALL time rather than captured at import, so a test
can set them without reloading the module and an operator can override them on a task definition
without rebuilding.
"""

import os

#: Not configurable: this is an identity, not a setting. A service that can be renamed by
#: environment cannot answer "did this reply come from rag_api at all".
SERVICE_NAME = "rag_api"

UNKNOWN = "unknown"

#: Header names are part of the contract the moment a consumer reads them (Core asked for exactly
#: this so an rag_api reply can be told from an edge/proxy reply without sniffing the body shape).
HEADER_SERVICE = "X-Service-Name"
HEADER_REVISION = "X-Build-Revision"
HEADER_TREE = "X-Build-Tree"

#: A full git sha is not needed to identify a build and a short one is easier to compare by eye.
_REVISION_CHARS = 12

_DIRTY_TRUE = {"1", "true", "yes", "dirty"}
_DIRTY_FALSE = {"0", "false", "no", "clean"}


def build_revision() -> str:
    """The source revision this image was built from, or `unknown`.

    Truncated to 12 characters because that is what people compare; the full value stays on the
    image label for anyone who needs to resolve it exactly.
    """
    value = (os.getenv("BUILD_REVISION") or "").strip()
    if not value or value == UNKNOWN:
        return UNKNOWN
    return value[:_REVISION_CHARS]


def build_tree() -> str:
    """`clean`, `dirty`, or `unknown` — the state of the tree the image was built from.

    Kept SEPARATE from the revision rather than folded into it. A revision with an unknown tree
    state and a revision with a clean one are different claims, and a single string would have to
    either drop that difference or encode it somewhere a reader has to decode.
    """
    value = (os.getenv("BUILD_DIRTY") or "").strip().lower()
    if value in _DIRTY_TRUE:
        return "dirty"
    if value in _DIRTY_FALSE:
        return "clean"
    return UNKNOWN


def build_time() -> str:
    """When the image was built (UTC, ISO-8601), or `unknown`. Not on the wire — `/health` and the
    image label only, because it identifies nothing by itself."""
    value = (os.getenv("BUILD_TIME") or "").strip()
    return value or UNKNOWN


def build_summary() -> dict:
    """The shape `/health` reports. Additive: no existing key changes."""
    return {
        "service": SERVICE_NAME,
        "revision": build_revision(),
        "tree": build_tree(),
        "built_at": build_time(),
    }


def stamp(headers) -> None:
    """Write the identity headers onto a response.

    Applied to EVERY response, including refusals and route misses, because those are precisely
    the ones a caller cannot otherwise attribute: a 404 from this service and a 404 from an edge
    that never reached it look identical to a client, and that ambiguity has already cost a
    consuming lane a misdiagnosis.
    """
    headers[HEADER_SERVICE] = SERVICE_NAME
    headers[HEADER_REVISION] = build_revision()
    headers[HEADER_TREE] = build_tree()


async def build_stamp_middleware(request, call_next):
    """Outermost middleware, so the stamp survives a refusal from any layer below it.

    LIMIT, stated rather than implied: a response produced ABOVE the user middleware stack — an
    unhandled exception rendered by Starlette's ServerErrorMiddleware — is not stamped. That case
    is a crash, and an unstamped 500 is itself a signal worth keeping distinguishable from a
    refusal this service chose to make.
    """
    response = await call_next(request)
    stamp(response.headers)
    return response
