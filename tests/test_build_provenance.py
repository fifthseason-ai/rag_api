"""Build provenance: does a running image say which build it is, on every answer it gives?

These tests exist because the deployed question -- "which image is serving this?" -- was
unanswerable from the wire, and the answer is only useful if it survives the responses nobody
plans for: a refusal, a validation error, a route that does not exist. The happy-path 200 is the
least interesting case here.

Scope note kept deliberately: these prove the SERVICE reports its stamp. They cannot prove any
particular deployed image carries one, and nothing here says anything about an image built before
this code existed.
"""

import datetime
import os

import jwt
import pytest
from fastapi.testclient import TestClient

from app import build_info
from app.routes import document_routes
from main import app

client = TestClient(app)

STAMPED = {
    "BUILD_REVISION": "36d4fb6da44df2de12617ce83fa5e29bc50b5acc",
    "BUILD_DIRTY": "false",
    "BUILD_TIME": "2026-09-18T12:00:00Z",
}


@pytest.fixture
def unstamped(monkeypatch):
    for key in ("BUILD_REVISION", "BUILD_DIRTY", "BUILD_TIME"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def stamped(monkeypatch):
    for key, value in STAMPED.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def auth_configured(monkeypatch):
    """The service fails closed without JWT_SECRET (D-KSPT-1), answering 500 to every protected
    request. Tests that want to observe a 401 or a 404 must configure it first, or they are
    measuring the fail-closed path by accident."""
    monkeypatch.setenv("JWT_SECRET", "testsecret")


@pytest.fixture
def token(auth_configured):
    return jwt.encode(
        {
            "id": "testuser",
            "tid": "tenantA",
            "ent": ["testuser"],
            "act": ["read"],
            "exp": datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(hours=1),
        },
        "testsecret",
        algorithm="HS256",
    )


# --- what the values MEAN -------------------------------------------------------------


def test_unstamped_build_says_unknown_rather_than_guessing(unstamped):
    """A developer image is not a mystery to be papered over: it reports what is true of it."""
    assert build_info.build_revision() == "unknown"
    assert build_info.build_tree() == "unknown"
    assert build_info.build_time() == "unknown"


def test_revision_is_shortened_but_not_invented(stamped):
    assert build_info.build_revision() == "36d4fb6da44d"
    assert STAMPED["BUILD_REVISION"].startswith(build_info.build_revision())


def test_tree_state_is_three_valued(monkeypatch):
    """`clean`, `dirty` and `unknown` are three different claims and must not collapse into two.

    An image built from uncommitted edits cannot be rebuilt from the revision it names; an image
    whose tree state was never recorded might be either. Reporting the second as `clean` would be
    the fabrication this whole change exists to prevent.
    """
    monkeypatch.setenv("BUILD_DIRTY", "true")
    assert build_info.build_tree() == "dirty"
    monkeypatch.setenv("BUILD_DIRTY", "false")
    assert build_info.build_tree() == "clean"
    monkeypatch.setenv("BUILD_DIRTY", "")
    assert build_info.build_tree() == "unknown"
    monkeypatch.setenv("BUILD_DIRTY", "possibly")
    assert build_info.build_tree() == "unknown"


def test_dirty_is_not_swallowed_by_the_revision(stamped, monkeypatch):
    """The revision stays the revision when the tree was dirty: two facts, two fields."""
    monkeypatch.setenv("BUILD_DIRTY", "true")
    assert build_info.build_revision() == "36d4fb6da44d"
    assert build_info.build_tree() == "dirty"


# --- the answers that are hardest to attribute ----------------------------------------


def _headers_of(response):
    return {k.lower(): v for k, v in response.headers.items()}


def test_every_answer_carries_the_stamp(stamped, auth_configured, token, monkeypatch):
    """THE LOAD-BEARING ONE. A 404 from this service and a 404 from an edge that never reached it
    are indistinguishable to a caller -- that ambiguity has already cost a consuming lane a
    misdiagnosis. The stamp is what separates them, so it has to be on the refusals and on the
    route misses, not only on the successes.

    Each case is a DIFFERENT layer producing the response: a public route, the security
    middleware, the router, and FastAPI's own not-found. They are listed together because the
    ordering of the middleware stack is the only thing that makes all four carry the stamp, and
    that ordering is easy to break without noticing.
    """
    # `/health` is the public case, and its 200 has to be EARNED rather than inherited. Under
    # pytest there is no reachable store, so the real health check fails -- and this assertion was
    # green anyway, because the route answered a failed check with HTTP 200 (a returned
    # (body, status) tuple never sends its status). The test was passing on the defect it could not
    # see. Pinned to a HEALTHY check so this case means "a public route stamps its answer" and
    # nothing else; the DOWN behaviour is `tests/test_health_contract.py`'s subject, not this one's.
    async def healthy():
        return True

    monkeypatch.setattr(document_routes, "is_health_ok", healthy)

    cases = [
        ("a public route", 200, lambda: client.get("/health")),
        ("a refusal with no token", 401, lambda: client.get("/ids")),
        (
            "a refusal with a bad token",
            401,
            lambda: client.get("/ids", headers={"Authorization": "Bearer nonsense"}),
        ),
        (
            "an authorized caller on a path no route matches",
            404,
            lambda: client.post(
                "/query/an-entity/extra/segments",
                json={"query": "x"},
                headers={"Authorization": "Bearer %s" % token},
            ),
        ),
    ]
    for what, expected_status, call in cases:
        response = call()
        assert response.status_code == expected_status, what
        headers = _headers_of(response)
        assert headers["x-service-name"] == "rag_api", what
        assert headers["x-build-revision"] == "36d4fb6da44d", what
        assert headers["x-build-tree"] == "clean", what


def test_the_fail_closed_500_is_attributable_too(stamped, monkeypatch):
    """Without JWT_SECRET the service refuses every protected request with a 500. That is the one
    answer most likely to be mistaken for an infrastructure fault, so it carries the stamp as
    well -- a 500 that names the service is a configuration problem, an unstamped one is not this
    service at all (or is a crash above the middleware stack)."""
    monkeypatch.delenv("JWT_SECRET", raising=False)
    response = client.get("/ids")
    assert response.status_code == 500
    assert _headers_of(response)["x-service-name"] == "rag_api"


def test_an_unmatched_path_answers_the_frameworks_not_found(auth_configured, token):
    """MEASURED, not quoted from the framework docs. This body is the discriminator a consuming
    lane uses to tell "the app answered and no route matched" from "the request never reached the
    app", so it is pinned here rather than assumed -- and note it is only reachable WITH a valid
    token, because the security middleware runs before routing and answers 401 first.
    """
    response = client.post(
        "/query/an-entity/extra/segments",
        json={"query": "x"},
        headers={"Authorization": "Bearer %s" % token},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_an_unmatched_path_without_a_token_is_refused_before_routing(auth_configured):
    """The same path with no token is a 401, not a 404: authority is checked before the route
    table is consulted. Worth pinning because it means a 404 from this service always implies the
    caller WAS authenticated."""
    response = client.post("/query/an-entity/extra/segments", json={"query": "x"})
    assert response.status_code == 401


def test_stamp_is_present_but_honest_on_an_unstamped_build(unstamped):
    response = client.get("/health")
    headers = _headers_of(response)
    assert headers["x-service-name"] == "rag_api"
    assert headers["x-build-revision"] == "unknown"
    assert headers["x-build-tree"] == "unknown"


def test_health_reports_the_build_additively(stamped, monkeypatch):
    """`status` keeps its existing meaning; a caller reading only `status` is untouched.

    The database probe is forced healthy: this asserts the SHAPE of the build report, and letting
    it depend on a live database would make it a database test that fails for the wrong reason.
    """

    async def healthy():
        return True

    monkeypatch.setattr(document_routes, "is_health_ok", healthy)
    response = client.get("/health")
    body = response.json()
    assert body["status"] in ("ok", "UP")
    assert body["build"] == {
        "service": "rag_api",
        "revision": "36d4fb6da44d",
        "tree": "clean",
        "built_at": "2026-09-18T12:00:00Z",
    }


# --- the wiring, which is what makes any of the above true of a real image -------------
#
# Source-text assertions, labelled as such: they prove the build files SAY the right thing, not
# that a particular image was built correctly. They exist because every value above arrives
# through the Dockerfiles and `deploy/push.sh`, and a test suite that mocks that path away would
# pass just as happily against an image that carries no stamp at all.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.lite"])
def test_dockerfiles_accept_and_record_the_stamp(dockerfile):
    text = open(os.path.join(ROOT, dockerfile), encoding="utf-8").read()
    for arg in ("BUILD_REVISION", "BUILD_DIRTY", "BUILD_TIME"):
        assert "ARG %s=unknown" % arg in text, (
            "%s must DEFAULT the arg, or a plain `docker build .` breaks for developers"
            % dockerfile
        )
        # NOT `"ENV" in text and "${ARG}" in text`. That pair is satisfied without the stamp
        # existing at all -- the base images already carry `ENV SCARF_NO_ANALYTICS=true`, and
        # `${BUILD_REVISION}` also appears on the LABEL line. Deleting the whole ENV block from
        # both Dockerfiles left this file 15/15 green, and that block is the half of the PR that
        # puts the stamp ON THE WIRE. The assertion now names the assignment itself.
        assert "%s=${%s}" % (arg, arg) in text, (
            "%s must ENV the arg, or the build identity never reaches a response header"
            % dockerfile
        )
    assert 'org.opencontainers.image.revision="${BUILD_REVISION}"' in text, (
        "%s must label the image, because a deployed digest can only be resolved from the "
        "registry -- not from the wire" % dockerfile
    )
    assert 'ai.fifthseason.build.dirty="${BUILD_DIRTY}"' in text, dockerfile


def test_push_script_derives_and_passes_the_stamp_and_prints_the_pairing():
    text = open(os.path.join(ROOT, "deploy", "push.sh"), encoding="utf-8").read()
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain" in text, "a dirty build tree must be detected, not assumed away"
    for arg in ("BUILD_REVISION", "BUILD_DIRTY", "BUILD_TIME"):
        assert '--build-arg "%s=${%s}"' % (arg, arg) in text
    assert "RECORD THIS WITH THE DEPLOYMENT" in text, (
        "the digest and the revision are only connected by being written down together"
    )
    assert "image digest" in text and "source revision" in text


def test_push_script_refuses_a_production_build_from_a_dirty_tree():
    """A production image must be rebuildable from the revision it names, so a dirty tree is a
    REFUSAL here rather than a warning -- with one deliberate, explicit way through, because an
    absolute ban would be worked around by calling `docker build` directly, which loses the stamp
    altogether."""
    text = open(os.path.join(ROOT, "deploy", "push.sh"), encoding="utf-8").read()
    assert "REFUSED: the build tree has UNCOMMITTED CHANGES." in text
    assert "exit 1" in text, "the refusal must stop the deploy, not just print"
    assert "PUSH_ALLOW_DIRTY" in text, "an emergency needs a documented way through"


def test_dirty_refusal_actually_exits_and_the_override_actually_passes():
    """EXECUTED, not read. The script is run with its side effects stubbed out, in a dirty git
    repo, so the refusal and the override are observed rather than inferred from source text --
    a shell guard that is only grepped for is a guard nobody has ever seen fire.
    """
    import subprocess
    import tempfile

    NL = chr(10)
    QUOTE = chr(34)

    repo = tempfile.mkdtemp()
    stub_bin = os.path.join(repo, "stubs")
    os.makedirs(stub_bin)

    def write_stub(name, body_lines):
        path = os.path.join(stub_bin, name)
        with open(path, "w") as fh:
            fh.write(NL.join(["#!/bin/sh"] + body_lines + [""]))
        os.chmod(path, 0o755)

    # Everything the script reaches for is stubbed, so this test runs anywhere -- including
    # INSIDE the shipped runtime image, which has no git and no AWS credentials. `docker` echoes
    # its arguments so a build that should not have happened is visible in the output.
    # `cat >/dev/null` is not decoration. The script runs
    # `aws ecr get-login-password | docker login --password-stdin`, so a `docker` stub that exits
    # without draining stdin kills `aws` with SIGPIPE; `set -o pipefail` then aborts the script
    # with 141 and the run fails for a reason that has nothing to do with the guard under test.
    # It reproduced at roughly 3-4%, and the assertion it flaked was the NEGATIVE CONTROL -- the
    # one thing standing between this test and a script that refuses every build.
    for tool in ("aws", "docker"):
        write_stub(
            tool,
            ["cat >/dev/null 2>&1 || true",
             "echo STUB-%s %s$@%s" % (tool.upper(), QUOTE, QUOTE),
             "exit 0"],
        )
    # The `git` stub is what makes the tree dirty or clean, deterministically and without needing
    # a real repository: `porcelain` output is exactly the signal the script keys on.
    write_stub(
        "git",
        [
            'case "$*" in',
            '  "rev-parse HEAD") echo 1111111111111111111111111111111111111111 ;;',
            '  "rev-parse --git-dir") echo .git ;;',
            # STUB_DIRTY=fail is the case that was missing and that the script read as CLEAN:
            # a `git status` that FAILS prints nothing, exactly like a clean tree.
            '  "status --porcelain")',
            '    [ "${STUB_DIRTY:-1}" = "fail" ] && exit 128',
            '    [ "${STUB_DIRTY:-1}" = "1" ] && echo " M dirt.txt"',
            "    ;;",
            "esac",
            "exit 0",
        ],
    )

    os.makedirs(os.path.join(repo, "deploy"))
    with open(os.path.join(repo, "deploy", "push.sh"), "w", newline=NL) as fh:
        fh.write(open(os.path.join(ROOT, "deploy", "push.sh"), encoding="utf-8").read())
    with open(os.path.join(repo, "Dockerfile.lite"), "w") as fh:
        fh.write("FROM scratch" + NL)

    env = dict(os.environ)
    env["PATH"] = stub_bin + os.pathsep + env["PATH"]
    env["STUB_DIRTY"] = "1"

    refused = subprocess.run(
        ["bash", os.path.join(repo, "deploy", "push.sh")],
        cwd=repo, env=env, capture_output=True, text=True,
    )
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert "REFUSED" in refused.stderr
    assert "STUB-DOCKER build" not in refused.stdout, "it must refuse BEFORE building"

    env["PUSH_ALLOW_DIRTY"] = "1"
    allowed = subprocess.run(
        ["bash", os.path.join(repo, "deploy", "push.sh")],
        cwd=repo, env=env, capture_output=True, text=True,
    )
    assert "PUSH_ALLOW_DIRTY=1" in allowed.stdout
    assert "STUB-DOCKER build" in allowed.stdout, "the override must let the build proceed"
    assert "BUILD_DIRTY=true" in allowed.stdout, "and the image must still be stamped dirty"

    # THE NEGATIVE CONTROL. Without this, a script that refused EVERY build would pass both
    # assertions above and nobody would find out until a deploy failed.
    env.pop("PUSH_ALLOW_DIRTY")
    env["STUB_DIRTY"] = "0"
    clean = subprocess.run(
        ["bash", os.path.join(repo, "deploy", "push.sh")],
        cwd=repo, env=env, capture_output=True, text=True,
    )
    assert "REFUSED" not in clean.stderr, "a clean tree must not be refused"
    assert "STUB-DOCKER build" in clean.stdout
    assert "BUILD_DIRTY=false" in clean.stdout, "and it must be stamped clean, not merely allowed"

    # AN UNMEASURABLE TREE IS NOT A CLEAN ONE. `git status --porcelain` prints nothing when it
    # fails and nothing when the tree is clean, and the script used to read those as the same
    # thing: it built the image, stamped `ai.fifthseason.build.dirty=false`, served
    # `X-Build-Tree: clean`, and printed `build tree dirty=false` into the receipt -- a measured-
    # sounding claim nobody had measured. Reachable in ordinary life: `git rev-parse HEAD` does
    # not touch the index and `git status` does, so a stale `.git/index.lock` fails the second
    # while the first still answers.
    env["STUB_DIRTY"] = "fail"
    unmeasurable = subprocess.run(
        ["bash", os.path.join(repo, "deploy", "push.sh")],
        cwd=repo, env=env, capture_output=True, text=True,
    )
    assert unmeasurable.returncode == 1, unmeasurable.stdout + unmeasurable.stderr
    assert "could NOT BE MEASURED" in unmeasurable.stderr
    assert "STUB-DOCKER build" not in unmeasurable.stdout, "it must refuse BEFORE building"
    assert "BUILD_DIRTY=false" not in unmeasurable.stdout, (
        "a failed measurement must never be reported as a clean tree"
    )

    # ...and the same case goes through on the same deliberate hatch, still honestly stamped.
    env["PUSH_ALLOW_DIRTY"] = "1"
    forced = subprocess.run(
        ["bash", os.path.join(repo, "deploy", "push.sh")],
        cwd=repo, env=env, capture_output=True, text=True,
    )
    assert "STUB-DOCKER build" in forced.stdout, "the override must let the build proceed"
    assert "BUILD_DIRTY=unknown" in forced.stdout, (
        "and the image must be stamped unknown -- not clean, not dirty"
    )
