"""The NLTK bake is checked against what the INSTALLED library actually asks for.

F-NLTK-CONTROL-DERIVED-FROM-THE-LIBRARY. Follow-up to #87, which fixed the bake; this is
the control that keeps it fixed.

WHY #87's CONTROL IS NOT ENOUGH, which is the whole point of this file
----------------------------------------------------------------------
#87 pins the two resource NAMES -- `punkt_tab` and `averaged_perceptron_tagger_eng` --
with a three-arm offline/online control. That tests the names it was written with. nltk
3.9.x already renamed the POS tagger once (`averaged_perceptron_tagger` ->
`..._eng`), which is the defect #87 exists to repair. **The next rename re-opens exactly
the same hole and that control still passes**, because it would be asserting the old name
against a bake that also carries the old name: both sides wrong, agreeing.

This file derives the expected names from the INSTALLED LIBRARY -- the thing that
changes -- and compares them against every place the estate writes them down. A rename
upstream then reddens automatically instead of shipping quietly.

MEASURED BASIS (2026-09-23, unstructured==0.18.32):
  * `unstructured/nlp/tokenize.py` is the ONLY module in the package that imports nltk;
  * it requests exactly two resources, and the availability check is an OR over both, so
    ONE stale name makes it fetch BOTH -- which is why the original defect was total
    rather than partial, and why a tagger-only fix fully closed it;
  * `langchain_text_splitters/html.py:717` wants `stopwords`, which is NOT baked --
    latent, not live: rag_api imports only `RecursiveCharacterTextSplitter`, and that
    call sits in `HTMLSemanticPreservingSplitter.__init__` behind a flag. If that ever
    changes this guard will not catch it, because it checks the tokenizer module only.

WHY IT FAILS LOUDLY ON A VERSION CHANGE RATHER THAN SKIPPING
-------------------------------------------------------------
If the installed `unstructured` is not the pinned one, this test FAILS. It does not skip.
A skip would turn a version bump into a silent no-op, which is precisely the inert-guard
shape this card exists to remove -- the guard would go quiet at the exact moment the thing
it guards became most likely to have moved.

The pinned version is read from the requirements files rather than hardcoded here, so bumping
the pin and bumping the installed library stay one action instead of two.

THE ONE SKIP IN THIS FILE, AND WHY IT IS NOT THE SKIP FORBIDDEN ABOVE
---------------------------------------------------------------------
Two tests READ `.github/workflows/ci.yml`. `.dockerignore` excludes `.github/` from the shipped
image on purpose, so inside the CI job that runs this suite IN the shipped runtime there is
nothing to read -- found by that job (run 35859305310: FileNotFoundError at
/app/.github/workflows/ci.yml) after the same tests passed on the bare runner and in a
mounted-source container. Their subject is the REPOSITORY; the image is not the repository.

A version mismatch is skipped-never because it makes the guard MEANINGLESS. An absent workflow
inside the image means the SUBJECT is absent from the artifact by design. But "absent" is also
what a DELETED or MOVED workflow looks like, and that is the inert-guard shape this file exists
to remove -- so the skip is paired with `test_the_workflow_is_present_in_a_checkout`, which
ASSERTS the file wherever a checkout is identifiable. Same shape as `needs_deploy_dir` in
tests/test_build_provenance.py, for the same reason.
"""
import ast
import importlib.util
import re
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

DOCKERFILES = ("Dockerfile", "Dockerfile.lite")
#: Two files because there are two images: `Dockerfile` installs the first, `Dockerfile.lite`
#: (production) installs the second, and this suite runs INSIDE each of them on CI. Reading only
#: `requirements.txt` -- which the first version of this file did -- compares the lite image's
#: installed library against the OTHER image's pin.
REQUIREMENTS = ("requirements.txt", "requirements.lite.txt")
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"

#: Narrow on purpose -- see the module docstring. The file's absence is the ONLY condition, and
#: the paired checkout test below is what stops that condition from meaning "deleted".
needs_the_workflow = pytest.mark.skipif(
    not WORKFLOW.is_file(),
    reason=".github/ is excluded from the shipped image by .dockerignore, so a test that READS "
    "the workflow has nothing to read here; its subject is the repository, not the runtime",
)

#: How many nltk resources unstructured==0.18.32 requests. A DELIBERATE INDEPENDENT anchor,
#: not derived from the parse it checks -- deriving it from the same parse would make it
#: agree with itself and catch nothing. Bump it only after deciding what to do about the
#: new resource, which is the decision this number exists to force.
EXPECTED_RESOURCE_COUNT = 2


def _pinned_unstructured() -> str:
    """The pin, read from BOTH requirements files -- never hardcoded in this file.

    Both Dockerfiles bake the SAME resource names, which is only right while both install the
    SAME library. Two files pinning two versions is therefore a defect in its own right -- the
    bake for one image may be inert -- and it fails here, loudly, rather than being averaged
    away by whichever file happened to be read.
    """
    pins = {}
    for name in REQUIREMENTS:
        for line in (REPO / name).read_text(encoding="utf-8").splitlines():
            m = re.match(r"^unstructured==(\S+)", line.strip())
            if m:
                pins[name] = m.group(1)
                break
        else:
            pytest.fail("no `unstructured==` pin found in %s" % name)
    assert len(set(pins.values())) == 1, (
        "the two images pin DIFFERENT unstructured versions: %r. Dockerfile and Dockerfile.lite "
        "bake the same NLTK resource names, which is only correct while both install the same "
        "library; whichever image runs the other version may have an inert bake." % pins
    )
    return pins[REQUIREMENTS[0]]


def _requested_resources() -> set:
    """The nltk resource names the INSTALLED tokenizer module asks for.

    Parsed from its source with `ast`, not imported: MEASURED on 0.18.32, tokenize.py:47 is
    `if os.getenv("AUTO_DOWNLOAD_NLTK", "True").lower() == "true":` at MODULE SCOPE, so an
    import runs the download routine wherever a resource is missing and that variable is
    unset -- the guard that exists to keep nltk.org out of the runtime would reach it itself.

    LOCATED with `importlib.util.find_spec`, which imports the parent package
    `unstructured.nlp` (an empty `__init__`) and NOT the module. The first version of this
    file did `import unstructured.nlp.tokenize` to find the path -- running the module scope
    this docstring says must not run -- and nothing showed it, because every environment it
    ran in already carried the bake. The fixture's self-check pins it now.

    Collects both the `nltk.download("X")` arguments and the
    `check_for_nltk_package(package_name="Y")` keywords, because a future version could
    check one set and download another and this guard should see both.
    """
    spec = importlib.util.find_spec("unstructured.nlp.tokenize")
    assert spec is not None and spec.origin, "unstructured.nlp.tokenize is not installed"
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name == "download":
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    found.add(a.value)
        if name == "check_for_nltk_package":
            for kw in node.keywords:
                if kw.arg == "package_name" and isinstance(kw.value, ast.Constant):
                    found.add(kw.value.value)
    return found


@pytest.fixture(scope="module")
def resources():
    """Fail LOUDLY on a version mismatch. Never skip."""
    pinned, actual = _pinned_unstructured(), installed_version("unstructured")
    assert actual == pinned, (
        "installed unstructured==%s but requirements.txt pins %s. This guard derives the "
        "NLTK resource names from the INSTALLED library, so it is only meaningful against "
        "the pinned one. It fails rather than skips deliberately: a skip would make a "
        "version bump silently disable the check, which is the inert-guard shape this "
        "card exists to remove." % (actual, pinned)
    )

    # SELF-CHECK: the guard must not import the module it reads. Decidable only when nothing
    # earlier in the session imported it (a full run may have, through any loader test); a
    # single-file run always decides it, and the printed line below says which happened.
    imported_before = "unstructured.nlp.tokenize" in sys.modules
    found = _requested_resources()
    if not imported_before:
        assert "unstructured.nlp.tokenize" not in sys.modules, (
            "this guard IMPORTED unstructured.nlp.tokenize while locating it. Its module scope "
            "runs the nltk download routine unless AUTO_DOWNLOAD_NLTK is false, so the guard "
            "that exists to keep nltk.org out of the runtime would itself reach it wherever a "
            "resource is missing. Locate the file with importlib.util.find_spec; never import "
            "the module."
        )
    assert len(found) == EXPECTED_RESOURCE_COUNT, (
        "parsed %d resource name(s) out of unstructured/nlp/tokenize.py, expected exactly "
        "%d: %r.\n"
        "Non-empty is NOT sufficient here. If the module's shape changes so the parse picks "
        "up ONE of two names, every comparison below still passes -- on a half-read set. "
        "This count is a deliberate independent anchor: if upstream starts requesting a "
        "different NUMBER of resources, somebody decides whether to bake the new one, "
        "rather than this guard quietly checking only what it happened to find."
        % (len(found), EXPECTED_RESOURCE_COUNT, sorted(found))
    )

    # POSITIVE EVIDENCE THAT THIS RAN. Every compared set is printed on every run, so a
    # passing run is distinguishable from one that never executed. That is this lane's own
    # rule -- any control that can be satisfied by its own absence must emit evidence it
    # ran -- and the first version of this file did not follow it.
    #
    # print(), not warnings.warn(): a warning would show under `-q` on a passing run, which
    # is strictly better for visibility, but it would also move the suite's warning count,
    # and several open cards reconcile their acceptance against that number. Trading a
    # shared signal for a local one is the wrong trade. `pytest -s` shows these, and pytest
    # shows them automatically on any failure.
    print("\n[nltk-guard] pinned/installed unstructured : %s" % actual)
    print("[nltk-guard] requested by the library      : %s" % sorted(found))
    print("[nltk-guard] tokenizer module imported here : %s"
          % ("not decidable -- imported earlier in the session" if imported_before else "no"))
    return found


def test_both_dockerfiles_bake_exactly_what_the_library_requests(resources):
    """The bake is what production runs: deploy/push.sh builds Dockerfile.lite."""
    for name in DOCKERFILES:
        text = (REPO / name).read_text(encoding="utf-8")
        bake = [l for l in text.splitlines() if "nltk.downloader" in l]
        assert bake, "%s has no nltk.downloader line at all" % name
        baked = " ".join(bake)
        for res in sorted(resources):
            assert res in baked, (
                "%s does not bake %r, which unstructured==%s asks for at runtime. An "
                "un-baked resource is fetched from nltk.org on the first ingest -- the "
                "exact runtime dependency #87 removed. Bake line: %s"
                % (name, res, installed_version("unstructured"), baked.strip()))


@pytest.mark.skipif(
    not (REPO / ".git").exists(),
    # EXISTS, not is_dir(): in a git WORKTREE `.git` is a FILE pointing at the real directory,
    # and every lane in this program checks out into worktrees (tests/test_build_provenance.py
    # caught the is_dir() version by running it).
    reason="not a checkout -- cannot distinguish 'not shipped' from 'deleted' without git metadata",
)
def test_the_workflow_is_present_in_a_checkout():
    """The other half of `needs_the_workflow`. In any tree identifiable as a checkout the
    workflow must EXIST -- otherwise moving or deleting it would silently turn the two guards
    below into two skips, in the bare-runner job where they are the only thing checking the
    prefetch and the cache key."""
    assert WORKFLOW.is_file(), (
        "%s is missing from a checkout: the prefetch/cache-key and offline-guard tests below "
        "would SKIP rather than FAIL, which is the inert-guard shape this file exists to remove"
        % WORKFLOW.relative_to(REPO)
    )


@needs_the_workflow
def test_the_ci_prefetch_and_cache_key_name_the_same_resources(resources):
    """CI prefetches and caches BY NAME. A rename that reached the Dockerfiles but not the
    cache key would restore a stale cache over a correct prefetch."""
    text = WORKFLOW.read_text(encoding="utf-8")

    prefetch = re.search(r'for pkg in \(([^)]*)\)', text)
    assert prefetch, "no prefetch loop found in the workflow; this guard is reading nothing"
    listed = set(re.findall(r'"([^"]+)"', prefetch.group(1)))
    assert listed == resources, (
        "the CI prefetch loop fetches %r but unstructured==%s requests %r. The test phase "
        "runs with AUTO_DOWNLOAD_NLTK=false, so anything missing here fails mid-test "
        "instead of in the named prefetch step."
        % (sorted(listed), installed_version("unstructured"), sorted(resources)))

    key = re.search(r'key:\s*\$\{\{\s*runner\.os\s*\}\}-nltk-(\S+)', text)
    assert key, "no NLTK cache key found in the workflow"
    for res in sorted(resources):
        assert res in key.group(1), (
            "the NLTK cache key %r does not name %r. The key is what makes a rename "
            "invalidate the cache; without it a stale cache is restored over a corrected "
            "prefetch and the run passes for the wrong reason." % (key.group(1), res))


@needs_the_workflow
def test_the_offline_guard_is_still_in_the_workflow(resources):
    """AUTO_DOWNLOAD_NLTK=false is what makes a missing resource FAIL rather than silently
    reach nltk.org. Without it every assertion above still passes and the runtime
    dependency quietly returns -- the greens would be measuring nothing."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r'AUTO_DOWNLOAD_NLTK:\s*"false"', text), (
        "AUTO_DOWNLOAD_NLTK=false is no longer set in the workflow. The bake and prefetch "
        "assertions above would still pass while the test phase silently downloaded from "
        "nltk.org on demand, which is the dependency this whole card removes.")
