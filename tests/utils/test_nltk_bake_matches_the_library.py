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

The pinned version is read from requirements.txt rather than hardcoded here, so bumping
the pin and bumping the installed library stay one action instead of two.
"""
import ast
import re
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

DOCKERFILES = ("Dockerfile", "Dockerfile.lite")
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"


def _pinned_unstructured() -> str:
    """The pin, read from requirements.txt -- never hardcoded in this file."""
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^unstructured==(\S+)", line.strip())
        if m:
            return m.group(1)
    pytest.fail("no `unstructured==` pin found in requirements.txt")


def _requested_resources() -> set:
    """The nltk resource names the INSTALLED tokenizer module asks for.

    Parsed from its source with `ast`, not imported: importing it runs
    `download_nltk_packages()` at module scope when AUTO_DOWNLOAD_NLTK is unset, which
    would reach nltk.org from a test. Collects both the `nltk.download("X")` arguments and
    the `check_for_nltk_package(package_name="Y")` keywords, because a future version could
    check one set and download another and this guard should see both.
    """
    import unstructured.nlp.tokenize as tok

    tree = ast.parse(Path(tok.__file__).read_text(encoding="utf-8"))
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

    found = _requested_resources()
    assert found, (
        "parsed NO resource names out of unstructured/nlp/tokenize.py. The module's shape "
        "changed, so this guard is reading nothing and would pass vacuously. Re-derive it "
        "before trusting any green in this file."
    )
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


def test_the_offline_guard_is_still_in_the_workflow(resources):
    """AUTO_DOWNLOAD_NLTK=false is what makes a missing resource FAIL rather than silently
    reach nltk.org. Without it every assertion above still passes and the runtime
    dependency quietly returns -- the greens would be measuring nothing."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r'AUTO_DOWNLOAD_NLTK:\s*"false"', text), (
        "AUTO_DOWNLOAD_NLTK=false is no longer set in the workflow. The bake and prefetch "
        "assertions above would still pass while the test phase silently downloaded from "
        "nltk.org on demand, which is the dependency this whole card removes.")
