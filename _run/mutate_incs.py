"""Fabricate a completeness claim INSIDE the real retrieval path, or exit non-zero.

This is the mutation the #88 suite structurally could not run: it stubbed
`_retrieve_documents`, so a fabrication inside that function was invisible to it. Here the
function is real, so injecting the claim there is exactly the defect the new file must catch.

NOTE ON THE FIRST VERSION, kept because it is the interesting part: it stamped
`d.metadata` directly and the suite stayed green. That was not the suite failing to see the
fabrication -- `_retrieve_documents` returns `(Document, score)` PAIRS, so `getattr(tuple,
"metadata")` stamped nothing and the mutation never ran. A probe that never ran reads exactly
like a disproved hypothesis. The wrapper now unwraps the pair, and the run below is what
actually tests the guard.

Runs INSIDE the test container: there is no python on the host shell.
"""
import io
import sys

PATH = "/src/app/routes/document_routes.py"

WRAPPER = '''

async def _retrieve_documents(*a, **kw):  # MUTATION: fabricate
    docs = await _retrieve_documents_real(*a, **kw)
    stamped = 0
    for item in docs:
        d = item[0] if isinstance(item, (tuple, list)) and item else item
        md = getattr(d, "metadata", None)
        if isinstance(md, dict):
            md["index_status"] = "indexed"
            stamped += 1
    print("MUTATION fabricate: stamped %d document(s)" % stamped)
    return docs

'''


def main():
    if sys.argv[1] != "fabricate":
        print("unknown mutation %r" % sys.argv[1])
        return 2

    s = io.open(PATH, encoding="utf-8", newline="").read()

    anchor = "async def _retrieve_documents("
    if s.count(anchor) != 1:
        print("ANCHOR NOT UNIQUE: found %d" % s.count(anchor))
        return 3
    marker = '\n@router.post("/query"'
    if s.count(marker) != 1:
        print("WRAPPER INSERTION POINT NOT UNIQUE: found %d" % s.count(marker))
        return 3

    s = s.replace(anchor, "async def _retrieve_documents_real(", 1)
    s = s.replace(marker, WRAPPER + marker, 1)
    io.open(PATH, "w", encoding="utf-8", newline="").write(s)
    print("MUTATION fabricate APPLIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
