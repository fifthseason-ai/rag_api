"""Mutate the REAL retrieval path, or exit non-zero. Two mutations, one per direction.

This is the mutation family the #88 suite structurally could not run: it stubbed
`_retrieve_documents`, so anything done inside that function was invisible to it. Here the
function is real, so the wrapper below is exactly the defect the new file must catch.

NOTE ON THE FIRST VERSION, kept because it is the instructive part: it stamped `d.metadata`
directly and the suite stayed green. That was NOT the suite failing to see the fabrication --
`_retrieve_documents` returns `(Document, score)` PAIRS, so `getattr(tuple, "metadata")`
matched nothing and the mutation stamped zero documents. A probe that never ran reads exactly
like a disproved hypothesis.

**Hence the print.** Every wrapper reports how many documents it touched, a number only a live
execution can produce. The general rule: any control that can be satisfied by its own absence
must emit positive evidence that it ran. Ask what the control's output would be IF IT NEVER
RAN; if that matches its passing output, it is not yet a control.

TWO DIRECTIONS, RUN SEPARATELY. `test_the_retrieved_metadata_is_exactly_what_was_stored` holds
two assertions -- nothing invented, nothing dropped. Two assertions only ever exercised
together cannot be shown to be two, so each has its own mutation and each must redden alone.

Runs INSIDE the test container: there is no python on the host shell.
"""
import io
import sys

PATH = "/src/app/routes/document_routes.py"

#: Injected with the original function's name so every call site reaches it unchanged.
WRAPPERS = {
    # DIRECTION 1 -- INVENT. A completeness claim the store never held. Must redden the
    # `invented` assertion and leave `dropped` satisfied.
    "fabricate": '''

async def _retrieve_documents(*a, **kw):  # MUTATION: fabricate
    docs = await _retrieve_documents_real(*a, **kw)
    touched = 0
    for item in docs:
        d = item[0] if isinstance(item, (tuple, list)) and item else item
        md = getattr(d, "metadata", None)
        if isinstance(md, dict):
            md["index_status"] = "indexed"
            touched += 1
    print("MUTATION fabricate: stamped %d document(s)" % touched)
    return docs

''',
    # DIRECTION 2 -- DROP. Silently lose stored provenance. Must redden the `dropped`
    # assertion and leave `invented` satisfied. Losing provenance without saying so is the
    # other half of the same defect: a consumer cannot tell an absent locator from a
    # discarded one.
    "drop": '''

async def _retrieve_documents(*a, **kw):  # MUTATION: drop
    docs = await _retrieve_documents_real(*a, **kw)
    touched = 0
    for item in docs:
        d = item[0] if isinstance(item, (tuple, list)) and item else item
        md = getattr(d, "metadata", None)
        if isinstance(md, dict) and "ingest_id" in md:
            del md["ingest_id"]
            touched += 1
    print("MUTATION drop: removed ingest_id from %d document(s)" % touched)
    return docs

''',
}


def main():
    name = sys.argv[1]
    if name not in WRAPPERS:
        print("unknown mutation %r (have: %s)" % (name, ", ".join(sorted(WRAPPERS))))
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
    s = s.replace(marker, WRAPPERS[name] + marker, 1)
    io.open(PATH, "w", encoding="utf-8", newline="").write(s)
    print("MUTATION %s APPLIED" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
