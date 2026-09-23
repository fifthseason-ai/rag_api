"""Apply ONE named mutation to document_routes.py, or exit non-zero.

A mutation that did not apply reads exactly like a passing guard, so every anchor is
asserted unique before anything is written and the script fails loudly otherwise.

Runs INSIDE the test container: there is no python on the host shell.
"""
import io
import sys

PATH = "/src/app/routes/document_routes.py"

MUTATIONS = {
    # LAYER 1 — the defensive route post-filter. The arm filters still run, so what
    # reddens under this is only what the post-filter alone protects.
    "postfilter": (
        "def _authorized_only(documents, entity_ids):",
        "def _authorized_only(documents, entity_ids):\n"
        "    return documents  # MUTATION: post-filter neutralised",
    ),
    # LAYER 2 — the entity predicate on the /query retrieval arm. A leak here originates
    # INSIDE the retrieval path, which is the case a stubbed retrieval cannot observe.
    "armfilter": (
        '            {"file_id": body.file_id, "user_id": user_filter},',
        '            {"file_id": body.file_id},  # MUTATION: entity predicate dropped',
    ),
}


def main():
    name = sys.argv[1]
    old, new = MUTATIONS[name]
    s = io.open(PATH, encoding="utf-8", newline="").read()
    n = s.count(old)
    if n != 1:
        print("ANCHOR NOT UNIQUE for %s: found %d" % (name, n))
        return 3
    io.open(PATH, "w", encoding="utf-8", newline="").write(s.replace(old, new, 1))
    print("MUTATION %s APPLIED" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ---------------------------------------------------------------------------
# REACHABILITY, and why this file is now the ONLY harness for this card
# ---------------------------------------------------------------------------
#
# `_run/xt-slot-run.sh` was deleted (it lived beside this file and mutated a DIFFERENT
# anchor: `{"user_id": entity_id}` in query_embeddings_by_entity_id). Two committed
# harnesses that disagree are worse than one that is wrong, because each reads as
# corroboration of the other -- and the receipt named neither.
#
# The deleted anchor was also VACUOUS, measured rather than argued. Inserting a bare
# `raise` at each retrieval call and running the suite:
#
#   raise at document_routes.py:1253  (/query retrieval)            -> 2 failed   REACHABLE
#   raise at document_routes.py:1320  (/query/{entity_id} retrieval) -> 10 passed  NEVER RUNS
#
# `_require_entity` returns 403 for a foreign entity before retrieval is reached, so no
# content test can exercise that line. A mutation there cannot redden under ANY
# combination, which makes "SURVIVED" there indistinguishable from "unreachable" -- and
# only one of those is a finding.
#
# STANDING RULE (INTEGRATION, 2026-09-23): before citing a survivor as evidence of
# redundancy, break the site in a way that MUST redden and confirm that it does. A
# survivor proves nothing until the site is shown capable of failing at all.
