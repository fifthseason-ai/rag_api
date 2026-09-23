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
