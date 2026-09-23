#!/bin/bash
# P06-4 residual (a) — the SLOT phase only: the pgvector half plus its per-layer mutations.
# The hermetic half is already committed and proven off-slot (9481702).
#
# Git Bash rewrites `-w /src` into a drive path and mangles `-v` mounts; without this every
# docker run exits 125 with an invalid-working-directory error and NOT a test result.
export MSYS_NO_PATHCONV=1
set -u

WT=C:/fswt/files-dev-xt
SRC=/c/fswt/files-dev-xt/app/routes/document_routes.py
IMG=files01-ocr-test:wip
NET=xtnet
PG=xtpg
DSN="postgresql://postgres:xt@${PG}:5432/postgres"
F=tests/utils/test_cross_tenant_query_isolation.py

cleanup () {
  echo "--- teardown ---"
  timeout 60 docker rm -f "$PG" >/dev/null 2>&1
  timeout 60 docker network rm "$NET" >/dev/null 2>&1
  echo "teardown done; mine left: $(docker ps -a --format '{{.Names}}' | grep -c '^xt' || true)"
}
trap cleanup EXIT

run_suite () {
  timeout 900 docker run --rm --network "$NET" -v "$WT:/src" -w /src \
    -e RAG_TEST_PG_DSN="$DSN" -e RAG_TEST_PG_REQUIRED=1 \
    "$IMG" python -m pytest -q -p no:cacheprovider "$F" 2>&1 | tail -"${1:-8}"
  echo "EXIT=${PIPESTATUS[0]}"
}

echo "=== disposable real pgvector ==="
timeout 60 docker network create "$NET" >/dev/null 2>&1
timeout 180 docker run -d --name "$PG" --network "$NET" \
  -e POSTGRES_PASSWORD=xt -e POSTGRES_DB=postgres pgvector/pgvector:pg16 >/dev/null || exit 1
for i in $(seq 1 30); do
  timeout 20 docker exec "$PG" pg_isready -U postgres >/dev/null 2>&1 && break
  sleep 2
done
timeout 20 docker exec "$PG" pg_isready -U postgres || { echo "PG NEVER READY"; exit 1; }

echo
echo "############ GREEN — 10 tests, real pgvector, both tenants in ONE collection ############"
echo "CMD: docker run --rm --network $NET -v $WT:/src -w /src -e RAG_TEST_PG_DSN=$DSN -e RAG_TEST_PG_REQUIRED=1 $IMG python -m pytest -q -p no:cacheprovider $F"
run_suite 10

echo
echo "############ MUTATION A — drop the ROUTE post-filter (_authorized_only) ############"
# Two layers defend the query routes and a single mutation cannot speak for both. This one
# neutralises the defensive re-filter; the arm filters still run, so what reddens here is
# only what the post-filter alone protects.
cp "$SRC" /tmp/xt_src.bak
python - "$SRC" <<'PY'
import sys, io
p = sys.argv[1]
s = io.open(p, encoding="utf-8", newline="").read()
old = "def _authorized_only(documents, entity_ids):"
assert s.count(old) == 1, "anchor not unique: %d" % s.count(old)
# make it the identity function: the defensive layer stops defending
s = s.replace(old, old + "\n    return documents  # MUTATION A: post-filter neutralised", 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("MUTATION A applied")
PY
[ $? -eq 0 ] || { echo "MUTATION A NOT APPLIED — NOT EVIDENCE"; cp /tmp/xt_src.bak "$SRC"; exit 1; }
run_suite 8
cp /tmp/xt_src.bak "$SRC"

echo
echo "############ MUTATION B — drop the ARM filter (retrieval user_id predicate) ############"
python - "$SRC" <<'PY'
import sys, io
p = sys.argv[1]
s = io.open(p, encoding="utf-8", newline="").read()
old = '            {"user_id": entity_id},'
assert s.count(old) == 1, "anchor not unique: %d" % s.count(old)
s = s.replace(old, '            {},  # MUTATION B: entity predicate dropped from retrieval', 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("MUTATION B applied")
PY
[ $? -eq 0 ] || { echo "MUTATION B NOT APPLIED — NOT EVIDENCE"; cp /tmp/xt_src.bak "$SRC"; exit 1; }
run_suite 8
cp /tmp/xt_src.bak "$SRC"
rm -f /tmp/xt_src.bak

echo
echo "############ RESTORE CHECK — git, not md5 ############"
# md5 is the right control for "did the mutation apply" and the WRONG one for "is the tree
# restored": sed/python rewrites normalise line endings file-wide on this host, so the digest
# moves while the content does not. git status is the content-authoritative check.
cd /c/fswt/files-dev-xt
export PATH="/c/fsnode:$PATH"
git checkout -- app/routes/document_routes.py
echo "git residue (expect only the untracked new suite):"
git status --porcelain
echo
echo "############ DONE — release the slot immediately ############"
