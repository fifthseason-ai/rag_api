# rag_api — working notes for an agent session

This repository had **no agent instruction file at all**, so a session cloning it had nothing
pointing at the canonical program record. That is what this file fixes; everything below it is
measured fact about working here, not policy.

## The canonical program record

> **The canonical program record is `DECISIONS.md` §A**, indexed by
> `docs/governance/rule-ledger.json` in program-control PR #4. The ledger is an INDEX OVER §A and
> never a second policy store — G023 enforces that mechanically, so the pointer cannot quietly
> become a rival record.

That wording is reproduced verbatim across every repository carrying this pointer, deliberately
unreworded, so four repos share one sentence rather than four paraphrases that drift apart.

## Authority — the part that is not negotiable

- **Never merge. Never deploy. Never force-push.** Merge to `main`, deployment, production
  acceptance and any branch-protection or workflow change are **Demian's**. Push a branch and open
  a PR; that is where this lane's authority ends.
- **Never delete a `.env`.** Never commit secrets, client data, or real-data logs.
- Commit by **explicit named path**. Never `git add -A` or `git add .` — other lanes' work lives in
  adjacent worktrees sharing this `.git`.
- If a task needs an operator, production access, billing, credentials or a product ruling: record
  the **exact** dependency and stop. Do not route around it, and do not ask another session to
  perform something your own permissions refused. **A boundary that can be satisfied by asking a
  different worker is not a boundary.**

## Running the tests

They do not run on the host. Use the prepared image:

```
MSYS_NO_PATHCONV=1 docker run --rm -v "C:/fswt/<your-worktree>:/src" -w /src \
  files01-ocr-test:wip python -m pytest -q
```

`MSYS_NO_PATHCONV=1` is required for volume mounts under Git-Bash.

## Two things that will mislead you if nobody says them

**1. No test in this repository can build a real vector store.** `tests/conftest.py` replaces
`PGVector.__post_init__` with a no-op for the whole session and sets `DSN=dummy://`. So any SQL in
`app/services/vector_store/` is exercised by route-level doubles, never by a live query. A change
to a SQL predicate can be fully green here and still be wrong. The end-to-end journey that does use
a real Postgres lives **outside this repository**, under
`evidence/files-01-2026-09-16/journey/` in the program-control evidence branch.

**2. `TestClient(app)` as a context manager runs the app lifespan**, which opens a real Postgres
connection. The existing suite deliberately constructs it *without* the context manager and
initialises `app.state.thread_pool` by hand. `JWT_SECRET` must also be set **before `main` is
imported**, not inside a fixture — `app/config.py` refuses to start without it, on purpose.

## The locator contract, because consumers get it wrong

`_UNIT_LOCATOR_KEYS` in `app/routes/document_routes.py` is the single place that decides which
metadata key means "this chunk can be cited here". Measured against a real service:

| format | key | note |
|---|---|---|
| PDF | `page` | **0-BASED.** Page two is `page: 1`. Render `page_label` (a string, correct for roman front matter), never `page` raw |
| PPTX | `slide_number` | 1-based |
| XLSX | `page_name` | the sheet **name**. `page_number` is a **sheet index wearing a page's name** — rendering it says "page 1" about a spreadsheet |
| CSV | `row` | **0-BASED** over data rows, header excluded. Indices are **not densified**: a gappy file stores `[0, 2]` |
| DOCX | *none* | no per-unit locator exists. Render the absence, never a blank or a zero |

`source` and `file_directory` carry this service's internal upload path. Render `filename`.

## Evidence conventions

A checkpoint records what was **measured**, not what was intended, and keeps
IMPLEMENTED / TEST-PROVEN / REVIEWED / PUSHED / MERGED / DEPLOYED separate — they are independent
rungs and one never implies another. **Never label a local commit PUSHED**, and never quote an
older CI result against a newer commit: a run that dies in seconds is a non-start, not a failure.

**A test that cannot fail is not evidence.** Mutate the source, confirm the test reddens, restore.
If a mutation cannot reach the format or branch it targets, it proves nothing about that case.

## Platform constitution — preflight before editing

The Fifth Season Platform Constitution and Governance v2.0 (36 rules, canonical primitives, required controls, change workflow) is the decision hierarchy for platform work. Canonical repository copy: [`docs/governance/PLATFORM-CONSTITUTION-AND-GOVERNANCE.md`](https://github.com/fifthseason-ai/fifthseason.ai-core/blob/release/richard-vibe/docs/governance/PLATFORM-CONSTITUTION-AND-GOVERNANCE.md) in fifthseason.ai-core (`release/richard-vibe`). Program-control record copy: fifthseason-program-control `docs/governance/` (constitutional reconciliation index over `DECISIONS.md` §A; constitution provenance copy at `docs/governance/sources/constitution-v2.0-as-received-2026-09-19.md`). Tracking register: the INTEGRATION governance register in program-control.

Before editing: name the user outcome, the affected rule IDs, the canonical primitive and owner, consumers/contracts, and tenant/access/mobile/Antonio/export/upstream effects (Constitution §5.3 / §6.2). If the requested outcome conflicts with a rule, pause, explain the conflict plainly, and recommend a compliant path.

Its own standing: Richard-approved direction that must be reconciled with the canonical program-control record before it is sole authority; later explicit rulings take precedence; the rule ledger remains the program-control DECISIONS record.

Reference implementation for MCP connector entitlement, platform-owned data scope and typed refusal: core PR #518 (Aaron, merged 2026-09-23) — Rules 1, 2, 27, 30, 34; its self-named gaps are tracked in the register.

Files owns extraction and retrieval — Rule 3; Rule 31 knowledge lifecycle.
