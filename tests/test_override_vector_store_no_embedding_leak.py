"""N1 Part A regression: tests/test_main.py's autouse `override_vector_store` fixture must
patch `vector_store.embedding_function` via monkeypatch (auto-restored), NOT by direct
attribute assignment on the module-global singleton.

WHY: a direct `vector_store.embedding_function = DummyEmbedding()` is never undone at
teardown, so the dummy leaks into every later test in the same process that reads the real
embedding_function -- a silent cross-test contamination. Every other patch in that fixture
already uses monkeypatch; this pins that the embedding one does too.

This is a source-level control on the ACTUAL fixture (deterministic red-first): it reds the
moment a direct assignment to `vector_store.embedding_function` reappears in the fixture."""
import inspect
import re

import pytest

import tests.test_main as test_main


def test_override_vector_store_uses_monkeypatch_for_embedding_function():
    src = inspect.getsource(test_main.override_vector_store)
    # A direct assignment target (LHS), not a monkeypatch.setattr(...) call.
    direct_assign = re.search(r"^\s*vector_store\.embedding_function\s*=", src, re.MULTILINE)
    assert direct_assign is None, (
        "override_vector_store assigns vector_store.embedding_function directly; it leaks past "
        "teardown. Use monkeypatch.setattr(vector_store, 'embedding_function', ...)."
    )
    # And the dummy IS installed via monkeypatch, so the fixture still does its job.
    assert 'monkeypatch.setattr(vector_store, "embedding_function"' in src \
        or "monkeypatch.setattr(vector_store, 'embedding_function'" in src, src


def test_embedding_function_is_restored_after_the_fixture_tears_down():
    """RUNTIME proof (Richard 2026-09-24: the source-level check above is necessary but too weak
    alone -- it proves what the fixture SAYS, not what teardown DOES). Run test_main's ACTUAL
    fixture body under a controlled MonkeyPatch, confirm it really installed the dummy (so this
    cannot pass vacuously), then undo() -- which is exactly what pytest does at teardown -- and
    confirm the ORIGINAL embedding_function is back. With the old direct assignment undo() had
    nothing to restore and this reds."""
    from app.config import vector_store

    original = vector_store.embedding_function
    mp = pytest.MonkeyPatch()
    try:
        result = test_main.override_vector_store.__wrapped__(mp)   # the real fixture body
        if hasattr(result, "__next__"):                          # tolerate a yield-style fixture
            next(result, None)
        assert vector_store.embedding_function is not original, (
            "the fixture did not replace embedding_function -- this test would be vacuous"
        )
        assert type(vector_store.embedding_function).__name__ == "DummyEmbedding", (
            type(vector_store.embedding_function)
        )
    finally:
        mp.undo()                                                  # == fixture teardown
    assert vector_store.embedding_function is original, (
        "embedding_function LEAKED past teardown: %r" % (vector_store.embedding_function,)
    )

