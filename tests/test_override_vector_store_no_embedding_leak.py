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
