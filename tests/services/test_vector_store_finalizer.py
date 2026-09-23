"""The store finalizer must not assume the constructor finished.

REGRESSION for F-ASYNCPG-DEL-UNRAISABLE. `PGVector.__del__` (langchain_community
pgvector.py:368) reads `self._bind` unguarded, and `_bind` is assigned at the END of
PGVector's initialisation. So any instance that never got there raises AttributeError when
it is collected. Reproduced at 5816e13 as:

    PytestUnraisableExceptionWarning: Exception ignored in:
      <function PGVector.__del__ at 0x...>
      File ".../langchain_community/vectorstores/pgvector.py", line 368, in __del__
        if isinstance(self._bind, sqlalchemy.engine.Connection):
    AttributeError: 'AsyncPgVector' object has no attribute '_bind'

from tests/utils/test_delete_by_text_source.py, which builds a store with
`AsyncPgVector.__new__` to exercise the delete filter without a database.

WHY THIS IS TESTED AT THE PRODUCT AND NOT PATCHED IN THE TEST. Six test files already work
around it by hand-assigning `_bind = None`; two of them say so in a comment. That is a
product class leaking a construction requirement onto every double, and the seventh double
to forget it re-opens the same warning. The guard belongs in ExtendedPgVector, which is the
shared base of both AsyncPgVector and ExtendedPgVector, so neither can regress.

WHY THE WARNING IS WORTH REMOVING AT ALL. An unraisable exception is attributed to whichever
test happened to be running when the collection occurred, NOT to the test that created the
object -- so it is a permanent false lead pointing at an innocent test, and it is noise that
a real finalizer failure would later hide in.
"""

import gc
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from sqlalchemy.engine import Connection, Engine

from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.services.vector_store.extended_pg_vector import ExtendedPgVector


@contextmanager
def _captured_unraisables():
    """Collect everything CPython routes to sys.unraisablehook inside the block.

    This is what pytest's own unraisable plugin listens to; capturing it directly makes
    the assertion deterministic instead of depending on when pytest happens to flush.
    """
    seen = []
    previous = sys.unraisablehook
    sys.unraisablehook = seen.append
    try:
        yield seen
    finally:
        sys.unraisablehook = previous


class _AlwaysRaisesOnDel:
    def __del__(self):
        raise RuntimeError("POSITIVE CONTROL: this finalizer is meant to fail")


def test_the_capture_mechanism_actually_sees_a_failing_finalizer():
    """POSITIVE CONTROL for the two tests below.

    Without this, `seen == []` would also be satisfied by a hook that was never installed,
    an object that was never collected, or a CPython that stopped routing finalizer
    exceptions here -- i.e. the guard test would be indistinguishable from its own absence.
    """
    with _captured_unraisables() as seen:
        obj = _AlwaysRaisesOnDel()
        del obj
        gc.collect()

    assert len(seen) == 1, (
        "the unraisable capture saw %d events, expected exactly 1 -- the mechanism the "
        "other tests in this file rely on is not working, so their greens mean nothing"
        % len(seen))
    assert isinstance(seen[0].exc_value, RuntimeError)


@pytest.mark.parametrize("cls", [AsyncPgVector, ExtendedPgVector])
def test_collecting_a_store_that_never_bound_raises_nothing(cls):
    """THE REGRESSION. Remove the `__del__` override and this reddens with the
    AttributeError quoted in the module docstring."""
    with _captured_unraisables() as seen:
        store = cls.__new__(cls)  # no __init__: `_bind` is never assigned
        assert not hasattr(store, "_bind"), (
            "fixture no longer builds an UNBOUND store, so it cannot exercise the guard")
        del store
        gc.collect()

    assert seen == [], (
        "collecting an unbound %s raised inside its finalizer: %r. A destructor must not "
        "assume __init__ completed." % (cls.__name__, [u.exc_value for u in seen]))


@pytest.mark.parametrize("cls", [AsyncPgVector, ExtendedPgVector])
def test_an_owned_connection_is_still_closed(cls):
    """BEHAVIOUR PRESERVED. The guard must not turn the finalizer into a no-op -- that
    would trade a cosmetic warning for a leaked connection, which is the worse defect."""
    store = cls.__new__(cls)
    conn = MagicMock(spec=Connection)
    store._bind = conn

    store.__del__()

    conn.close.assert_called_once_with()


@pytest.mark.parametrize("cls", [AsyncPgVector, ExtendedPgVector])
def test_an_engine_is_left_alone(cls):
    """PGVector closes only what it did not create: a store given an Engine owns it and
    disposes it elsewhere (factory.close_vector_store_connections), so the finalizer must
    not touch it."""
    store = cls.__new__(cls)
    engine = MagicMock(spec=Engine)
    store._bind = engine

    store.__del__()

    engine.dispose.assert_not_called()


@pytest.mark.parametrize("cls", [AsyncPgVector, ExtendedPgVector])
def test_a_bind_of_none_raises_nothing(cls):
    """The shape six existing test doubles hand-assign. It must keep working, so this card
    does not silently invalidate them."""
    store = cls.__new__(cls)
    store._bind = None

    store.__del__()
