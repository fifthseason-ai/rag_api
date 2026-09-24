import os
import time
import logging
from collections import defaultdict
from typing import Optional, Any, Dict, List, Union
from sqlalchemy import event
from sqlalchemy import asc, delete, func
from sqlalchemy.orm import Session
from sqlalchemy.engine import Connection, Engine
from langchain_core.documents import Document
from langchain_community.vectorstores.pgvector import PGVector


class ExtendedPgVector(PGVector):
    _query_logging_setup = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_query_logging()

    def __del__(self) -> None:
        """Close an owned Connection, without assuming __init__ ever finished.

        `PGVector.__del__` reads `self._bind` unguarded. `_bind` is set at the END of
        PGVector's initialisation, so ANY instance that did not get there -- a
        `__new__`-constructed double, or a real store whose constructor raised while
        connecting -- raises AttributeError when it is collected. A finalizer cannot
        propagate that: Python routes it to sys.unraisablehook and the process prints
        "Exception ignored in: <function PGVector.__del__>". Under pytest that surfaces
        as a PytestUnraisableExceptionWarning attributed to whichever test happened to be
        running when the collection occurred, which is not where the object was made.

        A destructor must therefore assume NOTHING about the constructor having run. The
        `Connection is not None` check is the same rule applied to module teardown, where
        globals can already be cleared by the time the last objects are collected.

        Behaviour for a bound store is unchanged, deliberately: an owned Connection is
        still closed, an Engine is still left alone (PGVector only closes what it did not
        create), and a failing close() still raises exactly as before rather than being
        swallowed here -- silencing that would hide a real resource leak behind a fix for
        a cosmetic warning.
        """
        bind = getattr(self, "_bind", None)
        if Connection is not None and isinstance(bind, Connection):
            bind.close()

    def _query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection under a TOTAL order: distance, then row uuid.

        MIRRORS ``PGVector._query_collection`` from langchain_community **0.4.1** (the
        pinned version) and changes exactly one thing: the ``order_by`` gains
        ``EmbeddingStore.uuid`` as a final key. Upstream orders by ``distance`` ALONE,
        which is a PARTIAL order -- equal distances are common (duplicate or near-duplicate
        chunks) and Postgres may return tied rows in any order, measurably so under a
        parallel plan. Because ``LIMIT k`` is applied IN SQL, a tie at the k-boundary
        decides which rows are returned at all, so sorting in Python afterwards cannot
        repair it: the caller never sees the rows that lost the tie.

        Overriding means copying upstream's body, which is a maintenance cost taken
        deliberately: there is no hook to append an ``order_by``. The tie-order suite guards
        this three ways: the uuid key is pinned; this method's existence is pinned (catches a
        delete/rename); the PUBLIC ``similarity_search_with_score_by_vector`` is asserted to
        flow through here (catches an upstream re-route); and the ``langchain_community``
        version is pinned (a bump reds, forcing a re-read of upstream's body this override
        copied). It is NOT true, as an earlier version of this docstring claimed, that "a
        langchain upgrade fails the suite" on its own -- RV-122 measured upgrades that leave
        the identity pin green; the public-path and version pins are what close that gap.

        ``uuid`` is the embedding table's primary key -- unique and never null -- so it is
        a safe final key: it never changes which rows tie, only which tied row wins.
        """
        with Session(self._bind) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            filter_by = [self.EmbeddingStore.collection_id == collection.uuid]
            if filter:
                if self.use_jsonb:
                    filter_clauses = self._create_filter_clause(filter)
                    if filter_clauses is not None:
                        filter_by.append(filter_clauses)
                else:
                    # Old way of doing things
                    filter_clauses = self._create_filter_clause_json_deprecated(filter)
                    filter_by.extend(filter_clauses)

            results: List[Any] = (
                session.query(
                    self.EmbeddingStore,
                    self.distance_strategy(embedding).label("distance"),
                )
                .filter(*filter_by)
                .order_by(asc("distance"), asc(self.EmbeddingStore.uuid))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
                .all()
            )

        return results

    @staticmethod
    def _sanitize_parameters_for_logging(
        parameters: Union[Dict, List, tuple, Any]
    ) -> Any:
        """Sanitize parameters for logging by truncating embeddings and large values."""
        if parameters is None:
            return parameters

        if isinstance(parameters, dict):
            sanitized = {}
            for key, value in parameters.items():
                # Check if the key contains 'embedding' or if the value looks like an embedding vector
                if "embedding" in str(key).lower() or (
                    isinstance(value, (list, tuple))
                    and len(value) > 10
                    and all(isinstance(x, (int, float)) for x in value[:10])
                ):
                    sanitized[key] = f"<embedding vector of length {len(value)}>"
                elif isinstance(value, str) and len(value) > 500:
                    sanitized[key] = value[:500] + "... (truncated)"
                elif isinstance(value, (dict, list, tuple)):
                    sanitized[key] = ExtendedPgVector._sanitize_parameters_for_logging(
                        value
                    )
                else:
                    sanitized[key] = value
            return sanitized
        elif isinstance(parameters, (list, tuple)):
            sanitized = []
            # Check if this is a list of embeddings
            if len(parameters) > 0 and all(
                isinstance(item, (list, tuple))
                and len(item) > 10
                and all(isinstance(x, (int, float)) for x in item[: min(10, len(item))])
                for item in parameters
            ):
                return f"<{len(parameters)} embedding vectors>"

            for item in parameters:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) > 10
                    and all(isinstance(x, (int, float)) for x in item[:10])
                ):
                    sanitized.append(f"<embedding vector of length {len(item)}>")
                elif isinstance(item, str) and len(item) > 500:
                    sanitized.append(item[:500] + "... (truncated)")
                elif isinstance(item, (dict, list, tuple)):
                    sanitized.append(
                        ExtendedPgVector._sanitize_parameters_for_logging(item)
                    )
                else:
                    sanitized.append(item)
            return type(parameters)(sanitized)
        else:
            return parameters

    def setup_query_logging(self):
        """Enable query logging for this vector store only if DEBUG_PGVECTOR_QUERIES is set"""
        # Only setup logging if the environment variable is set to a truthy value
        debug_queries = os.getenv("DEBUG_PGVECTOR_QUERIES", "").lower()
        if debug_queries not in ["true", "1", "yes", "on"]:
            return

        # Only setup once per class
        if ExtendedPgVector._query_logging_setup:
            return

        logger = logging.getLogger("pgvector.queries")
        logger.setLevel(logging.INFO)

        # Create handler if it doesn't exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - PGVECTOR QUERY - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        @event.listens_for(Engine, "before_cursor_execute")
        def receive_before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if "langchain_pg_embedding" in statement:
                context._query_start_time = time.time()
                logger.info(f"STARTING QUERY: {statement}")
                sanitized_params = ExtendedPgVector._sanitize_parameters_for_logging(
                    parameters
                )
                logger.info(f"PARAMETERS: {sanitized_params}")

        @event.listens_for(Engine, "after_cursor_execute")
        def receive_after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if "langchain_pg_embedding" in statement:
                total = time.time() - context._query_start_time
                logger.info(f"COMPLETED QUERY in {total:.4f}s")
                logger.info("-" * 50)

        ExtendedPgVector._query_logging_setup = True

    def get_all_ids(self) -> list[str]:
        """EVERY file identifier in the store, unscoped.

        Kept as the primitive, and deliberately NOT renamed: it does what it says. What
        changed is that `GET /ids` no longer calls it -- an unscoped list reached every
        authenticated caller, whatever tenant they belonged to. Use
        `get_ids_for_entities` for anything a request can reach.
        """
        with Session(self._bind) as session:
            results = session.query(self.EmbeddingStore.custom_id).all()
            return [result[0] for result in results if result[0] is not None]

    def get_row_uuids(
        self,
        file_id: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> list[str]:
        """Primary keys of the rows that exist for one file RIGHT NOW (FILES-01 F3).

        Every other lookup in this class keys on `custom_id`, which is the file_id and
        is therefore shared by every chunk of every version of that file. Replacement
        needs the opposite: the identity of THESE rows, so a later delete cannot touch a
        row written after the capture.

        `uuid` is the table's primary key -- one value per row, independent of the
        chunk's text. That is deliberate and it is the whole point: if a new version
        produces a chunk byte-identical to an old one, content- or digest-based identity
        cannot tell the two rows apart and would either spare a superseded row or delete
        a freshly written one. The primary key can.

        Scoped by user and tenant when given, so a capture can never reach across an
        entitlement boundary even if a caller passes someone else's file_id.
        """
        with Session(self._bind) as session:
            query = session.query(self.EmbeddingStore.uuid).filter(
                self.EmbeddingStore.custom_id == file_id
            )
            if user_id is not None:
                query = query.filter(
                    self.EmbeddingStore.cmetadata["user_id"].astext == user_id
                )
            if tenant_id is not None:
                query = query.filter(
                    self.EmbeddingStore.cmetadata["tenant_id"].astext == tenant_id
                )
            return [str(r[0]) for r in query.all() if r[0] is not None]

    def count_rows_for_ingest(self, file_id: str, ingest_id: str) -> int:
        """How many rows of `file_id` carry THIS write's `ingest_id`, read back now (KC-FILES-1).

        This is the index half of the receipt: the parse half (`extraction`) says what
        the loader read; this says what the table actually holds from the write. It is
        keyed on `ingest_id` -- a new UUID per write -- so earlier versions kept by the
        additive default, and a concurrent writer's rows, can never be counted as ours.
        """
        with Session(self._bind) as session:
            return int(
                session.query(func.count(self.EmbeddingStore.uuid))
                .filter(self.EmbeddingStore.custom_id == file_id)
                .filter(self.EmbeddingStore.cmetadata["ingest_id"].astext == ingest_id)
                .scalar()
                or 0
            )

    def delete_rows_by_uuid(self, row_uuids: list[str]) -> int:
        """Delete exactly these rows by primary key; returns how many were removed.

        An EMPTY list deletes NOTHING and returns 0. That is stated because the other
        delete path in this class treats a falsy `ids` as "no id filter" and would
        remove the whole collection -- the same shape as the dropped-filter accident the
        `text_source` guard exists for. Here an empty capture legitimately means "this
        file had no rows before the call", which must not become "delete everything".

        Returns the count so the caller can report a partial removal instead of
        assuming one: a replacement that did not remove what it superseded leaves stale
        content retrievable, and that has to reach the response rather than be inferred.
        """
        if not row_uuids:
            return 0
        with Session(self._bind) as session:
            stmt = delete(self.EmbeddingStore).where(
                self.EmbeddingStore.uuid.in_(row_uuids)
            )
            result = session.execute(stmt)
            session.commit()
            return int(result.rowcount or 0)

    def get_ids_for_entities(self, entity_ids: list[str]) -> list[str]:
        """File identifiers owned by these entities, and nothing else.

        The predicate is the one `get_documents_by_ids` and `load_document_context`
        already apply at the route -- a document is visible when its `user_id` is within
        the token entitlement. This is that existing rule reaching a route that was
        missed, not a new policy invented here.

        An EMPTY entity list returns NOTHING. Stated because the sibling delete path in
        this class treats a falsy list as "no filter", and the same shape here would turn
        an entitlement with no entities into a disclosure of the whole store.
        """
        # The route hands this the entitlement's `entity_ids`, which the middleware builds
        # as a SET. SQLAlchemy tolerates one; pymongo does not, and the mongo sibling was
        # raising on every call because of it. Normalised in both implementations so the
        # contract is "any iterable of entity ids" and the next caller cannot reintroduce
        # the difference.
        entity_ids = list(entity_ids or [])
        if not entity_ids:
            return []
        with Session(self._bind) as session:
            results = (
                session.query(self.EmbeddingStore.custom_id)
                .filter(self.EmbeddingStore.cmetadata["user_id"].astext.in_(entity_ids))
                .all()
            )
            return [r[0] for r in results if r[0] is not None]

    def get_filtered_ids(
        self, ids: list[str], user_id: Optional[str] = None, document_origin_type: Optional[str] = None, subscription_id: Optional[str] = None
    ) -> list[str]:
        with Session(self._bind) as session:
            query = session.query(self.EmbeddingStore.custom_id)
            if ids:
                query = query.filter(self.EmbeddingStore.custom_id.in_(ids))
            if user_id is not None:
                query = query.filter(
                    self.EmbeddingStore.cmetadata["user_id"].astext == user_id
                )
            if document_origin_type is not None:
                query = query.filter(
                    self.EmbeddingStore.cmetadata["document_origin_type"].astext == document_origin_type
                )
            if subscription_id is not None:
                query = query.filter(
                    self.EmbeddingStore.cmetadata["subscription_id"].astext == subscription_id
                )
            results = query.all()
            return [result[0] for result in results if result[0] is not None]

    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        with Session(self._bind) as session:
            results = (
                session.query(self.EmbeddingStore)
                .filter(self.EmbeddingStore.custom_id.in_(ids))
                .all()
            )
            return [
                Document(page_content=result.document, metadata=result.cmetadata or {})
                for result in results
                if result.custom_id in ids
            ]

    def get_documents_grouped_by_file_id(self, user_id: str) -> dict[str, list[Document]]:
        with Session(self._bind) as session:
            results = (
                session.query(self.EmbeddingStore)
                .filter(
                    self.EmbeddingStore.cmetadata["user_id"].astext == user_id
                )
                .all()
            )
            grouped = defaultdict(list)
            for result in results:
                file_id = result.custom_id
                if file_id is not None:
                    grouped[file_id].append(
                        Document(
                            page_content=result.document,
                            metadata=result.cmetadata or {},
                        )
                    )
            return dict(grouped)

    def _delete_multiple(
        self,
        ids: Optional[list[str]] = None,
        collection_only: bool = False,
        user_id: Optional[str] = None,
        document_origin_type: Optional[str] = None,
        subscription_id: Optional[str] = None,
        text_source: Optional[str] = None,
    ) -> None:
        with Session(self._bind) as session:
            self.logger.debug(
                "Trying to delete vectors by ids (represented by the model "
                "using the custom ids field)"
            )
            stmt = delete(self.EmbeddingStore)
            if collection_only:
                collection = self.get_collection(session)
                if not collection:
                    self.logger.warning("Collection not found")
                    return
                stmt = stmt.where(
                    self.EmbeddingStore.collection_id == collection.uuid
                )
            if ids:
                stmt = stmt.where(self.EmbeddingStore.custom_id.in_(ids))
            if user_id is not None:
                stmt = stmt.where(
                    self.EmbeddingStore.cmetadata["user_id"].astext == user_id
                )
            if document_origin_type is not None:
                stmt = stmt.where(
                    self.EmbeddingStore.cmetadata["document_origin_type"].astext == document_origin_type
                )
            if subscription_id is not None:
                stmt = stmt.where(
                    self.EmbeddingStore.cmetadata["subscription_id"].astext == subscription_id
                )
            if text_source is not None:
                # Narrows the delete to the rows ONE producer wrote (FILES-01), so an
                # escalation can embed better text first and remove only what it
                # superseded. Every clause here narrows; a filter that failed to arrive
                # would therefore delete MORE than the caller asked, which is why this
                # parameter is covered end to end rather than only at this layer.
                stmt = stmt.where(
                    self.EmbeddingStore.cmetadata["text_source"].astext == text_source
                )
            session.execute(stmt)
            session.commit()
