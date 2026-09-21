import copy
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

class AtlasMongoVector(MongoDBAtlasVectorSearch):
    @property
    def embedding_function(self) -> Embeddings:
        return self.embeddings

    def add_documents(self, docs: list[Document], ids: list[str]):
        # {file_id}_{idx}
        new_ids = [id for id in range(len(ids))]
        file_id = docs[0].metadata['file_id']
        f_ids = [f'{file_id}_{id}' for id in new_ids]
        return super().add_documents(docs, f_ids)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        docs = self._similarity_search_with_score(
            embedding,
            k=k,
            pre_filter=filter,
            post_filter_pipeline=None,
            **kwargs,
        )
        processed_documents: List[Tuple[Document, float]] = []
        for document, score in docs:
            # Make a deep copy to avoid mutating the original document
            doc_copy = copy.deepcopy(document.__dict__)
            # Remove _id field from metadata if it exists
            if "metadata" in doc_copy and "_id" in doc_copy["metadata"]:
                del doc_copy["metadata"]["_id"]
            new_document = Document(**doc_copy)
            processed_documents.append((new_document, score))
        return processed_documents

    def get_all_ids(self) -> list[str]:
        # Return unique file_id fields in self._collection. UNSCOPED -- see
        # ExtendedPgVector.get_all_ids; `GET /ids` must not call this.
        return self._collection.distinct("file_id")

    def get_ids_for_entities(self, entity_ids: list[str]) -> list[str]:
        """File identifiers owned by these entities. Documents here carry a top-level
        `user_id` (see `get_documents_by_ids`), which is the same field the pgvector
        store keeps in `cmetadata`. An empty list returns nothing, never everything."""
        # MUST be a list. `ent["entity_ids"]` is a set and BSON cannot encode one:
        # `InvalidDocument: cannot encode object: {'userY','userX'}, of type: <class 'set'>`
        # raised for every caller on an atlas-mongo deployment. Fail-closed -- the 500
        # handler caught it and nothing leaked -- but the route was dead, and the suite is
        # pgvector-only so nothing noticed. Found by independent review, not by a test.
        entity_ids = list(entity_ids or [])
        if not entity_ids:
            return []
        return self._collection.distinct("file_id", {"user_id": {"$in": entity_ids}})
    
    def get_filtered_ids(self, ids: list[str]) -> list[str]:
        # Return unique file_id fields filtered by the provided ids
        return self._collection.distinct("file_id", {"file_id": {"$in": ids}})

    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        # Return documents filtered by file_id
        return [
            Document(
                page_content=doc["text"],
                metadata={
                    "file_id": doc["file_id"],
                    "user_id": doc["user_id"],
                    "digest": doc["digest"],
                    "source": doc["source"],
                    "page": int(doc.get("page", 0)),
                },
            )
            for doc in self._collection.find({"file_id": {"$in": ids}})
        ]

    def delete(self, ids: Optional[list[str]] = None) -> None:
        # Delete documents by file_id
        if ids is not None:
            self._collection.delete_many({"file_id": {"$in": ids}})