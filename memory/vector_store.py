"""ChromaDB persistent vector store for semantic memory search."""
from datetime import datetime, timedelta
from uuid import uuid4

import chromadb

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "nexus_memory"
DEFAULT_RESULTS = 5


class VectorStore:
    """Persistent semantic memory. Survives restarts. Never forgets."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[VectorStore] Ready — {self._collection.count()} existing entries")

    def add(self, text: str, metadata: dict) -> str:
        """Store a memory entry. Returns generated doc_id."""
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.utcnow().isoformat()
        doc_id = f"{metadata.get('type', 'mem')}_{uuid4().hex[:8]}"
        self._collection.add(documents=[text], metadatas=[metadata], ids=[doc_id])
        logger.debug(f"[VectorStore] Stored {doc_id} type={metadata.get('type')}")
        return doc_id

    def search(
        self,
        query: str,
        n_results: int = DEFAULT_RESULTS,
        where_filter: dict | None = None,
    ) -> list[dict]:
        """Semantic similarity search. Returns ranked list of memory entries."""
        total = self.count()
        if total == 0:
            return []
        n = min(n_results, total)
        kwargs: dict = {"query_texts": [query], "n_results": n}
        if where_filter:
            kwargs["where"] = where_filter
        try:
            result = self._collection.query(**kwargs)
            return [
                {"text": doc, "metadata": meta, "distance": dist, "id": id_}
                for doc, meta, dist, id_ in zip(
                    result["documents"][0],
                    result["metadatas"][0],
                    result["distances"][0],
                    result["ids"][0],
                )
            ]
        except Exception as exc:
            logger.error(f"[VectorStore] Search failed: {exc}")
            return []

    def delete(self, doc_id: str) -> bool:
        """Delete an entry by its ID."""
        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False

    def get_recent(self, hours: int = 24) -> list[dict]:
        """Return entries from the last N hours by timestamp in metadata."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        try:
            result = self._collection.get(where={"timestamp": {"$gte": cutoff}})
            return [
                {"text": doc, "metadata": meta, "id": id_}
                for doc, meta, id_ in zip(
                    result["documents"], result["metadatas"], result["ids"]
                )
            ]
        except Exception:
            return []

    def count(self) -> int:
        """Return total stored entry count."""
        return self._collection.count()
