import os
from uuid import uuid4
from datetime import datetime, timedelta
import chromadb
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    COLLECTION_NAME = "nexus_memory"
    DEFAULT_RESULTS = 5
    
    def __init__(self):
        # Ensure path directory exists
        os.makedirs(settings.chroma_db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB initialized at {settings.chroma_db_path}. Collection contains {self.count()} documents.")

    def add(self, text: str, metadata: dict) -> str:
        # Validate metadata has keys: type, source, timestamp, importance
        required_keys = ["type", "source", "timestamp", "importance"]
        for key in required_keys:
            if key not in metadata:
                raise ValueError(f"Missing required metadata key: {key}")
                
        doc_id = f"{metadata['type']}_{uuid4().hex[:8]}"
        
        try:
            dt = datetime.fromisoformat(metadata["timestamp"])
            epoch = dt.timestamp()
        except Exception:
            epoch = datetime.utcnow().timestamp()

        meta = {
            "type": str(metadata["type"]),
            "source": str(metadata["source"]),
            "timestamp": float(epoch),
            "timestamp_iso": str(metadata["timestamp"]),
            "importance": float(metadata["importance"])
        }
        
        self.collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[doc_id]
        )
        logger.info(f"Added document to vector store: id={doc_id}, type={meta['type']}")
        return doc_id

    def search(self, query: str, n_results: int = DEFAULT_RESULTS, where_filter: dict = None) -> list[dict]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            output = []
            if not results or not results.get("documents"):
                return output
                
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]
            
            for i in range(len(ids)):
                output.append({
                    "id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i]
                })
            return output
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return []

    def delete(self, doc_id: str) -> bool:
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document from vector store: id={doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting from vector store: {e}")
            return False

    def get_by_type(self, type_name: str, limit: int = 50) -> list[dict]:
        try:
            results = self.collection.get(
                where={"type": type_name},
                limit=limit
            )
            output = []
            if not results or not results.get("ids"):
                return output
                
            ids = results["ids"]
            documents = results["documents"]
            metadatas = results["metadatas"]
            
            for i in range(len(ids)):
                output.append({
                    "id": ids[i],
                    "text": documents[i] if documents else "",
                    "metadata": metadatas[i] if metadatas else {}
                })
            return output
        except Exception as e:
            logger.error(f"Error getting by type from vector store: {e}")
            return []

    def get_recent(self, hours: int = 24) -> list[dict]:
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            cutoff_epoch = cutoff.timestamp()
            
            # ChromaDB supports filtering
            results = self.collection.get(
                where={"timestamp": {"$gte": cutoff_epoch}}
            )
            
            output = []
            if not results or not results.get("ids"):
                return output
                
            ids = results["ids"]
            documents = results["documents"]
            metadatas = results["metadatas"]
            
            for i in range(len(ids)):
                output.append({
                    "id": ids[i],
                    "text": documents[i] if documents else "",
                    "metadata": metadatas[i] if metadatas else {}
                })
            return output
        except Exception as e:
            logger.error(f"Error getting recent from vector store: {e}")
            return []

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting vector store count: {e}")
            return 0
