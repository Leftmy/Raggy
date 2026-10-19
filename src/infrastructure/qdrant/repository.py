import logging
import uuid
from typing import List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from src.domains.documents.repositories import VectorRepository
from src.domains.documents.schemas import DocumentChunk
from src.core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class QdrantVectorRepository(VectorRepository):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "rag_documents",
        api_key: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.api_key = api_key
        self._client: Optional[AsyncQdrantClient] = None

    def get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
            )
        return self._client

    async def ensure_collection(
        self,
        vector_size: int = 384,
        distance: qmodels.Distance = qmodels.Distance.COSINE,
    ) -> bool:
        """Create the collection if it does not already exist."""
        try:
            client = self.get_client()
            exists = await client.collection_exists(collection_name=self.collection_name)
            if not exists:
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}' with vector size {vector_size}")
            return True
        except Exception as e:
            logger.error(f"Error ensuring collection '{self.collection_name}': {e}")
            raise VectorStoreError(f"Failed to ensure collection exists: {str(e)}") from e

    async def save_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Persist chunk embeddings in Qdrant collection."""
        if not chunks:
            return True

        client = self.get_client()
        try:
            points = []
            for chunk in chunks:
                if not chunk.embedding:
                    continue

                # Qdrant requires valid UUID or integer point IDs
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    **chunk.metadata,
                }
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=chunk.embedding,
                        payload=payload,
                    )
                )

            if points:
                await client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save chunks to Qdrant: {e}")
            raise VectorStoreError(f"Failed to persist document chunks: {str(e)}") from e

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> List[DocumentChunk]:
        """Perform vector similarity search against Qdrant."""
        client = self.get_client()
        try:
            query_filter = None
            if filters:
                conditions = [
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=val),
                    )
                    for key, val in filters.items()
                ]
                query_filter = qmodels.Filter(must=conditions)

            search_result = await client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )

            result_chunks = []
            for hit in search_result:
                payload = hit.payload or {}
                chunk_id = payload.get("chunk_id", str(hit.id))
                doc_id = payload.get("document_id", "")
                content = payload.get("content", "")
                
                # Strip reserved keys for clean metadata
                metadata = {k: v for k, v in payload.items() if k not in ["chunk_id", "document_id", "content"]}
                metadata["score"] = hit.score

                result_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        content=content,
                        metadata=metadata,
                    )
                )

            return result_chunks
        except Exception as e:
            logger.error(f"Failed to execute vector search: {e}")
            raise VectorStoreError(f"Vector search failed: {str(e)}") from e

    async def delete_by_document_id(self, document_id: str) -> bool:
        """Remove all points belonging to a specific document."""
        client = self.get_client()
        try:
            delete_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )
            await client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter,
                wait=True,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete document '{document_id}': {e}")
            raise VectorStoreError(f"Failed to delete document: {str(e)}") from e
