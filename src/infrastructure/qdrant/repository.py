import logging
import uuid
from typing import Any, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.core.exceptions import VectorStoreError
from src.domains.documents.repositories import VectorRepository
from src.domains.documents.schemas import DocumentChunk

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
        self._client = AsyncQdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
        )

    def get_client(self) -> AsyncQdrantClient:
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    async def ensure_collection(
        self,
        vector_size: int,
        distance: qmodels.Distance = qmodels.Distance.COSINE,
    ) -> bool:
        """Create the collection if it does not already exist."""
        try:
            client = self.get_client()
            exists = await client.collection_exists(collection_name=self.collection_name)
            if exists:
                collection = await client.get_collection(collection_name=self.collection_name)
                vectors = collection.config.params.vectors
                if isinstance(vectors, qmodels.VectorParams) and vectors.size != vector_size:
                    raise VectorStoreError(
                        f"Collection vector size is {vectors.size}; model produced {vector_size}"
                    )
            else:
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
                logger.info("Created Qdrant collection '%s' with vector size %d", self.collection_name, vector_size)
            return True
        except Exception as e:
            logger.error("Error ensuring Qdrant collection '%s' (%s)", self.collection_name, type(e).__name__)
            if isinstance(e, VectorStoreError):
                raise
            raise VectorStoreError("Failed to ensure vector collection exists") from e

    async def save_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Persist chunk embeddings in Qdrant collection."""
        if not chunks:
            return True

        client = self.get_client()
        try:
            points = []
            expected_vector_size = len(chunks[0].embedding or [])
            for chunk in chunks:
                if not chunk.embedding:
                    raise ValueError(f"Chunk '{chunk.chunk_id}' has no valid embedding")
                if len(chunk.embedding) != expected_vector_size:
                    raise ValueError("All chunk embeddings must have the same vector dimension")

                # Qdrant requires valid UUID or integer point IDs
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                payload = {
                    **{
                        key: value
                        for key, value in chunk.metadata.items()
                        if key not in {"chunk_id", "document_id", "content", "score"}
                    },
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                }
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=chunk.embedding,
                        payload=payload,
                    )
                )

            if points:
                await self.ensure_collection(vector_size=expected_vector_size)
                batch_size = 128
                for offset in range(0, len(points), batch_size):
                    await client.upsert(
                        collection_name=self.collection_name,
                        points=points[offset:offset + batch_size],
                        wait=True,
                    )
            return True
        except Exception as e:
            logger.error("Failed to save chunks to Qdrant (%s)", type(e).__name__)
            raise VectorStoreError("Failed to persist document chunks") from e

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Perform vector similarity search against Qdrant."""
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        client = self.get_client()
        try:
            query_filter = None
            if filters:
                if any(isinstance(value, (dict, list, set)) for value in filters.values()):
                    raise ValueError("Qdrant metadata filters must use scalar values")
                conditions = [
                    qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
                    for key, value in filters.items()
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
                result_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        content=content,
                        metadata=metadata,
                        score=hit.score,
                    )
                )

            return result_chunks
        except Exception as e:
            logger.error("Failed to execute vector search (%s)", type(e).__name__)
            raise VectorStoreError("Vector search failed") from e

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
            logger.error("Failed to delete document '%s' (%s)", document_id, type(e).__name__)
            raise VectorStoreError("Failed to delete document") from e
