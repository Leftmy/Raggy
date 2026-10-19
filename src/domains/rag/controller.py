from fastapi import APIRouter, Depends, status
from src.domains.rag.schemas import QueryRequest, QueryResponse
from src.domains.rag.pipeline import RAGPipelineService
from src.core.dependencies import get_rag_pipeline_service

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_rag(
    request: QueryRequest,
    pipeline_service: RAGPipelineService = Depends(get_rag_pipeline_service),
):
    """Execute RAG pipeline: retrieval -> reranking -> LLM answer."""
    return await pipeline_service.execute_query(request)
