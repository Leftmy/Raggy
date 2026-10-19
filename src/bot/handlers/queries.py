from aiogram import Router, types, F
from src.domains.rag.pipeline import RAGPipelineService
from src.domains.rag.schemas import QueryRequest

router = Router(name="query_handlers")


@router.message(F.text)
async def handle_user_query(
    message: types.Message,
    rag_pipeline: RAGPipelineService,
):
    """
    Handle plain text questions or messages.
    Triggers RAGPipelineService: FastEmbed query -> Qdrant search -> FlashRank rerank -> Gemini 1.5 Pro.
    """
    user_query = message.text
    if not user_query or user_query.startswith("/"):
        return

    status_msg = await message.reply("Searching knowledge base and formulating answer...")

    try:
        query_request = QueryRequest(query=user_query)
        response = await rag_pipeline.execute_query(query_request)

        # Format sources
        sources_text = ""
        if response.source_chunks:
            sources_text = "\n\n*Sources:*\n" + "\n".join(
                f"- {chunk.metadata.get('filename', 'Doc')} (score: {chunk.score:.2f})"
                for chunk in response.source_chunks
            )

        reply_content = f"{response.answer}{sources_text}"
        await status_msg.edit_text(reply_content, parse_mode="Markdown")
    except Exception as e:
        await status_msg.edit_text(f"Error generating answer: {str(e)}")
