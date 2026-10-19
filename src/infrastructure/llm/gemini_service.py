import logging
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.domains.rag.service import LLMService
from src.domains.rag.schemas import SearchResultChunk
from src.core.exceptions import LLMGenerationError

logger = logging.getLogger(__name__)

SYSTEM_RAG_PROMPT = """You are a helpful and professional AI assistant.
Your goal is to answer the user's questions based primarily or exclusively on the provided document context.
If the context does not contain enough information to formulate a reliable answer, state this clearly.
Respond in the language requested by the user.
"""


class GeminiLLMService(LLMService):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro", temperature: float = 0.2):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self._llm: Optional[ChatGoogleGenerativeAI] = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
            )
        return self._llm

    async def generate_response(self, query: str, context_chunks: List[SearchResultChunk]) -> str:
        llm = self._get_llm()
        try:
            # Build context string with chunk metadata
            if context_chunks:
                context_parts = []
                for idx, chunk in enumerate(context_chunks, 1):
                    source = chunk.metadata.get("filename", f"Doc_{chunk.document_id}")
                    context_parts.append(f"--- Document [{idx}] ({source}) ---\n{chunk.content}")
                context_text = "\n\n".join(context_parts)
            else:
                context_text = "No relevant context found in the knowledge base."

            human_content = (
                f"DOCUMENT CONTEXT:\n"
                f"{context_text}\n\n"
                f"USER QUERY:\n"
                f"{query}\n\n"
                f"Please provide a comprehensive and accurate answer:"
            )

            messages = [
                SystemMessage(content=SYSTEM_RAG_PROMPT),
                HumanMessage(content=human_content),
            ]

            response = await llm.ainvoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error(f"Error during Gemini LLM generation: {e}")
            raise LLMGenerationError(f"LLM generation failed: {str(e)}") from e
