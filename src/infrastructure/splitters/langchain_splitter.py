from typing import List
import logging
import threading
from src.domains.documents.splitters import TextSplitterService

logger = logging.getLogger(__name__)


class LangChainRecursiveSplitter(TextSplitterService):
    def __init__(self):
        self._splitters = {}
        self._splitter_lock = threading.Lock()

    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        super().split_text(text, chunk_size, overlap)
        if not text:
            return []

        # Graceful fallback custom recursive splitting if external package isn't directly installed
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            key = (chunk_size, overlap)
            if key not in self._splitters:
                with self._splitter_lock:
                    if key not in self._splitters:
                        self._splitters[key] = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=overlap,
                            length_function=len,
                        )
            splitter = self._splitters[key]
            return splitter.split_text(text)
        except ImportError:
            logger.warning("langchain-text-splitters unavailable; using native chunking fallback")
            # Native lightweight chunker respecting overlap
            chunks = []
            start = 0
            text_len = len(text)
            step = max(1, chunk_size - overlap)

            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end == text_len:
                    break
                start += step
            return chunks
