from typing import List
from src.domains.documents.splitters import TextSplitterService


class LangChainRecursiveSplitter(TextSplitterService):
    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        if not text:
            return []

        # Graceful fallback custom recursive splitting if external package isn't directly installed
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len,
            )
            return splitter.split_text(text)
        except ImportError:
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
