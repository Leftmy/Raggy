from abc import ABC, abstractmethod
from typing import List


class TextSplitterService(ABC):
    @abstractmethod
    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into smaller chunks."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and less than chunk_size")
