from abc import ABC, abstractmethod
from typing import List


class TextSplitterService(ABC):
    @abstractmethod
    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into smaller chunks."""
        pass
