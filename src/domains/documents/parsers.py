import io
from abc import ABC, abstractmethod
from pathlib import PurePath
from typing import BinaryIO

from docx import Document
from pypdf import PdfReader

from src.core.exceptions import DocumentProcessingError


def sanitize_filename(filename: str) -> str:
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    printable = "".join(character for character in basename if character.isprintable())
    return printable.strip()[:255]


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_content: BinaryIO | bytes, filename: str) -> str:
        """Parse raw file content into text."""
        pass


class PDFParser(DocumentParser):
    def parse(self, file_content: BinaryIO | bytes, filename: str) -> str:
        try:
            stream = file_content if isinstance(file_content, io.BytesIO) else io.BytesIO(
                file_content if isinstance(file_content, bytes) else file_content.read()
            )
            reader = PdfReader(stream)
            text_pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text.strip())
            return "\n\n".join(text_pages)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse PDF file '{filename}': {str(e)}") from e


class DocxParser(DocumentParser):
    def parse(self, file_content: BinaryIO | bytes, filename: str) -> str:
        try:
            stream = file_content if isinstance(file_content, io.BytesIO) else io.BytesIO(
                file_content if isinstance(file_content, bytes) else file_content.read()
            )
            doc = Document(stream)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse DOCX file '{filename}': {str(e)}") from e


class TextParser(DocumentParser):
    def parse(self, file_content: BinaryIO | bytes, filename: str) -> str:
        try:
            content = file_content if isinstance(file_content, bytes) else file_content.read()
            return content.decode("utf-8", errors="replace")
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse text file '{filename}': {str(e)}") from e


class ParserFactory:
    @staticmethod
    def get_parser(filename: str) -> DocumentParser:
        ext = PurePath(filename.replace("\\", "/")).suffix.lower()
        if ext == ".pdf":
            return PDFParser()
        elif ext == ".docx":
            return DocxParser()
        elif ext in [".txt", ".md"]:
            return TextParser()
        raise DocumentProcessingError(f"Unsupported file format: {ext or 'missing extension'}")
