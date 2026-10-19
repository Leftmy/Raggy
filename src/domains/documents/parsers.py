import io
from abc import ABC, abstractmethod
from typing import BinaryIO
from pypdf import PdfReader
from docx import Document
from src.core.exceptions import DocumentProcessingError


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
            if isinstance(file_content, bytes):
                return file_content.decode("utf-8", errors="ignore")
            return file_content.read().decode("utf-8", errors="ignore")
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse text file '{filename}': {str(e)}") from e


class ParserFactory:
    @staticmethod
    def get_parser(filename: str) -> DocumentParser:
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        if ext == "pdf":
            return PDFParser()
        elif ext in ["docx", "doc"]:
            return DocxParser()
        elif ext in ["txt", "md", ""]:
            return TextParser()
        raise ValueError(f"Unsupported file format: {ext}")
