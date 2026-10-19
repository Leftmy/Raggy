import io
import pytest
from src.domains.documents.parsers import ParserFactory, PDFParser, DocxParser, TextParser
from src.core.exceptions import DocumentProcessingError


def test_factory_returns_correct_parser():
    assert isinstance(ParserFactory.get_parser("report.pdf"), PDFParser)
    assert isinstance(ParserFactory.get_parser("document.docx"), DocxParser)
    assert isinstance(ParserFactory.get_parser("document.doc"), DocxParser)
    assert isinstance(ParserFactory.get_parser("notes.txt"), TextParser)
    assert isinstance(ParserFactory.get_parser("README.md"), TextParser)


def test_factory_unsupported_extension_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported file format"):
        ParserFactory.get_parser("image.png")


def test_text_parser_bytes_and_stream():
    parser = TextParser()
    raw_bytes = "Ukrainian: Привіт світ\nEnglish: Hello world".encode("utf-8")
    
    # Bytes
    parsed_bytes = parser.parse(raw_bytes, "test.txt")
    assert "Привіт світ" in parsed_bytes

    # Stream (BytesIO)
    stream = io.BytesIO(raw_bytes)
    parsed_stream = parser.parse(stream, "test.txt")
    assert "Hello world" in parsed_stream


def test_pdf_parser_corrupted_raises_document_processing_error():
    parser = PDFParser()
    with pytest.raises(DocumentProcessingError):
        parser.parse(b"not a real pdf content", "broken.pdf")


def test_docx_parser_corrupted_raises_document_processing_error():
    parser = DocxParser()
    with pytest.raises(DocumentProcessingError):
        parser.parse(b"not a real docx content", "broken.docx")
