class AppException(Exception):
    """Base application exception."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DocumentProcessingError(AppException):
    """Raised when parsing or processing document fails."""
    pass


class VectorStoreError(AppException):
    """Raised when vector storage operation fails."""
    pass


class LLMGenerationError(AppException):
    """Raised when Gemini generation fails."""
    pass
