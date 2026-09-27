import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class UnsupportedFileTypeError(Exception):
    pass

class DocumentNotFoundError(Exception):
    pass

class ConversationNotFoundError(Exception):
    pass

class IngestionError(Exception):
    pass

class RetrievalError(Exception):
    pass

class LLMError(Exception):
    pass


# Aliases for backward compatibility — both naming styles work
UnsupportedFileType = UnsupportedFileTypeError
DocumentNotFound = DocumentNotFoundError
ConversationNotFound = ConversationNotFoundError

def register_exception_handlers(app: FastAPI):
    @app.exception_handler(UnsupportedFileTypeError)
    async def unsupported_file_type_handler(request: Request, exc: UnsupportedFileTypeError):
        logger.error(f"Unsupported file type: {exc}")
        return JSONResponse(status_code=400, content={"message": "Unsupported file type."})

    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
        logger.error(f"Document not found: {exc}")
        return JSONResponse(status_code=404, content={"message": "Document not found."})

    @app.exception_handler(ConversationNotFoundError)
    async def conversation_not_found_handler(request: Request, exc: ConversationNotFoundError):
        logger.error(f"Conversation not found: {exc}")
        return JSONResponse(status_code=404, content={"message": "Conversation not found."})

    @app.exception_handler(IngestionError)
    async def ingestion_error_handler(request: Request, exc: IngestionError):
        logger.error(f"Ingestion error: {exc}")
        return JSONResponse(status_code=500, content={"message": "Error ingesting document."})

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(request: Request, exc: RetrievalError):
        logger.error(f"Retrieval error: {exc}")
        return JSONResponse(status_code=500, content={"message": "Error retrieving context."})

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError):
        logger.error(f"LLM error: {exc}")
        return JSONResponse(status_code=502, content={"message": "Error from LLM service."})

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.error(f"Value error: {exc}")
        return JSONResponse(status_code=400, content={"message": "Invalid value provided."})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(status_code=500, content={"message": "Internal server error."})
