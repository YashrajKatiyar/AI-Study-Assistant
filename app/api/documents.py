import logging
import os
import tempfile
from typing import Any

from fastapi import APIRouter, UploadFile, File, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import UnsupportedFileTypeError, DocumentNotFoundError, IngestionError
from app.database.database import get_db
from app.database.repositories import create_document, get_documents, get_document, delete_document
from app.rag.ingestion import ingest_document
from app.rag.retrieval import delete_document_vectors

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    created_at: str


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Header(default="default_user", alias="X-User-ID"),
    db: Session = Depends(get_db),
):
    """Upload and ingest a study document (PDF, TXT, or MD)."""
    # Validate file extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"File type '{ext}' is not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Validate file size
    settings = get_settings()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {settings.MAX_FILE_SIZE_MB} MB limit.",
        )

    # Save to temp file
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(content)
        tmp.close()

        # Create DB record
        doc = create_document(db, user_id, filename, ext.lstrip("."))
        logger.info(f"Document record created: {doc.id} ({filename})")

        # Ingest
        chunk_count = ingest_document(tmp.name, filename, user_id, doc.id)
        logger.info(f"Ingested {chunk_count} chunks for document {doc.id}")

        return DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            created_at=doc.created_at.isoformat(),
        )
    finally:
        if tmp and os.path.exists(tmp.name):
            os.unlink(tmp.name)


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(
    user_id: str = Header(default="default_user", alias="X-User-ID"),
    db: Session = Depends(get_db),
):
    """List all documents belonging to the current user."""
    docs = get_documents(db, user_id)
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            file_type=d.file_type,
            created_at=d.created_at.isoformat(),
        )
        for d in docs
    ]


@router.delete("/{document_id}")
async def delete_doc(
    document_id: str,
    user_id: str = Header(default="default_user", alias="X-User-ID"),
    db: Session = Depends(get_db),
):
    """Delete a document and its vectors from ChromaDB."""
    doc = get_document(db, document_id)
    if not doc or doc.user_id != user_id:
        raise DocumentNotFoundError(f"Document {document_id} not found for user {user_id}")

    # Remove vectors first
    delete_document_vectors(document_id)

    # Remove DB record
    delete_document(db, document_id)
    logger.info(f"Document {document_id} deleted successfully")
    return {"message": "Document deleted successfully"}
