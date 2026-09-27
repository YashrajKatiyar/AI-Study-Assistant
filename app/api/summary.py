import logging

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.exceptions import DocumentNotFoundError
from app.database.database import get_db
from app.database.repositories import get_document
from app.rag.retrieval import retrieve_document_chunks
from app.agents.summary_agent import generate_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["summary"])


class SummarizeRequest(BaseModel):
    document_id: str


class SummarizeResponse(BaseModel):
    document_id: str
    summary: str


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    request: SummarizeRequest,
    user_id: str = Header(default="default_user", alias="X-User-ID"),
    db: Session = Depends(get_db),
):
    """Summarize a specific uploaded document."""
    logger.info(f"Summarize request for document {request.document_id} from user {user_id}")

    # Verify document exists and belongs to user
    doc = get_document(db, request.document_id)
    if not doc or doc.user_id != user_id:
        raise DocumentNotFoundError(
            f"Document {request.document_id} not found for user {user_id}"
        )

    # Retrieve all chunks for this document
    chunks = retrieve_document_chunks(request.document_id, user_id)
    if not chunks:
        return SummarizeResponse(
            document_id=request.document_id,
            summary="No content found for this document. The document may not have been ingested properly.",
        )

    # Generate summary
    summary = generate_summary(chunks, f"Summarize the document: {doc.filename}")
    logger.info(f"Summary generated for document {request.document_id}")

    return SummarizeResponse(
        document_id=request.document_id,
        summary=summary,
    )
