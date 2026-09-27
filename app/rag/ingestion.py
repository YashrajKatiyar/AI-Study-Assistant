import logging
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pypdf import PdfReader

from app.core.config import get_settings
from app.core.exceptions import UnsupportedFileType, IngestionError
from app.llm.llm import get_embeddings

logger = logging.getLogger(__name__)

def ingest_document(file_path: str, filename: str, user_id: str, document_id: str) -> int:
    try:
        if filename.lower().endswith('.pdf'):
            reader = PdfReader(file_path)
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"page": i}))
        elif filename.lower().endswith(('.txt', '.md')):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            docs = [Document(page_content=content, metadata={"page": 0})]
        else:
            raise UnsupportedFileType(f"Unsupported file type for {filename}")

        settings = get_settings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        )
        
        chunks = text_splitter.split_documents(docs)
        
        for chunk in chunks:
            page_number = chunk.metadata.get('page', 0)
            chunk.metadata = {
                "user_id": user_id,
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "chunk_id": str(uuid.uuid4())
            }

        vector_store = Chroma(
            collection_name="study_materials",
            embedding_function=get_embeddings(),
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        
        if chunks:
            vector_store.add_documents(chunks)
            logger.info(f"Ingested {len(chunks)} chunks for document {document_id}")
            return len(chunks)
        
        logger.warning(f"No chunks created for document {document_id}")
        return 0
    except UnsupportedFileType:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document {document_id}: {str(e)}")
        raise IngestionError(f"Failed to ingest document: {str(e)}") from e
