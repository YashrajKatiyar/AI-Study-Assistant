import logging
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.llm.llm import get_embeddings

logger = logging.getLogger(__name__)

def _get_vector_store() -> Chroma:
    return Chroma(
        collection_name="study_materials",
        embedding_function=get_embeddings(),
        persist_directory=get_settings().CHROMA_PERSIST_DIR,
    )

def retrieve_documents(query: str, user_id: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        vector_store = _get_vector_store()
        k = top_k if top_k is not None else get_settings().RAG_TOP_K
        
        docs = vector_store.similarity_search(
            query, 
            k=k, 
            filter={"user_id": user_id}
        )
        
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    except Exception as e:
        logger.error(f"Error retrieving documents for user {user_id}: {str(e)}")
        raise RetrievalError(f"Failed to retrieve documents: {str(e)}") from e

def retrieve_document_chunks(document_id: str, user_id: str) -> List[Dict[str, Any]]:
    try:
        vector_store = _get_vector_store()
        
        # Using a large k value to get all chunks or retrieving with filter
        docs = vector_store.similarity_search(
            "", 
            k=10000, 
            filter={"document_id": document_id, "user_id": user_id}
        )
        
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    except Exception as e:
        logger.error(f"Error retrieving chunks for document {document_id}: {str(e)}")
        raise RetrievalError(f"Failed to retrieve document chunks: {str(e)}") from e

def delete_document_vectors(document_id: str) -> None:
    try:
        vector_store = _get_vector_store()
        vector_store._collection.delete(where={"document_id": document_id})
        logger.info(f"Deleted vectors for document {document_id}")
    except Exception as e:
        logger.error(f"Error deleting vectors for document {document_id}: {str(e)}")
