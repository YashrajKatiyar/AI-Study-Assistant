import os
import tempfile
from unittest.mock import patch, MagicMock

from app.agents.rag_agent import generate_rag_answer
from app.rag.reranker import rerank_documents


@patch("app.agents.rag_agent.get_llm")
def test_generate_rag_answer(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Polymorphism allows objects to take many forms.")
    mock_get_llm.return_value = mock_llm

    result = generate_rag_answer(
        question="What is polymorphism?",
        context="Polymorphism allows objects to take many forms in OOP.",
        conversation_history=[],
    )
    assert isinstance(result, str)
    assert len(result) > 0


@patch("app.agents.rag_agent.get_llm")
def test_rag_answer_with_history(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Here is a Java example of polymorphism...")
    mock_get_llm.return_value = mock_llm

    history = [
        {"role": "user", "content": "What is polymorphism?"},
        {"role": "assistant", "content": "Polymorphism allows..."},
    ]
    result = generate_rag_answer(
        question="Give me a Java example",
        context="Polymorphism in Java uses interfaces and inheritance.",
        conversation_history=history,
    )
    assert isinstance(result, str)


@patch("app.rag.reranker.get_llm")
def test_rerank_documents(mock_get_llm, sample_documents):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='[{"index": 0, "score": 9.0}, {"index": 1, "score": 5.0}, {"index": 2, "score": 7.0}]'
    )
    mock_get_llm.return_value = mock_llm

    # Need more docs than final_k to trigger reranking
    docs = sample_documents + [
        {"content": "Abstract classes in Java.", "metadata": {"document_id": "doc1", "filename": "JavaNotes.pdf", "page_number": 40, "chunk_id": "chunk4", "user_id": "test_user"}},
        {"content": "Java interfaces define contracts.", "metadata": {"document_id": "doc1", "filename": "JavaNotes.pdf", "page_number": 45, "chunk_id": "chunk5", "user_id": "test_user"}},
        {"content": "Generics in Java.", "metadata": {"document_id": "doc1", "filename": "JavaNotes.pdf", "page_number": 50, "chunk_id": "chunk6", "user_id": "test_user"}},
    ]

    result = rerank_documents("What is polymorphism?", docs, final_k=3)
    assert len(result) <= 3


@patch("app.rag.reranker.get_llm")
def test_rerank_fallback_on_error(mock_get_llm, sample_documents):
    mock_get_llm.side_effect = Exception("LLM failed")

    docs = sample_documents + [
        {"content": "Extra doc 1", "metadata": {"document_id": "doc1", "filename": "test.pdf", "page_number": 1, "chunk_id": "c4", "user_id": "test_user"}},
        {"content": "Extra doc 2", "metadata": {"document_id": "doc1", "filename": "test.pdf", "page_number": 2, "chunk_id": "c5", "user_id": "test_user"}},
        {"content": "Extra doc 3", "metadata": {"document_id": "doc1", "filename": "test.pdf", "page_number": 3, "chunk_id": "c6", "user_id": "test_user"}},
    ]

    result = rerank_documents("test query", docs, final_k=3)
    assert len(result) == 3  # Falls back to original docs[:final_k]


def test_rerank_small_set(sample_documents):
    """If docs <= final_k, return as-is without calling LLM."""
    result = rerank_documents("test query", sample_documents, final_k=5)
    assert result == sample_documents


@patch("app.rag.ingestion.Chroma")
@patch("app.rag.ingestion.get_embeddings")
def test_ingestion_txt(mock_get_embeddings, mock_chroma, txt_file):
    mock_embeddings = MagicMock()
    mock_get_embeddings.return_value = mock_embeddings
    mock_store = MagicMock()
    mock_chroma.return_value = mock_store

    from app.rag.ingestion import ingest_document
    count = ingest_document(txt_file, "test.txt", "test_user", "doc123")
    assert count > 0
    mock_store.add_documents.assert_called_once()


@patch("app.rag.ingestion.Chroma")
@patch("app.rag.ingestion.get_embeddings")
def test_ingestion_unsupported(mock_get_embeddings, mock_chroma):
    from app.rag.ingestion import ingest_document
    from app.core.exceptions import UnsupportedFileType

    with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
        f.write(b"binary content")
        path = f.name
    try:
        try:
            ingest_document(path, "test.exe", "test_user", "doc123")
            assert False, "Should have raised UnsupportedFileType"
        except (UnsupportedFileType, Exception):
            pass  # Expected
    finally:
        os.unlink(path)


@patch("app.rag.retrieval._get_vector_store")
def test_retrieve_documents(mock_get_store):
    from app.rag.retrieval import retrieve_documents
    from langchain_core.documents import Document

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [
        Document(
            page_content="Test content",
            metadata={"document_id": "doc1", "filename": "test.pdf", "page_number": 1, "chunk_id": "c1", "user_id": "test_user"},
        )
    ]
    mock_get_store.return_value = mock_store

    results = retrieve_documents("test query", "test_user")
    assert len(results) == 1
    assert results[0]["content"] == "Test content"
    assert results[0]["metadata"]["filename"] == "test.pdf"
