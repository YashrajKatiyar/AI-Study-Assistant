import io
from unittest.mock import patch, MagicMock
from app.database.repositories import create_document


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("app.api.documents.ingest_document")
def test_upload_txt(mock_ingest, client):
    mock_ingest.return_value = 5
    file_content = b"Test content for upload about Python programming."
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        headers={"X-User-ID": "test_user"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["filename"] == "test.txt"
    assert data["file_type"] == "txt"


@patch("app.api.documents.ingest_document")
def test_upload_pdf(mock_ingest, client):
    mock_ingest.return_value = 10
    file_content = b"%PDF-1.4 fake pdf content"
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notes.pdf", io.BytesIO(file_content), "application/pdf")},
        headers={"X-User-ID": "test_user"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "notes.pdf"
    assert data["file_type"] == "pdf"


def test_upload_unsupported(client):
    file_content = b"binary content"
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.exe", io.BytesIO(file_content), "application/octet-stream")},
        headers={"X-User-ID": "test_user"},
    )
    assert response.status_code == 400


@patch("app.api.documents.ingest_document")
def test_list_documents(mock_ingest, client, db_session):
    mock_ingest.return_value = 3
    # Upload a document first
    client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
        headers={"X-User-ID": "test_user"},
    )

    response = client.get(
        "/api/documents/",
        headers={"X-User-ID": "test_user"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1


@patch("app.api.documents.delete_document_vectors")
@patch("app.api.documents.ingest_document")
def test_delete_document(mock_ingest, mock_delete_vectors, client, db_session):
    mock_ingest.return_value = 3
    # Upload first
    upload_resp = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
        headers={"X-User-ID": "test_user"},
    )
    doc_id = upload_resp.json()["id"]

    # Delete
    response = client.delete(
        f"/api/documents/{doc_id}",
        headers={"X-User-ID": "test_user"},
    )
    assert response.status_code == 200
    mock_delete_vectors.assert_called_once_with(doc_id)


@patch("app.api.chat.build_workflow")
def test_chat(mock_build, client, db_session):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "Polymorphism allows objects to take many forms.",
        "route": "RAG",
        "sources": [{"document": "JavaNotes.pdf", "page": 25, "chunk_id": "c1"}],
    }
    mock_build.return_value = mock_graph

    response = client.post(
        "/api/chat",
        json={"question": "What is polymorphism?"},
        headers={"X-User-ID": "test_user"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["route"] == "RAG"
    assert "conversation_id" in data


@patch("app.api.chat.build_workflow")
def test_chat_with_conversation_id(mock_build, client, db_session):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "Here is a Java example.",
        "route": "RAG",
        "sources": [],
    }
    mock_build.return_value = mock_graph

    # First message
    resp1 = client.post(
        "/api/chat",
        json={"question": "What is polymorphism?"},
        headers={"X-User-ID": "test_user"},
    )
    conv_id = resp1.json()["conversation_id"]

    # Follow-up
    resp2 = client.post(
        "/api/chat",
        json={"conversation_id": conv_id, "question": "Give me a Java example"},
        headers={"X-User-ID": "test_user"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["conversation_id"] == conv_id


@patch("app.api.summary.generate_summary")
@patch("app.api.summary.retrieve_document_chunks")
@patch("app.api.documents.ingest_document")
def test_summarize(mock_ingest, mock_retrieve, mock_summary, client, db_session):
    mock_ingest.return_value = 5
    mock_retrieve.return_value = [
        {"content": "Chapter content", "metadata": {"filename": "notes.pdf"}}
    ]
    mock_summary.return_value = "This is a summary of the document."

    # Upload document first
    upload_resp = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"Study content"), "text/plain")},
        headers={"X-User-ID": "test_user"},
    )
    doc_id = upload_resp.json()["id"]

    # Summarize
    response = client.post(
        "/api/summarize",
        json={"document_id": doc_id},
        headers={"X-User-ID": "test_user"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == doc_id
    assert "summary" in data


def test_summarize_not_found(client, db_session):
    response = client.post(
        "/api/summarize",
        json={"document_id": "nonexistent"},
        headers={"X-User-ID": "test_user"},
    )
    assert response.status_code == 404
