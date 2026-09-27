import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.database.database import Base, get_db
from app.main import app

# In-memory SQLite for tests
TEST_DATABASE_URL = "sqlite:///./test_study_assistant.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=test_engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_llm():
    """Mock LLM that returns predictable responses."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Mocked LLM response")
    return mock


@pytest.fixture
def sample_documents():
    """Sample retrieved documents for testing."""
    return [
        {
            "content": "Polymorphism allows objects to take many forms.",
            "metadata": {
                "document_id": "doc1",
                "filename": "JavaNotes.pdf",
                "page_number": 25,
                "chunk_id": "chunk1",
                "user_id": "test_user",
            },
        },
        {
            "content": "Inheritance enables code reuse between classes.",
            "metadata": {
                "document_id": "doc1",
                "filename": "JavaNotes.pdf",
                "page_number": 30,
                "chunk_id": "chunk2",
                "user_id": "test_user",
            },
        },
        {
            "content": "Encapsulation hides internal implementation details.",
            "metadata": {
                "document_id": "doc1",
                "filename": "JavaNotes.pdf",
                "page_number": 35,
                "chunk_id": "chunk3",
                "user_id": "test_user",
            },
        },
    ]


@pytest.fixture
def sample_conversation_history():
    return [
        {"role": "user", "content": "What is polymorphism?"},
        {"role": "assistant", "content": "Polymorphism allows objects to take many forms in OOP."},
    ]


@pytest.fixture
def txt_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("This is a test document about Python programming.\n")
        f.write("Python is a high-level programming language.\n")
        f.write("It supports multiple programming paradigms.\n")
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)
