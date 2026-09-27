import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.database.database import init_db
from app.api import chat, documents, summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    init_db()
    logger.info("AI Study Assistant started successfully")
    yield
    logger.info("AI Study Assistant shutting down")


app = FastAPI(
    title="AI Study Assistant",
    description="An AI-powered study assistant using RAG, LangGraph, and ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(summary.router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
