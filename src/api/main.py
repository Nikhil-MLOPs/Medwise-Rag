from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import logging
from pathlib import Path

from src.core.logging_config import setup_logging
from src.retrieval.retriever import VectorRetriever
from src.rag.chain import RAGChain

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="MedWise RAG", version="0.5.0")

VECTORSTORE_DIR = Path("data/vectorstore")

retriever = VectorRetriever(
    vectorstore_dir=VECTORSTORE_DIR,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    k=5,
)

rag_chain = RAGChain(retriever)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/rag/stream")
async def stream_rag(question: str):
    return StreamingResponse(
        rag_chain.stream_answer(question),
        media_type="text/event-stream",
    )