from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path

from src.core.logging_config import setup_logging
from src.core.config import load_production_config
from src.retrieval.retriever import VectorRetriever
from src.rag.chain import RAGChain

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

setup_logging()
logger = logging.getLogger("api")

VECTORSTORE_DIR = Path("data/vectorstore")

# ---------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize heavy resources once at startup.
    """
    logger.info("Starting MedWise RAG backend")

    config = load_production_config()

    retriever = VectorRetriever(
        vectorstore_dir=VECTORSTORE_DIR,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        k=config["retrieval"]["k"],
        strategy=config["retrieval"]["strategy"],
    )

    rag_chain = RAGChain(retriever)

    app.state.config = config
    app.state.retriever = retriever
    app.state.rag_chain = rag_chain

    logger.info("Backend startup complete")

    yield

    logger.info("Backend shutdown complete")


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(
    title="MedWise RAG",
    version="0.6.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check(request: Request):
    ready = hasattr(request.app.state, "rag_chain")
    return {"ready": ready}


# ---------------------------------------------------------------------
# RAG streaming endpoint (async-safe)
# ---------------------------------------------------------------------

@app.get("/rag/stream")
async def stream_rag(question: str, request: Request):
    rag_chain: RAGChain = request.app.state.rag_chain

    async def token_stream():
        try:
            async for chunk in rag_chain.stream_answer(question):
                yield chunk
        except Exception as e:
            logger.exception("RAG streaming failed")
            yield f"\n[ERROR] {str(e)}"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
    )