from fastapi import FastAPI
import logging

from src.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="MedWise RAG", version="0.1.0")


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "ok"}