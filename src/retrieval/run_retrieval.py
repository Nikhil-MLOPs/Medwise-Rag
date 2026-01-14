import logging
import time
from pathlib import Path

from src.core.logging_config import setup_logging
from src.retrieval.retriever import VectorRetriever

setup_logging()
logger = logging.getLogger(__name__)


VECTORSTORE_DIR = Path("data/vectorstore")


def run():
    retriever = VectorRetriever(
        vectorstore_dir=VECTORSTORE_DIR,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        k=5,
    )

    query = "What is hypertension?"

    start = time.perf_counter()
    docs = retriever.retrieve(query)
    end = time.perf_counter()

    logger.info(f"Retrieval time: {(end - start):.3f} seconds")

    for idx, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "unknown")
        logger.info(f"Result {idx} (page {page})")
        logger.info(doc.page_content[:300] + "...")

if __name__ == "__main__":
    run()