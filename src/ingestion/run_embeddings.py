import logging
from pathlib import Path

from src.core.logging_config import setup_logging
from src.ingestion.chunking import chunk_documents
from src.ingestion.embeddings import build_vectorstore
from src.ingestion.loaders.pdf_loader import load_pdf
from src.ingestion.cleaning.text_cleaner import clean_documents

setup_logging()
logger = logging.getLogger(__name__)


RAW_DATA_DIR = Path("data/data_raw")
VECTORSTORE_DIR = Path("data/vectorstore")


def run():
    pdf_path = next(RAW_DATA_DIR.glob("*.pdf"))

    documents = load_pdf(pdf_path)
    cleaned_docs = clean_documents(documents)

    chunks = chunk_documents(
        cleaned_docs,
        chunk_size=800,
        chunk_overlap=200,
    )

    build_vectorstore(
        chunks=chunks,
        persist_dir=VECTORSTORE_DIR,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,
    )


if __name__ == "__main__":
    run()