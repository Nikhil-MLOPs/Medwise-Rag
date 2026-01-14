import logging
from pathlib import Path

from src.core.logging_config import setup_logging
from src.ingestion.loaders.pdf_loader import load_pdf
from src.ingestion.cleaning.text_cleaner import clean_documents

setup_logging()
logger = logging.getLogger(__name__)


RAW_DATA_DIR = Path("data/data_raw")
PROCESSED_DATA_DIR = Path("data/data_processed")


def run():
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data/data_raw")

    for pdf_path in pdf_files:
        logger.info(f"Processing PDF: {pdf_path.name}")

        documents = load_pdf(pdf_path)
        cleaned_documents = clean_documents(documents)

        output_path = PROCESSED_DATA_DIR / f"{pdf_path.stem}_cleaned.txt"

        with output_path.open("w", encoding="utf-8") as f:
            for doc in cleaned_documents:
                page = doc.metadata.get("page", "unknown")
                f.write(f"[PAGE {page}]\n")
                f.write(doc.page_content + "\n\n")

        logger.info(f"Saved cleaned output to {output_path}")


if __name__ == "__main__":
    run()