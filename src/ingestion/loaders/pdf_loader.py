import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: Path) -> List[Document]:
    logger.info(f"Starting PDF load: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    logger.info(f"Loaded {len(documents)} pages from PDF")
    return documents