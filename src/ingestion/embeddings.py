import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_vectorstore(
    chunks: List[Document],
    persist_dir: Path,
    model_name: str,
    batch_size: int,
) -> Chroma:
    logger.info(f"Initializing HuggingFace embeddings: {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )

    logger.info("Initializing Chroma vector store")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    logger.info("Starting batched embedding")

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)

    logger.info("Embedding completed successfully")
    return vectorstore