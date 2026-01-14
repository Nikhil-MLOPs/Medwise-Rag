import logging
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(
        self,
        vectorstore_dir: Path,
        model_name: str,
        k: int = 5,
    ):
        logger.info("Initializing embedding function for retrieval")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
        )

        logger.info("Loading Chroma vectorstore from disk")

        self.vectorstore = Chroma(
            persist_directory=str(vectorstore_dir),
            embedding_function=self.embeddings,
        )

        self.k = k

    def retrieve(self, query: str) -> List[Document]:
        logger.info(f"Running retrieval for query: {query}")

        results = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.k,
            fetch_k=self.k * 4,
        )

        logger.info(f"Retrieved {len(results)} documents")
        return results