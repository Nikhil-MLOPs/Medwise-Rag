import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(
        self,
        vectorstore_dir: Path,
        model_name: str,
        k: int = 5,
        strategy: str = "mmr",
        fetch_k: int = 20,
    ):
        """
        Vector retriever with configurable retrieval strategy.

        strategy:
          - "mmr": Maximal Marginal Relevance
          - "similarity": Pure similarity search
        """

        logger.info("Initializing embedding function for retrieval")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

        logger.info("Loading Chroma vectorstore from disk")

        self.vectorstore = Chroma(
            persist_directory=str(vectorstore_dir),
            embedding_function=self.embeddings,
        )

        self.k = k
        self.fetch_k = fetch_k
        self.strategy = strategy

        if strategy == "mmr":
            logger.info("Using MMR retrieval strategy")
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k},
            )
        elif strategy == "similarity":
            logger.info("Using similarity retrieval strategy")
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )
        else:
            raise ValueError(
                f"Unsupported retrieval strategy: {strategy}"
            )

    def retrieve(self, query: str):
        logger.info(f"Running retrieval for query: {query}")
        docs = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents")
        return docs
