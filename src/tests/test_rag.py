from src.rag.chain import RAGChain
from src.retrieval.retriever import VectorRetriever
from pathlib import Path


def test_rag_chain_initializes():
    retriever = VectorRetriever(
        vectorstore_dir=Path("data/vectorstore"),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        k=2,
    )

    rag = RAGChain(retriever)

    assert rag is not None