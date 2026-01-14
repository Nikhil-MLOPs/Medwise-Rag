from pathlib import Path

from src.retrieval.retriever import VectorRetriever


def test_retriever_returns_documents():
    retriever = VectorRetriever(
        vectorstore_dir=Path("data/vectorstore"),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        k=3,
    )

    results = retriever.retrieve("medical condition")

    assert results is not None
    assert len(results) > 0