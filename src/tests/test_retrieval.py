from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.retrieval.retriever import VectorRetriever


def test_retriever_returns_documents(tmp_path: Path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docs = [
        Document(page_content="Hypertension is high blood pressure", metadata={"page": 1}),
        Document(page_content="Diabetes is a metabolic disorder", metadata={"page": 2}),
    ]

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(tmp_path),
    )

    retriever = VectorRetriever(
        vectorstore_dir=tmp_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        k=1,
    )

    results = retriever.retrieve("What is hypertension")

    assert results is not None
    assert len(results) == 1
    assert "Hypertension" in results[0].page_content