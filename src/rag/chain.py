import logging
from typing import AsyncGenerator, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.retrieval.retriever import VectorRetriever

import os

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a medical assistant.\n"
    "Answer the question strictly using the provided context.\n"
    "If the answer is not in the context, say you do not know.\n"
    "Cite page numbers where relevant.\n"
    "You MUST cite sources.\n"
    "For every factual claim, add a citation using the format: (Page X)\n"
    "Only use page numbers that appear in the provided context.\n"
    "Do not invent page numbers./n"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}\n\nContext:\n{context}"),
    ]
)


class RAGChain:
    def __init__(self, retriever: VectorRetriever):
        logger.info("Initializing RAGChain with Ollama (llama3.2)")

        self.retriever = retriever

        self.llm = ChatOllama(
            model="llama3.2",
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            streaming=True,
            num_ctx=2048,      # limit context window for speed
            num_predict=256,   # cap generation length
        )

        self.prompt = PROMPT

    async def stream_answer(self, question: str) -> AsyncGenerator[bytes, None]:
        logger.info(f"Running RAG for question: {question}")

        docs: List[Document] = self.retriever.retrieve(question)

        context = "\n\n".join(
            f"(Page {d.metadata.get('page', 'N/A')}) {d.page_content}"
            for d in docs
        )

        messages = self.prompt.format_messages(
            question=question,
            context=context,
        )

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content.encode("utf-8")

    def generate_answer(self, question: str) -> str:
        """
        Non-streaming, bounded generation for evaluation only.
        """
        docs: List[Document] = self.retriever.retrieve(question)

        context = "\n\n".join(
            f"(Page {d.metadata.get('page', 'N/A')}) {d.page_content}"
            for d in docs
        )

        messages = self.prompt.format_messages(
            question=question,
            context=context,
        )

        response = self.llm.invoke(messages)
        return response.content