import logging
import os
from typing import AsyncGenerator, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from src.retrieval.retriever import VectorRetriever

logger = logging.getLogger(__name__)

# ---------------------------
# Environment configuration
# ---------------------------
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
)

USE_DUMMY_LLM = os.getenv("USE_DUMMY_LLM", "false").lower() == "true"

# ---------------------------
# Prompt
# ---------------------------
SYSTEM_PROMPT = (
    "You are a medical assistant.\n"
    "Answer the question strictly using the provided context.\n"
    "If the answer is not in the context, say you do not know.\n"
    "Cite page numbers where relevant.\n"
    "You MUST cite sources.\n"
    "For every factual claim, add a citation using the format: (Page X)\n"
    "Only use page numbers that appear in the provided context.\n"
    "Do not invent page numbers.\n"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}\n\nContext:\n{context}"),
    ]
)

# ---------------------------
# Dummy LLM for CI
# ---------------------------
class DummyChatModel(BaseChatModel):
    async def _agenerate(self, messages, stop=None, **kwargs):
        return AIMessage(content="CI dummy response")

    def _generate(self, messages, stop=None, **kwargs):
        return AIMessage(content="CI dummy response")

    @property
    def _llm_type(self) -> str:
        return "dummy"

# ---------------------------
# RAG Chain
# ---------------------------
class RAGChain:
    def __init__(self, retriever: VectorRetriever):
        self.retriever = retriever

        if USE_DUMMY_LLM:
            logger.warning("Initializing RAGChain with Dummy LLM (CI mode)")
            self.llm: BaseChatModel = DummyChatModel()
            self.streaming_enabled = False
        else:
            logger.info("Initializing RAGChain with Ollama (llama3.2)")
            self.llm = ChatOllama(
                model="llama3.2",
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
                streaming=True,
                num_ctx=2048,
                num_predict=256,
            )
            self.streaming_enabled = True

        self.prompt = PROMPT

    # ---------------------------
    # Streaming path (API)
    # ---------------------------
    async def stream_answer(self, question: str) -> AsyncGenerator[bytes, None]:
        logger.info(f"Running RAG (streaming) for question: {question}")

        docs: List[Document] = self.retriever.retrieve(question)

        context = "\n\n".join(
            f"(Page {d.metadata.get('page', 'N/A')}) {d.page_content}"
            for d in docs
        )

        messages = self.prompt.format_messages(
            question=question,
            context=context,
        )

        # Dummy LLM does not support streaming
        if not self.streaming_enabled:
            response = self.llm.invoke(messages)
            yield response.content.encode("utf-8")
            return

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content.encode("utf-8")

    # ---------------------------
    # Non-streaming path (evaluation)
    # ---------------------------
    def generate_answer(self, question: str) -> str:
        logger.info(f"Running RAG (non-streaming) for question: {question}")

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