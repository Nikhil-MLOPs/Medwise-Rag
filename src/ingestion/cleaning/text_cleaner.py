import logging
import re
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def clean_documents(documents: List[Document]) -> List[Document]:
    cleaned_docs = []

    logger.info("Starting document cleaning")

    for doc in documents:
        text = doc.page_content

        text = re.sub(r"\s+", " ", text)
        text = text.replace("\x00", "").strip()

        cleaned_doc = Document(
            page_content=text,
            metadata=doc.metadata,
        )

        cleaned_docs.append(cleaned_doc)

    logger.info("Document cleaning completed")
    return cleaned_docs