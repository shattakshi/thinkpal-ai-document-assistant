# app/utils_loader.py

"""
UI utility loaders for uploaded files (e.g., Streamlit).
NOT used in the core RAG ingestion pipeline.
"""

from typing import List
from langchain_core.documents import Document


def load_uploaded_text(content: str, filename: str) -> List[Document]:
    """
    Wrap uploaded text content into a LangChain Document.
    This is intended ONLY for UI-level experimentation,
    not for persistent RAG indexing.
    """
    return [
        Document(
            page_content=content,
            metadata={"source": filename, "page": 1},
        )
    ]
