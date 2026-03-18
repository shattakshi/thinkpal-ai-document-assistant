# app/rag_query.py

"""
Single-question RAG query script.

Usage (from project root):
    python -m app.rag_query "What is Retrieval-Augmented Generation?"
"""

import sys
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from .config import get_llm, get_embeddings
from .embeddings_store import get_or_create_vector_store
from .rag_pipeline import RAGPipeline


def format_sources(docs: List[Document]) -> str:
    """Pretty-print sources from retrieved documents."""
    if not docs:
        return "  (no sources returned)"

    lines = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        src_name = Path(str(src)).name if src else "unknown"
        lines.append(f"  {i}. {src_name} (page {page})")

    return "\n".join(lines)


def build_pipeline() -> RAGPipeline:
    """Initialise LLM, embeddings, vector store, and RAG pipeline."""
    print("🔧 Initialising LLM, embeddings, and vector store...")
    llm = get_llm()
    embeddings = get_embeddings()
    vectorstore = get_or_create_vector_store(embeddings)
    return RAGPipeline(llm=llm, vectorstore=vectorstore)


# Build once (CLI-friendly)
PIPELINE = build_pipeline()


def run_query(question: str) -> Tuple[str, List[Document]]:
    """Run a single RAG query and return answer text + source docs."""
    response, docs = PIPELINE.ask(question)

    # Extract text safely from Granite response
    if isinstance(response, BaseMessage):
        answer_text = response.content
    else:
        answer_text = str(response)

    return answer_text, docs


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("Enter your question: ").strip()

    if not question:
        print("No question provided. Exiting.")
        return

    print(f"\n❓ Question:\n{question}\n")

    answer, docs = run_query(question)

    print("🤖 Answer:\n")
    print(answer)
    print("\n📚 Sources:\n")
    print(format_sources(docs))


if __name__ == "__main__":
    main()
