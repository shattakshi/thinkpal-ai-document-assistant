# ingest.py

"""
Document ingestion script for the RAG pipeline.

Loads documents from data/documents/,
chunks them into ~300–500 token segments,
embeds them using IBM Granite embeddings,
and persists them in a Chroma vector store.
"""

from app.config import get_embeddings
from app.embeddings_store import get_or_create_vector_store


def main():
    print("📚 Starting document ingestion...")

    embeddings = get_embeddings()

    # This call handles:
    # - loading raw documents
    # - chunking (300–500 tokens)
    # - embedding with IBM Granite
    # - persisting the vector store
    vectordb = get_or_create_vector_store(embeddings)

    print("✅ Ingestion complete. Vector store is ready.")


if __name__ == "__main__":
    main()
