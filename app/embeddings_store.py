# app/embeddings_store.py

import os
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .loaders import load_documents_only


def get_or_create_vector_store(embeddings):
    """
    Creates or loads a persistent Chroma vector store.
    This is the ingestion layer of the RAG pipeline.
    Documents are chunked into ~300–500 token segments
    before embedding with IBM Granite embeddings.
    """

    persist_dir = Path(settings.vector_store_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Safe check: load existing vector DB only if non-empty
    if any(persist_dir.iterdir()):
        print(f"➡ Loading existing Chroma DB from {persist_dir}")
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    print("➡ Building vector store from raw documents...")

    # 1️⃣ Load raw documents (NO chunking here)
    raw_docs = load_documents_only(settings.raw_docs_dir)

    # 2️⃣ Chunk documents (resume-safe, RAG-safe)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,   # within 300–500 token target range
        chunk_overlap=50
    )

    docs = splitter.split_documents(raw_docs)

    # 3️⃣ Create and persist vector store
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    vectordb.persist()
    print("✅ Vector store built with RAG-ready chunks.")

    return vectordb
