import os
from pathlib import Path


def load_documents_only(folder_path):
    """
    Load raw documents from a folder WITHOUT chunking.
    Used by the ingestion pipeline before splitting.
    """

    all_docs = []
    folder = Path(folder_path)

    for file_path in folder.glob("*"):
        ext = file_path.suffix.lower()

        with open(file_path, "rb") as f:

            if ext == ".pdf":
                docs = load_pdf(f)

            elif ext == ".txt":
                docs = load_text(f)

            elif ext == ".docx":
                docs = load_docx(f)

            else:
                continue

            all_docs.extend(docs)

    return all_docs