# app/evaluation/eval_retriever.py

import os
import json
from pathlib import Path
from typing import List

from app.rag_pipeline import get_retriever


def load_eval_dataset(path: Path) -> List[dict]:
    """Load questions from eval_dataset.jsonl"""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def clean_source_name(source: str) -> str:
    """
    Convert metadata['source'] into a simple ID.

    Examples:
      'rag_intro.txt'                          -> 'rag_intro'
      'C:/path/to/rag_components.txt'         -> 'rag_components'
      'D:\\docs\\embeddings_basics.pdf'       -> 'embeddings_basics'
    """
    if not source:
        return ""

    # string + trim spaces/newlines
    s = str(source).strip()

    # last part after any folders
    base = os.path.basename(s).strip()

    # remove extension (.txt, .pdf, .md, etc.)
    root, _ = os.path.splitext(base)

    # final trimmed id
    return root.strip()


def evaluate_retriever(k: int = 5):
    """
    Simple Recall@k evaluation for the retriever.

    For each question in eval_dataset.jsonl:
    - run retriever
    - check if the expected `source_doc_id` is in the top-k retrieved docs
    """
    dataset_path = Path(__file__).parent / "eval_dataset.jsonl"
    examples = load_eval_dataset(dataset_path)

    retriever = get_retriever(k=k)

    total = len(examples)
    hits = 0

    print(f"\nRunning retriever evaluation on {total} examples (k={k})...\n")

    for ex in examples:
        q = ex["question"]
        # clean ground-truth id as well
        target_id = ex["source_doc_id"].strip()

        # LangChain retriever -> .invoke
        docs = retriever.invoke(q)
        docs = docs[:k]

        # raw sources for debug
        raw_sources = [doc.metadata.get("source", "") for doc in docs]

        # cleaned IDs used for comparison
        retrieved_ids = [clean_source_name(s) for s in raw_sources]

        hit = target_id in retrieved_ids
        if hit:
            hits += 1

        print("Q:", q)
        print("Expected id:", target_id)
        print("Raw sources:", raw_sources)
        print("Cleaned ids:", retrieved_ids)
        print("Hit:", hit)
        print("-" * 60)

    recall_at_k = hits / total if total else 0.0
    print(f"\nRecall@{k}: {recall_at_k:.3f} ({hits}/{total} questions)")


if __name__ == "__main__":
    evaluate_retriever(k=5)
