from app.config import get_embeddings
from app.embeddings_store import get_or_create_vector_store


def main():
    embeddings = get_embeddings()
    vectorstore = get_or_create_vector_store(embeddings)

    print("\n✅ Vector store loaded.")

    query = "What is Retrieval Augmented Generation?"

    print(f"\n🔎 Query: {query}\n")

    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("-" * 40)
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()