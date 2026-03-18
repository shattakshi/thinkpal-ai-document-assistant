# app/prompts.py

from langchain_core.messages import SystemMessage


def get_rag_system_prompt() -> SystemMessage:
    """
    Returns the system prompt used for the RAG pipeline.
    This prompt enforces grounding and prevents hallucinations.
    """
    return SystemMessage(
        content=(
            "You are a helpful AI assistant for document-based question answering. "
            "Use ONLY the information provided in the context to answer the question. "
            "If the answer cannot be found in the context, respond with: "
            "'I don't know based on the provided documents.' "
            "Be concise, clear, and factual."
        )
    )
