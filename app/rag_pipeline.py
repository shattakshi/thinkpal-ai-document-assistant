# app/rag_pipeline.py

from typing import List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


class RAGPipeline:
    """
    Core Retrieval-Augmented Generation pipeline.
    Retrieves relevant document chunks and uses IBM Granite
    to generate grounded answers.
    """

    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore

    def ask(
        self,
        question: str,
        chat_history: List[Tuple[str, str]] | None = None,
    ):
        """
        question: user query
        chat_history: list of (role, message)
        """

        if chat_history is None:
            chat_history = []

        # 1️⃣ Retrieve top-3 relevant chunks
        retrieved_docs = self.vectorstore.similarity_search(
            question, k=3
        )

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        # 2️⃣ System grounding prompt (Granite-friendly)
        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer the user's question using ONLY the provided context. "
            "If the answer is not present in the context, say you do not know."
        )

        messages = [SystemMessage(content=system_prompt)]

        # 3️⃣ Add retrieved context
        messages.append(
            SystemMessage(content=f"Context:\n{context}")
        )

        # 4️⃣ Add chat history (if any)
        for role, msg in chat_history:
            if role.lower() == "user":
                messages.append(HumanMessage(content=msg))
            elif role.lower() == "assistant":
                messages.append(AIMessage(content=msg))

        # 5️⃣ Add current user question
        messages.append(HumanMessage(content=question))

        # 6️⃣ Invoke IBM Granite
        response = self.llm.invoke(messages)

        return response, retrieved_docs
