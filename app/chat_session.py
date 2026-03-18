# app/chat_session.py

from typing import List, Tuple


class ChatSession:
    """
    Maintains conversation history for a RAG-based chat.
    History is passed to the RAG pipeline to support
    multi-turn, context-aware question answering.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._history: List[Tuple[str, str]] = []

    def ask(self, question: str):
        """
        Ask a question using the RAG pipeline while
        preserving conversation history.
        """
        response, docs = self.pipeline.ask(
            question, chat_history=self._history
        )

        # Store conversation turns
        self._history.append(("user", question))
        self._history.append(("assistant", response.content))

        return response, docs

    def clear(self):
        """Clear the conversation history."""
        self._history.clear()

    def get_history(self) -> List[Tuple[str, str]]:
        """Return chat history as (role, message) pairs."""
        return list(self._history)
