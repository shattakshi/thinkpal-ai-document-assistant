# web_app.py

import streamlit as st
from langchain_core.messages import BaseMessage

from app.config import get_llm, get_embeddings
from app.embeddings_store import get_or_create_vector_store
from app.rag_pipeline import RAGPipeline
from app.chat_session import ChatSession


# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="ThinkPal AI", layout="wide")


# ------------------ HEADER ------------------
st.markdown(
    "<h2 style='text-align:center;'>ThinkPal AI</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:gray;'>Internal Knowledge Assistant</p>",
    unsafe_allow_html=True
)

st.markdown("---")


# ------------------ INITIALIZE ------------------
if "session" not in st.session_state:
    llm = get_llm()
    embeddings = get_embeddings()
    vectorstore = get_or_create_vector_store(embeddings)

    pipeline = RAGPipeline(llm=llm, vectorstore=vectorstore)
    st.session_state.session = ChatSession(pipeline)

if "chat" not in st.session_state:
    st.session_state.chat = []


# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("ThinkPal AI")

    if st.button("New Chat"):
        st.session_state.chat = []
        st.session_state.session.clear()

    st.markdown("---")
    st.caption("Internal AI Assistant")


# ------------------ CHAT DISPLAY ------------------
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)


# ------------------ INPUT ------------------
user_input = st.chat_input("Ask about company documents...")


if user_input:

    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.chat.append(("user", user_input))

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):

            response, sources = st.session_state.session.ask(user_input)

            if isinstance(response, BaseMessage):
                answer_text = response.content
            else:
                answer_text = str(response)

            # Answer
            st.markdown("### Answer")
            st.write(answer_text)

            # References (hidden)
            if sources:
                with st.expander("References"):
                    for i, doc in enumerate(sources, start=1):
                        src = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "?")

                        st.markdown(f"**Reference {i}**")
                        st.caption(f"{src} (page {page})")
                        st.write(doc.page_content[:300] + "...")
                        st.markdown("---")

    st.session_state.chat.append(("assistant", answer_text))