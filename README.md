# ThinkPal AI – Document Knowledge Assistant

ThinkPal AI is a Retrieval-Augmented Generation (RAG) based application that enables users to query private document collections using natural language. It retrieves relevant information from documents and generates grounded answers with source references.

---

## 🚀 Features

- 📄 Query documents using natural language
- 🤖 Context-aware answers powered by IBM Granite LLM
- 📚 Retrieval-Augmented Generation (RAG) pipeline
- 🔍 Displays relevant document references for transparency
- 💬 Multi-turn conversational support

---

## 🧠 How It Works

1. Documents are loaded from the `data/documents/` directory  
2. Text is split into smaller chunks  
3. Chunks are converted into embeddings using IBM Granite  
4. Stored in a Chroma vector database  
5. User query retrieves relevant chunks  
6. LLM generates an answer using retrieved context  

---

## 🏗️ Project Structure


rag-granite-assistant/
│
├── app/
│ ├── config.py # Model + environment configuration
│ ├── loaders.py # Document loaders (PDF, TXT, DOCX)
│ ├── embeddings_store.py # Vector store creation (Chroma)
│ ├── rag_pipeline.py # Core RAG logic
│ ├── chat_session.py # Chat history management
│
├── data/
│ ├── documents/ # Input documents
│ ├── vector_store/ # Persisted embeddings
│
├── ingest.py # Document ingestion script
├── check_vector_store.py # Debug / verification script
├── web_app.py # Streamlit UI
├── requirements.txt


---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/thinkpal-ai-document-assistant.git
cd thinkpal-ai-document-assistant
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Set environment variables

Create a .env file:

WATSONX_API_KEY=your_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
5. Add documents

Place your files inside:

data/documents/
6. Run ingestion
python ingest.py
7. Launch the app
streamlit run web_app.py
💡 Use Cases

Document search and summarization

Knowledge base exploration

Research assistance

Internal document querying

🔮 Future Improvements

Multi-user authentication system

Role-based document access

File upload via UI

Enhanced retrieval (hybrid search)

Deployment as a web service

📜 License

This project is licensed under the MIT License.

👩‍💻 Author

Shatakshi Tiwari
