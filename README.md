# 📄 PDF Chatbot

A **Streamlit-based chatbot** that allows you to upload PDF documents and ask questions interactively. The bot leverages **LangChain**, **FAISS** vector storage, and **OpenAI embeddings** (or OpenRouter LLM) to provide accurate answers using the content from your PDFs.

---

## Features

- Upload PDF documents and automatically **extract and chunk text**.
- Store **embeddings** in FAISS for fast semantic search.
- **Interactive chat interface** powered by LLMs.
- **Citations**: each answer references the source page from the PDF.
- **User-configurable chunk size and overlap** for embeddings.
- **Conversation history** retained for continuous interaction.
- Modular design for easy experimentation with different embedding and LLM models.

---

## Setup steps
1. Clone Repository & Install Dependencies
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt

2. Configure Environment Variables
Create a .streamlit/secrets.toml file (do not commit real keys):

    - OPENAI_API_KEY = "your-openai-api-key"
    - OPENROUTER_API_KEY = "your-openrouter-api-key"

3. Run the App
streamlit run app.py

---

### Chunking Strategy

- Splitter: RecursiveCharacterTextSplitter
- Chunk Size: Configurable (default: 1000 characters)
- Chunk Overlap: Configurable (default: 200 characters)
- Purpose: Prevents loss of context across chunk boundaries.
- Users can control chunk size and overlap through sidebar sliders for fine-tuning.

---

### Vector Database & Retrieval

- Vector DB: FAISS (Facebook AI Similarity Search)
- Retrieval Strategy: MMR (Maximal Marginal Relevance)
- Balances relevance & diversity.
- Avoids redundant chunks by retrieving complementary sections.
- Search Params: k=5 (top 5 chunks returned).

---

### Conversation Memory

- Maintained in st.session_state.history.
- Stores tuples of (role, message) for both user and assistant.
- Displayed in the UI:
- Sidebar → Compact chat history view.
- Main Area → Rich chat-style messages.
- Context from history is passed into the LLM to support multi-turn conversations.

---

### Known Limitations

- PDF Formatting: Complex PDFs (scanned images, tables, etc.) may not parse cleanly.
- Index Persistence: FAISS index is local; redeployment or restart requires rebuilding.
- Token Limits: Very large PDFs may exceed token window when too many chunks are retrieved.
- Free Models: The OpenRouter free LLM may be slower or rate-limited compared to paid APIs.
- No Fine-Grained Citations: Answers reference page numbers but not exact line positions.

---