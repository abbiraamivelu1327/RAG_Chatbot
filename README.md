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