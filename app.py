# app.py
import os
import streamlit as st
from ingestion import ingest_pdf
from retriever import load_retriever

def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("📄 Chat with your PDF")

    # ---------------- API Keys ---------------- #
    st.sidebar.header("🔑 API Configuration")
    openai_key = st.secrets.get("OPENAI_API_KEY")
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY")

    if not openai_key:
        st.sidebar.warning("⚠️ OPENAI_API_KEY not found")
    else:
        st.sidebar.success("✅ OpenAI API key configured")

    if not openrouter_key:
        st.sidebar.warning("⚠️ OPENROUTER_API_KEY not found")
    else:
        st.sidebar.success("✅ OpenRouter API key configured")

    # ---------------- Model & Chunk Settings ---------------- #
    st.subheader("🔧 Model & Chunk Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Embedding Model**")
        st.info("🔥 OpenAI text-embedding-3-small")
    with col2:
        st.write("**LLM Model**")
        llm_model = st.selectbox("Select LLM", ["openai/gpt-oss-20b:free"])

    # Chunk sliders
    st.sidebar.subheader("⚙️ Chunk Settings")
    chunk_size = st.sidebar.slider("Chunk size", 100, 2000, 1000, 50)
    chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, 200, 50)

    if "history" not in st.session_state:
        st.session_state.history = []

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        current_dir = os.getcwd()
        index_path = os.path.join(current_dir, "storage", "faiss_openai_embeddings")
        if not os.path.exists(index_path):
            os.makedirs(index_path)

        if st.button("Build Index"):
            with st.spinner("Indexing..."):
                ingest_pdf(pdf_file, index_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.success("✅ PDF indexed")

        index_file = os.path.join(index_path, "index.faiss")
        if os.path.exists(index_file):
            qa_chain = load_retriever(index_path, llm_model)
            user_q = st.chat_input("Ask a question about the PDF...")
            if user_q:
                st.session_state.history.append(("user", user_q))
                result = qa_chain.invoke({"question": user_q, "chat_history": st.session_state.history})
                st.session_state.history.append(("assistant", result["answer"]))

            # Display chat
            st.sidebar.subheader("💬 Chat History")
            for role, msg in st.session_state.history:
                st.sidebar.write(f"**{role}**: {msg}")

            # Render chat in main area
            for role, msg in st.session_state.history:
                with st.chat_message(role):
                    st.write(msg)

            # Citations
            with st.expander("Citations"):
                for doc in result["source_documents"]:
                    st.markdown(f"- **p.{doc.metadata['page']}** — {doc.page_content[:200]}...")

        else:
            st.warning("⚠️ Please click **Build Index** after uploading your PDF.")

if __name__ == "__main__":
    main()
