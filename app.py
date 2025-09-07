import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


# ---------------- PDF ingestion ---------------- #
def ingest_pdf(pdf_file, index_path):
    """Load PDF → Chunk → Embed → Store in FAISS"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    # Clean up temporary file
    os.unlink(tmp_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # OpenAI Embeddings
    openai_key = st.secrets.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

    # Build FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index
    vectorstore.save_local(index_path)
    return vectorstore

# ---------------- Retrieval + LLM ---------------- #
def load_retriever(index_path, llm_model="openai/gpt-oss-20b:free"):
    """Load FAISS retriever + wrap with OpenRouter LLM"""
    # Use OpenAI embeddings (same as used during indexing)
    openai_key = st.secrets.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
        
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    # OpenRouter LLM
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY")
    llm = ChatOpenAI(
        model=llm_model,
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=1000,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "LangChain PDF Chat"
        }
    )

    # Conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ---------------- Streamlit UI ---------------- #
def main():
    st.set_page_config(page_title="LangChain PDF Chat", layout="wide")
    st.title("📄 Chat with your PDF (LangChain)")

    # API Key setup
    st.sidebar.header("🔑 API Configuration")
    
    # Check for API keys (environment variables or Streamlit secrets)
    openai_key = st.secrets.get("OPENAI_API_KEY")
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY")
    
    if not openai_key:
        st.sidebar.warning("⚠️ OPENAI_API_KEY not found")
        st.sidebar.info("💡 Configure in Streamlit Cloud secrets or .env file")
    else:
        st.sidebar.success("✅ OpenAI API key configured")
        
    if not openrouter_key:
        st.sidebar.warning("⚠️ OPENROUTER_API_KEY not found")
        st.sidebar.info("💡 Configure in Streamlit Cloud secrets or .env file")
    else:
        st.sidebar.success("✅ OpenRouter API key configured")

    # Model selection
    st.subheader("🔧 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Embedding Model** (for document search)")
        st.info("🔥 **OpenAI text-embedding-3-small** - High quality embeddings")
        st.caption("💡 Fast and accurate semantic search")
    
    with col2:
        st.write("**LLM Models** (for generating answers) - 🆓 Free!")
        llm_model = st.selectbox(
            "Select LLM (for answers)",
            [
                "openai/gpt-oss-20b:free"
            ]
        )
        st.caption("💰 This model is completely free on OpenRouter!")

    if "history" not in st.session_state:
        st.session_state.history = []

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        current_dir = os.getcwd()
        index_path = os.path.join(current_dir, "storage", "faiss_openai_embeddings")
        if not os.path.exists(index_path):
            os.makedirs(index_path)

        
        if st.button("Build Index"):
            if not openai_key:
                st.error("❌ Please set OPENAI_API_KEY in your environment variables")
                return
            with st.spinner("Indexing..."):
                ingest_pdf(pdf_file, index_path)
            st.success("✅ PDF indexed")

        if not openrouter_key:
            st.error("❌ Please set OPENROUTER_API_KEY in your environment variables to chat")
            st.info("💡 Create a .env file with: `OPENROUTER_API_KEY=your_api_key_here`")
            return

        qa_chain = load_retriever(index_path, llm_model)

        user_q = st.chat_input("Ask a question about the PDF...")
        if user_q:
            st.session_state.history.append(("user", user_q))
            result = qa_chain.invoke({"question": user_q, "chat_history": st.session_state.history})

            # Store assistant response
            st.session_state.history.append(("assistant", result["answer"]))

            # Render history
            for role, msg in st.session_state.history:
                with st.chat_message(role):
                    st.write(msg)

            # Citations
            with st.expander("Citations"):
                for doc in result["source_documents"]:
                    st.markdown(f"- **p.{doc.metadata['page']}** — {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()
