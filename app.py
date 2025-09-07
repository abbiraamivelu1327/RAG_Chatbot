import os
import tempfile
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


# ---------------- PDF ingestion ---------------- #
@st.cache_resource(show_spinner=False)
def ingest_pdf(pdf_file, index_path):
    """Load PDF → Chunk → Embed → Store in FAISS"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    openai_key = st.secrets.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore


# ---------------- Retrieval + LLM ---------------- #
@st.cache_resource(show_spinner=False)
def load_retriever(index_path, llm_model="openai/gpt-oss-20b:free"):
    openai_key = st.secrets.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

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

    custom_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""You are a helpful AI assistant that answers questions based on the provided PDF document context.

    Guidelines:
    - Use only the information provided in the context below
    - If the answer is not in the context, clearly state "I cannot find this information in the provided document"
    - Be specific and cite relevant details from the document
    - Mention page numbers when possible
    - Maintain a professional and helpful tone

    Previous conversation history:
    {chat_history}

    Context from the document:
    {context}

    Question: {question}

    Answer:"""
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    return qa_chain


# ---------------- Streamlit UI ---------------- #
def main():
    st.set_page_config(page_title="LangChain PDF Chat", layout="wide")
    st.title("📄 Chat with your PDF (LangChain)")

    # Session state for chat persistence
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}  # {session_name: [(role, msg), ...]}
    if "active_session" not in st.session_state:
        st.session_state.active_session = None

    # Sidebar: Conversation Management
    st.sidebar.header("💬 Conversations")
    new_name = st.sidebar.text_input("New conversation name")
    if st.sidebar.button("➕ Create") and new_name:
        st.session_state.chat_sessions[new_name] = []
        st.session_state.active_session = new_name

    if st.session_state.chat_sessions:
        chosen = st.sidebar.radio(
            "Select conversation:",
            list(st.session_state.chat_sessions.keys()),
            index=list(st.session_state.chat_sessions.keys()).index(st.session_state.active_session)
            if st.session_state.active_session else 0
        )
        st.session_state.active_session = chosen

        # Export chat option
        if st.sidebar.button("⬇️ Download JSON"):
            data = st.session_state.chat_sessions[st.session_state.active_session]
            st.sidebar.download_button(
                "Save Chat",
                json.dumps(data, indent=2),
                file_name=f"{st.session_state.active_session}.json"
            )

    # File upload & indexing
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        index_path = os.path.join(os.getcwd(), "storage", "faiss_openai_embeddings")
        os.makedirs(index_path, exist_ok=True)

        if st.button("⚡ Build Index"):
            with st.spinner("Indexing PDF..."):
                ingest_pdf(pdf_file, index_path)
            st.success("✅ Index built")

        index_file = os.path.join(index_path, "index.faiss")
        if os.path.exists(index_file) and st.session_state.active_session:
            qa_chain = load_retriever(index_path)

            user_q = st.chat_input("Ask a question about the PDF...")
            if user_q:
                st.session_state.chat_sessions[st.session_state.active_session].append(("user", user_q))
                result = qa_chain.invoke({
                    "question": user_q,
                    "chat_history": st.session_state.chat_sessions[st.session_state.active_session]
                })
                st.session_state.chat_sessions[st.session_state.active_session].append(("assistant", result["answer"]))

            # Render chat history
            for role, msg in st.session_state.chat_sessions[st.session_state.active_session]:
                with st.chat_message(role):
                    st.write(msg)

            # Show citations
            if user_q and result.get("source_documents"):
                with st.expander("📚 Citations"):
                    for doc in result["source_documents"]:
                        st.markdown(f"- **p.{doc.metadata.get('page', '?')}** — {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
