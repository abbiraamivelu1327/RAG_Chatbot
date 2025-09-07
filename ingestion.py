# ingestion.py
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

@st.cache_resource(show_spinner=False)
def ingest_pdf(pdf_file, index_path, chunk_size=1000, chunk_overlap=200):
    """Load PDF → Chunk → Embed → Store in FAISS"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)

    # Split into chunks using user-defined size & overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    openai_key = st.secrets.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

    # Build FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index
    vectorstore.save_local(index_path)
    return vectorstore
