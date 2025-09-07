# retriever.py
import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

@st.cache_resource(show_spinner=False)
def load_retriever(index_path, llm_model="openai/gpt-oss-20b:free"):
    """Load FAISS retriever + wrap with OpenRouter LLM"""
    # Embeddings
    openai_key = st.secrets.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    # LLM
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

    # Custom prompt template
    custom_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""You are a helpful AI assistant that answers questions based on the provided PDF document context.
Guidelines:
- Use only the information in context
- If unknown, state "I cannot find this information"
- Cite page numbers when possible
Previous conversation history:
{chat_history}
Context from the document:
{context}
Question: {question}
Answer: Based on the provided document, """
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    return qa_chain
