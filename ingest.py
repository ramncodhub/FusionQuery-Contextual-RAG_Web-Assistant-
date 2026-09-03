import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

FAISS_DB_DIR = "./faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@st.cache_resource
def get_embedding_model():
    """Cache open-source HuggingFace embedding model in memory."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def clear_vectorstore():
    """Deletes the FAISS index directory directly."""
    if os.path.exists(FAISS_DB_DIR):
        try:
            shutil.rmtree(FAISS_DB_DIR)
        except Exception as e:
            st.error(f"Error clearing vector store: {e}")


def process_and_store_document(file_path: str) -> FAISS:
    """Loads a PDF, clears old vectors, splits into dense chunks,
    embeds them, and saves to a local FAISS index.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    clear_vectorstore()

    # 1. Load document
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2. Dense chunking: 500 chars with 80 overlap isolates specific facts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)

    # 3. Create embeddings & persist locally in FAISS
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_DB_DIR)

    return vectorstore


def get_vectorstore() -> FAISS | None:
    """Loads existing FAISS vector store from disk if present."""
    if not os.path.exists(FAISS_DB_DIR):
        return None

    embeddings = get_embedding_model()
    return FAISS.load_local(
        FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True
    )
