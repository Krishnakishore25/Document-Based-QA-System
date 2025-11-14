import streamlit as st
from doc_qa import extract_text, split_text, create_faiss_index, answer_question
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from io import BytesIO
import os

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

st.set_page_config(page_title="Chat with Document", layout="wide")
st.title("💬 Chat with Document - Senzmate")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

def read_uploaded_file(uploaded_file):
    """Extract text from PDF or TXT directly from BytesIO"""
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8")
    else:
        raise ValueError("Unsupported file format. Upload PDF or TXT.")

if uploaded_file:
    # Create FAISS index once
    if st.session_state.faiss_index is None:
        document_text = read_uploaded_file(uploaded_file)
        embedding_model = OpenAIEmbeddings()
        chunks = split_text(document_text)
        st.session_state.faiss_index = create_faiss_index(chunks, embedding_model)

    # --- Chat interface ---
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.markdown(chat["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(chat["content"])

    # Chat input
    question = st.chat_input("Ask a question about the document:")

    if question:
        # Display user question immediately
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.experimental_rerun()  # Rerun to show user message first

# --- Generate assistant answers separately ---
if st.session_state.chat_history:
    last_msg = st.session_state.chat_history[-1]
    if last_msg["role"] == "user" and (len(st.session_state.chat_history) < 2 or st.session_state.chat_history[-1]["role"] != "assistant"):
        with st.spinner("Generating answer..."):
            answer = answer_question(last_msg["content"], st.session_state.faiss_index)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.experimental_rerun()
else:
    st.info("Please upload a PDF or TXT document to start chatting.")
