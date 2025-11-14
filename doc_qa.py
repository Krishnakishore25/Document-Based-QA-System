import os
from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Function to extract text from PDF
#Opens the PDF File -> Loops through all pages -> Extracts text and concatenates into a single string
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to extract text from plain text file
# Reads plain text files directly
def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to extract text from document (PDF or TXT)
#Ensures only pdf or txt files are processed
def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format. Upload PDF or TXT.")

# Function to split text into chunks for embedding
def split_text(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create FAISS vector store from chunks
#Converts each text chunk into embeddings and stores them in FAISS, which allows fast similairty search
def create_faiss_index(chunks: List[str], embedding_model) -> FAISS:
    return FAISS.from_texts(chunks, embedding_model)

# Function to load FAISS index from disk
def load_faiss_index(file_path: str, embedding_model) -> FAISS:
    return FAISS.load_local(file_path, embedding_model)

# Function to save FAISS index to disk
def save_faiss_index(faiss_index: FAISS, file_path: str):
    faiss_index.save_local(file_path)

# Function to answer questions from document text using FAISS and GPT 3.5 Turbo
def answer_question(question: str, faiss_index: FAISS) -> str:
    # Get top relevant documents/chunks
    docs = faiss_index.similarity_search(question, k=3)
    # Load chat model
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # Load question answering chain
    qa_chain = load_qa_chain(chat, chain_type="stuff")
    # Run chain on retrieved docs
    answer = qa_chain.run(input_documents=docs, question=question)
    return answer
