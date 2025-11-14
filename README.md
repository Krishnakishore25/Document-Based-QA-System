# Document-Based-QA-System
This project is an interactive document-based question answering system that allows users to upload PDF or TXT files and ask questions directly about the content. The system uses OpenAI GPT-3.5 for natural language understanding and FAISS for fast retrieval of relevant text chunks from large documents.

## Features 
- Upload PDF or plain text documents.
- Extract and process document text automatically.
- Conversational chatbot interface with persistent chat history.
- Contextual question answering using document embeddings and GPT-3.5 Turbo.
- Easy-to-use web interface with Streamlit.

# Technology Stack
- Python
- Streamlit for UI
- PyPDF2 for PDF text extraction
- LangChain for text processing, embeddings, and QA
- FAISS for fast vector similarity search
- OpenAI API for embeddings and GPT-3.5 Turbo

# INSTALLATION

1. Clone this repository:

```bash
git clone https://github.com/Krishnakishore25/Document-Based-QA-System.git
cd DocumentBasedChatBot
code .
````

2. Create and activate a virtual environment:
# Create virtual environment (macOS/Linux/Windows)
```bash
python -m venv venv

# Activate virtual environment

# macOS/Linux
source venv/bin/activate


# Windows (Command Prompt)
venv\Scripts\activate


# Windows (PowerShell)
venv\Scripts\Activate.ps1
````

3. Install Dependencies
```bash
pip install -r requirements.txt
````

4. Add your OpenAI key
by creating a .streamlit folder in the project root and add a secrets.toml file with:
```bash
OPENAI_API_KEY = "your_openai_api_key_here"
````

5.Run the Application
```bash
streamlit run app.py
````

   

