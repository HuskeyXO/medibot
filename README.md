# MediBot

MediBot is a **Retrieval-Augmented Generation (RAG)** based chatbot that allows users to ask questions about medical PDF documents. It retrieves relevant information from the documents using semantic search and uses an LLM to generate context-aware answers.

## Features

* Ask questions about medical PDF documents
* PDF text extraction and chunking
* Semantic search using vector embeddings
* FAISS-based vector storage and retrieval
* LLM-powered answer generation
* Source document visibility
* Interactive Streamlit interface

## Tech Stack

* **Language:** Python
* **LLM:** Hugging Face
* **Framework:** LangChain
* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS
* **Document Processing:** PyPDFLoader
* **Frontend:** Streamlit
* **Architecture:** Retrieval-Augmented Generation (RAG)

## Architecture

```text
Medical PDFs
     ↓
PDF Loader
     ↓
Text Chunking
     ↓
Embeddings
     ↓
FAISS Vector Store
     ↓
User Query
     ↓
Relevant Document Retrieval
     ↓
LLM
     ↓
Generated Answer + Sources
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd MediBot
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your Hugging Face API token:

```env
HF_TOKEN=your_huggingface_token
```

## Run

Start the Streamlit application:

```bash
streamlit run app.py
```

## Disclaimer

MediBot is an educational project and should not be used for medical diagnosis or treatment. Always verify medical information with qualified healthcare professionals.
