# 📚 Research Paper Q&A Assistant

A semantic search and question-answering system for research papers using **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**.

## 🚀 Overview

This system enables users to:

- Upload research papers (PDF)
- Ask natural language questions
- Receive concise, context-grounded answers with traceable sources

### 🔧 Built With:

- [LangChain](https://github.com/langchain-ai/langchain)
- [Hugging Face Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Pinecone](https://www.pinecone.io/)
- [Google Gemini 1.5 Pro](https://deepmind.google/technologies/gemini/)
- [Flask](https://flask.palletsprojects.com/)

---

## 🧠 Features

- 📄 PDF parsing with text cleaning
- 🔍 Section segmentation & semantic chunking
- 🧬 Embedding generation with `all-MiniLM-L6-v2`
- 📦 Vector indexing with Pinecone
- 🤖 Contextual answer generation using Gemini 1.5 Pro
- 🌐 Web UI for upload and Q&A
- ✅ Transparent answers with source document snippets

---

## 🛠️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/ajay-del-bot/research_paper_RAG_chain.git
cd research_paper_RAG_chain
```

2. **Create and activate a virtual environment**

```bash
# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables in `.env`**

```env
PINECONE_API_KEY=YOUR_PINNECONE_API_KEY
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
INDEX_NAME='test-db'
```

### 💻 Running the App

```
python3 src/server.py
```

## 🧪 Future Enhancements

- Support for tables, figures, equations
- Better layout handling for multi-column PDFs
- User authentication & session history
- Integration with multiple LLMs
