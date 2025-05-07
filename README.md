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

## 🛠️ Installation

```
git clone https://github.com/your-username/research-paper-qa-assistant.git
cd research-paper-qa-assistant
pip install -r requirements.txt
```
Note : Make sure you have Python and pip installed.

### Set up environment variables in .env:
```
PINECONE_API_KEY=your_pinecone_key
GOOGLE_API_KEY=your_google_generative_ai_key
``` 

## 💻 Running the App
python app.py


## 🧪 Future Enhancements
- Support for tables, figures, equations
- Better layout handling for multi-column PDFs
- User authentication & session history
- Integration with multiple LLMs



MIT License - see the LICENSE file for details.