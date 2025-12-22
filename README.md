# Mental Health Chatbot (RAG-based)

## Introduction

This project is a Retrieval-Augmented Generation (RAG) based mental health chatbot.  
The system retrieves relevant information from PDF documents and uses a large language model to generate context-aware and supportive responses.  
The chatbot is intended for informational and supportive use only and does not replace professional mental health services.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/nnkhanhduy/mh-chatbot.git
cd mh-chatbot
```
### 2. Install dependencies:

```bash
pip install -r requirement.txt
```
### 3. Configure API key
```bash
GROQ_API_KEY="your_api_key_here"
```

### 4. Run
```bash
python main.py
```
## Technologies Used

- **Python** – Core programming language  
- **LangChain** – Framework for building RAG pipelines and LLM applications  
- **Groq (LLaMA 3.3)** – Large language model used for response generation  
- **HuggingFace Sentence Transformers** – Text embedding generation  
- **ChromaDB** – Vector database for semantic search and retrieval  
- **Gradio** – Web-based interface for interactive chatbot deployment  


