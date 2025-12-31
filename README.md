# 🌿 Mental Health Chatbot (RAG-based)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://langchain.com/)

A modern, empathetic, and context-aware mental health support chatbot. Built using **Retrieval-Augmented Generation (RAG)**, this assistant combines curated psychological insights with advanced Large Language Models to provide a calming space for reflection and support.

---

## Key Features

-   **Context-Aware Memory**: Remembers previous parts of the conversation for seamless, natural dialogue.
-   **RAG Integration**: Retrieves high-quality, relevant information from curated mental health PDFs (via ChromaDB).
-   **Multi-modal Support**: 
    -   **Text Chat**: Clean, responsive messaging interface.
    -   **Voice Chat**: Real-time Speech-to-Text (Whisper) and Text-to-Speech (gTTS).
-   **High Performance**: Powered by Groq (LLaMA 3.3) for near-instant responses.

---

## Technology Stack

-   **Language**: Python
-   **Orchestration**: LangChain
-   **LLM**: Groq (Llama 3.3-70B)
-   **Embeddings**: HuggingFace (MiniLM-L6)
-   **Vector Store**: ChromaDB
-   **Voice processing**: OpenAI Whisper (STT) & gTTS (TTS)
-   **UI Framework**: Gradio + Custom CSS

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/nnkhanhduy/mh-chatbot.git
cd mh-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

> [!NOTE]
> **System Dependency**: For Voice Chat (Whisper), you may need to install **ffmpeg** on your system.
> - **Windows**: `choco install ffmpeg`
> - **macOS**: `brew install ffmpeg`
> - **Linux**: `sudo apt install ffmpeg`

### 4. Run the Application
```bash
python main.py
```

---

## Project Structure

```text
├── chroma_db/      # Persisted vector database
├── data/           # PDF source documents
├── src/            # Core logic (LLM, RAG, Voice, DB)
├── styles/         # Custom UI enhancements (CSS)
├── main.py         # Application entry point
└── requirements.txt
```

---

## ⚠️ Disclaimer

**This chatbot is for informational and emotional support purposes only.** 
It is **not** a substitute for professional mental health care, diagnosis, or treatment. If you are in crisis or experiencing a psychological emergency, please contact your local emergency services or a qualified mental health professional immediately.


