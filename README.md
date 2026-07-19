# 📄 Chat with PDF (Hybrid Local/Cloud RAG)

> A secure, high-performance RAG chatbot that lets you query any text-based PDF using local Ollama embeddings and cloud-based Groq LLM API. 

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local-000000?style=flat&logo=ollama&logoColor=white)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

---

## 🎬 Demo

> **Upload any PDF and start chatting!**  
> *(No data leaves your computer for embeddings. The LLM runs via Groq API for lightning-fast inference.)*

<!-- Add a screenshot or GIF here -->
<!-- ![Demo](assets/demo.gif) -->

**To add a demo GIF:** Record your screen using [ShareX](https://getsharex.com/) or [LICEcap](https://www.cockos.com/licecap/), save as `assets/demo.gif`, and uncomment the line above.

---

## ✨ Features

| | Feature | Description |
|---|---------|-------------|
| 📄 | **PDF Upload + Chat** | Drag-and-drop any text-based PDF, then ask questions in a ChatGPT-style interface |
| 📑 | **Page-Level Citations** | Every answer cites the exact page number(s) it came from — fully verifiable |
| 🛡️ | **Hallucination Prevention** | Strict prompt engineering forces the LLM to answer ONLY from PDF content |
| 🔒 | **Secure by Design** | Local embeddings with Ollama. API keys via env vars only, magic-byte PDF validation |
| 💬 | **Chat History** | Full conversation memory within each session with a one-click clear button |
| 🔍 | **Debug Mode** | Toggle "Show retrieved chunks" to see exactly what the retriever found in the vector index |
| ⚡ | **Hybrid Architecture** | Local `nomic-embed-text` embeddings + cloud `llama-3.3-70b-versatile` via Groq |
| 💰 | **100% Free** | Ollama runs locally + Groq free tier API. No credit card required |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER (Browser)                          │
│                     Streamlit Chat Interface                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Upload PDF / Ask Question
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT SERVER                           │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │  PDF Upload  │───▶│   PyPDFLoader │───▶│ Text Splitter     │   │
│  │  Validation  │    │   (Extract)   │    │ (1000 char chunks)│   │
│  │ (Magic bytes)│    └──────────────┘    └────────┬──────────┘   │
│  └─────────────┘                                  │              │
│                                                   ▼              │
│                              ┌─────────────────────────────┐    │
│                              │   Ollama Local Embedding     │    │
│                              │   (nomic-embed-text)         │    │
│                              │   Text → 768-D vectors       │    │
│                              └──────────────┬──────────────┘    │
│                                             │                    │
│                                             ▼                    │
│  ┌──────────────┐           ┌──────────────────────────┐        │
│  │  User Query   │──embed──▶│    FAISS Vector Store     │        │
│  └──────┬───────┘           │    (In-Memory Index)      │        │
│         │                   │    Similarity Search (k=4) │        │
│         │                   └──────────────┬───────────┘        │
│         │                                  │ Top 4 chunks       │
│         ▼                                  ▼                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              PROMPT TEMPLATE (Strict RAG)                │    │
│  │  System: "Answer ONLY from context. Cite pages.          │    │
│  │           Say 'I couldn't find that' if not present."    │    │
│  │  Context: [Page 2]: chunk... [Page 5]: chunk...          │    │
│  │  Human: {user_question}                                  │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│               ┌──────────────────────────┐                      │
│               │   Groq API Cloud LLM     │                      │
│               │   (llama-3.3-70b)        │                      │
│               │   temperature=0          │                      │
│               └──────────────┬───────────┘                      │
│                              │                                   │
│                              ▼                                   │
│                 ┌────────────────────────┐                       │
│                 │  Answer + Page Sources  │                       │
│                 └────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Tool | Why This Choice |
|-----------|------|-----------------|
| **LLM** | `llama-3.3-70b-versatile` | Ultra-fast inference, high instruction following, free tier via Groq |
| **Embeddings** | `nomic-embed-text` | 768-D local embeddings, 100% private, no cloud latency or cost |
| **Vector Store** | FAISS (in-memory) | Pre-built binaries, no C++ build tools, high-performance local index |
| **RAG Framework** | LangChain 0.3 | Industry standard, modular, well-documented |
| **Frontend** | Streamlit 1.45 | Python-native, sleek glassmorphism UI, fast prototyping |
| **PDF Parsing** | PyPDF | Pure Python, no system dependencies |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- [Groq API key](https://console.groq.com/) (free)

### Local Infrastructure Setup

1. **Start Ollama** and pull the required embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```
2. *(Optional)* Pull the verification model used in the setup script:
   ```bash
   ollama pull gemma2:2b
   ```

### Application Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/ydvraman555-del/pdf-chat-rag.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pdf-chat-rag
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```
4. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
5. Set up your environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and add your Groq API Key:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     ```
6. **Verify the environment**:
   Run the verification script to test imports, local Ollama connections, and Groq LLM API connectivity:
   ```bash
   python test_setup.py
   ```
7. Start the application:
   ```bash
   streamlit run app.py
   ```

The application will open in your browser at `http://localhost:8501`.

---

## 🔒 Security & Privacy

- **Data Privacy:** 100% of the PDF text is processed and embedded locally on your computer via Ollama. No raw document content or embeddings are uploaded to the cloud for indexing.
- **Secure Key Management:** API keys are loaded via environment variables and are never hardcoded. `.env` is explicitly added to `.gitignore`.
- **Memory Storage:** Uploaded PDFs are processed in-memory. The FAISS vector database exists only in RAM and is destroyed when the Streamlit session closes.
- **Safety Safeguards:** Exception messages automatically redact API key values, preventing secret leaks in logs.

---

## ⚠️ Limitations

- **No Scanned PDFs:** PyPDF extracts text only, not OCR (optical character recognition).
- **Complex Tables & Charts:** Simple chunking can break structural layouts of tables and charts.
- **Hardware Bound:** Embedding performance is bound by your local hardware capabilities.

---

## 📁 Project Structure

```
pdf-chat-rag/
├── app.py                    # Main Streamlit application (production UI)
├── requirements.txt          # Python packages (langchain, streamlit, faiss, etc.)
├── .env.example              # Template for API keys
├── .gitignore                # Security: excludes .env, venv, and generated PDFs
├── .streamlit/
│   └── config.toml           # Streamlit server and theme configurations
├── step2_chunking.py         # Learning Script: PDF chunking demo
├── step3_embeddings.py       # Learning Script: Local embedding + FAISS index demo
├── step4_qa.py               # Learning Script: Full RAG pipeline CLI demo
├── create_sample_pdf.py      # Utility: generates sample.pdf for testing
├── test_setup.py             # Utility: verifies Ollama, models, and Groq API
├── brain.md                  # Reference: Quick summary of overall project details
└── rag_pipeline_explained.md # Reference: Detailed explanation of the architecture
```

---

## 👨‍💻 Author

**Raman Yadav**

[![GitHub](https://img.shields.io/badge/GitHub-ydvraman555--del-181717?style=flat&logo=github)](https://github.com/ydvraman555-del)

---

<p align="center">
  Built with ❤️ using LangChain, Ollama, Groq, and Streamlit
</p>
