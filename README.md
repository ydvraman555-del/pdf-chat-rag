# 📄 Chat with PDF (Hybrid Local/Cloud RAG)

> A secure, high-performance RAG chatbot that lets you query any text-based PDF using local Ollama embeddings and cloud-based Groq LLM API. 

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Visit_App-blue?style=for-the-badge)](https://pdf-chat-rag-w342.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local-000000?style=flat&logo=ollama&logoColor=white)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

---

## 🎬 Demo

> **[▶️ Try the live cloud demo on Render](https://pdf-chat-rag-w342.onrender.com)** — Upload any PDF and start chatting!  
> *(Runs in **Cloud Mode** using Google Gemini. Free tier — first load may take ~30s if the server is cold.)*

To run the application locally in **Local Mode** (100% private local embeddings via Ollama + lightning-fast Groq LLM API), follow the setup instructions below.

<!-- Add a screenshot or GIF here -->
<!-- ![Demo](assets/demo.gif) -->

**To add a demo GIF:** Record your screen using [ShareX](https://getsharex.com/) or [LICEcap](https://www.cockos.com/licecap/), save as `assets/demo.gif`, and uncomment the line above.

---

## ✨ Features

| | Feature | Description |
|---|---------|-------------|
| 🏠 | **Multi-Page Layout** | Features an informative landing/info page explaining the app and an interactive chat assistant page |
| 📄 | **PDF Upload + Chat** | Drag-and-drop any text-based PDF, then ask questions in a ChatGPT-style interface |
| ⚙️ | **RAG Hyperparameters** | Tune LLM Temperature (creativity) and Retrieval Top K (context size) dynamically via sidebar sliders |
| 📑 | **Page-Level Citations** | Every answer cites the exact page number(s) it came from — fully verifiable |
| 🛡️ | **Hallucination Prevention** | Strict prompt engineering forces the LLM to answer ONLY from PDF content |
| 🔒 | **Secure by Design** | Local embeddings with Ollama. API keys via env vars only, magic-byte PDF validation |
| 💬 | **Chat History** | Full conversation memory within each session with a one-click clear button |
| 🔍 | **Debug Mode** | Toggle "Show retrieved chunks" to see exactly what the retriever found in the vector index |
| ⚡ | **Hybrid Architecture** | Local `nomic-embed-text` embeddings + cloud `llama-3.3-70b-versatile` via Groq |
| 💰 | **100% Free** | Ollama runs locally + Groq free tier API. No credit card required |

---

## 🏗️ Architecture & Pipeline Steps

The RAG pipeline is broken down into **6 granular steps**, each represented by a dedicated learning script:

```
    [PDF File]
        │
        ▼ (Step 1: step1_loading.py)
 1. Document Loading (Extract raw text page-by-page)
        │
        ▼ (Step 2: step2_chunking.py)
 2. Document Chunking (Split pages into overlapping 1000-char chunks)
        │
        ▼ (Step 3: step3_embeddings.py)
 3. Vector Embeddings (Convert text to 768-D dense vectors locally)
        │
        ▼ (Step 4: step4_vector_store.py)
 4. Vector Storage (Index vectors in FAISS & persist to disk)
        │
        ▼ (Step 5: step5_retrieval.py)
 5. Context Retrieval (Query local FAISS via similarity search)
        │
        ▼ (Step 6: step6_generation.py)
 6. Answer Generation (Construct prompt & invoke ChatGroq LLM)
        │
        ▼
 [Grounded Answer + Citations]
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

---

## 📖 Walkthrough: Run the RAG Pipeline Step-by-Step

Anyone can understand the full pipeline by running these scripts in sequence:

1. **Generate the sample PDF**:
   ```bash
   python create_sample_pdf.py
   ```
2. **Step 1 - PDF Text Loading**:
   Extract and inspect raw text from page 1 and page 2.
   ```bash
   python step1_loading.py
   ```
3. **Step 2 - Document Chunking**:
   Split the loaded pages into overlapping text segments.
   ```bash
   python step2_chunking.py
   ```
4. **Step 3 - Vector Embeddings**:
   Convert a text string into a 768-D semantic vector locally.
   ```bash
   python step3_embeddings.py
   ```
5. **Step 4 - Vector Database persistence**:
   Index the document chunks and save the FAISS database to the local `faiss_index/` folder.
   ```bash
   python step4_vector_store.py
   ```
6. **Step 5 - Semantic Retrieval**:
   Load the saved index from disk and run similarity searches.
   ```bash
   python step5_retrieval.py
   ```
7. **Step 6 - LLM Generation**:
   Retrieve matching chunks, build the prompt, and get grounded answers from Groq.
   ```bash
   python step6_generation.py
   ```

### Run the Web Interface

Start the Streamlit web application:
```bash
streamlit run app.py
```
The interface opens at `http://localhost:8501`.

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
├── .gitignore                # Security: excludes .env, venv, and generated database index files
├── .streamlit/
│   └── config.toml           # Streamlit server and theme configurations
├── step1_loading.py          # Step 1: PDF loading and text extraction
├── step2_chunking.py         # Step 2: Document text chunking
├── step3_embeddings.py       # Step 3: Local vector embeddings generation
├── step4_vector_store.py      # Step 4: Vector index creation and local persistence
├── step5_retrieval.py         # Step 5: Similarity search retrieval
├── step6_generation.py       # Step 6: Grounded LLM Q&A answer generation
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
