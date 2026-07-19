# 🧠 brain.md — PDF Chat RAG Project Summary
> **Purpose:** Read this file first before any task. It contains the full project context so no file scanning is needed.
> **Last Updated:** 2026-07-09

---

## 📌 Project Overview

**Name:** PDF Chat RAG  
**Type:** Retrieval-Augmented Generation (RAG) web app  
**Goal:** Upload a PDF → ask questions → get grounded answers with page citations  
**UI:** Streamlit (dark glassmorphism design, animated gradient background)  
**Stack:** Python + LangChain + FAISS + Ollama (local embeddings) + Groq (LLM API)

---

## 🗂️ File Structure

```
pdf-chat-rag/
├── app.py                  # Main Streamlit UI app (production entry point)
├── step2_chunking.py       # Learning script: PDF loading + chunking demo
├── step3_embeddings.py     # Learning script: embeddings + FAISS vector store demo
├── step4_qa.py             # Learning script: full RAG Q&A pipeline demo
├── test_setup.py           # Verifies Ollama is running + required models are pulled
├── create_sample_pdf.py    # Utility: generates a sample.pdf for testing
├── sample.pdf              # Test PDF file (2.1 MB)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (GOOGLE_API_KEY, GROQ_API_KEY)
├── .env.example            # Template for .env
├── .gitignore              # Git ignore rules
├── .streamlit/             # Streamlit config directory
├── venv/                   # Python virtual environment
└── brain.md                # ← THIS FILE
```

---

## 🔑 Environment Variables (`.env`)

```env
GOOGLE_API_KEY=<google-api-key>      # Not actively used (kept for future use)
GROQ_API_KEY=<groq-api-key>          # Used by ChatGroq LLM in app.py and step4_qa.py
```

---

## 🧩 Tech Stack & Dependencies (`requirements.txt`)

| Package | Version | Purpose |
|---|---|---|
| `langchain` | 0.3.25 | Core RAG orchestration |
| `langchain-community` | 0.3.24 | PyPDFLoader, FAISS integration |
| `langchain-core` | 0.3.86 | ChatPromptTemplate, base types |
| `langchain-text-splitters` | 0.3.11 | RecursiveCharacterTextSplitter |
| `langchain-ollama` | 0.2.2 | OllamaEmbeddings (local embeddings) |
| `langchain-groq` | 0.2.0 | ChatGroq (cloud LLM via Groq API) |
| `langchain-google-genai` | 2.1.4 | Installed but not actively used |
| `faiss-cpu` | 1.14.2 | In-memory vector store |
| `pypdf` | 5.5.0 | PDF text extraction |
| `streamlit` | 1.45.1 | Web UI |
| `python-dotenv` | 1.1.0 | Load `.env` |

---

## ⚙️ RAG Pipeline (How It Works)

```
PDF Upload
    ↓
PyPDFLoader  →  pages (1 Document per page, 0-indexed metadata)
    ↓
RecursiveCharacterTextSplitter  →  chunks (size=1000 chars, overlap=200 chars)
    ↓
OllamaEmbeddings (nomic-embed-text, local)  →  dense vectors
    ↓
FAISS.from_documents()  →  in-memory vector index
    ↓
User Question
    ↓
similarity_search(question, k=4)  →  top 4 relevant chunks
    ↓
ChatPromptTemplate (system + human messages)
    ↓
ChatGroq (llama-3.3-70b-versatile, temperature=0)  →  grounded answer
    ↓
Answer + Source Page Numbers displayed in Streamlit UI
```

---

## 📄 File-by-File Details

### `app.py` — Main Production App
- **Entry point:** `streamlit run app.py`
- **Key functions:**
  - `inject_custom_css()` — Injects all CSS (glassmorphism, animated gradient, particles, etc.)
  - `build_vectorstore(file_bytes, api_key)` — `@st.cache_resource` cached; loads PDF → chunks → FAISS
  - `answer_question(vector_store, question, api_key)` — retrieves chunks, builds prompt, calls Groq
  - `validate_pdf(uploaded_file)` — checks file size (max 20MB) and PDF magic bytes
  - `trigger_confetti()` — JS canvas-confetti on first PDF upload
  - `stream_response(text)` — typewriter generator for `st.write_stream()`
  - `main()` — Streamlit app orchestrator
- **LLM model used:** `llama-3.3-70b-versatile` (via `langchain_groq.ChatGroq`)
- **Embedding model:** `nomic-embed-text` (via `langchain_ollama.OllamaEmbeddings`)
- **Max PDF size:** 20 MB
- **Retrieval k:** 4 chunks
- **Features:** Chat history, source page pills, debug mode (show retrieved chunks), clear chat button, confetti on upload

### `step4_qa.py` — Learning Script (Full RAG Pipeline)
- Standalone script demonstrating the full RAG pipeline
- Runs 3 test questions on `sample.pdf`
- **LLM model used:** `llama-3.3-70b-versatile` (via ChatGroq)
- **Embedding model:** `nomic-embed-text` (Ollama, local/offline)
- **Retrieval k:** 4 chunks

### `step2_chunking.py` — Learning Script (Chunking)
- Demonstrates PDF loading + chunking only (no embeddings/LLM)
- Uses `PyPDFLoader` + `RecursiveCharacterTextSplitter`
- Chunk size: 1000 chars, overlap: 200 chars
- Prints stats: pages, chunks, first chunk preview, 5th chunk preview

### `step3_embeddings.py` — Learning Script (Embeddings)
- Demonstrates embedding + FAISS vector store + similarity search
- Uses `nomic-embed-text` via Ollama (local, offline)
- Embedding dimensions: 768 (nomic-embed-text)
- Test query: "What is the main topic of this document?" → returns top 3 chunks

### `test_setup.py` — Setup Verifier
- Checks Ollama server is running at `http://localhost:11434`
- Verifies `gemma2:2b` and `nomic-embed-text` models are pulled
- Tests package imports (langchain, faiss, streamlit, etc.)
- Tests live generation with `gemma2:2b` and embeddings with `nomic-embed-text`

---

## 🖥️ Local Infrastructure Requirements

| Service | Requirement |
|---|---|
| **Ollama** | Must be running locally at `http://localhost:11434` |
| **nomic-embed-text** | Must be pulled: `ollama pull nomic-embed-text` |
| **gemma2:2b** | Used in `test_setup.py` only: `ollama pull gemma2:2b` |
| **Groq API Key** | Required for LLM answers (`GROQ_API_KEY` in `.env`) |

> **Note:** Embeddings are 100% local (Ollama). Only the LLM answer generation uses the cloud (Groq API).

---

## 🤖 LLM Configuration

| Setting | Value |
|---|---|
| **Provider** | Groq (cloud) |
| **Model** | `llama-3.3-70b-versatile` |
| **Temperature** | `0` (deterministic) |
| **Was deprecated:** | `llama-3.1-8b-instant` (migrated 2026-06-27, decommissioned Aug 16, 2026) |

---

## 🎨 UI Design (app.py)

- **Theme:** Dark mode glassmorphism
- **Background:** Animated gradient mesh (`#0a0a0f`, `#150f24`, `#0a1128`)
- **Accent colors:** Purple `#8b5cf6`, Blue `#3b82f6`, Pink `#ec4899`
- **Font:** Inter (Google Fonts)
- **Effects:** CSS particle system, floating animation, fade-in chat messages, typewriter response, confetti on PDF upload
- **Layout:** Centered, sidebar for file upload + settings

---

## 🚀 How to Run

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Make sure Ollama is running (separate terminal)
ollama serve

# 3. Run the app
streamlit run app.py

# OR run individual learning scripts
python step2_chunking.py
python step3_embeddings.py
python step4_qa.py

# Verify setup
python test_setup.py
```

---

## 🔒 Security Notes

- API keys are in `.env` — never printed or exposed
- `.gitignore` should exclude `.env` and `venv/`
- PDF validation: magic bytes check + 20MB size limit
- User data never leaves the machine for embeddings (Ollama is local)

---

## 📝 Known History / Changes

| Date | Change |
|---|---|
| 2026-07-09 | Updated and verified Ollama (nomic-embed-text) local embeddings and Groq LLM integration |
| 2026-06-27 | Migrated LLM from `llama-3.1-8b-instant` → `llama-3.3-70b-versatile` (Groq deprecation) |
| 2026-06-27 | Created `brain.md` for quick project context |
