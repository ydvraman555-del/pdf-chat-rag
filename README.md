# 📄 Chat with PDF

> A secure, open-source RAG chatbot that lets you query any text-based PDF using Google Gemini — deployed free on Render.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Visit_App-blue?style=for-the-badge)](https://pdf-chat-rag-w342.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

---

## 🎬 Demo

> **[▶️ Try it live](https://pdf-chat-rag-w342.onrender.com)** — Upload any PDF and start chatting!  
> *(Free tier — first load may take ~30s if the server is cold.)*

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
| 🔒 | **Secure by Design** | API keys via env vars only, magic-byte PDF validation, temp files auto-deleted |
| 💬 | **Chat History** | Full conversation memory within each session with a one-click clear button |
| 🔍 | **Debug Mode** | Toggle "Show retrieved chunks" to see exactly what the retriever found |
| 🚀 | **One-Click Deploy** | Push to GitHub → Render auto-deploys. Zero DevOps needed |
| 💰 | **100% Free** | Gemini free tier + Render free tier. No credit card required |

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
│                              │   Google Gemini Embedding    │    │
│                              │   API (gemini-embedding-001) │    │
│                              │   Text → 3072-D vectors      │    │
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
│               │   Google Gemini LLM      │                      │
│               │   (gemini-2.5-flash)     │                      │
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
| **LLM** | Gemini 2.5 Flash | Free tier, fast inference, strong instruction following |
| **Embeddings** | Gemini Embedding 001 | 3072-D vectors, free, same provider = no extra API keys |
| **Vector Store** | FAISS (in-memory) | Pre-built binaries, no C++ build tools, Render-compatible |
| **RAG Framework** | LangChain 0.3 | Industry standard, modular, well-documented |
| **Frontend** | Streamlit 1.45 | Python-native, chat UI built-in, fast prototyping |
| **Deployment** | Render (Free) | Auto-deploy from GitHub, env var support, zero DevOps |
| **PDF Parsing** | PyPDF | Pure Python, no system dependencies |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Google Gemini API key](https://aistudio.google.com/app/apikey) (free)

### Local Setup

```bash
# Clone
git clone https://github.com/ydvraman555-del/pdf-chat-rag.git
cd pdf-chat-rag

# Virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate          # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env → add your GOOGLE_API_KEY

# Run
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Deploy to Render (Free)

1. Fork this repo
2. Go to [render.com](https://render.com) → **New +** → **Web Service**
3. Connect your GitHub repo
4. Configure:

| Field | Value |
|-------|-------|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true` |
| Instance Type | Free |

5. Add environment variable: `GOOGLE_API_KEY` = your key
6. Click **Create Web Service** → Live in ~3 minutes!

---

## 🔒 Security

This project follows security best practices for API key management:

- ✅ API key loaded exclusively via environment variables (`os.getenv`)
- ✅ `.env` excluded in `.gitignore` — never committed to Git
- ✅ Key validated at startup — graceful error if missing
- ✅ Key scrubbed from all error messages before display
- ✅ Uploaded PDFs validated via magic bytes (not just file extension)
- ✅ Temp files deleted immediately after processing
- ✅ All API calls happen server-side — key never sent to the browser
- ✅ No `print()` or `st.write()` ever exposes the key

---

## ⚠️ Limitations

Being transparent about limitations shows engineering maturity:

| Limitation | Reason | Workaround |
|-----------|--------|------------|
| ❌ No scanned/image PDF support | PyPDF extracts text only, not OCR | Use Adobe Acrobat to OCR the PDF first |
| ❌ Tables & charts not understood | Chunking breaks table structure | Ask about the text surrounding the table |
| ❌ Can't summarize entire 200+ page docs | Only top 4 chunks retrieved per question | Ask specific questions instead |
| ❌ Limited multi-hop reasoning | Retriever finds locally similar chunks, not chains of logic | Break complex questions into simpler ones |
| ❌ Free tier cold starts | Render sleeps after 15 min idle | First load takes ~30s; use [UptimeRobot](https://uptimerobot.com) to keep alive |
| ✅ Best for | Text-heavy PDFs under 100 pages (research papers, reports, manuals) | |

---

## 📁 Project Structure

```
pdf-chat-rag/
├── app.py                    # Main Streamlit application (production)
├── requirements.txt          # Pinned dependencies
├── .env.example              # Template for API key setup
├── .gitignore                # Security: excludes .env, venv, etc.
├── .streamlit/
│   └── config.toml           # Streamlit server configuration
├── step2_chunking.py         # Learning: PDF chunking demo
├── step3_embeddings.py       # Learning: Embeddings + FAISS demo
├── step4_qa.py               # Learning: RAG QA chain demo
├── create_sample_pdf.py      # Utility: generates test PDF
└── test_setup.py             # Utility: validates environment setup
```

---

## 🧠 How RAG Works (For Recruiters & Non-Technical Readers)

**RAG = Retrieval-Augmented Generation**

Instead of asking an AI to answer from memory (which can hallucinate), RAG forces the AI to:

1. **Search** your specific document for relevant paragraphs
2. **Read** only those paragraphs
3. **Answer** based on what it read — citing page numbers
4. **Refuse** to answer if the information isn't in the document

This is the same architecture used by ChatGPT's "upload a file" feature, Notion AI, and enterprise document Q&A systems.

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- [ ] Add OCR support for scanned PDFs (using `pytesseract`)
- [ ] Support multiple file uploads
- [ ] Add conversation export (download chat as PDF)
- [ ] Implement streaming responses
- [ ] Add authentication for multi-user deployment
- [ ] Support DOCX and TXT uploads

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Raman Yadav**

[![GitHub](https://img.shields.io/badge/GitHub-ydvraman555--del-181717?style=flat&logo=github)](https://github.com/ydvraman555-del)

---

<p align="center">
  Built with ❤️ using LangChain, Google Gemini, and Streamlit
</p>
