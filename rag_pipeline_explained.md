# 🛠️ PDF Chat RAG Pipeline - How It Works

This document explains the step-by-step architecture of the Retrieval-Augmented Generation (RAG) pipeline implemented in this project, covering both local execution (on your laptop) and cloud execution (when deployed to Render).

---

## 🗺️ Architectural Workflow Diagram

```
[Uploaded PDF]
      │
      ▼
1. Parsing (PyPDFLoader)
      │
      ▼
2. Chunking (RecursiveCharacterTextSplitter)
      │
      ▼
3. Embedding (Ollama nomic-embed-text / Gemini Cloud)
      │
      ▼
4. Vector Storage (In-Memory FAISS Database)
      │
      ├───────────────────────────────┐
      ▼ (When User Asks Question)      │
5. Conversational Memory Check         │
      │                               │
      ▼ (If prior history exists)      │
6. Query Condensation (Groq Llama 3)   │
      │                               │
      ▼ (Condensed Query)              ▼ (Original Query if no history)
7. Semantic Search (FAISS retrieval, k=4)
      │
      ▼
8. Context Injection (Prompt Template)
      │
      ▼
9. Generation (Groq Llama 3.3 70B Cloud)
      │
      ▼
[Grounded Answer + Page Citations]
```

---

## 🔍 Detailed Step-by-Step Breakdown

### 1. Document Parsing (PyPDFLoader)
* **Goal:** Convert raw PDF binary bytes into plain text that Python can process.
* **How it works:** 
  - The application reads the uploaded PDF and saves it temporarily to disk.
  - It loads the file using LangChain's `PyPDFLoader` (which uses the `pypdf` engine).
  - The document is divided page-by-page. Each page is converted into a LangChain `Document` object containing:
    * `page_content`: The extracted text of that page.
    * `metadata`: E.g., `{"source": "temp_uploaded.pdf", "page": 0}` (0-indexed).

### 2. Text Chunking (RecursiveCharacterTextSplitter)
* **Goal:** Break the large document text into small, manageable pieces (chunks) so that:
  1. Semantic search can identify very specific passages.
  2. The text fits within the LLM's maximum prompt window (context limit).
* **How it works:**
  - The project uses `RecursiveCharacterTextSplitter`.
  - **Parameters:**
    * `chunk_size = 1000`: Each text chunk will be maximum 1000 characters long (roughly 150-200 words).
    * `chunk_overlap = 200`: Successive chunks overlap by 200 characters. This ensures that words or concepts split across chunk boundaries are not lost.
  - The splitter tries to divide the document at logical separators: paragraphs (`\n\n`), sentences (`\n` or `.`), words (` `), and finally individual characters.

### 3. Vector Embeddings
* **Goal:** Convert textual chunks into numerical vectors (coordinate points in multi-dimensional space) that represent the semantic meaning of the text.
* **How it works:**
  - Text is passed to an embedding model, which returns a vector (list of floating-point numbers).
  - **Laptop/Local Mode:** Uses **Ollama** running `nomic-embed-text` locally. It produces **768-dimensional vectors**.
  - **Render/Cloud Mode:** Uses **Google Gemini** `models/gemini-embedding-001`. It produces **3072-dimensional vectors**.
  - Text passages that talk about similar concepts (e.g., "economy" and "finance") are given coordinates that are close to each other.

### 4. Vector Database Storage (FAISS)
* **Goal:** Store the vectors in a data structure optimized for fast math searches (finding similar vectors).
* **How it works:**
  - The vectors, text content, and metadata are stored in-memory using **FAISS** (Facebook AI Similarity Search).
  - No files are written to your local hard drive. The database lives only in RAM for the duration of your Streamlit session, making it secure and fast.

### 5. Conversational Memory & Query Condensation
* **Goal:** Remember prior conversation turns (like ChatGPT) and ensure that search works correctly even when the user asks ambiguous follow-up questions.
* **How it works:**
  - All messages are stored in Streamlit's `st.session_state.messages`.
  - If it's a follow-up question (e.g. User asks: *"Where did he work?"* after previously talking about *"Steve Jobs"*):
    - The app calls the LLM (`llama-3.3-70b-versatile` via Groq) with a special instruction: *"Given this chat history and the question 'Where did he work?', rephrase it as a standalone question."*
    - The LLM returns: *"Where did Steve Jobs work?"*
  - This **condensed query** is used to perform the vector search, ensuring we retrieve the correct information from the PDF.

### 6. Semantic Retrieval (Search)
* **Goal:** Fetch the most relevant text passages from the PDF database based on the query.
* **How it works:**
  - The user's query (or condensed query) is converted into a vector.
  - The FAISS database calculates the distance (Cosine Similarity or L2 distance) between the query vector and all chunk vectors.
  - It retrieves the **top 4 ($k=4$)** closest text chunks.

### 7. Prompt Formatting & Context Injection
* **Goal:** Instruct the LLM to only answer based on the retrieved PDF data.
* **How it works:**
  - The retrieved text chunks are joined together with their corresponding page numbers into a single block of text called **Context**.
  - They are injected into a prompt template:
    ```
    System: You are a precise document assistant. Answer the user's question using ONLY the context provided below. Always mention page numbers used. If the answer is not in the context, say 'I couldn't find that in the document.'. NEVER hallucinate.

    CONTEXT:
    [Page 3]:
    Steve Jobs co-founded Apple in 1976.

    [Page 5]:
    He later left Apple to found NeXT.

    Human: Where did Steve Jobs work?
    ```

### 8. LLM Generation (Groq API)
* **Goal:** Generate a fluent, correct, and grounded answer from the prompt.
* **How it works:**
  - The prompt is sent to `llama-3.3-70b-versatile` via the **Groq API**.
  - **Temperature is set to 0** (making the model deterministic and literal, avoiding creative fabrications/hallucinations).
  - The LLM reads the context, extracts the answer, formats the page citations, and streams the response text back to the Streamlit UI.

---

## 🛠️ File References in the Project

* **Step 1: Document Loading:** See [step1_loading.py](file:///c:/Users/HP/Downloads/RAG Project/pdf-chat-rag/step1_loading.py)
* **Step 2: Document Chunking:** See [step2_chunking.py](file:///c:/Users/HP/Downloads/RAG Project/pdf-chat-rag/step2_chunking.py)
* **Step 3: Vector Embeddings:** See [step3_embeddings.py](file:///c:/Users/HP/Downloads/RAG Project/pdf-chat-rag/step3_embeddings.py)
* **Step 4: Vector Storage:** See [step4_vector_store.py](file:///c:/Users/HP/Downloads/RAG Project/pdf-chat-rag/step4_vector_store.py)
* **Step 5: Context Retrieval:** See [step5_retrieval.py](file:///c:/Users/HP/Downloads/RAG Project/pdf-chat-rag/step5_retrieval.py)
* **Step 6: Answer Generation:** See [step6_generation.py](file:///c:/Users/HP/Downloads/RAG Project/pdf-chat-rag/step6_generation.py)
* **Production UI Web App:** See [app.py](file:///c:/Users/HP/Downloads/RAG Project/pdf-chat-rag/app.py)
