import os
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered"
)

# ─────────────────────────────────────────────
# CONSTANTS & SETUP
# ─────────────────────────────────────────────
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main background with dynamic gradient mesh effect */
    .stApp {
        background: radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.15), transparent 25%),
                    radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.15), transparent 25%),
                    linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        background-attachment: fixed;
        color: white;
    }
    
    /* Sidebar glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.5) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 15px !important;
        padding: 10px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(139, 92, 246, 0.5) !important;
    }

    /* Chat message glassmorphism */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.25) !important;
    }
    
    /* Chat message user vs assistant */
    [data-testid="stChatMessage"][data-baseweb="chat-message-user"] {
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        border-bottom-right-radius: 4px !important;
    }
    
    [data-testid="stChatMessage"][data-baseweb="chat-message-assistant"] {
        background: linear-gradient(145deg, rgba(139, 92, 246, 0.15) 0%, rgba(139, 92, 246, 0.05) 100%) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-bottom-left-radius: 4px !important;
    }
    
    /* Chat input box glassmorphism */
    [data-testid="stChatInput"] {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 24px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 8px 40px 0 rgba(139, 92, 246, 0.2) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        transition: background 0.2s !important;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Typography & Titles */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #93c5fd, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        border: none !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
    }
    
    /* Info/Alert boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: #e2e8f0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
# Inject CSS
inject_custom_css()



def load_api_key() -> str:
    """
    SECURITY FIRST: Load API key safely.
    Priority: 1) System env var (Render), 2) .env file (Local)
    Never prints or displays the key.
    """
    # Load from .env if running locally
    load_dotenv()
    
    # Get from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        st.error("⚠️ API key not configured. Check your environment variables.")
        st.stop()
        
    return api_key


def validate_pdf(uploaded_file) -> bool:
    """
    Validate that the uploaded file is a PDF by checking its magic bytes
    and ensuring it's within the size limit.
    """
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(f"⚠️ PDF too large. Please upload under {MAX_FILE_SIZE_MB} MB.")
        st.stop()
        
    # Read the first 4 bytes to check for PDF magic number (%PDF)
    magic_bytes = uploaded_file.read(4)
    uploaded_file.seek(0)  # Reset pointer for future reading
    
    if magic_bytes != b"%PDF":
        st.error("⚠️ Invalid file type. The file is not a valid PDF document.")
        st.stop()
        
    return True


@st.cache_resource(show_spinner=False)
def build_vectorstore(file_bytes: bytes, api_key: str):
    """
    Process the PDF: extract text, chunk it, embed it, and store in FAISS.
    Cached based on file_bytes so we only process a PDF ONCE per upload session.
    
    Note: We use FAISS here instead of ChromaDB because FAISS ships with pre-built
    Windows binaries (no C++ Build Tools needed) and works perfectly in-memory
    for Render deployments.
    """
    # 1. Create a temporary file to save the uploaded bytes
    # Render has ephemeral storage, but we delete it immediately anyway
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # 2. Load PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        # Check if PDF has extractable text
        if sum(len(p.page_content.strip()) for p in pages) == 0:
            st.error("⚠️ Could not extract text from this PDF. It may be scanned or image-based.")
            st.stop()

        # 3. Chunk text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(pages)

        # 4. Create embeddings and store in FAISS (in-memory)
        # Using gemini-embedding-001 (Google's latest, replaced deprecated embedding-001)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            st.error("⚠️ Too many requests to the embedding API. Please wait a minute and try again.")
        else:
            # Log internally, show generic error to user
            print(f"Embedding Error: {error_msg}")
            st.error("⚠️ Something went wrong processing the PDF. Please try again.")
        st.stop()
        
    finally:
        # 5. Clean up: Delete temp file immediately
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def answer_question(vector_store, question: str, api_key: str) -> tuple[str, list[int], list]:
    """
    Retrieves context and generates an answer.
    Tries multiple Gemini models as fallback if one is rate-limited.
    """
    try:
        # 1. Retrieve top 4 most relevant chunks
        retrieved_docs = vector_store.similarity_search(question, k=4)
        
        # 2. Build context string with 1-indexed page numbers
        context_parts = []
        for doc in retrieved_docs:
            page_num = doc.metadata.get("page", 0) + 1
            context_parts.append(f"[Page {page_num}]:\n{doc.page_content}")
            
        context = "\n\n".join(context_parts)
        
        # Collect unique, sorted page numbers
        source_pages = sorted(list(set(
            doc.metadata.get("page", 0) + 1 for doc in retrieved_docs
        )))
        
        # 3. Define STRICT prompt template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a precise document assistant. Your ONLY job is to answer questions based on the provided document context.

STRICT RULES:
1. Answer ONLY using the context provided below. Do NOT use any outside knowledge.
2. Always mention which page number(s) your answer comes from.
3. If the answer is NOT in the context, say exactly:
   "I couldn't find that in the document."
4. NEVER make up information or hallucinate facts.
5. Be concise but complete.

CONTEXT FROM DOCUMENT:
{context}"""
            ),
            ("human", "{question}"),
        ])
        
        # 4. Try multiple models (free tier has separate quotas per model)
        # gemini-2.5-flash is newer and has its own quota pool
        models_to_try = [
            "gemini-2.5-flash",       # Best: newest, separate quota
            "gemini-2.5-flash-lite",  # Lightweight fallback
            "gemini-2.0-flash-lite",  # Older but separate quota
            "gemini-2.0-flash",       # Last resort
        ]
        
        messages = prompt.format_messages(context=context, question=question)
        
        last_error = None
        for model_name in models_to_try:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0,
                    max_retries=1,  # Don't let LangChain retry endlessly
                )
                response = llm.invoke(messages)
                return response.content, source_pages, retrieved_docs
            except Exception as model_err:
                last_error = str(model_err)
                # If it's a rate limit, try the next model
                if "429" in last_error or "quota" in last_error.lower():
                    continue
                # For non-rate-limit errors, break immediately
                break
        
        # All models failed
        if last_error and ("429" in last_error or "quota" in last_error.lower()):
            return "⚠️ Too many requests. All model quotas are exhausted. Please wait a few minutes and try again.", [], []
        else:
            print(f"LLM Error: {last_error}")
            return "⚠️ Something went wrong generating the answer. Please try again.", [], []

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "⚠️ Too many requests. Please wait a minute and try again.", [], []
        else:
            print(f"LLM Error: {error_msg}")
            return "⚠️ Something went wrong generating the answer. Please try again.", [], []


# ─────────────────────────────────────────────
# MAIN APP UI
# ─────────────────────────────────────────────
def main():
    # 1. Load API Key gracefully
    api_key = load_api_key()
    
    # 2. Main Area UI
    st.title("📄 Chat with Your PDF")
    
    # 3. Sidebar UI
    with st.sidebar:
        st.header("Document Setup")
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        
        if uploaded_file:
            # Show file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"**{uploaded_file.name}**\n\nSize: {file_size_mb:.1f} MB")
            
            # Clear chat button
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            # Debug toggle
            st.session_state.debug_mode = st.checkbox("🔍 Show retrieved chunks", value=st.session_state.get("debug_mode", False))
                
        st.markdown("---")
        st.markdown("<small>🔒 Your PDF is processed in-memory and never stored.</small>", unsafe_allow_html=True)

    # 4. Process PDF & Chat Logic
    if not uploaded_file:
        st.write("👈 Upload a PDF from the sidebar to get started.")
        return

    # Validate and process the PDF
    validate_pdf(uploaded_file)
    
    with st.spinner("Processing PDF (extracting and embedding text)..."):
        # We pass file.getvalue() so Streamlit can hash it for caching
        vector_store = build_vectorstore(uploaded_file.getvalue(), api_key)
    
    st.sidebar.success("✅ PDF processed and ready!")
    
    # 5. Initialize Chat History in session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # 6. Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                st.caption(f"📑 Sources: Page(s) {', '.join(map(str, msg['sources']))}")
            # Show debug chunks if saved and debug mode is on
            if st.session_state.get("debug_mode") and "debug_chunks" in msg:
                with st.expander("🔍 Retrieved Chunks (Debug)"):
                    for j, chunk in enumerate(msg["debug_chunks"], 1):
                        pg = chunk.get("page", "?")
                        st.markdown(f"**Chunk {j} — Page {pg}**")
                        st.code(chunk.get("text", "")[:500], language=None)
                        st.markdown("---")

    # 7. Handle User Input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        # Add user message to UI and history
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and show response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources, retrieved_docs = answer_question(vector_store, prompt, api_key)
                
            st.markdown(answer)
            if sources:
                st.caption(f"📑 Sources: Page(s) {', '.join(map(str, sources))}")
            # Show debug expander for current response
            if st.session_state.get("debug_mode") and retrieved_docs:
                with st.expander("🔍 Retrieved Chunks (Debug)"):
                    for j, doc in enumerate(retrieved_docs, 1):
                        pg = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Chunk {j} — Page {pg}**")
                        st.code(doc.page_content[:500], language=None)
                        st.markdown("---")
                
        # Save assistant message to history
        # Build debug chunk data for history
        debug_chunks = []
        if retrieved_docs:
            debug_chunks = [
                {"page": doc.metadata.get("page", 0) + 1, "text": doc.page_content}
                for doc in retrieved_docs
            ]
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources,
            "debug_chunks": debug_chunks,
        })


if __name__ == "__main__":
    main()
