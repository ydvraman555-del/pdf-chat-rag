import os
import time
import base64
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Check if running in cloud (Render) or local
is_cloud = "RENDER" in os.environ or os.getenv("RAG_MODE", "local").lower() == "cloud"

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONSTANTS & SETUP
# ─────────────────────────────────────────────
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024



def trigger_confetti():
    """Injects JS to trigger canvas-confetti."""
    confetti_js = """
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    <script>
        confetti({
            particleCount: 150,
            spread: 80,
            origin: { y: 0.6 },
            colors: ['#8b5cf6', '#3b82f6', '#ec4899', '#ffffff']
        });
    </script>
    """
    components.html(confetti_js, height=0)


def stream_response(text):
    """Generator for typewriter effect."""
    for char in text:
        yield char
        time.sleep(0.01) # 10ms delay per character


def inject_custom_css():
    """Injects dynamic CSS based on theme."""
    
    # Theme variables
    bg_colors = "linear-gradient(-45deg, #0a0a0f, #150f24, #0a1128, #0a0a0f)"
    glass_bg = "rgba(255, 255, 255, 0.05)"
    glass_border = "rgba(255, 255, 255, 0.1)"
    text_primary = "#ffffff"
    text_secondary = "#a1a1aa"
    particle_color = "rgba(139, 92, 246, 0.2)"
    assistant_bg = "rgba(255, 255, 255, 0.03)"
    input_bg = "rgba(10, 10, 15, 0.8)"

    # Generate CSS box-shadow particles
    import random
    particles_shadows = []
    for _ in range(30):
        x = random.randint(0, 100)
        y = random.randint(0, 150) # Taller spread
        blur = random.randint(2, 6)
        particles_shadows.append(f"{x}vw {y}vh {blur}px {particle_color}")
    particles_css = ", ".join(particles_shadows)

    st.markdown(f"""
    <style>
    /* 1. GLOBAL STYLES */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --bg-color: transparent;
        --accent-1: #8b5cf6;
        --accent-2: #3b82f6;
        --accent-3: #ec4899;
        --glass-bg: {glass_bg};
        --glass-border: {glass_border};
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
    }}

    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(139, 92, 246, 0.3); border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(139, 92, 246, 0.6); }}

    /* Animated Gradient Mesh Background */
    @keyframes gradientBG {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .stApp {{
        background: {bg_colors};
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        background-attachment: fixed;
    }}
    
    /* CSS Particles System */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 3px;
        height: 3px;
        border-radius: 50%;
        box-shadow: {particles_css};
        animation: floatParticles 80s linear infinite;
        z-index: 0;
        pointer-events: none;
    }}
    @keyframes floatParticles {{
        0% {{ transform: translateY(0); }}
        100% {{ transform: translateY(-50vh); }}
    }}
    
    /* 2. HEADER & ANIMATED LOGO */
    .premium-title {{
        text-align: center;
        background: linear-gradient(90deg, var(--accent-2), var(--accent-1), var(--accent-3));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(139, 92, 246, 0.3);
        letter-spacing: -1px;
        position: relative;
    }}
    .animated-icon {{
        display: inline-block;
        transition: transform 0.5s ease;
        animation: floatIcon 3s ease-in-out infinite;
    }}
    .premium-title:hover .animated-icon {{
        transform: rotateY(180deg) scale(1.1);
    }}
    @keyframes floatIcon {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-5px); }}
    }}
    .premium-subtitle {{
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }}
    
    /* 3. SIDEBAR */
    [data-testid="stSidebar"] {{
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--glass-border) !important;
    }}
    
    [data-testid="stFileUploader"] {{
        background: transparent !important;
        border: 1px dashed var(--glass-border) !important;
        border-radius: 16px !important;
        padding: 15px !important;
        transition: all 0.3s ease !important;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: var(--accent-1) !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.2) !important;
    }}

    /* 4. MAIN CHAT AREA */
    [data-testid="stChatMessage"] {{
        background: var(--glass-bg) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s !important;
        animation: fadeIn 0.5s ease forwards;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    [data-testid="stChatMessage"][data-baseweb="chat-message-user"] {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15)) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-bottom-right-radius: 4px !important;
    }}
    
    [data-testid="stChatMessage"][data-baseweb="chat-message-assistant"] {{
        background: {assistant_bg} !important;
        border: 1px solid var(--glass-border) !important;
        border-bottom-left-radius: 4px !important;
    }}
    
    /* 5. SOURCE CITATIONS */
    .source-pill {{
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: var(--text-primary);
        margin-top: 12px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        cursor: default;
    }}
    .source-pill:hover {{
        background: rgba(139, 92, 246, 0.3);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.4);
        transform: translateY(-1px);
    }}
    
    /* 6. INPUT BOX */
    [data-testid="stChatInput"] {{
        background: {input_bg} !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 24px !important;
        box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s ease !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: var(--accent-1) !important;
        box-shadow: 0 0 25px rgba(139, 92, 246, 0.3), 0 -10px 40px rgba(0, 0, 0, 0.5) !important;
    }}
    
    /* 8. BUTTONS */
    .stButton > button {{
        background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%) !important;
        border: none !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6) !important;
    }}
    
    /* 9. SUCCESS/ERROR MESSAGES */
    .stAlert {{
        background: var(--glass-bg) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        color: var(--text-primary) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        animation: slideInTop 0.4s ease forwards;
        border: 1px solid var(--glass-border) !important;
    }}
    
    /* 10. EMPTY STATE */
    .empty-state {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 45vh;
        text-align: center;
        animation: fadeIn 1s ease;
    }}
    .empty-icon {{
        font-size: 5rem;
        margin-bottom: 1.5rem;
        animation: bounce 2s infinite ease-in-out;
        text-shadow: 0 10px 30px rgba(139, 92, 246, 0.4);
    }}
    .empty-title {{
        background: linear-gradient(90deg, var(--accent-2), var(--accent-1));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 1rem;
    }}
    .empty-text {{
        font-size: 1.1rem;
        color: var(--text-secondary);
        max-width: 450px;
        line-height: 1.5;
    }}
    @keyframes bounce {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-15px); }}
    }}

    /* 7. LOADING STATE */
    .custom-spinner {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px 24px;
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        width: fit-content;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
    }}
    .dot {{
        width: 8px;
        height: 8px;
        background: var(--accent-1);
        border-radius: 50%;
        animation: pulse 1.5s infinite ease-in-out;
    }}
    .dot:nth-child(2) {{ animation-delay: 0.2s; }}
    .dot:nth-child(3) {{ animation-delay: 0.4s; }}
    @keyframes pulse {{
        0%, 100% {{ transform: scale(0.8); opacity: 0.3; }}
        50% {{ transform: scale(1.2); opacity: 1; box-shadow: 0 0 10px var(--accent-1); }}
    }}
    </style>
    """, unsafe_allow_html=True)


def load_api_key() -> str:
    # If in cloud mode, load and validate GOOGLE_API_KEY
    if is_cloud:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            st.error("⚠️ GOOGLE_API_KEY not configured. Check your environment variables on Render.")
            st.stop()
        return api_key
    return ""

def validate_pdf(uploaded_file) -> bool:
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(f"⚠️ PDF too large. Please upload under {MAX_FILE_SIZE_MB} MB.")
        st.stop()
    magic_bytes = uploaded_file.read(4)
    uploaded_file.seek(0)
    if magic_bytes != b"%PDF":
        st.error("⚠️ Invalid file type. The file is not a valid PDF document.")
        st.stop()
    return True

@st.cache_resource(show_spinner=False)
def build_vectorstore(file_bytes: bytes, api_key: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        if sum(len(p.page_content.strip()) for p in pages) == 0:
            st.error("⚠️ Could not extract text from this PDF.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)

        if is_cloud:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
            )
        else:
            from langchain_ollama import OllamaEmbeddings
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
            )
        return FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        st.error("⚠️ Something went wrong processing the PDF.")
        st.stop()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def answer_question(vector_store, question: str, api_key: str) -> tuple[str, list[int], list]:
    try:
        retrieved_docs = vector_store.similarity_search(question, k=4)
        context_parts = []
        for doc in retrieved_docs:
            page_num = doc.metadata.get("page", 0) + 1
            context_parts.append(f"[Page {page_num}]:\n{doc.page_content}")
            
        context = "\n\n".join(context_parts)
        source_pages = sorted(list(set(doc.metadata.get("page", 0) + 1 for doc in retrieved_docs)))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise document assistant. Answer ONLY using the context provided below. Always mention page numbers used. If answer is not in context, say 'I couldn't find that in the document.'. NEVER hallucinate.\n\nCONTEXT:\n{context}"),
            ("human", "{question}"),
        ])
        
        messages = prompt.format_messages(context=context, question=question)
        
        if is_cloud:
            from langchain_google_genai import ChatGoogleGenerativeAI
            models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash"]
            for model_name in models_to_try:
                try:
                    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0, max_retries=1)
                    response = llm.invoke(messages)
                    return response.content, source_pages, retrieved_docs
                except Exception as model_err:
                    if "429" in str(model_err) or "quota" in str(model_err).lower():
                        continue
                    return f"⚠️ Gemini LLM error: {model_err}", [], []
            return "⚠️ Too many requests or model error.", [], []
        else:
            from langchain_groq import ChatGroq
            try:
                groq_key = os.getenv("GROQ_API_KEY")
                if not groq_key or groq_key == "your_groq_api_key_here":
                    return "⚠️ GROQ_API_KEY not found or not configured in your .env file.", [], []
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_key)
                response = llm.invoke(messages)
                return response.content, source_pages, retrieved_docs
            except Exception as model_err:
                if groq_key in str(model_err):
                    model_err = str(model_err).replace(groq_key, "***REDACTED***")
                return f"⚠️ Groq LLM error: {model_err}", [], []
            
    except Exception as e:
        return "⚠️ Something went wrong generating the answer.", [], []


def main():
    # 1. Initialize State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    # 2. Inject CSS
    inject_custom_css()
    api_key = load_api_key()
    
    # 3. Header
    st.markdown("""
        <h1 class="premium-title"><span class="animated-icon">📄</span> Chat with Your PDF</h1>
        <p class="premium-subtitle">Ask anything. Get answers with verified sources.</p>
    """, unsafe_allow_html=True)
    
    # 4. Sidebar UI
    with st.sidebar:
        st.header("📄 Document Setup")
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], label_visibility="collapsed")
        
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"**{uploaded_file.name}**\n\nSize: {file_size_mb:.1f} MB")
            
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            st.session_state.debug_mode = st.checkbox("🔍 Show retrieved chunks", value=st.session_state.get("debug_mode", False))
                
        st.markdown("---")
        if is_cloud:
            st.markdown("<small style='color: var(--text-secondary);'>🌐 Cloud Mode: Google Gemini. Processed in-memory. Zero retention.</small>", unsafe_allow_html=True)
        else:
            st.markdown("<small style='color: var(--text-secondary);'>🔒 Local Mode: Local Ollama & Groq. 100% private & secure.</small>", unsafe_allow_html=True)

    # 5. Empty State
    if not uploaded_file:
        st.session_state.pdf_processed = False # Reset
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">👈</div>
            <div class="empty-title">Welcome to your AI Assistant</div>
            <p class="empty-text">Upload a PDF from the sidebar to start extracting insights with our RAG engine.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # 6. Process PDF
    validate_pdf(uploaded_file)
    with st.spinner("Processing PDF (extracting and embedding text)..."):
        vector_store = build_vectorstore(uploaded_file.getvalue(), api_key)
    
    # Trigger confetti only once per upload
    if not st.session_state.pdf_processed:
        st.session_state.pdf_processed = True
        trigger_confetti()
        st.sidebar.success("✅ PDF processed and ready!")
        
    # 7. Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                sources_str = ", ".join(map(str, msg["sources"]))
                st.markdown(f'<div class="source-pill">📑 Sources: Page(s) {sources_str}</div>', unsafe_allow_html=True)
            if st.session_state.get("debug_mode") and "debug_chunks" in msg:
                with st.expander("🔍 Retrieved Chunks (Debug)"):
                    for j, chunk in enumerate(msg["debug_chunks"], 1):
                        st.markdown(f"**Chunk {j} — Page {chunk.get('page', '?')}**")
                        st.code(chunk.get("text", "")[:500], language=None)

    # 8. Handle User Input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            loading_html = """
            <div class="custom-spinner">
                <span style="color:var(--text-secondary); font-weight:500; font-size:0.9rem;">Thinking</span>
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>
            """
            placeholder = st.empty()
            placeholder.markdown(loading_html, unsafe_allow_html=True)
            
            answer, sources, retrieved_docs = answer_question(vector_store, prompt, api_key)
            
            placeholder.empty()
            
            # Streaming Typewriter effect
            st.write_stream(stream_response(answer))
            
            if sources:
                sources_str = ", ".join(map(str, sources))
                st.markdown(f'<div class="source-pill">📑 Sources: Page(s) {sources_str}</div>', unsafe_allow_html=True)
                
            if st.session_state.get("debug_mode") and retrieved_docs:
                with st.expander("🔍 Retrieved Chunks (Debug)"):
                    for j, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Chunk {j} — Page {doc.metadata.get('page', 0) + 1}**")
                        st.code(doc.page_content[:500], language=None)
                
        debug_chunks = [{"page": doc.metadata.get("page", 0) + 1, "text": doc.page_content} for doc in retrieved_docs] if retrieved_docs else []
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "debug_chunks": debug_chunks})


if __name__ == "__main__":
    main()
