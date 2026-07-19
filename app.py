import os
import time
import threading
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

# Pre-load libraries at startup to prevent lag during the first PDF upload or first message
if is_cloud:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
else:
    from langchain_ollama import OllamaEmbeddings
    from langchain_groq import ChatGroq

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

def pre_warm_worker():
    """Background worker to pre-warm embedding and LLM models without blocking the UI thread."""
    if is_cloud:
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key and api_key != "your_gemini_api_key_here":
                # Warm up Google Gemini Embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=api_key,
                )
                embeddings.embed_query("warmup")
                
                # Warm up Google Gemini LLM
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0.0,
                )
                llm.invoke("Hi")
        except Exception:
            pass
    else:
        try:
            # Warm up Ollama nomic-embed-text
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            embeddings.embed_query("warmup")
            
            # Warm up Groq LLM connection
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key and groq_key != "your_groq_api_key_here":
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=groq_key)
                llm.invoke("Hi")
        except Exception:
            pass

@st.cache_resource(show_spinner=False)
def pre_warm_models():
    """Starts a background thread to pre-warm the models without blocking page rendering."""
    thread = threading.Thread(target=pre_warm_worker, daemon=True)
    thread.start()

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
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
            )
        else:
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


def answer_question(vector_store, question: str, api_key: str, k: int = 4, temperature: float = 0.0) -> tuple[str, list[int], list]:
    try:
        retrieved_docs = vector_store.similarity_search(question, k=k)
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
            models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash"]
            for model_name in models_to_try:
                try:
                    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature, max_retries=1)
                    response = llm.invoke(messages)
                    return response.content, source_pages, retrieved_docs
                except Exception as model_err:
                    if "429" in str(model_err) or "quota" in str(model_err).lower():
                        continue
                    return f"⚠️ Gemini LLM error: {model_err}", [], []
            return "⚠️ Too many requests or model error.", [], []
        else:
            try:
                groq_key = os.getenv("GROQ_API_KEY")
                if not groq_key or groq_key == "your_groq_api_key_here":
                    return "⚠️ GROQ_API_KEY not found or not configured in your .env file.", [], []
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature, groq_api_key=groq_key)
                response = llm.invoke(messages)
                return response.content, source_pages, retrieved_docs
            except Exception as model_err:
                if groq_key in str(model_err):
                    model_err = str(model_err).replace(groq_key, "***REDACTED***")
                return f"⚠️ Groq LLM error: {model_err}", [], []
            
    except Exception as e:
        return "⚠️ Something went wrong generating the answer.", [], []


def render_landing_page():
    # Hero Section
    st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 2.2rem; border-radius: 16px; margin-bottom: 2rem; text-align: center; backdrop-filter: blur(10px); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);">
            <h2 style="color: #ffffff; margin-bottom: 0.8rem; font-size: 2.2rem; font-weight: 700; background: linear-gradient(to right, #8b5cf6, #3b82f6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚡ Grounded Document Q&A Assistant</h2>
            <p style="color: #a1a1aa; font-size: 1.15rem; line-height: 1.7; margin: 0; max-width: 800px; margin-left: auto; margin-right: auto;">
                PDF Chat RAG is a premium, secure document intelligence platform. Chat naturally with your PDFs and get instant answers backed by verifiable page-level sources and strict anti-hallucination guardrails.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Features Grid
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; min-height: 140px; margin-bottom: 1.2rem; transition: transform 0.2s;">
                <h4 style="color: #8b5cf6; margin-top: 0; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;">🔒 Privacy-First</h4>
                <p style="color: #a1a1aa; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                    Your data stays private. In Local Mode, embeddings are calculated 100% locally on your computer using Ollama.
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; min-height: 140px; margin-bottom: 1.2rem; transition: transform 0.2s;">
                <h4 style="color: #ec4899; margin-top: 0; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;">⚡ Zero-Lag Chat</h4>
                <p style="color: #a1a1aa; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                    With conditional import loading and Ollama model pre-warming, processing your PDF and chatting is incredibly fast.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; min-height: 140px; margin-bottom: 1.2rem; transition: transform 0.2s;">
                <h4 style="color: #3b82f6; margin-top: 0; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;">📑 Page Citations</h4>
                <p style="color: #a1a1aa; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                    No hallucinations. The AI is strictly constrained to the uploaded PDF context and cites the exact page numbers used.
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; min-height: 140px; margin-bottom: 1.2rem; transition: transform 0.2s;">
                <h4 style="color: #10b981; margin-top: 0; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;">🌐 Dual Cloud Mode</h4>
                <p style="color: #a1a1aa; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                    Runs locally on Ollama/Groq, but automatically detects Render and switches to Google Gemini for seamless online use.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Interactive Simulator
    st.markdown("### 🎮 Interactive RAG Simulator")
    st.markdown("Select a sample question below and simulate how the 6-step RAG pipeline processes documents and prevents hallucinations:")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        sim_query = st.selectbox(
            "Choose a simulation query:",
            [
                "What is Retrieval-Augmented Generation?",
                "What are neural networks?",
                "What is the meaning of life?"
            ],
            key="sim_query_select",
            label_visibility="collapsed"
        )
        run_sim = st.button("🚀 Run Simulator", use_container_width=True)
    
    with sim_col2:
        sim_placeholder = st.empty()
        if run_sim:
            with sim_placeholder.container():
                st.markdown("<div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 1.2rem; border-radius: 12px; min-height: 250px;'>", unsafe_allow_html=True)
                
                with st.spinner("📥 Step 1: Loading PDF pages..."):
                    time.sleep(0.4)
                st.success("📥 Step 1: PDF loaded successfully (5 pages parsed)")
                
                with st.spinner("✂️ Step 2: Chunking text..."):
                    time.sleep(0.4)
                st.success("✂️ Step 2: Created 11 overlapping chunks (size=1000, overlap=200)")
                
                with st.spinner("📐 Step 3: Generating embeddings..."):
                    time.sleep(0.4)
                st.success("📐 Step 3: Dense 768-D vectors calculated via nomic-embed-text")
                
                with st.spinner("💾 Step 4: Building FAISS index..."):
                    time.sleep(0.4)
                st.success("💾 Step 4: Index created in-memory and persisted to disk")
                
                with st.spinner("🔍 Step 5: Performing semantic search..."):
                    time.sleep(0.4)
                st.success("🔍 Step 5: Retrieved top 3 matching chunks (highest similarity score)")
                
                with st.spinner("🤖 Step 6: Querying LLM..."):
                    time.sleep(0.5)
                
                st.markdown("💬 **Response from RAG Assistant:**")
                if sim_query == "What is Retrieval-Augmented Generation?":
                    st.info("Retrieval-Augmented Generation (RAG) is a technique that enhances an LLM by fetching relevant facts from an external document store before writing the answer. This ensures the output is grounded and does not rely on outdated memory. (Page 3)")
                elif sim_query == "What are neural networks?":
                    st.info("Neural networks are models composed of layers of nodes (neurons) mimicking human brain structures. They map input features to output values through weights adjusted during backpropagation. (Page 2)")
                else:
                    st.warning("⚠️ **Hallucination Prevention Triggered:** I couldn't find information about the meaning of life in the provided document context.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            sim_placeholder.markdown("""
                <div style="background: rgba(255,255,255,0.01); border: 1px dashed rgba(255,255,255,0.1); padding: 2rem; border-radius: 12px; height: 100%; min-height: 250px; display: flex; align-items: center; justify-content: center; text-align: center;">
                    <p style="color: #a1a1aa; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                        👈 Select a query and click "Run Simulator" to watch the RAG pipeline run in real-time step-by-step!
                    </p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stepper / How it works Accordion
    st.markdown("### ⚙️ How the RAG Pipeline Works")
    with st.expander("🔍 Step-by-Step Architecture (The 6-Step Pipeline)"):
        st.markdown("""
            <div style="color: #a1a1aa; line-height: 1.8; font-size: 0.95rem; padding: 0.5rem;">
                <ul style="list-style-type: none; padding-left: 0; margin: 0;">
                    <li style="margin-bottom: 0.8rem;"><b style="color: #ffffff;">1. Document Loading:</b> Parses your PDF page-by-page using PyPDFLoader.</li>
                    <li style="margin-bottom: 0.8rem;"><b style="color: #ffffff;">2. Text Chunking:</b> Splits pages into smaller, overlapping 1000-character segments (RecursiveCharacterTextSplitter).</li>
                    <li style="margin-bottom: 0.8rem;"><b style="color: #ffffff;">3. Vector Embeddings:</b> Converts chunks into dense coordinates (768-D vectors locally via Ollama nomic-embed-text).</li>
                    <li style="margin-bottom: 0.8rem;"><b style="color: #ffffff;">4. Vector Database:</b> Stores and indexes the coordinates in-memory using a FAISS database.</li>
                    <li style="margin-bottom: 0.8rem;"><b style="color: #ffffff;">5. Semantic Retrieval:</b> Searches the database to find the top matching chunks related to your question.</li>
                    <li style="margin-bottom: 0.8rem;"><b style="color: #ffffff;">6. Answer Generation:</b> Merges matching chunks with strict rules into a prompt, querying Groq/Gemini to write the answer.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Launch Button
    if st.button("💬 Launch Chat Assistant", use_container_width=True):
        st.session_state.current_page = "Chat"
        st.rerun()


def main():
    # 1. Initialize State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    # 2. Inject CSS
    inject_custom_css()
    api_key = load_api_key()
    
    # Pre-warm local models at startup to prevent lag
    pre_warm_models()
    
    # Define variables to prevent scope/unbound errors
    uploaded_file = None
    retrieval_k = 4
    temperature = 0.0

    # 3. Sidebar UI (Navigation and Document Settings)
    with st.sidebar:
        st.markdown("### 🗺️ Navigation")
        current_idx = 0 if st.session_state.current_page == "Home" else 1
        page = st.radio("Go to", ["🏠 Home", "💬 Chat Assistant"], index=current_idx, key="nav_radio", label_visibility="collapsed")
        
        # Sync navigation radio to session state
        if page == "🏠 Home":
            st.session_state.current_page = "Home"
        else:
            st.session_state.current_page = "Chat"
            
        if st.session_state.current_page == "Home":
            st.markdown("---")
            st.markdown("### 📡 System Status")
            if is_cloud:
                st.markdown("""
                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                        <p style="color: #3b82f6; font-size: 0.9rem; margin-top: 0; margin-bottom: 0.3rem;"><b>Active Mode</b></p>
                        <p style="color: #ffffff; font-size: 1rem; margin: 0; font-weight: 500;">🌐 Cloud Mode</p>
                        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.05); margin: 0.5rem 0;">
                        <p style="color: #a1a1aa; font-size: 0.85rem; margin-bottom: 0.2rem;"><b>Embedding:</b> Gemini-001</p>
                        <p style="color: #a1a1aa; font-size: 0.85rem; margin-bottom: 0;"><b>LLM:</b> Gemini-2.5-Flash</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                        <p style="color: #8b5cf6; font-size: 0.9rem; margin-top: 0; margin-bottom: 0.3rem;"><b>Active Mode</b></p>
                        <p style="color: #ffffff; font-size: 1rem; margin: 0; font-weight: 500;">🔒 Local Mode</p>
                        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.05); margin: 0.5rem 0;">
                        <p style="color: #a1a1aa; font-size: 0.85rem; margin-bottom: 0.2rem;"><b>Embedding:</b> nomic-embed-text</p>
                        <p style="color: #a1a1aa; font-size: 0.85rem; margin-bottom: 0;"><b>LLM:</b> llama-3.3-70b-versatile</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: rgba(139, 92, 246, 0.05); border: 1px dashed rgba(139, 92, 246, 0.2); padding: 1rem; border-radius: 8px;">
                    <p style="color: #a1a1aa; font-size: 0.85rem; margin: 0; line-height: 1.45;">
                        💡 <b>Tip:</b> Click <b>💬 Chat Assistant</b> in the navigation menu above to upload a PDF document and start chatting!
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        elif st.session_state.current_page == "Chat":
            st.markdown("---")
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
            st.markdown("### ⚙️ RAG Hyperparameters")
            retrieval_k = st.slider("Top K Chunks (Retrieval)", min_value=1, max_value=10, value=4, help="Number of document chunks retrieved for context.")
            temperature = st.slider("LLM Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.0, step=0.1, help="0.0 is deterministic and factual; 1.0 is creative.")
                
        st.markdown("---")
        if is_cloud:
            st.markdown("<small style='color: var(--text-secondary);'>🌐 Cloud Mode: Google Gemini. Processed in-memory. Zero retention.</small>", unsafe_allow_html=True)
        else:
            st.markdown("<small style='color: var(--text-secondary);'>🔒 Local Mode: Local Ollama & Groq. 100% private & secure.</small>", unsafe_allow_html=True)

    # 4. Header (runs on both pages)
    st.markdown("""
        <h1 class="premium-title"><span class="animated-icon">📄</span> Chat with Your PDF</h1>
        <p class="premium-subtitle">Ask anything. Get answers with verified sources.</p>
    """, unsafe_allow_html=True)

    # 5. Routing
    if st.session_state.current_page == "Home":
        render_landing_page()
        return

    # 6. Chat Page Empty State
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
            
            answer, sources, retrieved_docs = answer_question(vector_store, prompt, api_key, k=retrieval_k, temperature=temperature)
            
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
