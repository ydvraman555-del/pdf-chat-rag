"""
step4_qa.py — RAG Step 3: Question Answering with Retrieved Context + Groq LLM (Local Embeddings).

THE COMPLETE RAG PIPELINE:
1. User asks a question
2. Question is embedded → vector (locally using Ollama nomic-embed-text)
3. FAISS finds the k most similar chunks (retrieval)
4. Retrieved chunks are formatted into a context string
5. Context + question are sent to ChatGroq LLM via a prompt template
6. LLM generates an answer ONLY from the provided context
7. Page numbers are extracted for citations

SECURITY: API key from .env is loaded safely and scrubbed from any potential error messages.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
import os

# ─────────────────────────────────────────────
# Step 0: Load and validate environment
# ─────────────────────────────────────────────
load_dotenv()


def validate_api_key() -> str:
    """
    Load and validate the Groq API key from the environment.
    SECURITY: This function NEVER prints the actual API key.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file.")
        print("   → Copy .env.example to .env and add your real Groq key.")
        sys.exit(1)

    if api_key == "your_groq_api_key_here":
        print("❌ GROQ_API_KEY is still the placeholder value.")
        print("   → Replace it with your real Groq API key in .env")
        sys.exit(1)

    print(f"✅ Groq API key loaded (length: {len(api_key)} chars)")
    return api_key


# ─────────────────────────────────────────────
# Step 1: Reuse PDF loading + chunking + embeddings
# ─────────────────────────────────────────────
def build_vector_store(pdf_path: Path):
    """
    Load PDF → chunk → embed → store in FAISS.
    Reuses logic from step2_chunking.py and step3_embeddings.py.
    Runs embeddings fully locally.

    Returns:
        FAISS vector store ready for similarity search.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS

    # Load PDF
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path.resolve()}")
        sys.exit(1)

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    if sum(len(p.page_content.strip()) for p in pages) == 0:
        print("❌ PDF has no extractable text.")
        sys.exit(1)

    # Chunk (same params as previous steps)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)
    print(f"✅ Loaded {len(pages)} pages → {len(chunks)} chunks")

    # Embed + store in FAISS (local embeddings, 768-D)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    print("🔄 Building vector store (Local Embeddings)...")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        print(f"✅ FAISS index built with {len(chunks)} vectors")
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        print("   → Make sure Ollama is running locally ('ollama serve') and model is pulled.")
        sys.exit(1)

    return vector_store


# ─────────────────────────────────────────────
# Step 2: The Core RAG QA Function
# ─────────────────────────────────────────────
def answer_question(vector_store, question: str, groq_api_key: str) -> tuple:
    """
    Answer a question using RAG: retrieve relevant chunks, then generate answer via Groq.

    THIS IS WHERE RAG HAPPENS:
    1. Retrieve: Find chunks similar to the question (embedding similarity)
    2. Augment: Format chunks into a context string with page numbers
    3. Generate: Send context + question to ChatGroq LLM with strict instructions

    Args:
        vector_store: FAISS index with embedded chunks.
        question: User's question as a string.
        groq_api_key: Groq API key (never printed).

    Returns:
        Tuple of (answer_text: str, source_pages: list[int])
    """
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate

    # ── Step 2a: RETRIEVE top 4 chunks ──
    retrieved_docs = vector_store.similarity_search(question, k=4)

    # ── Step 2b: BUILD context string with page citations ──
    context_parts = []
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", 0) + 1  # Convert 0-indexed → 1-indexed
        context_parts.append(f"[Page {page_num}]:\n{doc.page_content}")

    # Join all chunks with double newlines for clear separation
    context = "\n\n".join(context_parts)

    # ── Step 2c: COLLECT unique source page numbers (sorted) ──
    source_pages = sorted(set(
        doc.metadata.get("page", 0) + 1 for doc in retrieved_docs
    ))

    # ── Step 2d: BUILD the prompt template ──
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a precise document assistant. Your ONLY job is to answer
questions based on the provided document context.

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
        (
            "human",
            "{question}"
        ),
    ])

    # ── Step 2e: INITIALIZE the Groq LLM ──
    # temperature=0: Makes the LLM deterministic for factual grounding.
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0,
    )

    # ── Step 2f: INVOKE the chain ──
    try:
        messages = prompt.format_messages(
            context=context,
            question=question,
        )

        # Send to Groq and get response
        response = llm.invoke(messages)
        answer_text = response.content

    except Exception as e:
        error_msg = str(e)
        # SECURITY: Scrub Groq API key from any error messages to prevent leakage
        if groq_api_key in error_msg:
            error_msg = error_msg.replace(groq_api_key, "***REDACTED***")
        
        answer_text = f"❌ Groq LLM error: {error_msg}"

    return answer_text, source_pages


# ─────────────────────────────────────────────
# Step 3: Pretty-print results
# ─────────────────────────────────────────────
def print_qa_result(question: str, answer: str, pages: list, question_num: int) -> None:
    """Display a Q&A result with clear formatting."""
    print(f"\n{'=' * 60}")
    print(f"❓ Question {question_num}: {question}")
    print(f"{'=' * 60}")
    print(f"\n💬 Answer:\n{answer}")
    print(f"\n📖 Source pages: {pages}")
    print(f"{'─' * 60}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Reconfigure stdout to support UTF-8 print (emojis on Windows)
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("🤖 RAG Step 3: Question Answering with Groq LLM")
    print("=" * 60)
    print()

    # 1. Validate Groq API key
    groq_key = validate_api_key()

    # 2. Build the vector store (uses local Ollama embeddings)
    project_root = Path(__file__).parent
    pdf_path = project_root / "sample.pdf"
    vector_store = build_vector_store(pdf_path)

    # 3. Test with 3 questions that demonstrate different RAG behaviors
    test_questions = [
        # Q1: Factual — answer IS in the PDF
        "What is Retrieval-Augmented Generation (RAG)?",

        # Q2: Synthesis — requires combining info from multiple sections
        "What are the main components of Natural Language Processing and Deep Learning mentioned?",

        # Q3: NOT in the PDF — tests hallucination prevention
        "What is the meaning of life?",
    ]

    print(f"\n📝 Testing with {len(test_questions)} questions...\n")

    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Processing question {i}/{len(test_questions)}...")

        answer, pages = answer_question(vector_store, question, groq_key)
        print_qa_result(question, answer, pages, i)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 RAG pipeline complete!")
    print("   PDF → Chunks → Local Embeddings → FAISS → Retrieval → Groq LLM → Answer")
    print("\n   Next: Wrap this in a Streamlit UI (app.py) for interactive use!")
    print("=" * 60)


if __name__ == "__main__":
    main()
