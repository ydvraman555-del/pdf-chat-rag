"""
step4_qa.py — RAG Step 3: Question Answering with Retrieved Context + Gemini LLM.

THE COMPLETE RAG PIPELINE:
1. User asks a question
2. Question is embedded → vector (same as Step 2)
3. FAISS finds the k most similar chunks (retrieval)
4. Retrieved chunks are formatted into a context string
5. Context + question are sent to Gemini LLM via a prompt template
6. LLM generates an answer ONLY from the provided context
7. Page numbers are extracted for citations

THIS IS THE KEY INSIGHT OF RAG:
The LLM doesn't answer from its training data (which could be outdated/wrong).
It answers from YOUR specific PDF content, grounded in retrieved evidence.

SECURITY: API key from .env, never printed.
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
    """Load and validate the Google API key. NEVER prints the key."""
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file.")
        print("   → Copy .env.example to .env and add your real key.")
        sys.exit(1)

    if api_key == "your_gemini_api_key_here":
        print("❌ GOOGLE_API_KEY is still the placeholder value.")
        sys.exit(1)

    print(f"✅ API key loaded (length: {len(api_key)} chars)")
    return api_key


# ─────────────────────────────────────────────
# Step 1: Reuse PDF loading + chunking + embeddings
# ─────────────────────────────────────────────
def build_vector_store(pdf_path: Path, api_key: str):
    """
    Load PDF → chunk → embed → store in FAISS.
    Reuses logic from step2_chunking.py and step3_embeddings.py.

    Returns:
        FAISS vector store ready for similarity search.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

    # Embed + store in FAISS
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    print("🔄 Building vector store...")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        print(f"✅ FAISS index built with {len(chunks)} vectors")
    except Exception as e:
        error_msg = str(e)
        if api_key in error_msg:
            error_msg = error_msg.replace(api_key, "***REDACTED***")
        print(f"❌ Embedding error: {error_msg}")
        sys.exit(1)

    return vector_store


# ─────────────────────────────────────────────
# Step 2: The Core RAG QA Function
# ─────────────────────────────────────────────
def answer_question(vector_store, question: str, api_key: str) -> tuple:
    """
    Answer a question using RAG: retrieve relevant chunks, then generate answer.

    THIS IS WHERE RAG HAPPENS:
    1. Retrieve: Find chunks similar to the question (embedding similarity)
    2. Augment: Format chunks into a context string with page numbers
    3. Generate: Send context + question to LLM with strict instructions

    Args:
        vector_store: FAISS index with embedded chunks.
        question: User's question as a string.
        api_key: Google API key (never printed).

    Returns:
        Tuple of (answer_text: str, source_pages: list[int])
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate

    # ── Step 2a: RETRIEVE top 4 chunks ──
    # similarity_search embeds the question and finds the 4 nearest chunks
    # k=4 is a good balance: enough context for a complete answer,
    # but not so much that irrelevant chunks dilute the signal
    retrieved_docs = vector_store.similarity_search(question, k=4)

    # ── Step 2b: BUILD context string with page citations ──
    # Format each chunk with its page number so the LLM can cite sources
    # We add 1 to page numbers because metadata uses 0-indexed pages
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
    #
    # WHY ChatPromptTemplate instead of f-strings?
    # 1. Separation of concerns: template structure vs. variable content
    # 2. Automatic escaping: prevents prompt injection from user input
    # 3. Reusability: same template, different variables each call
    # 4. LangChain integration: works with chains, memory, and tracing
    #
    # The system message sets STRICT rules for the LLM:
    # - ONLY use the provided context (no training knowledge)
    # - Cite page numbers
    # - Admit when info isn't available (prevents hallucination)
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

    # ── Step 2e: INITIALIZE the LLM ──
    #
    # temperature=0: Makes the LLM deterministic (always picks the most
    # likely token). Perfect for factual Q&A where we want consistent,
    # grounded answers — NOT creative writing.
    #
    # Think of temperature like regularization in ML:
    #   temperature=0 → argmax (greedy, deterministic)
    #   temperature=1 → standard sampling (creative, varied)
    #   temperature>1 → more random (too creative, unreliable)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0,        # Deterministic: same question → same answer
        max_output_tokens=1024,  # Cap response length
    )

    # ── Step 2f: INVOKE the chain ──
    # This sends the formatted prompt (system + human messages) to Gemini
    # The LLM sees the context + question and generates a grounded answer
    try:
        # Format the prompt template with our variables
        messages = prompt.format_messages(
            context=context,
            question=question,
        )

        # Send to Gemini and get response
        response = llm.invoke(messages)
        answer_text = response.content

    except Exception as e:
        error_msg = str(e)
        # Scrub API key from any error messages
        if api_key in error_msg:
            error_msg = error_msg.replace(api_key, "***REDACTED***")

        if "429" in error_msg or "quota" in error_msg.lower():
            answer_text = (
                "⚠️ Rate limited (429). The free tier quota is temporarily "
                "exhausted. Wait a few minutes and try again."
            )
        elif "invalid" in error_msg.lower() or "api key" in error_msg.lower():
            answer_text = "❌ Invalid API key. Check your .env file."
        else:
            answer_text = f"❌ LLM error: {error_msg}"

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
    print("=" * 60)
    print("🤖 RAG Step 3: Question Answering with Gemini LLM")
    print("=" * 60)
    print()

    # 1. Validate API key
    api_key = validate_api_key()

    # 2. Build the vector store (reuses Steps 1-2)
    project_root = Path(__file__).parent
    pdf_path = project_root / "sample.pdf"
    vector_store = build_vector_store(pdf_path, api_key)

    # 3. Test with 3 questions that demonstrate different RAG behaviors
    test_questions = [
        # Q1: Factual — answer IS in the PDF
        # Expected: LLM finds and summarizes the main topic from retrieved chunks
        "What is this document about?",

        # Q2: Reasonable synthesis — requires combining info from multiple chunks
        # Expected: LLM synthesizes key points from several retrieved chunks
        "Summarize the key points.",

        # Q3: NOT in the PDF — tests hallucination prevention
        # Expected: LLM responds with "I couldn't find that in the document."
        # WITHOUT RAG constraints, the LLM would happily answer from training data!
        "What is the meaning of life?",
    ]

    print(f"\n📝 Testing with {len(test_questions)} questions...\n")

    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Processing question {i}/{len(test_questions)}...")

        answer, pages = answer_question(vector_store, question, api_key)
        print_qa_result(question, answer, pages, i)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 RAG pipeline complete!")
    print("   PDF → Chunks → Embeddings → FAISS → Retrieval → LLM → Answer")
    print("\n   Next: Wrap this in a Streamlit UI for interactive use!")
    print("=" * 60)


if __name__ == "__main__":
    main()
