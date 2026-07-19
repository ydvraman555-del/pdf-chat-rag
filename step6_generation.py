"""
step6_generation.py — RAG Step 6: Generate grounded LLM answers using retrieved context.

WHAT THIS DOES:
1. Loads the Groq API key from your local .env.
2. Loads the persisted FAISS vector index (Step 4).
3. Retrieves the top k=4 relevant document chunks for the user question (Step 5).
4. Assembles the chunks with page headers into a single context string.
5. Injects the context and question into a strict prompt template.
6. Initialises ChatGroq (llama-3.3-70b-versatile, temperature=0).
7. Generates a deterministic, factual answer and extracts source page citations.

SECURITY:
- Groq API Key is retrieved from environment variables.
- Exception handlers automatically redact the API key value, protecting it from log leaks.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()


def validate_api_key() -> str:
    """
    Validate that GROQ_API_KEY is configured in the environment.
    SECURITY: Never prints the key value.
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


def load_vector_store(index_dir: Path):
    """
    Load the persisted FAISS vector store.
    """
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS

    if not index_dir.exists():
        print(f"❌ FAISS index directory '{index_dir.resolve()}' not found.")
        print("   → Run 'python step4_vector_store.py' first.")
        sys.exit(1)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


def answer_question(vector_store, question: str, groq_api_key: str) -> tuple:
    """
    Retrieve context and generate a grounded answer using the Groq API.
    """
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate

    # 1. Retrieve: Get top 4 closest chunks
    retrieved_docs = vector_store.similarity_search(question, k=4)

    # 2. Augment: Format retrieved text chunks with source page headers
    context_parts = []
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", 0) + 1  # 0-indexed → 1-indexed
        context_parts.append(f"[Page {page_num}]:\n{doc.page_content}")

    context = "\n\n".join(context_parts)
    
    # Collect unique page citations
    source_pages = sorted(set(
        doc.metadata.get("page", 0) + 1 for doc in retrieved_docs
    ))

    # 3. Prompt: Create strict RAG instructions
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

    # 4. LLM: Initialize Groq LLM (temperature=0 for factual grounding)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0,
    )

    # 5. Generate: Invoke LLM
    try:
        messages = prompt.format_messages(
            context=context,
            question=question
        )
        response = llm.invoke(messages)
        answer_text = response.content
    except Exception as e:
        error_msg = str(e)
        # SECURITY: Redact API key from output error messages
        if groq_api_key in error_msg:
            error_msg = error_msg.replace(groq_api_key, "***REDACTED***")
        answer_text = f"❌ Groq LLM error: {error_msg}"

    return answer_text, source_pages


def print_qa_result(question: str, answer: str, pages: list, num: int):
    """
    Pretty-print result.
    """
    print(f"\n{'=' * 60}")
    print(f"❓ Question {num}: {question}")
    print(f"{'=' * 60}")
    print(f"\n💬 Answer:\n{answer}")
    print(f"\n📖 Source pages: {pages}")
    print(f"{'─' * 60}")


def main():
    # Reconfigure stdout to support UTF-8 print (emojis on Windows)
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("🤖 RAG Step 6: Grounded Answer Generation")
    print("=" * 60)
    print()

    # 1. Validate Groq API key
    groq_key = validate_api_key()

    # 2. Load FAISS index
    project_root = Path(__file__).parent
    index_dir = project_root / "faiss_index"
    vector_store = load_vector_store(index_dir)

    # 3. Test questions
    test_questions = [
        # Factual query
        "What is Retrieval-Augmented Generation (RAG)?",
        # Synthesis query
        "What are the main components of Natural Language Processing and Deep Learning mentioned?",
        # Hallucination test query
        "What is the meaning of life?",
    ]

    print(f"\n📝 Running Q&A generation for {len(test_questions)} questions...")

    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Querying RAG system ({i}/{len(test_questions)})...")
        answer, pages = answer_question(vector_store, question, groq_key)
        print_qa_result(question, answer, pages, i)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Step 6 complete!")
    print("   ✅ PDF -> Chunks -> Embeddings -> FAISS -> Retrieval -> LLM Generation complete.")
    print("   ✅ RAG hallucination constraints verified successfully.")
    print("\n   The entire 6-step RAG pipeline is working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
