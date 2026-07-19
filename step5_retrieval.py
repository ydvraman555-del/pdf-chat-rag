"""
step5_retrieval.py — RAG Step 5: Retrieve relevant chunks using similarity search.

WHAT THIS DOES:
1. Loads the FAISS index that we saved locally in Step 4.
2. Accepts a natural language query (user question).
3. Converts the query into a vector representation (using local nomic-embed-text).
4. Runs L2 similarity search inside the FAISS database to find the top k=3 closest vectors.
5. Displays the retrieved text chunks alongside their source page numbers.

WHY RETRIEVAL IS ESSENTIAL (The "R" in RAG):
Retrieval filters out 99% of irrelevant document text, leaving only the most
specific passages related to the user's question. This minimizes the tokens we send
to the LLM, reducing latency and avoiding prompt length limits.

SECURITY:
- Runs 100% locally. The search is calculated locally on your CPU/RAM.
"""

import sys
from pathlib import Path


def load_local_vector_store(index_dir: Path):
    """
    Load the persisted FAISS vector store.
    """
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS

    if not index_dir.exists():
        print(f"❌ FAISS index directory '{index_dir.resolve()}' not found.")
        print("   → Run 'python step4_vector_store.py' to generate and save it first.")
        sys.exit(1)

    # Initialize local embeddings to map search query
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    try:
        # Load local FAISS index
        vector_store = FAISS.load_local(
            folder_path=str(index_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        sys.exit(1)


def perform_retrieval(vector_store, query: str, k: int = 3):
    """
    Query the vector store and print the top k retrieved text chunks.
    """
    print(f"❓ User query: \"{query}\"")
    print(f"🔍 Searching vector store for top k={k} matches...")

    try:
        # similarity_search embeds the query string and retrieves matches
        results = vector_store.similarity_search(query, k=k)

        print("\n" + "=" * 60)
        print("🔍 SEMANTIC RETRIEVAL RESULTS")
        print("=" * 60)

        for i, doc in enumerate(results, 1):
            page_num = doc.metadata.get("page", 0) + 1  # 0-indexed → 1-indexed
            print(f"\n📌 Match #{i} (Page {page_num}):")
            print("-" * 45)
            # Print a snippet of the retrieved chunk
            print(f"{doc.page_content[:300].strip()}...")
            print("-" * 45)
            print(f"📏 Full chunk size: {len(doc.page_content)} characters")

        print("=" * 60)

    except Exception as e:
        print(f"❌ Similarity search failed: {e}")
        sys.exit(1)


def main():
    # Reconfigure stdout to support UTF-8 print (emojis on Windows)
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("🔍 RAG Step 5: Semantic Context Retrieval")
    print("=" * 60)
    print()

    project_root = Path(__file__).parent
    index_dir = project_root / "faiss_index"

    # Step 1: Load FAISS index from disk
    vector_store = load_local_vector_store(index_dir)

    # Step 2: Perform similarity search on a query
    query = "What is Retrieval-Augmented Generation?"
    perform_retrieval(vector_store, query, k=3)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Step 5 complete!")
    print("   ✅ Top relevant chunks successfully retrieved from FAISS.")
    print("\n   Next: Generate grounded answers using Groq LLM + context (Step 6).")
    print("=" * 60)


if __name__ == "__main__":
    main()
