"""
step3_embeddings.py — RAG Step 2: Convert chunks into embeddings + store in FAISS (Local Ollama).

WHAT THIS DOES (in ML terms you already know):
1. Takes each text chunk from Step 1
2. Converts it into a fixed-size numerical vector (embedding) locally using Ollama
3. Stores all vectors in a FAISS index (like sklearn's NearestNeighbors, but faster)
4. Runs a similarity search to find chunks most relevant to a query

WHY FAISS instead of ChromaDB?
ChromaDB requires C++ Build Tools on Windows. FAISS ships pre-built binaries
and is actually faster for in-memory search. The LangChain API is nearly identical.

SECURITY: Run 100% locally. No API keys are required for embeddings, protecting raw data.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
import os

# ─────────────────────────────────────────────
# Step 0: Load and validate environment
# ─────────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────────
# Step 1: Reuse chunking logic from Step 2
# ─────────────────────────────────────────────
def load_and_chunk_pdf(pdf_path: Path) -> list:
    """
    Load a PDF and split it into overlapping chunks.
    Reuses the same logic from step2_chunking.py.

    Returns:
        List of Document objects (chunked).
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path.resolve()}")
        print(f"   → Place your PDF as 'sample.pdf' in the project root.")
        sys.exit(1)

    # Load PDF → one Document per page
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    # Check for empty/scanned PDFs
    total_text = sum(len(p.page_content.strip()) for p in pages)
    if total_text == 0:
        print("❌ PDF has no extractable text (possibly scanned).")
        print("   → Use a text-based PDF or apply OCR first.")
        sys.exit(1)

    # Split pages into smaller, overlapping chunks
    # Same parameters as step2_chunking.py for consistency
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # ~250 tokens per chunk
        chunk_overlap=200,     # 200 chars shared between adjacent chunks
        length_function=len,
    )
    chunks = splitter.split_documents(pages)

    print(f"✅ Loaded {len(pages)} pages → {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# Step 2: Create embeddings + vector store
# ─────────────────────────────────────────────
def create_vector_store(chunks: list):
    """
    Convert text chunks into embeddings and store them in FAISS.

    HOW EMBEDDINGS WORK (in ML terms):
    You know feature vectors from sklearn — e.g., TF-IDF gives you a sparse vector
    for each document. Embeddings are the same idea but DENSE and SEMANTIC:
      - TF-IDF: "king" and "monarch" get DIFFERENT vectors (different words)
      - Embeddings: "king" and "monarch" get SIMILAR vectors (similar meaning)

    Ollama's nomic-embed-text model:
      - Input: any text string (chunk or query)
      - Output: a 768-dimensional dense vector
      - Similar texts → vectors that are close in 768-D space

    WHAT FAISS DOES (in ML terms):
    Think of sklearn's NearestNeighbors, but optimized for high-dimensional vectors:
      - sklearn NN: brute-force, scans all vectors → O(n) per query
      - FAISS: uses IndexFlatL2 (exact L2 search) for small datasets,
               or HNSW/IVF (approximate search) for millions of vectors
      - For our use case (~50 chunks), exact search is fast enough

    Args:
        chunks: List of Document objects from load_and_chunk_pdf.

    Returns:
        Tuple of (FAISS vector store, embeddings model).
    """
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS

    # Initialize the embedding model
    # "nomic-embed-text" is Ollama's local text embedding model (768-D)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    print("\n🔄 Creating embeddings and building FAISS index...")
    print(f"   Embedding {len(chunks)} chunks via Local Ollama...")
    print(f"   (This processes each chunk locally on your machine)")

    try:
        # FAISS.from_documents does THREE things:
        # 1. Sends each chunk's text to Ollama → gets back 768-D vectors
        # 2. Builds a FAISS index (in-memory) from those vectors
        # 3. Stores the original text + metadata alongside the vectors
        #
        # In-memory = no files written to disk. The index lives only in RAM.
        vector_store = FAISS.from_documents(
            documents=chunks,       # Our text chunks with metadata
            embedding=embeddings,   # The model that converts text → vectors
        )
        print(f"✅ FAISS vector store created with {len(chunks)} vectors")

    except Exception as e:
        print(f"\n❌ Embedding error: {e}")
        print("   → Make sure Ollama is running locally (e.g. run 'ollama serve')")
        print("   → Verify that 'nomic-embed-text' is pulled ('ollama pull nomic-embed-text')")
        sys.exit(1)

    return vector_store, embeddings


# ─────────────────────────────────────────────
# Step 3: Show embedding dimensions
# ─────────────────────────────────────────────
def show_embedding_info(embeddings) -> None:
    """
    Embed a test string to reveal the vector dimensions.

    WHY THIS MATTERS:
    Ollama's nomic-embed-text outputs 768 dimensions — this is fixed.
    768-D captures rich semantic meaning while being highly efficient to search.
    """
    print("\n📐 EMBEDDING DIMENSIONS:")
    print("-" * 40)

    try:
        # embed_query() converts a single string → one vector
        test_vector = embeddings.embed_query("test")

        print(f"   Input: 'test' (4 characters)")
        print(f"   Output: vector of {len(test_vector)} dimensions")
        print(f"   Type: list of floats")
        print(f"   First 5 values: {[round(v, 6) for v in test_vector[:5]]}")
        print(f"\n   → Every chunk and every query becomes a {len(test_vector)}-D vector")
        print(f"   → Similar texts → nearby vectors in {len(test_vector)}-D space")

    except Exception as e:
        print(f"   ⚠️  Could not get embedding dimensions: {e}")


# ─────────────────────────────────────────────
# Step 4: Test similarity search
# ─────────────────────────────────────────────
def test_similarity_search(vector_store) -> None:
    """
    Run a similarity search to find chunks relevant to a query.

    WHAT HAPPENS MATHEMATICALLY:
    1. Your query text is converted to a 768-D vector (same embedding model)
    2. FAISS computes the distance between the query vector and ALL stored vectors
    3. Returns the k closest vectors (and their associated text chunks)

    DISTANCE METRIC:
    FAISS uses L2 (Euclidean) distance by default. Lower distance = more similar.
    """
    query = "What is the main topic of this document?"

    print(f"\n🔍 SIMILARITY SEARCH TEST")
    print(f"   Query: \"{query}\"")
    print(f"   Retrieving top k=3 most relevant chunks...")
    print("=" * 60)

    try:
        # similarity_search() does:
        # 1. Embeds the query string → 768-D vector
        # 2. Searches FAISS index for k nearest neighbors
        # 3. Returns the Document objects (text + metadata) for those neighbors
        results = vector_store.similarity_search(
            query=query,  # Your question as natural language
            k=3,          # Return top 3 most similar chunks
        )

        for i, doc in enumerate(results, 1):
            page_num = doc.metadata.get("page", "N/A")
            # Add 1 to convert from 0-indexed to human-readable page numbers
            display_page = page_num + 1 if isinstance(page_num, int) else page_num
            content_preview = doc.page_content[:200]

            print(f"\n📌 Result {i} — Page {display_page}")
            print(f"   {content_preview}...")
            print("-" * 60)

        print(f"\n✅ Retrieved {len(results)} relevant chunks for your query!")

    except Exception as e:
        print(f"\n❌ Search error: {e}")


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
    print("📊 RAG Step 2: Embeddings + FAISS Vector Store (Local Ollama)")
    print("=" * 60)
    print()

    # 1. Load and chunk the PDF (reuses Step 1 logic)
    project_root = Path(__file__).parent
    pdf_path = project_root / "sample.pdf"
    chunks = load_and_chunk_pdf(pdf_path)

    # 2. Create embeddings + store in FAISS
    vector_store, embeddings = create_vector_store(chunks)

    # 3. Show embedding dimensions (helps you understand the vector space)
    show_embedding_info(embeddings)

    # 4. Test similarity search
    test_similarity_search(vector_store)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Step 2 complete!")
    print("   ✅ PDF chunked → local embeddings created → stored in FAISS")
    print("   ✅ Similarity search working (fully local)")
    print("\n   Next: Build the full RAG chain (retriever + Groq LLM + prompt)")
    print("=" * 60)


if __name__ == "__main__":
    main()
