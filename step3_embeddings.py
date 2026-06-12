"""
step3_embeddings.py — RAG Step 2: Convert chunks into embeddings + store in FAISS.

WHAT THIS DOES (in ML terms you already know):
1. Takes each text chunk from Step 1
2. Converts it into a fixed-size numerical vector (embedding) using Google's API
3. Stores all vectors in a FAISS index (like sklearn's NearestNeighbors, but faster)
4. Runs a similarity search to find chunks most relevant to a query

WHY FAISS instead of ChromaDB?
ChromaDB requires C++ Build Tools on Windows. FAISS ships pre-built binaries
and is actually faster for in-memory search. The LangChain API is nearly identical.

SECURITY: API key loaded from .env, never printed.
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
    Load and validate the Google API key from environment.

    Returns:
        The API key string (never printed).

    Raises:
        SystemExit: If key is missing or still the placeholder.
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file.")
        print("   → Copy .env.example to .env and add your real key.")
        sys.exit(1)

    if api_key == "your_gemini_api_key_here":
        print("❌ GOOGLE_API_KEY is still the placeholder value.")
        print("   → Replace it with your real Gemini API key in .env")
        sys.exit(1)

    # Print length only — NEVER the actual key
    print(f"✅ API key loaded (length: {len(api_key)} chars)")
    return api_key


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
def create_vector_store(chunks: list, api_key: str):
    """
    Convert text chunks into embeddings and store them in FAISS.

    HOW EMBEDDINGS WORK (in ML terms):
    You know feature vectors from sklearn — e.g., TF-IDF gives you a sparse vector
    for each document. Embeddings are the same idea but DENSE and SEMANTIC:
      - TF-IDF: "king" and "monarch" get DIFFERENT vectors (different words)
      - Embeddings: "king" and "monarch" get SIMILAR vectors (similar meaning)

    Google's gemini-embedding-001 model:
      - Input: any text string (chunk or query)
      - Output: a 3072-dimensional dense vector
      - Similar texts → vectors that are close in 3072-D space

    WHAT FAISS DOES (in ML terms):
    Think of sklearn's NearestNeighbors, but optimized for high-dimensional vectors:
      - sklearn NN: brute-force, scans all vectors → O(n) per query
      - FAISS: uses IndexFlatL2 (exact L2 search) for small datasets,
               or HNSW/IVF (approximate search) for millions of vectors
      - For our use case (~50 chunks), exact search is fast enough

    Args:
        chunks: List of Document objects from load_and_chunk_pdf.
        api_key: Google API key for embedding generation.

    Returns:
        Tuple of (FAISS vector store, embeddings model).
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS

    # Initialize the embedding model
    # "gemini-embedding-001" is Google's latest text embedding model
    # It converts any text → 3072-dimensional vector
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    print("\n🔄 Creating embeddings and building FAISS index...")
    print(f"   Embedding {len(chunks)} chunks via Google API...")
    print(f"   (This sends each chunk's text to Google's embedding model)")

    try:
        # FAISS.from_documents does THREE things:
        # 1. Sends each chunk's text to Google's API → gets back 3072-D vectors
        # 2. Builds a FAISS index (in-memory) from those vectors
        # 3. Stores the original text + metadata alongside the vectors
        #
        # In-memory = no files written to disk. The index lives only in RAM.
        # Perfect for development. For production, you'd use vector_store.save_local().
        vector_store = FAISS.from_documents(
            documents=chunks,       # Our text chunks with metadata
            embedding=embeddings,   # The model that converts text → vectors
        )
        print(f"✅ FAISS vector store created with {len(chunks)} vectors")

    except Exception as e:
        error_msg = str(e)
        # Scrub API key from error messages
        if api_key in error_msg:
            error_msg = error_msg.replace(api_key, "***REDACTED***")

        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"\n⚠️  Rate limited by Google API (429).")
            print(f"   The embedding API has separate quotas from the chat API.")
            print(f"   → Wait a few minutes and try again.")
            print(f"   → Free tier: ~1,500 requests/minute for embeddings")
            print(f"   → Check: https://ai.google.dev/gemini-api/docs/rate-limits")
            sys.exit(1)
        elif "invalid" in error_msg.lower() or "api key" in error_msg.lower():
            print(f"\n❌ Invalid API key.")
            print(f"   → Verify at: https://aistudio.google.com/app/apikey")
            sys.exit(1)
        else:
            print(f"\n❌ Embedding error: {error_msg}")
            sys.exit(1)

    return vector_store, embeddings


# ─────────────────────────────────────────────
# Step 3: Show embedding dimensions
# ─────────────────────────────────────────────
def show_embedding_info(embeddings, api_key: str) -> None:
    """
    Embed a test string to reveal the vector dimensions.

    WHY THIS MATTERS:
    When you did PCA in sklearn, you chose n_components (e.g., 50).
    Google's gemini-embedding-001 outputs 3072 dimensions — this is fixed.
    3072-D captures rich semantic meaning while being efficient to search.
    """
    print("\n📐 EMBEDDING DIMENSIONS:")
    print("-" * 40)

    try:
        # embed_query() converts a single string → one vector
        # This is what happens internally for every chunk AND every search query
        test_vector = embeddings.embed_query("test")

        print(f"   Input: 'test' (4 characters)")
        print(f"   Output: vector of {len(test_vector)} dimensions")
        print(f"   Type: list of floats")
        print(f"   First 5 values: {[round(v, 6) for v in test_vector[:5]]}")
        print(f"\n   → Every chunk and every query becomes a {len(test_vector)}-D vector")
        print(f"   → Similar texts → nearby vectors in {len(test_vector)}-D space")

    except Exception as e:
        error_msg = str(e)
        if api_key in error_msg:
            error_msg = error_msg.replace(api_key, "***REDACTED***")
        print(f"   ⚠️  Could not get embedding dimensions: {error_msg}")


# ─────────────────────────────────────────────
# Step 4: Test similarity search
# ─────────────────────────────────────────────
def test_similarity_search(vector_store, api_key: str) -> None:
    """
    Run a similarity search to find chunks relevant to a query.

    WHAT HAPPENS MATHEMATICALLY:
    1. Your query text is converted to a 3072-D vector (same embedding model)
    2. FAISS computes the distance between the query vector and ALL stored vectors
    3. Returns the k closest vectors (and their associated text chunks)

    DISTANCE METRIC:
    FAISS uses L2 (Euclidean) distance by default:
      distance = sqrt(sum((query_i - chunk_i)^2 for i in range(3072)))
    Lower distance = more similar.

    This is equivalent to cosine similarity when vectors are normalized
    (which Google's embeddings approximately are).

    WHY NOT COMPUTE MANUALLY?
    You COULD do: sklearn.metrics.pairwise.cosine_similarity(query, all_chunks)
    But FAISS uses optimized C++ with SIMD instructions — it's 10-100x faster,
    especially with millions of vectors. For 50 chunks it doesn't matter,
    but the pattern scales to production.
    """
    query = "What is the main topic of this document?"

    print(f"\n🔍 SIMILARITY SEARCH TEST")
    print(f"   Query: \"{query}\"")
    print(f"   Retrieving top k=3 most relevant chunks...")
    print("=" * 60)

    try:
        # similarity_search() does:
        # 1. Embeds the query string → 3072-D vector (API call)
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
        error_msg = str(e)
        if api_key in error_msg:
            error_msg = error_msg.replace(api_key, "***REDACTED***")

        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"\n⚠️  Rate limited during search (429).")
            print(f"   → The search requires one more API call to embed the query.")
            print(f"   → Wait a few minutes and try again.")
        else:
            print(f"\n❌ Search error: {error_msg}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("📊 RAG Step 2: Embeddings + FAISS Vector Store")
    print("=" * 60)
    print()

    # 1. Validate API key (NEVER printed)
    api_key = validate_api_key()

    # 2. Load and chunk the PDF (reuses Step 1 logic)
    project_root = Path(__file__).parent
    pdf_path = project_root / "sample.pdf"
    chunks = load_and_chunk_pdf(pdf_path)

    # 3. Create embeddings + store in FAISS
    vector_store, embeddings = create_vector_store(chunks, api_key)

    # 4. Show embedding dimensions (helps you understand the vector space)
    show_embedding_info(embeddings, api_key)

    # 5. Test similarity search
    test_similarity_search(vector_store, api_key)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Step 2 complete!")
    print("   ✅ PDF chunked → embeddings created → stored in FAISS")
    print("   ✅ Similarity search working")
    print("\n   Next: Build the full RAG chain (retriever + LLM + prompt)")
    print("=" * 60)


if __name__ == "__main__":
    main()
