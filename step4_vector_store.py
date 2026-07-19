"""
step4_vector_store.py — RAG Step 4: Index chunks in FAISS and persist index to disk.

WHAT THIS DOES:
1. Loads the PDF and splits it into overlapping chunks (Steps 1 & 2).
2. Uses local Ollama nomic-embed-text to convert all chunks to vectors (Step 3).
3. Indexes all vectors in FAISS (Facebook AI Similarity Search).
4. Saves the FAISS index to the local file system (folder 'faiss_index').
5. Loads the index back from disk to demonstrate persistence.

WHY LOCAL PERSISTENCE MATTERS:
In a production app, embedding a large PDF (e.g. 500 pages) can take time.
By saving the FAISS index locally, we only embed the PDF ONCE. Future sessions
can load the index instantly from disk in milliseconds without repeating embeddings.

SECURITY:
- Both FAISS and Ollama run entirely locally on your computer.
- Saved files ('faiss_index/index.faiss' and 'faiss_index/index.pkl') remain on your hard drive.
"""

import sys
from pathlib import Path


def build_and_save_index(pdf_path: Path, output_dir: Path):
    """
    Load PDF → Chunk → Embed → Store in FAISS → Persist to disk.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS

    print("📄 Parsing PDF and creating chunks...")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    print(f"   Generated {len(chunks)} chunks from {len(pages)} pages.")

    print("\n🔄 Initializing Ollama local embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("🔄 Building FAISS index (generating vectors locally)...")
    try:
        # FAISS.from_documents embeds the chunks and loads them into the index
        vector_store = FAISS.from_documents(chunks, embeddings)
        print(f"✅ FAISS index successfully created with {len(chunks)} vectors.")

        # Persist index locally
        print(f"💾 Saving FAISS index to folder: '{output_dir.name}'...")
        vector_store.save_local(str(output_dir))
        print("✅ Index saved successfully.")
        return embeddings

    except Exception as e:
        print(f"❌ Failed to build/save FAISS vector store: {e}")
        print("   → Make sure Ollama is running locally.")
        sys.exit(1)


def verify_loaded_index(index_dir: Path, embeddings):
    """
    Load the saved FAISS index from disk and display basic verification details.
    """
    from langchain_community.vectorstores import FAISS

    print(f"\n📂 Loading FAISS index from: '{index_dir.name}'...")
    try:
        # allow_dangerous_deserialization=True is required to load FAISS pickle files locally.
        # SECURITY: Only set this to True when loading files you created yourself.
        loaded_store = FAISS.load_local(
            folder_path=str(index_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Successfully loaded FAISS index back from disk!")
        
        # Access underlying index structure to show total vectors stored
        total_vectors = loaded_store.index.ntotal
        print(f"📊 Verified vectors stored in loaded index: {total_vectors}")

    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        sys.exit(1)


def main():
    # Reconfigure stdout to support UTF-8 print (emojis on Windows)
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("💾 RAG Step 4: Vector Indexing & Local Persistence")
    print("=" * 60)
    print()

    project_root = Path(__file__).parent
    pdf_path = project_root / "sample.pdf"
    index_dir = project_root / "faiss_index"

    # Step 1: Build FAISS index and save it locally
    embeddings = build_and_save_index(pdf_path, index_dir)

    # Step 2: Load it back to verify persistence
    verify_loaded_index(index_dir, embeddings)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Step 4 complete!")
    print("   ✅ Document chunk vectors indexed and saved to disk.")
    print("\n   Next: Query the vector index to retrieve relevant chunks (Step 5).")
    print("=" * 60)


if __name__ == "__main__":
    main()
