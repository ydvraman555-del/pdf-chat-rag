"""
step2_chunking.py — RAG Step 2: Split document text into smaller chunks.

WHAT THIS DOES:
1. Loads the PDF using PyPDFLoader (from Step 1).
2. Uses LangChain's RecursiveCharacterTextSplitter to split the pages.
3. Configures:
   - chunk_size = 1000 characters (max length per chunk)
   - chunk_overlap = 200 characters (overlap between adjacent chunks)

WHY OVERLAP IS ESSENTIAL:
If a sentence falls exactly on a boundary, without overlap it gets sliced.
The first chunk gets the beginning, the second gets the ending. Neither chunk
has the full context! Overlap keeps boundaries intact, ensuring context is preserved.

SECURITY:
- Processing runs locally in memory. No network requests are made.
"""

import sys
from pathlib import Path


def load_pdf_pages(pdf_path: Path) -> list:
    from langchain_community.document_loaders import PyPDFLoader
    if not pdf_path.exists():
        print(f"❌ PDF not found. Run 'python create_sample_pdf.py' first.")
        sys.exit(1)
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def chunk_documents(pages: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Split page-level Document objects into smaller overlapping chunks.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # RecursiveCharacterTextSplitter tries to split by paragraphs (\n\n),
    # sentences (\n or .), and words (spaces) recursively until chunks fit.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    return chunks


def display_chunks(pages: list, chunks: list):
    """
    Display details about the generated text chunks.
    """
    print("\n" + "=" * 60)
    print("🧩 PDF CHUNKING RESULTS")
    print("=" * 60)
    print(f"📖 Original pages:  {len(pages)}")
    print(f"🧩 Chunks created:  {len(chunks)}")
    print(f"📊 Average chunks/page: {len(chunks) / len(pages):.1f}")
    print("-" * 60)

    # Let's inspect the first chunk
    if chunks:
        first_chunk = chunks[0]
        print(f"\n🔍 FIRST CHUNK PREVIEW (Page {first_chunk.metadata.get('page') + 1}):")
        print("-" * 40)
        print(f"{first_chunk.page_content[:300].strip()}...")
        print("-" * 40)
        print(f"📏 Chunk length: {len(first_chunk.page_content)} characters")

    # Let's inspect the fifth chunk
    if len(chunks) >= 5:
        fifth_chunk = chunks[4]
        print(f"\n🔍 FIFTH CHUNK PREVIEW (Page {fifth_chunk.metadata.get('page') + 1}):")
        print("-" * 40)
        print(f"{fifth_chunk.page_content[:300].strip()}...")
        print("-" * 40)
        print(f"📏 Chunk length: {len(fifth_chunk.page_content)} characters")


def main():
    # Reconfigure stdout to support UTF-8 print (emojis on Windows)
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("✂️  RAG Step 2: Document Text Chunking")
    print("=" * 60)
    print()

    project_root = Path(__file__).parent
    pdf_path = project_root / "sample.pdf"

    # Step 1: Load pages
    pages = load_pdf_pages(pdf_path)

    # Step 2: Split into chunks
    print("✂️  Splitting document text into chunks (size=1000, overlap=200)...")
    chunks = chunk_documents(pages, chunk_size=1000, chunk_overlap=200)

    # Step 3: Show chunks
    display_chunks(pages, chunks)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Step 2 complete!")
    print("   ✅ Text split into smaller, overlapping chunks.")
    print("\n   Next: Convert text chunks into numerical vectors (Step 3).")
    print("=" * 60)


if __name__ == "__main__":
    main()
