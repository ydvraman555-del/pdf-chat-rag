"""
step2_chunking.py — RAG Step 1: Load a PDF and split it into chunks.

WHY THIS STEP?
In RAG, we don't feed the entire PDF to the LLM (it won't fit, and even if it
did, retrieval would be imprecise). Instead, we break the document into small,
overlapping chunks that can later be embedded and searched individually.

SECURITY: Loads .env at the top as a good habit for future steps.
"""

import sys
from pathlib import Path  # Safe, cross-platform file path handling

from dotenv import load_dotenv  # Load .env even if not needed here — good habit

# ─────────────────────────────────────────────
# Step 0: Load environment variables
# ─────────────────────────────────────────────
load_dotenv()


def load_pdf(pdf_path: Path) -> list:
    """
    Load a PDF file and return a list of Document objects (one per page).

    Each Document has:
      - page_content: the extracted text from that page
      - metadata: dict with 'source' (file path) and 'page' (0-indexed page number)

    Args:
        pdf_path: Path object pointing to the PDF file.

    Returns:
        List of langchain Document objects.

    Raises:
        FileNotFoundError: If the PDF doesn't exist.
        ValueError: If the PDF is encrypted, corrupted, or has no text.
    """
    # --- Guard: File must exist ---
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}\n"
            f"   → Place your PDF as 'sample.pdf' in the project root."
        )

    # --- Guard: Must be a .pdf file ---
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: '{pdf_path.suffix}'")

    # Import here to keep top-level imports clean and error messages clear
    from langchain_community.document_loaders import PyPDFLoader

    try:
        # PyPDFLoader reads the PDF page by page
        # Each page becomes one Document object with text + metadata
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # Returns List[Document]
    except Exception as e:
        error_msg = str(e).lower()
        if "encrypt" in error_msg or "password" in error_msg:
            raise ValueError(
                "PDF appears to be encrypted/password-protected.\n"
                "   → Use an unprotected PDF or remove the password first."
            ) from e
        else:
            raise ValueError(
                f"Failed to read PDF (possibly corrupted).\n"
                f"   → Error: {e}"
            ) from e

    # --- Guard: PDF must contain extractable text ---
    # Scanned PDFs (images of text) will load but have empty page_content
    total_text = sum(len(page.page_content.strip()) for page in pages)
    if total_text == 0:
        raise ValueError(
            "PDF loaded but contains NO extractable text.\n"
            "   → This is likely a scanned PDF (images, not text).\n"
            "   → Use OCR (e.g., pytesseract) to extract text first,\n"
            "     or use a text-based PDF."
        )

    return pages


def chunk_documents(pages: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Split page-level Documents into smaller, overlapping chunks.

    WHY OVERLAP?
    Without overlap, a sentence at the boundary of two chunks gets cut in half.
    The first chunk loses the ending, the second loses the beginning — neither
    chunk has the full context. Overlap ensures boundary sentences appear in
    BOTH adjacent chunks, preserving context.

    Think of it like a sliding window (you know this from ML!):
      - chunk_size = window width
      - chunk_overlap = stride overlap (window moves by chunk_size - chunk_overlap)

    Args:
        pages: List of Document objects (from load_pdf).
        chunk_size: Max characters per chunk (not tokens — chars are simpler to reason about).
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of Document objects, now chunked (many more than original pages).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # RecursiveCharacterTextSplitter tries to split at natural boundaries:
    #   1. First tries "\n\n" (paragraph breaks)
    #   2. Then "\n" (line breaks)
    #   3. Then " " (word boundaries)
    #   4. Last resort: character-level split
    # This preserves meaning better than a naive every-N-chars split.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # Max chars per chunk (~250 tokens for English)
        chunk_overlap=chunk_overlap,  # 200 chars shared between adjacent chunks
        length_function=len,         # Use character count (not token count)
        is_separator_regex=False,    # Treat separators as literal strings
    )

    # split_documents preserves metadata from the original page Documents
    # Each chunk inherits the 'source' and 'page' from its parent page
    chunks = text_splitter.split_documents(pages)

    return chunks


def display_results(pages: list, chunks: list, pdf_path: Path) -> None:
    """Print a clear summary of the loading and chunking results."""

    print("=" * 60)
    print(f"📄 PDF LOADING & CHUNKING RESULTS")
    print(f"   Source: {pdf_path.name}")
    print("=" * 60)

    # --- Page stats ---
    print(f"\n📖 Total pages loaded:  {len(pages)}")
    print(f"🧩 Total chunks created: {len(chunks)}")

    # Show the ratio — helps you understand the chunking granularity
    if len(pages) > 0:
        ratio = len(chunks) / len(pages)
        print(f"📊 Chunks per page (avg): {ratio:.1f}")

    # --- First chunk preview ---
    print("\n" + "-" * 60)
    print("🔍 FIRST CHUNK PREVIEW (first 300 chars):")
    print("-" * 60)

    first_chunk = chunks[0]
    # Show only first 300 chars to keep output readable
    preview = first_chunk.page_content[:300]
    print(preview)
    if len(first_chunk.page_content) > 300:
        print(f"... [{len(first_chunk.page_content) - 300} more chars]")

    # --- First chunk metadata ---
    print("\n" + "-" * 60)
    print("🏷️  FIRST CHUNK METADATA:")
    print("-" * 60)
    # Metadata is a dict — typically contains 'source' and 'page'
    for key, value in first_chunk.metadata.items():
        print(f"   {key}: {value}")
    print(f"   chunk_length: {len(first_chunk.page_content)} chars")

    # --- 5th chunk (if exists) ---
    print("\n" + "-" * 60)
    if len(chunks) >= 5:
        fifth_chunk = chunks[4]  # 0-indexed, so index 4 = 5th chunk
        print(f"📌 5TH CHUNK — page number: {fifth_chunk.metadata.get('page', 'N/A')}")
        print(f"   Length: {len(fifth_chunk.page_content)} chars")
        print(f"   Preview: {fifth_chunk.page_content[:100]}...")
    else:
        print(f"📌 5TH CHUNK — Not available (only {len(chunks)} chunks total)")

    print("\n" + "=" * 60)
    print("✅ Chunking complete! Ready for Step 2: Embeddings + Vector Store")
    print("=" * 60)


def main():
    """Main entry point — load PDF, chunk it, display results."""

    # Resolve PDF path relative to this script's directory (project root)
    # Using pathlib ensures this works on Windows, Mac, and Linux
    project_root = Path(__file__).parent
    pdf_path = project_root / "sample.pdf"

    print("🔍 Loading PDF...\n")

    try:
        # Step 1: Load the PDF into page-level Documents
        pages = load_pdf(pdf_path)
        print(f"✅ PDF loaded successfully: {len(pages)} pages\n")

        # Step 2: Split pages into smaller, overlapping chunks
        print("✂️  Splitting into chunks (size=1000, overlap=200)...\n")
        chunks = chunk_documents(pages, chunk_size=1000, chunk_overlap=200)

        # Step 3: Display results
        display_results(pages, chunks, pdf_path)

    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
