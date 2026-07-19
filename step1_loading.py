"""
step1_loading.py — RAG Step 1: Parse a PDF page-by-page.

WHAT THIS DOES:
1. Locates 'sample.pdf' in the project root.
2. Uses LangChain's PyPDFLoader (powered by the pypdf library) to parse it.
3. Extracts raw page text and metadata (source document path and 0-indexed page numbers).

SECURITY:
- Raw document content remains local.
- No cloud calls or API keys are required for this parsing step.
"""

import sys
from pathlib import Path


def load_pdf(pdf_path: Path) -> list:
    """
    Load a PDF file and return a list of Document objects (one per page).
    """
    from langchain_community.document_loaders import PyPDFLoader

    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path.resolve()}")
        print(f"   → Run 'python create_sample_pdf.py' to generate a sample PDF first.")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print(f"❌ Expected a .pdf file, got: '{pdf_path.suffix}'")
        sys.exit(1)

    print(f"🔍 Loading PDF file from: {pdf_path.name}...")

    try:
        # PyPDFLoader parses the PDF and returns a list of Document objects
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        return pages
    except Exception as e:
        print(f"❌ Failed to read PDF: {e}")
        sys.exit(1)


def display_pages(pages: list):
    """
    Display basic details and previews of the loaded pages.
    """
    print("\n" + "=" * 60)
    print("📄 PDF LOADING RESULTS")
    print("=" * 60)
    print(f"📖 Total pages loaded: {len(pages)}")
    print("-" * 60)

    # Let's inspect a couple of pages
    pages_to_preview = [0, 1]  # Pages 1 and 2 (0-indexed)

    for p_idx in pages_to_preview:
        if p_idx < len(pages):
            page = pages[p_idx]
            print(f"\n📌 Page {p_idx + 1} Metadata:")
            print(f"   Source file: {page.metadata.get('source')}")
            print(f"   Page number: {page.metadata.get('page')}")
            print(f"   Text length: {len(page.page_content)} characters")
            print(f"\n--- Page {p_idx + 1} Content Preview (first 250 chars) ---")
            print(f"{page.page_content[:250].strip()}...")
            print("-" * 60)


def main():
    # Reconfigure stdout to support UTF-8 print (emojis on Windows)
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("📄 RAG Step 1: PDF Document Parsing")
    print("=" * 60)
    print()

    # Resolve path to sample.pdf
    project_root = Path(__file__).parent
    pdf_path = project_root / "sample.pdf"

    # Step 1: Load the PDF
    pages = load_pdf(pdf_path)

    # Step 2: Show output
    display_pages(pages)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Step 1 complete!")
    print("   ✅ PDF successfully loaded and parsed page-by-page.")
    print("\n   Next: Split pages into smaller overlapping chunks (Step 2).")
    print("=" * 60)


if __name__ == "__main__":
    main()
