"""
step3_embeddings.py — RAG Step 3: Convert text into numerical vector embeddings.

WHAT THIS DOES:
1. Explains what a vector embedding is (converting text to coordinates in semantic space).
2. Uses local OllamaEmbeddings (powered by 'nomic-embed-text') to embed text.
3. Converts a sample sentence into a dense numerical vector.
4. Prints vector statistics: dimensionality (768), data type, and preview of coordinates.

HOW EMBEDDINGS WORK (Semantic Mapping):
- TF-IDF: maps words to coordinates based on literal occurrences. "economy" and "financial" get different directions.
- Embeddings: maps words to coordinates based on meaning. "economy" and "financial" get nearby directions.

SECURITY:
- Runs 100% locally on your machine via Ollama.
- No document text or query data is sent to the cloud, ensuring complete privacy.
"""

import sys


def get_local_embeddings():
    """
    Initialize the local Ollama embedding model.
    """
    from langchain_ollama import OllamaEmbeddings
    
    # Initialize connection to local Ollama server running the 'nomic-embed-text' model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embeddings


def demonstrate_embeddings(embeddings, test_text: str):
    """
    Embed a sample text and print information about the resulting vector.
    """
    print(f"📝 Text to embed: '{test_text}'")
    print("🔄 Generating embedding locally via Ollama...")

    try:
        # embed_query() converts a single text string into a list of floats (vector)
        vector = embeddings.embed_query(test_text)

        print("\n" + "=" * 60)
        print("📐 EMBEDDING VECTOR INFO")
        print("=" * 60)
        print(f"📊 Vector dimensions: {len(vector)} dimensions")
        print(f"📦 Data type:         {type(vector)} of {type(vector[0]).__name__}s")
        print(f"🔢 First 5 coordinates: {[round(c, 6) for c in vector[:5]]}")
        print("-" * 60)
        print("💡 INSIGHT:")
        print(f"   - Ollama's 'nomic-embed-text' translates any text into a fixed-size {len(vector)}-D vector.")
        print(f"   - Closer distances between vectors = closer semantic meanings.")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        print("   → Make sure Ollama is running ('ollama serve') and model is pulled ('ollama pull nomic-embed-text').")
        sys.exit(1)


def main():
    # Reconfigure stdout to support UTF-8 print (emojis on Windows)
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("📐 RAG Step 3: Semantic Vector Embeddings")
    print("=" * 60)
    print()

    # Step 1: Initialize local Ollama embedding connection
    embeddings = get_local_embeddings()

    # Step 2: Embed a test phrase
    test_text = "Machine Learning algorithms discover patterns in data."
    demonstrate_embeddings(embeddings, test_text)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Step 3 complete!")
    print("   ✅ Text successfully converted to dense semantic vector.")
    print("\n   Next: Store these vectors in an in-memory vector database (Step 4).")
    print("=" * 60)


if __name__ == "__main__":
    main()
