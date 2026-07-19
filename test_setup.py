"""
test_setup.py — Verify local Ollama setup and required models.
"""

import sys
import os
import urllib.request
import json
from dotenv import load_dotenv


def main():
    # Reconfigure stdout to support UTF-8 print (emojis on Windows)
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # Step 1: Load .env (if present, to check OLLAMA_HOST override)
    load_dotenv()
    print("🔍 Checking environment setup for Local Ollama...\n")

    # Step 2: Verify Ollama is running
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    print(f"📡 Connecting to Ollama server at: {ollama_host}...")
    try:
        # Simple HTTP request to see if Ollama responds and get list of tags
        req = urllib.request.Request(f"{ollama_host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            installed_models = [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"❌ Could not connect to Ollama server.")
        print(f"   → Make sure Ollama is installed and running on your system.")
        print(f"   → Download link: https://ollama.com")
        print(f"   → Error: {e}")
        sys.exit(1)

    print("✅ Successfully connected to Ollama server.")

    # Step 3: Check models
    required_models = ["gemma2:2b", "nomic-embed-text"]
    missing_models = []
    
    # Normailze names to check them robustly (remove :latest tag matching)
    normalized_installed = []
    for m in installed_models:
        normalized_installed.append(m)
        if ":" in m:
            normalized_installed.append(m.split(":")[0])

    for req_model in required_models:
        # Check direct match or without tag if tag matches
        if req_model not in normalized_installed:
            missing_models.append(req_model)
            
    if missing_models:
        print("\n❌ Missing required models in Ollama:")
        for m in missing_models:
            print(f"   → '{m}' is not pulled.")
            print(f"     Run command: ollama pull {m}")
        sys.exit(1)
        
    print("✅ All required models (gemma2:2b, nomic-embed-text) are available.")

    # Step 4: Test imports
    try:
        import langchain
        import langchain_ollama
        import faiss
        import pypdf
        import streamlit
        print("✅ All packages imported successfully")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   → Run: pip install -r requirements.txt")
        sys.exit(1)

    # Step 5: Test Ollama chat generation
    print("\n🔗 Testing local model generation (gemma2:2b)...")
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="gemma2:2b", temperature=0)
        response = llm.invoke("Say 'hello' in one word.")
        print(f"✅ Local Gemma responded: '{response.content.strip()}'")
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        sys.exit(1)

    # Step 6: Test Ollama embeddings
    print("📐 Testing local embeddings (nomic-embed-text)...")
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector = embeddings.embed_query("test")
        print(f"✅ Embedding test succeeded (vector size: {len(vector)} dimensions)")
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 50)
    print("🎉 Local Setup complete! All checks passed.")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Place your PDF as sample.pdf in the project folder")
    print("  2. Run: streamlit run app.py")


if __name__ == "__main__":
    main()

