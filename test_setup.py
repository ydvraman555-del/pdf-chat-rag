"""
test_setup.py — Verify environment setup and Gemini API key.
SECURITY: This script NEVER prints the actual API key value.
"""

import sys
import os
from dotenv import load_dotenv


def main():
    # Step 1: Load .env
    load_dotenv()
    print("🔍 Checking environment setup...\n")

    # Step 2: Check GOOGLE_API_KEY exists
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file.")
        print("   → Copy .env.example to .env and add your real key.")
        sys.exit(1)

    # Step 3: Check it's not the placeholder
    if api_key == "your_gemini_api_key_here":
        print("❌ GOOGLE_API_KEY is still the placeholder value.")
        print("   → Replace it with your real Gemini API key in .env")
        sys.exit(1)

    print(f"✅ GOOGLE_API_KEY loaded (length: {len(api_key)} chars)")

    # Step 4: Test imports
    try:
        import langchain
        import faiss
        import pypdf
        import streamlit
        print("✅ All packages imported successfully")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   → Run: pip install -r requirements.txt")
        sys.exit(1)

    # Step 5: Test Gemini API call
    print("\n🔗 Testing Gemini API connection...")

    models_to_try = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]

    for model_name in models_to_try:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
            )
            response = llm.invoke("Say 'hello' in one word.")
            print(f"✅ Gemini responded ({model_name}): {response.content}")
            break  # Success — no need to try other models
        except Exception as e:
            error_msg = str(e)
            # Scrub any accidental key leakage from error messages
            if api_key in error_msg:
                error_msg = error_msg.replace(api_key, "***REDACTED***")

            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"⚠️  Rate limited on {model_name}, ", end="")
                if model_name != models_to_try[-1]:
                    print("trying next model...")
                    continue
                print("all models rate-limited.")
                print("\n⏳ This is a QUOTA issue, not a code issue.")
                print("   Your setup is correct! To fix:")
                print("   1. Wait a few minutes and try again")
                print("   2. Check quota: https://ai.google.dev/gemini-api/docs/rate-limits")
                print("   3. Free tier resets daily")
                print("\n✅ Key + imports verified. You can proceed to build the app!")
                return  # Don't sys.exit(1) — setup IS valid
            else:
                print(f"❌ Gemini API error: {error_msg}")
                print("   → Verify your API key at: https://aistudio.google.com/app/apikey")
                sys.exit(1)

    # Final summary
    print("\n" + "=" * 50)
    print("🎉 Setup complete! All checks passed.")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Place your PDF as sample.pdf in the project folder")
    print("  2. Run: streamlit run app.py  (once you build the app)")


if __name__ == "__main__":
    main()
