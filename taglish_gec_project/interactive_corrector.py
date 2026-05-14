#!/usr/bin/env python3
"""
Interactive Taglish Grammar Corrector
Load model ONCE and correct text continuously without reloading
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from standalone_inference import TaglishGrammarCorrector


MODEL_PATH = Path(__file__).resolve().parent / "taglish_gec_model"


def main():
    """Interactive corrector with persistent model loading."""
    
    print("=" * 80)
    print("TAGLISH GRAMMAR CORRECTOR - INTERACTIVE MODE")
    print("=" * 80)
    print("\n⏳ Loading model (this happens once)...\n")
    
    # Load model ONCE - this is the key!
    try:
        corrector = TaglishGrammarCorrector(
            model_path=MODEL_PATH,
            device="cuda"
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nFalling back to CPU...")
        try:
            corrector = TaglishGrammarCorrector(
                model_path=MODEL_PATH,
                device="cpu"
            )
        except Exception as e2:
            print(f"❌ Failed on CPU too: {e2}")
            return
    
    print("✅ Model loaded! Ready for corrections.\n")
    print("-" * 80)
    print("INSTRUCTIONS:")
    print("  • Type Taglish text and press Enter")
    print("  • Type 'exit' or 'quit' to stop")
    print("  • Press Ctrl+C to quit immediately")
    print("-" * 80 + "\n")
    
    correction_count = 0
    
    try:
        while True:
            # Get user input
            text = input("📝 Enter Taglish text: ").strip()
            
            # Check for exit commands
            if not text:
                print("   (empty input, try again)\n")
                continue
            
            if text.lower() in ('exit', 'quit', 'q', 'x'):
                break
            
            # Correct (model is already loaded, no reload!)
            try:
                corrected = corrector.correct(text, temperature=0.05)
                print(f"✅ Corrected: {corrected}\n")
                correction_count += 1
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user.")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"📊 Session Summary: {correction_count} corrections made")
    print("=" * 80)
    print("\n✨ Thank you for using Taglish Grammar Corrector!")


if __name__ == "__main__":
    main()
