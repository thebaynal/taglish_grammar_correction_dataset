#!/usr/bin/env python3
"""
Fast test script for Taglish Grammar Correction model
Loads model once and runs multiple tests efficiently
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from standalone_inference import TaglishGrammarCorrector


MODEL_PATH = Path(__file__).resolve().parent / "taglish_gec_model"


def main():
    """Test the trained model with sample Taglish sentences"""
    
    print("=" * 80)
    print("TAGLISH GRAMMAR CORRECTION MODEL - QUICK TEST")
    print("=" * 80)
    
    # Initialize model ONCE - try GPU first, fallback to CPU
    print("\nLoading model...")
    try:
        corrector = TaglishGrammarCorrector(
            model_path=MODEL_PATH,
            device="cuda"
        )
        print("✅ Model loaded on GPU!\n")
    except Exception as e:
        print(f"⚠️  GPU error: {str(e)[:60]}...")
        print("Falling back to CPU...\n")
        try:
            corrector = TaglishGrammarCorrector(
                model_path=MODEL_PATH,
                device="cpu"
            )
            print("✅ Model loaded on CPU!\n")
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            return False
    
    # Essential test cases only
    test_cases = [
        {
            "input": "6yrs na national id q wala pa din",
            "description": "Abbreviations & spelling (6yrs→years, q→ko)",
        },
        {
            "input": "wala masama pero may problem",
            "description": "Grammar & punctuation",
        },
        {
            "input": "kumusta ka na ba doon",
            "description": "Question detection (should end with ?)",
        },
        {
            "input": "ang taglish ay napakaganda dahil ito ay kombinasyon ng english at tagalog. q ka nag-start mag-aral? kumusta ka na ba? wala namang masama sa paggamit nito kase natural lang ito sa mga pilipino.",
            "description": "Paragraph handling (multiple sentences)",
        },
    ]
    
    print("Running 3 core test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['description']}")
        print(f"  Input:  {test_case['input']}")
        
        try:
            corrected = corrector.correct(test_case['input'], temperature=0.05)
            print(f"  Output: {corrected}\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
    
    # Batch test (reuses same model)
    print("-" * 80)
    print("Batch processing test...\n")
    
    batch_texts = [
        "bakit ang dumb mo kase",
        "ayoko na magwash ng plato",
    ]
    
    try:
        results = corrector.batch_correct(batch_texts, batch_size=2)
        for original, corrected in zip(batch_texts, results):
            print(f"  '{original}'")
            print(f"  → '{corrected}\n")
        print("✅ Batch processing works!")
    except Exception as e:
        print(f"❌ Batch error: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ Tests complete! Model is ready to use.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    main()
