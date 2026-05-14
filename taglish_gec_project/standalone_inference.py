"""
Standalone Taglish Grammar Correction Inference Script
Use this script outside the notebook to correct Taglish text
"""

from pathlib import Path
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
from typing import List
import argparse


class TaglishGrammarCorrector:
    """Production-ready Taglish grammar correction inference."""
    
    def __init__(self, model_path, device="cuda"):
        """
        Initialize the corrector with a trained LoRA model.
        
        Args:
            model_path: Path to the trained taglish_gec_model directory
            device: "cuda" or "cpu"
        """
        self.model_path = Path(model_path)
        self.device = device
        
        print(f"Loading model from {self.model_path}...")
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Question words for post-processing (pre-compiled tuple for speed)
        self.question_words = ('kumusta', 'bakit', 'saan', 'ano', 'sino', 'may', 'kaya', 'pwede', 'nag')
        
        print(f"✓ Model loaded on {device}")
    
    def correct(self, noisy_text: str, max_length: int = 100, temperature: float = 0.05) -> str:
        """
        Correct a single Taglish sentence.
        
        Args:
            noisy_text: Input text with grammar errors
            max_length: Maximum output length
            temperature: Creativity (0.05=strict, 0.7=creative)
        
        Returns:
            Corrected text (Taglish preserved, no translation)
        """
        # Minimal normalization: just strip whitespace
        noisy_text = noisy_text.strip()
        
        # Use training format: [INST] Correct this Taglish sentence:\n{text} [/INST]
        prompt = f"[INST] Correct this Taglish sentence:\n{noisy_text} [/INST]"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
            )
        
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output (keep only generated text)
        if "[/INST]" in corrected:
            corrected = corrected.split("[/INST]")[1].strip()
        
        # Keep only first line
        if "\n" in corrected:
            corrected = corrected.split("\n")[0].strip()
        
        # Quick question mark fix (only if necessary)
        if corrected.endswith('.') and corrected[0].lower() in ('k', 'b', 's', 'a', 'p', 'm', 'n'):
            corrected_lower = corrected.lower()
            if any(corrected_lower.startswith(qw) for qw in self.question_words):
                corrected = corrected[:-1] + '?'
        
        return corrected
    
    def batch_correct(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Correct multiple sentences efficiently."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                results.append(self.correct(text))
        return results


def main():
    parser = argparse.ArgumentParser(description="Taglish Grammar Correction")
    parser.add_argument("--model-path", type=str, 
                       default=str(Path(__file__).resolve().parent / "taglish_gec_model"),
                       help="Path to trained model")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--text", type=str, help="Single text to correct")
    parser.add_argument("--input-file", type=str, help="Input file with texts (one per line)")
    parser.add_argument("--output-file", type=str, help="Output file for results")
    parser.add_argument("--temperature", type=float, default=0.05,
                       help="Temperature (0.05=strict, 0.7=creative)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize corrector
    corrector = TaglishGrammarCorrector(args.model_path, device=args.device)
    
    # Single text correction
    if args.text:
        result = corrector.correct(args.text, temperature=args.temperature)
        print(f"Input:  {args.text}")
        print(f"Output: {result}")
    
    # File-based correction
    elif args.input_file:
        print(f"Reading from {args.input_file}...")
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = corrector.batch_correct(texts, batch_size=args.batch_size)
        
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                for original, corrected in zip(texts, results):
                    f.write(f"{original}\t{corrected}\n")
            print(f"Results saved to {args.output_file}")
        else:
            for original, corrected in zip(texts, results):
                print(f"Input:  {original}")
                print(f"Output: {corrected}\n")
    
    # Interactive mode
    elif args.interactive:
        print("\n=== Taglish Grammar Corrector (Interactive) ===")
        print("Type 'exit' or 'quit' to stop\n")
        
        while True:
            text = input("Enter Taglish text: ").strip()
            if text.lower() in ("exit", "quit"):
                break
            
            corrected = corrector.correct(text, temperature=args.temperature)
            print(f"Corrected: {corrected}\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
