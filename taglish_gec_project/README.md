# Taglish Grammar Correction Model

This project trains a Taglish grammar-correction model that rewrites noisy Filipino-English code-switched text into a cleaner, more natural form while preserving meaning, tone, slang, and code-mixing.

Everything needed for inference is included in the repository, so the project is runnable after installing dependencies.

The final model was trained in [taglish_grammar_correction.ipynb](taglish_grammar_correction.ipynb) and saved to [taglish_gec_model](taglish_gec_model).

## What the Model Does

The model corrects common Taglish errors such as:

- grammar marker mistakes like `ng` vs `nang`
- `din` vs `rin`, `daw` vs `raw`
- punctuation and capitalization
- hyphenation in Taglish verb forms like `nag-drive` and `mag-drawing`
- spelling issues while keeping conversational phrasing intact

It is designed to correct text without over-translating or making it sound too formal.

## Training Data

The model was built from the Hugging Face dataset [mggy/taglish-socialmedia-dataset](https://huggingface.co/datasets/mggy/taglish-socialmedia-dataset), which was filtered and modified into labeled noisy/clean Taglish pairs.

The working copy used in the notebook lives in [dataset/dataset_final (1).csv](dataset/dataset_final%20(1).csv).

Key data facts from the notebook:

- Total rows: 7,707
- Labeled Taglish training pairs after filtering: 4,247
- Split: 80% train / 10% validation / 10% test
- A small portion of unchanged pairs was kept so the model also learns when not to over-correct

The notebook normalizes the text, removes duplicates, and keeps only valid noisy/clean pairs before training.

## How the Model Was Trained

The training pipeline uses a decoder-only instruction-tuning setup with LoRA fine-tuning on `mistralai/Mistral-7B-Instruct-v0.2`.

Important training choices:

- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- Fine-tuning method: LoRA via PEFT
- LoRA target modules: `q_proj` and `v_proj`
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Epochs: 4
- Learning rate: `8e-5`
- Effective batch size: 16
- Mixed precision: FP16 on GPU
- Validation: enabled during training with best-model loading by `eval_loss`

The key training improvement is that the loss is computed only on the corrected output tokens, not on the prompt itself. That makes the model learn the correction task more directly instead of memorizing instructions.

## Results

Final notebook outputs reported:

- Final training loss: `0.5245`
- Average ROUGE-1 F-score: `0.9870`
- Average ROUGE-L F-score: `0.9864`
- Exact match rate: `70.00%`

These scores are strong for a generation-based correction task, especially because multiple valid corrections can exist for the same Taglish sentence.

## Outputs and Artifacts

The notebook creates the main artifact used for inference:

- [taglish_gec_model](taglish_gec_model) - fine-tuned model directory

## How to Use

For inference, the simplest entry point is [standalone_inference.py](standalone_inference.py).

Example:

```python
from standalone_inference import TaglishGrammarCorrector

corrector = TaglishGrammarCorrector(
    model_path="/path/to/taglish_gec_project/taglish_gec_model",
    device="cuda"
)

text = "wala naman masama pero may problem"
print(corrector.correct(text))
```

For batch correction, use `batch_correct()` on a list of sentences.

## Notebook Workflow

The notebook follows this process:

1. Load and inspect the CSV dataset.
2. Normalize and filter the Taglish correction pairs.
3. Split into train, validation, and test sets.
4. Build instruction-style prompts for correction.
5. Load the base Mistral model and apply LoRA.
6. Tokenize with prompt masking so only the target output contributes to loss.
7. Train with validation and checkpoint selection.
8. Evaluate on held-out test data using ROUGE and exact match.
9. Save the final model and evaluation artifacts.

## Limitations

The model is best at short to medium Taglish sentences with grammar and punctuation errors. It may still struggle with:

- highly ambiguous context
- very long posts with multiple style shifts
- cases where several corrections are equally valid
- meaning-preserving rewrites that require world knowledge

## Reproduce

Open [taglish_grammar_correction.ipynb](taglish_grammar_correction.ipynb) and run the cells in order. The notebook is already configured to use the dataset in [dataset/dataset_final (1).csv](dataset/dataset_final%20(1).csv) and save outputs under the project directory.
