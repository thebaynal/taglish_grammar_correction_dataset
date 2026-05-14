# Taglish Grammar Correction

Taglish Grammar Correction is a fully runnable Filipino-English code-switching project for correcting grammar, spelling, punctuation, and phrasing while preserving the original tone and meaning.

This repository contains the cleaned project folder, trained model artifacts, inference scripts, and the notebook used to build and evaluate the model.

## What’s In This Repo

- A fine-tuned Taglish grammar correction model under [taglish_gec_project/taglish_gec_model](taglish_gec_project/taglish_gec_model)
- A reusable inference helper in [taglish_gec_project/standalone_inference.py](taglish_gec_project/standalone_inference.py)
- An interactive CLI version in [taglish_gec_project/interactive_corrector.py](taglish_gec_project/interactive_corrector.py)
- A quick test script in [taglish_gec_project/test_model.py](taglish_gec_project/test_model.py)
- The training and evaluation notebook in [taglish_gec_project/taglish_grammar_correction.ipynb](taglish_gec_project/taglish_grammar_correction.ipynb)
- A project-specific README with deeper implementation details in [taglish_gec_project/README.md](taglish_gec_project/README.md)

## Quick Start

1. Install dependencies:

```bash
pip install -r taglish_gec_project/requirements.txt
```

2. Run a one-off correction:

```bash
python taglish_gec_project/standalone_inference.py --text "wala masama pero may problem"
```

3. Use interactive mode:

```bash
python taglish_gec_project/interactive_corrector.py
```

4. Open the notebook:

```bash
jupyter notebook taglish_gec_project/taglish_grammar_correction.ipynb
```

## Project Summary

The model is trained with a LoRA fine-tuning setup on a decoder-only instruction-tuning pipeline. It is designed to make minimal, meaning-preserving corrections rather than over-translating Taglish into pure Tagalog or English.

The training data comes from the Hugging Face dataset [mggy/taglish-socialmedia-dataset](https://huggingface.co/datasets/mggy/taglish-socialmedia-dataset), which was filtered and modified into labeled noisy/clean Taglish pairs.

The default scripts resolve the model path relative to the project folder, and the trained model is included in the repository, so it can be cloned and run without editing hardcoded paths.

## Repository Layout

```text
taglish_gec_project/
  dataset/
  model_checkpoints/
  taglish_gec_model/
  standalone_inference.py
  interactive_corrector.py
  test_model.py
  taglish_grammar_correction.ipynb
  requirements.txt
  README.md
```

## Model Notes

The project focuses on common Taglish correction cases such as:

- grammar markers like `ng` and `nang`
- `din` and `rin`, `daw` and `raw`
- capitalization and punctuation
- Taglish verb forms such as `nag-drive` or `mag-drawing`
- spelling fixes that preserve code-switching style

For the full training details, example outputs, and implementation notes, see [taglish_gec_project/README.md](taglish_gec_project/README.md).

## Before Publishing

Review the size of the model and checkpoint files before pushing the repository publicly. If you want a lighter public repo, move large artifacts to Git LFS or keep only the final model directory.