# Dataset Workflow (Taglish Grammar Correction)

Goal: build a **Taglish grammar correction** dataset with paired text:

- **noisy_text** = original / possibly-incorrect Taglish (model input)
- **clean_text** = corrected Taglish (model target)

This keeps annotation work minimal: you only need to fill in `clean_text`.

> Note: The file `dataset_final.jsonl` in this folder is actually a **JSON array** (starts with `[` and ends with `]`), not true JSONL.

---

## Files

### Raw (do not edit)
- `dataset_final.jsonl`  
  **Format:** JSON array of objects  
  This is the source of truth.

### Annotation (editable)
- `dataset_final.csv`  
  Generated from the raw JSON array. Edit in Excel / Google Sheets.

### Training-ready output (generated later)
- `dataset_annotated.jsonl`  
  True JSONL (one object per line) containing at least:
  - `id`
  - `noisy_text`
  - `clean_text`

Splits (train/val/test) will be created **automatically in code**, so annotators don’t need to deal with `split`.

---

## 1) Convert raw JSON array → CSV (for annotation)

From the project root (folder containing `FINAL_PROJECT/`):

1. Install dependency:
```bash
pip install pandas
```

2. Convert to CSV:
```bash
python FINAL_PROJECT/dataset/json_to_csv.py
```

Output:
- `FINAL_PROJECT/dataset/dataset_final.csv`

---

## 2) Annotate (minimal work)

Open `dataset_final.csv` and keep the sheet simple.

### Columns you should annotate
- `clean_text` (**required**)

### Columns you should NOT change
- `id`
- `noisy_text` (or `text`, depending on your CSV)

### Minimal annotation rules (Taglish GEC)
- **Do NOT translate** the sentence into pure Tagalog or pure English.
- Preserve the **Taglish code-switching style**.
- Make **minimal edits**:
  - capitalization and punctuation
  - obvious spelling fixes
  - clear English grammar errors inside English segments
- Preserve meaning and tone. Don’t add/remove facts.

### How to mark progress (optional)
If you want an easy progress tracker, add a `status` column in the sheet:
- `todo | doing | done`

But it’s optional.

---

## 3) Export annotated CSV

After annotation:
- Export/download the sheet as **CSV** (UTF-8)
- Overwrite `FINAL_PROJECT/dataset/dataset_final.csv` (or save as a new file)

---

## 4) Convert annotated CSV → training JSONL

Run:
```bash
python FINAL_PROJECT/dataset/csv_to_jsonl.py
```

Output:
- `FINAL_PROJECT/dataset/dataset_annotated.jsonl`

This file is what you feed into training/evaluation scripts.

---

## 5) Train/Val/Test split (no manual work)

We do **not** store `split` in the annotation CSV to reduce human work.

Instead, splitting is done automatically in code, for example:
- train 90%
- val 5%
- test 5%

(Implement in the training pipeline or a separate `split_dataset.py` script.)

---

## Troubleshooting

### JSONDecodeError during conversion
Make sure you are using `json_to_csv.py` (loads a JSON array) and not a line-by-line JSONL loader.

### Emojis look broken
Ensure your editor keeps UTF-8 encoding. Google Sheets is usually fine.