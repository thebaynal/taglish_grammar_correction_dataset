"""
Utilities for the Taglish grammar correction notebook.
Beginner-friendly helper functions for loading, cleaning, simple normalization,
metrics, saving outputs, and a lightweight inference helper.
"""

import os
import re
import json
from typing import Tuple, List

import pandas as pd
import numpy as np


def create_dirs(paths: List[str]):
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_jsonl(path: str) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame.

    Expects one JSON object per line with at least `noisy_text` and `clean_text` fields.
    """
    return pd.read_json(path, lines=True)


def clean_dataframe(df: pd.DataFrame, max_copies_per_pair: int = 5) -> Tuple[pd.DataFrame, int, int]:
    """Clean the dataset DataFrame.

    Steps:
    - Ensure `noisy_text` and `clean_text` columns exist.
    - Strip whitespace.
    - Drop rows with missing or empty `noisy_text` or `clean_text`.
    - Remove excessive exact duplicates for the same (noisy, clean) pair,
      but keep up to `max_copies_per_pair` copies so some unchanged examples remain.

    Returns:
    - cleaned DataFrame
    - number of rows removed overall
    - number of duplicate rows removed
    """
    df = df.copy()
    initial_len = len(df)

    # Ensure required columns
    for col in ["noisy_text", "clean_text"]:
        if col not in df.columns:
            df[col] = None

    # Convert to strings and strip whitespace
    df["noisy_text"] = df["noisy_text"].astype(str).str.strip()
    df["clean_text"] = df["clean_text"].astype(str).str.strip()

    # Drop rows with missing or empty noisy/clean text
    df_nonempty = df[(df["noisy_text"].notnull()) & (df["clean_text"].notnull()) &
                     (df["noisy_text"] != "") & (df["clean_text"] != "")].copy()
    after_nonempty = len(df_nonempty)

    # Reset index for stable behavior
    df_nonempty = df_nonempty.reset_index(drop=True)

    # Keep at most max_copies_per_pair for each identical (noisy, clean) pair
    # Assign a cumulative count per pair and filter
    pair_counts = df_nonempty.groupby(["noisy_text", "clean_text"]).cumcount()
    df_nonempty["_pair_idx"] = pair_counts
    df_limited = df_nonempty[df_nonempty["_pair_idx"] < max_copies_per_pair].copy()
    df_limited = df_limited.drop(columns=["_pair_idx"]).reset_index(drop=True)
    after_dedup = len(df_limited)

    removed_total = initial_len - after_dedup
    removed_duplicates = after_nonempty - after_dedup

    return df_limited, removed_total, removed_duplicates


def normalize_text_for_accuracy(s: str) -> str:
    """Conservative normalization used for approximate accuracy.

    Lowercases, removes most punctuation (keeps alphanumerics and spaces),
    collapses whitespace.
    """
    if s is None:
        return ""
    s = s.lower()
    # Remove punctuation except for apostrophes and hyphens which may be meaningful
    s = re.sub(r"[^\w\s'-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def postprocess_texts(texts: List[str]) -> List[str]:
    """Lightweight postprocessing for decoded model outputs and references."""
    return [t.strip() for t in texts]


def compute_approx_accuracy(preds: List[str], refs: List[str]) -> float:
    """Compute approximate exact-match accuracy after conservative normalization.

    Returns a float between 0 and 1.
    """
    if len(preds) == 0:
        return 0.0
    norm_preds = [normalize_text_for_accuracy(p) for p in preds]
    norm_refs = [normalize_text_for_accuracy(r) for r in refs]
    matches = sum(1 for p, r in zip(norm_preds, norm_refs) if p == r)
    return matches / len(preds)


def save_json(data, path: str):
    """Save JSON data to `path` (creates parent dirs if needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def correct_text(model, tokenizer, text: str, device: str = None, prefix: str = "fix grammar: ",
                 max_input_length: int = 128, max_target_length: int = 128, num_beams: int = 4) -> str:
    """Run a single inference pass with the given model and tokenizer.

    Returns the decoded predicted string.
    """
    try:
        import torch
    except Exception:
        torch = None

    if device is None:
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Move model to device
    if hasattr(model, "to"):
        model.to(device)

    # Tokenize and generate
    inputs = tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=max_input_length)
    if torch is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            pred_ids = model.generate(**inputs, max_length=max_target_length, num_beams=num_beams, early_stopping=True)
        decoded = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    else:
        # Best-effort fallback (won't actually run without torch)
        decoded = ""
    return decoded
