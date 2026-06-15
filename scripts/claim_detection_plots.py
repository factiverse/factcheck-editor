"""Claim detection F1 plots — based on the original reference script,
updated for current model names and JSONL pred files.

Usage:
    cd /home/azureuser/repos/ml-models/paper/ICDM-26/factcheck-editor
    uv run python -m scripts.claim_detection_plots
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

data_folder = "data/claim_detection"
split = "test"


def load_claim_pred_data(lang):
    """Load pred file — tries JSONL first, falls back to JSON array."""
    for ext in ("jsonl", "json"):
        path = f"{data_folder}/{lang}_{split}_pred.{ext}"
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if ext == "jsonl" or first != "[":
                return [json.loads(l) for l in f if l.strip()]
            return json.load(f)
    return None


if __name__ == "__main__":
    ISO639_FILE = {}
    with open("src/utils/lang_codes.json") as iso639_file:
        ISO639_FILE = json.load(iso639_file)

    # Backup existing claim_detection_f1_scores.tsv before overwriting
    import shutil
    tsv_path = f"{data_folder}/claim_detection_f1_scores.tsv"
    if os.path.exists(tsv_path):
        backup_path = f"{tsv_path}.backup"
        shutil.copy(tsv_path, backup_path)
        print(f"Backed up existing {tsv_path} to {backup_path}")

    with open(tsv_path, "w") as f1_file:
        f1_file.write(
            "lang\tollama_macro\tollama_micro\t"
            "claude_macro\tclaude_micro\t"
            "gpt52_macro\tgpt52_micro\t"
            "facti_macro\tfacti_micro\t"
            "openrouter_macro\topenrouter_micro\n"
        )
        for lang in ISO639_FILE.keys():
            data = load_claim_pred_data(lang)
            if data is None:
                continue

            print(lang)
            try:
                # Build paired (gt, pred) per system — skip rows missing the col
                # so misaligned rows (e.g. failed API calls) don't corrupt F1.
                def _pairs(col):
                    return [(item["checkworthy"], item[col])
                            for item in data if col in item and "checkworthy" in item]

                def _f1(pairs):
                    if not pairs:
                        return float("nan"), float("nan")
                    gt, preds = zip(*pairs)
                    return (
                        f1_score(list(gt), list(preds), average="macro",  zero_division=0),
                        f1_score(list(gt), list(preds), average="micro",  zero_division=0),
                    )

                ollama_macro,     ollama_micro     = _f1(_pairs("ollama_pred"))
                claude_macro,     claude_micro     = _f1(_pairs("claude_opus_4_6_pred"))
                gpt52_macro,      gpt52_micro      = _f1(_pairs("gpt52_pred"))
                facti_macro,      facti_micro      = _f1(_pairs("facti_local_pred"))
                openrouter_macro, openrouter_micro = _f1(_pairs("openrouter_pred"))

                f1_file.write(
                    f"{lang}\t"
                    f"{ollama_macro}\t{ollama_micro}\t"
                    f"{claude_macro}\t{claude_micro}\t"
                    f"{gpt52_macro}\t{gpt52_micro}\t"
                    f"{facti_macro}\t{facti_micro}\t"
                    f"{openrouter_macro}\t{openrouter_micro}\n"
                )
            except Exception as e:
                print(f"Failed to process {lang}: {e}")
                continue

    # ── Plotting ─────────────────────────────────────────────────────────────
    data = pd.read_csv(f"{data_folder}/claim_detection_f1_scores.tsv", delimiter="\t")

    lang_names = {
        lang: (meta["name"] if isinstance(meta, dict) else meta)
        for lang, meta in ISO639_FILE.items()
    }
    data["lang"] = data["lang"].map(lang_names)

    bar_width = 0.2
    opacity   = 0.8

    # ── Micro F1 plot ─────────────────────────────────────────────────────────
    # Sort by XLM-RoBERTa (facti) scores, showing all languages even if facti is missing
    data_sorted = data.sort_values(by="facti_micro", na_position="last")
    fig, ax = plt.subplots(figsize=(20, 10))
    index = np.arange(len(data_sorted))

    ax.bar(index - 2.0*bar_width, data_sorted["ollama_micro"], bar_width, alpha=opacity, label="Qwen3-8B",       color="green")
    ax.bar(index - 1.0*bar_width, data_sorted["claude_micro"], bar_width, alpha=opacity, label="Claude Opus 4.6",color="yellow")
    ax.bar(index + 0.0*bar_width, data_sorted["gpt52_micro"],  bar_width, alpha=opacity, label="GPT-5.2",        color="red")
    ax.bar(index + 2.0*bar_width, data_sorted["openrouter_micro"], bar_width, alpha=opacity, label="Gemini 3.5 Flash", color="purple")
    ax.bar(index + 1.0*bar_width, data_sorted["facti_micro"],  bar_width, alpha=opacity, label="XLM-RoBERTa-Large (fine-tuned)",     color="blue")
    

    ax.set_xlabel("Language", fontsize=16)
    ax.set_ylabel("Micro F1 Score", fontsize=16)
    ax.set_title("Micro F1 Scores by Language for Claim Detection.", fontsize=20)
    ax.set_xticks(index)
    ax.set_xticklabels(data_sorted["lang"], rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{data_folder}_test_micro.pdf", format="pdf")
    print(f"Saved {data_folder}_test_micro.pdf")

    # ── Macro F1 plot ─────────────────────────────────────────────────────────
    # Sort by XLM-RoBERTa (facti) scores, showing all languages even if facti is missing
    data_sorted = data.sort_values(by="facti_macro", na_position="last")
    fig, ax = plt.subplots(figsize=(20, 10))
    index = np.arange(len(data_sorted))

    ax.bar(index - 2.0*bar_width, data_sorted["ollama_macro"], bar_width, alpha=opacity, label="Qwen3-8B",       color="green")
    ax.bar(index - 1.0*bar_width, data_sorted["claude_macro"], bar_width, alpha=opacity, label="Claude Opus 4.6",color="yellow")
    ax.bar(index + 0.0*bar_width, data_sorted["gpt52_macro"],  bar_width, alpha=opacity, label="GPT-5.2",        color="red")
    ax.bar(index + 1.0*bar_width, data_sorted["facti_macro"],  bar_width, alpha=opacity, label="XLM-RoBERTa-Large (fine-tuned)",     color="blue")
    ax.bar(index + 2.0*bar_width, data_sorted["openrouter_macro"], bar_width, alpha=opacity, label="Gemini 3.5 Flash", color="purple")

    ax.set_xlabel("Language", fontsize=16)
    ax.set_ylabel("Macro F1 Score", fontsize=16)
    ax.set_title("Macro F1 Scores by Language for Claim Detection.", fontsize=20)
    ax.set_xticks(index)
    ax.set_xticklabels(data_sorted["lang"], rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{data_folder}_test_macro.pdf", format="pdf")
    print(f"Saved {data_folder}_test_macro.pdf")

    # ── Averages ──────────────────────────────────────────────────────────────
    for col, label in [
        ("ollama_micro",  "Qwen3-8B     Micro-F1"),
        ("ollama_macro",  "Qwen3-8B     Macro-F1"),
        ("claude_micro",  "Claude Opus  Micro-F1"),
        ("claude_macro",  "Claude Opus  Macro-F1"),
        ("gpt52_micro",   "GPT-5.2      Micro-F1"),
        ("gpt52_macro",   "GPT-5.2      Macro-F1"),
        ("facti_micro",   "XLM-RoBERTa-Large (fine-tuned)   Micro-F1"),
        ("facti_macro",   "XLM-RoBERTa-Large (fine-tuned)   Macro-F1"),
        ("openrouter_micro", "Gemini 3.5 Flash Micro-F1"),
        ("openrouter_macro", "Gemini 3.5 Flash Macro-F1"),
    ]:
        if col in data.columns:
            print(f"{label}: {data[col].mean():.4f}")
