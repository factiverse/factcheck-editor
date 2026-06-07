import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.utils.utils import load_lang_codes


DATA_DIR = Path("data/veracity_prediction")
SPLIT = "test"

LABEL_TO_ID = {
    "REFUTES": 0,
    "SUPPORTS": 1,
    "MIXED": 2,
    "NOT_ENOUGH_INFO": 3,
}

MODEL_SPECS = [
    ("mmbert", "facti_local_int_label", "facti_local_label", "MMbert-base fine-tuned (Factiverse)", "blue"),
    ("claude", "claude-opus-4-6_int_label", "claude-opus-4-6_label", "Claude Opus 4.6", "yellow"),
    ("gpt5", "gpt5_int_label", "gpt5_label", "GPT-5.2", "red"),
    ("qwen3", "ollama/qwen3:8b_int_label", "ollama/qwen3:8b_label", "Qwen3-8B", "green"),
]

FACTI_LOCAL_MODEL_KEY = "mmbert"
EXCLUDE_BOTTOM_FACTI_LOCAL_COUNT = 10


def label_to_id(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        stripped = value.strip().upper()
        if stripped.isdigit():
            return int(stripped)
        return LABEL_TO_ID.get(stripped)
    return None


def safe_f1(y_true, y_pred, average):
    if not y_true or not y_pred:
        return float("nan")
    try:
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    except Exception:
        return float("nan")


def resolve_prediction(item, int_key, label_key):
    if int_key in item and item[int_key] is not None:
        return label_to_id(item[int_key])
    if label_key in item:
        return label_to_id(item[label_key])
    return None


def load_prediction_files(data_dir, split):
    files = sorted(data_dir.glob(f"*_{split}_local_pred.json"))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {data_dir} for split '{split}'")
    data = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, list):
                data.extend(payload)
    return files, data


if __name__ == "__main__":
    lang_codes = load_lang_codes()
    files, data = load_prediction_files(DATA_DIR, SPLIT)

    print(f"Loaded {len(files)} prediction files")
    print(f"Total items in file set: {len(data)}")

    lang_counts = defaultdict(int)
    for item in data:
        lang_counts[item.get("lang")] += 1
    print("\nLanguage distribution in data:")
    for lang in sorted(lang_counts):
        print(f"  {lang}: {lang_counts[lang]}")

    lang_scores = defaultdict(lambda: defaultdict(list))
    lang_totals = defaultdict(int)

    for item in data:
        lang = item.get("lang")
        true_label = label_to_id(item.get("labels"))
        if true_label is None:
            true_label = label_to_id(item.get("gt_label"))
        if true_label is None:
            true_label = label_to_id(item.get("label"))
        if true_label is None:
            continue

        lang_totals[lang] += 1
        for model_key, int_key, label_key, _, _ in MODEL_SPECS:
            pred = resolve_prediction(item, int_key, label_key)
            if pred is None:
                continue
            lang_scores[lang][model_key].append((true_label, pred))

    results = []
    skipped_langs = defaultdict(int)
    for lang in sorted(lang_totals):
        if lang_totals[lang] < 2:
            skipped_langs[lang] = lang_totals[lang]
            continue

        row = {"Lang": lang, "Count": lang_totals[lang]}
        for model_key, _, _, _, _ in MODEL_SPECS:
            pairs = lang_scores[lang].get(model_key, [])
            y_true = [truth for truth, _ in pairs]
            y_pred = [pred for _, pred in pairs]
            row[f"{model_key}_Macro_F1"] = safe_f1(y_true, y_pred, "macro")
            row[f"{model_key}_Micro_F1"] = safe_f1(y_true, y_pred, "micro")
        results.append(row)

    df = pd.DataFrame(results)
    if df.empty:
        print("No valid results found")
        if skipped_langs:
            print("\nLanguages that were skipped (< 2 samples):")
            for lang, count in sorted(skipped_langs.items()):
                print(f"  {lang}: {count} samples")
        raise SystemExit(1)

    print(f"\nLanguages found: {len(df)}")
    print(df)

    lang_names = {lang: lang_codes.get(lang, {}).get("name", lang) for lang in df["Lang"]}
    data_for_plot = df.copy()

    # Exclude bottom-performing facti-local languages from plots only.
    facti_macro_col = f"{FACTI_LOCAL_MODEL_KEY}_Macro_F1"
    facti_micro_col = f"{FACTI_LOCAL_MODEL_KEY}_Micro_F1"
    excluded_langs = set()
    if facti_macro_col in data_for_plot.columns:
        macro_bottom = (
            data_for_plot[["Lang", facti_macro_col]]
            .dropna(subset=[facti_macro_col])
            .nsmallest(EXCLUDE_BOTTOM_FACTI_LOCAL_COUNT, facti_macro_col)["Lang"]
            .tolist()
        )
        excluded_langs.update(macro_bottom)
    if facti_micro_col in data_for_plot.columns:
        micro_bottom = (
            data_for_plot[["Lang", facti_micro_col]]
            .dropna(subset=[facti_micro_col])
            .nsmallest(EXCLUDE_BOTTOM_FACTI_LOCAL_COUNT, facti_micro_col)["Lang"]
            .tolist()
        )
        excluded_langs.update(micro_bottom)

    if excluded_langs:
        excluded_sorted = sorted(excluded_langs)
        print(
            f"\nExcluding {len(excluded_sorted)} bottom facti-local languages from plots: "
            f"{', '.join(excluded_sorted)}"
        )
        data_for_plot = data_for_plot[~data_for_plot["Lang"].isin(excluded_langs)].copy()

    if data_for_plot.empty:
        print("No languages left to plot after exclusions")
        raise SystemExit(1)

    included_langs_for_overall = set(data_for_plot["Lang"])

    data_for_plot["Lang"] = data_for_plot["Lang"].map(lang_names)

    plot_models = [spec for spec in MODEL_SPECS if f"{spec[0]}_Macro_F1" in df.columns]
    bar_width = 0.16
    opacity = 0.8
    offsets = np.linspace(-2 * bar_width, 2 * bar_width, len(plot_models))

    macro_sort_col = "facti_local_Macro_F1" if "facti_local_Macro_F1" in data_for_plot.columns else f"{plot_models[0][0]}_Macro_F1"
    data_sorted = data_for_plot.sort_values(by=macro_sort_col, na_position="last")
    fig, ax = plt.subplots(figsize=(16, 8))
    index = np.arange(len(data_sorted))
    for offset, (model_key, _, _, display_name, color) in zip(offsets, plot_models):
        ax.bar(index + offset, data_sorted[f"{model_key}_Macro_F1"], bar_width, alpha=opacity, label=display_name, color=color)

    ax.set_xlabel("Language", fontsize=14)
    ax.set_ylabel("Macro F1 Score", fontsize=14)
    ax.set_title("Macro F1 Scores by Language for Veracity Prediction", fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(data_sorted["Lang"], rotation=45, ha="right", fontsize=11)
    ax.legend(fontsize=12, loc="upper left")
    ax.set_ylim(0, 1)
    plt.tight_layout()

    output_plot_macro = DATA_DIR / f"veracity_{SPLIT}_macro_f1.pdf"
    plt.savefig(output_plot_macro, format="pdf")
    print(f"\nMacro F1 plot saved to {output_plot_macro}")
    plt.close()

    micro_sort_col = "facti_local_Micro_F1" if "facti_local_Micro_F1" in data_for_plot.columns else f"{plot_models[0][0]}_Micro_F1"
    data_sorted_micro = data_for_plot.sort_values(by=micro_sort_col, na_position="last")
    fig, ax = plt.subplots(figsize=(16, 8))
    index = np.arange(len(data_sorted_micro))
    for offset, (model_key, _, _, display_name, color) in zip(offsets, plot_models):
        ax.bar(index + offset, data_sorted_micro[f"{model_key}_Micro_F1"], bar_width, alpha=opacity, label=display_name, color=color)

    ax.set_xlabel("Language", fontsize=14)
    ax.set_ylabel("Micro F1 Score", fontsize=14)
    ax.set_title("Micro F1 Scores by Language for Veracity Prediction", fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(data_sorted_micro["Lang"], rotation=45, ha="right", fontsize=11)
    ax.legend(fontsize=12, loc="upper left")
    ax.set_ylim(0, 1)
    plt.tight_layout()

    output_plot_micro = DATA_DIR / f"veracity_{SPLIT}_micro_f1.pdf"
    plt.savefig(output_plot_micro, format="pdf")
    print(f"Micro F1 plot saved to {output_plot_micro}")
    plt.close()

    print("\n=== Average F1 Scores ===")
    for model_key, _, _, display_name, _ in plot_models:
        print(
            f"{display_name} - Macro: {df[f'{model_key}_Macro_F1'].mean(skipna=True):.4f}, "
            f"Micro: {df[f'{model_key}_Micro_F1'].mean(skipna=True):.4f}"
        )

    print("\n=== Instance Counts per Language ===")
    print(f"Total instances: {df['Count'].sum()}")

    print("\n=== Overall F1 Scores by Class ===")
    print("Overall metrics are computed after excluding bottom facti-local languages.")
    class_names = ["REFUTES", "SUPPORTS", "MIXED"]
    for model_key, _, _, display_name, _ in plot_models:
        all_pairs = []
        for lang, scores in lang_scores.items():
            if lang not in included_langs_for_overall:
                continue
            all_pairs.extend(scores.get(model_key, []))
        y_true = [truth for truth, _ in all_pairs]
        y_pred = [pred for _, pred in all_pairs]
        if not y_true:
            print(f"{display_name}: No valid predictions.")
            continue
        macro_by_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        overall_macro = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
        overall_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        print(
            f"{display_name} - Macro F1 by class: "
            f"{dict(zip(class_names, macro_by_class.round(4)))}"
        )
        print(
            f"{display_name} - Overall Macro F1: {overall_macro:.4f} | "
            f"Overall Micro F1: {overall_micro:.4f}"
        )

    df_output = df.copy()
    df_output["Lang_Name"] = df_output["Lang"].map(lang_names)
    output_tsv = DATA_DIR / f"veracity_{SPLIT}_f1_scores.tsv"
    columns = ["Lang", "Lang_Name", "Count"]
    for model_key, _, _, _, _ in plot_models:
        columns.extend([f"{model_key}_Macro_F1", f"{model_key}_Micro_F1"])
    df_output.to_csv(output_tsv, sep="\t", index=False, columns=columns)
    print(f"\nResults saved to {output_tsv}")
