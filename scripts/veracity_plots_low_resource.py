import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
from src.utils.utils import load_lang_codes

if __name__ == "__main__":
    lang_codes = load_lang_codes()
    
    # Input file - most recent results file
    input_file = "data/veracity_prediction/factisearch_train_20251230_091629_test_20260112_151237.jsonl"
    # input_file = "data/veracity_prediction/train_factisearch_20251225_results_debug_paraphrase-multilingual_20251230_091629.jsonl"
    
    
    # Read JSONL file
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Total items in file: {len(data)}")
    
    # Debug: Check language distribution
    lang_counts = defaultdict(int)
    for item in data:
        lang = item.get("lang")
        lang_counts[lang] += 1
    print(f"\nLanguage distribution in data:")
    for lang in sorted(lang_counts.keys()):
        print(f"  {lang}: {lang_counts[lang]}")
    
    if len(data) == 0:
        print("No data found")
        exit(1)
    
    # Map label strings to integers
    label_map = {
        "SUPPORTS": 1,
        "REFUTES": 0,
        "MIXED": 2,
        "NOT_ENOUGH_INFO": 3,
    }
    
    # Group by language and collect predictions
    lang_scores = defaultdict(lambda: {
        "factiverse": [],
        "ollama": [],
        "gpt3": [],
        "gpt4": [],
        "gpt5": [],
        "labels": []
    })
    
    for item in data:
        lang = item.get("lang")
        
        # Get ground truth label
        true_label = item.get("label", "")
        true_label_int = label_map.get(true_label, -1)
        
        if true_label_int == -1:
            continue
        
        # Get predictions from different models
        factiverse_label = item.get("factiverse_response", {}).get("finalLabelDescription", "")
        ollama_label = item.get("ollama_label", "")
        gpt3_label = item.get("gpt3_label", "")
        gpt4_label = item.get("gpt4o_label", "") or item.get("gpt4_label", "")
        gpt5_label = item.get("gpt5_label", "")
        
        factiverse_pred = label_map.get(factiverse_label, -1)
        ollama_pred = label_map.get(ollama_label, -1)
        gpt3_pred = label_map.get(gpt3_label, -1)
        gpt4_pred = label_map.get(gpt4_label, -1)
        gpt5_pred = label_map.get(gpt5_label, -1)
        
        # Only add if we have valid predictions
        if factiverse_pred >= 0 or ollama_pred >= 0 or gpt3_pred >= 0 or gpt4_pred >= 0 or gpt5_pred >= 0:
            lang_scores[lang]["factiverse"].append(factiverse_pred)
            lang_scores[lang]["ollama"].append(ollama_pred)
            lang_scores[lang]["gpt3"].append(gpt3_pred)
            lang_scores[lang]["gpt4"].append(gpt4_pred)
            lang_scores[lang]["gpt5"].append(gpt5_pred)
            lang_scores[lang]["labels"].append(true_label_int)
    
    # Calculate F1 scores for each language
    results = []
    skipped_langs = defaultdict(int)
    for lang, scores in sorted(lang_scores.items()):
        if len(scores["labels"]) < 2:  # Need at least 2 samples
            skipped_langs[lang] = len(scores["labels"])
            continue
        
        labels = scores["labels"]
        
        # Calculate F1 scores with proper error handling
        def safe_f1(y_true, y_pred, avg_type):
            try:
                if all(p >= 0 for p in y_pred):
                    return f1_score(y_true, y_pred, average=avg_type, zero_division=0)
                else:
                    return 0.0
            except:
                return 0.0
        
        results.append({
            "Lang": lang,
            "Count": len(scores["labels"]),
            "FV_Macro_F1": safe_f1(labels, scores["factiverse"], 'macro'),
            "FV_Micro_F1": safe_f1(labels, scores["factiverse"], 'micro'),
            "Ollama_Macro_F1": safe_f1(labels, scores["ollama"], 'macro'),
            "Ollama_Micro_F1": safe_f1(labels, scores["ollama"], 'micro'),
            "gpt3_Macro_F1": safe_f1(labels, scores["gpt3"], 'macro'),
            "gpt3_Micro_F1": safe_f1(labels, scores["gpt3"], 'micro'),
            "gpt4_Macro_F1": safe_f1(labels, scores["gpt4"], 'macro'),
            "gpt4_Micro_F1": safe_f1(labels, scores["gpt4"], 'micro'),
            "gpt5_Macro_F1": safe_f1(labels, scores["gpt5"], 'macro'),
            "gpt5_Micro_F1": safe_f1(labels, scores["gpt5"], 'micro'),
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No valid results found")
        print(f"\nLanguages that were skipped (< 2 samples):")
        for lang, count in sorted(skipped_langs.items()):
            print(f"  {lang}: {count} samples")
        exit(1)
    
    print(f"\nLanguages found: {len(df)}")
    print(df)
    
    # Map language codes to names
    lang_names = {lang: lang_codes.get(lang, {}).get("name", lang) for lang in df["Lang"]}
    
    # Create a copy for plotting with language names
    data_for_plot = df.copy()
    data_for_plot["Lang"] = data_for_plot["Lang"].map(lang_names)
    
    # ===== MACRO F1 PLOT =====
    data_sorted = data_for_plot.sort_values(by="FV_Macro_F1")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bar_width = 0.16
    opacity = 0.8
    index = np.arange(len(data_sorted))
    
    
    ax.bar(index - 2*bar_width, data_sorted["Ollama_Macro_F1"], bar_width, alpha=opacity, label="Qwen3-8b", color="green")
    # ax.bar(index - bar_width, data_sorted["gpt3_Macro_F1"], bar_width, alpha=opacity, label="GPT-3.5-turbo", color="pink")
    ax.bar(index, data_sorted["gpt4_Macro_F1"], bar_width, alpha=opacity, label="Claude-Opus-4.6", color="yellow")
    ax.bar(index + bar_width, data_sorted["gpt5_Macro_F1"], bar_width, alpha=opacity, label="GPT-5.2", color="red")
    ax.bar(index + 2*bar_width, data_sorted["FV_Macro_F1"], bar_width, alpha=opacity, label="Fine-tuned XLM-Roberta-Large", color="blue")
    
    # Formatting
    ax.set_xlabel("Language", fontsize=14)
    ax.set_ylabel("Macro F1 Score", fontsize=14)
    ax.set_title("Macro F1 Scores by Language for Veracity Prediction", fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(data_sorted["Lang"], rotation=45, ha="right", fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    output_plot_macro = input_file.replace(".jsonl", "_macro_f11.pdf")
    plt.savefig(output_plot_macro, format="pdf")
    print(f"\nMacro F1 plot saved to {output_plot_macro}")
    plt.close()
    
    # ===== MICRO F1 PLOT =====
    data_sorted_micro = data_for_plot.sort_values(by="FV_Micro_F1")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    index = np.arange(len(data_sorted_micro))
    
    # Plot bars for each model
    ax.bar(index - 2*bar_width, data_sorted_micro["Ollama_Micro_F1"], bar_width, alpha=opacity, label="Qwen3-8b", color="green")
    # ax.bar(index - bar_width, data_sorted_micro["gpt3_Micro_F1"], bar_width, alpha=opacity, label="GPT-3.5-turbo", color="pink")
    ax.bar(index, data_sorted_micro["gpt4_Micro_F1"], bar_width, alpha=opacity, label="Claude-Opus-4.6", color="yellow")
    ax.bar(index + bar_width, data_sorted_micro["gpt5_Micro_F1"], bar_width, alpha=opacity, label="GPT-5.2", color="red")
    ax.bar(index + 2*bar_width, data_sorted_micro["FV_Micro_F1"], bar_width, alpha=opacity, label="Factiverse (Fine-Tuned XLM-Roberta-Large)", color="blue")
    
    # Formatting
    ax.set_xlabel("Language", fontsize=14)
    ax.set_ylabel("Macro F1 Score", fontsize=14)
    ax.set_title("Macro F1 Scores by Language for Veracity Prediction", fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(data_sorted_micro["Lang"], rotation=45, ha="right", fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    output_plot_micro = input_file.replace(".jsonl", "_micro_f1.pdf")
    plt.savefig(output_plot_micro, format="pdf")
    print(f"Micro F1 plot saved to {output_plot_micro}")
    plt.close()
    
    # ===== SUMMARY STATISTICS =====
    print("\n=== Average F1 Scores ===")
    print(f"Ollama - Macro: {df['Ollama_Macro_F1'].mean():.4f}, Micro: {df['Ollama_Micro_F1'].mean():.4f}")
    # print(f"GPT-3.5 - Macro: {df['gpt3_Macro_F1'].mean():.4f}, Micro: {df['gpt3_Micro_F1'].mean():.4f}")
    print(f"GPT-4 - Macro: {df['gpt4_Macro_F1'].mean():.4f}, Micro: {df['gpt4_Micro_F1'].mean():.4f}")
    print(f"GPT-5.2 - Macro: {df['gpt5_Macro_F1'].mean():.4f}, Micro: {df['gpt5_Micro_F1'].mean():.4f}")
    print(f"Factiverse - Macro: {df['FV_Macro_F1'].mean():.4f}, Micro: {df['FV_Micro_F1'].mean():.4f}")
    
    # Print per-language instance counts
    print(f"\n=== Instance Counts per Language ===")
    print(f"Total instances: {df['Count'].sum()}")
    # Print overall performance by class for each model
    print("\n=== Overall F1 Scores by Class ===")
    from sklearn.metrics import f1_score
    import numpy as np

    # Aggregate all labels and predictions across all languages
    all_labels = []
    all_factiverse = []
    all_ollama = []
    all_gpt3 = []
    all_gpt4 = []
    all_gpt5 = []
    for scores in lang_scores.values():
        all_labels.extend(scores["labels"])
        all_factiverse.extend(scores["factiverse"])
        all_ollama.extend(scores["ollama"])
        all_gpt3.extend(scores["gpt3"])
        all_gpt4.extend(scores["gpt4"])
        all_gpt5.extend(scores["gpt5"])

    model_preds = {
        "Factiverse": all_factiverse,
        "Ollama": all_ollama,
        "GPT-3.5": all_gpt3,
        "GPT-4": all_gpt4,
        "GPT-5.2": all_gpt5,
    }
    class_names = ["SUPPORTS", "REFUTES", "MIXED", "NOT_ENOUGH_INFO"]
    print(f"Classes: {class_names}")
    for model, preds in model_preds.items():
        # Only consider valid predictions (>=0)
        valid_idx = [i for i, p in enumerate(preds) if p >= 0 and all_labels[i] >= 0]
        y_true = [all_labels[i] for i in valid_idx]
        y_pred = [preds[i] for i in valid_idx]
        if not y_true:
            print(f"{model}: No valid predictions.")
            continue
        macro = f1_score(y_true, y_pred, average=None, zero_division=0)
        micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        print(f"{model} - Macro F1 by class: {dict(zip(class_names, macro.round(4)))} | Micro F1: {micro:.4f}")
    # Save macro and micro F1 results to separate CSV files
    macro_cols = ["Lang", "Count", "FV_Macro_F1", "Ollama_Macro_F1", "gpt3_Macro_F1", "gpt4_Macro_F1", "gpt5_Macro_F1"]
    micro_cols = ["Lang", "Count", "FV_Micro_F1", "Ollama_Micro_F1", "gpt3_Micro_F1", "gpt4_Micro_F1", "gpt5_Micro_F1"]
    df[macro_cols].to_csv("veracity_macro_f1_results.csv", index=False)
    df[micro_cols].to_csv("veracity_micro_f1_results.csv", index=False)
    print("\nMacro F1 results saved to veracity_macro_f1_results.csv")
    print("Micro F1 results saved to veracity_micro_f1_results.csv")
    print("\nPer-language breakdown:")
    df_display = df.copy()
    df_display["Lang_Name"] = df_display["Lang"].map(lang_names)
    print(df_display[["Lang", "Lang_Name", "Count"]].to_string(index=False))
    
    # Save results to TSV for future use
    df_output = df.copy()
    df_output["Lang_Name"] = df_output["Lang"].map(lang_names)
    output_tsv = input_file.replace(".jsonl", "_f1_scores.tsv")
    df_output.to_csv(output_tsv, sep='\t', index=False, columns=["Lang", "Lang_Name", "Count", "FV_Macro_F1", "FV_Micro_F1", "Ollama_Macro_F1", "Ollama_Micro_F1", "gpt3_Macro_F1", "gpt3_Micro_F1", "gpt4_Macro_F1", "gpt4_Micro_F1", "gpt5_Macro_F1", "gpt5_Micro_F1"])
    print(f"\nResults saved to {output_tsv}")
