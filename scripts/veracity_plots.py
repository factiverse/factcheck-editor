import requests
import json
import os
from sklearn.metrics import f1_score

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils.utils import load_lang_codes

if __name__ == "__main__":
    lang_codes = load_lang_codes()

    # Load data from a JSON file into a pandas DataFrame
    split = "test"
    data = pd.read_csv(
        f"data/veracity_prediction/{split}.jsonl_f1.tsv", delimiter="\t"
    )
    # Assuming the JSON structure directly maps to the DataFrame structure
    # No need for transformation if the JSON structure matches the DataFrame exactly
    lang_names = {
        lang: lang_codes[lang]["name"] for lang in lang_codes.keys()
    }
    print(lang_names)
    # Sort the DataFrame by gpt3_Micro_F1
    data["Lang"] = data["Lang"].map(lang_names)
    print(data)
    data_sorted = data.sort_values(by="FV_Micro_F1")

    fig, ax = plt.subplots(figsize=(20, 10))
    bar_width = 0.2
    opacity = 0.8
    index = np.arange(len(data_sorted))

    # OpenAI
    Ollama_Micro_F1_bars = ax.bar(
        index - 2 * bar_width,
        data_sorted["Ollama_Micro_F1"],
        bar_width,
        alpha=opacity,
        label="Mistral-7b",
        color="green",
    )
    gpt3_Micro_F1_bars = ax.bar(
        index - bar_width,
        data_sorted["gpt3_Micro_F1"],
        bar_width,
        alpha=opacity,
        label="GPT-3.5-turbo",
        color="pink",
    )
    gpt3_Micro_F1_bars = ax.bar(
        index,
        data_sorted["gpt4_Micro_F1"],
        bar_width,
        alpha=opacity,
        label="GPT-4",
        color="red",
    )
    FV_Micro_F1_bars = ax.bar(
        index + bar_width,
        data_sorted["FV_Micro_F1"],
        bar_width,
        alpha=opacity,
        label="Factiverse",
        color="blue",
    )

    # Mistral

    # Facti

    # Formatting the plot
    ax.set_xlabel("Language", fontsize=16)
    ax.set_ylabel("Micro F1 Score", fontsize=16)
    ax.set_title(
        "Micro F1 Scores by Language for Veracity Prediction.", fontsize=20
    )
    ax.set_xticks(index)
    ax.set_xticklabels(
        data_sorted["Lang"], rotation=45, ha="right", fontsize=15
    )  # Use ha="right" to align labels
    ax.legend(fontsize=16)

    plt.tight_layout()

    # Save the plot to a PNG file
    plt.savefig(f"nli_{split}_test_micro.pdf", format="pdf")
    data_sorted = data.sort_values(by="FV_Macro_F1")
    fig, ax = plt.subplots(figsize=(20, 10))
    bar_width = 0.2
    index = np.arange(len(data_sorted))
    Ollama_Macro_F1_bars = ax.bar(
        index - 2 * bar_width,
        data_sorted["Ollama_Macro_F1"],
        bar_width,
        alpha=opacity,
        label="Mistral-7b",
        color="green",
    )
    gpt3_Macro_F1_bars = ax.bar(
        index - bar_width,
        data_sorted["gpt3_Macro_F1"],
        bar_width,
        alpha=opacity,
        label="GPT-3.5-turbo",
        color="pink",
    )
    gpt3_Macro_F1_bars = ax.bar(
        index,
        data_sorted["gpt4_Macro_F1"],
        bar_width,
        alpha=opacity,
        label="GPT-4",
        color="red",
    )
    FV_Macro_F1_bars = ax.bar(
        index + bar_width,
        data_sorted["FV_Macro_F1"],
        bar_width,
        alpha=opacity,
        label="Factiverse",
        color="blue",
    )

    ax.set_xlabel("Language", fontsize=16)
    ax.set_ylabel("Macro F1 Score", fontsize=16)
    ax.set_title(
        "Macro F1 Scores by Language for Veracity Prediction.", fontsize=20
    )
    ax.set_xticks(index)
    ax.set_xticks(index)
    ax.set_xticklabels(
        data_sorted["Lang"], rotation=45, ha="right", fontsize=15
    )  # Use ha="right" to align labels
    ax.legend(fontsize=16)
    plt.tight_layout()

    # Save the plot to a PNG file
    plt.savefig(f"nli_{split}_test_macro.pdf", format="pdf")

    average_gpt3_Micro_F1 = data["gpt3_Micro_F1"].mean()
    average_gpt3_Macro_F1 = data["gpt3_Macro_F1"].mean()

    average_FV_Micro_F1 = data["FV_Micro_F1"].mean()
    average_FV_Macro_F1 = data["FV_Macro_F1"].mean()

    average_Ollama_Micro_F1 = data["Ollama_Micro_F1"].mean()
    average_Ollama_Macro_F1 = data["Ollama_Macro_F1"].mean()

    # Print the results
    print(f"OpenAI Average Micro-F1: {average_gpt3_Micro_F1:.4f}")
    print(f"OpenAI Average Macro-F1: {average_gpt3_Macro_F1:.4f}\n")

    print(f"Facti Average Micro-F1: {average_FV_Micro_F1:.4f}")
    print(f"Facti Average Macro-F1: {average_FV_Macro_F1:.4f}\n")

    print(f"Mistral Average Micro-F1: {average_Ollama_Micro_F1:.4f}")
    print(f"Mistral Average Macro-F1: {average_Ollama_Macro_F1:.4f}")
