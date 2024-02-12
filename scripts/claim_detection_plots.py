import requests
import json
import os
from dotenv import load_dotenv
from sklearn.metrics import f1_score

data_folder = "data/claim_detection"
split = "test"
def load_claim_pred_data(lang):
    """Loads the claim data from the claim detection dataset."""
    with open(f"{data_folder}/{lang}_{split}_pred.json", "r") as json_file:
        return json.load(json_file)


if __name__ == "__main__":
    ISO639_FILE = {}
    with open("code/utils/lang_codes.json", "r") as iso639_file:
        ISO639_FILE = json.load(iso639_file)
    with open(f"{data_folder}/f1_scores.tsv", "w") as f1_file:
        f1_file.write(f"lang\tgpt3_macro\tgpt3_micro\tgpt4_macro\tgpt4_micro\tfacti_macro\tfacti_micro\tmistral_macro\tmistral_micro\n")
        for lang in ISO639_FILE.keys():
            if not os.path.exists(f"{data_folder}/{lang}_{split}_pred.json"):
                continue       
                
            print(lang)
            try:
                data = load_claim_pred_data(lang)
            except Exception as e:
                print("Failed to process lang: ", lang)
                print(e)
                continue 
            mistral_preds = [item['mistral_pred'] for item in data]
            gpt3_preds = [item['gpt3_pred'] for item in data]
            gpt4_preds = [item['gpt4_pred'] for item in data]
            facti_preds = [item['facti_pred'] for item in data]
            true_values = [item['checkworthy'] for item in data]
            gpt3_macro = f1_score(true_values, gpt3_preds, average='macro')
            gpt3_micro = f1_score(true_values, gpt3_preds, average='micro')
            gpt4_macro = f1_score(true_values, gpt4_preds, average='macro')
            gpt4_micro = f1_score(true_values, gpt4_preds, average='micro')
            facti_macro = f1_score(true_values, facti_preds, average='macro')
            facti_micro = f1_score(true_values, facti_preds, average='micro')
            mistral_macro = f1_score(true_values, mistral_preds, average='macro')
            mistral_micro = f1_score(true_values, mistral_preds, average='micro')
            print(f"{lang}\t{gpt3_macro}\t{gpt3_micro}\t{gpt4_macro}\t{gpt4_micro}\t{facti_macro}\t{facti_micro}\t{mistral_macro}\t{mistral_micro}\n")
            f1_file.write(f"{lang}\t{gpt3_macro}\t{gpt3_micro}\t{gpt4_macro}\t{gpt4_micro}\t{facti_macro}\t{facti_micro}\t{mistral_macro}\t{mistral_micro}\n")
            
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    # Load data from a JSON file into a pandas DataFrame
    data = pd.read_csv(f"{data_folder}/f1_scores.tsv", delimiter='\t')

    # Assuming the JSON structure directly maps to the DataFrame structure
    # No need for transformation if the JSON structure matches the DataFrame exactly
    lang_names = {lang: ISO639_FILE[lang]["name"] for lang in ISO639_FILE.keys()}
    print(lang_names)
    # Sort the DataFrame by gpt3_micro
    data['lang'] = data['lang'].map(lang_names)
    print(data)
    data_sorted = data.sort_values(by='facti_micro')
    

    fig, ax = plt.subplots(figsize=(20, 10))
    bar_width = 0.2
    opacity = 0.8
    index = np.arange(len(data_sorted))

    # gpt3
    mistral_micro_bars = ax.bar(index - 2*bar_width, data_sorted['mistral_micro'], bar_width, alpha=opacity, label='Mistral-7b', color='green')
    gpt3_micro_bars = ax.bar(index - bar_width, data_sorted['gpt3_micro'], bar_width,  alpha=opacity, label='GPT-3.5-turbo', color='yellow')
    gpt4_micro_bars = ax.bar(index, data_sorted['gpt4_micro'], bar_width,  alpha=opacity, label='GPT-4', color='red')
    

    # Facti
    facti_micro_bars = ax.bar(index  + bar_width, data_sorted['facti_micro'], bar_width, alpha=opacity, label='Factiverse', color='blue')
    

    # Mistral
    
    

    # Formatting the plot
    ax.set_xlabel('Language', fontsize=16)
    ax.set_ylabel('Micro F1 Score', fontsize=16)
    ax.set_title('Micro F1 Scores by Language for Claim Detection.', fontsize=20)
    ax.set_xticks(index)
    ax.set_xticklabels(data_sorted['lang'], rotation=45, ha="right", fontsize=10)  # Use ha="right" to align labels
    ax.legend(fontsize=16)

    plt.tight_layout()

    # Save the plot to a PNG file
    plt.savefig(f"{data_folder}_test_micro.pdf", format='pdf')
    data_sorted = data.sort_values(by='facti_macro')
    fig, ax = plt.subplots(figsize=(20, 10))
    bar_width = 0.2
    index = np.arange(len(data_sorted))
    mistral_macro_bars = ax.bar(index - 2*bar_width, data_sorted['mistral_macro'], bar_width,alpha=opacity,  label='Mistral-7b', color='green')
    gpt3_macro_bars = ax.bar(index - bar_width, data_sorted['gpt3_macro'], bar_width, alpha=opacity, label='GPT-3.5-turbo', color='yellow')
    gpt4_macro_bars = ax.bar(index, data_sorted['gpt4_macro'], bar_width, alpha=opacity, label='GPT-4', color='red')
    facti_macro_bars = ax.bar(index + bar_width, data_sorted['facti_macro'], bar_width, alpha=opacity, label='Factiverse', color='blue')
    
    ax.set_xlabel('Language', fontsize=16)
    ax.set_ylabel('Macro F1 Score', fontsize=16)
    ax.set_title('Macro F1 Scores by Language for Claim Detection.', fontsize=20)
    ax.set_xticks(index)
    ax.set_xticks(index)
    ax.set_xticklabels(data_sorted['lang'], rotation=45, ha="right", fontsize=10)  # Use ha="right" to align labels
    ax.legend(fontsize=16)
    plt.tight_layout()


    # Save the plot to a PNG file
    plt.savefig(f"{data_folder}_test_macro.pdf", format='pdf')
    
    average_gpt3_micro = data['gpt3_micro'].mean()
    average_gpt3_macro = data['gpt3_macro'].mean()
    
    average_gpt4_micro = data['gpt4_micro'].mean()
    average_gpt4_macro = data['gpt4_macro'].mean()

    average_facti_micro = data['facti_micro'].mean()
    average_facti_macro = data['facti_macro'].mean()

    average_mistral_micro = data['mistral_micro'].mean()
    average_mistral_macro = data['mistral_macro'].mean()

    # Print the results
    print(f"gpt3 Average Micro-F1: {average_gpt3_micro:.4f}")
    print(f"gpt3 Average Macro-F1: {average_gpt3_macro:.4f}\n")
    
    print(f"gpt4 Average Micro-F1: {average_gpt4_micro:.4f}")
    print(f"gpt4 Average Macro-F1: {average_gpt4_macro:.4f}\n")

    print(f"Facti Average Micro-F1: {average_facti_micro:.4f}")
    print(f"Facti Average Macro-F1: {average_facti_macro:.4f}\n")

    print(f"Mistral Average Micro-F1: {average_mistral_micro:.4f}")
    print(f"Mistral Average Macro-F1: {average_mistral_macro:.4f}")