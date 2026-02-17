import csv
import json
import os
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import math
from src.utils.utils import load_lang_codes
import argparse

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def find_majority_value_arbitrary_strings(string_list):
    """
    Finds the majority value in a list of arbitrary strings.
    If there's no majority, returns None.
    """
    count_dict = {}
    for item in string_list:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1

    max_count = 0
    majority_value = None
    for key, value in count_dict.items():
        if value > max_count and value > len(string_list) / 2:
            max_count = value
            majority_value = key

    return majority_value


def compute_metrics_binary(data, model):
    labels = [item["label"] for item in data if f"{model}_label" in item]
    pred = [
        1 if item[f"{model}_label"] == "True" else 0
        for item in data
        if f"{model}_label" in item
    ]
    per_class_f1 = f1_score(labels, pred, average=None, labels=[1, 0])
    macro_f1 = f1_score(labels, pred, average="macro")

    # Calculate Weighted F1 Score
    micro_f1 = f1_score(labels, pred, average="micro")
    return per_class_f1, macro_f1, micro_f1

def compute_metrics(data, model):
    labels = [item["label"] for item in data if f"{model}_label" in item]
    valid_labels = ["SUPPORTS", "REFUTES", "MIXED", "NOT_ENOUGH_INFO"]
        
    pred = [
        item[f"{model}_label"]
        for item in data
        if f"{model}_label" in item
    ]
    
    per_class_f1 = f1_score(labels, pred, average=None, labels=valid_labels)
    macro_f1 = f1_score(labels, pred, average="macro")

    # Calculate Weighted F1 Score
    weighted_f1 = f1_score(labels, pred, average="weighted")
    return per_class_f1, macro_f1, weighted_f1


def compute_metrics_majority(data):
    labels = [item["label"] for item in data if "factiverse_label" in item]
    pred = [0 for item in data if "factiverse_label" in item]
    print(len(labels), len(pred))
    conf_mat = confusion_matrix(labels, pred, labels=[1, 0])
    macro_f1 = f1_score(labels, pred, average="macro")

    # Calculate Weighted F1 Score
    weighted_f1 = f1_score(labels, pred, average="weighted")
    return conf_mat, macro_f1, weighted_f1


def compute_metrics_lang(data, lang):
    labels = [
        item["label"]
        for item in data
        if "factiverse_label" in item and item["lang"] == lang
    ]
    pred = [
        1 if item["factiverse_label"] == "True" else 0
        for item in data
        if "factiverse_label" in item and item["lang"] == lang
    ]
    conf_mat = confusion_matrix(labels, pred, labels=[1, 0])
    macro_f1 = f1_score(labels, pred, average="macro")

    # Calculate Weighted F1 Score
    weighted_f1 = f1_score(labels, pred, average="weighted")
    acc_score = accuracy_score(labels, pred)
    return conf_mat, macro_f1, weighted_f1, acc_score


def compute_aggregated_metrics(data):
    labels = [
        item["label"] for item in data if "factiverse_aggregated_label" in item
    ]
    pred = [
        item["factiverse_aggregated_label"]
        for item in data
        if "factiverse_aggregated_label" in item
    ]
    conf_mat = confusion_matrix(labels, pred, labels=[1, 0])
    macro_f1 = f1_score(labels, pred, average="macro")

    # Calculate Weighted F1 Score
    weighted_f1 = f1_score(labels, pred, average="weighted")
    return conf_mat, macro_f1, weighted_f1


def compute_aggregated_metrics_majority(data):
    labels = [
        item["label"] for item in data if "factiverse_aggregated_label" in item
    ]
    pred = [0 for item in data if "factiverse_aggregated_label" in item]
    conf_mat = confusion_matrix(labels, pred, labels=[1, 0])
    macro_f1 = f1_score(labels, pred, average="macro")

    # Calculate Weighted F1 Score
    weighted_f1 = f1_score(labels, pred, average="weighted")
    return conf_mat, macro_f1, weighted_f1


def compute_aggregated_metrics_lang(data, lang):
    labels = [
        item["label"]
        for item in data
        if "factiverse_aggregated_label" in item and item["lang"] == lang
    ]
    pred = [
        item["factiverse_aggregated_label"]
        for item in data
        if "factiverse_aggregated_label" in item and item["lang"] == lang
    ]
    conf_mat = confusion_matrix(labels, pred, labels=[1, 0])
    macro_f1 = f1_score(labels, pred, average="macro")

    # Calculate Weighted F1 Score
    weighted_f1 = f1_score(labels, pred, average="weighted")
    return conf_mat, macro_f1, weighted_f1


def compute_aggregated_metrics_lang_majority(data, lang):
    labels = [
        item["label"]
        for item in data
        if "factiverse_aggregated_label" in item and item["lang"] == lang
    ]
    majority_label = find_majority_value_arbitrary_strings(labels)
    pred = [
        majority_label
        for item in data
        if "factiverse_aggregated_label" in item and item["lang"] == lang
    ]
    conf_mat = confusion_matrix(labels, pred, labels=[1, 0])
    macro_f1 = f1_score(labels, pred, average="macro")

    # Calculate Weighted F1 Score
    weighted_f1 = f1_score(labels, pred, average="weighted")
    return conf_mat, macro_f1, weighted_f1


def aggregate_scores(data):
    for item in data:
        if "factiverse_response" in item:
            aggregated_final_score = 0
            if len(item["factiverse_response"]["evidence"]) == 0:
                continue

            num_docs = 0
            for evidence in item["factiverse_response"]["evidence"]:
                if evidence["simScore"] >= 0.6:
                    aggregated_final_score += evidence["softmaxScore"][1]
                    num_docs += 1

            if num_docs == 0:
                continue
            aggregated_final_score = aggregated_final_score / num_docs
            item["factiverse_aggregated_label"] = aggregated_final_score
            if aggregated_final_score >= 0.5:
                item["factiverse_aggregated_label"] = 1
            else:
                item["factiverse_aggregated_label"] = 0


def get_languages(data):
    languages = set()
    for item in data:
        if "lang" in item:
            languages.add(item["lang"])
    return languages


def fix_labels(csv_file, json_file):
    json_data = load_json(json_file)
    print(json_data[0])
    csv_data = csv.reader(open(csv_file, "r"), delimiter="\t")
    new_json_data = []
    new_json_item = {}
    for csv_item, json_item in zip(csv_data, json_data):
        new_json_item = json_item
        new_json_item["label"] = csv_item[2]
        new_json_item["lang"] = csv_item[0]
        json_item["label"] = csv_item[2]
        if "label" in new_json_item:
            new_json_data.append(new_json_item)
    return new_json_data


if __name__ == "__main__":

    lang_codes = load_lang_codes()
    split = "test"
    # pred_file_prefix = "test_dec_2025"
    
    # pred_file = f"{pred_file_prefix}_nli_pred.json"
    # pred_file = f"{pred_file_prefix}_results_debug.jsonl"
    parser = argparse.ArgumentParser(description='Compute NLI evaluation metrics')
    parser.add_argument('--pred-file', type=str, required=True, help='Prediction file name')
    args = parser.parse_args()
    
    pred_file = args.pred_file
    with open(
        f"data/veracity_prediction/{pred_file}_f1.tsv",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"Lang\tFV_Macro_F1\tFV_Micro_F1\tOllama_Macro_F1\tOllama_Micro_F1\tgpt3_Macro_F1\tgpt3_Micro_F1\tgpt4_Macro_F1\tgpt4_Micro_F1\n"
        )
        # for lang in lang_codes.keys():
        # try:
        # pred_file_prefix = f"{lang}_{split}"
        # if not os.path.exists(
        #     f"data/veracity_prediction/{pred_file}"
        # ):
        #     print("No data file for lang: ", lang)
        #     continue
        if pred_file.endswith(".jsonl"):
            data = load_jsonl(
                f"data/veracity_prediction/{pred_file}"
            )
        else:
            data = load_json(
                f"data/veracity_prediction/{pred_file}"
            )
        print(len(data))
        if len(data) == 0:
            print("No data for lang: ", lang)
            
        # Get all languages in the dataset
        languages = get_languages(data)
        print(f"Languages in dataset: {languages}")
        
        # Print overall scores
        print("\n" + "="*80)
        print("OVERALL SCORES (All Languages)")
        print("="*80)
        per_class_f1, fv_macro_f1, fv_micro_f1 = compute_metrics(
            data, "factiverse"
        )
        print("Factiverse Per-class F1 (SUPPORTS, REFUTES, MIXED, NOT_ENOUGH_INFO):", per_class_f1)
        print("Factiverse Macro F1:", fv_macro_f1)
        print("Factiverse Micro F1:", fv_micro_f1)
        if all("ollama_label" in item for item in data):
            per_class_f1, ollama_macro_f1, ollama_micro_f1 = compute_metrics(
                data, "ollama"
            )
            print("Ollama Per-class F1 (SUPPORTS, REFUTES, MIXED, NOT_ENOUGH_INFO):", per_class_f1)
            print("Ollama Macro F1:", ollama_macro_f1)
            print("Ollama Micro F1:", ollama_micro_f1)
        if all("gpt4o_label" in item for item in data):
            per_class_f1, gpt4o_macro_f1, gpt4o_micro_f1 = compute_metrics(
                data, "gpt4o"
            )
            print("GPT-4 Per-class F1 (SUPPORTS, REFUTES, MIXED, NOT_ENOUGH_INFO):", per_class_f1)
            print("GPT-4 Macro F1:", gpt4o_macro_f1)
            print("GPT-4 Micro F1:", gpt4o_micro_f1)
        if all("gpt5_label" in item for item in data):
            per_class_f1, gpt5_macro_f1, gpt5_micro_f1 = compute_metrics(
                data, "gpt5"
            )
            print("GPT-5 Per-class F1 (SUPPORTS, REFUTES, MIXED, NOT_ENOUGH_INFO):", per_class_f1)
            print("GPT-5 Macro F1:", gpt5_macro_f1)
            print("GPT-5 Micro F1:", gpt5_micro_f1)
        
        # Print scores for each language
        for lang in sorted(languages):
            print("\n" + "="*80)
            print(f"SCORES FOR LANGUAGE: {lang}")
            print("="*80)
            
            # Filter data for current language
            lang_data = [item for item in data if item.get("lang") == lang]
            print(f"Number of samples: {len(lang_data)}")
            
            # Compute metrics for Factiverse
            per_class_f1, fv_macro_f1_lang, fv_micro_f1_lang = compute_metrics(
                lang_data, "factiverse"
            )
            print("Factiverse Per-class F1 (SUPPORTS, REFUTES, MIXED, NOT_ENOUGH_INFO):", per_class_f1)
            print("Factiverse Macro F1:", fv_macro_f1_lang)
            print("Factiverse Micro F1:", fv_micro_f1_lang)
            
            # Compute metrics for Ollama if available
            if any("ollama_label" in item for item in lang_data):
                per_class_f1, ollama_macro_f1_lang, ollama_micro_f1_lang = compute_metrics(
                    lang_data, "ollama"
                )
                print("Ollama Per-class F1 (SUPPORTS, REFUTES, MIXED, NOT_ENOUGH_INFO):", per_class_f1)
                print("Ollama Macro F1:", ollama_macro_f1_lang)
                print("Ollama Micro F1:", ollama_micro_f1_lang)
            
            # Compute metrics for GPT-4 if available
            if any("gpt4o_label" in item for item in lang_data):
                per_class_f1, gpt4o_macro_f1_lang, gpt4o_micro_f1_lang = compute_metrics(
                    lang_data, "gpt4o"
                )
                print("GPT-4 Per-class F1 (SUPPORTS, REFUTES, MIXED, NOT_ENOUGH_INFO):", per_class_f1)
                print("GPT-4 Macro F1:", gpt4o_macro_f1_lang)
                print("GPT-4 Micro F1:", gpt4o_micro_f1_lang)
            
            # Compute metrics for GPT-5 if available
            if any("gpt5_label" in item for item in lang_data):
                per_class_f1, gpt5_macro_f1_lang, gpt5_micro_f1_lang = compute_metrics(
                    lang_data, "gpt5"
                )
                print("GPT-5 Per-class F1 (SUPPORTS, REFUTES, MIXED, NOT_ENOUGH_INFO):", per_class_f1)
                print("GPT-5 Macro F1:", gpt5_macro_f1_lang)
                print("GPT-5 Micro F1:", gpt5_micro_f1_lang)
        
        lang = "all"
        
        # if not math.isnan(fv_macro_f1) and not math.isnan(ollama_macro_f1) and not math.isnan(gpt4o_macro_f1) and not math.isnan(gpt5_macro_f1):
        #     f.write(
        #         f"{lang}\t{fv_macro_f1}\t{fv_micro_f1}\t{ollama_macro_f1}\t{ollama_micro_f1}\t{gpt4o_macro_f1}\t{gpt4o_micro_f1}\t{gpt5_macro_f1}\t{gpt5_micro_f1}\n"
        #     )
        #     print(
        #         (
        #             f"{lang}\t{fv_macro_f1}\t{fv_micro_f1}\t{ollama_macro_f1}\t{ollama_micro_f1}\t{gpt4o_macro_f1}\t{gpt4o_micro_f1}\t{gpt5_macro_f1}\t{gpt5_micro_f1}\n"
        #         )
        #     )
        # except Exception as e:
        #     print("Failed to process lang: ")
        #     print(e)
