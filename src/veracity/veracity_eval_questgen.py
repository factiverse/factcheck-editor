import csv
import json
import os
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import math
from src.utils.utils import load_lang_codes

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
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


def compute_metrics(data, model):
    labels = [item["label"] for item in data if f"{model}_label" in item]
    pred = [
        item[f"{model}_label"]
        for item in data
        if f"{model}_label" in item
    ]
    conf_mat = confusion_matrix(labels, pred, labels=["TRUE", "FALSE"])
    pos_f1 = f1_score(labels, pred, average="binary", pos_label="TRUE")
    neg_f1 = f1_score(labels, pred, average="binary", pos_label="FALSE")
    macro_f1 = f1_score(labels, pred, average="macro")

    # Calculate Weighted F1 Score
    micro_f1 = f1_score(labels, pred, average="micro")
    return conf_mat, macro_f1, micro_f1, pos_f1, neg_f1


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
        1 if item["factiverse_label"] == "TRUE" else 0
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
    variant = "questgen_all_search_engines"
    model_scores = {}
    with open(
        f"data/{variant}/questgen_f1.tsv",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"Model\tFV_Macro_F1\tFV_Micro_F1\tPOS_F1\tNEG_F1\n"
        )
        for model in ["human", "fv_qg", "no_qg", "BART", "flan-t5", "mistral", "T5"]:
            # try:
                print(f"data/{variant}/{model}_questgen_pred.json")
                if not os.path.exists(
                    f"data/{variant}/{model}_questgen_pred.json"
                ):
                    print("No data file for model: ", model)
                    continue
                data = load_json(
                    f"data/{variant}/{model}_questgen_pred.json"
                )
                print(len(data))
                if len(data) == 0:
                    print("No data for model: ", model)
                    continue
                # print(data[0])
                # scores = [item["factiverse_score"] for item in data if f"factiverse_score" in item]
                scores = [item["factiverse_score"] if "factiverse_score" in item else 0 for item in data]
                model_scores[model] = scores
                conf_mat, fv_macro_f1, fv_micro_f1, pos_f1, neg_f1 = compute_metrics(
                    data, "factiverse"
                )
                print(
                    fv_macro_f1
                )
                if not math.isnan(fv_macro_f1):
                    f.write(
                        f"{model}\t{fv_macro_f1}\t{fv_micro_f1}\t{pos_f1}\t{neg_f1}\n"
                    )
                    print(
                        (
                            f"{model}\t{fv_macro_f1}\t{fv_micro_f1}\t{pos_f1}\t{neg_f1}\n"
                        )
                    )
            # except Exception as e:
            #     print("Failed to process model: ", model)
                # print(e)
                # continue
    # print(model_scores)
    import numpy as np
    from scipy.stats import ttest_rel
    t5_scores = np.array(model_scores["T5"])
    for model, scores in model_scores.items():
        print(model, len(scores))
        # if model == "T5":
        #     continue
        # print(model, np.mean(scores), np.std(scores))
        # # Synthetic BLEU and ROUGE scores for two models
        
        # scores_b = np.array(scores)

        # # Perform paired t-tests
        # bleu_t_stat, bleu_p_val = ttest_rel(t5_scores, scores_b)

        # print(f"Model: {model},  p-value: {bleu_p_val}")
