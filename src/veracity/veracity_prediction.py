import csv
import json
import os
from typing import Dict, List

import dotenv
import pandas as pd
import requests
from tqdm import tqdm
from src.veracity.llm_nli import (
    predict_stance_ollama,
    predict_stance_openai,
)
from collections import Counter
from src.utils.utils import get_access_token, load_lang_codes, load_json
import random


dotenv.load_dotenv()



def factiverse_verify(query, lang, access_token):
    api_link = os.getenv("SERVER_ENDPOINT")
    api_endpoint = f"{api_link}/stance_detection"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = {
        "claim": query,  # Your search query "lang": "en", # Language code
        "lang": lang
    }
    # print(payload)
    response = requests.post(api_endpoint, headers=headers, json=payload)
    return response


if __name__ == "__main__":
    input = "data/veracity_prediction/test_dec_2025.jsonl"
    access_token = get_access_token()
    count = 0
    missing_evidence = 0
    ISO639_FILE = {}
    langs = load_lang_codes()
    # split = "test_dec_2025"
    split = "test_jan_2026"
    fact_checked_data = []
    cur_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    # data_file = f"data/veracity_prediction/test_dec_2025.json"
    # split = "train_factisearch_20251225"
    data_file = f"data/veracity_prediction/{split}.json"
    data = load_json(data_file)
    lang_distribution = Counter([item.get("lang") for item in data])
    print("Language distribution:")
    for lang, count in sorted(lang_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count}")
    retrieval_model = "paraphrase-multilingual"
    # retrieval_model = "bge-m3"
    run_llm_verification = True
    random.shuffle(data, random.Random(42).random)
    # sort data so English ('en') entries come first, then by language code
    data = sorted(data, key=lambda x: (
        0 if x.get("lang") == "en" and x.get("label") != "REFUTES" else 1 if x.get("lang") == "en" else 2,
        x.get("lang", "")
    ))
    # data = sorted(data, key=lambda x: (
    #     0 if x.get("lang") == "en" and x.get("label") == "REFUTES" else 1 if x.get("lang") == "en" else 2,
    #     x.get("lang", "")
    # ))
    


    with open(
        f"data/veracity_prediction/{split}_results_debug_{retrieval_model}_{cur_timestamp}.jsonl",
        mode="w",
        encoding="utf-8",
    ) as f:
        for item in tqdm(data[1176:]):
            # try:
            lang = item["lang"]
            lang_name = langs[lang]["name"]
            response = factiverse_verify(item["claim"], lang, access_token)
            verified_data = {}
            if response.status_code == 200:
                response_data = response.json()
                print("Response data:", response_data["finalLabelDescription"])
                if len(response_data["evidence"]) > 0:
                    ollama_preds = []
                    gpt3_preds = []
                    gpt4_preds = []
                    full_evidence = ""
                    for evidence in response_data["evidence"]:
                        if evidence.get("snippet", ""):
                            if evidence.get("rewritten_query", ""):
                                full_evidence += (
                                    evidence["rewritten_query"] + " "
                                )
                            full_evidence += evidence["snippet"] + " "
                    if run_llm_verification:
                        ollama_pred = predict_stance_ollama(
                            claim=response_data["claim"],
                            evidence=full_evidence,
                            lang=lang_name,
                        )
                        gpt4_pred = predict_stance_openai(
                            claim=response_data["claim"],
                            evidence=full_evidence,
                            lang=lang_name,
                            model="gpt4o",
                        )
                        gpt5_pred = predict_stance_openai(
                            claim=response_data["claim"],
                            evidence=full_evidence,
                            lang=lang_name,
                            model="gpt-5.2",
                        )
                        verified_data["ollama_label"] = ollama_pred
                        verified_data["gpt4o_label"] = gpt4_pred
                        verified_data["gpt5_label"] = gpt5_pred
                        print("OpenAI GPT-4", verified_data["gpt4o_label"])
                        print("OpenAI GPT-5.2", verified_data["gpt5_label"])
                        print("Ollama", verified_data["ollama_label"])
                    verified_data["claim"] = response_data["claim"]
                    verified_data["factiverse_response"] = response_data
                    verified_data["label"] = item["label"]
                    verified_data["lang"] = lang
                    verified_data["factiverse_score"] = response_data[
                        "finalScore"
                    ]
                    verified_data["factiverse_int_label"] = response_data[
                        "finalPrediction"
                    ]
                    verified_data["factiverse_label"] = response_data[
                        "finalLabelDescription"
                    ]
                    count += 1
                    print(
                        response_data["claim"],
                        str(item["label"]),
                        verified_data["factiverse_label"],
                        len(
                            verified_data["factiverse_response"][
                                "evidence"
                            ]
                        ),
                    )

                    f.write(json.dumps(verified_data) + "\n")
                    f.flush()
                else:
                    missing_evidence += 1
                    print("No evidence found ", missing_evidence)

            else:
                # Handle errors
                print("Error:", response.status_code, response.text, item["claim"])

            fact_checked_data.append(verified_data)
            # break
            # except Exception as e:
            #     print("exception", e)
            #     continue
    with open(
        f"data/veracity_prediction/{split}_nli_pred_debug_{retrieval_model}_{cur_timestamp}.json",
        mode="w",
        encoding="utf-8",
    ) as f:
        json.dump(fact_checked_data, f, indent=4)
    print("total checked", count)
    print("missing evidence", missing_evidence)
