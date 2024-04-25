import csv
import json
import os
from typing import Dict, List

import dotenv
import pandas as pd
import requests
from tqdm import tqdm
from code.veracity.llm_nli import (
    predict_stance_ollama,
    predict_stance_openai,
)
from collections import Counter
from code.utils.utils import get_access_token, load_lang_codes, load_json


dotenv.load_dotenv()



def factiverse_verify(query, lang, access_token, questions):
    api_link = os.getenv("SERVER_ENDPOINT")
    api_endpoint = f"{api_link}/stance_detection"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = {
        "claim": query,  # Your search query "lang": "en", # Language code
        "lang": lang,
        "questions": questions
    }
    # print(payload)
    response = requests.post(api_endpoint, headers=headers, json=payload)
    return response


if __name__ == "__main__":
    input = "data/veracity_prediction/dev.jsonl"
    access_token = get_access_token()
    count = 0
    missing_evidence = 0
    ISO639_FILE = {}
    langs = load_lang_codes()
    split = "test"
    for lang in langs.keys():
        lang_name = langs[lang]["name"]
        fact_checked_data = []
        data_file = f"data/veracity_prediction/{lang}_{split}.json"
        if not os.path.exists(data_file):
            continue
        data = load_json(data_file)
        with open(
            f"data/veracity_prediction/{lang}_{split}_results.tsv",
            mode="w",
            encoding="utf-8",
        ) as f:
            for item in tqdm(data):
                try:
                    response = factiverse_verify(item["claim"], lang, access_token)
                    verified_data = {}
                    if response.status_code == 200:
                        response_data = response.json()
                        if len(response_data["evidence"]) > 0:
                            ollama_preds = []
                            gpt3_preds = []
                            gpt4_preds = []
                            for evidence in response_data["evidence"]:
                                ollama_pred = predict_stance_ollama(
                                    claim=response_data["claim"],
                                    evidence=evidence["snippet"],
                                    lang=lang_name,
                                )
                                ollama_preds.append(ollama_pred)
                                evidence["ollama_pred"] = ollama_pred
                                gpt3_pred = predict_stance_openai(
                                    claim=response_data["claim"],
                                    evidence=evidence["snippet"],
                                    lang=lang_name,
                                )
                                gpt4_pred = predict_stance_openai(
                                    claim=response_data["claim"],
                                    evidence=evidence["snippet"],
                                    lang=lang_name,
                                    model="gpt-4",
                                )
                                gpt3_preds.append(gpt3_pred)
                                gpt4_preds.append(gpt4_pred)
                                evidence["gpt3_pred"] = gpt3_pred
                                evidence["gpt4_pred"] = gpt4_pred
                            ollama_count = Counter(ollama_preds)
                            if ollama_count["True"] > ollama_count["False"]:
                                verified_data["ollama_label"] = "True"
                            else:
                                verified_data["ollama_label"] = "False"
                            gpt3_count = Counter(gpt3_preds)
                            if gpt3_count["True"] > gpt3_count["False"]:
                                verified_data["gpt3_label"] = "True"
                            else:
                                verified_data["gpt3_label"] = "False"
                            gpt4_count = Counter(gpt4_preds)
                            if gpt4_count["True"] > gpt4_count["False"]:
                                verified_data["gpt4_label"] = "True"
                            else:
                                verified_data["gpt4_label"] = "False"
                            response_data["claim"] = response_data["claim"]
                            verified_data["factiverse_response"] = response_data
                            verified_data["label"] = item["label"]
                            verified_data["lang"] = lang
                            verified_data["factiverse_score"] = response_data[
                                "finalScore"
                            ]
                            if verified_data["factiverse_score"] > 0.5:
                                verified_data["factiverse_label"] = "True"
                            else:
                                verified_data["factiverse_label"] = "False"
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
                            print("OpenAI GPT-3", verified_data["gpt3_label"])
                            print("OpenAI GPT-4", verified_data["gpt4_label"])
                            print("Ollama", verified_data["ollama_label"])
                            f.write(
                                lang
                                + "\t"
                                + response_data["claim"]
                                + "\t"
                                + str(item["label"])
                                + "\t"
                                + verified_data["factiverse_label"]
                                + "\t"
                                + str(
                                    len(
                                        verified_data["factiverse_response"][
                                            "evidence"
                                        ]
                                    )
                                )
                                + "\n"
                            )
                        else:
                            missing_evidence += 1
                            print("No evidence found ", missing_evidence)

                    else:
                        # Handle errors
                        print("Error:", response.status_code, response.text)

                    fact_checked_data.append(verified_data)
                except Exception as e:
                    print("exception", e)
                    continue
        with open(
            f"data/veracity_prediction/{lang}_{split}_nli_pred.json",
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(fact_checked_data, f, indent=4)
        print("total checked", count)
        print("missing evidence", missing_evidence)
