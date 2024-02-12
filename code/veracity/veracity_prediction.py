import csv
import json
import os
from typing import Dict, List

import dotenv
import pandas as pd
import requests
from tqdm import tqdm
from src.search.nli_data.llm_nli import (
    predict_stance_ollama,
    predict_stance_openai,
)
from collections import Counter


dotenv.load_dotenv()
api_link = os.getenv("SERVER_ENDPOINT")
api_endpoint = f"{api_link}/v1/stance_detection"
token_url = "https://factiverse-dev.eu.auth0.com/oauth/token"


def load_manual_nli_data(lang):
    """Loads the claim data from the claim detection dataset."""
    data = []
    with open(f"src/search/nli_data/{lang}.json", "r") as json_file:
        return json.load(json_file)


def get_access_token(client_id, client_secret, token_url):
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "audience": "https://factiverse/api",
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(
            f"Failed to obtain token: {response.status_code} {response.text}"
        )


def load_data(csv_file_path: str) -> List[Dict[str, any]]:
    data_list = []
    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        column_names = next(csv_reader)
        print(column_names)
        for row in csv_reader:
            if len(row) != len(column_names):
                continue
            row_dict = {
                column_names[i]: row[i] for i in range(len(column_names))
            }
            data_list.append(row_dict)
    return data_list


def load_factisearch_json_data(json_file_path: str) -> List[Dict[str, any]]:
    data_list = []
    with open(json_file_path, mode="r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    df = pd.DataFrame(data_list)
    print(len(data_list))
    # print language distribution data_list unique values of lang.
    print(df["lang"].value_counts())
    print(df["label"].value_counts())
    return df


def verify(query, lang, access_token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = {
        "claim": query,  # Your search query "lang": "en", # Language code
        "lang": lang,
    }
    # print(payload)
    response = requests.post(api_endpoint, headers=headers, json=payload)
    return response


if __name__ == "__main__":
    input = "src/search/nli_data/dev.jsonl"
    client_id = os.getenv("AUTH0_CLIENT_ID")
    client_secret = os.getenv("AUTH0_SECRET")
    access_token = get_access_token(client_id, client_secret, token_url)
    count = 0
    missing_evidence = 0
    ISO639_FILE = {}
    queries = "expanded"
    with open("src/claim_search/iso639-1.json", "r") as iso639_file:
        ISO639_FILE = json.load(iso639_file)
    for lang in ISO639_FILE.keys():

        fact_checked_data = []
        if not os.path.exists(f"src/search/nli_data/{lang}.json"):
            continue
        data = load_manual_nli_data(lang)
        print(lang, len(data))
        if lang == "en":
            continue
        with open(
            f"src/search/nli_data/{lang}_{queries}_results.tsv",
            mode="w",
            encoding="utf-8",
        ) as f:
            for item in data:
                try:
                    print(item["claim"], str(item["label"]))
                    response = verify(item["claim"], lang, access_token)
                    verified_data = {}
                    if response.status_code == 200:
                        # Successful request
                        response_data = response.json()
                        # print(data)
                        # break
                        # print(json.dumps(data, indent=4))
                        # print(len(data["evidence"]))
                        if len(response_data["evidence"]) > 0:
                            # print(response_data["claim"])
                            ollama_preds = []
                            gpt3_preds = []
                            gpt4_preds = []
                            for evidence in response_data["evidence"]:
                                ollama_pred = predict_stance_ollama(
                                    claim=response_data["claim"],
                                    evidence=evidence["snippet"],
                                    lang=lang,
                                )
                                ollama_preds.append(ollama_pred)
                                evidence["ollama_pred"] = ollama_pred
                                gpt3_pred = predict_stance_openai(
                                    claim=response_data["claim"],
                                    evidence=evidence["snippet"],
                                    lang=lang,
                                )
                                gpt4_pred = predict_stance_openai(
                                    claim=response_data["claim"],
                                    evidence=evidence["snippet"],
                                    lang=lang,
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

                    # if len(fact_checked_data) == 10:
                    #     break
                except Exception as e:
                    print("exception", e)
                    continue
        with open(
            f"src/search/nli_data/{lang}_{queries}_nli_pred.json",
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(fact_checked_data, f, indent=4)
        print("total checked", count)
        print("missing evidence", missing_evidence)
