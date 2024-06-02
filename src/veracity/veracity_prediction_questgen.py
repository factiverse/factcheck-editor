import csv
import json
import os
from typing import Dict, List

import dotenv
import requests

from collections import Counter
from src.utils.utils import get_access_token, load_lang_codes, load_json





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
    dotenv.load_dotenv()
    access_token = get_access_token()
    
    data_file = f"data/questgen_100_claims.json"
  
    data = load_json(data_file)
    # models = data.keys()
    models = ["llama2", "BART", "flan-t5", "mistral", "T5", "fv_qg", "human"]
    variant = "serper_test"
    # models = ["no_qg"]
    for model in models:
        count = 0
        missing_evidence = 0
        fact_checked_data = []
        with open(
            f"data/{variant}/questgen_pred_{model}.json",
            mode="w",
            encoding="utf-8",
        ) as f:
            if model in data:
                claims = data[model]
            else:
                claims = data["BART"]
            for claim, item in claims.items():
                try:
                    # print(item)
                    questions = item["pred"]
                    if model not in data:
                        questions = []
                        if model == "human":
                            questions = item["questions-all"].split("\n")
                    # print(questions)
                    response = factiverse_verify(claim, "en", access_token, questions=questions)
                    print(model, questions)
                    # break
                    # response = {"status_code": 500}
                    verified_data = {}
                    print(response.status_code)
                    if response.status_code == 200:
                        response_data = response.json()
                        # print(response_data)
                        verified_data["factiverse_response"] = response_data
                        verified_data["label"] = item["label"]
                        verified_data["lang"] = "en"
                        if len(response_data["evidence"]) > 0:
                            verified_data["factiverse_score"] = response_data[
                                "finalScore"
                            ]
                            if verified_data["factiverse_score"] > 0.5:
                                verified_data["factiverse_label"] = "TRUE"
                            else:
                                verified_data["factiverse_label"] = "FALSE"
                            count += 1
                            # print(
                            #     response_data["claim"],
                            #     str(item["label"]),
                            #     verified_data["factiverse_label"],
                            #     len(
                            #         verified_data["factiverse_response"][
                            #             "evidence"
                            #         ]
                            #     ),
                            # )
                        
                            pred_data = (model
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
                                + "\n")
                            print(pred_data)
                            f.write(
                                pred_data
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
                # break
        with open(
            f"data/{variant}/{model}_questgen_pred.json",
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(fact_checked_data, f, indent=4)
        print("total checked", count)
        print("missing evidence", missing_evidence)
        # break
