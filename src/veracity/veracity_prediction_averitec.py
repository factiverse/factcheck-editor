import csv
import json
import os
from typing import Dict, List

import dotenv
import requests

from collections import Counter
from src.utils.utils import get_access_token, load_lang_codes, load_json


def factiverse_verify(query, lang, access_token):
    api_link = os.getenv("SERVER_ENDPOINT")
    api_endpoint = f"{api_link}/stance_detection"
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
    dotenv.load_dotenv()
    access_token = get_access_token()
    data_folder = f"data/veracity_prediction/averitec/"
    split = "dev"
    data_file = f"{data_folder}/averitec-{split}.json"

    data = load_json(data_file)
    print("data loaded", len(data))
    fact_checked_data = []
    count = 0
    missing_evidence = 0
    for claim_data in data:
        try:
            claim = claim_data["claim"]
            print(claim)
            response = factiverse_verify(claim, "en", access_token)
            print(response.status_code)
            if response.status_code == 200:
                response_data = response.json()

                # print(response_data)
                # claim_data["factiverse_response"] = response_data
                claim_data["raw_response"] = response_data
                evidences = []
                for evidence in response_data["evidence"]:
                    print(evidence["simScore"])
                    evidence_data = {
                        "url": evidence["url"],
                        "questions": (
                            evidence["rewritten_query"]
                            if evidence.get("rewritten_query", "")
                            else evidence.get("original_query", claim)
                        ),
                        "answer": evidence["snippet"],
                        "scrascraped_text": evidence["snippet"],
                    }
                    evidences.append(evidence_data)
                claim_data["evidence"] = evidences
                if len(response_data["evidence"]) > 0:
                    fv_score = response_data["finalScore"]
                    print("fv_score", fv_score)
                    if fv_score >= 0.55:
                        claim_data["pred_label"] = "Supported"
                    elif fv_score >= 0.5 and fv_score < 0.55:
                        claim_data["pred_label"] = (
                            "Conflicting Evidence/Cherrypicking"
                        )
                    else:
                        claim_data["pred_label"] = "Refuted"
                    # pred_data = (
                    #     +response_data["claim"]
                    #     + "\t"
                    #     + claim_data["pred_label"]
                    #     + "\t"
                    #     + str(len(claim_data["evidence"]))
                    #     + "\n"
                    # )
                    # print(claim_data)
                else:
                    missing_evidence += 1
                    print("No evidence found ", missing_evidence)
                    claim_data["pred_label"] = "Not Enough Evidence"
                print(claim_data["pred_label"])
                count += 1
            else:
                # Handle errors
                print("Error:", response.status_code, response.text)

            fact_checked_data.append(claim_data)
        except Exception as e:
            print("exception", e)
            continue
        # break
    with open(
        f"{data_folder}/averitec_{split}_pred_raw_evidence.json",
        mode="w",
        encoding="utf-8",
    ) as f:
        json.dump(fact_checked_data, f, indent=4)
    print("total checked", count)
    print("missing evidence", missing_evidence)
    # break
