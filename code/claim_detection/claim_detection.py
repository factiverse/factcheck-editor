import requests
import json
import logging
from tqdm import tqdm
import os
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from code.llm_utils.openai_utils import OpenAIUtils
import random

from code.llm_utils.ollama import Ollama
from code.prompts.prompts import CHECKWORTHY_PROMPT
from code.utils.utils import get_access_token, load_json


load_dotenv()
logger = logging.getLogger(__name__)

def predict_check_worthiness_using_ollama(text: str, lang: str) -> str:
    """Predict check worthiness using LLM.

    Args:
        text: Sentence to predict check-worthiness.
        lang: Language code.

    Returns:
        Return Yes if the sentence is check-worthy, else No.
    """
    lqg = Ollama()
    prompt = CHECKWORTHY_PROMPT.format(text=text, lang=lang)
    response = lqg.generate(prompt)
    return sanitize_llm_response(response)


def sanitize_llm_response(response: str) -> str:
    """Sanitize LLM response to valid labels.

    Args:
        response: LLM response text.

    Returns:
        Yes or No.
    """
    llm_prediction = response
    response_words = response.split(" ")
    if len(response_words) >= 1:
        llm_prediction = response_words[0].strip().replace(".", "").replace(",", "")
    if llm_prediction == "Yes":
        return "Yes"
    elif llm_prediction == "No":
        return "No"
    else: # If LLM response is anything other than Yes/No pick a random label.
        return "Yes" if random.random() >= 0.5 else "No"        


def predict_claim_check_worthiness_openai(
    text: str, lang: str, open_ai_utils: OpenAIUtils, model=None
):
    """Predict check worthiness using OpenAI's GPT-3/GPT-4.

    Args:
        text: Sentence to predict check-worthiness.
        lang: Language code.
        open_ai_utils: OpenAIUtils object.
        model (optional): gpt-3.5-turbo or gpt-4. Defaults to None.

    Returns:
        Yes if the sentence is check-worthy, else No.
    """
    response = open_ai_utils.generate(
        CHECKWORTHY_PROMPT.format(text=text, lang=lang), model
    )
    response = (
        response.strip()
        .strip()
        .replace(".", "")
        .replace(",", "")
        .replace('"', "")
    )
    return sanitize_llm_response(response)


def claim_detection(
    claim: str, access_token: str, lang: str = "en"
) -> requests.Response:
    """Search for claims.

    Args:
        query: Query string.
        lang: Language code.
        access_token: Access token for Factiverse API.

    Returns:
        Response object.
    """
    
    api_endpoint = os.getenv("SERVER_ENDPOINT")
    claim_detection_api_endpoint = f"{api_endpoint}/claim_detection"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "claimScoreThreshold": "0.01",
        "logging": "false",
    }
    payload = {"text": claim, "lang": lang}
    response = requests.post(
        claim_detection_api_endpoint, headers=headers, json=payload
    )
    return response

def predict_checkworthiness_using_factiverse(claim: str) -> int:
    response = claim_detection(
        claim=row["claim"],
        access_token=access_token,
        lang=lang,
    )
    if (
        "detectedClaims" in response.json()
        and len(response.json()["detectedClaims"]) > 0
    ):
        if response.json()["detectedClaims"][0]["score"] >= 0.5:
            return 1
        else:
            return 0
    else:
        return 0

if __name__ == "__main__":
    lang_codes = {}
    with open("code/utils/lang_codes.json", "r") as f:
        lang_codes = json.load(f)
    split = "test"
    access_token = get_access_token()
    for lang in lang_codes.keys():
        open_ai_utils = OpenAIUtils()
        logger.info("Running claim detection for ", lang)
        groundtruth_labels = []
        predicted_labels = []
        mistral_predicted_labels = []
        gpt3_preds = []
        gpt4_preds = []
        if not os.path.exists(f"data/claim_detection/{lang}_{split}.json"):
            continue
        with open(
            f"data/claim_detection/{lang}_{split}_pred.json", "w"
        ) as out_json_file:
            claim_preds = []
            claims = load_json(f"data/claim_detection/{lang}_{split}.json")
            for row in tqdm(claims):
                new_row = {}
                new_row["claim"] = row["claim"]
                try:
                    mistral_prediction = predict_check_worthiness_using_ollama(
                        text=row["claim"], lang=lang
                    )
                    gpt3_prediction = predict_claim_check_worthiness_openai(
                        text=row["claim"],
                        lang=lang,
                        open_ai_utils=open_ai_utils,
                    )
                    gpt4_prediction = predict_claim_check_worthiness_openai(
                        text=row["claim"],
                        lang=lang,
                        open_ai_utils=open_ai_utils,
                        model="gpt-4",
                    )
                except Exception as e:
                    logger.error("Exception", e)
                    continue
                logger.info(mistral_prediction, gpt3_prediction, gpt4_prediction)
                mistral_pred = 1 if mistral_prediction == "Yes" else 0
                mistral_predicted_labels.append(mistral_pred)
                gpt3_prediction_int = 1 if gpt3_prediction == "Yes" else 0
                gpt4_prediction_int = 1 if gpt4_prediction == "Yes" else 0
                gpt3_preds.append(gpt3_prediction_int)
                new_row["mistral_pred"] = mistral_pred
                new_row["gpt3_pred"] = gpt3_prediction_int
                new_row["gpt4_pred"] = gpt4_prediction_int
                gpt4_preds.append(gpt4_prediction_int)
                facti_pred = predict_checkworthiness_using_factiverse(row["claim"])
                predicted_labels.append(facti_pred)
                groundtruth_labels.append(int(row["checkworthy"]))
                logger.info(
                    row["claim"],
                    row["checkworthy"],
                    predicted_labels[-1],
                    mistral_predicted_labels[-1],
                    gpt3_preds[-1],
                    gpt4_preds[-1],
                )
                new_row["checkworthy"] = row["checkworthy"]
                new_row["facti_pred"] = predicted_labels[-1]
                claim_preds.append(new_row)
            json.dump(claim_preds, out_json_file, indent=4)
        intent_macro_f1 = f1_score(
            groundtruth_labels, predicted_labels, average="macro"
        )
        intent_micro_f1 = f1_score(
            groundtruth_labels, predicted_labels, average="micro"
        )
        logger.info("Factiverse", lang, intent_macro_f1, intent_micro_f1)
        intent_macro_f1 = f1_score(
            groundtruth_labels, mistral_predicted_labels, average="macro"
        )
        intent_micro_f1 = f1_score(
            groundtruth_labels, mistral_predicted_labels, average="micro"
        )
        logger.info("Mistral", lang, intent_macro_f1, intent_micro_f1)
        intent_macro_f1 = f1_score(
            groundtruth_labels, gpt3_preds, average="macro"
        )
        intent_micro_f1 = f1_score(
            groundtruth_labels, gpt3_preds, average="micro"
        )
        logger.info("OpenAI GPT-3", lang, intent_macro_f1, intent_micro_f1)
        intent_macro_f1 = f1_score(
            groundtruth_labels, gpt4_preds, average="macro"
        )
        intent_micro_f1 = f1_score(
            groundtruth_labels, gpt4_preds, average="micro"
        )
        logger.info("OpenAI GPT-4", lang, intent_macro_f1, intent_micro_f1)
