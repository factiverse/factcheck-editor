import requests
import json
import logging
from tqdm import tqdm
import os
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from src.llm_utils.openai_utils import OpenAIUtils
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm_utils.ollama import Ollama
from src.prompts.prompts import CHECKWORTHY_PROMPT
from src.utils.utils import get_access_token, load_json
from src.claim_detection.claim_detection_inference import BERTClaimPredictor


load_dotenv()
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.disabled = True
claim_detection_model = BERTClaimPredictor(
        "claim_detection_unquantized",
        "unquantized",
        "cache",
    )

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

def claim_detection_factiverse_local(claim: str) -> requests.Response:
    """
    Docstring for claim_detection_factiverse_local
    
    :param claim: Description
    :type claim: str
    :return: Description
    :rtype: Response
    """
    pred, scores = claim_detection_model.predict((claim,))
    return pred


def claim_detection_factiverse_local_batch(claims: list) -> list:
    """Predict check worthiness for multiple claims using local Factiverse model.
    
    Args:
        claims: List of claim texts
    
    Returns:
        List of predictions (0 or 1)
    """
    preds, scores = claim_detection_model.predict(tuple(claims))
    return preds.tolist() if hasattr(preds, 'tolist') else list(preds)


def predict_check_worthiness_using_ollama_batch(texts: list, lang: str) -> list:
    """Predict check worthiness for multiple texts using Ollama.
    
    Args:
        texts: List of sentences to predict check-worthiness
        lang: Language code
    
    Returns:
        List of predictions ("Yes" or "No")
    """
    lqg = Ollama()
    results = []
    for text in texts:
        prompt = CHECKWORTHY_PROMPT.format(text=text, lang=lang)
        response = lqg.generate(prompt)
        results.append(sanitize_llm_response(response))
    return results


def predict_claim_check_worthiness_openai_batch(
    texts: list, lang: str, open_ai_utils: OpenAIUtils, model=None
) -> list:
    """Predict check worthiness for multiple texts using OpenAI.
    
    Args:
        texts: List of sentences to predict check-worthiness
        lang: Language code
        open_ai_utils: OpenAIUtils object
        model: Model name (optional)
    
    Returns:
        List of predictions ("Yes" or "No")
    """
    results = []
    for text in texts:
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
        results.append(sanitize_llm_response(response))
    return results

    

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
    split = "fv_claim_test"
    access_token = get_access_token()
    batch_size = 20  # Process 20 claims per batch
    
    for lang in lang_codes.keys():
        open_ai_utils = OpenAIUtils()
        # logger.info("Running claim detection for %s", lang)
        groundtruth_labels = []
        predicted_labels = []
        mistral_predicted_labels = []
        gpt52_preds = []
        claude_opus_4_6_predictions = []
        input_path = f"data/claim_detection/{lang}_{split}.json"
        output_path = f"data/claim_detection/{lang}_{split}_pred.json"
        if not os.path.exists(input_path):
            continue
        if os.path.exists(output_path):
            continue
        with open(
            output_path, "w"
        ) as out_json_file:
            claim_preds = []
            claims = load_json(input_path)
            
            # Process claims in batches
            for i in tqdm(range(0, len(claims), batch_size), desc=f"Processing {lang}"):
                batch = claims[i:i+batch_size]
                batch_texts = [row["claim"] for row in batch]
                
                try:
                    # Run all model predictions in parallel on the batch
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        future_mistral = executor.submit(
                            predict_check_worthiness_using_ollama_batch,
                            texts=batch_texts,
                            lang=lang
                        )
                        future_gpt52 = executor.submit(
                            predict_claim_check_worthiness_openai_batch,
                            texts=batch_texts,
                            lang=lang,
                            open_ai_utils=open_ai_utils,
                            model="gpt-5.2"
                        )
                        future_claude = executor.submit(
                            predict_claim_check_worthiness_openai_batch,
                            texts=batch_texts,
                            lang=lang,
                            open_ai_utils=open_ai_utils,
                            model="claude-opus-4-6"
                        )
                        future_facti = executor.submit(
                            claim_detection_factiverse_local_batch,
                            claims=batch_texts
                        )
                        
                        # Wait for all to complete and get results
                        mistral_predictions = future_mistral.result()
                        gpt52_predictions = future_gpt52.result()
                        claude_predictions = future_claude.result()
                        facti_preds = future_facti.result()
                        
                except Exception as e:
                    logger.exception("Exception occurred while predicting batch: %s", str(e))
                    continue
                
                # Process batch results
                for idx, row in enumerate(batch):
                    new_row = {}
                    new_row["claim"] = row["claim"]
                    new_row["checkworthy"] = row["checkworthy"]
                    
                    mistral_pred = 1 if mistral_predictions[idx] == "Yes" else 0
                    gpt52_prediction_int = 1 if gpt52_predictions[idx] == "Yes" else 0
                    claude_prediction_int = 1 if claude_predictions[idx] == "Yes" else 0
                    facti_pred = facti_preds[idx]
                    
                    new_row["mistral_pred"] = mistral_pred
                    new_row["gpt52_pred"] = gpt52_prediction_int
                    new_row["claude_opus_4_6_pred"] = claude_prediction_int
                    new_row["facti_pred"] = facti_pred
                    
                    mistral_predicted_labels.append(mistral_pred)
                    gpt52_preds.append(gpt52_prediction_int)
                    claude_opus_4_6_predictions.append(claude_prediction_int)
                    predicted_labels.append(facti_pred)
                    groundtruth_labels.append(int(row["checkworthy"]))
                    
                    claim_preds.append(new_row)
            json.dump(claim_preds, out_json_file, indent=4)
        intent_macro_f1 = f1_score(
            groundtruth_labels, predicted_labels, average="macro"
        )
        intent_micro_f1 = f1_score(
            groundtruth_labels, predicted_labels, average="micro"
        )
        print(f"Factiverse [{lang}] - Macro F1: {intent_macro_f1:.4f}, Micro F1: {intent_micro_f1:.4f}")
        intent_macro_f1 = f1_score(
            groundtruth_labels, mistral_predicted_labels, average="macro"
        )
        intent_micro_f1 = f1_score(
            groundtruth_labels, mistral_predicted_labels, average="micro"
        )
        print(f"Mistral [{lang}] - Macro F1: {intent_macro_f1:.4f}, Micro F1: {intent_micro_f1:.4f}")
        intent_macro_f1 = f1_score(
            groundtruth_labels, gpt52_preds, average="macro"
        )
        intent_micro_f1 = f1_score(
            groundtruth_labels, gpt52_preds, average="micro"
        )
        print(f"OpenAI GPT-5.2 [{lang}] - Macro F1: {intent_macro_f1:.4f}, Micro F1: {intent_micro_f1:.4f}")
        intent_macro_f1 = f1_score(
            groundtruth_labels, claude_opus_4_6_predictions, average="macro"
        )
        intent_micro_f1 = f1_score(
            groundtruth_labels, claude_opus_4_6_predictions, average="micro"
        )
        print(f"OpenAI Claude Opus 4-6 [{lang}] - Macro F1: {intent_macro_f1:.4f}, Micro F1: {intent_micro_f1:.4f}")