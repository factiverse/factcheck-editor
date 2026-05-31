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
import argparse

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
FACTIVERSE_LOCAL_MODEL_PATH = os.path.expanduser(
    "~/repos/ml-models/wandb/chosen-20260530_075240-he3juxd5/files/best_model/"
)

# Lazy singleton — the XLM-R-XL local Factiverse model is ~7 GB on GPU,
# so we only construct it the first time `factiverse_local` is actually
# requested. Runs with --models gpt55 / claude / mistral skip the load
# entirely.
_claim_detection_model: BERTClaimPredictor | None = None


def get_claim_detection_model() -> BERTClaimPredictor:
    """Return the local Factiverse claim-detector, loading on first call."""
    global _claim_detection_model
    if _claim_detection_model is None:
        print(f"[factiverse_local] Loading model from {FACTIVERSE_LOCAL_MODEL_PATH} …")
        _claim_detection_model = BERTClaimPredictor(
            FACTIVERSE_LOCAL_MODEL_PATH,
            "unquantized",
            "cache",
        )
    return _claim_detection_model

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


_TEXT_KEYS = ("claim", "sentence", "text")
_LABEL_KEYS = ("checkworthy", "checkworthiness", "label", "labels")


def _pick(row: dict, candidates: tuple, what: str):
    """Return the first value found in ``row`` among ``candidates`` keys.

    Lets the loader accept JSONL rows from multiple producers:
      text  → 'claim' | 'sentence' | 'text'
      label → 'checkworthy' | 'checkworthiness' | 'label' | 'labels'
    """
    for key in candidates:
        if key in row:
            return row[key]
    raise KeyError(
        f"Row missing all of {candidates} (looking for {what}): {row}"
    )


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
    print("calling ", claim_detection_api_endpoint)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "claimScoreThreshold": "0.0001",
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
    pred, scores = get_claim_detection_model().predict((claim,))
    return pred


def claim_detection_factiverse_local_batch(claims: list) -> list:
    """Predict check worthiness for multiple claims using local Factiverse model.

    Returns:
        List of {"score": <float prob>, "label": <0 | 1>} dicts.
    """
    preds, scores = get_claim_detection_model().predict(tuple(claims))
    preds = preds.tolist() if hasattr(preds, "tolist") else list(preds)
    scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)
    return [{"score": float(s), "label": int(p)} for p, s in zip(preds, scores)]


def predict_check_worthiness_using_ollama_batch(texts: list, lang: str) -> list:
    """Predict check worthiness for multiple texts using Ollama.

    Returns:
        List of {"label": "Yes"|"No", "raw_text": str, "raw_response": dict}
        where raw_response is the full Ollama API response (timings,
        eval_count, context, etc.).
    """
    lqg = Ollama()
    results = []
    for text in texts:
        prompt = CHECKWORTHY_PROMPT.format(text=text, lang=lang)
        raw_response = lqg.generate_full(prompt)
        raw_text = (raw_response.get("response") or "").strip()
        results.append({
            "label": sanitize_llm_response(raw_text),
            "raw_text": raw_text,
            "raw_response": raw_response,
        })
    return results


def predict_claim_check_worthiness_openai_batch(
    texts: list, lang: str, open_ai_utils: OpenAIUtils, model=None
) -> list:
    """Predict check worthiness for multiple texts using OpenAI / Claude.

    Returns:
        List of {"label": "Yes"|"No", "raw_text": str, "raw_response": dict, "api": str}
        where raw_response is the full SDK response (usage, finish_reason,
        id, model, content_filter, reasoning trace if any, etc.).
    """
    results = []
    for text in texts:
        out = open_ai_utils.generate_full(
            CHECKWORTHY_PROMPT.format(text=text, lang=lang), model
        )
        cleaned = (
            out["text"].strip()
            .replace(".", "")
            .replace(",", "")
            .replace('"', "")
        )
        results.append({
            "label": sanitize_llm_response(cleaned),
            "raw_text": out["text"],
            "raw_response": out["raw_response"],
            "api": out["api"],
        })
    return results

    

def predict_checkworthiness_using_factiverse(claim: str) -> int:
    response = claim_detection(
        claim=row["claim"],
        access_token=access_token,
        lang=lang,
    )
    print("Factiverse response: ", response.json())
    if (
        "detectedClaims" in response.json()
        and len(response.json()["detectedClaims"]) > 0
    ):
        if response.json()["detectedClaims"][0]["score"] >= 0.8:
            return 1
        else:
            return 0
    else:
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run claim detection with specific models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mistral", "gpt54pro", "gpt52", "gpt55", "claude", "factiverse", "all", "factiverse_local"],
        default=["all"],
        help="Specify which models to run. Options: mistral, gpt54pro, gpt52, gpt55, claude, factiverse, factiverse_local, all. Default: all"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of claims to process per batch. Default: 20"
    )
    args = parser.parse_args()
    
    # Determine which models to run
    models_to_run = set(args.models)
    if "all" in models_to_run:
        models_to_run = {"mistral", "gpt54pro", "gpt52", "gpt55", "claude", "factiverse"}
    
    lang_codes = {}
    with open("src/utils/lang_codes.json", "r") as f:
        lang_codes = json.load(f)
    split = "clean_test"
    access_token = get_access_token()
    batch_size = args.batch_size
    
    # for lang in lang_codes.keys():
    for lang in ["en"]:
        open_ai_utils = OpenAIUtils()
        # logger.info("Running claim detection for %s", lang)
        groundtruth_labels = []
        predicted_labels = []
        mistral_predicted_labels = []
        gpt55_preds = []
        gpt54pro_preds = []
        gpt52_preds = []
        claude_opus_4_6_predictions = []
        
        # Print which models will be run
        print(f"Running models for {lang}: {', '.join(sorted(models_to_run))}")
        
        input_path = f"data/claim_detection/{lang}_{split}.jsonl"
        output_path = f"data/claim_detection/{lang}_{split}_pred.jsonl"
        # if not os.path.exists(input_path):
        #     continue
        # if os.path.exists(output_path):
        #     continue
        with open(
            output_path, "w"
        ) as out_json_file:
            claim_preds = []
            claims = load_json(input_path)
            
            # Process claims in batches
            for i in tqdm(range(0, len(claims), batch_size), desc=f"Processing {lang}"):
                batch = claims[i:i+batch_size]
                batch_texts = [_pick(row, _TEXT_KEYS, "text") for row in batch]
                
                try:
                    # Run selected model predictions in parallel on the batch
                    futures = {}
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        if "mistral" in models_to_run:
                            futures["mistral"] = executor.submit(
                                predict_check_worthiness_using_ollama_batch,
                                texts=batch_texts,
                                lang=lang
                            )
                        if "gpt55" in models_to_run:
                            futures["gpt55"] = executor.submit(
                                predict_claim_check_worthiness_openai_batch,
                                texts=batch_texts,
                                lang=lang,
                                open_ai_utils=open_ai_utils,
                                model="gpt-5.5"
                            )
                        if "gpt54pro" in models_to_run:
                            futures["gpt54pro"] = executor.submit(
                                predict_claim_check_worthiness_openai_batch,
                                texts=batch_texts,
                                lang=lang,
                                open_ai_utils=open_ai_utils,
                                model="gpt-5.4-pro"
                            )
                        if "gpt52" in models_to_run:
                            futures["gpt52"] = executor.submit(
                                predict_claim_check_worthiness_openai_batch,
                                texts=batch_texts,
                                lang=lang,
                                open_ai_utils=open_ai_utils,
                                model="gpt-5.2"
                            )
                        if "claude" in models_to_run:
                            futures["claude"] = executor.submit(
                                predict_claim_check_worthiness_openai_batch,
                                texts=batch_texts,
                                lang=lang,
                                open_ai_utils=open_ai_utils,
                                model="claude-opus-4-6"
                            )
                        if "factiverse_local" in models_to_run:
                            futures["factiverse_local"] = executor.submit(
                                claim_detection_factiverse_local_batch,
                                claims=batch_texts
                            )
                        
                        if "factiverse" in models_to_run:
                            futures["factiverse"] = executor.submit(
                                lambda texts: [predict_checkworthiness_using_factiverse(text) for text in texts],
                                batch_texts
                            )
                        
                        # Wait for all to complete and get results
                        results = {}
                        for model_name, future in futures.items():
                            results[model_name] = future.result()
                        
                except Exception as e:
                    import traceback
                    print(f"\n[BATCH FAILED] {type(e).__name__}: {e}")
                    traceback.print_exc()
                    continue
                
                # Process batch results
                for idx, row in enumerate(batch):
                    # Accept the ground-truth label under any of these names:
                    # checkworthy / checkworthiness / label / labels.
                    gt = _pick(row, _LABEL_KEYS, "label")
                    # Accept the text under any of: claim / sentence / text.
                    text = _pick(row, _TEXT_KEYS, "text")

                    new_row = {}
                    new_row["claim"] = text
                    new_row["checkworthy"] = gt
                    
                    # Process results for each selected model — save the
                    # sanitised prediction PLUS the model's raw_text reply
                    # AND the full raw_response (usage, finish_reason, …).
                    if "mistral" in models_to_run:
                        r = results["mistral"][idx]
                        mistral_pred = 1 if r["label"] == "Yes" else 0
                        new_row["mistral_pred"] = mistral_pred
                        new_row["mistral_raw_text"] = r["raw_text"]
                        new_row["mistral_raw_response"] = r["raw_response"]
                        mistral_predicted_labels.append(mistral_pred)

                    if "gpt55" in models_to_run:
                        r = results["gpt55"][idx]
                        gpt55_prediction_int = 1 if r["label"] == "Yes" else 0
                        new_row["gpt55_pred"] = gpt55_prediction_int
                        new_row["gpt55_raw_text"] = r["raw_text"]
                        new_row["gpt55_raw_response"] = r["raw_response"]
                        new_row["gpt55_api"] = r.get("api")
                        gpt55_preds.append(gpt55_prediction_int)

                    if "gpt54pro" in models_to_run:
                        r = results["gpt54pro"][idx]
                        gpt54pro_prediction_int = 1 if r["label"] == "Yes" else 0
                        new_row["gpt54pro_pred"] = gpt54pro_prediction_int
                        new_row["gpt54pro_raw_text"] = r["raw_text"]
                        new_row["gpt54pro_raw_response"] = r["raw_response"]
                        new_row["gpt54pro_api"] = r.get("api")
                        gpt54pro_preds.append(gpt54pro_prediction_int)

                    if "gpt52" in models_to_run:
                        r = results["gpt52"][idx]
                        gpt52_prediction_int = 1 if r["label"] == "Yes" else 0
                        new_row["gpt52_pred"] = gpt52_prediction_int
                        new_row["gpt52_raw_text"] = r["raw_text"]
                        new_row["gpt52_raw_response"] = r["raw_response"]
                        new_row["gpt52_api"] = r.get("api")
                        gpt52_preds.append(gpt52_prediction_int)

                    if "claude" in models_to_run:
                        r = results["claude"][idx]
                        claude_prediction_int = 1 if r["label"] == "Yes" else 0
                        new_row["claude_opus_4_6_pred"] = claude_prediction_int
                        new_row["claude_opus_4_6_raw_text"] = r["raw_text"]
                        new_row["claude_opus_4_6_raw_response"] = r["raw_response"]
                        new_row["claude_opus_4_6_api"] = r.get("api")
                        claude_opus_4_6_predictions.append(claude_prediction_int)

                    if "factiverse" in models_to_run:
                        # Factiverse API path still returns a plain int.
                        facti_pred = results["factiverse"][idx]
                        new_row["facti_pred"] = facti_pred
                        predicted_labels.append(facti_pred)

                    if "factiverse_local" in models_to_run:
                        r = results["factiverse_local"][idx]
                        facti_local_pred = int(r["label"])
                        new_row["facti_local_pred"] = facti_local_pred
                        new_row["facti_local_score"] = r["score"]
                        predicted_labels.append(facti_local_pred)
                    
                    groundtruth_labels.append(int(gt))
                    
                    claim_preds.append(new_row)
                # break
            json.dump(claim_preds, out_json_file, indent=4)
        
        # Calculate and print F1 scores only for models that were run
        if "factiverse" in models_to_run or "factiverse_local" in models_to_run:
            label = "Factiverse (API)" if "factiverse" in models_to_run else "Factiverse (local)"
            intent_macro_f1 = f1_score(
                groundtruth_labels, predicted_labels, average="macro"
            )
            intent_micro_f1 = f1_score(
                groundtruth_labels, predicted_labels, average="micro"
            )
            print(f"{label} [{lang}] - Macro F1: {intent_macro_f1:.4f}, Micro F1: {intent_micro_f1:.4f}")
        
        if "mistral" in models_to_run:
            intent_macro_f1 = f1_score(
                groundtruth_labels, mistral_predicted_labels, average="macro"
            )
            intent_micro_f1 = f1_score(
                groundtruth_labels, mistral_predicted_labels, average="micro"
            )
            print(f"Mistral [{lang}] - Macro F1: {intent_macro_f1:.4f}, Micro F1: {intent_micro_f1:.4f}")
        
        if "gpt55" in models_to_run:
            intent_macro_f1 = f1_score(
                groundtruth_labels, gpt55_preds, average="macro"
            )
            intent_micro_f1 = f1_score(
                groundtruth_labels, gpt55_preds, average="micro"
            )
            print(f"OpenAI GPT-5.5 [{lang}] - Macro F1: {intent_macro_f1:.4f}, Micro F1: {intent_micro_f1:.4f}")
        
        if "gpt54pro" in models_to_run:
            intent_macro_f1 = f1_score(
                groundtruth_labels, gpt54pro_preds, average="macro"
            )
            intent_micro_f1 = f1_score(
                groundtruth_labels, gpt54pro_preds, average="micro"
            )
            print(f"OpenAI GPT-5.4 Pro [{lang}] - Macro F1: {intent_macro_f1:.4f}, Micro F1: {intent_micro_f1:.4f}")

        if "gpt52" in models_to_run:
            intent_macro_f1 = f1_score(
                groundtruth_labels, gpt52_preds, average="macro"
            )
            intent_micro_f1 = f1_score(
                groundtruth_labels, gpt52_preds, average="micro"
            )
            print(f"OpenAI GPT-5.2 [{lang}] - Macro F1: {intent_macro_f1:.4f}, Micro F1: {intent_micro_f1:.4f}")
        
        if "claude" in models_to_run:
            intent_macro_f1 = f1_score(
                groundtruth_labels, claude_opus_4_6_predictions, average="macro"
            )
            intent_micro_f1 = f1_score(
                groundtruth_labels, claude_opus_4_6_predictions, average="micro"
            )
            print(f"OpenAI Claude Opus 4-6 [{lang}] - Macro F1: {intent_macro_f1:.4f}, Micro F1: {intent_micro_f1:.4f}")