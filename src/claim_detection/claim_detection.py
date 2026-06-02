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
from src.llm_utils.openrouter import OpenRouterUtils
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
    "claim_xlmr/"
)

# Lazy singleton — the XLM-R-XL local Factiverse model is ~7 GB on GPU,
# so we only construct it the first time `factiverse_local` is actually
# requested. Runs with --models gpt55 / claude / ollama skip the load
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


def predict_check_worthiness_using_openrouter(text: str, lang: str, model: str = "google/gemma-4-31b-it:free") -> str:
    """Predict check worthiness using OpenRouter.

    Args:
        text: Sentence to predict check-worthiness.
        lang: Language code.
        model: OpenRouter model to use (e.g., "google/gemma-4-31b-it:free" or "openrouter/google/gemma-4-31b-it:free")

    Returns:
        Return Yes if the sentence is check-worthy, else No.
    """
    openrouter = OpenRouterUtils(model=model)
    prompt = CHECKWORTHY_PROMPT.format(text=text, lang=lang)
    response = openrouter.generate(prompt)
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

    Returns Yes or No. Strips Qwen3-style <think>...</think> blocks
    before parsing so thinking-mode output doesn't fall through to the
    random fallback.
    """
    import re
    # Strip <think>...</think> blocks (Qwen3 thinking mode, possibly multi-line)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    response_words = response.split()
    llm_prediction = response_words[0].strip(".,:\"'").capitalize() if response_words else ""
    if llm_prediction == "Yes":
        return "Yes"
    elif llm_prediction == "No":
        return "No"
    else:
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

    ``score`` is the probability of the positive (check-worthy) class —
    handy for downstream threshold sweeps. The underlying ``predict``
    returns per-class probability vectors (shape ``[batch, num_labels]``),
    so we pick column 1 for binary check-worthiness.
    """
    preds, scores = get_claim_detection_model().predict(tuple(claims))
    preds = preds.tolist() if hasattr(preds, "tolist") else list(preds)
    scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)

    def _pos_score(s):
        # ``s`` is either a flat scalar or a per-class probability vector.
        # For binary check-worthiness we always want P(class 1).
        if isinstance(s, (list, tuple)):
            return float(s[1] if len(s) > 1 else s[0])
        return float(s)

    return [
        {"score": _pos_score(s), "label": int(p)}
        for p, s in zip(preds, scores)
    ]


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


def predict_check_worthiness_using_openrouter_batch(texts: list, lang: str, model: str = "google/gemma-4-31b-it:free") -> list:
    """Predict check worthiness for multiple texts using OpenRouter.

    Args:
        texts: List of texts to predict
        lang: Language code
        model: OpenRouter model to use (e.g., "google/gemma-4-31b-it:free")

    Returns:
        List of {"label": "Yes"|"No", "raw_text": str, "raw_response": dict}
        where raw_response is the full OpenRouter API response.
    """
    openrouter = OpenRouterUtils(model=model)
    results = []
    for text in texts:
        prompt = CHECKWORTHY_PROMPT.format(text=text, lang=lang)
        response_data = openrouter.generate_full(prompt)
        raw_text = response_data.get("text", "").strip()
        results.append({
            "label": sanitize_llm_response(raw_text),
            "raw_text": raw_text,
            "raw_response": response_data.get("raw_response", {}),
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
        choices=["ollama", "gpt54pro", "gpt52", "gpt55", "claude", "factiverse", "all", "factiverse_local", "openrouter"],
        default=["all"],
        help="Specify which models to run. Options: ollama, gpt54pro, gpt52, gpt55, claude, factiverse, factiverse_local, openrouter, all. Default: all"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of claims to process per batch. Default: 20"
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="google/gemma-4-31b-it:free",
        help="OpenRouter model to use. Default: google/gemma-4-31b-it:free"
    )
    args = parser.parse_args()

    # Determine which models to run
    models_to_run = set(args.models)
    if "all" in models_to_run:
        models_to_run = {"ollama", "gpt54pro", "gpt52", "gpt55", "claude", "factiverse", "openrouter"}

    print(f"\n{'='*60}")
    print(f"Running ONLY these models: {', '.join(sorted(models_to_run))}")
    print(f"{'='*60}\n")

    lang_codes = {}
    with open("src/utils/lang_codes.json", "r") as f:
        lang_codes = json.load(f)
    split = "test"
    access_token = get_access_token()
    batch_size = args.batch_size

    for lang in lang_codes.keys():
    # for lang in ["en"]:
        if lang == "en":
            continue
        open_ai_utils = OpenAIUtils()
        groundtruth_labels = []
        predicted_labels = []
        ollama_predicted_labels = []
        gpt55_preds = []
        gpt54pro_preds = []
        gpt52_preds = []
        claude_opus_4_6_predictions = []
        openrouter_preds = []

        # Print which models will be run
        print(f"Running models for {lang}: {', '.join(sorted(models_to_run))}")

        input_path = f"data/claim_detection/{lang}_{split}.jsonl"
        if not os.path.exists(input_path):
            print(f"Input file not found for {lang}: {input_path}. Skipping.")
            continue
        output_path = f"data/claim_detection/{lang}_{split}_pred.jsonl"

        claims = load_json(input_path)

        # Load existing predictions if the output file already exists so we
        # only add/overwrite the prediction columns for the models being run
        # now — all other model columns are preserved. Both JSONL and JSON
        # array formats are accepted for backward compat.
        existing_by_claim = {}
        if os.path.exists(output_path):
            try:
                rows_existing = []
                with open(output_path, encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows_existing.append(json.loads(line))
                        except json.JSONDecodeError:
                            if line_no == 1 and line.startswith("["):
                                # Legacy JSON-array file: load whole file
                                f.seek(0)
                                rows_existing = json.load(f)
                                break
                            else:
                                print(f"  [warn] Skipping malformed line {line_no} in {output_path}")
                existing_by_claim = {row.get("claim", ""): row for row in rows_existing}
                print(f"  Loaded {len(existing_by_claim)} existing rows from {output_path}")
            except Exception as e:
                print(f"  [warn] Could not load {output_path}: {e} — starting fresh")

        # Map: model key → the pred column it writes.
        # Used for lang-level and row-level skip decisions.
        _model_pred_col = {
            "mistral":          "mistral_pred",
            "gpt55":            "gpt55_pred",
            "gpt54pro":         "gpt54pro_pred",
            "gpt52":            "gpt52_pred",
            "claude":           "claude_opus_4_6_pred",
            "factiverse":       "facti_pred",
            "factiverse_local": "facti_local_pred",
            "ollama":           "ollama_pred",
            "openrouter":       "openrouter_pred",
        }
        needed_cols = [_model_pred_col[m] for m in models_to_run if m in _model_pred_col]

        # ── Lang-level skip ───────────────────────────────────────────────
        # If every row already has every needed col populated, skip the lang.
        if existing_by_claim and needed_cols and claims:
            all_done = all(
                all(col in existing_by_claim.get(_pick(r, _TEXT_KEYS, "text"), {})
                    for col in needed_cols)
                for r in claims
            )
            if all_done:
                print(
                    f"  [{lang}] All {len(claims)} rows already have "
                    f"{needed_cols} — skipping lang."
                )
                continue

        claim_preds = []

        # Process claims in batches
        for i in tqdm(range(0, len(claims), batch_size), desc=f"Processing {lang}"):
            full_batch = claims[i:i+batch_size]

            # ── Row-level skip ────────────────────────────────────────────
            # Rows with every needed col go straight to output; only rows
            # missing at least one col are sent to the model API.
            needs_work, already_done = [], []
            for row in full_batch:
                text = _pick(row, _TEXT_KEYS, "text")
                existing = existing_by_claim.get(text, {})
                if needed_cols and all(col in existing for col in needed_cols):
                    already_done.append(row)
                else:
                    needs_work.append(row)

            seen_in_batch = set()
            for row in already_done:
                text = _pick(row, _TEXT_KEYS, "text")
                if text in seen_in_batch:
                    continue
                seen_in_batch.add(text)
                gt = _pick(row, _LABEL_KEYS, "label")
                merged = dict(existing_by_claim.get(text, {}))
                merged.setdefault("claim", text)
                merged.setdefault("checkworthy", gt)
                groundtruth_labels.append(int(gt))
                claim_preds.append(merged)

            batch = needs_work
            if not batch:
                continue   # nothing new in this chunk

            batch_texts = [_pick(row, _TEXT_KEYS, "text") for row in batch]

            try:
                # Run selected model predictions in parallel on the batch
                futures = {}
                with ThreadPoolExecutor(max_workers=4) as executor:
                    if "ollama" in models_to_run:
                        futures["ollama"] = executor.submit(
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
                    if "openrouter" in models_to_run:
                        futures["openrouter"] = executor.submit(
                            predict_check_worthiness_using_openrouter_batch,
                            texts=batch_texts,
                            lang=lang,
                            model=args.openrouter_model
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

            # Process batch results — always start from the existing row so
            # columns written by previous runs (e.g. gpt52_pred from an
            # earlier --models gpt52 run) are preserved even when the current
            # run is for a different model.
            for idx, row in enumerate(batch):
                gt = _pick(row, _LABEL_KEYS, "label")
                text = _pick(row, _TEXT_KEYS, "text")

                # Seed from any existing data for this claim; add base fields.
                new_row = dict(existing_by_claim.get(text, {}))
                new_row["claim"] = text
                new_row["checkworthy"] = gt

                if "ollama" in models_to_run:
                    r = results["ollama"][idx]
                    ollama_pred = 1 if r["label"] == "Yes" else 0
                    new_row["ollama_pred"] = ollama_pred
                    new_row["ollama_raw_text"] = r["raw_text"]
                    new_row["ollama_raw_response"] = r["raw_response"]
                    ollama_predicted_labels.append(ollama_pred)

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
                    facti_pred = results["factiverse"][idx]
                    new_row["facti_pred"] = facti_pred
                    predicted_labels.append(facti_pred)

                if "factiverse_local" in models_to_run:
                    r = results["factiverse_local"][idx]
                    facti_local_pred = int(r["label"])
                    new_row["facti_local_pred"] = facti_local_pred
                    new_row["facti_local_score"] = r["score"]
                    predicted_labels.append(facti_local_pred)

                if "openrouter" in models_to_run:
                    r = results["openrouter"][idx]
                    openrouter_prediction_int = 1 if r["label"] == "Yes" else 0
                    new_row["openrouter_pred"] = openrouter_prediction_int
                    new_row["openrouter_raw_text"] = r["raw_text"]
                    new_row["openrouter_raw_response"] = r["raw_response"]
                    openrouter_preds.append(openrouter_prediction_int)

                groundtruth_labels.append(int(gt))

                # Merge with existing row if present, else start fresh
                if text in existing_by_claim:
                    merged = existing_by_claim[text]
                    merged.update(new_row)
                    claim_preds.append(merged)
                else:
                    claim_preds.append(new_row)

        # Write merged predictions as JSONL — one row per line.
        # Guard: refuse to write an empty file so a crash or unexpected
        # empty batch can never truncate a previously-populated pred file.
        if not claim_preds:
            print(f"  [{lang}] claim_preds is empty — skipping write to protect {output_path}")
        else:
            # Write atomically via a temp file so a mid-write crash doesn't
            # leave the output file in a half-written state.
            import shutil
            tmp_path = output_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as out_json_file:
                for row in claim_preds:
                    out_json_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            shutil.move(tmp_path, output_path)

        # Per-model F1 reporting — macro, micro, and per-class
        def _print_f1(name: str, gt: list, preds: list):
            if not gt or not preds or len(gt) != len(preds):
                print(f"{name} [{lang}] - no predictions to evaluate")
                return
            macro = f1_score(gt, preds, average="macro", zero_division=0)
            micro = f1_score(gt, preds, average="micro", zero_division=0)
            per_class = f1_score(gt, preds, average=None, labels=[0, 1], zero_division=0)
            # Count confusion matrix cells for TP/FP/TN/FN
            tp = sum(1 for g, p in zip(gt, preds) if g == 1 and p == 1)
            fp = sum(1 for g, p in zip(gt, preds) if g == 0 and p == 1)
            tn = sum(1 for g, p in zip(gt, preds) if g == 0 and p == 0)
            fn = sum(1 for g, p in zip(gt, preds) if g == 1 and p == 0)
            print(
                f"{name} [{lang}] "
                f"Macro={macro:.4f}  Micro={micro:.4f}  "
                f"F1(no-claim)={per_class[0]:.4f}  F1(claim)={per_class[1]:.4f}  "
                f"TP={tp} FP={fp} TN={tn} FN={fn}  n={len(gt)}"
            )

        if "factiverse" in models_to_run or "factiverse_local" in models_to_run:
            label = "Factiverse (API)" if "factiverse" in models_to_run else "Factiverse (local)"
            _print_f1(label, groundtruth_labels, predicted_labels)

        if "ollama" in models_to_run:
            _print_f1("ollama", groundtruth_labels, ollama_predicted_labels)

        if "gpt55" in models_to_run:
            _print_f1("OpenAI GPT-5.5", groundtruth_labels, gpt55_preds)

        if "gpt54pro" in models_to_run:
            _print_f1("OpenAI GPT-5.4 Pro", groundtruth_labels, gpt54pro_preds)

        if "gpt52" in models_to_run:
            _print_f1("OpenAI GPT-5.2", groundtruth_labels, gpt52_preds)

        if "claude" in models_to_run:
            _print_f1("Claude Opus 4-6", groundtruth_labels, claude_opus_4_6_predictions)

        if "openrouter" in models_to_run:
            _print_f1("Gemini 3.5 Flash", groundtruth_labels, openrouter_preds)