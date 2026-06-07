import json
import os
import re
import time
from src.llm_utils.openai_utils import OpenAIUtils

from src.llm_utils.ollama import Ollama
try:
    from src.llm_utils.openrouter import OpenRouterUtils
except ModuleNotFoundError:
    # openrouter is optional; if its module isn't present, only the
    # openrouter-based predictors are unavailable (they raise on use).
    OpenRouterUtils = None
from src.utils.utils import load_lang_codes
from src.prompts.prompts import IDENTIFY_STANCE_PROMPT
import random


def predict_stance_ollama(claim, evidence, lang, model="mistral"):
    prompt = IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang)
    # print(prompt)
    ollama = Ollama()
    response = ollama.generate(prompt, model=model)
    return sanitize_response(response)

def predict_stance_openai(claim, lang, evidence, model=None):
    open_ai_utils = OpenAIUtils()
    response = open_ai_utils.generate(
        IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang),
        model
    )
    if " " in response:
        response = response.split(" ")[0]
    return sanitize_response(response)

def predict_stance_ollama_batch(claims: list, evidences: list, lang: str, model: str = "mistral") -> list:
    """Predict stance for multiple claim-evidence pairs using Ollama.

    Args:
        claims: List of claims
        evidences: List of evidence texts
        lang: Language name
        model: Ollama model tag (e.g. "mistral", "qwen3:8b")

    Returns:
        List of stance predictions
    """
    results = []
    ollama = Ollama()
    for claim, evidence in zip(claims, evidences):
        prompt = IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang)
        try:
            response = ollama.generate(prompt, model=model, think=False)
            results.append(sanitize_response(response) if response else "ERROR")
        except Exception as e:
            print(f"  [warn] ollama({model}) item failed: {type(e).__name__}: {str(e)[:100]}")
            results.append("ERROR")
    return results


def predict_stance_openai_batch(
    claims: list, evidences: list, lang: str, open_ai_utils: OpenAIUtils, model=None
) -> list:
    """Predict stance for multiple claim-evidence pairs using OpenAI.

    Args:
        claims: List of claims
        evidences: List of evidence texts
        lang: Language name
        open_ai_utils: OpenAIUtils object
        model: Model name (optional)

    Returns:
        List of stance predictions
    """
    results = []
    for claim, evidence in zip(claims, evidences):
        prompt = IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang)
        label = "ERROR"
        # Azure/Foundry occasionally returns an empty payload under concurrent
        # load (throttling). Retry a few times with backoff before giving up;
        # leftover ERRORs are retried again on the next run.
        for attempt in range(3):
            try:
                response = open_ai_utils.generate(prompt, model)
                if response:
                    if " " in response:
                        response = response.split(" ")[0]
                    label = sanitize_response(response)
                    break
            except Exception as e:
                if attempt == 2:
                    print(f"  [warn] {model} item failed after retries: {type(e).__name__}: {str(e)[:100]}")
            time.sleep(1.0 * (attempt + 1))
        results.append(label)
    return results


def predict_stance_openrouter(claim: str, evidence: str, lang: str, model: str = "google/gemma-4-31b-it:free") -> str:
    """Predict stance using OpenRouter.

    Args:
        claim: Claim text
        evidence: Evidence text
        lang: Language name
        model: OpenRouter model to use (e.g., "google/gemma-4-31b-it:free")

    Returns:
        Stance prediction
    """
    if OpenRouterUtils is None:
        raise ModuleNotFoundError(
            "src.llm_utils.openrouter is not available — cannot run openrouter predictions."
        )
    openrouter = OpenRouterUtils(model=model)
    prompt = IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang)
    response = openrouter.generate(prompt)
    return sanitize_response(response)


def predict_stance_openrouter_batch(
    claims: list, evidences: list, lang: str, model: str = "google/gemma-4-31b-it:free"
) -> list:
    """Predict stance for multiple claim-evidence pairs using OpenRouter.

    Args:
        claims: List of claims
        evidences: List of evidence texts
        lang: Language name
        model: OpenRouter model to use (e.g., "google/gemma-4-31b-it:free")

    Returns:
        List of stance predictions
    """
    results = []
    if OpenRouterUtils is None:
        raise ModuleNotFoundError(
            "src.llm_utils.openrouter is not available — cannot run openrouter predictions."
        )
    openrouter = OpenRouterUtils(model=model)
    for claim, evidence in zip(claims, evidences):
        prompt = IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang)
        try:
            response = openrouter.generate(prompt)
            results.append(sanitize_response(response) if response else "ERROR")
        except Exception as e:
            print(f"  [warn] openrouter item failed: {type(e).__name__}: {str(e)[:100]}")
            results.append("ERROR")
    return results


def sanitize_response(response: str) -> str:
    """Sanitize LLM response to valid labels.

    Args:
        response: LLm response text.

    Returns:
        A valid label string.
    """
    # Strip <think>...</think> traces (e.g. Qwen3 thinking mode) before parsing
    # so the label isn't picked up from the reasoning text.
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = response.strip().replace(".", "").replace(",", "").upper()
    valid_labels = ["SUPPORTS", "REFUTES", "MIXED"]
    for label in valid_labels:
        if label in response or response in label:
            return label
    print("Invalid response label:", response)
    return random.choice(valid_labels)