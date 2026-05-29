import json
import os
from src.llm_utils.openai_utils import OpenAIUtils

from src.llm_utils.ollama import Ollama
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

def predict_stance_ollama_batch(claims: list, evidences: list, lang: str) -> list:
    """Predict stance for multiple claim-evidence pairs using Ollama.
    
    Args:
        claims: List of claims
        evidences: List of evidence texts
        lang: Language name
    
    Returns:
        List of stance predictions
    """
    results = []
    ollama = Ollama()
    for claim, evidence in zip(claims, evidences):
        prompt = IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang)
        response = ollama.generate(prompt)
        results.append(sanitize_response(response))
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
        response = open_ai_utils.generate(
            IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang),
            model
        )
        if " " in response:
            response = response.split(" ")[0]
        results.append(sanitize_response(response))
    return results


def sanitize_response(response: str) -> str:
    """Sanitize LLM response to valid labels.

    Args:
        response: LLm response text.

    Returns:
        A valid label string.
    """
    response = response.strip().replace(".", "").replace(",", "").upper()
    valid_labels = ["SUPPORTS", "REFUTES", "MIXED", "NOT_ENOUGH_INFO"]
    for label in valid_labels:
        if label in response or response in label:
            return label
    print("Invalid response label:", response)
    return random.choice(valid_labels)