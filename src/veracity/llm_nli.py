import json
import os
from src.llm_utils.openai_utils import OpenAIUtils

from src.llm_utils.ollama import Ollama
from src.utils.utils import load_lang_codes
from src.prompts.prompts import IDENTIFY_STANCE_PROMPT
import random


def predict_stance_ollama(claim, evidence, lang):
    prompt = IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang)
    # print(prompt)
    response = Ollama().generate(prompt)
    return sanitize_response(response)

def predict_stance_openai(claim, lang, evidence, model=None):
    open_ai_utils = OpenAIUtils()
    response = open_ai_utils.generate(IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang))
    if " " in response:
        response = response.split(" ")[0]
    return sanitize_response(response)

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