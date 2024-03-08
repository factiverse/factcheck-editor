import json
import os
from code.llm_utils.openai_utils import OpenAIUtils

from code.llm_utils.ollama import Ollama
from code.utils.utils import load_lang_codes
from code.prompts.prompts import IDENTIFY_STANCE_PROMPT
import random


def predict_stance_ollama(claim, evidence, lang, ollama):
    prompt = IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang)
    # print(prompt)
    response = ollama.generate(prompt)
    return sanitize_response(response)

def predict_stance_openai(claim, lang, evidence, model=None):
    open_ai_utils = OpenAIUtils()
    response = open_ai_utils.generate(IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence, lang=lang, model=model))
    if " " in response:
        response = response.split(" ")[0]
    return sanitize_response(response)

def sanitize_response(response: str) -> str:
    """Sanitize LLM response to valid labels.

    Args:
        response: LLm response text.

    Returns:
        True or False. If the response is not True or False, pick a random label.
    """
    response = response.strip().replace(".", "").replace(",", "")
    if response == "A":
        label = "True"
    elif response == "B":
        label = "False"
    else:
        label = "True" if random.random() >= 0.5 else "False"
    return label

