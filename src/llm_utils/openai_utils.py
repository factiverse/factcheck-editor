"""Module containing the OpenAI Summary Generator class."""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

# client = OpenAI()

from dotenv import load_dotenv

# from openai import AzureOpenAI


class OpenAIUtils:
    def __init__(self) -> None:
        """Initializes the OpenAI API and Elasticsearch client."""
        load_dotenv()
        # openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        # openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g., https://YOUR_RESOURCE_NAME.openai.azure.com/
        # openai.api_type = "azure"
        # openai.api_version = "2024-05-13"  # Ensure this is the correct version
        self._engine = os.getenv("AZURE_OPENAI_MODEL_ID")
        self._client = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-05-13",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        # Logging for debug purposes
        # logging.basicConfig(level=logging.DEBUG)
        # logging.debug(f"API Key: {openai.api_key}")
        # logging.debug(f"API Base: {openai.api_base}")
        # logging.debug(f"API Type: {openai.api_type}")
        # logging.debug(f"API Version: {openai.api_version}")
        # logging.debug(f"Engine: {self._engine}")

    def generate(self, prompt: str, model=None) -> str:
        """Generates a response from the OpenAI API for the given prompt.

        Args:
            prompt: The prompt to send to the API.
            model: The model to use for generating the response. Default is None.

        Returns:
            The API's response as a string.
        """
        if model is None:
            model = self._engine
        try:
            response = self._client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_MODEL_ID"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

if __name__ == "__main__":
    openai_utils = OpenAIUtils()
    prompt = "What is the capital of France?"
    try:
        response = openai_utils.generate(prompt)
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")
