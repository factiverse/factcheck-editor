"""This module contains the Mistral class, which is used to generate questions 
from a given claim.

Usage:
    ollama = Ollama(config_path="code/llm_utils/mistral.yaml")
    ollama.generate("Write a story about a happy llama.")
    
"""

from ollama import Client, Options
from typing import List, Dict, Any
import json
import yaml
import os
from dotenv import load_dotenv


class Ollama:
    """A class for generating questions and interacting with the Ollama API."""

    def __init__(self, config_path: str = "code/llm_utils/mistral.yaml"):
        """Initializes the Ollama client and loads necessary configurations."""
        load_dotenv()
        self._ollama_client = Client(host=os.environ["OLLAMA_HOST"], timeout=120)
        self._config_path = config_path
        self._config = self._load_config()
        self._stream = self._config.get("stream", False)
        self._model_name = self._config.get("model", "mistral")
        self._llm_options = self._get_llm_config()

    def generate(self, prompt: str) -> str:
        """Generate text using Ollama LLM for the given prompt.

        Args:
            prompt: Prompt for the LLM.

        Returns:
            Response text from an Ollama LLM.
        """
        response = self._ollama_client.generate(
            model=self._model_name,
            prompt=prompt,
            options=self._llm_options,
            stream=self._stream,
        )
        response_text = response["response"].strip()
        return response_text

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from a YAML file.

        Raises:
            FileNotFoundError: If the config file is not found.

        Returns:
            A dictionary with configuration values.
        """
        if not os.path.isfile(self._config_path):
            raise FileNotFoundError(
                f"Config file {self._config_path} not found."
            )
        with open(self._config_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data

    def _get_llm_config(self) -> Options:
        """Extracts and returns the LLM (language learning model) configuration.

        Returns:
            An Options object with the LLM configuration.
        """
        return Options(self._config.get("options", {}))

