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

    def __init__(self, config_path: str = "src/llm_utils/mistral.yaml"):
        """Initializes the Ollama client and loads necessary configurations."""
        load_dotenv()
        self._ollama_client = Client(host=os.environ["OLLAMA_HOST"], timeout=20)
        self._config_path = config_path
        self._config = self._load_config()
        # print("Ollama config:", self._config)
        self._stream = self._config.get("stream", False)
        self._model_name = self._config.get("model", "mistral")
        self._llm_options = self._get_llm_config()
        # print("Ollama LLM options:", self._llm_options)

    def generate(self, prompt: str, model: str = None) -> str:
        """Generate text using Ollama LLM for the given prompt.

        Args:
            prompt: Prompt for the LLM.
            model: Optional model name to override the default config model.

        Returns:
            Response text from an Ollama LLM.
        """
        return self.generate_full(prompt, model)["response"].strip()

    def generate_full(self, prompt: str, model: str = None) -> Dict[str, Any]:
        """Return the full Ollama response dict (response, model, timings, etc.).

        Useful when callers want to log token counts, eval_count,
        total_duration, etc. alongside the text — without re-running.
        """
        model_name = model or self._model_name
        response = self._ollama_client.generate(
            model=model_name,
            prompt=prompt,
            options=self._llm_options,
            stream=self._stream,
        )
        # ollama-python returns either a dict or a GenerateResponse pydantic
        # model depending on version; normalise to dict so downstream
        # json.dump works cleanly.
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        elif hasattr(response, "dict"):
            response = response.dict()
        return response

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
        opts = self._config.get("options", {}) or {}
        # Try to construct Options with kwargs; if the library's Options
        # takes no args, fall back to no-arg construction. If that also
        # fails, return the raw dict so Client.generate can accept it.
        try:
            return Options(**opts)
        except TypeError:
            try:
                return Options()
            except TypeError:
                return opts

