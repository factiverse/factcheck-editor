"""Module containing the OpenAI Summary Generator class."""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

# client = OpenAI()

from dotenv import load_dotenv
from anthropic import AnthropicFoundry

# from openai import AzureOpenAI

logger = logging.getLogger(__name__)

class OpenAIUtils:
    # Model-specific API version mapping
    MODEL_API_VERSIONS = {
        "Kimi-K2.5": "2024-05-01-preview",
        "gpt-5.2": "2024-12-01-preview",
        "claude-opus-4-6": "1"
        # Add more model-specific versions here as needed
        # "other-model": "2024-xx-xx-preview",
    }
    ENDPOINTS = {
        "Kimi-K2.5": "https://sponsorship-150k-resource.services.ai.azure.com/openai/v1/",
        "gpt-5.2": "https://sponsorship-150k-resource.cognitiveservices.azure.com/",
        "claude-opus-4-6": "https://sponsorship-150k-resource.services.ai.azure.com/anthropic"
        # Add more model-specific endpoints here as needed
        # "other-model": "https://your-endpoint.openai.azure.com/",
    }
    
    def __init__(self) -> None:
        """Initializes the OpenAI API and Elasticsearch client."""
        load_dotenv()
        # deployment id / model name to use (from env)
        self._engine = os.getenv("AZURE_OPENAI_MODEL_ID")
        self._api_key = os.getenv("AZURE_OPENAI_KEY")
        self._azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self._default_api_version = OpenAIUtils.MODEL_API_VERSIONS.get(self._engine, "2024-05-01-preview")
        
        # Create default client
        self._client = AzureOpenAI(
            api_key=self._api_key,
            azure_endpoint=self._azure_endpoint,
            api_version=self._default_api_version,
        )
        
        # Cache for model-specific clients
        self._model_clients = {}
        # Logging for debug purposes
        # logging.basicConfig(level=logging.DEBUG)
        # logging.debug(f"API Key: {openai.api_key}")
        # logging.debug(f"API Base: {openai.api_base}")
        # logging.debug(f"API Type: {openai.api_type}")
        # logging.debug(f"API Version: {openai.api_version}")
        # logging.debug(f"Engine: {self._engine}")
    
    def _get_client_for_model(self, model: str) -> AzureOpenAI:
        """Get the appropriate Azure OpenAI client for the given model.
        
        Args:
            model: The model/deployment name.
            
        Returns:
            AzureOpenAI client configured for the model's API version.
        """
        # Check if model requires a specific API version
        if model in self.MODEL_API_VERSIONS:
            api_version = self.MODEL_API_VERSIONS[model]
            endpoint = self.ENDPOINTS.get(model, self._azure_endpoint)
            # Return cached client if available
            if model in self._model_clients:
                return self._model_clients[model]
            
            # Create and cache new client for this model
            if model == "claude-opus-4-6":
                client = AnthropicFoundry(
                    api_key=self._api_key,
                    base_url=endpoint,
                )
            else:
                client = AzureOpenAI(
                    api_key=self._api_key,
                    azure_endpoint=endpoint,
                    api_version=api_version,
                )
            self._model_clients[model] = client
            return client
        
        # Use default client
        return self._client

    def generate(self, prompt: str, model=None, max_tokens=None, system_message=None, response_format=None) -> str:
        """Generates a response from the OpenAI API for the given prompt.

        Args:
            prompt: The prompt to send to the API.
            model: The deployment/model to use for generating the response. If None, uses env AZURE_OPENAI_MODEL_ID.
            max_tokens: Maximum tokens for completion. If None, uses 60.
            system_message: Custom system message. If None, uses default helpful assistant message.
            response_format: Response format type. Can be {"type": "json_object"} for JSON mode.

        Returns:
            The API's response as a string.
        """
        if model is None:
            model = self._engine
        if max_tokens is None:
            max_tokens = 60
        if system_message is None:
            system_message = "You are a helpful assistant."
            
        try:
            # Get the appropriate client for this model
            client = self._get_client_for_model(model)
            api_version = self.MODEL_API_VERSIONS.get(model, self._default_api_version)
            logger.debug(f"Using API version: {api_version} for model: {model}")
            
            if model == "claude-opus-4-6":
                # For Anthropic Claude, the API call structure is different
                api_params = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                    "system": system_message,
                    "temperature": 0.0,
                }
                logger.debug(f"API call parameters: {api_params}")
                response = client.messages.create(**api_params)
                if not response.content or len(response.content) == 0:
                    logger.error(f"Empty response from Claude API: {response}")
                    raise ValueError("Empty response from Claude API")
                return response.content[0].text.strip()
            else:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
                
                # Build the API call parameters
                api_params = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
                
                # Add response_format if specified
                if response_format is not None:
                    api_params["response_format"] = response_format
                
                logger.debug(f"API call parameters: {api_params}")
                response = client.chat.completions.create(**api_params)
                if not response.choices or len(response.choices) == 0:
                    logger.error(f"Empty response from OpenAI API: {response}")
                    raise ValueError("Empty response from OpenAI API")
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

if __name__ == "__main__":
    openai_utils = OpenAIUtils()
    prompt = "What is the capital of France?"
    try:
        response = openai_utils.generate(prompt)
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")
