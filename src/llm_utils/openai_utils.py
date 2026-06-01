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
    # Models that only expose the Responses API on Azure (no chat/completions).
    # For these we call client.responses.create() with input= instead of
    # messages=. Includes gpt-5.x and the newer "responses-only" variants.
    RESPONSES_API_MODELS = {"gpt-5.5", "gpt-5.4-pro"}

    # Cheapest valid reasoning.effort per model. gpt-5.4-pro rejects
    # 'minimal'/'low' — supported values are medium/high/xhigh only.
    # gpt-5.5 accepts the full minimal/low/medium/high range.
    REASONING_EFFORT = {
        "gpt-5.5":     "low",
        "gpt-5.4-pro": "medium",
    }

    # Model-specific API version mapping
    MODEL_API_VERSIONS = {
        "Kimi-K2.5": "2024-05-01-preview",
        "gpt-5.2": "2024-12-01-preview",
        "gpt-5.5": "2025-04-01-preview",
        "gpt-5.4-pro": "2025-04-01-preview",
        "claude-opus-4-6": "1"
        # Add more model-specific versions here as needed
        # "other-model": "2024-xx-xx-preview",
    }
    ENDPOINTS = {
        "Kimi-K2.5": "https://sponsorship-150k-resource.services.ai.azure.com/openai/v1/",
        "gpt-5.2": "https://sponsorship-150k-resource.cognitiveservices.azure.com/",
        "gpt-5.5": "https://sponsorship-150k-resource.cognitiveservices.azure.com/",
        "gpt-5.4-pro": "https://sponsorship-150k-resource.cognitiveservices.azure.com/",
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

    def generate_full(self, prompt: str, model=None, max_tokens=None,
                      system_message=None, response_format=None,
                      reasoning_summary: str = None) -> Dict[str, Any]:
        """Same as generate() but returns the full structured response.

        Returns a dict with at least:
            - "text":         the assistant's reply (str), same as generate()
            - "raw_response": the full SDK response serialised via .model_dump()
                              (Pydantic) or asdict() (dataclass) — includes
                              usage, finish_reason, id, model, etc.
            - "api":          which API surface was used: "chat_completions",
                              "responses", or "anthropic_messages"

        Useful for logging token usage, reasoning traces, content_filter
        results, and any other metadata the API surfaces.
        """
        if model is None:
            model = self._engine
        if max_tokens is None:
            max_tokens = 60
        if system_message is None:
            system_message = "You are a helpful assistant."

        client = self._get_client_for_model(model)

        def _dump(resp):
            if hasattr(resp, "model_dump"):
                return resp.model_dump()
            if hasattr(resp, "dict"):
                return resp.dict()
            return resp  # already a plain dict

        if model == "claude-opus-4-6":
            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "system": system_message,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
            response = client.messages.create(**api_params)
            if not response.content or len(response.content) == 0:
                raise ValueError("Empty response from Claude API")
            text = response.content[0].text.strip()
            return {"text": text, "raw_response": _dump(response), "api": "anthropic_messages"}

        if model in self.RESPONSES_API_MODELS:
            # gpt-5.x is a reasoning model: it spends "reasoning tokens"
            # internally before emitting output_text. With the chat-completions
            # default of 60 max_tokens, all of them get consumed by reasoning
            # and the response comes back status="incomplete" with no text.
            # Set effort="medium" so reasoning is balanced, and give a much
            # larger budget so the final answer has room to materialise.
            reasoning_budget = max(max_tokens, 200)
            effort = self.REASONING_EFFORT.get(model, "medium")
            # Raw chain-of-thought is encrypted by policy. Requesting a
            # `summary` ("auto" | "concise" | "detailed") populates
            # output[].summary with a natural-language description of the
            # model's reasoning. Costs extra output tokens.
            reasoning_cfg = {"effort": effort}
            if reasoning_summary:
                reasoning_cfg["summary"] = reasoning_summary
            api_params = {
                "model": model,
                "input": [{"role": "user", "content": prompt}],
                "instructions": system_message,
                "max_output_tokens": reasoning_budget,
                "reasoning": reasoning_cfg,
            }
            if response_format is not None:
                api_params["response_format"] = response_format
            response = client.responses.create(**api_params)
            # Fail loudly when the model returned no usable answer text —
            # e.g. status="incomplete" with reason="max_output_tokens"
            # (reasoning burned the whole budget). Silent fallback to random
            # would have masked the misconfiguration.
            status = getattr(response, "status", None)
            if status and status != "completed":
                reason = getattr(
                    getattr(response, "incomplete_details", None), "reason", "unknown"
                )
                raise ValueError(
                    f"Responses API status={status} (reason={reason}); "
                    f"no usable text. Raise max_output_tokens or lower "
                    f"reasoning.effort."
                )
            text = getattr(response, "output_text", None)
            if not text:
                # Walk output[] looking for the message item that holds text.
                for item in (response.output or []):
                    if getattr(item, "type", None) == "message":
                        content = getattr(item, "content", None) or []
                        for c in content:
                            if getattr(c, "type", None) in ("output_text", "text"):
                                text = c.text
                                break
                    if text:
                        break
            if not text:
                raise ValueError(
                    f"Responses API returned no text. output types: "
                    f"{[getattr(o, 'type', None) for o in (response.output or [])]}"
                )
            return {"text": text.strip(), "raw_response": _dump(response), "api": "responses"}

        # Chat completions fallback
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        api_params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        if response_format is not None:
            api_params["response_format"] = response_format
        response = client.chat.completions.create(**api_params)
        if not response.choices or len(response.choices) == 0:
            raise ValueError("Empty response from OpenAI API")
        text = response.choices[0].message.content.strip()
        return {"text": text, "raw_response": _dump(response), "api": "chat_completions"}

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
                    "input": [{"role": "user", "content": prompt}],
                    "system": system_message,
                    "temperature": 0.0,
                }
                logger.debug(f"API call parameters: {api_params}")
                response = client.messages.create(**api_params)
                if not response.content or len(response.content) == 0:
                    logger.error(f"Empty response from Claude API: {response}")
                    raise ValueError("Empty response from Claude API")
                return response.content[0].text.strip()
            elif model in self.RESPONSES_API_MODELS:
                # Newer Azure models (gpt-5.5, gpt-5.4-pro) only expose the
                # Responses API — call client.responses.create with input=
                # instead of messages=. System message goes in `instructions`.
                api_params = {
                    "model": model,
                    "input": [{"role": "user", "content": prompt}],
                    "instructions": system_message,
                    "max_output_tokens": max_tokens,
                }
                if response_format is not None:
                    api_params["response_format"] = response_format
                logger.debug(f"Responses API call parameters: {api_params}")
                response = client.responses.create(**api_params)
                # Convenience property added in openai>=1.50; falls back to
                # manual traversal if absent (older SDK or unusual shape).
                text = getattr(response, "output_text", None)
                if text is None:
                    if not response.output:
                        logger.error(f"Empty response.output: {response}")
                        raise ValueError("Empty response from Responses API")
                    first = response.output[0]
                    content = getattr(first, "content", None) or []
                    if not content:
                        logger.error(f"Empty response.output[0].content: {response}")
                        raise ValueError("Empty content from Responses API")
                    text = content[0].text
                return text.strip()
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
