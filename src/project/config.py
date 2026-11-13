import os
import sys
from typing import Optional

from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig


class Config:
    """Central configuration holder.

    Reads environment variables and exposes model routing and defaults.
    Created once in ``main`` and passed to all components that need it.
    """

    MODEL_ROUTING = {
        "coding": "qwen3.8-27b",
        "math": "qwen3.8-27b-noreason",
        "data_analysis": "qwen3.6-35b-a3b-noreason",
        "search": "qwen3.6-35b-a3b",
    }

    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 4096

    OUTPUT_TOKEN_WEIGHT = 10

    def __init__(self) -> None:
        """Build the provider-level client configuration from env vars."""
        self.client_config = self._build_client_config()

        # Read global token budget from environment, default to 100,000
        self._budget = int(os.environ.get("BUDGET", 100_000))

    def _build_client_config(self) -> ModelClientConfig:
        provider = os.getenv("MODEL_PROVIDER", "OpenAI")
        api_key = os.environ.get("LLM_API_KEY")
        api_base = os.environ.get("LLM_BASE_URL")

        if not api_key:
            sys.exit(
                "API_KEY is not set. Export a LiteLLM virtual key (starts with 'sk-')."
            )
        if not api_key.startswith("sk-"):
            sys.exit(
                "API_KEY must be a LiteLLM virtual key starting with 'sk-'. "
                "The invite code is not an API key."
            )
        if not api_base:
            raise RuntimeError("BASE_URL is not set")

        return ModelClientConfig(
            client_provider=provider,
            api_key=api_key,
            api_base=api_base,
            verify_ssl=False,
        )

    @property
    def budget(self) -> int:
        """Return the global token budget."""
        return self._budget

    def get_model_name(self, task_type: str) -> Optional[str]:
        """Return the model name configured for a given task type."""
        return self.MODEL_ROUTING.get(task_type)

    def get_model_config(self, model_name: str) -> ModelRequestConfig:
        """Create a model request configuration for the given model name."""
        return ModelRequestConfig(model=model_name)
    
    def normalize_tokens(self, input_tokens: int, output_tokens: int) -> int:
        """Привести пару (input, output) к единой шкале «входных» токенов.

        1 выходной токен = OUTPUT_TOKEN_WEIGHT входных.
        """
        return input_tokens + output_tokens * self.OUTPUT_TOKEN_WEIGHT