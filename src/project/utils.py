import os
import sys

from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig

def build_client_config() -> ModelClientConfig:
    """Create a client configuration for openjiuwen.

    Reads API credentials from environment variables:
    - LLM_API_KEY: API key (must start with 'sk-')
    - LLM_BASE_URL: Base URL for the API
    - MODEL_PROVIDER: Provider name (defaults to 'OpenAI')

    Returns:
        A ``ModelClientConfig`` instance.

    Raises:
        RuntimeError: If LLM_BASE_URL is not set.
        SystemExit: If API key is missing or invalid.
    """
    provider = os.getenv("MODEL_PROVIDER", "OpenAI")
    api_key = os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_BASE_URL")

    if not api_key:
        sys.exit("API_KEY is not set. Export a LiteLLM virtual key (starts with 'sk-').")
    if not api_key.startswith("sk-"):
        sys.exit(
            "API_KEY must be a LiteLLM virtual key starting with 'sk-'. "
            "The invite code is not an API key."
        )
    if not api_base:
        raise RuntimeError("LLM_BASE_URL is not set")

    return ModelClientConfig(
        client_provider=provider,
        api_key=api_key,
        api_base=api_base,
        verify_ssl=False,
    )


def build_model_config(model_name: str) -> ModelRequestConfig:
    """Create a model request configuration for openjiuwen.

    Args:
        model_name: Name of the model to use.

    Returns:
        A ``ModelRequestConfig`` instance for the specified model.
    """
    return ModelRequestConfig(model=model_name)