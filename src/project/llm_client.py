from typing import Any

from openjiuwen.core.foundation.llm import (
    ModelClientConfig,
    ModelRequestConfig,
)
from openjiuwen.core.foundation.tool import LocalFunction

# Try to import Model - path may differ depending on openjiuwen version
try:
    from openjiuwen.core.foundation.llm import Model
except ImportError:
    from openjiuwen.core.foundation.llm.model import Model

from project.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE


class LLMClient:
    """Thin wrapper around openjiuwen native Model invocation.

    This class is intentionally minimal: it forwards a list of messages
    and a list of tools to the configured LLM via the native ``Model``
    API. Higher-level logic such as conversation history management
    and tool dispatch lives in ``TaskSolver``.
    """

    def __init__(
        self,
        client_config: ModelClientConfig,
        model_config: ModelRequestConfig,
    ):
        """Initialize the client with model and provider configuration.

        Args:
            client_config: Provider-level configuration (API key, base URL).
            model_config: Model-level configuration (model name, etc.).
        """
        self.client_config = client_config
        self.model_config = model_config

    async def call(
        self,
        messages: list,
        tools: list[LocalFunction],
    ) -> Any:
        """Execute a single LLM call with native tool support.

        Args:
            messages: Conversation history as a list of message dicts
                (with ``role``, ``content``, and optionally ``tool_calls``).
            tools: List of ``LocalFunction`` tools available to the model.

        Returns:
            The ``AssistantMessage`` returned by the model. The caller is
            responsible for inspecting ``content`` and ``tool_calls``.
        """
        model = Model(
            model_client_config=self.client_config,
            model_config=self.model_config,
        )

        # Extract ToolInfo from each LocalFunction's card
        tool_infos = [t.card.tool_info() for t in tools]

        response = await model.invoke(
            messages=messages,
            tools=tool_infos,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return response


def extract_usage(response: Any) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from an LLM response.

    Returns ``(0, 0)`` if usage metadata is not available.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    return (
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )