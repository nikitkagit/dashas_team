from typing import Any, Optional

from openjiuwen.core.foundation.tool import LocalFunction

# Try to import Model - path may differ depending on openjiuwen version
try:
    from openjiuwen.core.foundation.llm import Model
except ImportError:
    from openjiuwen.core.foundation.llm.model import Model

from project.config import Config


class LLMClient:
    """Thin wrapper around openjiuwen native Model invocation.

    This class is intentionally minimal: it forwards a list of messages
    and a list of tools to the configured LLM via the native ``Model``
    API. Higher-level logic such as conversation history management
    and tool dispatch lives in ``TaskSolver``.
    """

    def __init__(self, config: Config, model_name: str):
        """Initialize the client with shared configuration and a model name.

        Args:
            config: Shared ``Config`` instance providing client config and defaults.
            model_name: Name of the model to use for this client.
        """
        self.config = config
        self.client_config = config.client_config
        self.model_config = config.get_model_config(model_name)

    async def call(
        self,
        messages: list,
        tools: list[LocalFunction],
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Execute a single LLM call with native tool support.

        Args:
            messages: Conversation history as a list of message dicts
                (with ``role``, ``content``, and optionally ``tool_calls``).
            tools: List of ``LocalFunction`` tools available to the model.
            max_tokens: Per-call output token limit. If ``None``, falls back
                to ``Config.DEFAULT_MAX_TOKENS``.

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

        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.config.DEFAULT_MAX_TOKENS
        )

        response = await model.invoke(
            messages=messages,
            tools=tool_infos,
            temperature=self.config.DEFAULT_TEMPERATURE,
            max_tokens=effective_max_tokens,
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