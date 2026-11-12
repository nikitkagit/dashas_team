import json
import uuid
from typing import Any, Optional

from openjiuwen.core.foundation.tool import LocalFunction

from project.llm_client import LLMClient


class MockUsageMetadata:
    """Mimics the ``usage_metadata`` attribute of an LLM response."""

    def __init__(self, input_tokens: int = 0, output_tokens: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockToolCall:
    """Mimics a native ToolCall object (Pydantic-like, attribute access).

    ``arguments`` is always a JSON string, as returned by the real API.
    """

    def __init__(
        self,
        name: str,
        arguments: dict | str,
        id: Optional[str] = None,
    ):
        self.id = id or f"call_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.arguments = (
            arguments if isinstance(arguments, str) else json.dumps(arguments)
        )


class MockResponse:
    """Mimics an AssistantMessage returned by ``Model.invoke``."""

    def __init__(
        self,
        content: str = "",
        tool_calls: Optional[list[MockToolCall]] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        self.content = content
        self.tool_calls = tool_calls
        self.usage_metadata = MockUsageMetadata(input_tokens, output_tokens)


class MockLLMClient(LLMClient):
    """Fake LLM client for testing without real network calls.

    Returns predefined responses sequentially. Each response item may be:
      * ``str`` - a plain final answer,
      * ``list[MockToolCall]`` - native tool calls to execute,
      * ``MockResponse`` - a fully specified response.

    Inherits from LLMClient to satisfy type checking.
    """

    def __init__(
        self,
        responses: Optional[list[Any]] = None,
        input_tokens: int = 10,
        output_tokens: int = 5,
    ):
        """Initialize the mock.

        Args:
            responses: Responses to return sequentially. Exhausted -> empty answer.
            input_tokens: Default input token count reported per call.
            output_tokens: Default output token count reported per call.
        """
        # Do not call super().__init__() to avoid requiring real configs
        self.responses: list[Any] = responses or []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def _build_response(self, item: Any) -> MockResponse:
        """Convert a queued response item into a MockResponse."""
        if isinstance(item, MockResponse):
            return item
        if isinstance(item, list):
            return MockResponse(
                content="",
                tool_calls=item,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            )
        return MockResponse(
            content=str(item),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    async def call(self, messages: list, tools: list[LocalFunction]) -> Any:
        """Simulate a native LLM call.

        Args:
            messages: Conversation history (snapshot recorded for inspection).
            tools: Tools offered to the model (recorded for inspection).

        Returns:
            A MockResponse with ``content``, ``tool_calls`` and ``usage_metadata``.
        """
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "tools": list(tools),
        })

        item = (
            self.responses[self.call_count]
            if self.call_count < len(self.responses)
            else ""
        )
        self.call_count += 1
        return self._build_response(item)