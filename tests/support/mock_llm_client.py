from typing import Any
from unittest.mock import MagicMock

from project.llm_client import LLMClient


class MockLLMClient(LLMClient):
    """Fake LLM client for testing without real network calls.

    Returns predefined text responses sequentially on each call,
    wrapped in a mock object that mimics the openjiuwen workflow result.
    Inherits from LLMClient to satisfy type checking.
    """

    def __init__(self, responses: list[str] | None = None):
        """Initialize the mock with a list of responses.

        Args:
            responses: List of text responses to return sequentially.
                       If exhausted, returns an empty string.
        """
        # Do not call super().__init__() to avoid requiring real configs
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict[str, str]] = []

    async def call(self, system_prompt: str, user_prompt: str) -> Any:
        """Simulate an LLM call.

        Args:
            system_prompt: The system prompt (recorded for inspection).
            user_prompt: The user prompt (recorded for inspection).

        Returns:
            A mock object with a ``result`` attribute containing the
            expected ``output.answer`` structure.
        """
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })

        response_text = (
            self.responses[self.call_count]
            if self.call_count < len(self.responses)
            else ""
        )
        self.call_count += 1

        mock_result = MagicMock()
        mock_result.result = {
            "output": {
                "answer": response_text
            }
        }
        return mock_result