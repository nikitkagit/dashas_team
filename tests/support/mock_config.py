from typing import Any, Optional

from project.config import Config


class MockConfig(Config):
    """Lightweight Config mock for tests.

    Avoids reading environment variables and constructing real
    ``ModelClientConfig`` / ``ModelRequestConfig`` objects while
    preserving the same public interface used by ``TaskSolver``
    and ``LLMClient``.
    """

    def __init__(self) -> None:
        # Intentionally skip Config.__init__ to avoid env-var lookups
        # and real provider configuration construction.
        self.client_config = None
        self.DEFAULT_TEMPERATURE = 0.1
        self.DEFAULT_MAX_TOKENS = 4096

    def get_model_name(self, task_type: str) -> Optional[str]:
        if task_type == "coding":
            return "mock-coding-model"
        if task_type == "math":
            return "mock-math-model"
        return None

    def get_model_config(self, model_name: str) -> Any:
        return {"model": model_name}