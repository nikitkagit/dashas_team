from typing import Any

from openjiuwen.core.foundation.llm import (
    ModelClientConfig,
    ModelRequestConfig,
    SystemMessage,
    UserMessage,
)
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.workflow import (
    End,
    LLMCompConfig,
    LLMComponent,
    Start,
    Workflow,
    WorkflowCard,
)

class LLMClient:
    """Thin wrapper around openjiuwen workflow execution.

    This class is intentionally minimal: it only builds and runs a
    single-step workflow that forwards a system prompt and a user
    prompt to the configured LLM. Higher-level logic such as
    conversation history formatting lives in ``TaskSolver``.
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

    async def call(self, system_prompt: str, user_prompt: str) -> Any:
        """Execute a single LLM call.

        Args:
            system_prompt: System prompt content. May be empty.
            user_prompt: User prompt content.

        Returns:
            The workflow result object. The caller is responsible for
            extracting the assistant's text answer from it.
        """
        comp_config = LLMCompConfig(
            model_config=self.model_config,
            model_client_config=self.client_config,
            system_prompt_template=SystemMessage(content=system_prompt) if system_prompt else None,
            user_prompt_template=UserMessage(content=user_prompt),
            response_format={"type": "text"},
            output_config={"answer": {"type": "string"}},
        )

        workflow_card = WorkflowCard(
            id="llm_step",
            name="llm_step",
            version="1.0",
            input_params={"type": "object", "properties": {}},
        )
        workflow = Workflow(workflow_card)
        workflow.set_start_comp("start", Start())
        workflow.add_workflow_comp("llm", LLMComponent(comp_config))
        workflow.set_end_comp("end", End(), inputs_schema={"answer": "${llm.answer}"})
        workflow.add_connection("start", "llm")
        workflow.add_connection("llm", "end")

        return await Runner.run_workflow(workflow, inputs={})