"""Query an LLM through openJiuwen and the NeuralDeep OpenAI-compatible API."""

import asyncio
import os
import sys

from openjiuwen.core.foundation.llm import (
    ModelClientConfig,
    ModelRequestConfig,
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


def build_config() -> tuple[ModelClientConfig, ModelRequestConfig]:
    """Read environment variables and build the model client/config."""
    provider = os.getenv("MODEL_PROVIDER", "OpenAI")
    api_key = os.getenv("API_KEY")
    api_base = os.getenv("API_BASE", "https://api.neuraldeep.ru/v1")
    model_name = os.getenv("MODEL_NAME", "qwen3.8-27b")

    if not api_key:
        sys.exit("API_KEY is not set. Export a LiteLLM virtual key (starts with 'sk-').")
    if not api_key.startswith("sk-"):
        sys.exit(
            "API_KEY must be a LiteLLM virtual key starting with 'sk-'. "
            "The invite code is not an API key."
        )

    client_config = ModelClientConfig(
        client_provider=provider,
        api_key=api_key,
        api_base=api_base,
        verify_ssl=False,
    )
    model_config = ModelRequestConfig(model=model_name)
    return client_config, model_config

def build_llm_component(
    client_config: ModelClientConfig,
    model_config: ModelRequestConfig,
    query: str,
) -> LLMComponent:
    """Create an LLM workflow component with a query already inlined."""
    comp_config = LLMCompConfig(
        model_config=model_config,
        model_client_config=client_config,
        user_prompt_template=UserMessage(content=f"Запрос пользователя: {query}"),
        response_format={"type": "text"},
        output_config={"answer": {"type": "string"}},
    )
    return LLMComponent(comp_config)


def build_workflow_card() -> WorkflowCard:
    """Define the input schema for the workflow."""
    return WorkflowCard(
        id="simple_llm_workflow",
        name="simple_llm",
        version="1.0",
        input_params={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )


async def ask_llm(
    query: str,
    client_config: ModelClientConfig,
    model_config: ModelRequestConfig,
    workflow_card: WorkflowCard,
) -> object:
    workflow = Workflow(workflow_card)
    workflow.set_start_comp("start", Start())
    workflow.add_workflow_comp(
        "llm", build_llm_component(client_config, model_config, query)
    )
    workflow.set_end_comp("end", End())
    workflow.add_connection("start", "llm")
    workflow.add_connection("llm", "end")
    return await Runner.run_workflow(workflow, inputs={"query": query})


async def main() -> None:
    client_config, model_config = build_config()
    workflow_card = build_workflow_card()

    query = "Hello, how are you?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    result = await ask_llm(query, client_config, model_config, workflow_card)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())