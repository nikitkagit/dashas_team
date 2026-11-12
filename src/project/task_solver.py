import json
from pathlib import Path
from typing import Any

from project.llm_client import LLMClient, extract_usage
from project.task_repo import TaskRepo
from project.tool_box import ToolBox
from project.config import Config


class TaskSolver:
    """Solves tasks using an LLM client and the provided ToolBox.

    This class owns the conversation history, tool-dispatch loop,
    token accounting, and answer extraction logic. The underlying
    LLM invocation is delegated to an ``LLMClient`` instance, which
    can be replaced with a test double in unit tests.
    """

    def __init__(self, task_repo: TaskRepo, llm_client: LLMClient, config: Config):
        """Initialize the solver.

        Args:
            task_repo: Repository providing access to tasks by ID.
            llm_client: Client used to invoke the LLM. Can be a
                stub or mock in tests.
            config: Shared ``Config`` instance for model routing.
        """
        self.task_repo = task_repo
        self.llm_client = llm_client
        self.config = config

    async def solve(self, task_id: str) -> tuple[str, bool, int, int]:
        """Solve a specific task by its ID.

        Args:
            task_id: The unique identifier of the task to solve.

        Returns:
            A tuple ``(answer_text, success, total_input_tokens, total_output_tokens)``.
            ``success`` indicates whether the solving process completed
            (and, for ``unittest`` tasks, whether the test passed).
        """
        task = self.task_repo.get_task(task_id)
        task_dir = self.task_repo.get_task_dir(task_id)

        if not task or not task_dir:
            return f"Task {task_id} not found", False, 0, 0

        task_type = task.get("task_type", "")
        model_name = self.config.get_model_name(task_type)
        if not model_name:
            return f"Unknown task type: {task_type}", False, 0, 0

        toolbox = ToolBox(task_dir)

        tools = [
            toolbox.read_file(),
            toolbox.write_file(),
            toolbox.list_files(),
            toolbox.search_files(),
            toolbox.run_python(),
            toolbox.read_csv_table(),
            toolbox.remember(),
            toolbox.recall(),
        ]

        system_prompt = (
            "You are an agent that solves tasks. Available tools are provided natively.\n"
            "Use tools as needed. When the task is solved, write the final answer "
            "and save it to answer.txt via write_file."
        )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task["instruction"]},
        ]

        total_input_tokens = 0
        total_output_tokens = 0

        max_rounds = 10
        for round_idx in range(max_rounds):
            try:
                response = await self.llm_client.call(messages, tools)
            except Exception as e:
                return f"LLM Error: {e}", False, total_input_tokens, total_output_tokens

            print(f"[{task['task_id']}] round {round_idx}: model={model_name}")

            in_tok_round, out_tok_round = extract_usage(response)
            total_input_tokens += in_tok_round
            total_output_tokens += out_tok_round

            # Extract assistant content and tool calls from the response
            assistant_content = getattr(response, "content", "") or ""
            tool_calls = getattr(response, "tool_calls", None)

            # If there are tool calls, execute each and continue the loop
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    # ToolCall is a Pydantic object, access attributes directly
                    tool_name = tc.name
                    tool_args_str = tc.arguments
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}

                    tool_result = "Tool not found"
                    for t in tools:
                        if t.card.name == tool_name:
                            try:
                                tool_result = await t.invoke(tool_args)
                            except Exception as e:
                                tool_result = f"Error: {e}"
                            break

                    messages.append({
                        "role": "tool",
                        "content": str(tool_result)[:4000],
                        "tool_call_id": tc.id,
                    })
                continue

            # No tool calls - this is the final answer
            answer_text = assistant_content.strip()

            answer_file = toolbox.workspace / "answer.txt"
            if answer_file.exists():
                answer_text = answer_file.read_text(encoding="utf-8").strip()

            (task_dir / "answer.txt").write_text(answer_text, encoding="utf-8")

            print(
                f"  [tokens] input={total_input_tokens}, "
                f"output={total_output_tokens}, "
                f"total={total_input_tokens + total_output_tokens}"
            )

            # return the answer
            return answer_text, True, total_input_tokens, total_output_tokens

        return "Max rounds exceeded", False, total_input_tokens, total_output_tokens