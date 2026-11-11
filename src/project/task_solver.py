import json
import re
from pathlib import Path
from typing import Any

from project.config import MODEL_ROUTING
from project.llm_client import LLMClient
from project.task_repo import TaskRepo
from project.tool_box import ToolBox

class TaskSolver:
    """Solves tasks using an LLM client and the provided ToolBox.

    This class owns the conversation history, tool-dispatch loop,
    and answer extraction logic. The underlying LLM invocation is
    delegated to an ``LLMClient`` instance, which can be replaced
    with a test double in unit tests.
    """

    def __init__(self, task_repo: TaskRepo, llm_client: LLMClient):
        """Initialize the solver.

        Args:
            task_repo: Repository providing access to tasks by ID.
            llm_client: Client used to invoke the LLM. Can be a
                stub or mock in tests.
        """
        self.task_repo = task_repo
        self.llm_client = llm_client

    async def solve(self, task_id: str) -> tuple[str, bool]:
        """Solve a specific task by its ID.

        Args:
            task_id: The unique identifier of the task to solve.

        Returns:
            A tuple containing the final answer text and a boolean
            indicating whether the solving process completed
            successfully (not whether the answer is correct).
        """
        task = self.task_repo.get_task(task_id)
        task_dir = self.task_repo.get_task_dir(task_id)

        if not task or not task_dir:
            return f"Task {task_id} not found", False

        task_type = task.get("task_type", "")
        model_name = MODEL_ROUTING.get(task_type)
        if not model_name:
            return f"Unknown task type: {task_type}", False

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

        tools_desc = "\n".join([
            f"- {t.card.name}: {t.card.description}"
            for t in tools
        ])

        system_prompt = (
            "You are an agent that solves tasks. Available tools:\n"
            f"{tools_desc}\n\n"
            "To call a tool, respond with JSON of the form: "
            '{"tool": "tool_name", "args": {...}}\n'
            "When the task is solved, write the final answer and save it to answer.txt via write_file."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task["instruction"]},
        ]

        max_rounds = 10
        for round_idx in range(max_rounds):
            system_content, user_prompt = self._format_messages(messages)

            try:
                result = await self.llm_client.call(system_content, user_prompt)
            except Exception as e:
                return f"LLM Error: {e}", False

            answer_text = self._extract_answer(result)

            tool_match = re.search(
                r'\{"tool":\s*"(\w+)",\s*"args":\s*(\{.*?\})\}',
                answer_text,
                re.DOTALL,
            )

            if tool_match:
                tool_name = tool_match.group(1)
                try:
                    tool_args = json.loads(tool_match.group(2))
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

                messages.append({"role": "assistant", "content": answer_text})
                messages.append({"role": "tool", "content": str(tool_result)[:4000]})
                continue

            answer_file = toolbox.workspace / "answer.txt"
            if answer_file.exists():
                answer_text = answer_file.read_text(encoding="utf-8").strip()

            (task_dir / "answer.txt").write_text(answer_text, encoding="utf-8")
            return answer_text, True

        return "Max rounds exceeded", False

    @staticmethod
    def _format_messages(messages: list[dict]) -> tuple[str, str]:
        """Split messages into system prompt and flattened user prompt.

        Args:
            messages: Conversation history as a list of message dicts.

        Returns:
            A tuple ``(system_content, user_prompt)``.
        """
        history_parts = []
        system_content = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
                continue
            if role == "tool":
                history_parts.append(f"[Tool result]: {content}")
            elif role == "assistant":
                history_parts.append(f"[Assistant]: {content}")
            else:
                history_parts.append(f"[User]: {content}")

        return system_content, "\n".join(history_parts)

    @staticmethod
    def _extract_answer(result: Any) -> str:
        """Extract the assistant's text answer from a workflow result.

        Args:
            result: The object returned by ``LLMClient.run``.

        Returns:
            The extracted answer string, or an empty string if the
            result structure is unexpected.
        """
        if hasattr(result, "result") and isinstance(result.result, dict):
            output = result.result.get("output", {})
            if isinstance(output, dict):
                return output.get("answer", "")
        return ""