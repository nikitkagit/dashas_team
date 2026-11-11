import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from project.llm_client import LLMClient
from tests.support.mock_llm_client import MockLLMClient
from project.task_repo import TaskRepo
from project.task_solver import TaskSolver

@pytest.fixture
def task_repo(tmp_path: Path) -> TaskRepo:
    """Create a temporary TaskRepo with mock tasks."""
    # Valid coding task
    task1_dir = tmp_path / "task_001"
    task1_dir.mkdir()
    task1_data = {
        "task_id": "task_001",
        "task_type": "coding",
        "instruction": "Write the answer to answer.txt",
        "expected": "hello",
    }
    (task1_dir / "task.json").write_text(json.dumps(task1_data), encoding="utf-8")

    # Task with unknown type
    task2_dir = tmp_path / "task_002"
    task2_dir.mkdir()
    task2_data = {
        "task_id": "task_002",
        "task_type": "unknown_type",
        "instruction": "Do something",
    }
    (task2_dir / "task.json").write_text(json.dumps(task2_data), encoding="utf-8")

    return TaskRepo(tmp_path)


@pytest.mark.asyncio
async def test_solve_task_not_found(task_repo: TaskRepo):
    """Test solving a task that does not exist in the repo."""
    mock_llm = MockLLMClient(responses=[])
    solver = TaskSolver(task_repo, mock_llm)

    answer, success = await solver.solve("non_existent_task")

    assert not success
    assert "not found" in answer.lower()
    assert mock_llm.call_count == 0


@pytest.mark.asyncio
async def test_solve_unknown_task_type(task_repo: TaskRepo):
    """Test solving a task with a type not in MODEL_ROUTING."""
    mock_llm = MockLLMClient(responses=[])
    solver = TaskSolver(task_repo, mock_llm)

    answer, success = await solver.solve("task_002")

    assert not success
    assert "Unknown task type" in answer
    assert mock_llm.call_count == 0


@pytest.mark.asyncio
async def test_solve_direct_answer(task_repo: TaskRepo):
    """Test solving a task where the LLM returns the answer directly."""
    mock_llm = MockLLMClient(responses=["The final answer is 42"])
    solver = TaskSolver(task_repo, mock_llm)

    answer, success = await solver.solve("task_001")

    assert success
    assert answer == "The final answer is 42"
    assert mock_llm.call_count == 1
    
    # Verify answer.txt was written to the task directory
    task_dir = task_repo.get_task_dir("task_001")
    assert task_dir is not None  # или if task_dir is None: pytest.skip(...)
    assert (task_dir / "answer.txt").read_text(encoding="utf-8") == "The final answer is 42"


@pytest.mark.asyncio
async def test_solve_with_tool_call(task_repo: TaskRepo):
    """Test solving a task that requires a tool call before finishing."""
    # First response: tool call to write the answer
    tool_call_response = json.dumps({
        "tool": "write_file",
        "args": {"path": "answer.txt", "content": "tool_generated_answer"}
    })
    # Second response: LLM acknowledges completion
    final_response = "Task completed."
    
    mock_llm = MockLLMClient(responses=[tool_call_response, final_response])
    solver = TaskSolver(task_repo, mock_llm)

    answer, success = await solver.solve("task_001")

    assert success
    # TaskSolver reads from answer.txt if it exists after tool execution
    assert answer == "tool_generated_answer"
    assert mock_llm.call_count == 2

    # Verify the tool result was passed back to the LLM in the next turn
    assert "[Tool result]:" in mock_llm.calls[1]["user_prompt"]
    assert "tool_generated_answer" in mock_llm.calls[1]["user_prompt"]


@pytest.mark.asyncio
async def test_solve_tool_execution_error(task_repo: TaskRepo):
    """Test handling of a tool execution that raises an exception."""
    # Request a tool with invalid arguments to trigger an error
    tool_call_response = json.dumps({
        "tool": "read_file",
        "args": {"path": "../escape_attempt.txt"} # Will trigger PermissionError in ToolBox
    })
    final_response = "Handled error."
    
    mock_llm = MockLLMClient(responses=[tool_call_response, final_response])
    solver = TaskSolver(task_repo, mock_llm)

    answer, success = await solver.solve("task_001")

    assert success
    assert mock_llm.call_count == 2
    # Verify error message was passed to LLM
    assert "Error:" in mock_llm.calls[1]["user_prompt"]
    assert "Path outside workspace" in mock_llm.calls[1]["user_prompt"]


@pytest.mark.asyncio
async def test_solve_max_rounds_exceeded(task_repo: TaskRepo):
    """Test that the solver stops after max_rounds (10) of tool calls."""
    tool_call_response = json.dumps({
        "tool": "list_files",
        "args": {"subdir": "."}
    })
    # Provide more responses than max_rounds
    mock_llm = MockLLMClient(responses=[tool_call_response] * 15)
    solver = TaskSolver(task_repo, mock_llm)

    answer, success = await solver.solve("task_001")

    assert not success
    assert "Max rounds exceeded" in answer
    assert mock_llm.call_count == 10


@pytest.mark.asyncio
async def test_solve_llm_client_exception(task_repo: TaskRepo):
    """Test handling of an exception raised by the LLM client."""
    class FailingLLMClient(LLMClient):
        def __init__(self):
            pass  # skip real config initialization

        async def call(self, system_prompt: str, user_prompt: str) -> Any:
            raise RuntimeError("Simulated network failure")

    solver = TaskSolver(task_repo, FailingLLMClient())

    answer, success = await solver.solve("task_001")

    assert not success
    assert "LLM Error" in answer
    assert "Simulated network failure" in answer