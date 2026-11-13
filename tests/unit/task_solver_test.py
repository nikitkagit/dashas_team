import json
from pathlib import Path
from typing import Any, Optional

import pytest

from project.llm_client import LLMClient
from project.task_repo import TaskRepo
from project.task_solver import TaskSolver
from tests.support.mock_config import MockConfig
from tests.support.mock_llm_client import MockLLMClient, MockResponse, MockToolCall


def write_task(
    root: Path,
    dirname: str,
    data: dict,
    files: Optional[dict[str, str]] = None,
) -> Path:
    """Create a task directory with task.json and optional workspace files."""
    task_dir = root / dirname
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    if files:
        workspace = task_dir / "workspace"
        workspace.mkdir(exist_ok=True)
        for name, content in files.items():
            (workspace / name).write_text(content, encoding="utf-8")
    return task_dir


@pytest.fixture
def task_repo(tmp_path: Path) -> TaskRepo:
    """Create a temporary TaskRepo with several mock tasks."""
    # Valid coding task
    write_task(
        tmp_path,
        "task_001",
        {
            "task_id": "task_001",
            "task_type": "coding",
            "instruction": "Write the answer to answer.txt",
            "expected": "hello",
            "reward": 10,
        },
    )

    # Task with unknown type
    write_task(
        tmp_path,
        "task_002",
        {
            "task_id": "task_002",
            "task_type": "unknown_type",
            "instruction": "Do something",
            "reward": 5,
        },
    )

    # Task with a pre-existing file in the workspace
    write_task(
        tmp_path,
        "task_003",
        {
            "task_id": "task_003",
            "task_type": "coding",
            "instruction": "Read data.txt and report its content",
            "expected": "sample data",
            "reward": 15,
        },
        files={"data.txt": "sample data"},
    )

    return TaskRepo(tmp_path)


@pytest.fixture
def mock_config() -> MockConfig:
    """Create a MockConfig instance for tests."""
    return MockConfig()


@pytest.mark.asyncio
async def test_solve_task_not_found(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """Solving a non-existent task returns failure without calling the LLM."""
    mock_llm = MockLLMClient(responses=[])
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    answer, success, in_tok, out_tok = await solver.solve("non_existent_task")

    assert not success
    assert "not found" in answer.lower()
    assert in_tok == 0 and out_tok == 0
    assert mock_llm.call_count == 0


@pytest.mark.asyncio
async def test_solve_unknown_task_type(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """Solving a task with an unknown type returns failure without calling the LLM."""
    mock_llm = MockLLMClient(responses=[])
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    answer, success, in_tok, out_tok = await solver.solve("task_002")

    assert not success
    assert "Unknown task type" in answer
    assert in_tok == 0 and out_tok == 0
    assert mock_llm.call_count == 0


@pytest.mark.asyncio
async def test_solve_direct_answer(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """When the LLM returns a plain text answer, it is written to answer.txt."""
    mock_llm = MockLLMClient(responses=["The final answer is 42"])
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    answer, success, in_tok, out_tok = await solver.solve("task_001")

    assert success
    assert answer == "The final answer is 42"
    assert mock_llm.call_count == 1

    # Tokens reported by the mock
    assert in_tok == 10
    assert out_tok == 5

    # answer.txt was written to the task directory
    task_dir = task_repo.get_task_dir("task_001")
    assert task_dir is not None
    assert (task_dir / "answer.txt").read_text(encoding="utf-8") == "The final answer is 42"


@pytest.mark.asyncio
async def test_solve_with_native_tool_call(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """Native tool_calls on the response are executed and their results fed back."""
    # First response: native tool call to write_file
    tool_call = MockToolCall(
        name="write_file",
        arguments={"path": "answer.txt", "content": "tool_generated_answer"},
    )
    # Second response: plain final answer
    final_response = MockResponse(
        content="Task completed.",
        input_tokens=10,
        output_tokens=5,
    )

    mock_llm = MockLLMClient(responses=[[tool_call], final_response])
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    answer, success, in_tok, out_tok = await solver.solve("task_001")

    assert success
    # TaskSolver reads from answer.txt if it exists after tool execution
    assert answer == "tool_generated_answer"
    assert mock_llm.call_count == 2

    # Tokens accumulated over both rounds
    assert in_tok == 20
    assert out_tok == 10

    # Verify the tool result was passed back to the LLM in the next turn
    second_call = mock_llm.calls[1]
    messages = second_call["messages"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "Written 21 characters to answer.txt" in tool_messages[0]["content"]
    assert tool_messages[0]["tool_call_id"] == tool_call.id


@pytest.mark.asyncio
async def test_solve_tool_execution_error(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """A tool that raises an error reports the error back to the LLM."""
    # Request a tool with an invalid path to trigger PermissionError
    tool_call = MockToolCall(
        name="read_file",
        arguments={"path": "../escape_attempt.txt"},
    )
    final_response = MockResponse(content="Handled error.")

    mock_llm = MockLLMClient(responses=[[tool_call], final_response])
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    answer, success, _, _ = await solver.solve("task_001")

    assert success
    assert mock_llm.call_count == 2

    # Verify error message was passed to LLM
    second_call = mock_llm.calls[1]
    messages = second_call["messages"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "Error:" in tool_messages[0]["content"]
    assert "Path outside workspace" in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_solve_unknown_tool_name(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """Calling a tool that does not exist reports 'Tool not found'."""
    tool_call = MockToolCall(
        name="nonexistent_tool",
        arguments={},
    )
    final_response = MockResponse(content="Final answer")

    mock_llm = MockLLMClient(responses=[[tool_call], final_response])
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    answer, success, _, _ = await solver.solve("task_001")

    assert success
    assert mock_llm.call_count == 2

    second_call = mock_llm.calls[1]
    messages = second_call["messages"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert tool_messages[0]["content"] == "Tool not found"


@pytest.mark.asyncio
async def test_solve_max_rounds_exceeded(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """The solver stops after max_rounds (10) of tool calls."""
    tool_call = MockToolCall(
        name="list_files",
        arguments={"subdir": "."},
    )
    # Provide more responses than max_rounds
    mock_llm = MockLLMClient(responses=[[tool_call]] * 15)
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    answer, success, in_tok, out_tok = await solver.solve("task_001")

    assert not success
    assert "Max rounds exceeded" in answer
    assert mock_llm.call_count == 10
    # Tokens accumulated across 10 rounds
    assert in_tok == 100
    assert out_tok == 50


@pytest.mark.asyncio
async def test_solve_llm_client_exception(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """An exception raised by the LLM client is propagated as a failure."""

    class FailingLLMClient(LLMClient):
        def __init__(self) -> None:
            pass  # skip real config initialization

        async def call(
            self, messages: list, tools: list, max_tokens: Optional[int] = None
        ) -> Any:
            raise RuntimeError("Simulated network failure")

    solver = TaskSolver(task_repo, FailingLLMClient(), mock_config)

    answer, success, in_tok, out_tok = await solver.solve("task_001")

    assert not success
    assert "LLM Error" in answer
    assert "Simulated network failure" in answer
    assert in_tok == 0 and out_tok == 0


@pytest.mark.asyncio
async def test_solve_reads_answer_from_file_if_present(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """If answer.txt already exists in workspace, its content is used as the answer."""
    # Pre-populate answer.txt in the workspace
    task_dir = task_repo.get_task_dir("task_001")
    assert task_dir is not None
    workspace = task_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "answer.txt").write_text("file-based answer", encoding="utf-8")

    # LLM returns a different text, but the file takes precedence
    mock_llm = MockLLMClient(responses=["ignored text"])
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    answer, success, _, _ = await solver.solve("task_001")

    assert success
    assert answer == "file-based answer"


@pytest.mark.asyncio
async def test_solve_passes_tools_to_client(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """The solver passes the full list of tools to the LLM client on every call."""
    mock_llm = MockLLMClient(responses=["Final answer"])
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    await solver.solve("task_001")

    first_call = mock_llm.calls[0]
    tool_names = {t.card.name for t in first_call["tools"]}
    expected = {
        "read_file", "write_file", "list_files", "search_files",
        "run_python", "read_csv_table", "remember", "recall",
    }
    assert tool_names == expected


@pytest.mark.asyncio
async def test_solve_system_prompt_in_first_call(task_repo: TaskRepo, mock_config: MockConfig) -> None:
    """The first LLM call contains a system prompt and the task instruction."""
    mock_llm = MockLLMClient(responses=["Final answer"])
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    await solver.solve("task_001")

    first_call = mock_llm.calls[0]
    messages = first_call["messages"]
    assert messages[0]["role"] == "system"
    assert "Available tools are provided natively" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Write the answer to answer.txt" in messages[1]["content"]


@pytest.mark.asyncio
async def test_solve_max_tokens_budget(task_repo, mock_config):
    """Solver stops and returns failure if max_tokens budget is exceeded."""
    mock_llm = MockLLMClient(responses=["The final answer is 42"])
    solver = TaskSolver(task_repo, mock_llm, mock_config)

    # Before call: remaining = 5, call_max_tokens = max(1, 5) = 5.
    # After call: used = int(5 + 10/10) = 6. 6 >= 5 -> fails.
    answer, success, in_tok, out_tok = await solver.solve("task_001", max_tokens=5)

    assert not success
    assert "Max tokens exceeded" in answer
    assert in_tok == 10
    assert out_tok == 5
    assert mock_llm.calls[0]["max_tokens"] == 5