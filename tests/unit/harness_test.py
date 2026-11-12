import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from project.harness import Harness, check_answer, normalize_int


class TestNormalizeInt:
    """Tests for the normalize_int helper function."""

    def test_normalize_int_simple(self):
        assert normalize_int("42") == 42

    def test_normalize_int_with_spaces(self):
        assert normalize_int("  42  ") == 42

    def test_normalize_int_with_non_digits(self):
        assert normalize_int("answer: 42") == 42

    def test_normalize_int_negative(self):
        assert normalize_int("-10") == -10

    def test_normalize_int_multiline(self):
        assert normalize_int("42\nextra text") == 42

    def test_normalize_int_invalid(self):
        assert normalize_int("no numbers here") is None

    def test_normalize_int_empty(self):
        assert normalize_int("") is None


class TestCheckAnswer:
    """Tests for the check_answer helper function."""

    def test_check_answer_exact_str_match(self):
        task = {"check": "exact_str", "expected": "hello"}
        assert check_answer(task, "hello") is True

    def test_check_answer_exact_str_case_insensitive(self):
        task = {"check": "exact_str", "expected": "Hello"}
        assert check_answer(task, "hello") is True

    def test_check_answer_exact_str_mismatch(self):
        task = {"check": "exact_str", "expected": "hello"}
        assert check_answer(task, "world") is False

    def test_check_answer_exact_int_match(self):
        task = {"check": "exact_int", "expected": "42"}
        assert check_answer(task, "42") is True

    def test_check_answer_exact_int_with_text(self):
        task = {"check": "exact_int", "expected": "42"}
        assert check_answer(task, "The answer is 42") is True

    def test_check_answer_exact_int_mismatch(self):
        task = {"check": "exact_int", "expected": "42"}
        assert check_answer(task, "43") is False

    def test_check_answer_contains_match(self):
        task = {"check": "contains", "expected": "hello"}
        assert check_answer(task, "hello world") is True

    def test_check_answer_contains_case_insensitive(self):
        task = {"check": "contains", "expected": "HELLO"}
        assert check_answer(task, "hello world") is True

    def test_check_answer_contains_mismatch(self):
        task = {"check": "contains", "expected": "hello"}
        assert check_answer(task, "world") is False

    def test_check_answer_default_exact_match(self):
        task = {"expected": "hello"}
        assert check_answer(task, "hello") is True

    def test_check_answer_default_exact_mismatch(self):
        task = {"expected": "hello"}
        assert check_answer(task, "world") is False


class TestHarnessInit:
    """Tests for Harness initialization."""

    def test_harness_init_valid_dir(self, tmp_path):
        """Harness initializes successfully with a valid tasks directory."""
        mock_config = MagicMock()
        mock_solver = MagicMock()
        harness = Harness(tmp_path, mock_config, mock_solver)
        assert harness.tasks_dir == tmp_path
        assert harness.repo is not None
        assert harness.config is mock_config
        assert harness.solver is mock_solver

    def test_harness_init_invalid_dir(self, tmp_path):
        """Harness raises ValueError when tasks directory doesn't exist."""
        invalid_dir = tmp_path / "nonexistent"
        mock_config = MagicMock()
        mock_solver = MagicMock()
        with pytest.raises(ValueError, match="Tasks directory not found"):
            Harness(invalid_dir, mock_config, mock_solver)


@pytest.fixture
def mock_config():
    """Create a mock Config instance."""
    config = MagicMock()
    config.get_model_name.return_value = "test-model"
    return config


@pytest.fixture
def mock_solver():
    """Create a mock TaskSolver instance."""
    solver = MagicMock()
    solver.solve = AsyncMock()
    return solver


@pytest.fixture
def tasks_dir_empty(tmp_path):
    """Empty tasks directory."""
    return tmp_path


@pytest.fixture
def tasks_dir_single_task(tmp_path):
    """Tasks directory with a single successful task."""
    task_dir = tmp_path / "task_001"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps({
            "task_id": "task_001",
            "task_type": "coding",
            "instruction": "Test task",
            "expected": "answer",
            "reward": 10,
        }, ensure_ascii=False),
        encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def tasks_dir_single_task_failure(tmp_path):
    """Tasks directory with a single task that will fail."""
    task_dir = tmp_path / "task_001"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps({
            "task_id": "task_001",
            "task_type": "coding",
            "instruction": "Test task",
            "expected": "correct",
            "reward": 10,
        }, ensure_ascii=False),
        encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def tasks_dir_multiple_tasks(tmp_path):
    """Tasks directory with multiple tasks."""
    for i, (task_id, reward) in enumerate([("task_001", 10), ("task_002", 20)], 1):
        task_dir = tmp_path / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.json").write_text(
            json.dumps({
                "task_id": task_id,
                "task_type": "coding",
                "instruction": f"Task {i}",
                "expected": f"answer{i}",
                "reward": reward,
            }, ensure_ascii=False),
            encoding="utf-8"
        )
    return tmp_path


@pytest.fixture
def tasks_dir_unknown_type(tmp_path):
    """Tasks directory with a task of unknown type."""
    task_dir = tmp_path / "task_001"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps({
            "task_id": "task_001",
            "task_type": "unknown",
            "instruction": "Test task",
            "reward": 10,
        }, ensure_ascii=False),
        encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def tasks_dir_unittest(tmp_path):
    """Tasks directory with a unittest task."""
    task_dir = tmp_path / "task_001"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps({
            "task_id": "task_001",
            "task_type": "coding",
            "instruction": "Test task",
            "check": "unittest",
            "reward": 10,
        }, ensure_ascii=False),
        encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def tasks_dir_solver_error(tmp_path):
    """Tasks directory with a task that will cause solver error."""
    task_dir = tmp_path / "task_001"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps({
            "task_id": "task_001",
            "task_type": "coding",
            "instruction": "Test task",
            "expected": "answer",
            "reward": 10,
        }, ensure_ascii=False),
        encoding="utf-8"
    )
    return tmp_path


class TestHarnessRun:
    """Tests for Harness.run method."""

    @pytest.mark.asyncio
    async def test_run_empty_repo(self, tasks_dir_empty, mock_config, mock_solver):
        """Running on an empty repository returns zero rewards and tokens."""
        harness = Harness(tasks_dir_empty, mock_config, mock_solver)
        report = await harness.run()

        assert report["total_reward"] == 0
        assert report["total_input_tokens"] == 0
        assert report["total_output_tokens"] == 0
        assert report["total_tokens"] == 0
        assert report["results"] == []

    @pytest.mark.asyncio
    async def test_run_single_task_success(self, tasks_dir_single_task, mock_config, mock_solver):
        """Running a single successful task returns correct reward and tokens."""
        harness = Harness(tasks_dir_single_task, mock_config, mock_solver)
        mock_solver.solve.return_value = ("answer", True, 100, 50)

        report = await harness.run()

        assert report["total_reward"] == 10
        assert report["total_input_tokens"] == 100
        assert report["total_output_tokens"] == 50
        assert report["total_tokens"] == 150
        assert len(report["results"]) == 1
        assert report["results"][0]["task_id"] == "task_001"
        assert report["results"][0]["success"] is True
        assert report["results"][0]["reward"] == 10

    @pytest.mark.asyncio
    async def test_run_single_task_failure(self, tasks_dir_single_task_failure, mock_config, mock_solver):
        """Running a single failed task returns zero reward."""
        harness = Harness(tasks_dir_single_task_failure, mock_config, mock_solver)
        mock_solver.solve.return_value = ("wrong", True, 100, 50)

        report = await harness.run()

        assert report["total_reward"] == 0
        assert len(report["results"]) == 1
        assert report["results"][0]["reward"] == 0

    @pytest.mark.asyncio
    async def test_run_specific_task_id(self, tasks_dir_multiple_tasks, mock_config, mock_solver):
        """Running with a specific task_id only processes that task."""
        harness = Harness(tasks_dir_multiple_tasks, mock_config, mock_solver)
        mock_solver.solve.return_value = ("answer1", True, 100, 50)

        report = await harness.run(task_id="task_001")

        assert report["total_reward"] == 10
        assert len(report["results"]) == 1
        assert report["results"][0]["task_id"] == "task_001"

    @pytest.mark.asyncio
    async def test_run_unknown_task_type(self, tasks_dir_unknown_type, mock_config, mock_solver):
        """Tasks with unknown type are skipped with zero reward."""
        mock_config.get_model_name.return_value = None
        harness = Harness(tasks_dir_unknown_type, mock_config, mock_solver)

        report = await harness.run()

        assert report["total_reward"] == 0
        assert len(report["results"]) == 1
        assert report["results"][0]["success"] is False
        assert report["results"][0]["reward"] == 0

    @pytest.mark.asyncio
    async def test_run_multiple_tasks(self, tasks_dir_multiple_tasks, mock_config, mock_solver):
        """Running multiple tasks accumulates rewards and tokens correctly."""
        harness = Harness(tasks_dir_multiple_tasks, mock_config, mock_solver)
        mock_solver.solve.side_effect = [
            ("answer1", True, 100, 50),
            ("answer2", True, 200, 100),
        ]

        report = await harness.run()

        assert report["total_reward"] == 30
        assert report["total_input_tokens"] == 300
        assert report["total_output_tokens"] == 150
        assert report["total_tokens"] == 450
        assert len(report["results"]) == 2

    @pytest.mark.asyncio
    async def test_run_unittest_task(self, tasks_dir_unittest, mock_config, mock_solver):
        """Tasks with check='unittest' are always considered successful if solve returns success."""
        harness = Harness(tasks_dir_unittest, mock_config, mock_solver)
        mock_solver.solve.return_value = ("any answer", True, 100, 50)

        report = await harness.run()

        assert report["total_reward"] == 10
        assert report["results"][0]["reward"] == 10

    @pytest.mark.asyncio
    async def test_run_solver_exception(self, tasks_dir_solver_error, mock_config, mock_solver):
        """If solver.solve returns success=False, reward is zero."""
        harness = Harness(tasks_dir_solver_error, mock_config, mock_solver)
        mock_solver.solve.return_value = ("Error occurred", False, 100, 50)

        report = await harness.run()

        assert report["total_reward"] == 0
        assert report["results"][0]["success"] is False
        assert report["results"][0]["reward"] == 0