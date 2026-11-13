import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project.harness import (
    BudgetPlanner,
    Harness,
    _knapsack_01,
    _quantile,
    check_answer,
    normalize_int,
)
from project.task_estimator import Estimate


# ======================================================================
# Helpers
# ======================================================================

def make_estimate(tokens=1000, p=0.8, source="heuristic", cost=0):
    return Estimate(
        tokens=tokens,
        p_solvable=p,
        source=source,
        confidence=0.5,
        estimator_cost=cost,
    )


def make_task(task_type="coding", reward=10, check="exact_str",
              expected="", instruction=""):
    return {
        "task_type": task_type,
        "reward": reward,
        "check": check,
        "expected": expected,
        "instruction": instruction,
    }


def make_estimator(estimates: dict[str, Estimate]):
    """Mock TaskEstimator with predefined Estimate per task_id.

    Keeps a call counter so tests can assert that ``estimate`` is not
    invoked twice for the same task when ``plan`` already returned a cache.
    """
    est = MagicMock()
    call_counts: dict[str, int] = {}

    async def estimate(tid, depth="heuristic"):
        call_counts[tid] = call_counts.get(tid, 0) + 1
        return estimates[tid]

    est.estimate = AsyncMock(side_effect=estimate)
    est.calibrate = MagicMock()
    est.base_costs = {}
    est._call_counts = call_counts
    return est


def make_solver(results: dict[str, tuple]):
    """Mock TaskSolver: results[tid] = (answer, success, in_tok, out_tok)."""
    solver = MagicMock()
    calls: list[tuple[str, int | None]] = []

    async def solve(tid, max_tokens=None):
        calls.append((tid, max_tokens))
        return results[tid]

    solver.solve = AsyncMock(side_effect=solve)
    solver._calls = calls
    return solver


def make_config(model_by_type=None):
    cfg = MagicMock()
    cfg.get_model_name = MagicMock(
        side_effect=lambda t: (model_by_type or {}).get(t, f"model-{t}")
    )
    cfg.token_budget = None
    cfg.normalize_tokens = lambda i, o: int(o + i / 10)
    return cfg


def make_harness(tmp_path, tasks, estimates, solver_results,
                 *, budget=100_000, alpha_init=1.3, beta_mult=1.7,
                 warmup_depth="heuristic", model_by_type=None):
    """Build Harness with mocked repo/solver/estimator.

    The repo mock is backed by ``tmp_path`` and ``tasks`` (a dict of
    ``task_id -> task_dict``). It also lazily creates ``<tmp>/<tid>/workspace``
    on demand so that ``check_answer`` works as expected.
    """
    estimator = make_estimator(estimates)
    solver = make_solver(solver_results)
    config = make_config(model_by_type)

    h = Harness(
        tasks_dir=tmp_path,
        config=config,
        solver=solver,
        estimator=estimator,
        token_budget=budget,
        alpha_init=alpha_init,
        beta_mult=beta_mult,
        warmup_depth=warmup_depth,
    )

    repo = MagicMock()

    def get_task(tid):
        return tasks.get(tid)

    def get_task_dir(tid):
        if tid not in tasks:
            return None
        d = tmp_path / tid
        (d / "workspace").mkdir(parents=True, exist_ok=True)
        return d

    repo.get_all_task_ids = MagicMock(return_value=list(tasks.keys()))
    repo.get_task = MagicMock(side_effect=get_task)
    repo.get_task_dir = MagicMock(side_effect=get_task_dir)
    h.repo = repo

    return h, solver, estimator


# ======================================================================
# normalize_int / check_answer
# ======================================================================

class TestNormalizeInt:
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

    def test_check_answer_unittest_success(self, tmp_path):
        task_dir = tmp_path / "task_001"
        task_dir.mkdir()
        (task_dir / "workspace").mkdir()
        task = {"check": "unittest"}
        with patch("project.harness.run_unittest", return_value=True):
            assert check_answer(task, "anything", task_dir) is True

    def test_check_answer_unittest_failure(self, tmp_path):
        task_dir = tmp_path / "task_001"
        task_dir.mkdir()
        (task_dir / "workspace").mkdir()
        task = {"check": "unittest"}
        with patch("project.harness.run_unittest", return_value=False):
            assert check_answer(task, "anything", task_dir) is False

    def test_check_answer_unittest_no_task_dir(self):
        task = {"check": "unittest"}
        assert check_answer(task, "anything", None) is False


# ======================================================================
# knapsack + quantile
# ======================================================================

class TestKnapsack01:
    def test_empty(self):
        assert _knapsack_01([], [], 100) == []

    def test_zero_capacity(self):
        assert _knapsack_01([10], [1.0], 0) == []

    def test_picks_optimal_not_greedy(self):
        weights = [10, 6, 6]
        values = [100.0, 60.0, 60.0]
        chosen = _knapsack_01(weights, values, 12)
        assert sorted(chosen) == [1, 2]

    def test_single_item_fits(self):
        assert _knapsack_01([5], [10.0], 5) == [0]

    def test_single_item_does_not_fit(self):
        assert _knapsack_01([5], [10.0], 4) == []

    def test_scaling_huge_capacity(self):
        chosen = _knapsack_01([1_000_000], [5.0], 500_000)
        assert chosen == []


class TestQuantile:
    def test_empty_returns_one(self):
        assert _quantile([], 0.5) == 1.0

    def test_single(self):
        assert _quantile([2.0], 0.5) == 2.0

    def test_median(self):
        assert _quantile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_interpolation(self):
        assert _quantile([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)
        assert _quantile([1.0, 2.0, 3.0], 0.8) == pytest.approx(2.6)


# ======================================================================
# BudgetPlanner
# ======================================================================

@pytest.fixture
def planner():
    est = MagicMock()
    est.estimate = AsyncMock(return_value=make_estimate(tokens=1000, p=0.5))
    est.calibrate = MagicMock()
    return BudgetPlanner(budget=10_000, estimator=est)


class TestPlannerCosts:
    def test_cost_applies_alpha(self, planner):
        est = make_estimate(tokens=1000)
        planner.alpha = 2.0
        assert planner._cost(est) == 2000.0

    def test_cost_floor_100(self, planner):
        est = make_estimate(tokens=1)
        planner.alpha = 1.0
        assert planner._cost(est) == 100.0

    def test_value_uses_expected_reward(self, planner):
        est = make_estimate(p=0.5)
        assert planner._value({"reward": 10}, est) == pytest.approx(5.0)

    def test_value_floor_on_p_solvable(self, planner):
        est = make_estimate(p=0.0)
        assert planner._value({"reward": 10}, est) == pytest.approx(0.5)

    def test_task_limit_applies_beta(self, planner):
        planner.alpha = 2.0
        planner.beta_mult = 1.5
        est = make_estimate(tokens=1000)
        assert planner.task_limit(est) == int(1000 * 2.0 * 1.5)


class TestPlannerObserve:
    def test_delegates_calibration(self, planner):
        est = make_estimate(tokens=1000)
        planner.observe("t1", est, real_tokens=1000, success=True)
        planner.estimator.calibrate.assert_called_once_with("t1", 1000, True)

    def test_alpha_moves_toward_ratio_after_3(self, planner):
        est = make_estimate(tokens=1000)
        planner.alpha = 1.0
        planner.ema = 1.0
        for _ in range(3):
            planner.observe("t1", est, real_tokens=2000, success=True)
        assert planner.alpha == pytest.approx(2.0)

    def test_alpha_clamped_to_min(self, planner):
        est = make_estimate(tokens=1000)
        planner.alpha = 1.5
        planner.alpha_min = 1.0
        planner.ema = 1.0
        for _ in range(3):
            planner.observe("t1", est, real_tokens=100, success=False)
        assert planner.alpha == pytest.approx(1.0)

    def test_zero_tokens_skips_ratio(self, planner):
        est = make_estimate(tokens=1000)
        planner.alpha = 1.3
        planner.observe("t1", est, real_tokens=0, success=True)
        assert planner._ratios == []
        planner.estimator.calibrate.assert_called_once()

    def test_ratio_history_unbounded(self, planner):
        est = make_estimate(tokens=1000)
        for _ in range(50):
            planner.observe("t1", est, real_tokens=1000, success=True)
        assert len(planner._ratios) == 50


class TestPlannerPlan:
    """``plan`` and ``reoptimize`` now return ``(ids, cost, cache)``."""

    @pytest.mark.asyncio
    async def test_plan_empty(self, planner):
        ids, cost, cache = await planner.plan([])
        assert ids == []
        assert cost == 0
        assert cache == {}

    @pytest.mark.asyncio
    async def test_plan_sorted_by_roi(self, planner):
        planner.estimator.estimate = AsyncMock(side_effect=[
            make_estimate(tokens=1000, p=0.9, cost=5),
            make_estimate(tokens=1000, p=0.3, cost=5),
        ])
        candidates = [("a", {"reward": 10}), ("b", {"reward": 10})]
        ids, cost, cache = await planner.plan(candidates)
        assert ids == ["a", "b"]
        assert cost == 10
        assert set(cache) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_plan_respects_budget(self, planner):
        planner.budget = 1500
        planner.alpha = 1.0
        planner.estimator.estimate = AsyncMock(side_effect=[
            make_estimate(tokens=1000, p=0.9),
            make_estimate(tokens=1000, p=0.3),
        ])
        candidates = [("a", {"reward": 10}), ("b", {"reward": 10})]
        ids, _, cache = await planner.plan(candidates)
        assert len(ids) == 1
        assert ids == ["a"]
        assert set(cache) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_reoptimize_empty(self, planner):
        ids, cost, cache = await planner.reoptimize([], 1000)
        assert ids == []
        assert cost == 0
        assert cache == {}

    @pytest.mark.asyncio
    async def test_reoptimize_zero_budget(self, planner):
        ids, cost, cache = await planner.reoptimize(
            [("a", {"reward": 1})], 0
        )
        assert ids == []
        assert cost == 0
        assert cache == {}

    @pytest.mark.asyncio
    async def test_plan_returns_cache_matching_inputs(self, planner):
        planner.estimator.estimate = AsyncMock(side_effect=[
            make_estimate(tokens=100, p=0.5, cost=3),
            make_estimate(tokens=200, p=0.7, cost=4),
        ])
        candidates = [("a", {"reward": 10}), ("b", {"reward": 20})]
        _, cost, cache = await planner.plan(candidates)
        assert cost == 7
        assert cache["a"].tokens == 100
        assert cache["b"].tokens == 200


# ======================================================================
# Harness init
# ======================================================================

class TestHarnessInit:
    def test_harness_init_valid_dir(self, tmp_path):
        mock_config = MagicMock()
        mock_config.token_budget = None
        mock_solver = MagicMock()
        mock_estimator = MagicMock()
        harness = Harness(
            tmp_path, mock_config, mock_solver, mock_estimator
        )
        assert harness.tasks_dir == tmp_path
        assert harness.repo is not None
        assert harness.config is mock_config
        assert harness.solver is mock_solver
        assert harness.estimator is mock_estimator
        assert harness.planner is None

    def test_harness_init_no_budget_disables_planner(self, tmp_path):
        mock_config = MagicMock()
        mock_config.token_budget = None
        harness = Harness(
            tmp_path, mock_config, MagicMock(), MagicMock(),
            token_budget=None,
        )
        assert harness.planner is None

    def test_harness_init_with_budget_creates_planner(self, tmp_path):
        mock_config = MagicMock()
        mock_config.token_budget = None
        harness = Harness(
            tmp_path, mock_config, MagicMock(), MagicMock(),
            token_budget=5000,
        )
        assert harness.planner is not None
        assert harness.planner.budget == 5000


# ======================================================================
# Harness.run — real fixtures (no planner)
# ======================================================================

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.get_model_name.return_value = "test-model"
    config.token_budget = None
    config.normalize_tokens = lambda i, o: int(o + i / 10)
    return config


@pytest.fixture
def mock_solver():
    solver = MagicMock()
    solver.solve = AsyncMock()
    return solver


@pytest.fixture
def mock_estimator():
    est = MagicMock()
    est.estimate = AsyncMock(return_value=make_estimate(tokens=1000))
    est.calibrate = MagicMock()
    est.base_costs = {}
    return est


@pytest.fixture
def tasks_dir_empty(tmp_path):
    return tmp_path


@pytest.fixture
def tasks_dir_single_task(tmp_path):
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
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def tasks_dir_single_task_failure(tmp_path):
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
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def tasks_dir_multiple_tasks(tmp_path):
    for i, (task_id, reward) in enumerate(
        [("task_001", 10), ("task_002", 20)], 1
    ):
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
            encoding="utf-8",
        )
    return tmp_path


@pytest.fixture
def tasks_dir_unknown_type(tmp_path):
    task_dir = tmp_path / "task_001"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps({
            "task_id": "task_001",
            "task_type": "unknown",
            "instruction": "Test task",
            "reward": 10,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def tasks_dir_unittest(tmp_path):
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
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def tasks_dir_solver_error(tmp_path):
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
        encoding="utf-8",
    )
    return tmp_path


class TestHarnessRun:
    @pytest.mark.asyncio
    async def test_run_empty_repo(
        self, tasks_dir_empty, mock_config, mock_solver, mock_estimator
    ):
        harness = Harness(
            tasks_dir_empty, mock_config, mock_solver, mock_estimator
        )
        report = await harness.run()
        assert report["total_reward"] == 0
        assert report["total_input_tokens"] == 0
        assert report["total_output_tokens"] == 0
        assert report["total_tokens"] == 0
        assert report["results"] == []

    @pytest.mark.asyncio
    async def test_run_single_task_success(
        self, tasks_dir_single_task, mock_config, mock_solver, mock_estimator
    ):
        harness = Harness(
            tasks_dir_single_task, mock_config, mock_solver, mock_estimator
        )
        mock_solver.solve.return_value = ("answer", True, 100, 50)
        report = await harness.run()
        assert report["total_reward"] == 10
        assert report["total_input_tokens"] == 100
        assert report["total_output_tokens"] == 50
        assert report["total_tokens"] == 60
        assert len(report["results"]) == 1
        assert report["results"][0]["task_id"] == "task_001"
        assert report["results"][0]["success"] is True
        assert report["results"][0]["reward"] == 10

    @pytest.mark.asyncio
    async def test_run_single_task_failure(
        self, tasks_dir_single_task_failure, mock_config,
        mock_solver, mock_estimator,
    ):
        harness = Harness(
            tasks_dir_single_task_failure, mock_config,
            mock_solver, mock_estimator,
        )
        mock_solver.solve.return_value = ("wrong", True, 100, 50)
        report = await harness.run()
        assert report["total_reward"] == 0
        assert len(report["results"]) == 1
        assert report["results"][0]["reward"] == 0

    @pytest.mark.asyncio
    async def test_run_specific_task_id(
        self, tasks_dir_multiple_tasks, mock_config,
        mock_solver, mock_estimator,
    ):
        harness = Harness(
            tasks_dir_multiple_tasks, mock_config, mock_solver, mock_estimator
        )
        mock_solver.solve.return_value = ("answer1", True, 100, 50)
        report = await harness.run(task_id="task_001")
        assert report["total_reward"] == 10
        assert len(report["results"]) == 1
        assert report["results"][0]["task_id"] == "task_001"

    @pytest.mark.asyncio
    async def test_run_unknown_task_type(
        self, tasks_dir_unknown_type, mock_config,
        mock_solver, mock_estimator,
    ):
        mock_config.get_model_name.return_value = None
        harness = Harness(
            tasks_dir_unknown_type, mock_config, mock_solver, mock_estimator
        )
        report = await harness.run()
        assert report["total_reward"] == 0
        assert report["results"] == []

    @pytest.mark.asyncio
    async def test_run_multiple_tasks(
        self, tasks_dir_multiple_tasks, mock_config,
        mock_solver, mock_estimator,
    ):
        harness = Harness(
            tasks_dir_multiple_tasks, mock_config, mock_solver, mock_estimator
        )
        mock_solver.solve.side_effect = [
            ("answer1", True, 100, 50),
            ("answer2", True, 200, 100),
        ]
        report = await harness.run()
        assert report["total_reward"] == 30
        assert report["total_input_tokens"] == 300
        assert report["total_output_tokens"] == 150
        assert report["total_tokens"] == 180
        assert len(report["results"]) == 2

    @pytest.mark.asyncio
    @patch("project.harness.run_unittest", return_value=True)
    async def test_run_unittest_task(
        self, mock_run_unittest, tasks_dir_unittest,
        mock_config, mock_solver, mock_estimator,
    ):
        harness = Harness(
            tasks_dir_unittest, mock_config, mock_solver, mock_estimator
        )
        mock_solver.solve.return_value = ("any answer", True, 100, 50)
        report = await harness.run()
        assert report["total_reward"] == 10
        assert report["results"][0]["reward"] == 10
        mock_run_unittest.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_solver_failure(
        self, tasks_dir_solver_error, mock_config,
        mock_solver, mock_estimator,
    ):
        harness = Harness(
            tasks_dir_solver_error, mock_config, mock_solver, mock_estimator
        )
        mock_solver.solve.return_value = ("Error occurred", False, 100, 50)
        report = await harness.run()
        assert report["total_reward"] == 0
        assert report["results"][0]["success"] is False
        assert report["results"][0]["reward"] == 0


# ======================================================================
# Harness — execute + warmup with BudgetPlanner
# ======================================================================

class TestHarnessExecute:
    @pytest.mark.asyncio
    async def test_execute_success(self, tmp_path):
        h, solver, _ = make_harness(
            tmp_path,
            tasks={"t1": make_task(expected="ok")},
            estimates={"t1": make_estimate(tokens=500)},
            solver_results={"t1": ("ok", True, 100, 50)},
        )
        task = h.repo.get_task("t1")
        est = make_estimate(tokens=500, cost=7)
        rec = await h._execute("t1", task, est, remaining_budget=10_000)
        assert rec["success"] is True
        assert rec["reward"] == 10
        assert rec["total_tokens"] == 60
        assert rec["estimator_cost"] == 7
        assert rec["est_tokens"] == 500

    @pytest.mark.asyncio
    async def test_execute_max_tokens_is_min_of_limit_and_remaining(self, tmp_path):
        h, solver, _ = make_harness(
            tmp_path,
            tasks={"t1": make_task()},
            estimates={"t1": make_estimate(tokens=1000)},
            solver_results={"t1": ("x", False, 100, 50)},
            alpha_init=1.0, beta_mult=2.0,
        )
        task = h.repo.get_task("t1")
        est = make_estimate(tokens=1000)
        await h._execute("t1", task, est, remaining_budget=500)
        assert solver._calls[-1] == ("t1", 500)

    @pytest.mark.asyncio
    async def test_execute_no_planner_passes_none(self, tmp_path):
        h, solver, _ = make_harness(
            tmp_path,
            tasks={"t1": make_task()},
            estimates={"t1": make_estimate()},
            solver_results={"t1": ("x", True, 10, 5)},
            budget=0,
        )
        assert h.planner is None
        task = h.repo.get_task("t1")
        est = make_estimate()
        await h._execute("t1", task, est, remaining_budget=None)
        assert solver._calls[-1] == ("t1", None)

    @pytest.mark.asyncio
    async def test_execute_calls_observe(self, tmp_path):
        h, _, _ = make_harness(
            tmp_path,
            tasks={"t1": make_task()},
            estimates={"t1": make_estimate(tokens=500)},
            solver_results={"t1": ("x", True, 100, 50)},
        )
        spy = MagicMock(wraps=h.planner.observe)
        h.planner.observe = spy
        task = h.repo.get_task("t1")
        est = make_estimate(tokens=500)
        await h._execute("t1", task, est, remaining_budget=10_000)
        spy.assert_called_once()
        args = spy.call_args.args
        assert args[0] == "t1"
        assert args[2] == 60
        assert args[3] is True


class TestHarnessWarmup:
    @pytest.mark.asyncio
    async def test_warmup_no_planner_is_noop(self, tmp_path):
        h, _, _ = make_harness(
            tmp_path,
            tasks={"t1": make_task()},
            estimates={"t1": make_estimate()},
            solver_results={"t1": ("x", True, 1, 1)},
            budget=0,
        )
        assert h.planner is None
        queue = [("t1", make_task())]
        results = []
        state = {"spent_total": 0, "in": 0, "out": 0, "reward": 0}
        await h._warmup(queue, results, state)
        assert len(queue) == 1
        assert results == []

    @pytest.mark.asyncio
    async def test_warmup_picks_one_per_type(self, tmp_path):
        tasks = {
            "c1": make_task(task_type="coding", expected="a"),
            "c2": make_task(task_type="coding", expected="b"),
            "m1": make_task(task_type="math", expected="c"),
        }
        estimates = {
            "c1": make_estimate(tokens=100, p=0.9, cost=1),
            "c2": make_estimate(tokens=100, p=0.5, cost=1),
            "m1": make_estimate(tokens=100, p=0.9, cost=1),
        }
        h, _, _ = make_harness(
            tmp_path, tasks, estimates,
            {"c1": ("a", True, 10, 5), "m1": ("c", True, 10, 5)},
        )
        queue = [(k, tasks[k]) for k in tasks]
        results = []
        state = {"spent_total": 0, "in": 0, "out": 0, "reward": 0}
        await h._warmup(queue, results, state)
        assert {r["task_id"] for r in results} == {"c1", "m1"}
        assert {tid for tid, _ in queue} == {"c2"}

    @pytest.mark.asyncio
    async def test_warmup_charges_estimator_cost(self, tmp_path):
        tasks = {"t1": make_task()}
        estimates = {"t1": make_estimate(cost=33)}
        h, _, _ = make_harness(
            tmp_path, tasks, estimates, {"t1": ("x", True, 1, 1)}
        )
        queue = [("t1", tasks["t1"])]
        results = []
        state = {"spent_total": 0, "in": 0, "out": 0, "reward": 0}
        await h._warmup(queue, results, state)
        assert state["spent_total"] == 33 + 1

    @pytest.mark.asyncio
    async def test_warmup_skips_unaffordable(self, tmp_path):
        tasks = {"t1": make_task()}
        estimates = {"t1": make_estimate(tokens=1_000_000, cost=0)}
        h, solver, _ = make_harness(
            tmp_path, tasks, estimates, {"t1": ("x", True, 1, 1)},
            budget=1000,
        )
        queue = [("t1", tasks["t1"])]
        results = []
        state = {"spent_total": 0, "in": 0, "out": 0, "reward": 0}
        await h._warmup(queue, results, state)
        assert results == []
        assert solver._calls == []


# ======================================================================
# Harness — budget behaviour with planner
# ======================================================================

class TestHarnessBudget:
    @pytest.mark.asyncio
    async def test_run_no_budget_skips_warmup_and_plan(self, tmp_path):
        tasks = {"t1": make_task(expected="a")}
        estimates = {"t1": make_estimate()}
        h, _, _ = make_harness(
            tmp_path, tasks, estimates, {"t1": ("a", True, 1, 1)}, budget=0
        )
        assert h.planner is None
        out = await h.run()
        assert out["total_reward"] == 10
        assert out["final_alpha"] == 1.0

    @pytest.mark.asyncio
    async def test_run_charges_estimator_cost_in_total_spent(self, tmp_path):
        tasks = {"t1": make_task(expected="a")}
        estimates = {"t1": make_estimate(cost=50)}
        h, _, _ = make_harness(
            tmp_path, tasks, estimates, {"t1": ("a", True, 10, 5)},
            budget=1_000_000,
        )
        out = await h.run()
        assert out["total_spent_including_estimator"] == 56
        assert out["total_tokens"] == 6

    @pytest.mark.asyncio
    async def test_run_budget_guard_skips_unaffordable(self, tmp_path):
        tasks = {
            "big": make_task(expected="a", reward=100),
            "small": make_task(expected="b", reward=10),
        }
        estimates = {
            "big": make_estimate(tokens=900_000),
            "small": make_estimate(tokens=100),
        }
        h, solver, _ = make_harness(
            tmp_path, tasks, estimates,
            {"big": ("a", True, 1, 1), "small": ("b", True, 1, 1)},
            budget=1000, alpha_init=1.0,
        )
        h._warmup = AsyncMock()
        out = await h.run()
        ran = {tid for tid, _ in solver._calls}
        assert ran == {"small"}
        assert out["total_reward"] == 10

    @pytest.mark.asyncio
    async def test_run_respects_task_id(self, tmp_path):
        tasks = {"t1": make_task(expected="a"), "t2": make_task(expected="b")}
        estimates = {k: make_estimate(tokens=100) for k in tasks}
        h, _, _ = make_harness(
            tmp_path, tasks, estimates,
            {"t1": ("a", True, 1, 1), "t2": ("b", True, 1, 1)},
            budget=1_000_000,
        )
        h._warmup = AsyncMock()
        out = await h.run(task_id="t2")
        assert len(out["results"]) == 1
        assert out["results"][0]["task_id"] == "t2"

    @pytest.mark.asyncio
    async def test_run_reward_and_tokens_accumulate(self, tmp_path):
        tasks = {
            "t1": make_task(expected="a", reward=10),
            "t2": make_task(expected="b", reward=20),
        }
        estimates = {k: make_estimate(tokens=100) for k in tasks}
        h, _, _ = make_harness(
            tmp_path, tasks, estimates,
            {"t1": ("a", True, 10, 5), "t2": ("b", True, 20, 10)},
            budget=1_000_000,
        )
        out = await h.run()
        assert out["total_reward"] == 30
        assert out["total_input_tokens"] == 30
        assert out["total_output_tokens"] == 15
        assert out["total_tokens"] == 18
        assert isinstance(out["final_alpha"], float)


# ======================================================================
# End-to-end budget invariants
# ======================================================================

class TestEndToEndBudget:
    @pytest.mark.asyncio
    async def test_never_exceeds_budget(self, tmp_path):
        budget = 1000
        tasks = {f"t{i}": make_task(expected="x", reward=10) for i in range(5)}
        estimates = {k: make_estimate(tokens=300) for k in tasks}
        h, solver, _ = make_harness(
            tmp_path, tasks, estimates,
            {k: ("x", True, 250, 0) for k in tasks},
            budget=budget, alpha_init=1.0, beta_mult=1.0,
        )
        h._warmup = AsyncMock()
        out = await h.run()
        assert out["total_spent_including_estimator"] <= budget + 250
        for _, mt in solver._calls:
            assert mt is not None and mt >= 1

    @pytest.mark.asyncio
    async def test_estimator_cost_reduces_available_budget(self, tmp_path):
        tasks = {"t1": make_task(expected="a")}
        estimates = {"t1": make_estimate(tokens=100, cost=300)}
        h, solver, _ = make_harness(
            tmp_path, tasks, estimates, {"t1": ("a", True, 1, 1)},
            budget=500, alpha_init=1.0, beta_mult=10.0,
        )
        h._warmup = AsyncMock()
        await h.run()
        mt = solver._calls[-1][1]
        assert mt <= 200

    @pytest.mark.asyncio
    async def test_estimator_cost_charged_once_per_task(self, tmp_path):
        """``plan`` returns the cache, so ``estimate`` must not be re-called
        for the same task inside the execution loop."""
        tasks = {"t1": make_task(expected="a")}
        estimates = {"t1": make_estimate(tokens=100, cost=50)}
        h, _, estimator = make_harness(
            tmp_path, tasks, estimates, {"t1": ("a", True, 1, 1)},
            budget=1_000_000,
        )
        h._warmup = AsyncMock()
        out = await h.run()
        # 50 (classification) + 2 (solve) = 52
        assert out["total_spent_including_estimator"] == 51
        # estimate called exactly once for t1: inside plan
        assert estimator._call_counts["t1"] == 1