from __future__ import annotations

from typing import Optional
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project.knapsack_solver import KnapsackSolver, Task
from project.task_repo import TaskRepo


# ---------- fake TaskRepo ----------

class FakeTaskRepo(TaskRepo):
    """Minimal in-memory TaskRepo for tests."""

    def __init__(self, tasks: Optional[list[dict]] = None) -> None:
        self._tasks: dict[str, dict] = {}
        self._task_dirs: dict[str, Path] = {}
        for t in tasks or []:
            self._tasks[t["task_id"]] = t

    def set_reward(self, task_id: str, reward: float) -> None:
        if task_id in self._tasks:
            self._tasks[task_id]["reward"] = reward
        else:
            self._tasks[task_id] = {"task_id": task_id, "reward": reward}

    def get_task(self, task_id: str) -> Optional[dict]:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[dict]:
        return list(self._tasks.values())


@pytest.fixture
def repo() -> FakeTaskRepo:
    return FakeTaskRepo()


# ---------- hypothesis strategies ----------

ids = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=12,
)
toks = st.integers(min_value=0, max_value=10_000)
rews = st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)
ps = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ---------- add_task ----------

class TestAddTask:
    def test_add_task_basic(self, repo):
        repo.set_reward("a", 5.0)
        s = KnapsackSolver(repo)
        t = s.add_task("a", 100)
        assert t.task_id == "a"
        assert t.tokens == 100
        assert t.reward == 5.0
        assert t.p_solvable == 1.0

    def test_add_task_default_reward(self):
        s = KnapsackSolver(FakeTaskRepo(), default_reward=2.5)
        t = s.add_task("x", 10)
        assert t.reward == 2.5

    def test_add_task_overwrite(self, repo):
        repo.set_reward("a", 5.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)
        repo.set_reward("a", 7.0)
        t = s.add_task("a", 200)
        assert t.tokens == 200
        assert t.reward == 7.0
        assert t.version == 2

    def test_add_task_negative_tokens_raises(self, repo):
        s = KnapsackSolver(repo)
        with pytest.raises(ValueError):
            s.add_task("a", -1)

    def test_add_task_unknown_id_uses_default(self):
        s = KnapsackSolver(FakeTaskRepo(), default_reward=3.0)
        t = s.add_task("missing", 50)
        assert t.reward == 3.0


# ---------- pop_tastiest ----------

class TestPopTastiest:
    def test_pop_empty(self, repo):
        s = KnapsackSolver(repo)
        assert s.pop_tastiest() is None

    def test_pop_returns_best_roi(self, repo):
        repo.set_reward("a", 10.0)
        repo.set_reward("b", 10.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)   # roi = 0.1
        s.add_task("b", 10)    # roi = 1.0
        assert s.pop_tastiest() == "b"
        assert s.pop_tastiest() == "a"
        assert s.pop_tastiest() is None

    def test_pop_removes_task(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 10)
        s.pop_tastiest()
        assert "a" not in s


# ---------- update_estimate ----------

class TestUpdateEstimate:
    def test_update_tokens(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)
        s.update_estimate("a", tokens=50)
        t = s.get("a")
        assert t is not None
        assert t.tokens == 50

    def test_update_p_solvable(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)
        s.update_estimate("a", p_solvable=0.3)
        t = s.get("a")
        assert t is not None
        assert t.p_solvable == 0.3

    def test_update_unknown_returns_none(self, repo):
        s = KnapsackSolver(repo)
        assert s.update_estimate("missing", tokens=10) is None

    def test_update_invalid_p_solvable_raises(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 10)
        with pytest.raises(ValueError):
            s.update_estimate("a", p_solvable=1.5)

    def test_update_negative_tokens_raises(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 10)
        with pytest.raises(ValueError):
            s.update_estimate("a", tokens=-1)


# ---------- record_attempt ----------

class TestRecordAttempt:
    def test_success_increases_p_solvable(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)
        s.update_estimate("a", p_solvable=0.5)
        s.record_attempt("a", success=True, real_tokens=100)
        t = s.get("a")
        assert t is not None
        assert t.p_solvable > 0.5

    def test_failure_decreases_p_solvable(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)
        s.update_estimate("a", p_solvable=0.5)
        s.record_attempt("a", success=False, real_tokens=100)
        t = s.get("a")
        assert t is not None
        assert t.p_solvable < 0.5

    def test_tokens_smoothed(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)
        s.record_attempt("a", success=True, real_tokens=200)
        # 0.5*100 + 0.5*200 = 150
        t = s.get("a")
        assert t is not None
        assert t.tokens == 150

    def test_attempts_counter(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)
        s.record_attempt("a", success=True, real_tokens=100)
        s.record_attempt("a", success=False, real_tokens=100)
        t = s.get("a")
        assert t is not None
        assert t.attempts == 2

    def test_unknown_task_returns_none(self, repo):
        s = KnapsackSolver(repo)
        assert s.record_attempt("missing", success=True, real_tokens=10) is None


# ---------- can_start ----------

class TestCanStart:
    def test_can_start_within_budget(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)
        # safety=1.3 => нужно 130
        assert s.can_start("a", 200) is True
        assert s.can_start("a", 100) is False

    def test_can_start_unknown_task(self, repo):
        s = KnapsackSolver(repo)
        assert s.can_start("missing", 1000) is False

    def test_can_start_zero_tokens(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 0)
        assert s.can_start("a", 0) is True

    def test_can_start_zero_p_solvable(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)
        s.update_estimate("a", p_solvable=0.0)
        assert s.can_start("a", 10_000) is False


# ---------- fit ----------

class TestFit:
    def test_fit_respects_budget(self, repo):
        repo.set_reward("a", 10.0)
        repo.set_reward("b", 10.0)
        repo.set_reward("c", 10.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)
        s.add_task("b", 200)
        s.add_task("c", 300)
        chosen = s.fit(250)
        total = 0
        for tid in chosen:
            t = s.get(tid)
            assert t is not None
            total += t.tokens
        assert total <= 250

    def test_fit_excludes(self, repo):
        repo.set_reward("a", 10.0)
        repo.set_reward("b", 10.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 10)
        s.add_task("b", 10)
        chosen = s.fit(1000, exclude={"a"})
        assert "a" not in chosen
        assert "b" in chosen

    def test_fit_skips_zero_p_solvable(self, repo):
        repo.set_reward("a", 10.0)
        repo.set_reward("b", 10.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 10)
        s.add_task("b", 10)
        s.update_estimate("a", p_solvable=0.0)
        chosen = s.fit(1000)
        assert "a" not in chosen

    def test_fit_prefers_high_roi(self, repo):
        repo.set_reward("a", 1.0)
        repo.set_reward("b", 100.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 100)   # roi = 0.01
        s.add_task("b", 10)    # roi = 10
        chosen = s.fit(10)
        assert chosen == ["b"]


# ---------- property-based tests ----------

class TestProperties:
    @given(st.lists(st.tuples(ids, toks, rews, ps), max_size=50))
    @settings(max_examples=50)
    def test_pop_never_returns_more_than_inserted(self, items):
        repo = FakeTaskRepo()
        for tid, _, r, _ in items:
            repo.set_reward(tid, r)
        s = KnapsackSolver(repo)
        for tid, t, _, _ in items:
            s.add_task(tid, tokens=t)
        unique = len({tid for tid, *_ in items})

        popped = []
        while True:
            x = s.pop_tastiest()
            if x is None:
                break
            popped.append(x)

        assert len(popped) == unique
        assert len(set(popped)) == unique

    @given(
        st.lists(st.tuples(ids, toks, rews, ps), max_size=30),
        st.integers(min_value=0, max_value=10**6),
    )
    @settings(max_examples=50)
    def test_fit_never_exceeds_budget(self, items, budget):
        repo = FakeTaskRepo()
        for tid, _, r, _ in items:
            repo.set_reward(tid, r)
        s = KnapsackSolver(repo)
        for tid, t, _, _ in items:
            s.add_task(tid, tokens=t)
        chosen = s.fit(budget)
        total = 0
        for tid in chosen:
            t = s.get(tid)
            assert t is not None
            total += t.tokens
        assert total <= budget

    @given(st.lists(st.tuples(ids, toks, rews, ps), max_size=30))
    @settings(max_examples=50)
    def test_pop_order_is_non_increasing_roi(self, items):
        repo = FakeTaskRepo()
        for tid, _, r, _ in items:
            repo.set_reward(tid, r)
        s = KnapsackSolver(repo)
        for tid, t, _, _ in items:
            s.add_task(tid, tokens=t)

        seq = []
        while (tid := s.peek_tastiest()) is not None:
            t = s.get(tid)
            assert t is not None
            seq.append(t.roi)
            s.pop_tastiest()

        assert seq == sorted(seq, reverse=True)


# ---------- misc ----------

class TestMisc:
    def test_len_and_contains(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        assert len(s) == 0
        s.add_task("a", 10)
        assert len(s) == 1
        assert "a" in s
        assert "b" not in s

    def test_remove(self, repo):
        repo.set_reward("a", 1.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 10)
        removed = s.remove("a")
        assert removed is not None
        assert "a" not in s
        assert s.remove("a") is None

    def test_iter(self, repo):
        repo.set_reward("a", 1.0)
        repo.set_reward("b", 2.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 10)
        s.add_task("b", 20)
        tasks = list(s)
        assert len(tasks) == 2
        assert all(isinstance(t, Task) for t in tasks)

    def test_expected_value_total(self, repo):
        repo.set_reward("a", 10.0)
        repo.set_reward("b", 20.0)
        s = KnapsackSolver(repo)
        s.add_task("a", 10)
        s.add_task("b", 10)
        s.update_estimate("a", p_solvable=0.5)
        # 10*0.5 + 20*1.0 = 25
        assert s.expected_value_total() == 25.0