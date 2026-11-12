# tests/test_knapsack_estimates.py
import pytest
from knapsack import KnapsackSolver


def test_update_estimate_changes_only_given_fields():
    s = KnapsackSolver()
    s.add_task("a", tokens=100, reward=10, p_solvable=0.9)
    s.update_estimate("a", tokens=50)
    t = s.get("a")
    assert t.tokens == 50
    assert t.reward == 10        # не тронуто
    assert t.p_solvable == 0.9   # не тронуто


def test_update_estimate_missing_returns_none():
    assert KnapsackSolver().update_estimate("nope", tokens=1) is None


def test_record_attempt_success_moves_p_up():
    s = KnapsackSolver()
    s.add_task("a", tokens=100, p_solvable=0.5)
    s.record_attempt("a", success=True, real_tokens=80, alpha=0.5)
    t = s.get("a")
    assert t.p_solvable > 0.5
    assert t.tokens < 100        # факт дешевле оценки → средняя упала
    assert t.attempts == 1


def test_record_attempt_failure_moves_p_down():
    s = KnapsackSolver()
    s.add_task("a", tokens=100, p_solvable=0.9)
    s.record_attempt("a", success=False, real_tokens=0, alpha=0.5)
    assert s.get("a").p_solvable < 0.9


def test_can_start_respects_safety_margin():
    s = KnapsackSolver()
    s.add_task("a", tokens=100, p_solvable=1.0)
    assert s.can_start("a", remaining_budget=130, safety=1.3) is True
    assert s.can_start("a", remaining_budget=129, safety=1.3) is False
    assert s.can_start("a", remaining_budget=100, safety=1.0) is True


def test_can_start_false_for_zero_probability():
    s = KnapsackSolver()
    s.add_task("a", tokens=10, p_solvable=0.0)
    assert s.can_start("a", remaining_budget=10**9) is False


def test_fit_respects_budget_and_roi_order():
    s = KnapsackSolver()
    s.add_task("a", tokens=50,  reward=5,  p_solvable=1.0)   # ROI=0.10
    s.add_task("b", tokens=100, reward=30, p_solvable=1.0)   # ROI=0.30
    s.add_task("c", tokens=200, reward=100, p_solvable=1.0)  # ROI=0.50
    chosen = s.fit(budget=150)
    assert chosen == ["b", "a"]                              # c (200) не влез бы? смотрим ниже

def test_fit_skips_items_that_do_not_fit():
    s = KnapsackSolver()
    s.add_task("expensive", tokens=500, reward=100, p_solvable=1.0)  # ROI=0.2
    s.add_task("cheap",     tokens=10,  reward=1,   p_solvable=1.0)  # ROI=0.1
    chosen = s.fit(budget=100)
    assert chosen == ["cheap"]                               # expensive не влез
