# tests/test_knapsack_contract.py
import pytest
from knapsack import KnapsackSolver


def test_pop_tastiest_returns_none_on_empty():
    s = KnapsackSolver()
    assert s.pop_tastiest() is None


def test_add_task_then_pop_returns_that_id():
    s = KnapsackSolver()
    s.add_task("a", tokens=10, reward=1.0, p_solvable=1.0)
    assert s.pop_tastiest() == "a"
    assert len(s) == 0                      # задача удалена из пула


def test_rewrite_same_id_overwrites_tokens_not_duplicates():
    """Ключевое требование ТЗ: id существует → число токенов перезаписывается."""
    s = KnapsackSolver()
    s.add_task("a", tokens=10)
    s.add_task("a", tokens=999)
    assert len(s) == 1                      # не задублировалось
    assert s.get("a").tokens == 999         # перезаписано


def test_pop_picks_highest_roi_not_most_tokens():
    """«Самая вкусная» = max ROI, а не max tokens (критично для конкурса)."""
    s = KnapsackSolver()
    s.add_task("big",   tokens=1000, reward=100, p_solvable=1.0)   # ROI=0.1
    s.add_task("small", tokens=10,   reward=5,   p_solvable=1.0)   # ROI=0.5
    assert s.pop_tastiest() == "small"
    assert s.pop_tastiest() == "big"
    assert s.pop_tastiest() is None


def test_pop_respects_p_solvable():
    s = KnapsackSolver()
    s.add_task("sure",   tokens=10, reward=5, p_solvable=1.0)   # EV=5,  ROI=0.5
    s.add_task("gamble", tokens=10, reward=100, p_solvable=0.01) # EV=1, ROI=0.1
    assert s.pop_tastiest() == "sure"


def test_tie_broken_by_insertion_order():
    """При равном ROI — задача, добавленная раньше."""
    s = KnapsackSolver()
    s.add_task("first",  tokens=10, reward=1.0, p_solvable=1.0)
    s.add_task("second", tokens=10, reward=1.0, p_solvable=1.0)
    assert s.pop_tastiest() == "first"
    assert s.pop_tastiest() == "second"


@pytest.mark.parametrize("bad_tokens", [-1, -100])
def test_negative_tokens_rejected(bad_tokens):
    s = KnapsackSolver()
    with pytest.raises(ValueError):
        s.add_task("a", tokens=bad_tokens)


@pytest.mark.parametrize("bad_p", [-0.1, 1.1, 2.0])
def test_invalid_p_solvable_rejected(bad_p):
    s = KnapsackSolver()
    with pytest.raises(ValueError):
        s.add_task("a", tokens=10, p_solvable=bad_p)
