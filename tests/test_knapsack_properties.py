# tests/test_knapsack_properties.py
from hypothesis import given, strategies as st
from knapsack import KnapsackSolver

ids  = st.text(min_size=1, max_size=8)
toks = st.integers(min_value=0, max_value=10**6)
rew  = st.floats(min_value=0, max_value=10**6, allow_nan=False, allow_infinity=False)
ps   = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(st.lists(st.tuples(ids, toks, rew, ps), max_size=50))
def test_pop_never_returns_more_than_inserted(items):
    s = KnapsackSolver()
    for i, (tid, t, r, p) in enumerate(items):
        s.add_task(tid, tokens=t, reward=r, p_solvable=p)
    unique = len({tid for tid, *_ in items})
    popped = []
    while True:
        x = s.pop_tastiest()
        if x is None:
            break
        popped.append(x)
    assert len(popped) == unique
    assert len(set(popped)) == unique


@given(st.lists(st.tuples(ids, toks, rew, ps), max_size=30), st.integers(min_value=0, max_value=10**6))
def test_fit_never_exceeds_budget(items, budget):
    s = KnapsackSolver()
    for tid, t, r, p in items:
        s.add_task(tid, tokens=t, reward=r, p_solvable=p)
    chosen = s.fit(budget)
    total = sum(s.get(tid).tokens for tid in chosen)
    assert total <= budget


@given(st.lists(st.tuples(ids, toks, rew, ps), max_size=30))
def test_pop_order_is_non_increasing_roi(items):
    s = KnapsackSolver()
    for tid, t, r, p in items:
        s.add_task(tid, tokens=t, reward=r, p_solvable=p)
    rois = []
    while (tid := s.pop_tastiest()) is not None:
        # сохраняем roi до удаления — уже удалили, поэтому храним заранее
        pass
    # вариант чище — запоминаем перед pop:
    s2 = KnapsackSolver()
    for tid, t, r, p in items:
        s2.add_task(tid, tokens=t, reward=r, p_solvable=p)
    seq = []
    while (tid := s2.peek_tastiest()) is not None:
        seq.append(s2.get(tid).roi)
        s2.pop_tastiest()
    assert seq == sorted(seq, reverse=True)
