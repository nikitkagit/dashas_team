import math
import re
import sys
import subprocess
from pathlib import Path
from typing import Optional

from project.config import Config
from project.task_estimator import Estimate, TaskEstimator
from project.task_repo import TaskRepo
from project.task_solver import TaskSolver


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def normalize_int(s: str) -> Optional[int]:
    """Normalize a string to an integer, ignoring non-digit characters."""
    s = s.strip()
    s = re.sub(r"[^\d\-+]", "", s.split("\n")[0])
    try:
        return int(s)
    except ValueError:
        return None


def run_unittest(task_dir: Path, timeout: int = 60) -> bool:
    """Run the task's test suite inside the workspace and report pass/fail.

    Looks for a ``tests/`` directory or ``test_*.py`` files inside the
    workspace and runs ``pytest`` if available, otherwise ``unittest``.
    Returns ``True`` only if the runner exits with code 0.
    """
    workspace = (task_dir / "workspace").resolve()
    if not workspace.is_dir():
        return False

    has_tests = (workspace / "tests").is_dir() or any(
        workspace.rglob("test_*.py")
    )
    if not has_tests:
        return False

    env = {
        **__import__("os").environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(workspace),
    }

    try:
        import pytest  # noqa: F401
        cmd = [sys.executable, "-m", "pytest", "-q", str(workspace)]
    except ImportError:
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(workspace)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def check_answer(
    task: dict,
    answer_text: str,
    task_dir: Optional[Path] = None,
) -> bool:
    """Check if the agent's answer matches the expected result.

    For ``check == "unittest"`` the workspace test suite is executed;
    ``task_dir`` is required in that case.
    """
    check_type = task.get("check", "exact_str")
    expected = str(task.get("expected", "")).strip()
    actual = (answer_text or "").strip()

    if check_type == "unittest":
        if task_dir is None:
            return False
        return run_unittest(task_dir)

    if check_type == "exact_int":
        exp_int = normalize_int(expected)
        act_int = normalize_int(actual)
        return exp_int is not None and act_int is not None and exp_int == act_int
    if check_type == "exact_str":
        return actual.lower() == expected.lower()
    if check_type == "contains":
        return expected.lower() in actual.lower()
    return actual == expected


# ---------------------------------------------------------------------------
# Knapsack helpers
# ---------------------------------------------------------------------------

def _knapsack_01(
    weights: list[int],
    values: list[float],
    capacity: int,
) -> list[int]:
    """Classic 0/1 knapsack. Returns indices of chosen items.

    DP over capacity: O(n * capacity). Capacity is scaled down if it
    exceeds ``max_cap`` to keep memory bounded.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return []

    scale = 1
    max_cap = 200_000
    if capacity > max_cap:
        scale = math.ceil(capacity / max_cap)
    cap = max(1, capacity // scale)

    dp = [0.0] * (cap + 1)
    take = [[False] * (cap + 1) for _ in range(n)]
    for i in range(n):
        w = max(1, weights[i] // scale)
        v = values[i]
        for c in range(cap, w - 1, -1):
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
                take[i][c] = True

    chosen: list[int] = []
    c = cap
    for i in range(n - 1, -1, -1):
        if c >= 0 and take[i][c]:
            chosen.append(i)
            c -= max(1, weights[i] // scale)
    return chosen


def _quantile(xs: list[float], q: float) -> float:
    """Empirical quantile with linear interpolation."""
    if not xs:
        return 1.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo)


# ---------------------------------------------------------------------------
# Budget planner
# ---------------------------------------------------------------------------

class BudgetPlanner:
    """Adaptive planner on top of ``TaskEstimator``.

    Holds only the planning-specific state:

    * ``alpha`` — safety factor on top of ``estimate.tokens``;
    * ``beta_mult`` — per-task limit multiplier for ``solve(max_tokens=...)``;
    * adaptive ratio history to tune ``alpha`` from observations.

    All per-task cost/solvability calibration is delegated to
    ``TaskEstimator.calibrate``.
    """

    def __init__(
        self,
        budget: int,
        estimator: TaskEstimator,
        alpha_init: float = 1.3,
        beta_mult: float = 1.7,
        alpha_min: float = 1.0,
        alpha_quantile: float = 0.8,
        ema_alpha_for_safety: float = 0.2,
    ) -> None:
        self.budget = budget
        self.estimator = estimator
        self.alpha = alpha_init
        self.alpha_min = alpha_min
        self.beta_mult = beta_mult
        self.alpha_quantile = alpha_quantile
        self.ema = ema_alpha_for_safety
        self._ratios: list[float] = []

    # --- estimates -----------------------------------------------------
    async def estimate(self, task_id: str, depth: str = "heuristic") -> Estimate:
        return await self.estimator.estimate(task_id, depth=depth)

    # --- cost/value for the knapsack ----------------------------------
    def _cost(self, est: Estimate) -> float:
        return max(100.0, est.tokens * self.alpha)

    def _value(self, task: dict, est: Estimate) -> float:
        # expected reward, not nominal
        return float(task.get("reward", 0)) * max(0.05, est.p_solvable)

    # --- per-task limits ----------------------------------------------
    def task_limit(self, est: Estimate) -> int:
        """Beta-limit for ``solve(max_tokens=...)``."""
        return max(1, int(est.tokens * self.alpha * self.beta_mult))

    def cost(self, est: Estimate) -> float:
        return self._cost(est)

    # --- feedback ------------------------------------------------------
    def observe(
        self,
        task_id: str,
        est: Estimate,
        real_tokens: int,
        success: bool,
    ) -> None:
        """Calibrate the estimator and softly adjust alpha."""
        prior = est.tokens
        self.estimator.calibrate(task_id, real_tokens, success)
        if real_tokens > 0 and prior > 0:
            ratio = real_tokens / prior
            self._ratios.append(ratio)
            if len(self._ratios) >= 3:
                q = _quantile(self._ratios[-20:], self.alpha_quantile)
                target = max(self.alpha_min, q)
                self.alpha = (1 - self.ema) * self.alpha + self.ema * target

    # --- plan / re-plan ------------------------------------------------
    async def plan(
        self,
        candidates: list[tuple[str, dict]],
    ) -> tuple[list[str], int, dict[str, Estimate]]:
        """Build the initial plan. Returns (ordered_ids, estimator_cost)."""
        if not candidates:
            return [], 0, {}
        weights: list[int] = []
        values: list[float] = []
        cache: dict[str, Estimate] = {}
        total_est_cost = 0
        for tid, task in candidates:
            est = await self.estimate(tid)
            cache[tid] = est
            total_est_cost += est.estimator_cost
            weights.append(int(self._cost(est)))
            values.append(self._value(task, est))
        idx = _knapsack_01(weights, values, self.budget)
        chosen = [candidates[i] for i in idx]
        chosen.sort(
            key=lambda kv: self._value(kv[1], cache[kv[0]])
            / max(1.0, self._cost(cache[kv[0]])),
            reverse=True,
        )
        return [tid for tid, _ in chosen], total_est_cost, cache

    async def reoptimize(
        self,
        remaining: list[tuple[str, dict]],
        remaining_budget: int,
        depth: str = "heuristic",
    ) -> tuple[list[str], int, dict[str, Estimate]]:
        """Re-plan against the remaining budget."""
        if not remaining or remaining_budget <= 0:
            return [], 0, {}
        weights: list[int] = []
        values: list[float] = []
        cache: dict[str, Estimate] = {}
        total_est_cost = 0
        for tid, task in remaining:
            est = await self.estimate(tid, depth=depth)
            cache[tid] = est
            total_est_cost += est.estimator_cost
            weights.append(int(self._cost(est)))
            values.append(self._value(task, est))
        idx = _knapsack_01(weights, values, remaining_budget)
        chosen = [remaining[i] for i in idx]
        chosen.sort(
            key=lambda kv: self._value(kv[1], cache[kv[0]])
            / max(1.0, self._cost(cache[kv[0]])),
            reverse=True,
        )
        return [tid for tid, _ in chosen], total_est_cost, cache


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Harness:
    """Orchestrates running the agent over a repository of tasks.

    Owns the task loop, reward and token accounting, and result collection.
    Console output and file I/O for the final report are left to the caller.
    """

    def __init__(
        self,
        tasks_dir: Path,
        config: Config,
        solver: TaskSolver,
        estimator: TaskEstimator,
        token_budget: Optional[int] = None,
        alpha_init: float = 1.3,
        beta_mult: float = 1.7,
        warmup_depth: str = "heuristic",
    ) -> None:
        self.tasks_dir = tasks_dir
        self.repo = TaskRepo(tasks_dir)
        self.config = config
        self.solver = solver
        self.estimator = estimator
        self.warmup_depth = warmup_depth

        raw_budget = token_budget
        if raw_budget is None:
            raw_budget = getattr(config, "token_budget", None)
        if not isinstance(raw_budget, int):
            raw_budget = 0
        budget = raw_budget
        self.planner: Optional[BudgetPlanner] = (
            BudgetPlanner(
                budget,
                estimator,
                alpha_init=alpha_init,
                beta_mult=beta_mult,
            )
            if budget > 0
            else None
        )

    # --- candidate collection -----------------------------------------
    def _candidates(self, task_id: Optional[str]) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        for tid in self.repo.get_all_task_ids():
            if task_id and tid != task_id:
                continue
            task = self.repo.get_task(tid)
            if task is None:
                continue
            if not self.config.get_model_name(task.get("task_type", "")):
                continue
            out.append((tid, task))
        return out

    # --- single task execution ----------------------------------------
    async def _execute(
        self,
        tid: str,
        task: dict,
        est: Estimate,
        remaining_budget: Optional[int],
    ) -> dict:
        """Run one task and return the result record.

        ``max_tokens`` bounds the **sum** of input+output tokens, so
        ``remaining_budget`` is a hard cap for this task.
        """
        task_dir = self.repo.get_task_dir(tid)
        task_type = task.get("task_type", "")
        model_name = self.config.get_model_name(task_type)
        reward = task.get("reward", 0)

        max_tokens: Optional[int] = None
        if self.planner is not None and remaining_budget is not None:
            limit = self.planner.task_limit(est)
            max_tokens = max(1, min(limit, remaining_budget))

        alpha = self.planner.alpha if self.planner else 1.0
        print(
            f"[{tid}] type={task_type} model={model_name} "
            f"est={est.tokens} p={est.p_solvable:.2f} "
            f"alpha={alpha:.2f} max_tokens={max_tokens}"
        )
        answer, success, in_tok, out_tok = await self.solver.solve(
            tid, max_tokens=max_tokens
        )
        spent = in_tok + out_tok

        got_reward = False
        if success and task_dir is not None:
            got_reward = check_answer(task, answer, task_dir)
            if got_reward:
                print(f"  ok +{reward}")
            else:
                print(f"  answer mismatch (expected: {task.get('expected')})")
        else:
            print(f"  error/interrupted: {(answer or '')[:200]}")

        if self.planner is not None:
            self.planner.observe(tid, est, spent, success)

        return {
            "task_id": tid,
            "success": success,
            "answer": (answer or "")[:500],
            "reward": reward if got_reward else 0,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": spent,
            "est_tokens": est.tokens,
            "est_p_solvable": est.p_solvable,
            "est_source": est.source,
            "estimator_cost": est.estimator_cost,
            "alpha_at_run": alpha,
        }

    # --- warmup --------------------------------------------------------
    async def _warmup(
        self,
        queue: list[tuple[str, dict]],
        results: list[dict],
        state: dict,
    ) -> None:
        """Run one best task per task type before the main plan.

        Guarantees that ``TaskEstimator.calibrate`` receives signal for
        every task type present in the set, so per-type base costs and
        p_solvable are calibrated before the main plan is built.
        """
        if self.planner is None or not queue:
            return

        by_type: dict[str, list[tuple[str, dict, Estimate]]] = {}
        for tid, task in queue:
            est = await self.planner.estimate(tid, depth=self.warmup_depth)
            state["spent_total"] += est.estimator_cost
            ttype = task.get("task_type", "")
            by_type.setdefault(ttype, []).append((tid, task, est))

        picked: list[tuple[str, dict, Estimate]] = []
        for items in by_type.values():
            items.sort(
                key=lambda x: self.planner._value(x[1], x[2])
                / max(1.0, self.planner._cost(x[2])),
                reverse=True,
            )
            picked.append(items[0])

        taken: set[str] = set()
        for tid, task, est in picked:
            remaining = self.planner.budget - state["spent_total"]
            if remaining <= 0:
                break
            if self.planner._cost(est) > remaining:
                continue
            rec = await self._execute(tid, task, est, remaining)
            state["spent_total"] += rec["total_tokens"]
            state["in"] += rec["input_tokens"]
            state["out"] += rec["output_tokens"]
            state["reward"] += rec["reward"]
            results.append(rec)
            taken.add(tid)

        queue[:] = [(tid, t) for tid, t in queue if tid not in taken]

    # --- main entrypoint -----------------------------------------------
    async def run(self, task_id: Optional[str] = None) -> dict:
        """Execute the harness over the loaded tasks.

        Args:
            task_id: Optional specific task ID to run. If ``None``, all
                tasks in the repository are processed.

        Returns:
            A dict with reward/token accounting and per-task results.
        """
        results: list[dict] = []
        state = {"spent_total": 0, "in": 0, "out": 0, "reward": 0}

        candidates = self._candidates(task_id)

        # --- Stage 0. Warmup (only with a planner) ---
        if self.planner is not None:
            await self._warmup(candidates, results, state)

        # --- Stage 1. Build the plan ---
        if self.planner is not None:
            planned_ids, est_cost, est_cache = await self.planner.plan(candidates)
            state["spent_total"] += est_cost
            by_id = dict(candidates)
            queue = [(tid, by_id[tid]) for tid in planned_ids if tid in by_id]
            deferred = [(tid, t) for tid, t in candidates if tid not in set(planned_ids)]
        else:
            queue = list(candidates)
            deferred = []

        # --- Stage 2. Execution with re-planning ---
        while queue:
            tid, task = queue.pop(0)
            remaining: Optional[int] = None
            if self.planner is not None:
                remaining = self.planner.budget - state["spent_total"]
                if remaining <= 0:
                    break
                est = est_cache.get(tid)
                if est is None:
                    est = await self.planner.estimate(tid)
                    state["spent_total"] += est.estimator_cost
                    remaining = self.planner.budget - state["spent_total"]
                    if remaining <= 0:
                        break
                if self.planner._cost(est) > remaining:
                    deferred.append((tid, task))
                    continue
            else:
                est = Estimate(
                    tokens=0, p_solvable=1.0, source="none", confidence=0.0
                )

            rec = await self._execute(tid, task, est, remaining)
            state["spent_total"] += rec["total_tokens"]
            state["in"] += rec["input_tokens"]
            state["out"] += rec["output_tokens"]
            state["reward"] += rec["reward"]
            results.append(rec)

            if self.planner is not None:
                remaining = self.planner.budget - state["spent_total"]
                if remaining <= 0:
                    break
                pending = queue + deferred
                if pending:
                    pending_map = dict(pending)
                    new_ids, est_cost, new_cache = await self.planner.reoptimize(pending, remaining)
                    state["spent_total"] += est_cost
                    est_cache.update(new_cache)
                    remaining = self.planner.budget - state["spent_total"]
                    queue = [
                        (i, pending_map[i]) for i in new_ids if i in pending_map
                    ]
                    chosen_set = set(new_ids)
                    deferred = [
                        (i, t) for i, t in pending if i not in chosen_set
                    ]
                else:
                    queue = []
                    deferred = []

        # --- Stage 3. Backfill (only if a planner exist and if alpha was overestimated)---
        if self.planner is not None:
            # One initial re-plan over everything deferred.
            while deferred:
                remaining = self.planner.budget - state["spent_total"]
                if remaining <= 0:
                    break

                new_ids, est_cost, est_cache = await self.planner.reoptimize(
                    deferred, remaining
                )
                state["spent_total"] += est_cost
                remaining = self.planner.budget - state["spent_total"]

                if not new_ids or remaining <= 0:
                    break

                # Prepare the backfill queue once per re-plan; skip anything that
                # no longer fits, but keep trying cheaper tasks instead of `break`.
                deferred_map = dict(deferred)
                stage3_queue: list[tuple[str, dict, Estimate]] = []
                for tid in new_ids:
                    est = est_cache.get(tid)
                    if est is None:
                        est = await self.planner.estimate(tid)
                        state["spent_total"] += est.estimator_cost
                        remaining = self.planner.budget - state["spent_total"]
                        if remaining <= 0:
                            break
                    if self.planner._cost(est) <= remaining:
                        stage3_queue.append((tid, deferred_map[tid], est))

                # Remove everything we might run from `deferred` so the next
                # `reoptimize` does not see it again.
                chosen_ids = {tid for tid, _, _ in stage3_queue}
                deferred = [(i, t) for i, t in deferred if i not in chosen_ids]

                if not stage3_queue:
                    break

                # Execute the prepared queue until the budget is exhausted.
                for tid, task, est in stage3_queue:
                    remaining = self.planner.budget - state["spent_total"]
                    if remaining <= 0:
                        break
                    if self.planner._cost(est) > remaining:
                        # can't fit now; push back for a possible re-plan
                        deferred.append((tid, task))
                        continue

                    rec = await self._execute(tid, task, est, remaining)
                    state["spent_total"] += rec["total_tokens"]
                    state["in"] += rec["input_tokens"]
                    state["out"] += rec["output_tokens"]
                    state["reward"] += rec["reward"]
                    results.append(rec)

        # Loop again: if some tasks were pushed back to `deferred`, the
        # next iteration will re-plan them with fresh estimates.
        return {
            "total_reward": state["reward"],
            "total_input_tokens": state["in"],
            "total_output_tokens": state["out"],
            "total_tokens": state["in"] + state["out"],
            "total_spent_including_estimator": state["spent_total"],
            "results": results,
            "estimator_base_costs": self.estimator.base_costs,
            "final_alpha": self.planner.alpha if self.planner else 1.0,
        }