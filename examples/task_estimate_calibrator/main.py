"""Experiment runner.

Expected config file (path passed via ``--config``):
  {
    "tasks_dir": "/abs/path/to/tasks",
    "model": "qwen3.8-27b",
    "repetitions": 3,
    "estimate_depth": "llm",
    "output_dir": "/tmp/experiment",
    "estimator": { ...optional EstimatorConfig overrides... }
  }

Task IDs are taken from ``TaskRepo(tasks_dir)`` — all ``task.json`` files
found under ``tasks_dir`` are used.

Usage:
  python -m project.task_estimate_calibrator.main --config config.json [--seed 42]
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

from project.config import Config
from project.harness import check_answer
from project.llm_client import LLMClient
from project.task_estimator import EstimatorConfig, TaskEstimator
from project.task_repo import TaskRepo
from project.task_solver import TaskSolver

MIN_TASK_BUDGET = 8000  # safety floor; must exceed one-round LLM overhead


def load_json_dict(path: Path) -> dict[str, Any]:

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


async def run_one(
    tid: str,
    config: Config,
    estimator: TaskEstimator,
    solver: TaskSolver,
    repo: TaskRepo,
    model_name: str,
    estimate_depth: str,
    repetition: int,
) -> dict:
    task = repo.get_task(tid)
    task_dir = repo.get_task_dir(tid)
    task_type = (task or {}).get("task_type", "")
    check_type = (task or {}).get("check", "exact_str")
    reward = float((task or {}).get("reward", 0) or 0)

    # --- ОЦЕНКА до решения ---
    estimate = await estimator.estimate(tid, depth=estimate_depth)
    budget = max(estimate.tokens, MIN_TASK_BUDGET)

    print(
        f"[rep {repetition}] [{tid}] ESTIMATE "
        f"tokens={estimate.tokens} "
        f"source={estimate.source} "
        f"p_solvable={estimate.p_solvable:.2f} "
        f"budget={budget}"
    )

    # --- РЕШЕНИЕ ---
    answer, completed, in_tok, out_tok = await solver.solve(tid, max_tokens=budget)
    real_tokens = config.normalize_tokens(in_tok, out_tok)

    if completed and task is not None and task_dir is not None:
        task_ok = check_answer(task, answer, task_dir)
    else:
        task_ok = False

    if completed:
        estimator.calibrate(tid, real_tokens, task_ok, estimate=estimate)

    # --- СРАВНЕНИЕ после решения ---
    ratio = (real_tokens / estimate.tokens) if estimate.tokens else None
    ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
    print(
        f"[rep {repetition}] [{tid}] RESULT   "
        f"est={estimate.tokens} real={real_tokens} ratio={ratio_str} "
        f"in={in_tok} out={out_tok} "
        f"completed={completed} passed={task_ok}"
    )

    return {
        "repetition": repetition,
        "task_id": tid,
        "task_type": task_type,
        "check": check_type,
        "model": model_name,
        "estimated_tokens": estimate.tokens,
        "estimated_source": estimate.source,
        "estimated_p_solvable": estimate.p_solvable,
        "real_input_tokens": in_tok,
        "real_output_tokens": out_tok,
        "real_tokens": real_tokens,
        "error_ratio": ratio,
        "completed": completed,
        "passed": task_ok,
        "reward": reward if task_ok else 0.0,
        "answer_preview": (answer or "")[:200],
    }


async def run_experiment(
    task_ids: list[str],
    config: Config,
    estimator: TaskEstimator,
    solver: TaskSolver,
    repo: TaskRepo,
    model_name: str,
    repetitions: int,
    estimate_depth: str,
    rng: random.Random,
) -> list[dict]:
    results: list[dict] = []
    for rep in range(repetitions):
        order = list(task_ids)
        rng.shuffle(order)
        print(f"\n=== repetition {rep + 1}/{repetitions} — {len(order)} tasks ===")
        for tid in order:
            results.append(
                await run_one(
                    tid=tid,
                    config=config,
                    estimator=estimator,
                    solver=solver,
                    repo=repo,
                    model_name=model_name,
                    estimate_depth=estimate_depth,
                    repetition=rep,
                )
            )
    return results


def summarize(results: list[dict]) -> dict:
    if not results:
        return {}

    def _bucketed(field: str) -> dict:
        buckets: dict[str, dict] = {}
        for r in results:
            key = r[field]
            b = buckets.setdefault(key, {
                "runs": 0, "passed": 0, "completed": 0,
                "estimated_tokens": 0, "real_tokens": 0,
            })
            b["runs"] += 1
            b["passed"] += int(r["passed"])
            b["completed"] += int(r["completed"])
            b["estimated_tokens"] += r["estimated_tokens"]
            b["real_tokens"] += r["real_tokens"]
        return buckets

    def _per_task() -> dict:
        """Разбивка токенов по каждой задаче (сумма + среднее на запуск)."""
        per: dict[str, dict] = {}
        for r in results:
            tid = r["task_id"]
            t = per.setdefault(tid, {
                "task_type": r["task_type"],
                "check": r["check"],
                "runs": 0,
                "passed": 0,
                "completed": 0,
                "estimated_tokens_total": 0,
                "real_tokens_total": 0,
                "real_input_tokens_total": 0,
                "real_output_tokens_total": 0,
            })
            t["runs"] += 1
            t["passed"] += int(r["passed"])
            t["completed"] += int(r["completed"])
            t["estimated_tokens_total"] += r["estimated_tokens"]
            t["real_tokens_total"] += r["real_tokens"]
            t["real_input_tokens_total"] += r["real_input_tokens"]
            t["real_output_tokens_total"] += r["real_output_tokens"]

        for t in per.values():
            n = t["runs"]
            est = t["estimated_tokens_total"]
            t["avg_estimated_tokens"] = round(est / n, 1) if n else None
            t["avg_real_tokens"] = round(t["real_tokens_total"] / n, 1) if n else None
            t["avg_real_input_tokens"] = round(t["real_input_tokens_total"] / n, 1) if n else None
            t["avg_real_output_tokens"] = round(t["real_output_tokens_total"] / n, 1) if n else None
            t["error_ratio"] = round(t["real_tokens_total"] / est, 3) if est else None
            t["pass_rate"] = round(t["passed"] / n, 3) if n else 0.0
            t["completed_rate"] = round(t["completed"] / n, 3) if n else 0.0
        return per

    total_est = sum(r["estimated_tokens"] for r in results)
    total_real = sum(r["real_tokens"] for r in results)
    ratios = [r["error_ratio"] for r in results if r["error_ratio"] is not None]
    within_2x = sum(1 for x in ratios if 0.5 <= x <= 2.0)
    mean_abs_error = (
        sum(abs(1.0 - x) for x in ratios) / len(ratios) if ratios else None
    )
    completed = sum(1 for r in results if r["completed"])
    passed = sum(1 for r in results if r["passed"])

    return {
        "runs": len(results),
        "total_estimated_tokens": total_est,
        "total_real_tokens": total_real,
        "overall_error_ratio": (total_real / total_est) if total_est else None,
        "mean_abs_error_ratio": mean_abs_error,
        "runs_within_2x": within_2x,
        "completed_rate": completed / len(results),
        "pass_rate": passed / len(results),
        "by_check": _bucketed("check"),
        "by_task_type": _bucketed("task_type"),
        "by_source": _bucketed("estimated_source"),
        "by_task": _per_task(),                      # ← новая секция
    }


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate + solve tasks repeatedly, calibrating the estimator."
    )
    DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to experiment config JSON (default: config.json)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    exp_cfg = load_json_dict(config_path)

    tasks_dir = Path(exp_cfg["tasks_dir"])
    if not tasks_dir.is_absolute():
        tasks_dir = (config_path.resolve().parent / tasks_dir).resolve()
    if not tasks_dir.is_dir():
        print(f"tasks_dir not found: {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    model_name = exp_cfg.get("model", "qwen3.8-27b")
    repetitions = int(exp_cfg.get("repetitions", 1))
    estimate_depth = exp_cfg.get("estimate_depth", "llm")
    output_dir = Path(exp_cfg.get("output_dir", "/tmp/experiment"))
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator_overrides = exp_cfg.get("estimator") or {}
    estimator_cfg = EstimatorConfig(**estimator_overrides)

    repo = TaskRepo(tasks_dir)
    task_ids = repo.get_all_task_ids()
    if not task_ids:
        print(f"No tasks found under {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    config = Config()
    estimator_client = LLMClient(config, "qwen3.6-35b-a3b-noreason")  # дёшево
    solver_client    = LLMClient(config, model_name)                  # по routing

    estimator = TaskEstimator(repo, estimator_client, config)
    solver    = TaskSolver(repo, solver_client, config)

    rng = random.Random(args.seed)

    results = await run_experiment(
        task_ids=task_ids,
        config=config,
        estimator=estimator,
        solver=solver,
        repo=repo,
        model_name=model_name,
        repetitions=repetitions,
        estimate_depth=estimate_depth,
        rng=rng,
    )

    summary = summarize(results)
    report = {
        "config": {
            "config_file": str(config_path),
            "tasks_dir": str(tasks_dir),
            "model": model_name,
            "repetitions": repetitions,
            "estimate_depth": estimate_depth,
        },
        "summary": summary,
        "calibrated_base_costs": estimator.base_costs,
        "results": results,
    }
    out_file = output_dir / "experiment_results.json"
    out_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    by_task = summary.get("by_task", {})
    if by_task:
        print("\n=== TOKEN COST PER TASK (normalized) ===")
        header = (
            f"{'task_id':<20} {'type':<14} {'check':<10} "
            f"{'runs':>4} {'est':>8} {'real':>8} {'ratio':>6} "
            f"{'in':>8} {'out':>6} {'pass':>5}"
        )
        print(header)
        print("-" * len(header))
        for tid, t in sorted(by_task.items()):
            ratio = t["error_ratio"]
            ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
            print(
                f"{tid:<20} {t['task_type']:<14} {t['check']:<10} "
                f"{t['runs']:>4} "
                f"{t['estimated_tokens_total']:>8} "
                f"{t['real_tokens_total']:>8} "
                f"{ratio_str:>6} "
                f"{t['real_input_tokens_total']:>8} "
                f"{t['real_output_tokens_total']:>6} "
                f"{t['passed']:>5}"
            )
        print("-" * len(header))
        print(
            f"{'TOTAL':<20} {'':<14} {'':<10} "
            f"{summary['runs']:>4} "
            f"{summary['total_estimated_tokens']:>8} "
            f"{summary['total_real_tokens']:>8}"
        )
    print(f"\nFull report: {out_file}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
