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

Usage:
  python -m examples.task_estimate_calibrator.main [--tasks-dir /data/tasks] [--config config.json] [--seed 42]
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

MIN_TASK_BUDGET = 2500  # safety floor; must exceed one-round LLM overhead


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

    estimate = await estimator.estimate(tid, depth=estimate_depth)

    # estimate.tokens is a total budget, but enforce a floor so a broken
    # (e.g. "tiny") estimate still allows one full LLM round to complete.
    budget = max(estimate.tokens, MIN_TASK_BUDGET)
    answer, completed, in_tok, out_tok = await solver.solve(tid, max_tokens=budget)
    real_tokens = config.normalize_tokens(in_tok, out_tok)

    if completed and task is not None and task_dir is not None:
        task_ok = check_answer(task, answer, task_dir)
    else:
        task_ok = False

    # Calibrate ONLY on completed runs. A budget-exceeded or transport-failed
    # run says nothing about the true task cost or its solvability, and
    # feeding it back would poison p_solvable and heuristic_safety.
    if completed:
        estimator.calibrate(tid, real_tokens, task_ok)

    ratio = (real_tokens / estimate.tokens) if estimate.tokens else None
    ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
    print(
        f"[rep {repetition}] [{tid}] type={task_type} check={check_type} "
        f"est={estimate.tokens} real={real_tokens} ratio={ratio_str} "
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
            b = buckets.setdefault(key, {"runs": 0, "passed": 0, "real_tokens": 0})
            b["runs"] += 1
            b["passed"] += int(r["passed"])
            b["real_tokens"] += r["real_tokens"]
        return buckets

    total_est = sum(r["estimated_tokens"] for r in results)
    total_real = sum(r["real_tokens"] for r in results)
    within_2x = sum(
        1
        for r in results
        if r["error_ratio"] is not None and 0.5 <= r["error_ratio"] <= 2.0
    )
    completed = sum(1 for r in results if r["completed"])
    passed = sum(1 for r in results if r["passed"])
    return {
        "runs": len(results),
        "total_estimated_tokens": total_est,
        "total_real_tokens": total_real,
        "overall_error_ratio": (total_real / total_est) if total_est else None,
        "runs_within_2x": within_2x,
        "completed_rate": completed / len(results),
        "pass_rate": passed / len(results),
        "by_check": _bucketed("check"),
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
    parser.add_argument(
        "--tasks-dir",
        default=None,
        help="Override tasks_dir from config (e.g., /data/tasks)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    exp_cfg = load_json_dict(config_path)

    tasks_dir_str = args.tasks_dir or exp_cfg.get("tasks_dir", "/data/tasks")
    tasks_dir = Path(tasks_dir_str)
    
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
    llm_client = LLMClient(config, model_name)
    estimator = TaskEstimator(repo, llm_client, config, estimator_cfg)
    solver = TaskSolver(repo, llm_client, config)

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
    print(f"\nFull report saved to: {out_file}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()