import argparse
import asyncio
import json
import sys
from pathlib import Path

from project.harness import Harness
from project.config import Config
from project.llm_client import LLMClient
from project.task_estimator import EstimatorConfig, TaskEstimator
from project.task_repo import TaskRepo
from project.task_solver import TaskSolver


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the agent harness on a directory of tasks."
    )
    parser.add_argument(
        "--tasks-dir",
        required=True,
        help="Directory containing task subdirectories",
    )
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.is_dir():
        print(f"Tasks directory not found: {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        repo = TaskRepo(tasks_dir)
    except ValueError as e:
        print(f"Failed to initialize repository: {e}", file=sys.stderr)
        sys.exit(1)

    config = Config()

    # Default client model (used for estimation and as a fallback for solving).
    default_model = config.MODEL_ROUTING.get("coding", "qwen3.8-27b")
    llm_client = LLMClient(config, default_model)

    # Estimator uses the same client; defaults are baked into EstimatorConfig.
    estimator_cfg = EstimatorConfig()
    estimator = TaskEstimator(repo, llm_client, config, estimator_cfg)

    # Solver does the actual per-task work.
    solver = TaskSolver(repo, llm_client, config)

    # Budget comes from Config (BUDGET env var, default 100k).
    harness = Harness(
        tasks_dir,
        config,
        solver,
        estimator,
        token_budget=config.budget,
    )

    report = await harness.run()

    print(
        f"\n=== TOTAL: {report['total_reward']} points, "
        f"tokens: in={report['total_input_tokens']}, "
        f"out={report['total_output_tokens']}, "
        f"total={report['total_tokens']}, "
        f"spent (with estimator)={report['total_spent_including_estimator']} ==="
    )

    # Always write the report to /tmp.
    out_dir = Path("/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    out_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Full report saved to: {out_file}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()