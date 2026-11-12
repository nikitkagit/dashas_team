import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from project.harness import Harness
from project.config import Config
from project.llm_client import LLMClient
from project.task_repo import TaskRepo
from project.task_solver import TaskSolver


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the agent harness on a directory of tasks."
    )
    parser.add_argument(
        "--tasks-dir", required=True,
        help="Directory containing task subdirectories",
    )
    parser.add_argument(
        "--task-id", default=None,
        help="Optional: specific task_id to run (e.g., 'coding_0003')",
    )
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.is_dir():
        print(f"Каталог не найден: {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        repo = TaskRepo(tasks_dir)
    except ValueError as e:
        print(f"Ошибка инициализации репозитория: {e}", file=sys.stderr)
        sys.exit(1)

    config = Config()
    
    # Create a default LLM client (using coding model as default)
    default_model = config.MODEL_ROUTING.get("coding", "qwen3.8-27b")
    llm_client = LLMClient(config, default_model)
    
    # Create solver with the repo, client, and config
    solver = TaskSolver(repo, llm_client, config)
    
    # Initialize harness with config and solver
    harness = Harness(tasks_dir, config, solver)

    report = await harness.run(task_id=args.task_id)

    print(
        f"\n=== ИТОГО: {report['total_reward']} баллов, "
        f"токенов: in={report['total_input_tokens']}, "
        f"out={report['total_output_tokens']}, "
        f"total={report['total_tokens']} ==="
    )

    out_dir = Path(os.environ.get("OUT", "/tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()