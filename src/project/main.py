import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from project.config import MODEL_ROUTING
from project.llm_client import LLMClient
from project.task_repo import TaskRepo
from project.task_solver import TaskSolver
from project.utils import build_client_config, build_model_config


def normalize_int(s: str) -> Optional[int]:
    """Normalize a string to an integer, ignoring non-digit characters."""
    s = s.strip()
    s = re.sub(r"[^\d\-+]", "", s.split("\n")[0])
    try:
        return int(s)
    except ValueError:
        return None


def check_answer(task: dict, answer_text: str) -> bool:
    """Check if the agent's answer matches the expected result."""
    check_type = task.get("check", "exact_str")
    expected = str(task.get("expected", "")).strip()
    actual = answer_text.strip()

    if check_type == "exact_int":
        exp_int = normalize_int(expected)
        act_int = normalize_int(actual)
        return exp_int is not None and act_int is not None and exp_int == act_int
    if check_type == "exact_str":
        return actual.lower() == expected.lower()
    if check_type == "contains":
        return expected.lower() in actual.lower()
    return actual == expected


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

    # Base client configuration (API keys, base URL)
    client_config = build_client_config()

    total_reward = 0
    results: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for task_id in repo.get_all_task_ids():
        if args.task_id and task_id != args.task_id:
            continue

        task = repo.get_task(task_id)
        task_dir = repo.get_task_dir(task_id)

        if task is None or task_dir is None:
            print(f"[{task_id}] Пропущено: задача или директория не найдена")
            continue

        task_type = task.get("task_type", "")
        model_name = MODEL_ROUTING.get(task_type)
        reward = task.get("reward", 0)

        if not model_name:
            print(f"[{task_id}] Пропущено: неизвестный тип задачи '{task_type}'")
            results.append({
                "task_id": task_id,
                "success": False,
                "answer": "Unknown task type",
                "reward": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            })
            continue

        print(f"[{task_id}] type={task_type} model={model_name}")

        # Create a specific LLM client configured for this task's model
        model_config = build_model_config(model_name)
        llm_client = LLMClient(client_config, model_config)

        # Initialize solver with the task-specific client
        solver = TaskSolver(repo, llm_client)
        answer, success, input_tokens, output_tokens = await solver.solve(task_id)

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        if success:
            # For unittest tasks, the check was already performed inside solve()
            if task.get("check") == "unittest":
                ok = True
            else:
                ok = check_answer(task, answer)

            if ok:
                total_reward += reward
                print(f"  ok +{reward} (всего: {total_reward})")
            else:
                print(f"  ответ не совпал (ожидалось: {task.get('expected')})")
        else:
            print(f"  ошибка: {answer[:200]}")

        results.append({
            "task_id": task_id,
            "success": success,
            "answer": answer[:500] if answer else "",
            "reward": reward if (success and (
                task.get("check") == "unittest" or check_answer(task, answer)
            )) else 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        })

    print(
        f"\n=== ИТОГО: {total_reward} баллов, "
        f"токенов: in={total_input_tokens}, out={total_output_tokens}, "
        f"total={total_input_tokens + total_output_tokens} ==="
    )

    out_dir = Path(os.environ.get("OUT", "/tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({
            "total_reward": total_reward,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "results": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()