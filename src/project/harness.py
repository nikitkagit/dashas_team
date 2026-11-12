import re
from pathlib import Path
from typing import Optional

from project.config import Config
from project.task_repo import TaskRepo
from project.task_solver import TaskSolver


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


class Harness:
    """Orchestrates running the agent over a repository of tasks.

    Owns the task loop, reward and token accounting, and result collection.
    Console output and file I/O for the final report are left to the caller.
    """

    def __init__(self, tasks_dir: Path, config: Config, solver: TaskSolver) -> None:
        """Initialize the harness.

        Args:
            tasks_dir: Root directory containing task subdirectories.
            config: Shared configuration instance.
            solver: Task solver instance to use for all tasks.

        Raises:
            ValueError: If the task repository cannot be initialized.
        """
        self.tasks_dir = tasks_dir
        self.repo = TaskRepo(tasks_dir)
        self.config = config
        self.solver = solver

    async def run(self, task_id: Optional[str] = None) -> dict:
        """Execute the harness over the loaded tasks.

        Args:
            task_id: Optional specific task ID to run. If ``None``, all
                tasks in the repository are processed.

        Returns:
            A dict with ``total_reward``, ``total_input_tokens``,
            ``total_output_tokens``, ``total_tokens`` and ``results``
            (a list of per-task result dicts).
        """
        total_reward = 0
        results: list[dict] = []
        total_input_tokens = 0
        total_output_tokens = 0

        for tid in self.repo.get_all_task_ids():
            if task_id and tid != task_id:
                continue

            task = self.repo.get_task(tid)
            task_dir = self.repo.get_task_dir(tid)

            if task is None or task_dir is None:
                print(f"[{tid}] Пропущено: задача или директория не найдена")
                continue

            task_type = task.get("task_type", "")
            model_name = self.config.get_model_name(task_type)
            reward = task.get("reward", 0)

            if not model_name:
                print(f"[{tid}] Пропущено: неизвестный тип задачи '{task_type}'")
                results.append({
                    "task_id": tid,
                    "success": False,
                    "answer": "Unknown task type",
                    "reward": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                })
                continue

            print(f"[{tid}] type={task_type} model={model_name}")

            answer, success, input_tokens, output_tokens = await self.solver.solve(tid)

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            if success:
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
                "task_id": tid,
                "success": success,
                "answer": answer[:500] if answer else "",
                "reward": reward if (success and (
                    task.get("check") == "unittest" or check_answer(task, answer)
                )) else 0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            })

        return {
            "total_reward": total_reward,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "results": results,
        }