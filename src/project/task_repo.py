import json
from pathlib import Path
from typing import Optional

class TaskRepo:
    """Repository for loading and managing tasks from a directory structure.

    Tasks are loaded from ``task.json`` files in subdirectories of the
    specified tasks directory. Each task must contain at least ``task_id``
    and ``task_type`` fields.
    """

    def __init__(self, tasks_dir: Path):
        """Initialize the repository and load all tasks.

        Args:
            tasks_dir: Root directory containing task subdirectories.
                Each subdirectory should contain a ``task.json`` file.

        Raises:
            ValueError: If tasks_dir is not a valid directory.
        """
        if not tasks_dir.is_dir():
            raise ValueError(f"Tasks directory not found: {tasks_dir}")

        self.tasks_dir = tasks_dir.resolve()
        self._tasks: dict[str, dict] = {}
        self._task_dirs: dict[str, Path] = {}
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load all tasks from task.json files in subdirectories."""
        task_files = [
            tf for tf in self.tasks_dir.rglob("task.json")
            if "workspace" not in tf.parts
        ]

        for tf in sorted(task_files):
            try:
                task = json.loads(tf.read_text(encoding="utf-8-sig"))
            except Exception:
                continue

            if "task_id" not in task or "task_type" not in task:
                continue

            task_id = task["task_id"]
            self._tasks[task_id] = task
            self._task_dirs[task_id] = tf.parent

    def get_task(self, task_id: str) -> Optional[dict]:
        """Get a task by its ID.

        Args:
            task_id: The unique identifier of the task.

        Returns:
            The task dictionary if found, None otherwise.
        """
        return self._tasks.get(task_id)

    def get_task_dir(self, task_id: str) -> Optional[Path]:
        """Get the directory path for a task by its ID.

        Args:
            task_id: The unique identifier of the task.

        Returns:
            The task directory path if found, None otherwise.
        """
        return self._task_dirs.get(task_id)

    def get_all_tasks(self) -> list[dict]:
        """Get all loaded tasks.

        Returns:
            A list of all task dictionaries.
        """
        return list(self._tasks.values())

    def get_all_task_ids(self) -> list[str]:
        """Get all loaded task IDs.

        Returns:
            A list of all task IDs.
        """
        return list(self._tasks.keys())

    def __len__(self) -> int:
        """Return the number of loaded tasks."""
        return len(self._tasks)

    def __iter__(self):
        """Iterate over all tasks."""
        return iter(self._tasks.values())
