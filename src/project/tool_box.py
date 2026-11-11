import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from openjiuwen.core.foundation.tool import LocalFunction, tool

class ToolBox:
    """Collection of tools available to the model.

    All file operations are confined to the task workspace directory
    to prevent accidental access outside the sandbox. Each tool method
    returns a ``LocalFunction`` object ready to be registered with the LLM.
    """

    def __init__(self, task_dir: Path):
        """Initialize the toolbox for a given task directory.

        Args:
            task_dir: Root directory of the task. A ``workspace``
                subdirectory is created (if missing) and used as the
                sandbox root for all file operations.
        """
        self.task_dir = task_dir.resolve()
        self.workspace = (task_dir / "workspace").resolve()
        self.workspace.mkdir(exist_ok=True)

    def _safe_path(self, user_path: str) -> Path:
        """Resolve a user-supplied path and ensure it stays inside the workspace.

        Args:
            user_path: Path provided by the model (relative to workspace).

        Returns:
            Resolved absolute ``Path`` inside the workspace.

        Raises:
            PermissionError: If the resolved path escapes the workspace.
        """
        p = (self.workspace / user_path.lstrip("/")).resolve()
        if not str(p).startswith(str(self.workspace)):
            raise PermissionError(f"Path outside workspace: {user_path}")
        return p

    def read_file(self) -> LocalFunction:
        """Return a LocalFunction for reading a file from the workspace."""

        @tool(name="read_file", description="Read the contents of a file from the workspace")
        def _impl(path: str) -> str:
            p = self._safe_path(path)
            return p.read_text(encoding="utf-8", errors="replace")

        return _impl

    def write_file(self) -> LocalFunction:
        """Return a LocalFunction for writing a file inside the workspace."""

        @tool(name="write_file", description="Write content to a file in the workspace")
        def _impl(path: str, content: str) -> str:
            p = self._safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written {len(content)} characters to {path}"

        return _impl

    def list_files(self) -> LocalFunction:
        """Return a LocalFunction for listing files in the workspace."""

        @tool(name="list_files", description="Show a list of files and folders in the workspace")
        def _impl(subdir: str = ".") -> str:
            p = self._safe_path(subdir)
            if not p.is_dir():
                return f"Not a directory: {subdir}"
            items = []
            for child in sorted(p.iterdir()):
                rel = child.relative_to(self.workspace)
                kind = "DIR" if child.is_dir() else f"FILE ({child.stat().st_size} bytes)"
                items.append(f"{rel}  [{kind}]")
            return "\n".join(items) if items else "(empty)"

        return _impl

    def search_files(self) -> LocalFunction:
        """Return a LocalFunction for searching files by regex."""

        @tool(
            name="search_files",
            description="Find lines matching a regex pattern in all text files in the workspace",
        )
        def _impl(pattern: str, path: str = ".") -> str:
            p = self._safe_path(path)
            results = []
            regex = re.compile(pattern, re.IGNORECASE)
            for f in p.rglob("*"):
                if f.is_file() and f.suffix in (".txt", ".md", ".json", ".csv", ".py", ".log"):
                    try:
                        text = f.read_text(encoding="utf-8", errors="replace")
                        for lineno, line in enumerate(text.splitlines(), 1):
                            if regex.search(line):
                                rel = f.relative_to(self.workspace)
                                results.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                                if len(results) >= 50:
                                    return "\n".join(results) + "\n... (truncated)"
                    except Exception:
                        continue
            return "\n".join(results) if results else "No matches found"

        return _impl

    def run_python(self) -> LocalFunction:
        """Return a LocalFunction for executing Python code."""

        @tool(name="run_python", description="Run Python code and get stdout/stderr")
        def _impl(code: str) -> str:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", dir=self.workspace, delete=False
            ) as f:
                f.write(code)
                script = f.name
            try:
                proc = subprocess.run(
                    [sys.executable, script],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.workspace,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                out = proc.stdout[:4000]
                err = proc.stderr[:2000]
                return f"stdout:\n{out}\nstderr:\n{err}\nexit={proc.returncode}"
            except subprocess.TimeoutExpired:
                return "Error: 30-second timeout"
            finally:
                Path(script).unlink(missing_ok=True)

        return _impl

    def read_csv_table(self) -> LocalFunction:
        """Return a LocalFunction for reading a CSV table."""

        @tool(
            name="read_csv_table",
            description="Read a CSV table and show the first rows as a table",
        )
        def _impl(path: str, max_rows: int = 50) -> str:
            p = self._safe_path(path)
            with p.open(newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = []
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        rows.append(["... (more rows omitted)"])
                        break
                    rows.append(row)
            if not rows:
                return "(empty table)"
            widths = [
                max(len(str(r[j])) for r in rows if j < len(r))
                for j in range(max(len(r) for r in rows))
            ]
            lines = []
            for row in rows:
                lines.append(
                    " | ".join(str(row[j]).ljust(widths[j]) for j in range(len(row)))
                )
            return "\n".join(lines)

        return _impl

    def remember(self) -> LocalFunction:
        """Return a LocalFunction for saving a value to the task memory."""

        @tool(name="remember", description="Save a value to the task memory")
        def _impl(key: str, value: str) -> str:
            mem_file = self.task_dir / ".memory.json"
            mem = {}
            if mem_file.exists():
                try:
                    mem = json.loads(mem_file.read_text())
                except Exception:
                    pass
            mem[key] = value
            mem_file.write_text(json.dumps(mem, ensure_ascii=False, indent=2))
            return f"Remembered: {key}"

        return _impl

    def recall(self) -> LocalFunction:
        """Return a LocalFunction for reading a value from the task memory."""

        @tool(name="recall", description="Read a value from the task memory by key")
        def _impl(key: str) -> str:
            mem_file = self.task_dir / ".memory.json"
            if not mem_file.exists():
                return "(memory is empty)"
            try:
                mem = json.loads(mem_file.read_text())
            except Exception:
                return "(error reading memory)"
            return mem.get(key, f"(no key {key})")

        return _impl