"""Unit tests for the ToolBox module."""

import json
import tempfile
from pathlib import Path

import pytest

from project.tool_box import ToolBox


@pytest.fixture
def task_dir(tmp_path):
    """Create a temporary task directory for testing."""
    return tmp_path


@pytest.fixture
def toolbox(task_dir):
    """Create a ToolBox instance for testing."""
    return ToolBox(task_dir)


class TestSafePath:
    """Tests for the _safe_path method."""

    def test_safe_path_normal(self, toolbox):
        """Test that normal paths are resolved correctly."""
        result = toolbox._safe_path("test.txt")
        assert str(result).startswith(str(toolbox.workspace))
        assert result.name == "test.txt"

    def test_safe_path_with_subdir(self, toolbox):
        """Test that paths with subdirectories are resolved correctly."""
        result = toolbox._safe_path("subdir/test.txt")
        assert "subdir" in str(result)
        assert result.name == "test.txt"

    def test_safe_path_strips_leading_slash(self, toolbox):
        """Test that leading slashes are stripped."""
        result = toolbox._safe_path("/test.txt")
        assert result.name == "test.txt"
        assert str(result).startswith(str(toolbox.workspace))

    def test_safe_path_prevents_escape(self, toolbox):
        """Test that paths trying to escape workspace are rejected."""
        with pytest.raises(PermissionError, match="Path outside workspace"):
            toolbox._safe_path("../escape.txt")

    def test_safe_path_prevents_escape_with_dots(self, toolbox):
        """Test that complex escape attempts are rejected."""
        with pytest.raises(PermissionError, match="Path outside workspace"):
            toolbox._safe_path("subdir/../../escape.txt")


class TestReadFile:
    """Tests for the read_file tool."""

    @pytest.mark.asyncio
    async def test_read_file_exists(self, toolbox):
        """Test reading an existing file."""
        test_file = toolbox.workspace / "test.txt"
        test_file.write_text("Hello, World!", encoding="utf-8")

        read_tool = toolbox.read_file()
        result = await read_tool.invoke({"path": "test.txt"})

        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_file_not_exists(self, toolbox):
        """Test reading a non-existent file raises an error."""
        read_tool = toolbox.read_file()
        with pytest.raises(FileNotFoundError):
            await read_tool.invoke({"path": "nonexistent.txt"})

    @pytest.mark.asyncio
    async def test_read_file_with_encoding_errors(self, toolbox):
        """Test reading a file with invalid UTF-8 bytes."""
        test_file = toolbox.workspace / "test.txt"
        test_file.write_bytes(b"Hello \xff\xfe World")

        read_tool = toolbox.read_file()
        result = await read_tool.invoke({"path": "test.txt"})

        assert "Hello" in result
        assert "World" in result


class TestWriteFile:
    """Tests for the write_file tool."""

    @pytest.mark.asyncio
    async def test_write_file_new(self, toolbox):
        """Test writing a new file."""
        write_tool = toolbox.write_file()
        result = await write_tool.invoke({"path": "test.txt", "content": "Hello, World!"})

        assert "Written 13 characters" in result
        assert (toolbox.workspace / "test.txt").read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_write_file_overwrite(self, toolbox):
        """Test overwriting an existing file."""
        test_file = toolbox.workspace / "test.txt"
        test_file.write_text("Old content", encoding="utf-8")

        write_tool = toolbox.write_file()
        result = await write_tool.invoke({"path": "test.txt", "content": "New content"})

        assert test_file.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(self, toolbox):
        """Test that parent directories are created automatically."""
        write_tool = toolbox.write_file()
        await write_tool.invoke({"path": "subdir/nested/test.txt", "content": "test"})

        assert (toolbox.workspace / "subdir" / "nested" / "test.txt").exists()


class TestListFiles:
    """Tests for the list_files tool."""

    @pytest.mark.asyncio
    async def test_list_files_empty(self, toolbox):
        """Test listing an empty directory."""
        list_tool = toolbox.list_files()
        result = await list_tool.invoke({"subdir": "."})

        assert result == "(empty)"

    @pytest.mark.asyncio
    async def test_list_files_with_content(self, toolbox):
        """Test listing a directory with files and subdirectories."""
        (toolbox.workspace / "file1.txt").write_text("content1")
        (toolbox.workspace / "file2.txt").write_text("content2")
        (toolbox.workspace / "subdir").mkdir()

        list_tool = toolbox.list_files()
        result = await list_tool.invoke({"subdir": "."})

        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" in result
        assert "DIR" in result
        assert "FILE" in result

    @pytest.mark.asyncio
    async def test_list_files_not_directory(self, toolbox):
        """Test listing a non-existent directory."""
        list_tool = toolbox.list_files()
        result = await list_tool.invoke({"subdir": "nonexistent"})

        assert "Not a directory" in result


class TestSearchFiles:
    """Tests for the search_files tool."""

    @pytest.mark.asyncio
    async def test_search_files_found(self, toolbox):
        """Test searching for a pattern that exists."""
        test_file = toolbox.workspace / "test.txt"
        test_file.write_text("Line 1\nHello World\nLine 3", encoding="utf-8")

        search_tool = toolbox.search_files()
        result = await search_tool.invoke({"pattern": "Hello", "path": "."})

        assert "test.txt:2:" in result
        assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_search_files_not_found(self, toolbox):
        """Test searching for a pattern that doesn't exist."""
        test_file = toolbox.workspace / "test.txt"
        test_file.write_text("Line 1\nLine 2", encoding="utf-8")

        search_tool = toolbox.search_files()
        result = await search_tool.invoke({"pattern": "NotFound", "path": "."})

        assert result == "No matches found"

    @pytest.mark.asyncio
    async def test_search_files_case_insensitive(self, toolbox):
        """Test that search is case-insensitive."""
        test_file = toolbox.workspace / "test.txt"
        test_file.write_text("Hello WORLD", encoding="utf-8")

        search_tool = toolbox.search_files()
        result = await search_tool.invoke({"pattern": "hello", "path": "."})

        assert "Hello WORLD" in result

    @pytest.mark.asyncio
    async def test_search_files_truncates_results(self, toolbox):
        """Test that results are truncated at 50 matches."""
        test_file = toolbox.workspace / "test.txt"
        lines = ["match" for _ in range(100)]
        test_file.write_text("\n".join(lines), encoding="utf-8")

        search_tool = toolbox.search_files()
        result = await search_tool.invoke({"pattern": "match", "path": "."})

        assert "(truncated)" in result


class TestRunPython:
    """Tests for the run_python tool."""

    @pytest.mark.asyncio
    async def test_run_python_success(self, toolbox):
        """Test running successful Python code."""
        run_tool = toolbox.run_python()
        result = await run_tool.invoke({"code": "print('Hello, World!')"})

        assert "Hello, World!" in result
        assert "exit=0" in result

    @pytest.mark.asyncio
    async def test_run_python_with_error(self, toolbox):
        """Test running Python code with an error."""
        run_tool = toolbox.run_python()
        result = await run_tool.invoke({"code": "raise ValueError('test error')"})

        assert "ValueError" in result
        assert "test error" in result
        assert "exit=1" in result

    @pytest.mark.asyncio
    async def test_run_python_timeout(self, toolbox):
        """Test that long-running code times out."""
        run_tool = toolbox.run_python()
        result = await run_tool.invoke({"code": "import time; time.sleep(35)"})

        assert "timeout" in result.lower()


class TestReadCsvTable:
    """Tests for the read_csv_table tool."""

    @pytest.mark.asyncio
    async def test_read_csv_table_normal(self, toolbox):
        """Test reading a normal CSV file."""
        csv_file = toolbox.workspace / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")

        read_tool = toolbox.read_csv_table()
        result = await read_tool.invoke({"path": "test.csv", "max_rows": 50})

        assert "name" in result
        assert "age" in result
        assert "Alice" in result
        assert "30" in result
        assert "Bob" in result
        assert "25" in result

    @pytest.mark.asyncio
    async def test_read_csv_table_empty(self, toolbox):
        """Test reading an empty CSV file."""
        csv_file = toolbox.workspace / "test.csv"
        csv_file.write_text("", encoding="utf-8")

        read_tool = toolbox.read_csv_table()
        result = await read_tool.invoke({"path": "test.csv", "max_rows": 50})

        assert result == "(empty table)"

    @pytest.mark.asyncio
    async def test_read_csv_table_truncates_rows(self, toolbox):
        """Test that rows are truncated at max_rows."""
        csv_file = toolbox.workspace / "test.csv"
        lines = ["col1,col2"] + [f"val{i},val{i}" for i in range(100)]
        csv_file.write_text("\n".join(lines), encoding="utf-8")

        read_tool = toolbox.read_csv_table()
        result = await read_tool.invoke({"path": "test.csv", "max_rows": 10})

        assert "(more rows omitted)" in result


class TestRememberRecall:
    """Tests for the remember and recall tools."""

    @pytest.mark.asyncio
    async def test_remember_and_recall(self, toolbox):
        """Test saving and retrieving a value."""
        remember_tool = toolbox.remember()
        await remember_tool.invoke({"key": "test_key", "value": "test_value"})

        recall_tool = toolbox.recall()
        result = await recall_tool.invoke({"key": "test_key"})

        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_recall_empty_memory(self, toolbox):
        """Test recalling from empty memory."""
        recall_tool = toolbox.recall()
        result = await recall_tool.invoke({"key": "nonexistent"})

        assert result == "(memory is empty)"

    @pytest.mark.asyncio
    async def test_remember_overwrites(self, toolbox):
        """Test that remember overwrites existing values."""
        remember_tool = toolbox.remember()
        await remember_tool.invoke({"key": "test_key", "value": "value1"})
        await remember_tool.invoke({"key": "test_key", "value": "value2"})

        recall_tool = toolbox.recall()
        result = await recall_tool.invoke({"key": "test_key"})

        assert result == "value2"

    @pytest.mark.asyncio
    async def test_remember_persists_to_file(self, toolbox):
        """Test that remember writes to the memory file."""
        remember_tool = toolbox.remember()
        await remember_tool.invoke({"key": "test_key", "value": "test_value"})

        mem_file = toolbox.task_dir / ".memory.json"
        assert mem_file.exists()

        mem_data = json.loads(mem_file.read_text())
        assert mem_data["test_key"] == "test_value"