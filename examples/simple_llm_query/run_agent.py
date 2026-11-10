#!/usr/bin/env python3
"""
Harness для решения задач четырёх типов с маршрутизацией по моделям.
Использует openjiuwen для вызова LLM через OpenAI-совместимый API.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from openjiuwen.core.foundation.llm import (
    ModelClientConfig,
    ModelRequestConfig,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
)
from openjiuwen.core.foundation.tool import tool, ToolCard, LocalFunction
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.workflow import (
    End,
    LLMCompConfig,
    LLMComponent,
    Start,
    Workflow,
    WorkflowCard,
)

# =============================================================================
# Конфигурация маршрутизации: task_type -> model
# =============================================================================
MODEL_ROUTING = {
    "coding": "qwen3.8-27b",
    "math": "qwen3.8-27b-noreason",
    "data_anylisis": "qwen3.6-35b-a3b-noreason",
    "search": "qwen3.6-35b-a3b",
}

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096


# =============================================================================
# Конфигурация LLM через openjiuwen
# =============================================================================
def build_model_configs(model_name: str) -> tuple[ModelClientConfig, ModelRequestConfig]:
    #Создаёт конфигурацию клиента и модели для openjiuwen
    provider = os.getenv("MODEL_PROVIDER", "OpenAI")
    api_key = os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_BASE_URL")
    

    if not api_key:
            sys.exit("API_KEY is not set. Export a LiteLLM virtual key (starts with 'sk-').")
    if not api_key.startswith("sk-"):
            sys.exit(
            "API_KEY must be a LiteLLM virtual key starting with 'sk-'. "
            "The invite code is not an API key.")
    if not api_base:
        raise RuntimeError("LLM_BASE_URL is not set")

    client_config = ModelClientConfig(
        client_provider=provider,
        api_key=api_key,
        api_base=api_base,
        verify_ssl=False,
    )
    model_config = ModelRequestConfig(model=model_name)
    return client_config, model_config


# =============================================================================
# Инструменты (tools) для модели — через декоратор @tool openjiuwen
# =============================================================================

class ToolBox:
    #Набор функций, доступных модели. Пути ограничены workspace задачи

    def __init__(self, task_dir: Path):
        self.task_dir = task_dir.resolve()
        self.workspace = (task_dir / "workspace").resolve()
        self.workspace.mkdir(exist_ok=True)

    def _safe_path(self, user_path: str) -> Path:
        p = (self.workspace / user_path.lstrip("/")).resolve()
        if not str(p).startswith(str(self.workspace)):
            raise PermissionError(f"Путь вне workspace: {user_path}")
        return p

    def read_file(self, path: str) -> str:
        p = self._safe_path(path)
        return p.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> str:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Записано {len(content)} символов в {path}"

    def list_files(self, subdir: str = ".") -> str:
        p = self._safe_path(subdir)
        if not p.is_dir():
            return f"Не директория: {subdir}"
        items = []
        for child in sorted(p.iterdir()):
            rel = child.relative_to(self.workspace)
            kind = "DIR" if child.is_dir() else f"FILE ({child.stat().st_size} bytes)"
            items.append(f"{rel}  [{kind}]")
        return "\n".join(items) if items else "(пусто)"

    def search_files(self, pattern: str, path: str = ".") -> str:
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
                                return "\n".join(results) + "\n... (обрезано)"
                except Exception:
                    continue
        return "\n".join(results) if results else "Совпадений не найдено"

    def run_python(self, code: str) -> str:
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
            return "Ошибка: таймаут 30 секунд"
        finally:
            Path(script).unlink(missing_ok=True)

    def read_csv_table(self, path: str, max_rows: int = 50) -> str:
        import csv
        p = self._safe_path(path)
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    rows.append(["... (ещё строки опущены)"])
                    break
                rows.append(row)
        if not rows:
            return "(пустая таблица)"
        widths = [max(len(str(r[j])) for r in rows if j < len(r)) for j in range(max(len(r) for r in rows))]
        lines = []
        for row in rows:
            lines.append(" | ".join(str(row[j]).ljust(widths[j]) for j in range(len(row))))
        return "\n".join(lines)

    def remember(self, key: str, value: str) -> str:
        mem_file = self.task_dir / ".memory.json"
        mem = {}
        if mem_file.exists():
            try:
                mem = json.loads(mem_file.read_text())
            except Exception:
                pass
        mem[key] = value
        mem_file.write_text(json.dumps(mem, ensure_ascii=False, indent=2))
        return f"Запомнено: {key}"

    def recall(self, key: str) -> str:
        mem_file = self.task_dir / ".memory.json"
        if not mem_file.exists():
            return "(память пуста)"
        try:
            mem = json.loads(mem_file.read_text())
        except Exception:
            return "(ошибка чтения памяти)"
        return mem.get(key, f"(нет ключа {key})")


def build_tools(toolbox: ToolBox) -> list[LocalFunction]:
    """Создаёт LocalFunction-инструменты openjiuwen, привязанные к toolbox."""

    @tool(name="read_file", description="Прочитать содержимое файла из workspace")
    def read_file(path: str) -> str:
        return toolbox.read_file(path)

    @tool(name="write_file", description="Записать содержимое в файл в workspace")
    def write_file(path: str, content: str) -> str:
        return toolbox.write_file(path, content)

    @tool(name="list_files", description="Показать список файлов и папок в workspace")
    def list_files(path: str = ".") -> str:
        return toolbox.list_files(path)

    @tool(name="search_files", description="Найти строки по regex-паттерну во всех текстовых файлах workspace")
    def search_files(pattern: str, subdir: str = ".") -> str:
        return toolbox.search_files(pattern, subdir)

    @tool(name="run_python", description="Запустить Python-код и получить stdout/stderr")
    def run_python(code: str) -> str:
        return toolbox.run_python(code)

    @tool(name="read_csv_table", description="Прочитать CSV-таблицу и показать первые строки в виде таблицы")
    def read_csv_table(path: str, max_rows: int = 50) -> str:
        return toolbox.read_csv_table(path, max_rows)

    @tool(name="remember", description="Сохранить значение в память задачи")
    def remember(key: str, value: str) -> str:
        return toolbox.remember(key, value)

    @tool(name="recall", description="Прочитать значение из памяти задачи по ключу")
    def recall(key: str) -> str:
        return toolbox.recall(key)

    return [read_file, write_file, list_files, search_files, run_python, read_csv_table, remember, recall]


# =============================================================================
# Проверка ответа
# =============================================================================
def normalize_int(s: str) -> Optional[int]:
    s = s.strip()
    s = re.sub(r"[^\d\-+]", "", s.split("\n")[0])
    try:
        return int(s)
    except ValueError:
        return None


def check_answer(task: dict, answer_text: str) -> bool:
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


# =============================================================================
# Вызов LLM через openjiuwen Workflow
# =============================================================================
async def call_llm_with_history(
    client_config: ModelClientConfig,
    model_config: ModelRequestConfig,
    messages: list,
    tools: list[LocalFunction],
) -> Any:
    """Один вызов LLM через openjiuwen.
    Возвращает assistant-сообщение (с возможными tool_calls).
    """
    # Собираем контекст из истории сообщений в один prompt
    # openjiuwen LLMComponent принимает user_prompt_template
    # Для многошагового диалога собираем текстовое представление истории
    history_text_parts = []
    for msg in messages:
        role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role == "system":
            continue  # system обрабатывается отдельно
        if role == "tool":
            history_text_parts.append(f"[Результат инструмента]: {content}")
        elif role == "assistant":
            history_text_parts.append(f"[Ассистент]: {content}")
        else:
            history_text_parts.append(f"[Пользователь]: {content}")

    prompt = "\n".join(history_text_parts)

    # Извлекаем system prompt
    system_content = ""
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        if role == "system":
            system_content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            break

    comp_config = LLMCompConfig(
        model_config=model_config,
        model_client_config=client_config,
        system_prompt_template=SystemMessage(content=system_content) if system_content else None,
        user_prompt_template=UserMessage(content=prompt),
        response_format={"type": "text"},
        output_config={"answer": {"type": "string"}},
        # Передаём инструменты, если openjiuwen поддерживает в LLMCompConfig
        # В текущей версии API может потребоваться кастомная обвязка
    )

    workflow_card = WorkflowCard(
        id="llm_step",
        name="llm_step",
        version="1.0",
        input_params={"type": "object", "properties": {}},
    )
    workflow = Workflow(workflow_card)
    workflow.set_start_comp("start", Start())
    workflow.add_workflow_comp("llm", LLMComponent(comp_config))
    workflow.set_end_comp("end", End(), inputs_schema={"answer": "${llm.answer}"})
    workflow.add_connection("start", "llm")
    workflow.add_connection("llm", "end")

    result = await Runner.run_workflow(workflow, inputs={})
    print("TYPE:", type(result))
    print("DIR:", [a for a in dir(result) if not a.startswith("_")])
    if hasattr(result, "__dict__"):
        print("DICT:", result.__dict__)
    return result


# =============================================================================
# Основной цикл решения задачи (упрощённый, без tool calling внутри openjiuwen)
# =============================================================================
async def solve_task(
    client_config: ModelClientConfig,
    task: dict,
    task_dir: Path,
) -> tuple[str, bool]:
    """
    Решает задачу. Поскольку openjiuwen LLMComponent не предоставляет
    прямого доступа к tool calling в одном вызове, используем итеративный
    подход: модель вызывается с описанием доступных инструментов,
    парсим её ответ, выполняем инструменты вручную и повторяем.
    """
    task_type = task.get("task_type", "")
    model_name = MODEL_ROUTING.get(task_type)
    if not model_name:
        return f"Неизвестный тип задачи: {task_type}", False

    # Пересоздаём model_config с нужной моделью
    _, model_config = build_model_configs(model_name)#####################################################################3is error????

    toolbox = ToolBox(task_dir)
    tools = build_tools(toolbox)

    # Формируем описание инструментов для промпта (упрощённо)
    tools_desc = "\n".join([
        f"- {t.card.name}: {t.card.description}"
        for t in tools
    ])

    system_prompt = (
    "You are an agent that solves tasks. Available tools:\n"
    f"{tools_desc}\n\n"
    "To call a tool, respond with JSON of the form: "
    '{"tool": "tool_name", "args": {...}}\n'
    "When the task is solved, write the final answer and save it to answer.txt via write_file."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task["instruction"]},
    ]

    max_rounds = 10
    for round_idx in range(max_rounds):
        # Вызываем LLM через openjiuwen
        try:
            
            result = await call_llm_with_history(
                client_config, model_config, messages, tools
            )
        except Exception as e:
            return f"Ошибка LLM: {e}", False
        
        print(f"[{task['task_id']}] round {round_idx}: model={model_name}")
        # Извлекаем текст ответа из результата workflow
        # Структура result зависит от версии openjiuwen; пробуем разные варианты
        if hasattr(result, "result") and isinstance(result.result, dict):
            output = result.result.get("output", {})
            answer_text = output.get("answer", "") if isinstance(output, dict) else ""
        else:
            answer_text = ""

        # Проверяем, не вызвала ли модель инструмент
        tool_match = re.search(r'\{"tool":\s*"(\w+)",\s*"args":\s*(\{.*?\})\}', answer_text, re.DOTALL)
        if tool_match:
            tool_name = tool_match.group(1)
            try:
                tool_args = json.loads(tool_match.group(2))
            except json.JSONDecodeError:
                tool_args = {}

            # Находим инструмент и выполняем
            tool_result = "Инструмент не найден"
            for t in tools:
                if t.card.name == tool_name:
                    try:
                        tool_result = await t.invoke(tool_args)
                    except Exception as e:
                        tool_result = f"Ошибка: {e}"
                    break

            # Добавляем результат в историю
            messages.append({"role": "assistant", "content": answer_text})
            messages.append({"role": "tool", "content": str(tool_result)[:4000]})
            continue

        # Модель дала финальный ответ
        answer_file = toolbox.workspace / "answer.txt"
        if answer_file.exists():
            answer_text = answer_file.read_text(encoding="utf-8").strip()

        (task_dir / "answer.txt").write_text(answer_text, encoding="utf-8")
        return answer_text, True

    return "Превышено число раундов", False


# =============================================================================
# main
# =============================================================================
async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", required=True)
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.is_dir():
        print(f"Каталог не найден: {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    # Базовая конфигурация клиента (модель будет переопределяться)
    client_config, _ = build_model_configs("placeholder")##########################################################тут проверить

    total_reward = 0
    results = []

    task_files = [
    tf for tf in tasks_dir.rglob("task.json")
    if "workspace" not in tf.parts
    ]
    for tf in sorted(task_files):
        
        try:
            task = json.loads(tf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "task_id" not in task or "task_type" not in task:
            continue
        if task["task_id"] != "coding_0003":
            continue
        task_dir = tf.parent
        task_id = task["task_id"]
        task_type = task["task_type"]
        reward = task.get("reward", 0)
        #print(f"[{task_id}] file={tf}")
        print(f"[{task_id}] type={task_type} model={MODEL_ROUTING.get(task_type)}")

        answer, success = await solve_task(client_config, task, task_dir)

        if success:
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
            "answer": answer[:500],
            "reward": reward if success else 0,
        })

    print(f"\n=== ИТОГО: {total_reward} баллов ===")
    out_dir = Path(os.environ.get("OUT", "/tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"total_reward": total_reward, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    asyncio.run(async_main())##асинхронный вызов.


if __name__ == "__main__":#типа запуск
    main()