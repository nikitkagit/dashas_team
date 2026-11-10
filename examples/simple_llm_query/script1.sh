#!/usr/bin/env bash
# Локальный запуск run_agent.py без Docker.
# Переменные берутся из .env (если есть) или из окружения.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 0. Загрузка .env ----------
if [[ -f "/home/dasha/juiwen/.env" ]]; then
    echo ">>> Загружаю переменные из /home/dasha/juiwen/.env"
    set -a
    # shellcheck disable=SC1091
    source "/home/dasha/juiwen/.env"
    set +a
fi

# ---------- 1. Проверка обязательных переменных ----------
: "${LLM_BASE_URL:?LLM_BASE_URL не задан (проверьте .env)}"
: "${LLM_API_KEY:?LLM_API_KEY не задан (проверьте .env)}"
: "${TASKS:?TASKS не задан (проверьте .env)}"
: "${OUT:?OUT не задан (проверьте .env)}"

if [[ ! -d "$TASKS" ]]; then
    echo "Ошибка: каталог задач не найден: $TASKS" >&2
    exit 1
fi

mkdir -p "$OUT"

# ---------- 2. Пути ----------
AGENT="$SCRIPT_DIR/run_agent.py"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "$AGENT" ]]; then
    echo "Ошибка: не найден $AGENT" >&2
    exit 1
fi

# ---------- 3. venv ----------
if [[ ! -d "$VENV_DIR" ]]; then
    echo ">>> Создаю виртуальное окружение: $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---------- 4. Зависимости ----------
if [[ ! -f "$VENV_DIR/.deps_installed" ]]; then
    echo ">>> Устанавливаю зависимости..."
    pip install --upgrade pip >/dev/null
    pip install -r "/home/dasha/juiwen/requirements.txt"
    touch "$VENV_DIR/.deps_installed"
fi

# ---------- 5. Проверка импорта ----------
if ! python -c "import openjiuwen" 2>/dev/null; then
    echo "Ошибка: openjiuwen не импортируется. Пересоздайте venv: rm -rf $VENV_DIR" >&2
    exit 1
fi

# ---------- 6. Запуск ----------
echo ">>> Запуск harness"
echo "    TASKS=$TASKS"
echo "    OUT=$OUT"
echo "    LLM_BASE_URL=$LLM_BASE_URL"
echo

cd "$SCRIPT_DIR"
exec python "$AGENT" --tasks-dir "$TASKS"