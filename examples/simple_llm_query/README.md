# Простой запрос к LLM через openJiuwen

Минимальный Python-скрипт, который отправляет один запрос к LLM через
[openJiuwen](https://github.com/openjiuwen), используя OpenAI-совместимый API
(по умолчанию — шлюз NeuralDeep).

## Требования

- Python 3.11–3.13
- Доступ к OpenAI-совместимому LLM-endpoint (NeuralDeep, DashScope и т.п.)

## Установка

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U openjiuwen
```

## Настройка

Скрипт читает конфигурацию из переменных окружения:

| Переменная       | Обязательна | По умолчанию                    | Описание                                             |
|------------------|-------------|---------------------------------|------------------------------------------------------|
| `API_KEY`        | да          | —                               | LiteLLM virtual key, должен начинаться с `sk-`      |
| `API_BASE`       | нет         | `https://api.neuraldeep.ru/v1`  | OpenAI-совместимый базовый URL (без слэша в конце)  |
| `MODEL_PROVIDER` | нет         | `OpenAI`                        | Провайдер для openJiuwen (`OpenAI`, `DashScope`, `SiliconFlow` и т.д.) |
| `MODEL_NAME`     | нет         | `qwen3.8-27b`                   | Идентификатор модели из `/v1/models`                 |

## Использование

```sh
export API_KEY="sk-..."
python main.py "Привет, как дела?"
```

Без аргументов скрипт отправляет промпт по умолчанию.

Результат выводится как сырой результат workflow, например:

```
result=None state=<WorkflowExecutionState.COMPLETED: 'COMPLETED'>
```

Сам ответ LLM виден в логах openJiuwen (`response_content`).

## Как это работает

1. Читает переменные окружения и собирает `ModelClientConfig` / `ModelRequestConfig`.
2. Создаёт `LLMComponent` с шаблоном промпта в виде `UserMessage`.
3. Собирает минимальный workflow: `start → llm → end`.
4. Запускает его через `Runner.run_workflow`.
5. Печатает результат.

## Решение проблем

### `AuthenticationError: 401 - LiteLLM Virtual Key expected`

Передан инвайт-код вместо API-ключа. Сгенерируй virtual key.
Он должен начинаться с `sk-`.

### `APIStatusError: Error code: 308`

`API_BASE` вызывает редирект. OpenAI-клиенты не следуют редиректам в POST.
Проверь базовый URL:

```sh
curl -i https://api.neuraldeep.ru/v1/models -H "Authorization: Bearer $API_KEY"
```

- Если `Location:` указывает на `/v1/v1/...`, убери лишний `/v1`.
- В `API_BASE` не должно быть слэша в конце.

### `RuntimeError: There is no current event loop in thread 'MainThread'`

Сборка workflow идёт вне асинхронного контекста (типично для ячеек IPython).
Оберни сборку и `Runner.run_workflow` в `async def` и вызывай через `await`.

### `ValidationError: response format {} is invalid`

`LLMCompConfig` требует `response_format` и `output_config`. Для простого текста:

```python
response_format={"type": "text"},
output_config={"answer": {"type": "string"}},
```

### `AttributeError: 'str' object has no attribute 'get'`

`template_content` должен быть списком словарей-сообщений, а не строкой.
Либо используй `user_prompt_template=UserMessage(content="...")`, либо
`template_content=[{"role": "user", "content": "..."}]`.

### Плейсхолдер `{query}` не подставляется

Используй двойные фигурные скобки `{{query}}` в шаблоне.

### `ModuleNotFoundError: No module named 'mcp'`

Запуск идёт системным Python, а не из venv. Используй
`.venv/bin/python main.py` или сначала активируй venv.

## Безопасность

- Никогда не коммить API-ключи. Храни их в переменных окружения или в локальном
  `.env`, исключённом из системы контроля версий.
- Если ключ утёк — отзови его в личном кабинете и выпусти новый.