# Использование `main.py`

## Подготовка

Установи зависимости и активируй виртуальное окружение:

```sh
cd /mnt/c/Dev/projects/dashas_team
source .venv/bin/activate
pip install mcp
```

Рядом со скриптом должен лежать `team.json` с анкетой команды: 2–3 участника и блок `consent`. Пример структуры:

```json
{
  "team_name": "Команда Пример",
  "participants": [
    {
      "full_name": "Иванов Иван Иванович",
      "email": "ivanov@example.org",
      "telegram": "@ivanov",
      "phone": "+70000000001",
      "university": "НГУ",
      "city": "Новосибирск",
      "citizenship": "Россия"
    },
    {
      "full_name": "Петрова Анна Сергеевна",
      "email": "petrova@example.org",
      "telegram": "@petrova",
      "phone": "+70000000002",
      "university": "НГУ",
      "city": "Новосибирск",
      "citizenship": "Россия"
    }
  ],
  "consent": {
    "accepted": true,
    "text_version": "nsu-privacy-registration-v1"
  }
}
```

## Запуск

Скрипту нужна хотя бы одна переменная окружения `JIUWENSWARM_*`. Проще всего выставить её на текущую сессию:

```sh
export JIUWENSWARM_DATA_DIR=$(pwd)
python main.py
```

Либо одной строкой, без изменения окружения:

```sh
JIUWENSWARM_DATA_DIR=$(pwd) python main.py
```

## Параметры командной строки

| Флаг | Назначение | По умолчанию |
|------|-----------|--------------|
| `--name` | имя агента в реестре | `Scout` |
| `--version` | версия агента | `1.0` |
| `--team` | путь к файлу анкеты | `./team.json` |
| `--key` | путь к SSH-ключу Ed25519 | `~/.ssh/agent-registry-key` |
| `--no-team` | чек-ин без обновления анкеты | выключено |

## Примеры

Обычная регистрация команды:

```sh
JIUWENSWARM_DATA_DIR=$(pwd) python main.py
```

Регистрация под другим именем и версией:

```sh
JIUWENSWARM_DATA_DIR=$(pwd) python main.py --name Alpha --version 2.0
```

Повторный чек-ин без изменения анкеты:

```sh
JIUWENSWARM_DATA_DIR=$(pwd) python main.py --no-team
```

Своя анкета и свой ключ:

```sh
JIUWENSWARM_DATA_DIR=$(pwd) python main.py \
  --team my_team.json \
  --key ~/.ssh/my-key
```

## Что делает скрипт

1. Проверяет наличие SSH-ключа, при отсутствии создаёт Ed25519-пару.
2. Читает `team.json` и считает SHA-256 канонического JSON анкеты.
3. Собирает `harness_report` из переменных `JIUWENSWARM_*`.
4. Подключается к `https://openjiuwen.iir.nsu.ru/mcp`.
5. Вызывает `create_registration_challenge`, получает `nonce`.
6. Формирует сообщение `<nonce>:<name>:<version>:<sha256(report)>:team-v1:<team_digest>`.
7. Подписывает его через `ssh-keygen -Y sign -n agent-registry`.
8. Вызывает `register_agent` с анкетой, отчётом, подписью и публичным ключом.
9. Печатает JSON-ответ и `registration_id`.

## Результат

В конце вывода будет блок вида:

```json
{
  "created": true,
  "agent": {
    "name": "Scout",
    "registration_id": "fc5428f8eda93eb6b76065e2f9014398"
  },
  "attested": true
}
```

И отдельной строкой:

```
registration_id: fc5428f8eda93eb6b76065e2f9014398
```

Сохрани `registration_id`: он нужен для `heartbeat` и `deregister_agent`. При каждой успешной регистрации выдаётся новый ID.

## Типичные ошибки

- `No JIUWENSWARM_* environment variables found` — не выставлена переменная. Выполни `export JIUWENSWARM_DATA_DIR=$(pwd)`.
- `operation authorization failed` — истёк challenge (TTL 300 с) или подпись не совпала. Перезапусти скрипт, он возьмёт новый `nonce`.
- `team questionnaire is required` — нет `team.json`, и не указан `--no-team`.
- `team.participants` — ошибка в анкете: неверное поле, пустое значение или нарушен формат. Исправь `team.json` и перезапусти.
- `ModuleNotFoundError: No module named 'mcp'` — запускаешь системным Python вместо venv. Используй `.venv/bin/python main.py`.