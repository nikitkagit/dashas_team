from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class Task:
    task_id: str
    tokens: int                # ожидаемый расход в выходных токенах (входные/10)
    reward: float = 0.0        # баллы из task.json
    p_solvable: float = 1.0    # оценка выполнимости [0..1]
    attempts: int = 0          # фактических попыток
    version: int = 0           # растёт при любом обновлении — для диагностики

    @property
    def expected_value(self) -> float:
        """Математическое ожидание баллов."""
        return self.reward * self.p_solvable

    @property
    def roi(self) -> float:
        """«Вкусность» = ожидаемые баллы на один токен."""
        return self.expected_value / max(self.tokens, 1)


class KnapsackSolver:
    """Решатель задачи о рюкзаке для конкурсного harness.

    Базовый контракт (по ТЗ):
      * add_task(task_id, tokens)  — добавить задачу; при существующем
                                     task_id число токенов перезаписывается;
      * pop_tastiest() -> id | None — вернуть id самой «вкусной» задачи
                                     и удалить её из рюкзака.

    «Вкусность» — это ROI = (reward * p_solvable) / tokens, а не сами
    токены. Это позволяет учитывать, что задача с 10 токенами и reward=1
    выгоднее, чем задача с 100 токенами и reward=2.

    Дополнительно:
      * update_estimate()    — уточнить оценку, не сбрасывая остальные поля;
      * record_attempt()     — обновить p_solvable и tokens по факту попытки;
      * can_start()          — проверка «влезает ли задача в остаток бюджета»;
      * fit(budget)          — жадный подбор набора задач под бюджет.
    """

    def __init__(self, *, default_reward: float = 1.0) -> None:
        self._tasks: dict[str, Task] = {}
        self._default_reward = default_reward

    # -------------------- методы из ТЗ --------------------

    def add_task(
        self,
        task_id: str,
        tokens: int,
        *,
        reward: float | None = None,
        p_solvable: float = 1.0,
    ) -> Task:
        """Добавить задачу. Если task_id существует — поля перезаписываются.

        Параметры
        ---------
        task_id     : идентификатор задачи (обычно имя каталога);
        tokens      : ожидаемый расход в выходных токенах; >= 0;
        reward      : баллы из task.json; при None берётся default_reward;
        p_solvable  : априорная вероятность выполнимости из триажа [0..1].

        Возвращает актуальный Task.
        """
        if tokens < 0:
            raise ValueError(f"tokens должен быть >= 0, получено {tokens!r}")
        if not 0.0 <= p_solvable <= 1.0:
            raise ValueError(f"p_solvable должен быть в [0,1], получено {p_solvable!r}")
        reward_val = self._default_reward if reward is None else float(reward)

        if task_id in self._tasks:
            t = self._tasks[task_id]
            t.tokens = int(tokens)
            t.reward = reward_val
            t.p_solvable = float(p_solvable)
            t.version += 1
            return t

        t = Task(
            task_id=task_id,
            tokens=int(tokens),
            reward=reward_val,
            p_solvable=float(p_solvable),
        )
        self._tasks[task_id] = t
        return t

    def pop_tastiest(self) -> Optional[str]:
        """Вернуть id самой «вкусной» задачи и удалить её из рюкзака.

        Возвращает None, если задач нет.
        При равном ROI выигрывает задача, добавленная раньше
        (dict в Python сохраняет порядок вставки).
        """
        if not self._tasks:
            return None
        best_id = max(self._tasks, key=lambda k: self._tasks[k].roi)
        del self._tasks[best_id]
        return best_id

    # -------------------- обновления оценки --------------------

    def update_estimate(
        self,
        task_id: str,
        *,
        tokens: Optional[int] = None,
        p_solvable: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> Optional[Task]:
        """Уточнить оценку, не сбрасывая остальные поля. None — если задачи нет."""
        t = self._tasks.get(task_id)
        if t is None:
            return None
        if tokens is not None:
            if tokens < 0:
                raise ValueError("tokens >= 0")
            t.tokens = int(tokens)
        if p_solvable is not None:
            if not 0.0 <= p_solvable <= 1.0:
                raise ValueError("p_solvable in [0,1]")
            t.p_solvable = float(p_solvable)
        if reward is not None:
            t.reward = float(reward)
        t.version += 1
        return t

    def record_attempt(
        self,
        task_id: str,
        *,
        success: bool,
        real_tokens: int,
        alpha: float = 0.5,
    ) -> Optional[Task]:
        """Обновить оценку по факту попытки.

        p_solvable — EMA (alpha сдвигает к 1.0 при успехе, к 0.0 при провале);
        tokens     — скользящее среднее между прежней оценкой и фактом.
        """
        t = self._tasks.get(task_id)
        if t is None:
            return None
        t.attempts += 1
        target = 1.0 if success else 0.0
        t.p_solvable = (1.0 - alpha) * t.p_solvable + alpha * target
        if real_tokens > 0:
            t.tokens = int(0.5 * t.tokens + 0.5 * real_tokens)
        t.version += 1
        return t

    # -------------------- выбор под бюджет --------------------

    def can_start(
        self,
        task_id: str,
        remaining_budget: int,
        safety: float = 1.3,
    ) -> bool:
        """Стоит ли стартовать задачу при данном остатке бюджета.

        safety — коэффициент запаса: не начинаем задачу, если ожидаемая
        стоимость * safety превышает остаток. 1.3 по умолчанию — компромисс
        между «не терять дешёвые задачи» и «не влезать в долги».
        """
        t = self._tasks.get(task_id)
        if t is None or t.p_solvable <= 0.0:
            return False
        if t.tokens == 0:
            return True
        return remaining_budget >= t.tokens * safety

    def fit(self, budget: int, *, exclude: Optional[set[str]] = None) -> list[str]:
        """Жадный подбор набора задач под бюджет.

        Порядок — по убыванию ROI. Возвращает список task_id, суммарные
        tokens которых не превышают budget. Для 100 задач жадности
        достаточно; если понадобится оптимум — замените на DP по бюджету.
        """
        exclude = exclude or set()
        candidates = [
            t for t in self._tasks.values()
            if t.task_id not in exclude and t.p_solvable > 0.0
        ]
        candidates.sort(key=lambda t: t.roi, reverse=True)

        chosen: list[str] = []
        used = 0
        for t in candidates:
            if t.tokens == 0:
                chosen.append(t.task_id)
                continue
            if used + t.tokens <= budget:
                chosen.append(t.task_id)
                used += t.tokens
        return chosen

    # -------------------- вспомогательное --------------------

    def peek_tastiest(self) -> Optional[str]:
        if not self._tasks:
            return None
        return max(self._tasks, key=lambda k: self._tasks[k].roi)

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def remove(self, task_id: str) -> Optional[Task]:
        return self._tasks.pop(task_id, None)

    def expected_value_total(self) -> float:
        return sum(t.expected_value for t in self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks.values())
