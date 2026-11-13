"""
extract_and_classify_spacy.py
Извлечение инструкций + распознавание вопросов через spaCy.
Требует: spacy, en_core_web_sm
"""

from dataclasses import dataclass, field
from typing import List, Optional

import spacy
from spacy.tokens import Doc, Token

nlp = spacy.load("en_core_web_sm")


# --- 1. Словари ---------------------------------------------------------

ACTION_VERBS = {
    "run", "fix", "write", "edit", "do", "work", "create", "delete",
    "add", "remove", "update", "install", "build", "test", "check",
    "ensure", "make", "use", "execute", "modify", "implement", "verify",
    "commit", "push", "save", "open", "close", "move", "copy", "rename",
    "read", "parse", "return", "print", "call", "set", "get", "put",
    "start", "stop", "restart", "deploy", "merge", "revert", "refactor",
}

WH_WORDS = {
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
}

# Вспомогательные глаголы для yes/no вопросов (инверсия)
AUX_FOR_YESNO = {
    "is", "are", "was", "were", "do", "does", "did",
    "have", "has", "had", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must",
}


# --- 2. Структура результата --------------------------------------------

@dataclass
class Sentence:
    text: str
    kind: str                       # instruction | question | description | mixed
    score: int
    is_question: bool = False
    question_type: str = ""         # wh | yesno | indirect | ""
    verbs: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


# --- 3. Разбиение на предложения через spaCy ----------------------------

def split_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# --- 4. Извлечение артефактов -------------------------------------------

def extract_artifacts(doc: Doc) -> List[str]:
    """Файлы, команды, пути — через форму токенов и эвристики."""
    artifacts = []
    # Имена файлов: токен содержит точку и расширение
    for token in doc:
        t = token.text
        if "." in t and len(t) > 3 and not t.startswith("."):
            # похоже на file.ext
            if any(c.isalpha() for c in t.split(".")[-1]):
                artifacts.append(t)
        # backticks (spaCy сохраняет как отдельные токены ` и `)
        if t == "`":
            artifacts.append("`")
    # Соберём содержимое backticks как единое целое
    text = doc.text
    import re
    for m in re.finditer(r"`([^`]+)`", text):
        artifacts.append(m.group(1).strip())
    # Убираем одиночные "`"
    artifacts = [a for a in artifacts if a != "`"]
    # Уникализируем, сохраняя порядок
    seen = set()
    return [a for a in artifacts if not (a in seen or seen.add(a))]


# --- 5. Детектор вопросов через spaCy -----------------------------------

def detect_question(doc: Doc) -> tuple:
    """
    Возвращает (is_question, question_type, reasons).
    Типы: 'wh' | 'yesno' | 'indirect' | ''
    """
    reasons = []
    text_stripped = doc.text.strip()

    # 5.1. Явный знак вопроса в конце
    has_qmark = text_stripped.endswith("?")

    # 5.2. Вопросительное слово в начале (по лемме, не по POS)
    first_token = next((t for t in doc if not t.is_space), None)
    if first_token is not None and first_token.lemma_.lower() in WH_WORDS:
        reasons.append(f"wh_start:{first_token.lemma_.lower()}")
        return True, "wh", reasons

    # 5.3. Косвенный вопрос: whether / if внутри — это НЕ вопрос,
    #      если главный глагол императивный
    has_indirect = any(
        t.lemma_.lower() in {"whether", "if"} for t in doc
    )
    if has_indirect:
        root = _get_root(doc)
        if root is not None and root.lemma_.lower() in ACTION_VERBS:
            reasons.append("imperative_with_indirect")
            return False, "indirect", reasons
        # Без императива — считаем вопросом, если есть ?
        if has_qmark:
            return True, "yesno", reasons + ["indirect_qmark"]
        return False, "indirect", reasons + ["indirect_no_qmark"]

    # 5.4. Yes/no вопрос: инверсия aux + subject в начале
    if len(doc) >= 2:
        first_two = [t for t in doc if not t.is_space][:2]
        if (len(first_two) == 2
                and first_two[0].lemma_.lower() in AUX_FOR_YESNO
                and first_two[1].pos_ in ("NOUN", "PROPN", "PRON")):
            reasons.append(f"yesno_inversion:{first_two[0].lemma_.lower()}")
            return True, "yesno", reasons

    # 5.5. Знак вопроса без инверсии и без wh — всё равно вопрос
    if has_qmark:
        reasons.append("question_mark")
        return True, "yesno", reasons

    return False, "", reasons


def _get_root(doc: Doc) -> Optional[Token]:
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None


# --- 6. Анализ предложения ----------------------------------------------

def analyze_sentence(sentence: str) -> Sentence:
    doc = nlp(sentence)
    score = 0
    reasons: List[str] = []

    # 6.1. Глаголы действия (по лемме)
    verbs = [t.lemma_.lower() for t in doc
             if t.pos_ in ("VERB", "AUX") and t.lemma_.lower() in ACTION_VERBS]
    if verbs:
        score += 2 * len(verbs)
        reasons.append(f"action_verbs:{verbs}")

    # 6.2. Императив: ROOT-глагол в базовой форме в начале
    root = _get_root(doc)
    imperative_start = False
    if root is not None and root.lemma_.lower() in ACTION_VERBS:
        # ROOT — первый смысловой токен?
        content_tokens = [t for t in doc if not t.is_space and not t.is_punct]
        if content_tokens and content_tokens[0] == root:
            imperative_start = True
            score += 3
            reasons.append(f"imperative_start:{root.lemma_.lower()}")

    # 6.3. Модальный глагол + глагол действия (MUST/SHOULD и т.д.)
    for token in doc:
        if token.tag_ == "MD" and any(
            child.lemma_.lower() in ACTION_VERBS for child in token.children
        ):
            score += 1
            reasons.append(f"modal+verb:{token.lemma_.lower()}")
            break

    # 6.4. Артефакты
    artifacts = extract_artifacts(doc)
    if artifacts:
        score += 2 * len(artifacts)
        reasons.append(f"artifacts:{artifacts}")

    # 6.5. Shell-подсказка
    if any(t.lemma_.lower() in {"bash", "sh", "shell", "terminal", "echo"}
           for t in doc):
        score += 1
        reasons.append("shell_hint")

    # 6.6. Отрицание
    negation = any(t.lemma_.lower() in {"not", "never"} for t in doc) \
               or any(t.dep_ == "neg" for t in doc)
    if negation and verbs:
        reasons.append("negation")

    # 6.7. Описательные маркеры
    description_markers = {"some", "currently", "fail", "failing",
                           "broken", "error", "issue"}
    if any(t.lemma_.lower() in description_markers for t in doc) \
            and not artifacts and not imperative_start:
        score -= 2
        reasons.append("description_penalty")

    # 6.8. Детектор вопросов
    is_question, q_type, q_reasons = detect_question(doc)
    reasons.extend(q_reasons)
    if is_question:
        score -= 5
        reasons.append("question_penalty")

    # 6.9. Итоговый тип
    if is_question:
        kind = "question"
    elif imperative_start:
        kind = "instruction"
    elif score >= 3:
        kind = "instruction"
    elif score <= -2:
        kind = "description"
    else:
        kind = "mixed"

    return Sentence(
        text=sentence, kind=kind, score=score, is_question=is_question,
        question_type=q_type, verbs=verbs, artifacts=artifacts,
        reasons=reasons,
    )


# --- 7. Публичные функции -----------------------------------------------

def analyze_text(text: str) -> List[Sentence]:
    return [analyze_sentence(s) for s in split_sentences(text)]


def extract_instructions(text: str, min_score: int = 3) -> List[Sentence]:
    return [s for s in analyze_text(text)
            if s.kind == "instruction" and s.score >= min_score]


def extract_questions(text: str) -> List[Sentence]:
    return [s for s in analyze_text(text) if s.kind == "question"]


# --- 8. Демонстрация ----------------------------------------------------

if __name__ == "__main__":
    task_text = (
        "Run `python3 test_mylib.py` in the current directory; "
        "some tests fail. "
        "Why do the tests fail? "
        "Fix the bugs in mylib.py (do not edit the tests) until all tests pass. "
        "Work only inside the current directory. "
        "Is there a backup of mylib.py? "
        "When you are done, write ONLY the word DONE to the file answer.txt "
        "in the current directory (for example with bash: echo 42 > answer.txt). "
        "Check whether the file exists."
    )

    sentences = analyze_text(task_text)

    print("=" * 82)
    print(f"{'ТИП':<12} {'SCORE':>6} {'ВОПРОС?':<10} {'ТЕКСТ'}")
    print("=" * 82)
    for s in sentences:
        q = s.question_type if s.is_question else "—"
        print(f"{s.kind:<12} {s.score:>6} {q:<10} {s.text[:58]}")

    print("\n" + "=" * 82)
    print("ТОЛЬКО ВОПРОСЫ:")
    print("=" * 82)
    for q in extract_questions(task_text):
        print(f"  [{q.question_type}] {q.text}")

    print("\n" + "=" * 82)
    print("ТОЛЬКО ИНСТРУКЦИИ:")
    print("=" * 82)
    for ins in extract_instructions(task_text, min_score=3):
        print(f"  [score={ins.score}] {ins.text[:70]}")