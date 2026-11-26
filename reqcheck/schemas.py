"""Схемы структурированных ответов модели (pydantic).

Поля без значений по умолчанию — так проще передавать схему в Gemini
(response_schema) и в Anthropic (tool input_schema). Пустое значение модель
пишет явно: "" или [] или 0.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Kind = Literal[
    "functional", "nonfunctional", "data", "integration", "ui",
    "process", "api", "acceptance", "constraint", "goal", "other",
]
Status = Literal["ok", "violated", "na"]
Severity = Literal["critical", "major", "minor"]


# --- Этап 1. Нарезка --------------------------------------------------------
class Segment(BaseModel):
    doc_id: str = Field(description="Идентификатор требования как в документе (например «Ф-1», «FR-3»); "
                                    "если номера нет — пустая строка")
    section: str = Field(description="Раздел документа, в котором находится требование")
    kind: Kind
    quote: str = Field(description="ДОСЛОВНАЯ цитата требования из документа (без изменений)")
    summary: str = Field(description="Краткая формулировка требования, до 140 символов")
    line_from: int = Field(description="Первая строка цитаты (номер из L###)")
    line_to: int = Field(description="Последняя строка цитаты")


class SegmentationResult(BaseModel):
    document_title: str
    requirements: list[Segment]


# --- Этап 2. Проверка каждого требования ------------------------------------
class CriterionVerdict(BaseModel):
    criterion: str = Field(description="Код критерия: NEC APP UNA COM SIN FEA VER COR CON TRA")
    status: Status
    severity: Severity = Field(description="Если status=violated — важность; иначе minor")
    quote: str = Field(description="Если violated — ДОСЛОВНАЯ цитата из документа, где виден дефект; иначе \"\"")
    problem: str = Field(description="Если violated — в чём конкретно нарушение; иначе \"\"")
    recommendation: str = Field(description="Если violated — как исправить (конкретная формулировка); иначе \"\"")


class RequirementCheck(BaseModel):
    req_id: str = Field(description="Идентификатор в отчёте: R-01, R-02, ...")
    criteria: list[CriterionVerdict]


class RequirementBatchResult(BaseModel):
    checks: list[RequirementCheck]


# --- Этап 3. Проверка набора -------------------------------------------------
class QuoteRef(BaseModel):
    text: str = Field(description="ДОСЛОВНАЯ цитата из документа")


class SetDefect(BaseModel):
    criterion: str = Field(description="Код критерия набора: S-COM S-CNS S-FEA S-CMP S-VAL S-RNK S-MOD S-TRA")
    kind: Literal["contradiction", "omission", "terminology", "traceability", "validation", "other"]
    severity: Severity
    related_req_ids: list[str] = Field(description="R-идентификаторы затронутых требований (может быть пусто)")
    quotes: list[QuoteRef] = Field(description="Минимум одна цитата; для противоречия — обе стороны; "
                                               "для пропуска — цитата места, где ответ должен быть")
    problem: str
    recommendation: str


class ChecklistItem(BaseModel):
    item: str = Field(description="Пункт чек-листа так, как он записан в документе")
    status: Literal["yes", "partial", "no", "not_required"]
    justification: str = Field(description="Обоснование со ссылкой на конкретное место документа")


class SetResult(BaseModel):
    defects: list[SetDefect]
    checklist: list[ChecklistItem] = Field(description="Если в документе есть чек-лист ревью — оценка каждого "
                                                       "его пункта; иначе пустой список")


class ConflictScanResult(BaseModel):
    analysis: str = Field(description="Рабочие заметки: правила, сценарии, где они сталкиваются, и итог разбора (сначала это, потом defects)")
    defects: list[SetDefect]


# --- Этап 3б. Объединение дублей ----------------------------------------------
class DuplicateGroup(BaseModel):
    keep_id: str = Field(description="D-### дефекта, который остаётся (самый полный и точный)")
    drop_ids: list[str] = Field(description="D-### дефектов, описывающих ТУ ЖЕ проблему")
    reason: str


class DuplicateResult(BaseModel):
    groups: list[DuplicateGroup]


# --- Этап 4. Верификация найденного ------------------------------------------
class Decision(BaseModel):
    defect_id: str
    keep: bool
    reason: str


class VerificationResult(BaseModel):
    decisions: list[Decision]


# --- Этап 6. Уточняющие вопросы ------------------------------------------------
Owner = Literal["owner", "analyst", "architect", "security", "qa"]


class Question(BaseModel):
    defect_ids: list[str] = Field(description="D-### дефектов, которые закроет ответ на этот вопрос")
    owner: Owner = Field(description="Кто может ответить: owner — владелец продукта/бизнес, analyst — бизнес-аналитик, "
                                     "architect — архитектор/разработка, security — безопасность, qa — тестирование")
    question: str = Field(description="Один конкретный вопрос автору документа, на который можно ответить решением")
    options: list[str] = Field(description="2–3 готовых варианта ответа (значение или формулировка) либо пустой список")
    blocking: bool = Field(description="true — без ответа нельзя начать разработку или написать тест")


class QuestionsResult(BaseModel):
    questions: list[Question]


# --- Режим ответов автора -------------------------------------------------------
class AnswerAssessment(BaseModel):
    question_id: str = Field(description="Q-NN вопроса, на который дан ответ")
    verdict: Literal["closes", "partial", "insufficient", "contradicts"]
    closed_defect_ids: list[str] = Field(description="D-### дефектов, которые ответ закрывает (для closes — все дефекты вопроса)")
    reason: str = Field(description="1–2 предложения: почему такая оценка")
    proposed_text: str = Field(description="Готовая формулировка требования для вставки в документ; \"\" если ответ не закрывает дефекты")
    follow_up: str = Field(description="Новый, более узкий вопрос, если ответ не закрывает всё; иначе \"\"")
    conflict_quote: str = Field(description="ДОСЛОВНАЯ цитата документа, с которой ответ конфликтует; иначе \"\"")
    conflict_problem: str = Field(description="В чём конфликт ответа с документом; иначе \"\"")


class AnswersResult(BaseModel):
    assessments: list[AnswerAssessment]


# --- Этап 5. Итог ------------------------------------------------------------
class FixItem(BaseModel):
    defect_ids: list[str] = Field(description="Идентификаторы дефектов D-###, которые закрывает это действие")
    action: str = Field(description="Что исправить, конкретно")


class ConclusionText(BaseModel):
    rationale: str = Field(description="2–5 предложений: почему можно/нельзя начинать разработку и тестирование")
    top_fixes: list[FixItem] = Field(description="До 7 действий в порядке приоритета")
