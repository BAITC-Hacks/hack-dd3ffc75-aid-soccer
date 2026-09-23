# Архитектура Adaptive Campaign Portfolio Agent

## Поток решения

1. **Candidate ranking** строит проверяемые гипотезы из истории смен тарифов.
   Исторический lift используется только как prior/ranking signal.
2. **Adaptive pilots** получают новые наблюдения на текущей аудитории и приводят
   их к единицам hypothetical channel multiplier `1.0`.
3. **Portfolio optimizer** независимо восстанавливает аудитории из
   `env.customer_profile`, рассчитывает экономику каналов и выбирает набор
   кампаний bounded beam search.
4. **Agent coordinator** объединяет модули и хранит JSON-compatible trace.
5. **Reporter** читает сохранённый trace и создаёт self-contained HTML; он не
   вызывает агента, пилоты или evaluator.

Границы модулей держат runtime простым: optimizer принимает plain dictionaries,
не импортирует candidate/pilot modules и не меняет счётчики среды. Это позволяет
тестировать его на маленьком FakeEnv до интеграции с остальными ветками.

## Почему bounded beam search

Обычная сортировка по standalone profit может потратить весь лимит на одну
крупную кампанию, хотя несколько меньших дают больший суммарный conservative
net. Для каждой cell поиск рассматривает `skip` и допустимые target/channel
альтернативы. State хранит точные contacts, cost, campaign count, выбранные
options и objective.

На каждом шаге поиск:

- отбрасывает варианты, не помещающиеся целиком в остаток ресурсов;
- объединяет состояния с одинаковыми resource totals;
- сохраняет представителей разных областей бюджета/охвата;
- применяет детерминированные tie-breakers;
- сравнивает результат с greedy incumbent и не возвращает худший portfolio.

Это bounded heuristic ширины 64, а не заявление о глобальном оптимуме. На
маленькой синтетической задаче тест дополнительно сравнивает результат с полным
перебором.

## Согласование со scorer

Аудитория определяется только документированными фильтрами current tariff и
ARPU segment. Строки сортируются по числовому `ID_NUMBER`, после чего берутся
первые `min(5000, cell size)`. Стоимость и контакты рассчитываются для всей этой
effective audience — план не полагается на случайное усечение scorer по бюджету.

Для измеренного normalized lift:

```text
mean gross = mean_lift × channel_multiplier × served_arpu
safe gross = safe_lift × channel_multiplier × served_arpu
safe net   = safe gross − served_size × cost_per_contact
```

Multiplier применяется один раз. Отрицательный safe lift не обнуляется. Call
не входит в обычную эксплуатацию из-за возможного saturation; поддерживаются
push, SMS и digital ads.

## Trace и честность оценки

Trace отделяет известные факты (стоимость, аудитория, pilot sample size),
оценки (mean/safe lift, conservative final-only proxy) и неизвестное (реальное
пересечение pilot/final audience). `observed_lift_total` пилотов не добавляется к
proxy objective. Реальный mock net gain показывается только как отдельный
результат внешнего evaluator.

Fallback явно называется emergency и сохраняет оценённый downside либо
отсутствие уверенности. Пустая или невыполнимая аудитория даёт infeasible event,
а не выдуманную кампанию.

## Связь с judging rubric

### Functionality — 25

- публичная функция возвращает только документированные campaign keys;
- проверяются non-self transition, существующий target и поддерживаемый channel;
- финальный план повторно валидируется по budget, contacts и campaign count;
- sanitizer/validator из публичного scorer включены в тест.

### Technical quality — 25

- экономика пересчитывается по реальной effective audience;
- uncertainty переносится между каналами в нормализованных единицах;
- beam search учитывает multiple-choice и совместные ограничения;
- решения воспроизводимы, входы и env counters не мутируются.

### Reproducibility — 25

- независимые pytest fixtures не используют mock truth;
- версии проверенного окружения записаны в README;
- reporter работает из сохранённого trace;
- benchmark-команды и место для 10-seed evidence зафиксированы без выдуманных
  результатов.

### Applicability — 15

- отчёт показывает расходы, остатки ресурсов, uncertainty и причины отказа;
- emergency fallback не маскирует бизнес-риск;
- production path требует экспериментов, calibration, drift monitoring и
  approval.

### Originality — 10

- адаптивная разведка отделена от portfolio allocation;
- bounded multiple-choice search сильнее независимого выбора кампаний;
- audit trace связывает техническое решение с объяснением для аналитика.

## Известные ограничения

Исторические switchers не являются случайной выборкой; before/after изменение
не устанавливает причинность. Mock и hidden effects различаются. Интервалы после
адаптивного отбора не являются simultaneous guarantees. Идентификаторы
pilot-клиентов недоступны, поэтому точная дедупликация с финальными кампаниями
невозможна на стороне planner.


## Optional LLM decision module

`llm_advisor.py` supplies bounded exploration nominations between candidate
generation and pilots. The release replays `artifacts/llm_policy.json` only for
identical aggregate public inputs. Changed contexts fall back to the numerical
policy. Explicit live runs can refresh the artifact for a new dataset. Existing
shortlist members keep their order; novel nominations can replace up to two
positions. Measured pilot effects and portfolio constraints remain numerical.

`adaptive_confirmation_n=True` additionally sizes confirmations from current
uncertainty near profit, channel, and competing-target boundaries. It is opt-in
because the measured ten-seed minimum declined. See
[LLM integration](LLM_INTEGRATION.md) for API setup, actual results, replay, and
the reviewed training-data workflow. No model weights have been trained.
