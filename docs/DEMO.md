# Демо на 2–3 минуты

## 0:00–0:25 — задача

«У нас 23 тысячи синтетических абонентов, ограниченные бюджет и охват. Нужно
выбрать до десяти кампаний: аудиторию, целевой тариф и канал. Цель — не просто
рост ARPU, а чистый прирост после стоимости коммуникации».

Показать README и схему: candidate ranking → adaptive pilots → normalized
observations → portfolio search → trace/report.

## 0:25–1:05 — как evidence меняет решение

Открыть HTML-блок **Pilot evidence and uncertainty**. Выбрать реальное событие
текущего trace, где pilot mean и safe lift отличаются достаточно, чтобы
изменить channel или убрать кандидата. Сказать:

«Исторический score только предложил эту гипотезу. Финальное решение использует
пилот на текущей аудитории. Средняя оценка выглядит привлекательной, но нижняя
эвристическая граница учитывает шум. Поэтому дорогой канал [название из trace]
не прошёл conservative net».

Если такого события в реальном trace нет, честно показать синтетический тест
`test_zero_cost_push_and_negative_safe_lift_use_honest_fallback` и прямо назвать
его тестовым сценарием, а не live run.

## 1:05–1:40 — portfolio tradeoff

Показать тест
`test_beam_finds_cheaper_combination_that_beats_standalone_greedy`:

«Standalone greedy берёт одну крупную кампанию с proxy-objective 1430. Две
меньшие кампании целиком помещаются в те же 20 контактов и дают 1600. Beam search
рассматривает skip и альтернативы по каждой cell совместно. На маленьком примере
результат совпадает с полным перебором».

Уточнить, что это bounded heuristic, а не доказательство глобальной
оптимальности на полном наборе.

## 1:40–2:15 — соблюдение ограничений

Показать верхние карточки HTML:

- pilot spend и planned final spend;
- total contacts;
- budget/contacts after plan;
- список selected campaigns.

«Оптимизатор начинает с остатка после пилотов. Каждая финальная аудитория
пересчитывается из профиля, сортируется по числовому ID и ограничивается 5000.
Мы не рассчитываем на то, что scorer случайно обрежет не помещающуюся кампанию».

## 2:15–2:40 — воспроизводимость

Запустить:

```bash
python -m pytest tests/test_portfolio_optimizer.py tests/test_reporting.py -q
python local_eval.py --runs 10
```

Показать `reports/benchmark.md`: 10/10 положительных adaptive-прогонов,
медиана 3 103 097.10 у.е., минимум 1 567 730.55. У неизменённого starter
медиана −357 947.80 и 0/10 положительных; на seed 3 нет финальных кампаний.
Это фактическая mock-оценка, а не обещание результата на скрытых эффектах.

## 2:40–3:00 — честные границы

«Final-only objective — оценочный proxy. Пилоты могут пересекаться с финальными
аудиториями, их точные IDs planner не знает. История не даёт причинной оценки,
а hidden effects отличаются от mock. В production нужны randomized controls,
propensity logging, calibration, drift monitoring и human approval».
