# Adaptive Campaign Portfolio Agent

Агент для синтетического кейса Beeline: история ранжирует гипотезы, SMS-пилоты
измеряют эффект на текущей аудитории, оптимизатор выбирает до 10 кампаний с
учётом неопределённости, бюджета и охвата. Интегрированная версия запускается
**из корня репозитория**. По умолчанию он работает без сети и ключей API.

На ветке `feature/llm-integration` добавлена опциональная LLM-интеграция:
[настройка ключа, режимы и подготовка обучающих примеров](docs/LLM_INTEGRATION.md).
Ключ вставляется локально в `.env`; официальная генерация submission оставляет
LLM выключенной. Реальная API-проверка и обучение модели ещё не выполнялись.

## Запуск

Проверено на Python 3.12.3 с pandas 3.0.3, NumPy 2.4.4, pytest 9.0.2.
Установка зависимостей требует доступа к пакету Python; работа агента — нет.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python local_eval.py
python local_eval.py --runs 10
python -m tools.benchmark --runs 10 --compare-starter --trace-seed 42
python make_submission.py
python -m reporting --input reports/decision_trace.json --output reports/demo.html
```

Для запуска без тестов достаточно `requirements.txt`. Все публичные CSV,
неизменённые скрипты организаторов и рабочие инструкции `PROMTS/` включены в
репозиторий. Копия `beeline_case_participants ` с пробелом в конце — история
предыдущего прототипа; команды выше используют корневые модули и данные.

## Измеренные результаты

Ниже результаты проверенной политики без LLM; они не являются результатами
LLM-assisted режима. На новой ветке прошли **125 офлайн-тестов**, а default
10-seed результаты и submission сохранились. Проверки подготовки новой ветки: `reports/llm_preparation_tests.txt`
и `reports/llm_preparation_verification.json`.

Публичный mock evaluator, seed 0–9, один и тот же набор seed для обоих агентов:

| Метрика net gain, у.е. | Интегрированный агент | Исходный agent_template |
|---|---:|---:|
| Среднее | 2,898,763.77 | -480,836.18 |
| Медиана | 3,103,097.10 | -357,947.80 |
| Минимум | 1,567,730.55 | -1,019,431.06 |
| Положительных прогонов | 10/10 | 0/10 |

Побед по одинаковым seed: **10/10**. Разница медиан: **3,461,044.89** у.е.
Starter не изменялся: на seed 3 он возвращает ноль финальных кампаний, а его
реальный pilot-only результат включён в сравнение. Это отдельно отмечено
`meets_final_count_requirement=false`; успешное вычисление не означает
соответствия требованию 1–10 кампаний.

На базовом релизе **93 теста прошли**, в том числе в чистом экспорте без `task/`; submission
в нём совпал побайтово. Во всех десяти adaptive-прогонах 15 пилотов, 3–6 финальных
кампаний, валидные непустые non-self аудитории, различные финальные ячейки,
нет отброшенных кампаний или усечения по оставшемуся бюджету/охвату.
Максимальные расходы: **96,542/100 000**;
максимальные контакты: **8,201/15 000**.
Максимальное измеренное время полного adaptive-evaluation: **0.554 с**
при лимите 600 с; это замер на текущей машине, не обещание для любого окружения.

Seed 42: net gain **3,206,241.49**, **15 пилотов + 4 финальные кампании**,
**74 256** расходов, **4 848** контактов. Число 19 в официальном отчёте включает
пилоты; лимит 10 относится к финальному списку. Напечатанный evaluator остаток
бюджета 92 800 — после пилотов; после всего плана остаётся **25 744**.

Доказательства: [benchmark](reports/benchmark.md), [все прогоны JSON](reports/benchmark.json),
[официальные 10 seed](reports/local_eval_10_seeds.txt), [seed 42](reports/local_eval_seed42.txt),
[тесты](reports/tests.txt), [отчёт HTML](reports/demo.html),
[итоговая передача](reports/FINAL_RELEASE.md).

## Submission

`submission.csv` создан неизменённым `make_submission.py` с seed 42. Два запуска
дали одинаковые байты; `sanitize_campaigns` и `validate_strategy` приняли все
четыре кампании. Файл совпадает с решениями в `reports/decision_trace.json`.

```bash
python make_submission.py
cp submission.csv /tmp/beeline-submission-first.csv
python make_submission.py
cmp submission.csv /tmp/beeline-submission-first.csv
```

SHA-256: `3cba2865d369ad86c93e110476a4436c42f25c3583cbb9265382844e8e848e5e`.
Подробная проверка: [submission_verification.json](reports/submission_verification.json).
CSV не редактируется вручную. На скрытой среде Agent может выбрать другой план.

## Модули и логика

```text
generate_candidates(profile, history, tariffs)
  -> run_adaptive_pilots(env, candidates)
  -> build_campaigns(env, observations)
  -> Agent.act(env): list[dict]
```

- `candidate_engine.py`: robust lift среди switchers, shrinkage n/(n+50),
  разнообразие ячеек и каталожные альтернативы; default pool 40.
- `pilot_policy.py`: до 10 SMS-разведок по 80 контактов и до 5 адаптивных
  подтверждений по 200. Учитываются фактически возвращённые размеры выборок,
  отрицательные наблюдения, остатки ресурсов; необязательные слоты не расходуются.
- `portfolio_optimizer.py`: push/SMS/digital_ads, один target/channel на ячейку,
  целые эффективные аудитории в numeric-ID порядке с пределом 5000. Bounded beam
  search совместно выбирает каналы, бюджет, охват и skip; сравнивается с greedy.
- `agent.py`: передаёт общий config/trace, читает публичную историю относительно
  файла, сбрасывает `last_trace` на каждый запуск, проверяет финальные лимиты.
- `tools/benchmark.py`: внешний запуск официального evaluator, проверка реальных
  фильтров/ресурсов, сравнение со starter, JSON/CSV/Markdown и сохранение trace.
- `reporting.py`: рендерит только сохранённый trace, не запускает пилоты. Реальный
  evaluator net gain прикрепляет benchmark отдельно от оценок планировщика.

Mean/SE/safe lift после пилота нормализованы на channel multiplier 1.0;
экономика умножает lift на коэффициент канала один раз. Call исключён из-за
насыщения конверсии. Исторический positive_rate не является конверсией,
prior_score — не прогноз прибыли. При отсутствии положительного conservative
net аварийный push помечается как рискованный fallback; бесплатный контакт
может уменьшить выручку.

## Ограничения и целостность

История и текущая аудитория не пересекаются по ID. Before/after изменение среди
switchers не доказывает причинный эффект. После адаптивного отбора uncertainty
bands являются эвристическими. Поиск ограниченной ширины не гарантирует
глобальный оптимум. Точные ID пилотов недоступны: final-only денежные оценки
не включают выдуманную дедупликацию с пилотами. Net gain измеряет официальный
evaluator, включая пилоты и настоящую дедупликацию.

Все данные синтетические, mock effects отличаются от скрытых; эти числа не
являются прогнозом реального бизнеса или результатом судейства. Никаких
настроек под конкретные seed/тарифы при интеграции не добавлено.

[Контракт](PROMTS/SHARED_CONTRACT.md), [аудит](docs/DATA_FINDINGS.md),
[архитектура](docs/ARCHITECTURE.md), [демо](docs/DEMO.md).
[Контрольные суммы](reports/organizer_integrity.json) подтверждают побайтовое
совпадение 15 корневых файлов организаторов/данных с исходным пакетом.
