# Объединение и финальная проверка — капитан

Применяйте после получения рабочих commits студентов 1 и 3. Общий контракт и первые три промпта уже готовы; менять их для старта не нужно.

## Сначала общий starter kit

Каждый clone должен содержать одинаковые CSV, файлы организаторов и папку prompts/. Проверьте это сейчас: в просмотренной локальной копии starter kit и prompts/ ещё были untracked, поэтому обычный clone сам по себе их не получит. Один участник публикует эти конкретные файлы в согласованную общую ветку либо передаёт одинаковый архив всем. Не добавляйте ключи, .env и виртуальное окружение.

## Как объединить код

1. Каждый участник фиксирует только свои файлы и передаёт капитану SHA рабочего commit; commit должен быть доступен в опубликованной ветке.
2. Капитан сохраняет собственную работу в commit и проверяет, что рабочая папка чистая.
3. В своём clone капитан создаёт ветку интеграции от своей актуальной ветки с pilot_policy.py и agent.py:

```bash
git switch -c integration/final
git fetch origin
git cherry-pick <SHA_PRIOR>
git cherry-pick <SHA_PORTFOLIO>
```

Замените placeholders настоящими SHA. Если модуль сделан несколькими commits, переносите все нужные commits в исходном порядке. Последующие исправления переносите только новыми SHA, не повторяйте уже перенесённые commits. Если integration/final уже существует, переключитесь на неё вместо повторного создания.

4. Капитан связывает три функции в agent.py:

```text
generate_candidates(profile, history, tariffs)
  -> run_adaptive_pilots(env, candidates)
  -> build_campaigns(env, observations)
  -> список финальных кампаний
```

Передавайте config и trace по SHARED_CONTRACT.md. Отчёт читает agent.last_trace; он не запускает пилоты заново.

5. До финальной передачи чужие ошибки исправляет владелец модуля. После явной остановки работы владельцев капитан может исправить небольшие ошибки интеграции. Не используйте force-push или автоматическое принятие всей стороны при конфликте.

## Короткий промпт для Codex капитана

Скопируйте текст ниже после объединения:

---

Finish and verify our existing Adaptive Campaign Portfolio Agent. Read prompts/SHARED_CONTRACT.md and the integrated modules. Implement small necessary integration fixes; do not redesign the project or change organizer files. Teammates have handed off their final code. Preserve unrelated changes.

Connect candidate_engine.generate_candidates -> pilot_policy.run_adaptive_pilots -> portfolio_optimizer.build_campaigns in Agent.act. Preserve the agreed dictionary keys and normalized lift units. Reset last_trace per run. Keep the runtime offline and ensure every imported custom module is included.

Check actual returned pilot sample sizes, negative observations, residual budgets/contacts, at most 10 final campaigns, 5000-person ID-ordered cap, valid filters and non-self targets. Do not treat free push as risk-free, historical positive_rate as conversion, or unknown pilot overlap as exactly known. Fix report-field mismatches in the integration/report layer without changing the frozen contract.

Run from repository root:

```text
python -m pytest tests -q
python local_eval.py
python local_eval.py --runs 10
python -m tools.benchmark --runs 10 --compare-starter --trace-seed 42
python make_submission.py
python -m reporting --input reports/decision_trace.json --output reports/demo.html
```

Inspect output, not just exit codes: local_eval catches agent exceptions. Require pilots > 0, 1–10 valid final campaigns, no discarded campaigns, resource compliance and runtime below the limit. Evaluator campaign count includes pilots. Compare the actual ten-seed results with starter; report failures honestly and never hardcode mock winners.

Regenerate submission.csv twice from unchanged code with the supplied seed and verify identical content. Validate it using the public validators. Keep the final generated CSV; do not edit it manually. Update README with actual setup commands and measured results, removing unverified claims. If optional benchmark/report tooling is missing, prioritize the official evaluator, submission and runnable README; explicitly report the omission.

Finish with a brief release report: changed files, test results, median/minimum/positive runs, starter comparison, submission reproduction, and remaining limitations. Do not add APIs, services or new features during final verification.

---

## Что передать на сдачу

agent.py, candidate_engine.py, pilot_policy.py, portfolio_optimizer.py, сгенерированный submission.csv, requirements.txt, README.md и остальные реально используемые модули. Сохраните тесты и результаты проверки в репозитории. Финальный push выполняет капитан по согласованному командой процессу.
