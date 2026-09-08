# Adaptive k4 с новой калибровкой: результат 2026-09-08

## Вывод

Реконструкция выполнена и проверена на прежних 200 задачах ровно один раз:
**156 PASS, 44 FAIL, 0 BLOCKED; weighted decode 174,17 ток/с**.
Оснований заменить текущий Adaptive нет: decode выше всего на 1,47%, медиана
полного ответа немного хуже, а одинаковый балл скрывает разные ошибки.
Это межсессионное сравнение, не доказательство воспроизводимого выигрыша 1,47%.

Рекомендация предыдущей серии сохраняется: DFlash2 fast/n7 — приоритетный кандидат
для выделенного LLM-профиля, не автоматическое переключение. Рабочий Adaptive
восстановлен, пользовательский доступ проверен отдельно от качества.

## Идентичность

Пользователь отдельно разрешил новую калибровку после доказанного отсутствия
исторических весов/Hessians. Новый кандидат `adaptive-k4-reconstructed-20260908`
не подменяет архивную строку: **точный исторический Adaptive остаётся BLOCKED**.

- Исходный AutoRound revision `f958acb153eb8c7bbd25638149c8c75185559226`, receipt
  `a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a`.
  Все 16 исходных файлов проверены до и после подготовки. Неизменяемые тела весов
  переиспользованы hardlink с read-only защитой в подготовительных контейнерах.
- Исторический syv recipe revision `69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c`;
  точные adaptive helpers SHA `936ddad9d726b75e0a170368a2c8620e356d65285685792665a13b56e336fea1`
  и `51306be000fd3c0563b8fec6c5a2859c1b91f60dcc6ca44dcf6e2476566e7d76`.
- Калибровка: 128 записей из восьми закреплённых базовых prompts, максимум входа
  259 при лимите512, максимум выхода1024. Собрано135547 строк активаций, незавершённых
  последовательностей0; lm_head использовал115547 строк. Это ограниченный
  синтетический корпус, не полноценный coding-корпус.
- Новый Hessians SHA `af92591a9510f1be42105aa2a1abf0a80bb8b53dfa4f273f2b49ed7596bdff39`
  отличается от исторического `1e272c…b470`. Все восемь проверок матриц PASS,
  первоначальный damping factor1, RTN fallback отсутствует.
- Итог: lm_head INT4, MTP INT4, group128, vocab40960, MTP k4, CTX fast,
  MAX_LEN57344, GPU_UTIL .93, BF16 KV, PREFIX_CACHE0, vision OFF.
  Реальные process arguments/draft head подтверждены в window report.
- Новый receipt SHA `6b098f064a894e3efccf1adba96aabcaee88025e52ab01bde13857f7cb72011c`.
- Образ `sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0`,
  runtime сообщает vLLM0.27.1. Новые образы не скачивались и не собирались.

## Сравнение

Все 200 request SHA совпадают между семью измеренными конфигурациями.
Новые: 200 ответов + один исключённый warmup. Старые ответы не повторялись
и не переоценивались; рубрика/критичность/параметры не менялись.

| Режим | Всего /200 | Coding /120 | RU /40 | Instructions /20 | Retrieval /20 | Decode, ток/с |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GGUF, сохранённый | 134 | 83 | 20 | 11 | 20 | 37,79 |
| AutoRound BF16-MTP, сохранённый | 157 | 106 | 19 | 12 | 20 | 135,84 |
| Текущий Adaptive, сохранённый | 156 | 105 | 19 | 12 | 20 | 171,64 |
| Fast MTP INT8, сохранённый | 157 | 106 | 19 | 12 | 20 | 158,80 |
| DFlash2 fast/n7, сохранённый | 157 | 106 | 19 | 12 | 20 | 240,31 |
| DFlash2 huge/n7, сохранённый | 154 | 104 | 19 | 12 | 19 | 161,37 |
| **Adaptive, новая калибровка** | **156** | **105** | **18** | **13** | **20** | **174,17** |
| Точный исторический Adaptive | BLOCKED | — | — | — | — | — |

Против текущего Adaptive: один PASS→FAIL и один FAIL→PASS, новых critical FAIL0.
Против AutoRound/INT8/DFlash fast: два PASS→FAIL, один FAIL→PASS, один новый critical FAIL.

- `expanded-he-103`: как у текущего Adaptive, пропущено деление на2 в среднем.
  `(1,5)` даёт `0b110` вместо `0b11`; неверны10/12 векторов. Алгоритмическая ошибка.
- `expanded-ru-09`: к300г сплава с40% меди для достижения60% надо добавить150г;
  реконструкция ответила100. Ранее этот случай проходил.
- `expanded-tool-01`: теперь `city: Казань`, а не `Kazan`, поэтому strict scorer
  засчитал PASS. Это соответствие буквальному эталону, не доказательство общего
  улучшения tool calling. Ранее описанное ограничение рубрики сохраняется.

Три ошибки исполнения сгенерированных функций и две RU-обрезки сохранились.
`candidate_runtime` означает ошибку Python-ответа в песочнице, не сбой GPU.
FAIL не тюнились и не повторялись.

## Время, ресурсы, ограничения

| Метрика | Текущий Adaptive, сохранённый | Новая калибровка |
| --- | ---: | ---: |
| Медиана полного ответа, с | 0,492185 | 0,498601 |
| Медиана правильного ответа, с | 0,502358 | 0,521677 |
| Медиана TTFT, с | 0,193921 | 0,193538 |
| P95 полного ответа, с | 4,400213 | 3,575835 |
| Пик VRAM, MiB | 21962 | 21962 |

Новый cold-ready148,231с; максимум ответа9,362с. Во время измеряемых ответов:
максимум74C, минимум MemAvailable26038456320B, power limit300W (мгновенный
измеренный максимум300,77W). Нет OOM.

Decode — сумма completion tokens / сумма измеренных decode seconds, не скорость
полного пользовательского запроса. Новые ответы суммарно короче:10111 completion
tokens против10856 у текущего Adaptive. Выигрыш1,47% нельзя отделить от вариативности
серии и состава ответов без отдельного опыта. Архивный1024-token benchmark с этим
числом напрямую не сравнивается. P95 — nearest-rank,190-е значение из200.

Набор `qwen38-quality-expanded-200-v1`, canonical SHA
`2ea25b195358d0d78a1f4e025c6c63e604693bc00ec4cbc5224e4772e77f187f`.
Thinking OFF, temperature0, прежние seeds/caps, concurrency1. Максимальный prompt
11017 токенов: **200k не проверены**. Это адаптированный скрининг, не официальный
HumanEval и не многошаговый coding-agent. Сравнивались конфигурации целиком;
причинный эффект одной MTP-квантизации не изолирован.

## Возврат и доказательства

Окно11:32:29–12:09:31 UTC,2222с (37:02), `captured_restored`, error/restore_error
null. После возврата выполнена одна CPU-оценка. Финальный read-only снимок12:12:53 UTC:

- Исходный container `15fc9d27b9c035e8d1304c0fcfc5c6a3170b4dc2c00e32a3b178696465f2ea5f`,
  image и model `/srv/ai-server/artifacts/extensions/qwen-syv-k4-4d4166b891489bfc/model`
  сохранены. Qwen ready/healthy; fast/MTPk4/57344/.93/PREFIX_CACHE0.
- Два authenticated gateway completion PASS; public admission, очередь/ledger
  пусты, Gateway/Open WebUI healthy. Реконструкция не назначена рабочим сервисом.
- Все133 новых контейнера stopped/retained, OOM0, ненулевых exit code0.
- Свободно191961800704B (178,78GiB). Дельта свободного места от pre-upload
  11624923136B (10,83GiB) — всего filesystem, не точный эксклюзивный размер модели.
  Бюджет21260679496B (~19,80GiB), резерв150GiB соблюдены.
- Два отсутствовавших BF16-фрагмента скачаны по pinned revision один раз,
  суммарно5085604168B; размер/SHA проверены до использования. Ingress, калибровка,
  backups, candidate и остановленные контейнеры сохранены. Удалений нет.

Серверный пакет `/srv/ai-server/benchmarks/qwen-adaptive-reconstruction-20260908`,
manifest SHA `545e26c70783d7bbbd967521107482b5e0e51b469ba6bf0f632b742dceed00db`.
Точный пакет получил одну независимую Sol-проверку PASS.11 подготовительных шагов
завершены один раз, затем один capture и один scorer; оба transient unit exit0.

Кандидат сохранён в
`/srv/ai-server/staging/experiments/qwen38-syv-uncensored/qwen-adaptive-reconstruction-20260908-candidate`;
калибровка в соседнем `qwen-adaptive-reconstruction-20260908-work`.

Локальные доказательства: `E:\Projects\home_lab\outputs\qwen-adaptive-reconstruction-20260908`:
`comparison-reconstruction.json`, `window-report.json`, `preparation-evidence.json`,
`final-runtime.json`, raw/scored JSONL, frozen sources/manifest.
Raw SHA `301d2f0726bd246d05bd3554cbbe15afe6e76a14c0133336b99362092ec42cc3`,
scored SHA `5ad98e494a7da971cc397161d0c4072709a569cb58fc1be877215364f36d7031`.
Raw/scored/window/preparation совпали с сервером побайтно. В локальной копии
`grading-summary.json` добавлен только завершающий LF; исходный SHA совпадает после
его удаления. Полные подготовительные логи и receipt сохранены на сервере.

Единственные команды выполнения через Remote SSH `server-a` (не повторять):

```text
/usr/bin/python3 -B /srv/ai-server/benchmarks/qwen-adaptive-reconstruction-20260908/run_reconstruction.py 545e26c70783d7bbbd967521107482b5e0e51b469ba6bf0f632b742dceed00db
/usr/bin/python3 -B /srv/ai-server/benchmarks/qwen-adaptive-reconstruction-20260908/grade_reconstruction.py 545e26c70783d7bbbd967521107482b5e0e51b469ba6bf0f632b742dceed00db
```

Реконструкция и её конечный тест завершены. Архивная строка остаётся невоспроизведённой;
новых тестов, promotion, очистки и постоянных изменений нет.
