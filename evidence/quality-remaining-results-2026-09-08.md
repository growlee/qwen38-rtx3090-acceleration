# Остальные конфигурации Qwen: результаты 2026-09-08

## Вывод

**DFlash2 fast/n7 — приоритетный кандидат для выделенного LLM-профиля:** на этой
выборке его набор PASS/FAIL полностью совпал с AutoRound BF16-MTP, weighted decode
выше на 76,91%, медиана времени правильного ответа ниже на 29,87%. Это рекомендация,
не переключение production. При необходимости запаса VRAM более экономный вариант —
Fast MTP INT8: тот же набор PASS/FAIL, +16,91% decode против AutoRound.

Huge не рекомендован по этой серии: три новых критических провала, два ответа
выродились в повтор `Register`, почти отсутствует запас VRAM. Причина не локализована:
сравнивались профили целиком, нельзя приписывать это одному размеру контекста или KV.

## Полный учёт матрицы

Проведено **600 новых измеряемых ответов + 3 исключённых warmup**. Все 600 ответов
однократно оценены, инфраструктурных BLOCKED среди них нет. Для архивного Adaptive
записаны отдельные 200 BLOCKED-координат: точные веса/калибровка не найдены, текущий
Adaptive не подставлялся. Таким образом учтены все 800 плановых координат.
Старые 600 ответов расширенной серии и первая 60-case серия не запускались повторно
и не переоценивались. Совпали request SHA всех 200 задач между всеми шестью измеренными
конфигурациями. Ни одного повторного или подстроенного модельного случая.

| Конфигурация | Всего / 200 | Coding / 120 | RU / 40 | Instructions / 20 | Retrieval / 20 | Decode, ток/с |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GGUF Q4_K_M, сохранённый контроль | 134 | 83 | 20 | 11 | 20 | 37,79 |
| AutoRound BF16-MTP k4, сохранённый контроль | 157 | 106 | 19 | 12 | 20 | 135,84 |
| Текущий Adaptive k4, сохранённый контроль | 156 | 105 | 19 | 12 | 20 | 171,64 |
| **Fast MTP INT8 k4** | **157** | **106** | 19 | 12 | 20 | **158,80** |
| **DFlash2 fast/n7** | **157** | **106** | 19 | 12 | 20 | **240,31** |
| DFlash2 huge/n7 | 154 | 104 | 19 | 12 | 19 | 161,37 |
| Архивный Adaptive k4 | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | — |

Ток/с здесь — прежняя frozen weighted-decode метрика: сумма completion tokens,
делённая на сумму измеренных decode seconds. Это короткие разные ответы, не старые
три задачи с фиксированным выходом 1024 токена. Архивные 94,78/101,31/100,51 ток/с
не являются опровергнутыми результатами и напрямую с этой таблицей не сравниваются.
В сравнении с текущим Adaptive DFlash fast быстрее по данной метрике на 40,01%.

## Конкретные изменения качества

- AutoRound → INT8: **0 PASS→FAIL, 0 FAIL→PASS**, новых критических ошибок нет.
- AutoRound → DFlash fast: **0 PASS→FAIL, 0 FAIL→PASS**, новых критических ошибок нет.
- AutoRound или DFlash fast → huge: **3 PASS→FAIL, 0 FAIL→PASS**, все три critical:
  - `expanded-he-140`: функция `fix_spaces` заменяет два пробела одним `_`, а не двумя;
    один из пяти векторов неверен. Это ошибка значения, не форматный отказ.
  - `expanded-fix-checked_percent`: `def Register Register ...` до лимита 2048 токенов;
    `finish_reason=length`, 28,80 секунды, готового решения нет.
  - `expanded-context-420-1`: повтор `Register` до лимита 128 токенов при входе
    11004 токена; готового ответа нет. Это не нагрузка 200k.
- Архивный Adaptive против INT8/текущего Adaptive: сравнение BLOCKED, а не 0/200.

У новых INT8/DFlash fast сохранены те же 3 ошибки исполнения сгенерированных функций,
11 ошибок coding-векторов и не-кодовые провалы AutoRound. `candidate_runtime` в
scorer — исключение сгенерированного Python в изоляции, не падение GPU-сервиса.
У huge две дополнительные обрезки поверх двух общих RU-обрезок. Рубрика, включая
известный строгий `Казань`/`Kazan`, не менялась. По сравнению с GGUF у семейства
AutoRound остаются прежние критические регрессии: «без потерь вообще» не доказано.

## Время и ресурсы

| Режим | Медиана полного ответа, с | Медиана правильного ответа, с | P95 полного ответа, с | Максимум, с | Пик VRAM, MiB | Холодная готовность, с |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AutoRound, сохранённый контроль | 0,616 | 0,625 | 4,445 | 9,398 | 21882 | 178,930 |
| INT8 | 0,527 | 0,548 | 4,429 | 9,391 | 21904 | 143,125 |
| DFlash fast | 0,420 | 0,438 | 3,417 | 9,335 | 23860 | 235,044 |
| DFlash huge | 0,455 | 0,463 | 4,536 | 28,798 | 24122 | 306,527 |

P95 — nearest-rank, 190-е отсортированное значение из 200. Медианы по категориям,
TTFT и все времена конкретных случаев сохранены в comparison/score JSON. Медиана
retrieval: AutoRound 3,170 с, INT8 3,149 с, DFlash fast 3,111 с, huge 4,135 с:
ускорение decode не означает такого же ускорения длинного входа.

Холодная готовность включает компиляцию в изолированном tmpfs-кэше теста, не обещает
такую задержку у прогретого пользовательского сервиса. Во всех новых профилях максимум
температуры 74 C, лимит 300 W; минимальный MemAvailable при измеряемых ответах:
INT8 25 942 003 712 B, fast 25 736 777 728 B, huge 25 353 199 616 B.
Ни одного OOM-контейнера. Huge достиг 24122 MiB при примерно 24128 MiB доступных
GPU: практически без запаса. У fast также мало резерва для другой GPU-нагрузки.

## Идентичность и условия

- Набор `qwen38-quality-expanded-200-v1`, canonical SHA
  `2ea25b195358d0d78a1f4e025c6c63e604693bc00ec4cbc5224e4772e77f187f`.
  120 coding, 40 RU, 20 instructions/tools, 20 retrieval; это адаптированный
  синтетический скрининг, не официальный HumanEval и не полноценный coding-agent.
- Thinking OFF, temperature 0, прежние per-case seed и output caps, concurrency 1,
  один исключённый warmup; максимум фактического prompt **11017 токенов**.
- Target `leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound`, revision
  `f958acb153eb8c7bbd25638149c8c75185559226`, receipt `a1a385…621a`.
  Все 16 файлов проверены по SHA/размеру/custody перед каждым DFlash.
- Образ `ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-69ba4d0`, digest
  `sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0`.
  Runtime и его process arguments зафиксированы в `window-report.json`.
- INT8: все 18 итоговых файлов точно совпали с историческим manifest; receipt
  `33ef6b…89521`. Скопированы только три изменяемых файла; 13 неизменяемых — hardlink
  с индивидуальными RO bind поверх них в подготовительном CPU-контейнере. Все
  upstream backups сохранены. Исторические физические счётчики receipt — provenance,
  сегодняшние значения отдельно в `fast-recovery-evidence.json`.
- DFlash: revision `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`, receipt `4908dc…88238`.
  Один drafter 1 280 636 224 логических байта скачан после CAS-проверки и совпал по SHA.
  Fast: CTX=fast, n7, MAX=65536, GPU_UTIL=.93, BF16 KV. Huge: CTX=huge,
  n7, MAX=245760, GPU_UTIL=.97, `kvarn_k4v2_g128` KV. PREFIX_CACHE=0, VISION=0.
  Ограничения huge не смягчались; второй drafter не скачивался.
- Архивные Adaptive helper bytes найдены с исходными SHA в home_lab. Отсутствуют
  точные weights/calibration (historical Hessians `1e272c…b470`); текущая калибровка
  отличается. Новая калибровка не делалась и за историческую не выдавалась.

Сравнение с тремя сохранёнными контролями межсессионное. По одному ответу на задачу
не доказывает отсутствие деградации вообще, воспроизводимость процента ускорения
или работу на 200k. Причина двух вырожденных huge-ответов в этой конечной матрице
не диагностировалась и не исправлялась.

## Возврат и артефакты

Окно capture/preparation/restore: **1630 с (27:10)**, `captured_restored`, оба кода
ошибки null. После него CPU-оценка выполнена один раз. Финальный live-снимок
**2026-09-08 11:00:10 UTC**: исходные container/image/model/CTXfast/MTPk4/.93/57344
сохранены, Qwen/gateway/WebUI healthy, public admission, очередь/ledger пусты.
Два authenticated gateway completion PASS. Оба transient unit inactive/dead exit0;
366 тестовых контейнеров retained/stopped, OOM0, nonzero exit0, volume mounts0.

Новый расход диска **3 263 086 592 B (3,04 GiB)** из лимита 7 GiB; свободно
203 587 833 856 B (189,61 GiB), резерв 150 GiB соблюдён. INT8+backups и один
drafter оставлены и учтены. Модели, кэши, данные и контейнеры не удалялись;
постоянные профили/маршруты не менялись, компьютеры не перезагружались.

Локальные доказательства: `E:\Projects\home_lab\outputs\qwen-quality-remaining-20260908`:
`comparison-remaining.json`, `window-report.json`, `grading-summary.json`,
`responses-*.jsonl`, `scores-*.jsonl`, `final-runtime.json`, `fast-recovery-evidence.json`,
`execution.md`, точные controller/preparation/scorer sources и bundle manifest.
Серверный пакет `/srv/ai-server/benchmarks/quality-remaining-20260908`, ответы в `results`.
Manifest `0a6ba209c1a9320743c9e455af3934afe47f9d856e7c01e35ff6665cb2ab35e3`.

Команды единственного выполнения (на `server-a`, через Remote SSH), внутри transient units:

```text
/usr/bin/python3 -B /srv/ai-server/benchmarks/quality-remaining-20260908/run_quality_window_remaining.py --output-dir /srv/ai-server/benchmarks/quality-remaining-20260908/results --bundle-manifest-sha256 0a6ba209c1a9320743c9e455af3934afe47f9d856e7c01e35ff6665cb2ab35e3
/usr/bin/python3 -B /srv/ai-server/benchmarks/quality-remaining-20260908/grade_remaining.py 0a6ba209c1a9320743c9e455af3934afe47f9d856e7c01e35ff6665cb2ab35e3
```

Это запись выполненных команд, не приглашение повторить их: существующие outputs,
имена контейнеров и O_EXCL намеренно запрещают повтор. Задача конечной матрицы завершена.
