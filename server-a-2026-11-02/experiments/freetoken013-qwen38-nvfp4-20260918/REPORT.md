# FreeToken 0.1.3 / Qwen3.8-27B NVFP4 — PASS, 2026-09-18

Модель загрузилась на RTX 3090; loopback endpoint работал; все пять согласованных streaming-запросов прошли. Повторное использование контекста подтверждено серверными cache hits. После проверки FreeToken остановлен, исходное состояние Qwen и admission восстановлено. Это конечная smoke-проверка, не бенчмарк качества или устойчивой производительности.

## Установленный runtime и действительная модель

- FreeToken `0.1.3`, commit `cac247a860e316e06580d05aeb05f2e647bde214`.
- Образ `local/ai-server-freetoken:0.1.3-cac247a-nvfp4-20260918`; ID `sha256:10bc9d29f79941c71bb27a77974be4b2312ae64f02636dd5445db20e77675134`.
- Реально загружена согласованная `preetpatel/Qwen3.8-27B-Uncensored-NVFP4`, revision `37b5130a2d2a1f7d4456ab3f8d05d0b2a45ea350`.
- Проверенные существующие веса: 19,709,883,808 bytes; SHA-256 `bc642c1159ab9b0fcd2fdc65be4fee72fa560cf0570604741db50061ea063277`. Model directory, CAS weights и tokenizer смонтированы read-only. Веса не копировались и не скачивались.
- Python 3.12.3, Torch 2.11.0 / CUDA 13, Triton 3.6.0, Transformers 5.16.1, Torchvision 0.26.0, flashlib 0.3.0, freetoken-kernel-cache 0.1.3+cu130; все 95 версий закреплены и проверены в `receipts/installed-versions.json`. GCC 13 и nvcc 13.0.88 присутствуют.
- Runtime `/opt/freetoken013` внутри контейнера; каталог эксперимента `/srv/ai-server/artifacts/experiments/freetoken013-qwen38-nvfp4-20260918`.

Первоначальная `leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound` revision `f958acb153eb8c7bbd25638149c8c75185559226` не загружалась: INT4/INT8 compressed-tensors integer schemes отвергаются `_scheme_of` в выбранном commit. Историческое доказательство — `../qwen38-20260918/RESULT.md`. Этот успешный запуск относится к согласованной NVFP4; сравнивать его напрямую с прежними AutoRound/vLLM TTFT как чистое сравнение движков нельзя.

## Настройки и результаты

Контекст/KV 32768, concurrency 1, text-only, thinking off, temperature 0, max output 256, max prefill chunk 8192. Attention Triton; NVFP4 linear Triton W4A16 на SM86. `--cache-type radix` разрешился в `hybrid_radix` с повторным использованием GDN state; cache report включён.

| Запрос | Вход / выход, токены | TTFT, с | Cached tokens | Пик VRAM, MiB | Скорость после первого content, ток/с | Triton compile, с |
|---|---:|---:|---:|---:|---:|---:|
| Короткий streaming | 24 / 26 | 7.209 | 0 | 22194 | 40.72 | 3.908 |
| ~8k новый | 7976 / 47 | 12.424 | 0 | 23334 | 37.91 | 2.944 |
| ~8k продолжение | 8049 / 40 | 1.903 | 8023 | 23334 | 37.89 | 1.514 |
| ~28k новый | 27994 / 47 | 27.883 | 0 | 23394 | 33.96 | 0 |
| ~28k продолжение | 28067 / 40 | 0.214 | 28041 | 23394 | 33.61 | 0 |

Все HTTP 200, SSE завершены `[DONE]`, finish_reason `stop`, reasoning_chars=0. Ответы осмысленные; длинные запросы описали фильтрацию неотрицательных чисел и циклическую прибавку 0–6, продолжения предложили соответствующий unit test.

TTFT измерен от начала запроса до первого непустого `delta.content`. Времена необработанные: JIT не вычитался. Колонка compile — сумма событий Triton compiler listener, не полный профайлер всех возможных затрат JIT. Скорость после первого content — оценка `(output_tokens - 1) / (duration - TTFT)`, а не чистый GPU decode benchmark.

Наблюдаемое отношение TTFT: **6.53×** для 8k и **130.28×** для 28k. Для 8k сравнение включает разные затраты JIT. В паре 28k новых Triton-компиляций не было. В обоих продолжениях сервер повторно использовал весь предыдущий запрос вместе с ответом и обработал только 26 новых prompt-токенов. Это подтверждает пользу сохранённой истории в данном smoke, без гарантии таких отношений на произвольном коде.

Короткий запрос выполнен один раз в `run-20260918T084024672232Z`; остальные четыре — один раз в `run-20260918T084817863237Z`. Между ними исправлен только тестовый подсчёт токенов. Старое `local_prompt_tokens=2` у короткого запроса ошибочно считало поля BatchEncoding; авторитетное серверное usage=24 отдельно воспроизведено токенизатором. У всех четырёх длинных запросов локальные и серверные длины совпали.

## Загрузка и устранённые причины отказов

Финальный старт до проверенной готовности endpoint: **45.528 с**; отдельно веса 13.59 с, CUDA graphs 6.20 с, prefill warmup 10.216 с. Пик во время старта 22194 MiB; пик запросов **23394 MiB = 22.85 GiB**. OOM не было.

Исправления ограничены контейнером и тестовой обвязкой; образ и зависимости после установки не пересобирались:

1. Убрана искусственная остановка при `MemAvailable < 27 GiB`. Недобор порядка 50 MB не доказывал невозможность запуска. Фактическая RAM записывается; лимит контейнера 24 GiB остаётся.
2. `/cache` tmpfs получил `exec,nosuid,nodev`: иначе Triton `.so` не мог загрузиться (`failed to map segment`).
3. Задан `TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor`: UID 65532 отсутствует в passwd, поэтому default cache path через getpass ломал импорт Torch. Повторный частичный импорт давал вторичное `Artifact of type=precompile already registered`.
4. Dedicated bridge `freetoken013-nvfp4-loopback` заменил sole internal bridge, на котором Docker 29.8 не устанавливал фактическую публикацию порта. Только `127.0.0.1:19191 -> 1919`; masquerading и ICC отключены. Это не заявление о полном запрете любого исходящего трафика. CPU HTTP-проба и live `/health`, `/v1/stats`, `/v1/cache/status` подтвердили host-loopback доступ. [Docker bridge contract](https://docs.docker.com/engine/network/drivers/bridge/), [Moby reproduction](https://github.com/moby/moby/discussions/53256).
5. Готовность проверяется по `/health` и startup `cache/status.geometry` (32768 pages × 1 token), поскольку `/v1/stats.kv` до первого ответа равен null.
6. Токенизатор вызывается с `return_dict=False`: Transformers 5.16.1 по умолчанию возвращает BatchEncoding. Исправленный тест продолжен с `--skip-short-stream`, без повторения успешного случая.

Вспомогательные ошибки также сохранены в доказательствах: неверный `/usr/bin/python3` в первой CPU-пробе исправлен на venv Python; после HTTP-пробы был краткий отказ bind порта, следующая read-only проверка доказала его освобождение до повтора; сбой передачи полного runner не применил изменений; последующая частичная установка проверена по точным SHA и завершена привязкой receipt. Эти события не являются отказами модели. Автоматического retry loop нет.

## Восстановление и диск

Свежая проверка **2026-09-18 08:51:22 UTC**, `final-verification-resumed.json`:

- `ai-server-qwen` остался stopped/exited, как до работы; остановлен с 2026-09-10 17:02:28 UTC. Полностью совпали ID, image, State, command, entrypoint, env names/проверенные runtime env, mounts, ports, restart, devices и Compose SHA `88c7de590dd7dea693dba225242d81e52387415a1e4e0f5796b568dfc4468c3c`.
- Gateway healthy, supervisor active; admission public, epoch 349, ledger/queue пусты, restart/reconciliation false. Оба штатных lock доступны.
- FreeToken-контейнер `4a741eeb0dd8f832f222f0b458756f4755b72f19ef34588f918138a714afc7b7` остановлен, exit 0, OOMKilled=false. GPU свободен, 1 MiB фонового использования. Runtime и тестовые контейнеры сохранены остановленными; сеть пустая.
- Root free 122,295,734,272 bytes (113.90 GiB); AI usage 117,506,465,792 bytes (109.44 GiB). От исходного baseline root free уменьшился примерно на 18.82 GiB, в пределах предварительного пика 35 GiB.
- После первоначальной подготовки новых образов/моделей/зависимостей не скачивали и не собирали. Во время исправлений сохранялись малые квитанции и container/network metadata; JIT-кэши находились в tmpfs.
- Пользователь отменил floor 150 GiB только для этого теста; сохранены reserve 32 GiB и AI cap 200 GiB. Глобальная storage policy не изменялась. Ничего не удалялось.

Драйвер, системный Python, постоянный gateway, маршрутизация и автозапуск не изменялись. Endpoint сейчас остановлен в соответствии с требованием восстановления.

## Повторный запуск

Через **Remote SSH, alias `server-a`**:

```sh
sudo -n python3 -E -B /srv/ai-server/artifacts/experiments/freetoken013-qwen38-nvfp4-20260918/run_candidate.py --image-id sha256:10bc9d29f79941c71bb27a77974be4b2312ae64f02636dd5445db20e77675134 --bundle-sha256 b0bbaf4fddce284d2987b8dcabcc47d295c2f27686a7e4ff1ca45d246102dc95
```

Эта точная команда запускает endpoint `http://127.0.0.1:19191/v1` с model ID `qwen38-nvfp4-freetoken`, выполняет конечную матрицу из пяти запросов и останавливает контейнер с восстановлением admission. Она не оставляет постоянный endpoint. Нужны свободный GPU, Qwen stopped, public/empty admission и свободные штатные locks. Произвольного порога 27 GiB больше нет. Не запускать контейнер напрямую в обход этих процедур.

## Доказательства

Итоговые `matrix-summary.json`, `final-verification-resumed.json`; точные raw receipts в `resume6-receipts/` (16 файлов, SHA сверены), короткий запрос и промежуточные события в `resume45-receipts/` (18 файлов), предыдущие исправления в `resume1-receipts/`, `resume2-receipts/`, `resume3-receipts/`. Старый отчёт с остановкой по RAM сохранён в `before-resume-report/`.

Один независимый Sol reviewer `/root/nvfp4_operational_review` проверял материальные изменения того же кандидата; финальный verdict APPROVE для bundle `b0bbaf4fddce284d2987b8dcabcc47d295c2f27686a7e4ff1ca45d246102dc95`. Узкие проверки управляющего кода прошли; здесь PASS подтверждён реальными streaming-ответами, server cache hits и восстановлением.
