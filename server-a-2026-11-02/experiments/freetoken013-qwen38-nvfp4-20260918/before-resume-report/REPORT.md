# FreeToken 0.1.3 / Qwen3.8-27B NVFP4 — итог 2026-09-18

**BLOCKED_HOST_RAM_RESERVE. Runtime установлен; модель не загружалась, endpoint не запускался, выполнено 0 из 5 запросов.** Единственный запуск остановился на защитной проверке RAM до создания контейнера кандидата. Исходное состояние сервисов восстановлено и проверено.

## Установлено и сохранено

- FreeToken `0.1.3`, commit `cac247a860e316e06580d05aeb05f2e647bde214`.
- Образ `local/ai-server-freetoken:0.1.3-cac247a-nvfp4-20260918`.
- Image ID: `sha256:10bc9d29f79941c71bb27a77974be4b2312ae64f02636dd5445db20e77675134`.
- Каталог Server A: `/srv/ai-server/artifacts/experiments/freetoken013-qwen38-nvfp4-20260918`; окружение внутри образа: `/opt/freetoken013`.
- Python 3.12.3, Torch 2.11.0 / CUDA 13.0, Triton 3.6.0, Transformers 5.16.1, Torchvision 0.26.0, flashlib 0.3.0, freetoken-kernel-cache 0.1.3+cu130. Все 95 закреплённых версий совпали с manifest; полный список — `receipts/installed-versions.json`.
- GCC 13, nvcc 13.0.88, Python.h и stdio.h доступны из существующей базы. Сборка прошла offline, без обновления базы; pip check и проверка импортов/версий завершились успешно.
- Internal Docker bridge `freetoken013-nvfp4-internal`, без подключённых контейнеров. Проверочный контейнер `ai-server-freetoken013-version-check` сохранён остановленным, exit 0, без GPU.
- Runtime-контейнер `ai-server-freetoken-q38-013-nvfp4` отсутствует. Системный Python, драйвер 595.84, постоянный gateway и автозапуск не изменялись.

Исходный образ и его слои закреплены в `receipts/installed-image.json`. Некоторые описательные OCI labels унаследованы от syv-базы и не являются результатами этого эксперимента.

## Модель и граница совместимости

Согласованная цель: `preetpatel/Qwen3.8-27B-Uncensored-NVFP4`, revision `37b5130a2d2a1f7d4456ab3f8d05d0b2a45ea350`.

Существующий файл весов: 19,709,883,808 bytes; SHA-256 `bc642c1159ab9b0fcd2fdc65be4fee72fa560cf0570604741db50061ea063277`. Все 12 файлов complete.json проверены; перед установкой повторно проверен полный SHA весов. Веса не скачивались и не копировались. Runtime подготовлен для read-only mounts model/, CAS весов и внешней цели symlink tokenizer.json.

В исходниках проверен путь для `Qwen3_5ForConditionalGeneration`, compressed-tensors NVFP4 / float4 / tensor_group / group_size=16, dynamic=local. Тензоры: U8 packed weights, F8_E4M3 scales, BF16 embedding/lm_head. Triton NVFP4 W4A16 использует декодирование E4M3 через uint8 для SM < 89; это соответствует предусмотренному пути для RTX 3090 SM86.

**Это исходниковая совместимость, не доказательство исполнения ядер, вместимости 32768 или успешного inference.** Успешный предыдущий inference этой NVFP4 в проверенных материалах не найден. Статические 18,788,076,416 bytes текстовых tensor payloads не являются оценкой полного пика VRAM.

Первоначальная AutoRound и эта NVFP4 имеют разные веса/формат. Сравнение с прежними TTFT AutoRound не было бы чистым сравнением движков; сейчас новых TTFT вообще нет.

## Почему остановлено

Проверка runner требует `MemAvailable >= 27 GiB`: лимит контейнера 24 GiB плюс 3 GiB для хоста.

- В 07:51:50 UTC непосредственно перед запуском: 29,001,302,016 bytes, выше порога 28,991,029,248 bytes всего на 10,272,768 bytes.
- В 07:52:07 UTC runner получил `RuntimeError: insufficient_available_host_ram`. Проверка выполняется до `docker create`.
- Точное значение RAM в момент отказа не записано; из кода и ошибки известно только, что оно было ниже 27 GiB.
- В финальной проверке 07:53:36 UTC: 28,940,169,216 bytes (~26.953 GiB), на 50,860,032 bytes ниже порога.

Механический `run-20260918T075207675291Z/result.json` содержит `status=FAIL` и `restoration_verified=true`. Итог эксперимента классифицирован **BLOCKED**: не выполнено условие безопасного старта. Это не OOM модели, не доказанная нехватка VRAM и не доказанная несовместимость NVFP4.

Повтора, снижения порога или остановки посторонних задач не было. Загрузка модели, GPU JIT и проверка KV capacity не начинались.

## Результаты запросов

План: context 32768, text-only, concurrency 1, prefix/radix cache, thinking off, temperature 0, max_tokens 256; только loopback `127.0.0.1:19191`.

| Случай | Результат | Фактическая длина / TTFT / cache hits / скорость / VRAM |
|---|---|---|
| Короткий streaming smoke | NOT RUN | Нет измерений |
| Новый контекст ~8k | NOT RUN | Нет измерений |
| Продолжение истории ~8k | NOT RUN | Нет измерений |
| Новый контекст ~28k | NOT RUN | Нет измерений |
| Продолжение истории ~28k | NOT RUN | Нет измерений |

Эффект повторного использования контекста не установлен. 1 MiB на GPU в финальном snapshot — фоновое состояние, а не потребление модели.

## Диск и ограничения

До скачивания рассчитан полный дополнительный пик **35 GiB**, включая wheelhouse/partial, build snapshots/cache, распакованные слои и резерв overhead. Существующие веса и базовые слои переиспользованы. 95 wheel-файлов: 2,974,808,160 bytes; полные SHA, размеры и ZIP metadata проверены.

Порог 150 GiB свободного root снят пользователем только для этого теста. Сохранены scoped reserve 32 GiB и AI budget 200 GiB; глобальная storage policy не изменена.

После сборки уменьшение свободного места root относительно baseline: 20,082,311,168 bytes. В финальном snapshot — 20,082,991,104 bytes (~18.70 GiB), ниже бюджета 35 GiB. Это измеренная разница свободного места файловой системы, не сумма логических размеров образов.

Финально: root free 122,419,924,992 bytes (~114.01 GiB); AI usage по storage guard 117,505,826,816 bytes (~109.44 GiB). Логический размер образа 23,244,253,967 bytes включает переиспользованные базовые слои и не равен приросту диска. Wheelhouse, build cache, образ, квитанции и небольшие транспортные файлы сохранены; удаления не выполнялись.

## Восстановление

Проверено 2026-09-18 07:53:36 UTC:

- `ai-server-qwen` остался stopped/exited, как до эксперимента; он остановлен с 2026-09-10 17:02:28 UTC. В этой работе его не запускали и не останавливали.
- Container ID `15fc9d27b9c035e8d1304c0fcfc5c6a3170b4dc2c00e32a3b178696465f2ea5f`, image, state, command, mounts, ports, restart и devices совпали с точным свежим prestate.
- Compose SHA-256 остался `88c7de590dd7dea693dba225242d81e52387415a1e4e0f5796b568dfc4468c3c`.
- Gateway healthy, supervisor active; admission возвращён в public, ledger и очередь пусты, restart/reconciliation false. Epoch штатно изменился 335 -> 336 -> 337 при закрытии и открытии admission.
- GPU compute processes отсутствуют; кандидат не создан; internal network пуст. Существующие locks освобождены при завершении runner.

## Повторный запуск

Команда для будущего явно разрешённого повторного теста через **Remote SSH, alias `server-a`**:

```sh
sudo -n python3 -E -B /srv/ai-server/artifacts/experiments/freetoken013-qwen38-nvfp4-20260918/run_candidate.py --image-id sha256:10bc9d29f79941c71bb27a77974be4b2312ae64f02636dd5445db20e77675134 --bundle-sha256 2e0089de02cce4e5801f8fdfedc9b7ec4bf4402c14cbbabb9110b623cd5a2c23
```

Это конечный тест из пяти случаев с остановкой кандидата и восстановлением admission в finally. Постоянный endpoint команда не оставляет. Нужны свободный GPU, исходный Qwen stopped, public/empty admission, свободные штатные locks и >=27 GiB MemAvailable. Runner заново проверяет закреплённые входы, образ, модель, сеть и ресурсы; при отклонении останавливается. Не использовать Python -O: проверки подготовки используют assert. Текущий запуск завершён; автоматического повтора нет.

## Review и доказательства

Независимый Sol critic `/root/nvfp4_operational_review` одобрил конкретный исправленный кандидат. После обнаружения унаследованного VOLUME /app/models тот же reviewer проверил узкую поправку read-only tmpfs и привязку receipt: финальный runtime bundle `2e0089de02cce4e5801f8fdfedc9b7ec4bf4402c14cbbabb9110b623cd5a2c23`. Исходный build bundle `884bff69cd920ff42d424d758e1b2f5d7445141c1b14bec903ed05633ff1c85a` сохранён отдельно; образ не пересобирался.

11 офлайн-проверок адаптеров прошли; после поправки VOLUME прошли 2 затронутые RuntimeGateTests. Это проверки управляющего кода, а не inference.

Основные квитанции: `receipts/installed-image.json`, `receipts/installed-versions.json`, `receipts/preswitch-preflight.json`, `receipts/run-20260918T075207675291Z/result.json`, `receipts/run-20260918T075207675291Z/events.jsonl`, `receipts/final-verification.json`. Все 20 файлов off-host export сверены с remote SHA-256 и размером побайтно. Старые статусы до установки сохранены в `before-final-report/`.

Предшествующий конфликт OpenCodex 409 исправлен отдельно: родитель и subagent разделяли session-id, и mutex ошибочно объединял их. Ключ теперь предпочитает thread-id, корректно освобождается при завершении/ошибках; GET/HEAD не захватывают turn. 18 целевых тестов и реальный запуск Sol reviewer прошли. Локальный отчёт и rollback: `D:\Programs\OpenCodex\patches\session-turn-20260918\README.md`. Это локальная правка OpenCodex, которую обновление приложения может перезаписать.
