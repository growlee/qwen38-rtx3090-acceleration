"""Deterministic long-context fixtures for the Qwen3.8 comparison.

This module only builds JSON-native fixtures.  ``padding_count`` counts
synthetic distractor records; actual chat-template token counts are measured by
the runtime adapter.  Candidate-produced Python remains the responsibility of
the existing ``quality_sandbox`` boundary.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping


SCHEMA_VERSION = 1
SUITE_ID = "qwen38-context-12-v1"
POSITIONS = ("start", "middle", "end")
FAMILY_IDS = tuple(
    f"context-{task_type}-{position}"
    for task_type in ("fact", "join", "version", "code")
    for position in POSITIONS
)


def _fc(args: list[Any], expected: Any, kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"args": args, "kwargs": kwargs or {}, "expected": expected}


def _fe(args: list[Any], kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"args": args, "kwargs": kwargs or {}, "expected_exception": "ValueError"}


_FACT = {
    "start": {
        "anchor": "[КАРТОЧКА A17] Активный узел Север-17: владелец Ирина Левина; порт 6417; код amber-92.",
        "question": "Для активного узла Север-17 верни ровно владелец|порт|код.",
        "expected": "Ирина Левина|6417|amber-92",
    },
    "middle": {
        "anchor": "[КАРТОЧКА M42] Активный узел Волга-42: владелец Тимур Сафиуллин; порт 7284; код cobalt-31.",
        "question": "Для активного узла Волга-42 верни ровно владелец|порт|код.",
        "expected": "Тимур Сафиуллин|7284|cobalt-31",
    },
    "end": {
        "anchor": "[КАРТОЧКА Z09] Активный узел Тайга-09: владелец Ольга Чернова; порт 5909; код jade-64.",
        "question": "Для активного узла Тайга-09 верни ровно владелец|порт|код.",
        "expected": "Ольга Чернова|5909|jade-64",
    },
}

_JOIN = {
    "start": {
        "anchor": "[ЗАКАЗ J11] Заказ R-204 связан с клиентом C-73 и проектом Вектор; приоритет P2.",
        "support": (
            "[КЛИЕНТ J11] Клиент C-73 обслуживается в городе Томск.",
            "[ПРОЕКТ J11] Проект Вектор сопровождает инженер Орлова.",
        ),
        "question": "Для заказа R-204 соедини записи и верни ровно город|инженер|приоритет.",
        "expected": "Томск|Орлова|P2",
    },
    "middle": {
        "anchor": "[ЗАКАЗ J52] Заказ R-518 связан с клиентом C-28 и проектом Каскад; приоритет P1.",
        "support": (
            "[КЛИЕНТ J52] Клиент C-28 обслуживается в городе Псков.",
            "[ПРОЕКТ J52] Проект Каскад сопровождает инженер Нестерова.",
        ),
        "question": "Для заказа R-518 соедини записи и верни ровно город|инженер|приоритет.",
        "expected": "Псков|Нестерова|P1",
    },
    "end": {
        "anchor": "[ЗАКАЗ J93] Заказ R-907 связан с клиентом C-41 и проектом Маяк; приоритет P3.",
        "support": (
            "[КЛИЕНТ J93] Клиент C-41 обслуживается в городе Курск.",
            "[ПРОЕКТ J93] Проект Маяк сопровождает инженер Белова.",
        ),
        "question": "Для заказа R-907 соедини записи и верни ровно город|инженер|приоритет.",
        "expected": "Курск|Белова|P3",
    },
}

_VERSION = {
    "start": {
        "anchor": "[ПОЛИТИКА V14] Nimbus версия 7 имеет статус CURRENT: порог 83, владелец Дарья.",
        "support": (
            "[ПОЛИТИКА V14] Nimbus версия 6 имеет статус OBSOLETE: порог 71, владелец Борис.",
            "[ПОЛИТИКА V14] Nimbus версия 8 имеет статус DRAFT: порог 91, владелец Елена.",
        ),
        "question": "Выбери только CURRENT-версию Nimbus, не максимальный номер. Верни ровно версия|порог|владелец.",
        "expected": "7|83|Дарья",
    },
    "middle": {
        "anchor": "[ПОЛИТИКА V55] Aurora версия 12 имеет статус CURRENT: порог 64, владелец Роман.",
        "support": (
            "[ПОЛИТИКА V55] Aurora версия 11 имеет статус OBSOLETE: порог 59, владелец Анна.",
            "[ПОЛИТИКА V55] Aurora версия 13 имеет статус DRAFT: порог 70, владелец Павел.",
        ),
        "question": "Выбери только CURRENT-версию Aurora, не максимальный номер. Верни ровно версия|порог|владелец.",
        "expected": "12|64|Роман",
    },
    "end": {
        "anchor": "[ПОЛИТИКА V96] Meridian версия 4 имеет статус CURRENT: порог 76, владелец Инга.",
        "support": (
            "[ПОЛИТИКА V96] Meridian версия 3 имеет статус OBSOLETE: порог 68, владелец Лев.",
            "[ПОЛИТИКА V96] Meridian версия 5 имеет статус DRAFT: порог 88, владелец Нина.",
        ),
        "question": "Выбери только CURRENT-версию Meridian, не максимальный номер. Верни ровно версия|порог|владелец.",
        "expected": "4|76|Инга",
    },
}

_CODE = {
    "start": {
        "function_name": "shipping_fee",
        "anchor": "[SPEC C18/1] Define shipping_fee(total, region, fragile). Base fee: north=300, central=200, south=400; any other region raises ValueError.",
        "support": (
            "[SPEC C18/2] total must be non-negative; otherwise raise ValueError. If total >= 5000, subtract 150 from the base fee, never below zero.",
            "[SPEC C18/3] After the discount, add 250 when fragile is true. Return the integer fee.",
        ),
        "question": "По всем распределённым записям SPEC C18 напиши требуемую функцию shipping_fee.",
        "source": "def shipping_fee(total, region, fragile):\n    if total < 0 or region not in {'north', 'central', 'south'}: raise ValueError('input')\n    fee = {'north': 300, 'central': 200, 'south': 400}[region]\n    if total >= 5000: fee = max(0, fee - 150)\n    if fragile: fee += 250\n    return fee",
        "cases": [_fc([0, "central", False], 200), _fc([4999, "north", True], 550), _fc([5000, "north", False], 150), _fc([9000, "central", True], 300), _fc([5000, "south", True], 500), _fe([-1, "north", False]), _fe([10, "west", False])],
    },
    "middle": {
        "function_name": "retention_days",
        "anchor": "[SPEC C57/1] Define retention_days(kind, encrypted, legal_hold). Base days: logs=30, backups=90, exports=7; any other kind raises ValueError.",
        "support": (
            "[SPEC C57/2] Add 15 days when encrypted is true, except that exports never receive this addition.",
            "[SPEC C57/3] A true legal_hold overrides every computed value and returns 365 days.",
        ),
        "question": "По всем распределённым записям SPEC C57 напиши требуемую функцию retention_days.",
        "source": "def retention_days(kind, encrypted, legal_hold):\n    if kind not in {'logs', 'backups', 'exports'}: raise ValueError('kind')\n    if legal_hold: return 365\n    days = {'logs': 30, 'backups': 90, 'exports': 7}[kind]\n    if encrypted and kind != 'exports': days += 15\n    return days",
        "cases": [_fc(["logs", False, False], 30), _fc(["logs", True, False], 45), _fc(["backups", True, False], 105), _fc(["exports", True, False], 7), _fc(["exports", False, True], 365), _fc(["logs", True, True], 365), _fe(["cache", False, False])],
    },
    "end": {
        "function_name": "retry_budget",
        "anchor": "[SPEC C94/1] Define retry_budget(priority, attempts, outage). Normal limits: low=2, normal=4, high=6; any other priority raises ValueError.",
        "support": (
            "[SPEC C94/2] attempts must be a non-negative integer (bool is not an integer here); otherwise raise ValueError.",
            "[SPEC C94/3] During outage add 2 to the selected limit. Return remaining attempts, with a minimum of zero.",
        ),
        "question": "По всем распределённым записям SPEC C94 напиши требуемую функцию retry_budget.",
        "source": "def retry_budget(priority, attempts, outage):\n    if priority not in {'low', 'normal', 'high'} or type(attempts) is not int or attempts < 0: raise ValueError('input')\n    limit = {'low': 2, 'normal': 4, 'high': 6}[priority] + (2 if outage else 0)\n    return max(0, limit - attempts)",
        "cases": [_fc(["low", 0, False], 2), _fc(["normal", 4, False], 0), _fc(["high", 5, False], 1), _fc(["low", 2, True], 2), _fc(["high", 20, True], 0), _fe(["urgent", 0, False]), _fe(["normal", -1, False]), _fe(["normal", True, False])],
    },
}

_SPECS: dict[str, dict[str, Any]] = {}
for _task_type, _group in (("fact", _FACT), ("join", _JOIN), ("version", _VERSION), ("code", _CODE)):
    for _position, _spec in _group.items():
        _SPECS[f"context-{_task_type}-{_position}"] = {"task_type": _task_type, "position": _position, **_spec}


def _distractors(count: int) -> list[str]:
    return [
        f"[ШУМ D{index:06d}] Архивная запись не относится к запрошенным идентификаторам; сектор={(index * 17) % 997:03d}; checksum={(index * 7919 + 104729) % 1000003:06d}."
        for index in range(count)
    ]


def _arrange(anchor: str, support: tuple[str, ...], padding: list[str], position: str) -> list[str]:
    if not support:
        split = len(padding) // 2
        if position == "start":
            return [anchor, *padding]
        if position == "middle":
            return [*padding[:split], anchor, *padding[split:]]
        return [*padding, anchor]
    if len(support) != 2:
        raise ValueError("fixture_support_count_invalid")
    first = len(padding) // 3
    second = (2 * len(padding)) // 3
    a, b, c = padding[:first], padding[first:second], padding[second:]
    if position == "start":
        return [anchor, *a, support[0], *b, support[1], *c]
    if position == "middle":
        return [support[0], *a, *b[: len(b) // 2], anchor, *b[len(b) // 2 :], *c, support[1]]
    return [support[0], *a, support[1], *b, *c, anchor]


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_case(family_id: str, padding_count: int) -> dict[str, Any]:
    """Build one family with exactly ``padding_count`` irrelevant records."""
    if family_id not in _SPECS:
        raise ValueError("unknown_family_id")
    if type(padding_count) is not int or padding_count < 0:
        raise ValueError("padding_count_must_be_non_negative_integer")
    spec = _SPECS[family_id]
    records = _arrange(spec["anchor"], tuple(spec.get("support", ())), _distractors(padding_count), spec["position"])
    system = (
        "Используй только предоставленные записи. Не считай архивный шум относящимся к вопросу. "
        "Верни только запрошенный ответ без объяснений."
    )
    max_tokens = 256
    if spec["task_type"] == "code":
        system = (
            "Use only the supplied specification records. Return exactly one Python function definition with the requested name. "
            "Do not use markdown, imports, input/output, network, files, or global state."
        )
        max_tokens = 2048
    request = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "КОНТЕКСТ:\n" + "\n".join(records) + "\n\nВОПРОС:\n" + spec["question"]},
        ],
        "generation": {"temperature": 0, "seed": 9100 + FAMILY_IDS.index(family_id), "max_tokens": max_tokens},
    }
    case: dict[str, Any] = {
        "case_id": family_id,
        "category": "coding" if spec["task_type"] == "code" else "retrieval",
        "critical": True,
        "fixture": {"family_id": family_id, "task_type": spec["task_type"], "position": spec["position"], "padding_count": padding_count},
        "request": request,
    }
    if spec["task_type"] == "code":
        case["check"] = {"type": "python_function_cases", "function_name": spec["function_name"], "cases": copy.deepcopy(spec["cases"])}
        case["reference_source"] = spec["source"]
    else:
        case["check"] = {"type": "exact", "expected": spec["expected"], "casefold": False}
    return case


def build_suite(padding_count: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "padding_count": padding_count,
        "cases": [build_case(family_id, padding_count) for family_id in FAMILY_IDS],
    }

