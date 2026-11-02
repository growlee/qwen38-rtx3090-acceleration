"""Hand-authored miniature repositories for the bounded coding-agent matrix."""

from textwrap import dedent


def _text(value: str) -> str:
    return dedent(value).lstrip("\n")


CASES = [
    {
        "id": "cursor-pagination-validation",
        "input_target": 8000,
        "instruction": (
            "Read README.md, then implement its contract in app.py. Keep the public "
            "function name and use only the Python standard library. Run the provided "
            "tests when you are done."
        ),
        "files": {
            "README.md": _text(
                r'''
                # Customer cursor pagination

                Implement `list_customers(records, cursor=None, limit=50)` in `app.py`.
                The service receives an in-memory snapshot and must expose deterministic
                cursor pagination without mutating the caller's records.

                ## Record validation

                `records` must be a list of dictionaries. Every record must contain exactly
                `id`, `name`, and `active`:

                - `id` is a positive integer. Booleans are not integers here. IDs must be unique.
                - `name` is a non-empty string after trimming. Preserve the original name in output.
                - `active` is a boolean.

                Reject invalid input with `ValueError`. Input order is arbitrary; results are
                ordered by ascending `id`.

                ## Cursor and limit

                `limit` must be an integer from 1 through 100, excluding booleans.
                A cursor is either `None` or a string previously returned by this function.
                Returned cursors are the unpadded URL-safe Base64 encoding of compact UTF-8
                JSON containing exactly `{"last_id": positive_integer}`. Reject malformed
                cursors with `ValueError`, including decoded payloads with extra fields or a
                non-positive ID.
                A valid cursor whose ID is absent from the current snapshot resumes after that
                numeric ID; this allows records to be deleted between requests.

                Return `{"items": [...], "next_cursor": string_or_none}`. Items are shallow
                copies. Return a next cursor only when at least one matching record remains
                after the current page. Empty pages never have a next cursor.
                '''
            ),
            "app.py": _text(
                r'''
                """Cursor-based access to an in-memory customer snapshot."""


                def list_customers(records, cursor=None, limit=50):
                    """Return one page of customers.

                    This provisional implementation predates the opaque cursor contract.
                    """
                    start = int(cursor or 0)
                    page = records[start : start + limit]
                    next_cursor = str(start + limit) if page else None
                    return {"items": page, "next_cursor": next_cursor}
                '''
            ),
        },
        "editable": ["app.py"],
        "tests": _text(
            r'''
            import base64
            import json
            import unittest

            import app


            class CustomerPaginationTests(unittest.TestCase):
                def setUp(self):
                    self.records = [
                        {"id": 9, "name": "Nine", "active": False},
                        {"id": 2, "name": "Two", "active": True},
                        {"id": 5, "name": "Five", "active": True},
                    ]

                def test_orders_pages_and_returns_opaque_cursor(self):
                    first = app.list_customers(self.records, limit=2)
                    self.assertEqual([row["id"] for row in first["items"]], [2, 5])
                    self.assertIsInstance(first["next_cursor"], str)
                    self.assertNotEqual(first["next_cursor"], "2")
                    second = app.list_customers(self.records, first["next_cursor"], 2)
                    self.assertEqual([row["id"] for row in second["items"]], [9])
                    self.assertIsNone(second["next_cursor"])

                def test_cursor_survives_deletion_of_boundary_record(self):
                    first = app.list_customers(self.records, limit=2)
                    changed = [row for row in self.records if row["id"] != 5]
                    page = app.list_customers(changed, first["next_cursor"], 10)
                    self.assertEqual([row["id"] for row in page["items"]], [9])

                def test_does_not_mutate_or_alias_records(self):
                    original_order = list(self.records)
                    result = app.list_customers(self.records, limit=1)
                    self.assertEqual(self.records, original_order)
                    self.assertIsNot(result["items"][0], self.records[1])

                def test_empty_page_has_no_cursor(self):
                    full_page = app.list_customers(self.records, limit=3)
                    self.assertEqual([row["id"] for row in full_page["items"]], [2, 5, 9])
                    self.assertIsNone(full_page["next_cursor"])
                    self.assertEqual(
                        app.list_customers([], limit=3),
                        {"items": [], "next_cursor": None},
                    )

                def test_rejects_invalid_records(self):
                    bad_sets = [
                        "not-a-list",
                        [{"id": 1, "name": "A", "active": True}, {"id": 1, "name": "B", "active": False}],
                        [{"id": True, "name": "A", "active": True}],
                        [{"id": 1, "name": "   ", "active": True}],
                        [{"id": 1, "name": "A", "active": 1}],
                        [{"id": 1, "name": "A", "active": True, "tier": "gold"}],
                    ]
                    for records in bad_sets:
                        with self.subTest(records=records), self.assertRaises(ValueError):
                            app.list_customers(records)

                def test_rejects_invalid_limits(self):
                    for limit in (True, 0, 101, 2.5):
                        with self.subTest(limit=limit), self.assertRaises(ValueError):
                            app.list_customers(self.records, limit=limit)

                def test_rejects_malformed_or_noncanonical_cursors(self):
                    def encoded(payload):
                        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

                    for cursor in ("%%%", encoded({"last_id": 0}), encoded({"last_id": 2, "extra": 1}), encoded({"last_id": True})):
                        with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                            app.list_customers(self.records, cursor=cursor)


            if __name__ == "__main__":
                unittest.main()
            '''
        ),
        "reference": _text(
            r'''
            """Cursor-based access to an in-memory customer snapshot."""

            import base64
            import binascii
            import json


            _FIELDS = {"id", "name", "active"}


            def _validate_records(records):
                if not isinstance(records, list):
                    raise ValueError("records must be a list")
                seen = set()
                validated = []
                for record in records:
                    if not isinstance(record, dict) or set(record) != _FIELDS:
                        raise ValueError("each record must contain exactly id, name, and active")
                    customer_id = record["id"]
                    if isinstance(customer_id, bool) or not isinstance(customer_id, int) or customer_id <= 0:
                        raise ValueError("id must be a positive integer")
                    if customer_id in seen:
                        raise ValueError("ids must be unique")
                    if not isinstance(record["name"], str) or not record["name"].strip():
                        raise ValueError("name must be a non-empty string")
                    if not isinstance(record["active"], bool):
                        raise ValueError("active must be a boolean")
                    seen.add(customer_id)
                    validated.append(record)
                return sorted(validated, key=lambda row: row["id"])


            def _encode_cursor(last_id):
                raw = json.dumps({"last_id": last_id}, separators=(",", ":")).encode("utf-8")
                return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


            def _decode_cursor(cursor):
                if not isinstance(cursor, str) or not cursor:
                    raise ValueError("cursor must be a non-empty string")
                try:
                    padding = "=" * (-len(cursor) % 4)
                    raw = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
                    payload = json.loads(raw.decode("utf-8"))
                except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError("malformed cursor") from exc
                if not isinstance(payload, dict) or set(payload) != {"last_id"}:
                    raise ValueError("malformed cursor payload")
                last_id = payload["last_id"]
                if isinstance(last_id, bool) or not isinstance(last_id, int) or last_id <= 0:
                    raise ValueError("malformed cursor id")
                return last_id


            def list_customers(records, cursor=None, limit=50):
                if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
                    raise ValueError("limit must be an integer from 1 through 100")
                ordered = _validate_records(records)
                last_id = 0 if cursor is None else _decode_cursor(cursor)
                remaining = [record for record in ordered if record["id"] > last_id]
                selected = remaining[:limit]
                items = [dict(record) for record in selected]
                next_cursor = None
                if selected and len(remaining) > len(selected):
                    next_cursor = _encode_cursor(selected[-1]["id"])
                return {"items": items, "next_cursor": next_cursor}
            '''
        ),
    },
    {
        "id": "decimal-invoice-calculation",
        "input_target": 16000,
        "instruction": (
            "Read README.md and repair app.py so calculate_invoice follows the stated "
            "money and validation rules. Use only the Python standard library and do not "
            "change the public API. Run the provided tests when finished."
        ),
        "files": {
            "README.md": _text(
                r'''
                # Invoice totals

                Implement `calculate_invoice(lines, tax_rate)` in `app.py`. Inputs come from
                JSON, so all decimal values are strings. Never use binary floating point.

                `lines` is a non-empty list of dictionaries with exactly these fields:

                - `sku`: non-empty string after trimming; SKUs must be unique.
                - `quantity`: positive integer, excluding booleans.
                - `unit_price`: non-negative plain decimal string with at most two fractional digits.
                - optional `discount_rate`: plain decimal string between `0` and `1` inclusive,
                  with at most four fractional digits. Missing means zero.

                `tax_rate` follows the same format and range as `discount_rate`. Scientific
                notation, signs, whitespace, `NaN`, and infinities are invalid. Raise
                `ValueError` for every domain or shape error.

                Calculate each line independently:

                1. `gross = quantity * unit_price` (already exact cents).
                2. `discount = gross * discount_rate`, rounded to cents with `ROUND_HALF_UP`.
                3. `net = gross - discount`.

                Invoice `subtotal`, `discount`, and `net` are sums of those line values. Apply
                tax once to invoice net and round it to cents with `ROUND_HALF_UP`. Total is net
                plus tax. Return money as fixed two-decimal strings and rates as the original
                validated strings:

                ```python
                {
                    "lines": [
                        {"sku": ..., "quantity": ..., "unit_price": ..., "discount_rate": ...,
                         "gross": ..., "discount": ..., "net": ...},
                    ],
                    "subtotal": ..., "discount": ..., "net": ...,
                    "tax_rate": ..., "tax": ..., "total": ...,
                }
                ```

                Preserve input order and do not mutate input dictionaries.
                '''
            ),
            "app.py": _text(
                r'''
                """Invoice arithmetic."""


                def calculate_invoice(lines, tax_rate):
                    subtotal = sum(row["quantity"] * float(row["unit_price"]) for row in lines)
                    tax = subtotal * float(tax_rate)
                    return {
                        "lines": lines,
                        "subtotal": f"{subtotal:.2f}",
                        "discount": "0.00",
                        "net": f"{subtotal:.2f}",
                        "tax_rate": tax_rate,
                        "tax": f"{tax:.2f}",
                        "total": f"{subtotal + tax:.2f}",
                    }
                '''
            ),
        },
        "editable": ["app.py"],
        "tests": _text(
            r'''
            import copy
            import unittest

            import app


            class InvoiceTests(unittest.TestCase):
                def test_calculates_line_discounts_then_invoice_tax(self):
                    lines = [
                        {"sku": "A-1", "quantity": 3, "unit_price": "19.99", "discount_rate": "0.1250"},
                        {"sku": "B-2", "quantity": 2, "unit_price": "5.00"},
                    ]
                    result = app.calculate_invoice(lines, "0.0825")
                    self.assertEqual(result["lines"], [
                        {"sku": "A-1", "quantity": 3, "unit_price": "19.99", "discount_rate": "0.1250", "gross": "59.97", "discount": "7.50", "net": "52.47"},
                        {"sku": "B-2", "quantity": 2, "unit_price": "5.00", "discount_rate": "0", "gross": "10.00", "discount": "0.00", "net": "10.00"},
                    ])
                    self.assertEqual(result["subtotal"], "69.97")
                    self.assertEqual(result["discount"], "7.50")
                    self.assertEqual(result["net"], "62.47")
                    self.assertEqual(result["tax"], "5.15")
                    self.assertEqual(result["total"], "67.62")
                    self.assertEqual(result["tax_rate"], "0.0825")

                def test_rounds_half_up_not_bankers(self):
                    result = app.calculate_invoice(
                        [{"sku": "HALF", "quantity": 1, "unit_price": "0.05", "discount_rate": "0.10"}],
                        "0.10",
                    )
                    self.assertEqual(result["lines"][0]["discount"], "0.01")
                    self.assertEqual(result["lines"][0]["net"], "0.04")
                    self.assertEqual(result["tax"], "0.00")
                    self.assertEqual(result["total"], "0.04")

                def test_tax_is_rounded_once_after_summing_net(self):
                    lines = [
                        {"sku": "X", "quantity": 1, "unit_price": "0.05"},
                        {"sku": "Y", "quantity": 1, "unit_price": "0.05"},
                    ]
                    result = app.calculate_invoice(lines, "0.10")
                    self.assertEqual(result["tax"], "0.01")
                    self.assertEqual(result["total"], "0.11")

                def test_preserves_input_order_without_mutation(self):
                    lines = [
                        {"sku": "Z", "quantity": 1, "unit_price": "2.00"},
                        {"sku": "A", "quantity": 1, "unit_price": "1.00"},
                    ]
                    before = copy.deepcopy(lines)
                    result = app.calculate_invoice(lines, "0")
                    self.assertEqual(lines, before)
                    self.assertEqual([row["sku"] for row in result["lines"]], ["Z", "A"])
                    self.assertIsNot(result["lines"][0], lines[0])

                def test_rejects_bad_shapes_and_duplicate_skus(self):
                    bad = [
                        [],
                        "lines",
                        [{"sku": "A", "quantity": 1, "unit_price": "1.00", "extra": 1}],
                        [{"sku": " ", "quantity": 1, "unit_price": "1.00"}],
                        [{"sku": "A", "quantity": True, "unit_price": "1.00"}],
                        [{"sku": "A", "quantity": 1, "unit_price": "1.00"}, {"sku": "A", "quantity": 2, "unit_price": "2.00"}],
                    ]
                    for lines in bad:
                        with self.subTest(lines=lines), self.assertRaises(ValueError):
                            app.calculate_invoice(lines, "0.2")

                def test_rejects_noncanonical_decimals_and_out_of_range_rates(self):
                    base = {"sku": "A", "quantity": 1, "unit_price": "1.00"}
                    for price in (1.0, "1.000", "+1.00", " 1.00", "1e0", "NaN", "-0.01"):
                        with self.subTest(price=price), self.assertRaises(ValueError):
                            app.calculate_invoice([{**base, "unit_price": price}], "0")
                    for rate in (0.1, "0.12345", "+0.1", "1.0001", "-0.1", "NaN"):
                        with self.subTest(rate=rate), self.assertRaises(ValueError):
                            app.calculate_invoice([base], rate)


            if __name__ == "__main__":
                unittest.main()
            '''
        ),
        "reference": _text(
            r'''
            """Exact invoice arithmetic with explicit domain validation."""

            from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
            import re


            _MONEY = Decimal("0.01")
            _PRICE_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,2})?")
            _RATE_PATTERN = re.compile(r"(?:0|1|0\.[0-9]{1,4}|1\.0{1,4})")
            _REQUIRED_FIELDS = {"sku", "quantity", "unit_price"}


            def _decimal(value, pattern, label):
                if not isinstance(value, str) or pattern.fullmatch(value) is None:
                    raise ValueError(f"{label} must be a plain decimal string")
                try:
                    return Decimal(value)
                except InvalidOperation as exc:
                    raise ValueError(f"invalid {label}") from exc


            def _rate(value, label):
                result = _decimal(value, _RATE_PATTERN, label)
                if not Decimal("0") <= result <= Decimal("1"):
                    raise ValueError(f"{label} must be between zero and one")
                return result


            def _money(value):
                return format(value.quantize(_MONEY, rounding=ROUND_HALF_UP), ".2f")


            def calculate_invoice(lines, tax_rate):
                if not isinstance(lines, list) or not lines:
                    raise ValueError("lines must be a non-empty list")
                tax_decimal = _rate(tax_rate, "tax_rate")
                seen_skus = set()
                output_lines = []
                subtotal = Decimal("0")
                discount_total = Decimal("0")
                net_total = Decimal("0")

                for line in lines:
                    if not isinstance(line, dict):
                        raise ValueError("each line must be a dictionary")
                    fields = set(line)
                    if not _REQUIRED_FIELDS <= fields or fields - (_REQUIRED_FIELDS | {"discount_rate"}):
                        raise ValueError("invalid line fields")
                    sku = line["sku"]
                    if not isinstance(sku, str) or not sku.strip():
                        raise ValueError("sku must be a non-empty string")
                    if sku in seen_skus:
                        raise ValueError("skus must be unique")
                    quantity = line["quantity"]
                    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
                        raise ValueError("quantity must be a positive integer")
                    unit_price_text = line["unit_price"]
                    unit_price = _decimal(unit_price_text, _PRICE_PATTERN, "unit_price")
                    discount_text = line.get("discount_rate", "0")
                    discount_rate = _rate(discount_text, "discount_rate")

                    gross = unit_price * quantity
                    discount = (gross * discount_rate).quantize(_MONEY, rounding=ROUND_HALF_UP)
                    net = gross - discount
                    subtotal += gross
                    discount_total += discount
                    net_total += net
                    seen_skus.add(sku)
                    output_lines.append({
                        "sku": sku,
                        "quantity": quantity,
                        "unit_price": unit_price_text,
                        "discount_rate": discount_text,
                        "gross": _money(gross),
                        "discount": _money(discount),
                        "net": _money(net),
                    })

                tax = (net_total * tax_decimal).quantize(_MONEY, rounding=ROUND_HALF_UP)
                return {
                    "lines": output_lines,
                    "subtotal": _money(subtotal),
                    "discount": _money(discount_total),
                    "net": _money(net_total),
                    "tax_rate": tax_rate,
                    "tax": _money(tax),
                    "total": _money(net_total + tax),
                }
            '''
        ),
    },
    {
        "id": "jsonl-event-log-aggregation",
        "input_target": 28000,
        "instruction": (
            "Read README.md and implement aggregate_events in app.py. The existing code "
            "is incomplete; preserve the public API, use only the Python standard library, "
            "and run the supplied tests after editing."
        ),
        "files": {
            "README.md": _text(
                r'''
                # JSONL event-log aggregation

                Implement `aggregate_events(text)` in `app.py`. The input is one string in
                JSON Lines format. Ignore empty or whitespace-only lines. Every other line
                must decode to a JSON object containing exactly the common fields
                `event_id`, `user_id`, `timestamp`, and `type`, plus the conditional field
                described below.

                Common fields:

                - `event_id` and `user_id` are non-empty strings after trimming. Preserve them.
                - `timestamp` is an RFC 3339 string with an explicit timezone. `Z` is accepted.
                  Normalize it to UTC with exactly six fractional digits and a trailing `Z`.
                - `type` is one of `login`, `view`, `purchase`, or `refund`.

                `purchase` and `refund` require an `amount` field. It is a non-negative plain
                decimal string with at most two fractional digits. Other event types must not
                have `amount`. Normalize accepted amounts to fixed two-decimal strings.

                Raise `ValueError` for invalid JSON, unpaired UTF-16 surrogate code points in
                parsed strings (including IDs), blank IDs, invalid timestamps, invalid fields,
                or invalid amounts. The exception message for a bad physical line must start
                with `line N:` where N is its one-based input line number.

                ## Deduplication and ordering

                Deduplicate by `event_id` after normalization. An identical repeat increments
                `duplicate_count` and is otherwise ignored. If the same ID has different
                normalized content, raise `ValueError` at the later line. Sort unique events by
                normalized timestamp, then by `event_id`, independent of input order.

                ## Result

                Return:

                ```python
                {
                    "total_unique": integer,
                    "duplicate_count": integer,
                    "ordered_events": [normalized event dictionaries],
                    "users": [
                        {
                            "user_id": string,
                            "first_seen": normalized timestamp,
                            "last_seen": normalized timestamp,
                            "counts": {"login": int, "view": int, "purchase": int, "refund": int},
                            "net_revenue": fixed_two_decimal_string,
                        }
                    ],
                }
                ```

                Normalized event dictionaries use common fields in the documented order and
                add `amount` last for financial events. User aggregates are based on unique
                events, ordered by `user_id`, and include zero counts. Purchases add revenue;
                refunds subtract it. Negative net revenue is valid. For no events, return zero
                totals and empty lists.
                '''
            ),
            "app.py": _text(
                r'''
                """Aggregate a JSONL activity log."""

                import json


                def aggregate_events(text):
                    events = [json.loads(line) for line in text.splitlines() if line.strip()]
                    users = {}
                    for event in events:
                        user = users.setdefault(event["user_id"], {"user_id": event["user_id"], "counts": {}})
                        kind = event["type"]
                        user["counts"][kind] = user["counts"].get(kind, 0) + 1
                    return {
                        "total_unique": len(events),
                        "duplicate_count": 0,
                        "ordered_events": events,
                        "users": list(users.values()),
                    }
                '''
            ),
        },
        "editable": ["app.py"],
        "tests": _text(
            r'''
            import json
            import unittest

            import app


            def line(**event):
                return json.dumps(event, separators=(",", ":"))


            class EventAggregationTests(unittest.TestCase):
                def test_parses_normalizes_deduplicates_orders_and_aggregates(self):
                    purchase = line(event_id="e3", user_id="bob", timestamp="2026-09-19T10:00:00+02:00", type="purchase", amount="12.5")
                    text = "\n".join([
                        line(event_id="e2", user_id="alice", timestamp="2026-09-19T08:00:01Z", type="view"),
                        purchase,
                        "   ",
                        line(event_id="e1", user_id="alice", timestamp="2026-09-19T07:59:59.123Z", type="login"),
                        purchase,
                        line(event_id="e4", user_id="bob", timestamp="2026-09-19T08:00:02+00:00", type="refund", amount="2.25"),
                    ])
                    result = app.aggregate_events(text)
                    self.assertEqual(result["total_unique"], 4)
                    self.assertEqual(result["duplicate_count"], 1)
                    self.assertEqual([event["event_id"] for event in result["ordered_events"]], ["e1", "e3", "e2", "e4"])
                    self.assertEqual(result["ordered_events"][0]["timestamp"], "2026-09-19T07:59:59.123000Z")
                    self.assertEqual(result["ordered_events"][1]["timestamp"], "2026-09-19T08:00:00.000000Z")
                    self.assertEqual(result["ordered_events"][1]["amount"], "12.50")
                    self.assertEqual(result["users"], [
                        {
                            "user_id": "alice",
                            "first_seen": "2026-09-19T07:59:59.123000Z",
                            "last_seen": "2026-09-19T08:00:01.000000Z",
                            "counts": {"login": 1, "view": 1, "purchase": 0, "refund": 0},
                            "net_revenue": "0.00",
                        },
                        {
                            "user_id": "bob",
                            "first_seen": "2026-09-19T08:00:00.000000Z",
                            "last_seen": "2026-09-19T08:00:02.000000Z",
                            "counts": {"login": 0, "view": 0, "purchase": 1, "refund": 1},
                            "net_revenue": "10.25",
                        },
                    ])

                def test_orders_equal_timestamps_by_event_id(self):
                    text = "\n".join([
                        line(event_id="z", user_id="u", timestamp="2026-01-01T00:00:00Z", type="view"),
                        line(event_id="a", user_id="u", timestamp="2026-01-01T00:00:00+00:00", type="login"),
                    ])
                    result = app.aggregate_events(text)
                    self.assertEqual([row["event_id"] for row in result["ordered_events"]], ["a", "z"])

                def test_normalized_equivalent_duplicates_are_identical(self):
                    first = line(event_id="p", user_id="u", timestamp="2026-01-01T01:00:00+01:00", type="purchase", amount="1")
                    second = line(event_id="p", user_id="u", timestamp="2026-01-01T00:00:00Z", type="purchase", amount="1.00")
                    result = app.aggregate_events(first + "\n" + second)
                    self.assertEqual(result["total_unique"], 1)
                    self.assertEqual(result["duplicate_count"], 1)

                def test_conflicting_duplicate_reports_later_line(self):
                    text = "\n".join([
                        line(event_id="same", user_id="u", timestamp="2026-01-01T00:00:00Z", type="view"),
                        line(event_id="same", user_id="u", timestamp="2026-01-01T00:00:00Z", type="login"),
                    ])
                    with self.assertRaisesRegex(ValueError, r"^line 2:"):
                        app.aggregate_events(text)

                def test_rejects_bad_json_and_shape_with_physical_line_number(self):
                    cases = [
                        ("\n{bad", 2),
                        ("[]", 1),
                        (line(event_id="e", user_id="u", timestamp="2026-01-01T00:00:00Z", type="view", extra=True), 1),
                        (line(event_id=" ", user_id="u", timestamp="2026-01-01T00:00:00Z", type="view"), 1),
                    ]
                    for text, number in cases:
                        with self.subTest(text=text), self.assertRaisesRegex(ValueError, rf"^line {number}:"):
                            app.aggregate_events(text)

                def test_rejects_naive_or_invalid_timestamps(self):
                    for timestamp in ("2026-01-01T00:00:00", "yesterday", "2026-02-30T00:00:00Z"):
                        text = line(event_id="e", user_id="u", timestamp=timestamp, type="view")
                        with self.subTest(timestamp=timestamp), self.assertRaisesRegex(ValueError, r"^line 1:"):
                            app.aggregate_events(text)

                def test_enforces_conditional_amount_and_decimal_format(self):
                    bad_events = [
                        dict(event_id="e", user_id="u", timestamp="2026-01-01T00:00:00Z", type="purchase"),
                        dict(event_id="e", user_id="u", timestamp="2026-01-01T00:00:00Z", type="view", amount="1.00"),
                        dict(event_id="e", user_id="u", timestamp="2026-01-01T00:00:00Z", type="refund", amount=1),
                        dict(event_id="e", user_id="u", timestamp="2026-01-01T00:00:00Z", type="purchase", amount="1.000"),
                        dict(event_id="e", user_id="u", timestamp="2026-01-01T00:00:00Z", type="purchase", amount="-1.00"),
                        dict(event_id="e", user_id="u", timestamp="2026-01-01T00:00:00Z", type="purchase", amount="1e2"),
                    ]
                    for event in bad_events:
                        with self.subTest(event=event), self.assertRaisesRegex(ValueError, r"^line 1:"):
                            app.aggregate_events(line(**event))

                def test_empty_log(self):
                    self.assertEqual(app.aggregate_events("\n  \n"), {
                        "total_unique": 0,
                        "duplicate_count": 0,
                        "ordered_events": [],
                        "users": [],
                    })


            if __name__ == "__main__":
                unittest.main()
            '''
        ),
        "reference": _text(
            r'''
            """Parse, normalize, deduplicate, and aggregate a JSONL activity log."""

            from datetime import datetime, timezone
            from decimal import Decimal, InvalidOperation
            import json
            import re


            EVENT_TYPES = ("login", "view", "purchase", "refund")
            FINANCIAL_TYPES = {"purchase", "refund"}
            COMMON_FIELDS = {"event_id", "user_id", "timestamp", "type"}
            AMOUNT_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,2})?")
            TIMESTAMP_PATTERN = re.compile(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
                r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})"
            )


            def _fail(line_number, message, cause=None):
                error = ValueError(f"line {line_number}: {message}")
                if cause is None:
                    raise error
                raise error from cause


            def _identifier(value, label, line_number):
                if not isinstance(value, str) or not value.strip():
                    _fail(line_number, f"{label} must be a non-empty string")
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError as exc:
                    _fail(line_number, f"{label} contains invalid Unicode", exc)
                return value


            def _timestamp(value, line_number):
                if not isinstance(value, str) or TIMESTAMP_PATTERN.fullmatch(value) is None:
                    _fail(line_number, "timestamp must be an RFC 3339 string")
                candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
                try:
                    parsed = datetime.fromisoformat(candidate)
                except ValueError as exc:
                    _fail(line_number, "invalid timestamp", exc)
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    _fail(line_number, "timestamp must include a timezone")
                normalized = parsed.astimezone(timezone.utc)
                return normalized, normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


            def _amount(value, line_number):
                if not isinstance(value, str) or AMOUNT_PATTERN.fullmatch(value) is None:
                    _fail(line_number, "amount must be a non-negative plain decimal string")
                try:
                    amount = Decimal(value)
                except InvalidOperation as exc:
                    _fail(line_number, "invalid amount", exc)
                return amount, format(amount, ".2f")


            def _normalize(raw, line_number):
                if not isinstance(raw, dict):
                    _fail(line_number, "event must be a JSON object")
                kind = raw.get("type")
                if kind not in EVENT_TYPES:
                    _fail(line_number, "unknown event type")
                expected = COMMON_FIELDS | ({"amount"} if kind in FINANCIAL_TYPES else set())
                if set(raw) != expected:
                    _fail(line_number, "event has invalid fields")
                event_id = _identifier(raw["event_id"], "event_id", line_number)
                user_id = _identifier(raw["user_id"], "user_id", line_number)
                moment, timestamp = _timestamp(raw["timestamp"], line_number)
                event = {
                    "event_id": event_id,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "type": kind,
                }
                amount = None
                if kind in FINANCIAL_TYPES:
                    amount, amount_text = _amount(raw["amount"], line_number)
                    event["amount"] = amount_text
                return event, moment, amount


            def aggregate_events(text):
                if not isinstance(text, str):
                    raise ValueError("text must be a string")
                unique = {}
                duplicate_count = 0

                for line_number, source_line in enumerate(text.splitlines(), start=1):
                    if not source_line.strip():
                        continue
                    try:
                        raw = json.loads(source_line)
                    except (json.JSONDecodeError, UnicodeError) as exc:
                        _fail(line_number, "invalid JSON", exc)
                    event, moment, amount = _normalize(raw, line_number)
                    event_id = event["event_id"]
                    previous = unique.get(event_id)
                    if previous is not None:
                        if previous[0] != event:
                            _fail(line_number, "conflicting duplicate event_id")
                        duplicate_count += 1
                        continue
                    unique[event_id] = (event, moment, amount)

                ordered = sorted(unique.values(), key=lambda item: (item[1], item[0]["event_id"]))
                users = {}
                for event, moment, amount in ordered:
                    user_id = event["user_id"]
                    state = users.get(user_id)
                    if state is None:
                        state = {
                            "first": moment,
                            "last": moment,
                            "counts": {kind: 0 for kind in EVENT_TYPES},
                            "revenue": Decimal("0"),
                        }
                        users[user_id] = state
                    state["first"] = min(state["first"], moment)
                    state["last"] = max(state["last"], moment)
                    state["counts"][event["type"]] += 1
                    if event["type"] == "purchase":
                        state["revenue"] += amount
                    elif event["type"] == "refund":
                        state["revenue"] -= amount

                user_rows = []
                for user_id in sorted(users):
                    state = users[user_id]
                    user_rows.append({
                        "user_id": user_id,
                        "first_seen": state["first"].isoformat(timespec="microseconds").replace("+00:00", "Z"),
                        "last_seen": state["last"].isoformat(timespec="microseconds").replace("+00:00", "Z"),
                        "counts": state["counts"],
                        "net_revenue": format(state["revenue"], ".2f"),
                    })

                return {
                    "total_unique": len(ordered),
                    "duplicate_count": duplicate_count,
                    "ordered_events": [item[0] for item in ordered],
                    "users": user_rows,
                }
            '''
        ),
    },
]
