from __future__ import annotations

import json
import sys
import urllib.request


def main() -> int:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as response:
            if response.status != 200:
                return 1
            payload = json.load(response)
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                return 1
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
