from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_engine.zatch_mongo_recommender import ZatchConfigError, ensure_indexes, get_db


def main() -> int:
    try:
        db = get_db()
    except ZatchConfigError as exc:
        print(f"Cannot create indexes without MongoDB: {exc}")
        return 1

    ensure_indexes(db)
    print("Indexes ensured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
