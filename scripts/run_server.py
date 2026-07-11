from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import main


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.INFO)
    main()
