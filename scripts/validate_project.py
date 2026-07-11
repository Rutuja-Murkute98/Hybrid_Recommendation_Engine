from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "app.py",
    "extensions.py",
    "reel_engine/__init__.py",
    "reel_engine/routes.py",
    "product_engine/__init__.py",
    "product_engine/zatch_mongo_recommender.py",
    "product_engine/recommender.py",
    "product_engine/final_recommender.py",
    "product_engine/routes.py",
    "requirements.txt",
    "requirements-dev.txt",
    ".env.example",
    "render.yaml",
    "Procfile",
    "README.md",
    "scripts/train_model.py",
    "scripts/ensure_indexes.py",
    "scripts/run_server.py",
]

def validate_files() -> list[str]:
    errors = []
    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).exists():
            errors.append(f"Missing required file: {relative_path}")
    return errors


def validate_env_example() -> list[str]:
    path = ROOT / ".env.example"
    if not path.exists():
        return ["Missing required file: .env.example"]

    content = path.read_text(encoding="utf-8")
    required_keys = ["MONGO_URI=", "MONGO_DB_NAME=", "MONGO_TIMEOUT_MS="]
    missing = [key for key in required_keys if key not in content]
    if missing:
        return [f".env.example missing keys: {missing}"]
    return []


def main() -> int:
    errors = []
    errors.extend(validate_files())
    errors.extend(validate_env_example())

    if errors:
        print("Project validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Project validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
