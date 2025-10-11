import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.server import app


def main():
    spec = app.openapi()
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    out_json = docs_dir / "openapi.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    print(f"Exported OpenAPI spec to {out_json}")


if __name__ == "__main__":
    main()