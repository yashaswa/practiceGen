from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict

from config import DATA_DIR

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def processed_path(doc_id: str) -> Path:
    return PROCESSED_DIR / f"{doc_id}.json"

def save_processed(doc_id: str, payload: Dict[str, Any]) -> None:
    path = processed_path(doc_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def load_processed(doc_id: str) -> Dict[str, Any] | None:
    path = processed_path(doc_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None