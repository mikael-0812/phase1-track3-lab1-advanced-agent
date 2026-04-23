from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable
from .schemas import QAExample, RunRecord

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def dataset_paths(path: str | Path) -> list[Path]:
    if isinstance(path, Path):
        return [path]
    return [Path(part.strip()) for part in path.split(",") if part.strip()]


def dataset_label(path: str | Path) -> str:
    parts = dataset_paths(path)
    if len(parts) == 1:
        return parts[0].name
    return "+".join(part.stem for part in parts) + ".json"


def load_dataset(path: str | Path) -> list[QAExample]:
    rows: list[QAExample] = []
    seen_qids: set[str] = set()
    for dataset_path in dataset_paths(path):
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
        for item in raw:
            example = QAExample.model_validate(item)
            if example.qid in seen_qids:
                raise ValueError(f"Duplicate qid found while loading datasets: {example.qid}")
            seen_qids.add(example.qid)
            rows.append(example)
    return rows

def save_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
