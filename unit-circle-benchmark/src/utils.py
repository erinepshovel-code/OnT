# src/utils.py
import json
import orjson
from pathlib import Path
from typing import Any


def save_json(path: str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def load_json(path: str) -> Any:
    return orjson.loads(Path(path).read_bytes())


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def pretty_print_results(runs: dict) -> None:
    from src.metrics import format_results_table
    print(format_results_table(runs))
