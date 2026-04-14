# src/data.py
import orjson
from pathlib import Path
from typing import List, Dict, Any


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records


def load_corpus(path: str = "data/corpus.jsonl") -> List[Dict[str, Any]]:
    """Each record must have at minimum 'id' and 'text'. Optional: 'label'."""
    return _load_jsonl(path)


def load_queries(path: str = "data/queries.jsonl") -> List[Dict[str, Any]]:
    """Each record must have 'id' and 'text'."""
    return _load_jsonl(path)


def load_qrels(path: str = "data/qrels.jsonl") -> List[Dict[str, Any]]:
    """Each record must have 'query_id', 'doc_id', and 'relevance'."""
    return _load_jsonl(path)


def validate_corpus(corpus: List[Dict[str, Any]]) -> None:
    for i, doc in enumerate(corpus):
        if "id" not in doc or "text" not in doc:
            raise ValueError(f"corpus record {i} missing 'id' or 'text': {doc}")


def validate_queries(queries: List[Dict[str, Any]]) -> None:
    for i, q in enumerate(queries):
        if "id" not in q or "text" not in q:
            raise ValueError(f"query record {i} missing 'id' or 'text': {q}")


def validate_qrels(qrels: List[Dict[str, Any]], corpus: List[Dict[str, Any]], queries: List[Dict[str, Any]]) -> None:
    doc_ids = {d["id"] for d in corpus}
    query_ids = {q["id"] for q in queries}
    for i, rel in enumerate(qrels):
        if rel["query_id"] not in query_ids:
            raise ValueError(f"qrel {i}: unknown query_id '{rel['query_id']}'")
        if rel["doc_id"] not in doc_ids:
            raise ValueError(f"qrel {i}: unknown doc_id '{rel['doc_id']}'")
