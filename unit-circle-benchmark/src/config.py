# src/config.py
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbedConfig:
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    doc_prefix: Optional[str] = "search_document: "
    query_prefix: Optional[str] = "search_query: "
    batch_size: int = 64


# Named presets for the models tested in the benchmark
EMBED_PRESETS = {
    "nomic": EmbedConfig(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
    ),
    "bge-m3": EmbedConfig(
        model_name="BAAI/bge-m3",
        doc_prefix=None,
        query_prefix=None,
    ),
    # text-embedding-3-large requires the openai package; handled separately
}


@dataclass
class TransformConfig:
    name: str
    out_dim: int
    mode: Optional[str] = None  # used by unit_circle_transform


# Ablation ladder as described in the spec
ABLATION_LADDER: List[TransformConfig] = [
    TransformConfig("native",    768, None),
    TransformConfig("pca_256",   256, None),
    TransformConfig("uc_256",    256, "direct"),
    TransformConfig("pca_64",     64, None),
    TransformConfig("uc_64",      64, "direct"),
    TransformConfig("uc_256_rank", 256, "rank"),
    TransformConfig("uc_64_rank",   64, "rank"),
]


@dataclass
class BenchmarkConfig:
    corpus_path: str = "data/corpus.jsonl"
    queries_path: str = "data/queries.jsonl"
    qrels_path: str = "data/qrels.jsonl"
    output_dir: str = "outputs"
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    geometry_k: int = 10
    embed: EmbedConfig = field(default_factory=EmbedConfig)
