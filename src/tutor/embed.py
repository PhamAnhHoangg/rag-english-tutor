"""Embedding backends, picked in configs/default.yaml under `embeddings`.

- fastembed (default): small English model on ONNX, fast on a plain CPU, no PyTorch needed.
- bge-m3: the large multilingual model; much slower on CPU, use it with a GPU.
"""
import os
from typing import Protocol


def default_cache_dir() -> str | None:
    """Keep fastembed's model files under HF_HOME (on E: for Hai), not the system temp dir."""
    return os.environ.get("FASTEMBED_CACHE_PATH") or (
        os.path.join(os.environ["HF_HOME"], "fastembed") if os.environ.get("HF_HOME") else None)


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 by default (384 dims). Queries get the model's search prefix."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model, cache_dir=cache_dir or default_cache_dir())
        self.batch_size = batch_size
        self.dim = len(self.embed_query("probe"))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


class BgeM3Embedder:
    dim = 1024

    def __init__(self, model: str = "BAAI/bge-m3", batch_size: int = 64):
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(model, use_fp16=True)
        self.batch_size = batch_size

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.encode(texts, batch_size=self.batch_size)["dense_vecs"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_passages([text])[0]


def make_embedder(cfg: dict) -> Embedder:
    backend = cfg.get("backend", "fastembed")
    kwargs = {"model": cfg["model"], "batch_size": cfg.get("batch_size", 64)}
    if backend == "fastembed":
        return FastEmbedder(**kwargs)
    if backend == "bge-m3":
        return BgeM3Embedder(**kwargs)
    raise ValueError(f"unknown embeddings.backend {backend!r}: use 'fastembed' or 'bge-m3'")
