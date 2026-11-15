"""Retriever interface. Steps on the experiment ladder each add one implementation."""
from typing import Protocol

from tutor.chunking import Chunk


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Chunk]: ...


class DenseRetriever:
    """Baseline: bge-m3 embeddings in Qdrant. TODO in the baseline milestone."""

    def __init__(self, qdrant_url: str, collection: str, embed_model: str):
        self.qdrant_url, self.collection, self.embed_model = qdrant_url, collection, embed_model

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        raise NotImplementedError("Implement in milestone 2 (baseline)")
