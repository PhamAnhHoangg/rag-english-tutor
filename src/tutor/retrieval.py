"""Retriever interface. Steps on the experiment ladder each add one implementation."""
from collections.abc import Callable
from typing import Protocol

from tutor.chunking import Chunk

Encoder = Callable[[list[str]], list[list[float]]]


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Chunk]: ...


def bge_m3_encoder(model_name: str = "BAAI/bge-m3") -> Encoder:
    """Dense bge-m3 vectors. Needs the `ml` extras; loads the model once."""
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(model_name, use_fp16=True)
    return lambda texts: [v.tolist() for v in model.encode(texts)["dense_vecs"]]


def payload_to_chunk(payload: dict) -> Chunk:
    meta = {k: v for k, v in payload.items() if k not in {"text", "source", "ref"}}
    return Chunk(payload["text"], payload["source"], payload["ref"], meta)


class DenseRetriever:
    """Baseline: embed the query with bge-m3 and take the nearest chunks from Qdrant."""

    def __init__(self, qdrant_url: str, collection: str, embed_model: str = "BAAI/bge-m3",
                 encoder: Encoder | None = None, client=None):
        if client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=qdrant_url)
        self.client, self.collection = client, collection
        self._encoder, self._embed_model = encoder, embed_model

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:  # lazy: loading bge-m3 takes a while
            self._encoder = bge_m3_encoder(self._embed_model)
        return self._encoder

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        vec = self.encoder([query])[0]
        res = self.client.query_points(self.collection, query=vec, limit=k, with_payload=True)
        chunks = []
        for p in res.points:
            c = payload_to_chunk(p.payload)
            c.meta["score"] = p.score
            chunks.append(c)
        return chunks
