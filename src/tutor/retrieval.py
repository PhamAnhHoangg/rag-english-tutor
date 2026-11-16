"""Retriever interface. Steps on the experiment ladder each add one implementation."""
from typing import Protocol

from tutor.chunking import Chunk



class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Chunk]: ...


def payload_to_chunk(payload: dict) -> Chunk:
    meta = {k: v for k, v in payload.items() if k not in {"text", "source", "ref"}}
    return Chunk(payload["text"], payload["source"], payload["ref"], meta)


class DenseRetriever:
    """Baseline: embed the query and take the nearest chunks from Qdrant."""

    def __init__(self, qdrant_url: str, collection: str, embedder, client=None, timeout: int = 60):
        if client is None:
            from tutor.index import make_qdrant

            client = make_qdrant(qdrant_url, timeout)
        self.client, self.collection, self.embedder = client, collection, embedder

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        vec = self.embedder.embed_query(query)
        res = self.client.query_points(self.collection, query=vec, limit=k, with_payload=True)
        chunks = []
        for p in res.points:
            c = payload_to_chunk(p.payload)
            c.meta["score"] = p.score
            chunks.append(c)
        return chunks
