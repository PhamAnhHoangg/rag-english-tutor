"""Embed chunks with bge-m3 and upsert them into Qdrant. Needs the `ml` extras."""
import uuid
from collections.abc import Iterable, Iterator

from tutor.chunking import Chunk


def chunk_id(c: Chunk) -> str:
    """Stable id, so re-running ingest updates points instead of duplicating them."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.ref}|{c.meta}|{c.text[:64]}"))


def batched(items: Iterable[Chunk], size: int) -> Iterator[list[Chunk]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class Indexer:
    def __init__(self, qdrant_url: str, collection: str, embed_model: str = "BAAI/bge-m3"):
        from FlagEmbedding import BGEM3FlagModel
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection
        self.model = BGEM3FlagModel(embed_model, use_fp16=True)

    def ensure_collection(self, dim: int = 1024) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def index(self, chunks: Iterable[Chunk], batch_size: int = 64) -> int:
        from qdrant_client.models import PointStruct

        self.ensure_collection()
        total = 0
        for batch in batched(chunks, batch_size):
            vecs = self.model.encode([c.text for c in batch], batch_size=batch_size)["dense_vecs"]
            points = [
                PointStruct(id=chunk_id(c), vector=v.tolist(),
                            payload={"text": c.text, "source": c.source, "ref": c.ref, **c.meta})
                for c, v in zip(batch, vecs)
            ]
            self.client.upsert(self.collection, points)
            total += len(points)
        return total
