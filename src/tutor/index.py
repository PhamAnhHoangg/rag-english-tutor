"""Embed chunks and upsert them into Qdrant. Needs the `ml` extras."""
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
    def __init__(self, qdrant_url: str, collection: str, embedder, client=None):
        if client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=qdrant_url)
        self.client, self.collection, self.embedder = client, collection, embedder

    def ensure_collection(self, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams

        dim = self.embedder.dim
        if recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if self.client.collection_exists(self.collection):
            have = self.client.get_collection(self.collection).config.params.vectors.size
            if have != dim:
                raise SystemExit(f"Collection {self.collection!r} holds {have}-dim vectors but the embedder makes "
                                 f"{dim}-dim ones (the embedding model changed). Re-run ingest with --recreate.")
        else:
            self.client.create_collection(self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def index(self, chunks: Iterable[Chunk], batch_size: int = 64, recreate: bool = False) -> int:
        from qdrant_client.models import PointStruct

        self.ensure_collection(recreate=recreate)
        total = 0
        for batch in batched(chunks, batch_size):
            vecs = self.embedder.embed_passages([c.text for c in batch])
            points = [
                PointStruct(id=chunk_id(c), vector=v,
                            payload={"text": c.text, "source": c.source, "ref": c.ref, **c.meta})
                for c, v in zip(batch, vecs)
            ]
            self.client.upsert(self.collection, points)
            total += len(points)
            print(f"indexed {total} chunks", flush=True)
        return total
