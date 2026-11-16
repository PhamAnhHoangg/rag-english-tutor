"""Embed chunks and upsert them into Qdrant. Needs the `ml` extras."""
import time
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


def make_qdrant(url: str, timeout: int = 60):
    """Qdrant client with a generous timeout: dropping or creating a collection on a
    Docker Desktop volume (Windows/WSL) can take longer than the 5 s default."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=timeout)


def with_retry(fn, attempts: int = 4, wait: float = 3.0, sleep=None):
    """Retry fn on timeouts and connection errors, waiting longer each time."""
    import httpx

    sleep = sleep or time.sleep
    for i in range(attempts):
        try:
            return fn()
        except (httpx.TimeoutException, httpx.TransportError, TimeoutError) as e:
            if i == attempts - 1:
                raise SystemExit(f"Qdrant did not respond after {attempts} tries ({e!r}). "
                                 "Check it is running: docker compose ps") from e
            print(f"Qdrant timed out, retrying in {wait * 2 ** i:.0f}s...", flush=True)
            sleep(wait * 2 ** i)


class Indexer:
    def __init__(self, qdrant_url: str, collection: str, embedder, client=None, timeout: int = 60):
        self.client = client or make_qdrant(qdrant_url, timeout)
        self.collection, self.embedder = collection, embedder

    def ensure_collection(self, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams

        dim = self.embedder.dim
        exists = lambda: with_retry(lambda: self.client.collection_exists(self.collection))  # noqa: E731
        if recreate and exists():
            # A timed-out delete may still finish on the server, so retry until it is gone.
            with_retry(lambda: exists() and self.client.delete_collection(self.collection))
        if not exists():
            # Same for create: retry only if the collection really is not there yet.
            with_retry(lambda: exists() or self.client.create_collection(
                self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)))
        else:
            have = self.client.get_collection(self.collection).config.params.vectors.size
            if have != dim:
                raise SystemExit(f"Collection {self.collection!r} holds {have}-dim vectors but the embedder makes "
                                 f"{dim}-dim ones (the embedding model changed). Re-run ingest with --recreate.")

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
            with_retry(lambda: self.client.upsert(self.collection, points))
            total += len(points)
            print(f"indexed {total} chunks", flush=True)
        return total
