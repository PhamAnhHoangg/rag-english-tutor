"""Chunkers per source type. Each chunk keeps its source so answers can cite it."""
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source: str               # e.g. "wiktionary", "tatoeba", "wikibooks"
    ref: str                  # stable id or URL of the original entry
    meta: dict = field(default_factory=dict)


def fixed_size(text: str, source: str, ref: str, max_chars: int = 800, overlap: int = 100) -> list[Chunk]:
    """Baseline: fixed windows with overlap, cut at the last space when possible."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(Chunk(text[start:end].strip(), source, ref, {"start": start}))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def dictionary_entry(word: str, senses: list[str], source: str, ref: str) -> list[Chunk]:
    """One chunk per sense, prefixed with the headword so it retrieves on the word."""
    return [Chunk(f"{word}: {s}", source, ref, {"word": word, "sense": i}) for i, s in enumerate(senses)]


def sentence_pairs(pairs: list[tuple[str, str]], source: str, ref_prefix: str, group: int = 5) -> list[Chunk]:
    """Example sentences are tiny, so group a few per chunk."""
    chunks = []
    for i in range(0, len(pairs), group):
        batch = pairs[i : i + group]
        text = "\n".join(f"{en} ({note})" if note else en for en, note in batch)
        chunks.append(Chunk(text, source, f"{ref_prefix}#{i}"))
    return chunks
