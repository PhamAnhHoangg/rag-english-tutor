"""Download the English sources, chunk them, and (unless --chunks-only) index them in Qdrant.

Examples:
  python scripts/ingest.py --limit 2000 --chunks-only   # quick look at chunks, no models needed
  python scripts/ingest.py --limit 2000                  # small index to build the baseline on
  python scripts/ingest.py                               # everything (Wiktionary is a ~2 GB download)
"""
import argparse
import itertools
import json
from pathlib import Path

from tutor import sources
from tutor.config import load_config


def iter_chunks(names: list[str], raw: Path, limit: int | None, cfg: dict):
    ck = cfg["chunking"]
    if "wiktionary" in names:
        path = sources.download(sources.KAIKKI_URL, raw / "kaikki-english.jsonl")
        with open(path, encoding="utf-8") as f:
            yield from sources.parse_kaikki(f, limit)
    if "tatoeba" in names:
        path = sources.download(sources.TATOEBA_URL, raw / "eng_sentences.tsv.bz2")
        yield from sources.parse_tatoeba(sources.tatoeba_lines(path), limit)
    if "wikibooks" in names:
        for title, text in sources.fetch_wikibooks(limit=limit):
            yield from sources.parse_wikibooks_page(title, text, ck["max_chars"], ck["overlap_chars"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["wiktionary", "tatoeba", "wikibooks"])
    p.add_argument("--limit", type=int, default=None, help="max entries per source")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--chunks-only", action="store_true", help="write data/chunks.jsonl and skip indexing")
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)

    chunks = iter_chunks(args.sources, args.data_dir / "raw", args.limit, cfg)
    out = args.data_dir / "chunks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    def logged(it):
        with open(out, "w", encoding="utf-8") as f:
            for n, c in enumerate(it, 1):
                f.write(json.dumps({"text": c.text, "source": c.source, "ref": c.ref, "meta": c.meta}) + "\n")
                if n % 10000 == 0:
                    print(f"{n} chunks")
                yield c

    if args.chunks_only:
        n = sum(1 for _ in logged(chunks))
        print(f"Wrote {n} chunks to {out}")
        return

    from tutor.index import Indexer

    r = cfg["retrieval"]
    n = Indexer(r["qdrant_url"], r["collection"], cfg["embeddings"]["model"]).index(logged(chunks))
    print(f"Indexed {n} chunks into {r['collection']}")


if __name__ == "__main__":
    main()
