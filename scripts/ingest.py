"""Download the English sources, chunk them, and (unless --chunks-only) index them in Qdrant.

Examples:
  python scripts/ingest.py --limit 2000 --chunks-only   # quick look at chunks, no models needed
  python scripts/ingest.py --limit 2000                  # small index to build the baseline on
  python scripts/ingest.py --vocab 5000 --recreate      # common words only, rebuild the index
  python scripts/ingest.py                               # everything (Wiktionary is a ~2 GB download)
"""
import argparse
import itertools
import json
from pathlib import Path

from tutor import sources
from tutor.config import load_config


def iter_chunks(names: list[str], raw: Path, limits: dict, cfg: dict, vocab_size: int | None = None):
    ck = cfg["chunking"]
    tatoeba_path = None
    if "tatoeba" in names or vocab_size:
        tatoeba_path = sources.download(sources.TATOEBA_URL, raw / "eng_sentences.tsv.bz2")
    if "wiktionary" in names:
        vocab = None
        if vocab_size:
            vocab = sources.build_vocab(sources.tatoeba_sentences(sources.tatoeba_lines(tatoeba_path)), vocab_size)
            print(f"Vocabulary: {vocab_size} most frequent words in Tatoeba")
        path = sources.download(sources.KAIKKI_URL, raw / "kaikki-english.jsonl")
        with open(path, encoding="utf-8") as f:
            yield from sources.parse_kaikki(f, limits["wiktionary"], vocab=vocab)
    if "tatoeba" in names:
        yield from sources.parse_tatoeba(sources.tatoeba_lines(tatoeba_path), limits["tatoeba"])
    if "wikibooks" in names:
        for title, text in sources.fetch_wikibooks(limit=limits["wikibooks"]):
            yield from sources.parse_wikibooks_page(title, text, ck["max_chars"], ck["overlap_chars"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["wiktionary", "tatoeba", "wikibooks"])
    p.add_argument("--limit", type=int, default=None, help="max entries per source")
    for src in ("wiktionary", "tatoeba", "wikibooks"):
        p.add_argument(f"--limit-{src}", type=int, default=None, help=f"override --limit for {src}")
    p.add_argument("--vocab", type=int, default=None,
                   help="keep only Wiktionary entries for the N most frequent words in Tatoeba")
    p.add_argument("--recreate", action="store_true", help="drop the Qdrant collection before indexing")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--chunks-only", action="store_true", help="write data/chunks.jsonl and skip indexing")
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)

    limits = {src: getattr(args, f"limit_{src}") or args.limit for src in ("wiktionary", "tatoeba", "wikibooks")}
    chunks = iter_chunks(args.sources, args.data_dir / "raw", limits, cfg, args.vocab)
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

    from tutor.embed import make_embedder
    from tutor.index import Indexer

    r = cfg["retrieval"]
    n = Indexer(r["qdrant_url"], r["collection"], make_embedder(cfg["embeddings"])).index(logged(chunks), recreate=args.recreate)
    print(f"Indexed {n} chunks into {r['collection']}")


if __name__ == "__main__":
    main()
