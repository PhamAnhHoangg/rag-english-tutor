"""Download sources, chunk them, embed and index into Qdrant.

TODO (milestone 1):
  1. Wiktionary English entries from kaikki.org (JSONL) -> chunking.dictionary_entry
  2. Tatoeba English sentences -> chunking.sentence_pairs
  3. Wikibooks English grammar pages -> chunking.fixed_size
  4. Embed with bge-m3, upsert into Qdrant with source + ref in the payload
"""

if __name__ == "__main__":
    raise SystemExit("Not implemented yet: see the TODO list in this file.")
