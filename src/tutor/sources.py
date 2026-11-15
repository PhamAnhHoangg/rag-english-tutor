"""Download and parse the three English sources into Chunks.

Parsers take plain iterables (lines, dicts) so they can be tested offline;
the download helpers stream files to data/raw/.
"""
import bz2
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import httpx

from tutor.chunking import Chunk, dictionary_entry, fixed_size, sentence_pairs

KAIKKI_URL = "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl"
TATOEBA_URL = "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2"
WIKIBOOKS_API = "https://en.wikibooks.org/w/api.php"
WIKIBOOKS_PREFIX = "English Grammar"
USER_AGENT = "rag-english-tutor/0.1 (educational project)"


def download(url: str, dest: Path) -> Path:
    """Stream a file to disk; skip if it is already there."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_bytes(1 << 20):
                f.write(block)
    tmp.rename(dest)
    return dest


# --- Wiktionary (kaikki.org JSONL, one entry per line) ---

def parse_kaikki(lines: Iterable[str], limit: int | None = None) -> Iterator[Chunk]:
    """English entries only. One chunk per sense: gloss plus up to two examples."""
    seen = 0
    for line in lines:
        entry = json.loads(line)
        if entry.get("lang_code") != "en" or not entry.get("word"):
            continue
        word, pos = entry["word"], entry.get("pos", "")
        senses = []
        for s in entry.get("senses", []):
            glosses = s.get("glosses") or []
            if not glosses:
                continue
            text = f"({pos}) {'; '.join(glosses)}" if pos else "; ".join(glosses)
            examples = [e["text"] for e in s.get("examples", []) if e.get("text")][:2]
            if examples:
                text += " Examples: " + " / ".join(examples)
            senses.append(text)
        if not senses:
            continue
        yield from dictionary_entry(word, senses, "wiktionary", f"wiktionary:{word}:{pos}")
        seen += 1
        if limit and seen >= limit:
            return


# --- Tatoeba (TSV: id <tab> lang <tab> text) ---

def parse_tatoeba(lines: Iterable[str], limit: int | None = None, group: int = 5) -> Iterator[Chunk]:
    batch, n = [], 0
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[1] != "eng":
            continue
        batch.append((parts[2], ""))
        n += 1
        if len(batch) == group:
            yield from sentence_pairs(batch, "tatoeba", f"tatoeba:{parts[0]}", group=group)
            batch = []
        if limit and n >= limit:
            break
    if batch:
        yield from sentence_pairs(batch, "tatoeba", "tatoeba:tail", group=group)


def tatoeba_lines(path: Path) -> Iterator[str]:
    with bz2.open(path, "rt", encoding="utf-8") as f:
        yield from f


# --- Wikibooks (MediaWiki API, plain-text extracts) ---

def split_sections(title: str, text: str) -> Iterator[tuple[str, str]]:
    """Split a plain-text extract on '== Heading ==' lines, keeping the heading."""
    heading, buf = title, []
    for line in text.splitlines():
        m = re.match(r"^=+\s*(.*?)\s*=+$", line.strip())
        if m:
            if "".join(buf).strip():
                yield heading, "\n".join(buf).strip()
            heading, buf = f"{title} > {m.group(1)}", []
        else:
            buf.append(line)
    if "".join(buf).strip():
        yield heading, "\n".join(buf).strip()


def parse_wikibooks_page(title: str, text: str, max_chars: int = 800, overlap: int = 100) -> list[Chunk]:
    """Chunk within sections so a chunk never mixes two grammar topics."""
    chunks = []
    for heading, body in split_sections(title, text):
        ref = "wikibooks:" + heading.replace(" ", "_")
        for c in fixed_size(body, "wikibooks", ref, max_chars, overlap):
            c.text = f"{heading}: {c.text}"
            chunks.append(c)
    return chunks


def fetch_wikibooks(prefix: str = WIKIBOOKS_PREFIX, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """Yield (title, plain text) for pages whose title starts with `prefix`."""
    headers = {"User-Agent": USER_AGENT}
    params = {"action": "query", "list": "allpages", "apprefix": prefix, "apnamespace": 0,
              "aplimit": 50, "format": "json"}
    titles = []
    while True:
        data = httpx.get(WIKIBOOKS_API, params=params, headers=headers, timeout=30).json()
        titles += [p["title"] for p in data["query"]["allpages"]]
        if "continue" not in data or (limit and len(titles) >= limit):
            break
        params.update(data["continue"])
    for title in titles[:limit] if limit else titles:
        data = httpx.get(WIKIBOOKS_API, headers=headers, timeout=30, params={
            "action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json"}).json()
        for page in data["query"]["pages"].values():
            if page.get("extract"):
                yield page["title"], page["extract"]
