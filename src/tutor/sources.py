"""Download and parse the three English sources into Chunks.

Parsers take plain iterables (lines, dicts) so they can be tested offline;
the download helpers stream files to data/raw/.
"""
import bz2
import json
import os
import re
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import httpx

from tutor.chunking import Chunk, dictionary_entry, fixed_size, sentence_pairs

KAIKKI_URL = "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl"
TATOEBA_URL = "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2"
WIKIBOOKS_API = "https://en.wikibooks.org/w/api.php"
WIKIBOOKS_PREFIX = "English Grammar"
# Wikimedia asks API clients for a User-Agent with contact info (a URL or email).
# Set RAG_TUTOR_CONTACT to add yours, e.g. your GitHub repo URL.
_CONTACT = os.environ.get("RAG_TUTOR_CONTACT", "").strip()
USER_AGENT = f"rag-english-tutor/0.1 ({_CONTACT + '; ' if _CONTACT else ''}educational RAG project) httpx/{httpx.__version__}"
RETRY_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    pass


def get_json(client: httpx.Client, url: str, params: dict, retries: int = 5, backoff: float = 2.0,
             sleep=time.sleep) -> dict:
    """GET JSON with clear errors and retries on rate limits, server errors and lag.

    Raises FetchError with the status and the start of the body when the reply is not JSON,
    instead of a bare JSONDecodeError.
    """
    for attempt in range(retries + 1):
        try:
            r = client.get(url, params=params)
        except httpx.TransportError as e:
            if attempt == retries:
                raise FetchError(f"network error for {url}: {e}") from e
            sleep(backoff * 2 ** attempt)
            continue
        is_json = "json" in r.headers.get("content-type", "")
        lagged = is_json and r.status_code == 200 and r.json().get("error", {}).get("code") == "maxlag"
        if r.status_code in RETRY_STATUS or lagged:
            if attempt == retries:
                raise FetchError(f"{url} still failing after {retries} retries: HTTP {r.status_code}")
            wait = float(r.headers.get("retry-after") or backoff * 2 ** attempt)
            sleep(min(wait, 60))
            continue
        if r.status_code != 200 or not is_json:
            snippet = r.text[:200].replace("\n", " ")
            raise FetchError(f"HTTP {r.status_code} ({r.headers.get('content-type', '?')}) from {url}: {snippet}")
        data = r.json()
        if "error" in data:
            raise FetchError(f"API error from {url}: {data['error']}")
        return data
    raise AssertionError("unreachable")


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

SKIP_POS = {"name", "character", "symbol", "punct", "prefix", "suffix", "infix", "affix", "phrase"}
SKIP_TAGS = {"form-of", "alt-of", "obsolete", "archaic", "rare", "dated", "misspelling", "abbreviation"}


def parse_kaikki(lines: Iterable[str], limit: int | None = None, vocab: set[str] | None = None,
                 max_senses: int = 4) -> Iterator[Chunk]:
    """English entries only. One chunk per sense: gloss plus up to two examples.

    `vocab` keeps only those words (lowercase), so a learner-sized index holds common words
    instead of whatever comes first in the file. Inflected forms, archaic/rare senses, proper
    names and affixes are skipped, and each entry keeps at most `max_senses` senses.
    """
    seen = 0
    for line in lines:
        entry = json.loads(line)
        word, pos = entry.get("word"), entry.get("pos", "")
        if entry.get("lang_code") != "en" or not word or pos in SKIP_POS:
            continue
        if vocab is not None and word.lower() not in vocab:
            continue
        senses = []
        for s in entry.get("senses", []):
            glosses = s.get("glosses") or []
            if not glosses or s.get("form_of") or s.get("alt_of") or SKIP_TAGS & set(s.get("tags", [])):
                continue
            text = f"({pos}) {'; '.join(glosses)}" if pos else "; ".join(glosses)
            examples = [e["text"] for e in s.get("examples", []) if e.get("text")][:2]
            if examples:
                text += " Examples: " + " / ".join(examples)
            senses.append(text)
            if len(senses) >= max_senses:
                break
        if not senses:
            continue
        yield from dictionary_entry(word, senses, "wiktionary", f"wiktionary:{word}:{pos}")
        seen += 1
        if limit and seen >= limit:
            return


def build_vocab(sentences: Iterable[str], size: int) -> set[str]:
    """The `size` most frequent words in English example sentences: a free, learner-focused word list."""
    counts: dict[str, int] = {}
    for text in sentences:
        for w in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()):
            counts[w] = counts.get(w, 0) + 1
    return {w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:size]}


def tatoeba_sentences(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3 and parts[1] == "eng":
            yield parts[2]


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


def fetch_wikibooks(prefix: str = WIKIBOOKS_PREFIX, limit: int | None = None,
                    client: httpx.Client | None = None, pause: float = 0.5) -> Iterator[tuple[str, str]]:
    """Yield (title, plain text) for pages whose title starts with `prefix`.

    Requests run one at a time with a short pause, as Wikimedia's API etiquette asks.
    A page that keeps failing is skipped with a warning rather than stopping the ingest.
    """
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)
    base = {"format": "json", "formatversion": 2, "maxlag": 5}
    try:
        params = {**base, "action": "query", "list": "allpages", "apprefix": prefix,
                  "apnamespace": 0, "aplimit": 50}
        titles: list[str] = []
        while True:
            data = get_json(client, WIKIBOOKS_API, params)
            titles += [p["title"] for p in data["query"]["allpages"]]
            if "continue" not in data or (limit and len(titles) >= limit):
                break
            params.update(data["continue"])
        for title in titles[:limit] if limit else titles:
            time.sleep(pause)
            try:
                data = get_json(client, WIKIBOOKS_API, {**base, "action": "query", "prop": "extracts",
                                                        "explaintext": 1, "titles": title})
            except FetchError as e:
                print(f"warning: skipping Wikibooks page {title!r}: {e}", file=sys.stderr)
                continue
            for page in data["query"]["pages"]:
                if page.get("extract"):
                    yield page["title"], page["extract"]
    finally:
        if own:
            client.close()
