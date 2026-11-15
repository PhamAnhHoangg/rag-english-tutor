"""Build a retrieval test set by asking the local LLM to write a learner question for sampled chunks.

The chunk a question was written from is its gold reference. Generated questions are a draft:
read them, fix or delete bad ones, and set "reviewed": true on the ones you keep.
"""
import json
import random
import re

from tutor.llm import LLM

PROMPT = """You help build a test set for an English-learning assistant.
Below is a passage from a study source ({source}). Write ONE question that an English learner
might ask, which this passage answers. Do not quote the passage or mention it. Use natural wording.
Then give a short correct answer based only on the passage.

Question types: {qtype}

Passage:
{text}

Reply with JSON only: {{"question": "...", "answer": "..."}}"""

QTYPES = {
    "wiktionary": "word meaning, difference between similar words, how a word is used",
    "tatoeba": "asking for example sentences or how to say something naturally",
    "wikibooks": "a grammar rule, when to use a form, why a sentence is correct",
}


def sample_chunks(chunks: list[dict], per_source: dict[str, int], seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for source, n in per_source.items():
        pool = [c for c in chunks if c["source"] == source and len(c["text"]) >= 40]
        out += rng.sample(pool, min(n, len(pool)))
    return out


def parse_reply(reply: str) -> dict | None:
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not str(data.get("question", "")).strip() or not str(data.get("answer", "")).strip():
        return None
    return {"question": data["question"].strip(), "answer": data["answer"].strip()}


def generate(chunks: list[dict], llm: LLM) -> list[dict]:
    items = []
    for i, c in enumerate(chunks):
        reply = llm.generate(PROMPT.format(source=c["source"], qtype=QTYPES.get(c["source"], "any"), text=c["text"]))
        qa = parse_reply(reply)
        if qa is None:
            continue
        items.append({"id": f"q{i + 1:03d}", "source": c["source"], **qa,
                      "gold_refs": [c["ref"]], "reviewed": False})
    return items
