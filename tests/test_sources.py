from pathlib import Path

from tutor.index import batched, chunk_id
from tutor.sources import parse_kaikki, parse_tatoeba, parse_wikibooks_page, split_sections

FIX = Path(__file__).parent / "fixtures"


def test_kaikki_english_senses_only():
    chunks = list(parse_kaikki(open(FIX / "kaikki_sample.jsonl")))
    assert [c.ref for c in chunks] == ["wiktionary:affect:verb"] * 2 + ["wiktionary:effect:noun"]
    assert chunks[0].text == "affect: (verb) To influence or alter. Examples: The weather affected my mood."


def test_kaikki_limit_counts_entries():
    assert len(list(parse_kaikki(open(FIX / "kaikki_sample.jsonl"), limit=1))) == 2


def test_tatoeba_keeps_english():
    chunks = list(parse_tatoeba(open(FIX / "tatoeba_sample.tsv")))
    assert len(chunks) == 1 and "faim" not in chunks[0].text
    assert "I am hungry." in chunks[0].text and "lived here" in chunks[0].text


def test_wikibooks_chunks_stay_inside_sections():
    text = "Intro line.\n== Present perfect ==\nUse it for past actions linked to now.\n== Past simple ==\nUse it with a finished time."
    assert [h for h, _ in split_sections("English Grammar/Tenses", text)] == [
        "English Grammar/Tenses", "English Grammar/Tenses > Present perfect", "English Grammar/Tenses > Past simple"]
    chunks = parse_wikibooks_page("English Grammar/Tenses", text)
    assert chunks[1].text.startswith("English Grammar/Tenses > Present perfect: Use it")
    assert chunks[1].ref == "wikibooks:English_Grammar/Tenses_>_Present_perfect"


def test_chunk_ids_are_stable_and_batches_cover_all():
    chunks = list(parse_kaikki(open(FIX / "kaikki_sample.jsonl")))
    assert chunk_id(chunks[0]) == chunk_id(chunks[0]) != chunk_id(chunks[1])
    assert [len(b) for b in batched(chunks, 2)] == [2, 1]
