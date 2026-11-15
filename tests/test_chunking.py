from tutor.chunking import dictionary_entry, fixed_size, sentence_pairs


def test_fixed_size_covers_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = fixed_size(text, "wikibooks", "p1", max_chars=200, overlap=40)
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].text.startswith("word0") and chunks[-1].text.endswith("word399")
    assert chunks[1].meta["start"] < len(chunks[0].text)  # overlap


def test_dictionary_entry_one_chunk_per_sense():
    chunks = dictionary_entry("run", ["to move fast", "to manage"], "wiktionary", "wiktionary:run")
    assert [c.text for c in chunks] == ["run: to move fast", "run: to manage"]


def test_sentence_pairs_grouped():
    pairs = [(f"Sentence {i}.", "") for i in range(7)]
    chunks = sentence_pairs(pairs, "tatoeba", "tatoeba:en", group=5)
    assert len(chunks) == 2 and chunks[1].ref == "tatoeba:en#5"
