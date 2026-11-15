from tutor.chunking import Chunk
from tutor.pipeline import answer, build_prompt


class FakeRetriever:
    def search(self, query, k=5):
        return [Chunk("affect: to influence", "wiktionary", "wiktionary:affect")]


class FakeLLM:
    def generate(self, prompt):
        assert "[1] (wiktionary) affect: to influence" in prompt
        return "'Affect' means to influence [1]."


def test_answer_cites_sources():
    a = answer("What does affect mean?", FakeRetriever(), FakeLLM())
    assert "[1]" in a.text and a.sources[0].ref == "wiktionary:affect"


def test_prompt_numbers_sources():
    p = build_prompt("q", [Chunk("x", "s", "r"), Chunk("y", "s", "r2")])
    assert "[1] (s) x" in p and "[2] (s) y" in p
