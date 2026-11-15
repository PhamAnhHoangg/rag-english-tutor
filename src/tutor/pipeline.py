"""Retrieve, build a prompt that forces citations, generate."""
from dataclasses import dataclass

from tutor.chunking import Chunk
from tutor.llm import LLM
from tutor.retrieval import Retriever

SYSTEM = (
    "You are an English tutor. Answer the learner's question using ONLY the numbered sources. "
    "Cite sources inline like [1]. Give one or two example sentences. "
    "If the sources do not contain the answer, say you don't know."
)


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    sources = "\n\n".join(f"[{i}] ({c.source}) {c.text}" for i, c in enumerate(chunks, 1))
    return f"{SYSTEM}\n\nSources:\n{sources}\n\nQuestion: {question}\nAnswer:"


@dataclass
class Answer:
    text: str
    sources: list[Chunk]


def answer(question: str, retriever: Retriever, llm: LLM, k: int = 5) -> Answer:
    chunks = retriever.search(question, k=k)
    return Answer(llm.generate(build_prompt(question, chunks)), chunks)
