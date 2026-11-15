"""Ask the tutor a question from the command line.

  python scripts/ask.py "What is the difference between affect and effect?"
"""
import argparse

from tutor.config import load_config
from tutor.llm import OllamaLLM
from tutor.pipeline import answer
from tutor.retrieval import DenseRetriever


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sources-only", action="store_true", help="show retrieved chunks, skip the LLM")
    args = p.parse_args()
    cfg = load_config(args.config)
    r = cfg["retrieval"]
    retriever = DenseRetriever(r["qdrant_url"], r["collection"], cfg["embeddings"]["model"])

    if args.sources_only:
        for i, c in enumerate(retriever.search(args.question, k=r["top_k"]), 1):
            print(f"[{i}] {c.meta.get('score', 0):.3f} {c.ref}\n    {c.text[:200]}")
        return

    a = answer(args.question, retriever, OllamaLLM(**cfg["llm"]), k=r["top_k"])
    print(a.text, "\n\nSources:")
    for i, c in enumerate(a.sources, 1):
        print(f"[{i}] {c.ref}")


if __name__ == "__main__":
    main()
