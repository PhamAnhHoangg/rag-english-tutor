"""Ask the tutor a question from the command line.

  python scripts/ask.py "What is the difference between affect and effect?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without PYTHONPATH or pip install -e

import argparse

from tutor.config import load_config
from tutor.llm import OllamaLLM
from tutor.pipeline import answer
from tutor.retrieval import MODES, make_retriever


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sources-only", action="store_true", help="show retrieved chunks, skip the LLM")
    p.add_argument("--mode", choices=MODES, default=None, help="override retrieval.mode from the config")
    args = p.parse_args()
    cfg = load_config(args.config)
    r = cfg["retrieval"]
    retriever = make_retriever(cfg, args.mode)

    if args.sources_only:
        for i, c in enumerate(retriever.search(args.question, k=r["top_k"]), 1):
            score = c.meta.get("rerank", c.meta.get("score", 0))
            print(f"[{i}] {score:.3f} {c.ref}\n    {c.text[:200]}")
        return

    a = answer(args.question, retriever, OllamaLLM(**cfg["llm"]), k=r["top_k"])
    print(a.text, "\n\nSources:")
    for i, c in enumerate(a.sources, 1):
        print(f"[{i}] {c.ref}")


if __name__ == "__main__":
    main()
