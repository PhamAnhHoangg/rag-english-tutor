"""Draft a test set from data/chunks.jsonl with the local LLM (Ollama).

  python scripts/make_testset.py --wiktionary 25 --tatoeba 10 --wikibooks 15
Then review eval/questions.jsonl by hand.
"""
import argparse
import json
from pathlib import Path

from tutor.config import load_config
from tutor.llm import OllamaLLM
from tutor.testset import generate, sample_chunks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    p.add_argument("--out", type=Path, default=Path("eval/questions.jsonl"))
    p.add_argument("--wiktionary", type=int, default=25)
    p.add_argument("--tatoeba", type=int, default=10)
    p.add_argument("--wikibooks", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    chunks = [json.loads(line) for line in open(args.chunks, encoding="utf-8")]
    picked = sample_chunks(chunks, {"wiktionary": args.wiktionary, "tatoeba": args.tatoeba,
                                    "wikibooks": args.wikibooks}, args.seed)
    llm = OllamaLLM(**load_config(args.config)["llm"])
    items = []
    for i, c in enumerate(picked, 1):
        print(f"{i}/{len(picked)} {c['ref']}")
        items += generate([c], llm)
    for i, it in enumerate(items, 1):
        it["id"] = f"q{i:03d}"
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} questions to {args.out}. Review them before trusting the scores.")


if __name__ == "__main__":
    main()
