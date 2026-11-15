"""Score retrieval on eval/questions.jsonl and save the result.

  python scripts/eval_retrieval.py --name baseline
"""
import argparse
import json
from pathlib import Path

from eval.run_retrieval import evaluate
from tutor.config import load_config
from tutor.retrieval import DenseRetriever


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    p.add_argument("--name", default="baseline")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    r = cfg["retrieval"]
    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    res = evaluate(DenseRetriever(r["qdrant_url"], r["collection"], cfg["embeddings"]["model"]), questions, k=args.k)

    out = Path("eval/results") / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{args.name}: n={res['n']}  recall@{args.k}={res[f'recall@{args.k}']}  MRR={res['mrr']}")
    for s, m in res["by_source"].items():
        print(f"  {s:<11} n={m['n']:<3} recall@{args.k}={m[f'recall@{args.k}']}  MRR={m['mrr']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
