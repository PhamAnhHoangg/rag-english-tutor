"""Score retrieval on a test set, save the result, and compare it with the baseline.

  python scripts/eval_retrieval.py --name baseline --mode dense
  python scripts/eval_retrieval.py --ladder            # every mode, one table
  python scripts/eval_retrieval.py --ladder --questions eval/questions_compare.jsonl

Modes: dense, bm25, hybrid, dense+rerank, hybrid+rerank (default: retrieval.mode in the config).
--split / --no-split turns the "X vs Y" query splitting on or off.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works even without `pip install -e .`

from tutor.config import load_config  # noqa: E402
from tutor.evaluation import evaluate  # noqa: E402
from tutor.retrieval import MODES, make_retriever  # noqa: E402


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def check_gold(questions: list[dict], chunks_path: Path) -> None:
    """Warn about gold refs that are not in the index: those questions can never score."""
    if not chunks_path.exists():
        return
    refs = {json.loads(line)["ref"] for line in open(chunks_path, encoding="utf-8") if line.strip()}
    for q in questions:
        gold = q.get("gold_refs") or [r for g in q.get("gold_groups", []) for r in g]
        missing = [r for r in gold if r not in refs]
        if missing and len(missing) == len(gold):
            print(f"  warning: {q['id']} has no gold ref in the index ({missing[0]}...)")


def run(name: str, mode: str, split: bool | None, cfg: dict, questions: list[dict], k: int, out_dir: Path) -> dict:
    res = evaluate(make_retriever(cfg, mode, split), questions, k=k)
    res["mode"], res["split_comparisons"] = mode, split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


def show(name: str, res: dict, k: int, base: dict | None) -> None:
    def delta(key):
        return f" ({res[key] - base[key]:+.3f})" if base and key in base else ""

    print(f"{name}: n={res['n']}  recall@{k}={res[f'recall@{k}']}{delta(f'recall@{k}')}  "
          f"MRR={res['mrr']}{delta('mrr')}  {res['ms_per_query']} ms/query")
    for s, m in res["by_source"].items():
        print(f"  {s:<11} n={m['n']:<3} recall@{k}={m[f'recall@{k}']}  MRR={m['mrr']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=ROOT / "eval/questions.jsonl")
    p.add_argument("--name", default=None, help="result name (default: the mode)")
    p.add_argument("--mode", choices=MODES, default=None)
    p.add_argument("--split", dest="split", action="store_true", default=None, help="split 'X vs Y' questions")
    p.add_argument("--no-split", dest="split", action="store_false")
    p.add_argument("--ladder", action="store_true", help="run every mode and print a comparison table")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    chunks_path = Path(cfg["retrieval"].get("chunks_path", "data/chunks.jsonl"))
    if not chunks_path.is_absolute():
        chunks_path = ROOT / chunks_path
    cfg["retrieval"]["chunks_path"] = str(chunks_path)
    questions = load_questions(args.questions)
    check_gold(questions, chunks_path)
    suffix = "" if args.questions.stem == "questions" else "-" + args.questions.stem.removeprefix("questions_")
    out_dir = ROOT / "eval/results"
    base_file = out_dir / f"baseline{suffix}.json"
    base = json.loads(base_file.read_text(encoding="utf-8")) if base_file.exists() else None

    if not args.ladder:
        mode = args.mode or cfg["retrieval"].get("mode", "dense")
        name = (args.name or mode.replace("+", "-")) + suffix
        res = run(name, mode, args.split, cfg, questions, args.k, out_dir)
        show(name, res, args.k, None if name == f"baseline{suffix}" else base)
        print(f"Saved {out_dir / name}.json")
        return

    # Each step adds one idea, so the table shows what each one is worth.
    steps = [("dense", False), ("bm25", False), ("hybrid", False), ("dense+rerank", False),
             ("hybrid+rerank", False), ("hybrid+rerank", True)]
    rows = []
    for mode, split in steps:
        name = mode.replace("+", "-") + ("-split" if split else "") + suffix
        print(f"running {name}...", flush=True)
        res = run(name, mode, split, cfg, questions, args.k, out_dir)
        show(name, res, args.k, base)
        rows.append((name, res))

    sources = sorted({s for _, r in rows for s in r["by_source"]})
    print(f"\n| Experiment | Recall@{args.k} | MRR | " + " | ".join(f"{s} R@{args.k}" for s in sources) + " | ms/query |")
    print("| --- " * (4 + len(sources)) + "|")
    for name, r in rows:
        per = " | ".join(str(r["by_source"].get(s, {}).get(f"recall@{args.k}", "")) for s in sources)
        print(f"| {name} | {r[f'recall@{args.k}']} | {r['mrr']} | {per} | {r['ms_per_query']} |")


if __name__ == "__main__":
    main()
