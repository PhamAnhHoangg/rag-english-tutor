# RAG English Tutor

A study assistant that answers English grammar, vocabulary and usage questions from openly licensed sources, and cites the passage behind every answer. Everything runs locally and free: Ollama for the LLM, Qdrant for search, and bge-small-en-v1.5 via fastembed for embeddings (fast on CPU; bge-m3 is an option with a GPU).

Later this becomes the core of a full language study app.

## Quick start

```bash
conda create -n rag python=3.11 -y && conda activate rag   # fresh env: avoids clashes with base Anaconda packages
docker compose up -d                 # Qdrant + Ollama
docker exec -it ollama ollama pull qwen2.5:7b
pip install -e ".[dev,ml]"           # add ,serve for the API and ,eval for MLflow + Ragas later
python scripts/ingest.py --limit 2000 --chunks-only   # quick look, writes data/chunks.jsonl
python scripts/ingest.py --limit 2000                 # small index for the baseline
python scripts/ingest.py                              # everything (Wiktionary is ~2 GB)
python scripts/ask.py "What is the difference between affect and effect?"
python scripts/make_testset.py       # draft eval/questions.jsonl with the local LLM, then review it
python scripts/eval_retrieval.py --name baseline --mode dense
python scripts/eval_retrieval.py --ladder            # every retrieval mode, one table
python scripts/eval_retrieval.py --ladder --questions eval/questions_compare.jsonl
uvicorn tutor.api:app --reload       # POST /ask {"question": "..."}
pytest
```

Wikimedia asks API clients to identify themselves. Set `RAG_TUTOR_CONTACT` to a URL or email (for example your repo URL) before ingesting, and it is added to the User-Agent.

## Layout

| Path | What it holds |
| --- | --- |
| `src/tutor/sources.py` | Download and parse Wiktionary (kaikki.org), Tatoeba, Wikibooks |
| `src/tutor/embed.py` | Embedding backends: fastembed (default, CPU) or bge-m3 (GPU) |
| `src/tutor/index.py` | Embed chunks and upsert them into Qdrant |
| `src/tutor/chunking.py` | Chunkers per source type (dictionary, sentences, grammar prose) |
| `src/tutor/retrieval.py` | Dense, BM25, hybrid (RRF), cross-encoder rerank, "X vs Y" splitting; `make_retriever` builds the stack from the config |
| `src/tutor/llm.py` | LLM interface and the Ollama client |
| `src/tutor/pipeline.py` | Retrieve, build a cited prompt, generate |
| `src/tutor/api.py` | FastAPI app |
| `src/tutor/testset.py` | Drafts test questions from sampled chunks with the local LLM |
| `src/tutor/evaluation.py` | Retrieval metrics (recall@k, MRR) and the eval loop |
| `eval/` | Test questions and saved results (data only) |
| `configs/default.yaml` | Model names, chunk sizes, top-k |

## Results

Retrieval on 49 LLM-drafted, hand-reviewed questions (22,438 chunks, CPU). "Compare" is a separate set of 20 hand-written "X vs Y" questions (`eval/questions_compare.jsonl`), where recall@5 is the share of the two words found.

| Experiment | Recall@5 | MRR | Tatoeba R@5 | Compare R@5 |
| --- | --- | --- | --- | --- |
| Baseline: dense, fixed chunks | 0.776 | 0.613 | 0.30 | |
| BM25 only | 0.735 | 0.585 | 0.90 | 0.43 |
| BM25 + "X vs Y" splitting | 0.755 | 0.589 | 0.90 | 0.98 |
| Hybrid (dense + BM25, RRF) | | | | |
| Hybrid + rerank | | | | |
| Hybrid + rerank + splitting | | | | |

### What each step does

- **BM25** scores exact word overlap, weighted by rarity. Dense embeddings blur exact words, so a learner quoting a sentence ("How do I say 'I can't believe my eyes'?") missed it; BM25 finds it. Built in memory from `data/chunks.jsonl`, so no re-ingest.
- **Hybrid** merges dense and BM25 rankings with reciprocal rank fusion: score = sum of 1 / (60 + rank). Ranks, not raw scores, so no weights to tune.
- **Rerank**: a cross-encoder (ms-marco-MiniLM-L-6, 80 MB ONNX) reads query and passage together and re-scores the top 30. More accurate than embeddings, too slow for all chunks.
- **"X vs Y" splitting**: one embedding of "affect vs effect" lands near one word only. The question is split, each word is looked up in the dictionary and searched on its own, and the results are interleaved.
