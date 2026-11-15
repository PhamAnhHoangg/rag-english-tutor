# RAG English Tutor

A study assistant that answers English grammar, vocabulary and usage questions from openly licensed sources, and cites the passage behind every answer. Everything runs locally and free: Ollama for the LLM, Qdrant for search, bge-m3 for embeddings.

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
uvicorn tutor.api:app --reload       # POST /ask {"question": "..."}
pytest
```

Wikimedia asks API clients to identify themselves. Set `RAG_TUTOR_CONTACT` to a URL or email (for example your repo URL) before ingesting, and it is added to the User-Agent.

## Layout

| Path | What it holds |
| --- | --- |
| `src/tutor/sources.py` | Download and parse Wiktionary (kaikki.org), Tatoeba, Wikibooks |
| `src/tutor/index.py` | Embed with bge-m3 and upsert into Qdrant |
| `src/tutor/chunking.py` | Chunkers per source type (dictionary, sentences, grammar prose) |
| `src/tutor/retrieval.py` | Retriever interface; dense, hybrid and reranked versions go here |
| `src/tutor/llm.py` | LLM interface and the Ollama client |
| `src/tutor/pipeline.py` | Retrieve, build a cited prompt, generate |
| `src/tutor/api.py` | FastAPI app |
| `eval/` | Test questions and retrieval metrics (recall@k, MRR) |
| `configs/default.yaml` | Model names, chunk sizes, top-k |

## Results

Filled in as the experiment ladder runs.

| Experiment | Recall@5 | MRR | Faithfulness |
| --- | --- | --- | --- |
| Baseline: dense, fixed chunks | | | |
