# RAG English Tutor

A study assistant that answers English grammar, vocabulary and usage questions from openly licensed sources, and cites the passage behind every answer. Everything runs locally and free: Ollama for the LLM, Qdrant for search, bge-m3 for embeddings.

Later this becomes the core of a full language study app.

## Quick start

```bash
docker compose up -d                 # Qdrant + Ollama
docker exec -it ollama ollama pull qwen2.5:7b
pip install -e ".[dev,ml,serve]"
python scripts/ingest.py             # download, chunk, embed, index (TODO)
uvicorn tutor.api:app --reload       # POST /ask {"question": "..."}
pytest
```

## Layout

| Path | What it holds |
| --- | --- |
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
