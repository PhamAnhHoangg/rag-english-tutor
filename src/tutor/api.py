from fastapi import FastAPI
from pydantic import BaseModel

from tutor.config import load_config
from tutor.llm import OllamaLLM
from tutor.pipeline import answer
from tutor.embed import make_embedder
from tutor.retrieval import DenseRetriever

cfg = load_config()
llm = OllamaLLM(**cfg["llm"])
retriever = DenseRetriever(cfg["retrieval"]["qdrant_url"], cfg["retrieval"]["collection"], make_embedder(cfg["embeddings"]))
app = FastAPI(title="RAG English Tutor")


class Ask(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Ask):
    a = answer(body.question, retriever, llm, k=cfg["retrieval"]["top_k"])
    return {"answer": a.text, "sources": [{"source": c.source, "ref": c.ref, "text": c.text} for c in a.sources]}
