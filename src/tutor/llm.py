"""LLM interface. Only local, free backends: swap models by changing the config."""
from typing import Protocol

import httpx


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class OllamaLLM:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b", timeout: float = 120):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    def generate(self, prompt: str) -> str:
        r = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["response"]
