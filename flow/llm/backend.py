"""Ollama LLM backend with OpenAI-compatible API."""
from __future__ import annotations

import json
from typing import Any, Generator
import httpx

from flow.config import OllamaConfig


class OllamaClient:
    """Thin sync client for Ollama's OpenAI-compatible API."""

    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self._client = httpx.Client(timeout=cfg.timeout)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> dict | Generator[str, None, None]:
        """Send chat completion request to Ollama."""
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "stream": stream,
            "options": {"num_ctx": self.cfg.context_length},
        }
        if tools:
            payload["tools"] = tools

        resp = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()

        if stream:
            return self._stream_response(resp)

        data = resp.json()
        return data

    def _stream_response(self, resp: httpx.Response) -> Generator[str, None, None]:
        for line in resp.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        yield delta["content"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def list_models(self) -> list[str]:
        """List available local models."""
        resp = self._client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def is_available(self) -> bool:
        try:
            self._client.get(f"{self.base_url}/api/tags", timeout=3)
            return True
        except Exception:
            return False
