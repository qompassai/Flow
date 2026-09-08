"""Ollama OpenAI-compatible client: bounded responses, explicit remote opt-in, no proxies."""

from __future__ import annotations

import json

import httpx

from flow.config import FlowConfig, OllamaConfig, validate_config

# A chat completion is at most a few tool calls plus prose; streaming the body with a hard cap
# keeps a misbehaving or hostile server from exhausting memory before json.loads runs.
RESPONSE_BYTES_MAX = 2 * 1024 * 1024
# Completions are capped below the context window so the prompt itself always fits.
COMPLETION_TOKENS_MAX = 8192

assert RESPONSE_BYTES_MAX > 0
assert COMPLETION_TOKENS_MAX > 0


class OllamaClient:
    def __init__(self, cfg: OllamaConfig):
        assert isinstance(cfg, OllamaConfig)
        validate_config(FlowConfig(ollama=cfg))
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self._client = httpx.Client(timeout=cfg.timeout, trust_env=False, follow_redirects=False)

    def _json(self, method: str, path: str, **kwargs):
        assert method in {"GET", "POST"}
        assert path.startswith("/")
        with self._client.stream(method, f"{self.base_url}{path}", **kwargs) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > RESPONSE_BYTES_MAX:
                    raise ValueError(f"Ollama response exceeded {RESPONSE_BYTES_MAX} byte limit")
                chunks.append(chunk)
        assert size <= RESPONSE_BYTES_MAX
        return json.loads(b"".join(chunks))

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None, *, model: str | None = None
    ) -> dict:
        assert isinstance(messages, list)
        assert len(messages) > 0
        assert model is None or isinstance(model, str)
        payload = {
            "model": model or self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "stream": False,
            "max_tokens": min(COMPLETION_TOKENS_MAX, self.cfg.context_length // 2),
        }
        if tools:
            payload["tools"] = tools
        return self._json("POST", "/v1/chat/completions", json=payload)

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._json("GET", "/api/tags").get("models", [])]

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except (httpx.HTTPError, ValueError, KeyError):
            return False

    def close(self):
        self._client.close()
