"""Web search tool — uses DuckDuckGo (local, no API key) or SearXNG."""

from __future__ import annotations

import json

import httpx


def duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo via instant answer API (no JS, no key)."""
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    }
    try:
        resp = httpx.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        # Abstract result
        if data.get("AbstractText"):
            results.append(
                {
                    "title": data.get("Heading", query),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data["AbstractText"][:500],
                }
            )

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(
                    {
                        "title": topic.get("Text", "")[:80],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")[:300],
                    }
                )

        return results[:max_results]
    except Exception as e:
        return [{"error": str(e)}]


def searxng_search(query: str, searxng_url: str, max_results: int = 5) -> list[dict]:
    """Search via local SearXNG instance."""
    url = f"{searxng_url.rstrip('/')}/search"
    params = {"q": query, "format": "json", "categories": "general"}
    try:
        resp = httpx.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", [])[:max_results]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:300],
                }
            )
        return results
    except Exception as e:
        return [{"error": str(e)}]


def make_web_search_tool(backend: str = "duckduckgo", searxng_url: str = ""):
    """Factory returning the search function configured for the backend."""

    def run(query: str, max_results: int = 5) -> str:
        if backend == "searxng" and searxng_url:
            results = searxng_search(query, searxng_url, max_results)
        else:
            results = duckduckgo_search(query, max_results)
        return json.dumps(results, indent=2)

    return run


TOOL_SPEC = {
    "name": "web_search",
    "description": "Search the web for information. Returns titles, URLs, and snippets.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Max results (default 5)"},
        },
        "required": ["query"],
    },
}
