"""Conversation context manager."""
from __future__ import annotations
from collections import deque


class ConversationContext:
    """Sliding window conversation context."""

    def __init__(self, max_messages: int = 40):
        self.max_messages = max_messages
        self._messages: deque[dict] = deque(maxlen=max_messages)

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
