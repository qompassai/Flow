from __future__ import annotations

import copy
import json

import pytest

from flow.config import AgentConfig, FlowConfig


def message(content="", calls=None):
    value = {"role": "assistant", "content": content}
    if calls:
        value["tool_calls"] = calls
    return {"choices": [{"message": value}]}


def tool(name, arguments, ident="call-1"):
    return {
        "id": ident,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def verdict(approved=True):
    return message(json.dumps({"approved": approved, "summary": "Reviewed", "issues": []}))


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def chat(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("Unexpected extra backend request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response(**kwargs) if callable(response) else response

    def close(self):
        self.closed = True


@pytest.fixture
def cfg(tmp_path):
    return FlowConfig(
        workspace_dir=str(tmp_path), trusted=True, agent=AgentConfig(max_iterations=4, max_cycles=1)
    )
