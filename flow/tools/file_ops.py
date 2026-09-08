"""Compatibility file factories backed by the same confined Workspace implementation."""

from __future__ import annotations

import json

from flow.workspace import Workspace


def make_file_ops_tools(workspace: str = ".", *, trusted=False):
    ws = Workspace(workspace, trusted=trusted)

    def wrap(fn):
        def call(**kwargs):
            try:
                return json.dumps(fn(**kwargs))
            except Exception as exc:
                return json.dumps({"status": "error", "error": str(exc)})

        return call

    def file_read(path):
        return wrap(ws.read)(path=path)

    def file_write(path, content):
        return wrap(ws.write)(path=path, content=content)

    def file_list(directory="."):
        return wrap(ws.list)(path=directory)

    def file_delete(path):
        return json.dumps({"status": "error", "error": "Deletion tools are disabled"})

    return file_read, file_write, file_list, file_delete
