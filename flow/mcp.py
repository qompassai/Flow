"""Newline-delimited MCP stdio server. Stdout is exclusively JSON-RPC."""

from __future__ import annotations

import contextlib
import json
import sys

from flow import __version__

PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_VERSIONS = {PROTOCOL_VERSION, "2025-06-18", "2025-03-26"}
# One newline-delimited JSON-RPC frame; a task plus arguments never approaches this, so anything
# larger is a framing fault and the connection is closed instead of draining the rest.
MAX_FRAME_BYTES = 1024 * 1024
TOOLS = [
    {
        "name": "flow_run",
        "description": "Bounded local planner/coder/reviewer run with real verification",
        "inputSchema": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
            "additionalProperties": False,
        },
    },
    {
        "name": "flow_status",
        "description": "Local health, configured models/checks, editor capabilities",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "flow_check",
        "description": "Run operator-configured named checks; never arbitrary argv",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        },
    },
]


def error(request_id, code: int, message: str):
    assert isinstance(code, int)
    assert isinstance(message, str)
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _result(request_id, result: dict):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _tool_result(request_id, result: dict, *, is_error: bool):
    assert isinstance(result, dict)
    text = json.dumps(result, ensure_ascii=True, allow_nan=False)
    return _result(
        request_id,
        {
            "content": [{"type": "text", "text": text}],
            "structuredContent": result,
            "isError": is_error,
        },
    )


def _invalid_constant(value):
    raise ValueError(f"Invalid JSON constant {value}")


class MCPServer:
    def __init__(self, runtime):
        assert runtime is not None
        self.runtime = runtime
        self.initialized = False
        self.ready = False
        self.frames_read = 0

    def handle(self, request):
        """Dispatch one decoded JSON-RPC message; returns the reply or None for notifications."""
        if not isinstance(request, dict):
            return error(None, -32600, "Invalid Request: expected one JSON-RPC object, not a batch")
        request_id = request.get("id")
        if (
            request.get("jsonrpc") != "2.0"
            or not isinstance(request.get("method"), str)
            or isinstance(request_id, bool)
            or not isinstance(request_id, (str, int, type(None)))
        ):
            return error(None, -32600, "Invalid JSON-RPC request")
        notification = "id" not in request
        method = request["method"]
        params = request.get("params", {})
        if not isinstance(params, dict):
            return None if notification else error(request_id, -32602, "Params must be an object")
        if notification:
            if method == "notifications/initialized" and self.initialized:
                self.ready = True
            # Notifications never execute tools and never receive a response.
            return None
        if method == "ping":
            return _result(request_id, {})
        if method == "initialize":
            return self._initialize(request_id, params)
        if not self.ready:
            return error(request_id, -32002, "Initialize and send notifications/initialized first")
        if method == "tools/list":
            return _result(request_id, {"tools": TOOLS})
        if method == "tools/call":
            return self._tools_call(request_id, params)
        return error(request_id, -32601, f"Method not found: {method}")

    def _initialize(self, request_id, params: dict):
        assert isinstance(params, dict)
        if self.initialized:
            return error(request_id, -32600, "Already initialized")
        if params.get("protocolVersion") not in SUPPORTED_VERSIONS:
            return error(
                request_id,
                -32602,
                "Unsupported protocolVersion; supported: " + ", ".join(sorted(SUPPORTED_VERSIONS)),
            )
        self.initialized = True
        return _result(
            request_id,
            {
                "protocolVersion": params["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "flow", "version": __version__},
                "instructions": "Trusted workspace named checks only. Missing checks never verify.",
            },
        )

    def _tools_call(self, request_id, params: dict):
        from flow.runtime import validate_arguments

        assert self.ready
        assert isinstance(params, dict)
        if set(params) - {"name", "arguments", "_meta"}:
            return error(request_id, -32602, "Unknown tools/call parameters")
        name = params.get("name")
        spec = next((tool for tool in TOOLS if tool["name"] == name), None)
        if not spec:
            return error(request_id, -32602, "Unknown Flow tool")
        args = params.get("arguments", {})
        try:
            validate_arguments(args, spec["inputSchema"])
        except (ValueError, TypeError) as exc:
            return error(request_id, -32602, str(exc))
        try:
            # Third-party clients must never contaminate the transport.
            with contextlib.redirect_stdout(sys.stderr):
                if name == "flow_status":
                    result = self.runtime.status()
                elif name == "flow_check":
                    result = self.runtime.check(**args)
                else:
                    result = self.runtime.run(**args)
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
            return _tool_result(request_id, result, is_error=True)
        return _tool_result(request_id, result, is_error=result.get("status") != "ok")

    def serve(self, source=None, target=None):
        source = source or sys.stdin.buffer
        target = target or sys.stdout
        try:
            # This is the transport event loop: it runs until the client closes stdin or sends
            # an oversize frame, so each iteration is bounded by MAX_FRAME_BYTES instead.
            while True:
                frame = source.readline(MAX_FRAME_BYTES + 1)
                if not frame:
                    break
                self.frames_read += 1
                assert len(frame) <= MAX_FRAME_BYTES + 1
                if len(frame) > MAX_FRAME_BYTES:
                    # Oversize input is a terminal framing error, not an unbounded drain.
                    reply = error(None, -32700, "Frame exceeds 1 MiB; connection closing")
                    target.write(json.dumps(reply) + "\n")
                    target.flush()
                    break
                if not frame.strip():
                    continue
                reply = self._reply(frame)
                if reply is not None:
                    target.write(json.dumps(reply, ensure_ascii=True, allow_nan=False) + "\n")
                    target.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.runtime.close()

    def _reply(self, frame: bytes):
        assert 0 < len(frame) <= MAX_FRAME_BYTES
        try:
            request = json.loads(frame, parse_constant=_invalid_constant)
            return self.handle(request)
        except (ValueError, UnicodeError, RecursionError):
            return error(None, -32700, "Parse error")
        except Exception:
            return error(None, -32603, "Internal error")
