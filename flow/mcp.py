"""Newline-delimited MCP stdio server. Stdout is exclusively JSON-RPC."""

import contextlib
import json
import sys
from typing import BinaryIO, NoReturn, Protocol, TextIO, TypedDict, cast

from flow import __version__
from flow.json_types import is_object

PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_VERSIONS = {PROTOCOL_VERSION, "2025-06-18", "2025-03-26"}
# One newline-delimited JSON-RPC frame; a task plus arguments never approaches this, so anything
# larger is a framing fault and the connection is closed instead of draining the rest.
MAX_FRAME_BYTES = 1024 * 1024


class ToolSpec(TypedDict):
    name: str
    description: str
    inputSchema: dict[str, object]


class MCPRuntime(Protocol):
    def status(self) -> object: ...
    def check(self, name: str | None = None) -> object: ...
    def run(self, task: str) -> object: ...
    def close(self) -> None: ...


RequestID = str | int | None
TOOLS: list[ToolSpec] = [
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


def error(request_id: RequestID, code: int, message: str) -> dict[str, object]:
    assert isinstance(code, int)
    assert isinstance(message, str)
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _result(request_id: RequestID, result: dict[str, object]) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _tool_result(
    request_id: RequestID, result: dict[str, object], *, is_error: bool
) -> dict[str, object]:
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


def _invalid_constant(value: str) -> NoReturn:
    raise ValueError(f"Invalid JSON constant {value}")


class MCPServer:
    runtime: MCPRuntime
    initialized: bool
    ready: bool
    frames_read: int

    def __init__(self, runtime: MCPRuntime) -> None:
        assert runtime is not None
        self.runtime = runtime
        self.initialized = False
        self.ready = False
        self.frames_read = 0

    def handle(self, request: object) -> dict[str, object] | None:
        """Dispatch one decoded JSON-RPC message; returns the reply or None for notifications."""
        if not is_object(request):
            return error(None, -32600, "Invalid Request: expected one JSON-RPC object, not a batch")
        request_id = request.get("id")
        if (
            request.get("jsonrpc") != "2.0"
            or not isinstance(request.get("method"), str)
            or ("id" in request and request_id is None)
            or isinstance(request_id, bool)
            or not isinstance(request_id, (str, int, type(None)))
        ):
            return error(None, -32600, "Invalid JSON-RPC request")
        notification = "id" not in request
        method = request["method"]
        params = request.get("params", {})
        if not is_object(params):
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

    def _initialize(self, request_id: RequestID, params: dict[str, object]) -> dict[str, object]:
        assert isinstance(params, dict)
        if self.initialized:
            return error(request_id, -32600, "Already initialized")
        version = params.get("protocolVersion")
        if not isinstance(version, str) or not version:
            return error(request_id, -32602, "protocolVersion must be a nonempty string")
        info = params.get("clientInfo")
        if (
            not is_object(params.get("capabilities"))
            or not is_object(info)
            or not isinstance(info.get("name"), str)
            or not isinstance(info.get("version"), str)
        ):
            return error(request_id, -32602, "Initialize requires capabilities and clientInfo")
        negotiated = version if version in SUPPORTED_VERSIONS else PROTOCOL_VERSION
        self.initialized = True
        return _result(
            request_id,
            {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "flow", "version": __version__},
                "instructions": "Trusted workspace named checks only. Missing checks never verify.",
            },
        )

    def _tools_call(self, request_id: RequestID, params: dict[str, object]) -> dict[str, object]:
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
        if not is_object(args):
            return error(request_id, -32602, "Arguments must be an object")
        result: object
        try:
            # Third-party clients must never contaminate the transport.
            with contextlib.redirect_stdout(sys.stderr):
                if name == "flow_status":
                    result = self.runtime.status()
                elif name == "flow_check":
                    check_name = args.get("name")
                    if check_name is not None and not isinstance(check_name, str):
                        return error(request_id, -32602, "Check name must be a string")
                    result = self.runtime.check(check_name)
                else:
                    task = args.get("task")
                    if not isinstance(task, str):
                        return error(request_id, -32602, "Task must be a string")
                    result = self.runtime.run(task)
        except Exception as exc:
            return _tool_result(request_id, {"status": "error", "error": str(exc)}, is_error=True)
        if not is_object(result):
            return error(request_id, -32603, "Runtime returned a non-object result")
        try:
            return _tool_result(request_id, result, is_error=result.get("status") != "ok")
        except (TypeError, ValueError, RecursionError):
            return error(request_id, -32603, "Runtime returned non-JSON data")

    def serve(self, source: BinaryIO | None = None, target: TextIO | None = None) -> None:
        source = cast(BinaryIO, sys.stdin.buffer) if source is None else source
        target = sys.stdout if target is None else target
        if target is None:
            raise ValueError("MCP requires input and output streams")
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
                    _ = target.write(json.dumps(reply) + "\n")
                    target.flush()
                    break
                if not frame.strip():
                    continue
                reply = self._reply(frame)
                if reply is not None:
                    _ = target.write(json.dumps(reply, ensure_ascii=True, allow_nan=False) + "\n")
                    target.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.runtime.close()

    def _reply(self, frame: bytes) -> dict[str, object] | None:
        assert 0 < len(frame) <= MAX_FRAME_BYTES
        try:
            request = cast(object, json.loads(frame, parse_constant=_invalid_constant))
        except (ValueError, UnicodeError, RecursionError):
            return error(None, -32700, "Parse error")
        try:
            return self.handle(request)
        except Exception:
            request_id = request.get("id") if is_object(request) else None
            if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
                request_id = None
            return error(request_id, -32603, "Internal error")
