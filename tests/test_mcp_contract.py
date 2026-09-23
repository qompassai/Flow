import json

from flow.mcp import PROTOCOL_VERSION, MCPServer


class RuntimeStub:
    def status(self):
        return {"status": "ok"}

    def close(self):
        pass


def send(params):
    return MCPServer(RuntimeStub())._reply(json.dumps(params).encode())


def test_null_id_is_invalid():
    assert send({"jsonrpc": "2.0", "id": None, "method": "ping"})["error"]["code"] == -32600


def test_unhashable_version_is_invalid_params():
    reply = send(
        {"jsonrpc": "2.0", "id": 9, "method": "initialize", "params": {"protocolVersion": []}}
    )
    assert reply["error"]["code"] == -32602
    assert reply["id"] == 9


def test_unknown_version_negotiates():
    reply = send(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "initialize",
            "params": {
                "protocolVersion": "2099-01-01",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        }
    )
    assert reply["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_dispatch_value_error_is_not_parse_error():
    class Broken(MCPServer):
        def handle(self, request):
            raise ValueError("handler failure")

    reply = Broken(RuntimeStub())._reply(b'{"jsonrpc":"2.0","id":3,"method":"ping"}')
    assert reply["error"]["code"] == -32603
    assert reply["id"] == 3


def test_validator_rejects_nonfinite_and_malformed_schema():
    import pytest

    from flow.runtime import validate_arguments

    with pytest.raises(ValueError):
        validate_arguments(float("inf"), {"type": "number"})
    with pytest.raises(ValueError):
        validate_arguments({}, {"type": []})
    with pytest.raises(ValueError):
        validate_arguments({"x": 1}, {"type": "object", "properties": {"x": []}})


def test_runtime_nonjson_result_preserves_request_id():
    class Invalid(RuntimeStub):
        def status(self):
            return {"status": "ok", "bad": object()}

    server = MCPServer(Invalid())
    server.ready = True
    reply = server._reply(
        b'{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"flow_status"}}'
    )
    assert reply["id"] == 4 and reply["error"]["code"] == -32603
