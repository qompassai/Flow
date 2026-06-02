"""File read/write/list tools."""
from __future__ import annotations

import json
from pathlib import Path


def make_file_ops_tools(workspace: str = "."):
    ws = Path(workspace).resolve()

    def _safe_path(relative: str) -> Path:
        p = (ws / relative).resolve()
        if not str(p).startswith(str(ws)):
            raise ValueError(f"Path escape detected: {relative}")
        return p

    def file_read(path: str) -> str:
        try:
            p = _safe_path(path)
            if not p.exists():
                return json.dumps({"error": f"File not found: {path}"})
            content = p.read_text(errors="replace")
            return json.dumps({"path": path, "content": content, "lines": len(content.splitlines())})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def file_write(path: str, content: str) -> str:
        try:
            p = _safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return json.dumps({"success": True, "path": path, "bytes": len(content)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def file_list(directory: str = ".") -> str:
        try:
            p = _safe_path(directory)
            if not p.exists():
                return json.dumps({"error": f"Directory not found: {directory}"})
            files = []
            for item in sorted(p.rglob("*")):
                if item.is_file() and ".git" not in str(item):
                    files.append(str(item.relative_to(ws)))
            return json.dumps({"directory": directory, "files": files})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def file_delete(path: str) -> str:
        try:
            p = _safe_path(path)
            if p.exists():
                p.unlink()
            return json.dumps({"success": True, "path": path})
        except Exception as e:
            return json.dumps({"error": str(e)})

    return file_read, file_write, file_list, file_delete
