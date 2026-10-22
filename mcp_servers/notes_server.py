"""Bundled notes MCP server (stdio). Run: python -m mcp_servers.notes_server"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("notes")


def _path() -> Path:
    return Path(os.environ.get("NOTES_FILE", "/data/notes.json"))


def _load() -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _save(notes: list[dict]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


@mcp.tool()
def add_note(text: str) -> str:
    """Save a note for later."""
    notes = _load()
    note_id = max((n["id"] for n in notes), default=0) + 1
    notes.append(
        {"id": note_id, "text": text, "created_at": datetime.now(timezone.utc).isoformat()}
    )
    _save(notes)
    return f"Saved note #{note_id}"


@mcp.tool()
def list_notes() -> str:
    """List all saved notes."""
    notes = _load()
    if not notes:
        return "No notes yet."
    return "\n".join(f"{n['id']}. {n['text']}" for n in notes)


@mcp.tool()
def search_notes(query: str) -> str:
    """Search saved notes (case-insensitive substring match)."""
    hits = [n for n in _load() if query.lower() in n["text"].lower()]
    if not hits:
        return f"No notes match '{query}'."
    return "\n".join(f"{n['id']}. {n['text']}" for n in hits)


if __name__ == "__main__":
    mcp.run()
