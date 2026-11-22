"""Observability layer (Module 1, defence layer 5): JSONL log of every agent step."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from agents import RunHooks

from . import config


class LogHooks(RunHooks):
    def __init__(self, verbose: bool = False, log_file: Path | None = None):
        self.verbose = verbose
        config.LOG_DIR.mkdir(exist_ok=True)
        self.path = log_file or config.LOG_DIR / "runs.jsonl"
        self.tool_calls: list[dict] = []
        self.handoffs: list[str] = []

    def _w(self, event: str, **data) -> None:
        rec = {"t": round(time.time(), 3), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if self.verbose:
            print(f"  [{event}] " + " ".join(f"{k}={str(v)[:90]}" for k, v in data.items()), file=sys.stderr)

    async def on_agent_start(self, context, agent):
        self._w("agent_start", agent=agent.name)

    async def on_handoff(self, context, from_agent, to_agent):
        self.handoffs.append(to_agent.name)
        self._w("handoff", frm=from_agent.name, to=to_agent.name)

    async def on_tool_start(self, context, agent, tool):
        self._w("tool_start", agent=agent.name, tool=tool.name)

    async def on_tool_end(self, context, agent, tool, result):
        self.tool_calls.append({"tool": tool.name, "result": str(result)})
        self._w("tool_end", agent=agent.name, tool=tool.name, result=str(result)[:200])
