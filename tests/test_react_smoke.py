"""Smoke test for the ReAct loop.

The LLM is stubbed with a scripted sequence of tool_calls so no Ollama is
required. The MCP server is replaced by a FakeToolkit that delegates disk
writes (digest + rejections) to the real store module so we can assert files
land on disk. No MCP subprocess is spawned.
"""

import json
from typing import Optional

import pytest

from langchain_core.messages import AIMessage

import app.mcp.store as store_module
from app.agent import build_graph, AgentState, MCPToolkit


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class FakeToolkit:
    """Replaces MCPToolkit: no subprocess, no network. Spawns counted at 0."""

    def __init__(self) -> None:
        self.spawn_count = 0

    async def __aenter__(self) -> "FakeToolkit":
        self.spawn_count += 1
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None

    async def call(self, name: str, arguments: dict) -> str:
        if name == "search_papers":
            return json.dumps([
                {"arxiv_id": "2401.00001", "title": "Paper A", "abstract": "..."},
                {"arxiv_id": "2401.00002", "title": "Paper B", "abstract": "..."},
            ])
        if name == "get_paper_details":
            return json.dumps({
                "arxiv_id": arguments["arxiv_id"],
                "title": f"Piece {arguments['arxiv_id']}",
                "abstract": "lorem ipsum",
                "already_summarized": False,
            })
        if name == "summarize_paper":
            return json.dumps({
                "arxiv_id": arguments["arxiv_id"],
                "path": f"papers/{arguments['arxiv_id']}.md",
                "summary": "ok",
            })
        if name == "skip_paper":
            store_module.write_rejection(
                arguments["arxiv_id"], arguments["reason"]
            )
            return json.dumps({
                "arxiv_id": arguments["arxiv_id"], "skipped": True,
            })
        if name == "compare_to_interest":
            results = []
            for aid in arguments["arxiv_ids"][:1]:
                results.append({
                    "arxiv_id": aid, "similarity": 0.9, "title": "x",
                })
            return json.dumps(results)
        if name == "list_papers":
            return json.dumps([])
        if name == "finish":
            store_module.write_digest(arguments["digest_markdown"])
            return json.dumps({"finished": True})
        return json.dumps({"error": f"unknown tool {name}"})


class FakeLLM:
    """Replays a scripted queue of AIMessages. Ignores `bind_tools`."""

    def __init__(self, scripts: list[AIMessage]) -> None:
        self._scripts = list(scripts)

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, **kwargs):
        if not self._scripts:
            pytest.fail("FakeLLM script exhausted — run did not terminate")
        return self._scripts.pop(0)


def _tc(name: str, args: dict, idx: int) -> dict:
    return {"name": name, "args": args, "id": f"call_{idx}"}


def _ai_with_calls(calls: list[dict]) -> AIMessage:
    msg = AIMessage(content="")
    msg.tool_calls = calls
    return msg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point the store module at a tmp papers dir so the test never touches
    the real ./papers volume."""
    index_dir = tmp_path / ".index"
    index_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store_module, "PAPERS_DIR", tmp_path)
    monkeypatch.setattr(store_module, "_REJECTIONS_FILE", index_dir / "rejections.json")
    yield


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_loop_writes_digest_and_rejections():
    script = [
        _ai_with_calls([_tc("search_papers", {"query": "multi-agent reasoning"}, 0)]),
        _ai_with_calls([
            _tc("get_paper_details", {"arxiv_id": "2401.00002"}, 1),
        ]),
        _ai_with_calls([
            _tc("summarize_paper", {"arxiv_id": "2401.00001"}, 2),
        ]),
        _ai_with_calls([
            _tc("skip_paper", {
                "arxiv_id": "2401.00002", "reason": "off-topic"
            }, 3),
        ]),
        _ai_with_calls([
            _tc("compare_to_interest", {
                "interest": "efficient inference",
                "arxiv_ids": ["2401.00001"],
            }, 4),
        ]),
        _ai_with_calls([
            _tc("finish", {"digest_markdown": "# Digest\n\nPaper: A"}, 5),
        ]),
    ]
    llm = FakeLLM(script)
    toolkit = FakeToolkit()

    app = build_graph(llm=llm, toolkit=toolkit)

    initial_state: AgentState = {
        "messages": [],
        "user_query": "multi-agent reasoning, interest: efficient inference",
        "user_interest": None,
        "kept_ids": [],
        "arxiv_calls": 0,
        "summaries_written": 0,
        "iterations": 0,
    }
    result = await app.ainvoke(initial_state)

    # Loops once per scripted AIMessage — six react steps total.
    assert result["iterations"] == 6
    # Only one summarize call counted.
    assert result["summaries_written"] == 1
    # Kept set has the summarized paper.
    assert "2401.00001" in result["kept_ids"]
    # No MCP subprocess is spawned by the Fake (one logical session per run,
    # not one per node). The fact that 6 react iterations ran against the
    # same toolkit instance exercises the single-session architecture.
    assert toolkit.spawn_count == 0

    # Disk side-effects (delegated to the real store).
    digest = store_module.PAPERS_DIR / "digest.md"
    assert digest.exists()
    assert "Digest" in digest.read_text()

    rejections_path = store_module._REJECTIONS_FILE
    assert rejections_path.exists()
    rej = json.loads(rejections_path.read_text())
    assert "2401.00002" in rej
    assert rej["2401.00002"] == "off-topic"


@pytest.mark.asyncio
async def test_react_loop_implicit_finish_on_free_text():
    """If the LLM replies with no tool_calls, the message content becomes the
    digest and the run terminates."""
    free_msg = AIMessage(content="# Plain digest via text")
    llm = FakeLLM([free_msg])
    toolkit = FakeToolkit()

    app = build_graph(llm=llm, toolkit=toolkit)

    initial_state: AgentState = {
        "messages": [],
        "user_query": "anything",
        "user_interest": None,
        "kept_ids": [],
        "arxiv_calls": 0,
        "summaries_written": 0,
        "iterations": 0,
    }
    result = await app.ainvoke(initial_state)

    assert result["iterations"] == 1
    digest = store_module.PAPERS_DIR / "digest.md"
    assert digest.exists()
    assert "Plain digest" in digest.read_text()