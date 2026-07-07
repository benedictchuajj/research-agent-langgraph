import asyncio
import json
import logging
import sys
from typing import Annotated, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.config import (
    ARXIV_MAX,
    ITER_MAX,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SUMMARY_MAX,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    user_interest: Optional[str]
    kept_ids: list[str]
    arxiv_calls: int
    summaries_written: int
    iterations: int


# ---------------------------------------------------------------------------
# System prompt — the agentic policy lives here
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a research scout. Goal: deliver a short markdown digest of arxiv "
    "papers relevant to the user's query and (if stated) interest, saved to "
    "papers/digest.md via the `finish` tool.\n\n"
    "Policy:\n"
    "1. Call `search_papers` first. Read the titles and abstracts of ALL "
    "candidates before deciding what to keep.\n"
    "2. For borderline papers, call `get_paper_details` before committing — "
    "summarizing is expensive (a full fetch + summary + embedding + file write).\n"
    "3. Summarize only the papers you judge relevant (target 3-8). Skip the rest "
    "with `skip_paper` and a short reason.\n"
    "4. If the first search returns <3 on-topic hits, REWRITE the query "
    "(narrower topic, synonyms, different subtopic) and search again — at most "
    "2 rewrites.\n"
    "5. If a user interest is stated, call `compare_to_interest` on your kept "
    "set before finishing; mention the most relevant papers in the digest.\n"
    "6. Call `finish` exactly once with the digest markdown (<400 words) when "
    "done. The digest is saved to papers/digest.md for you.\n\n"
    f"Budget: at most {ARXIV_MAX} arxiv searches, {SUMMARY_MAX} summaries, "
    f"{ITER_MAX} tool rounds overall. Hard caps — do not exceed.\n"
)


# ---------------------------------------------------------------------------
# MCP toolkit — one long-lived stdio session for the whole run
# ---------------------------------------------------------------------------


MCP_COMMAND = [sys.executable, "-m", "app.mcp.server"]


class MCPToolkit:
    """Holds one MCP stdio subprocess open for the whole run."""

    def __init__(self, command: Optional[list[str]] = None) -> None:
        self._command = command or MCP_COMMAND
        self._session: Optional[ClientSession] = None
        self._ctx = None
        self.spawn_count = 0

    async def __aenter__(self) -> "MCPToolkit":
        params = StdioServerParameters(
            command=self._command[0], args=self._command[1:]
        )
        self._ctx = stdio_client(params)
        read, write = await self._ctx.__aenter__()
        self.spawn_count += 1
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._session is not None:
            await self._session.__aexit__(*exc_info)
            self._session = None
        if self._ctx is not None:
            await self._ctx.__aexit__(*exc_info)
            self._ctx = None

    async def call(self, name: str, arguments: dict) -> str:
        assert self._session is not None, "MCPToolkit used outside `async with`"
        result = await self._session.call_tool(name, arguments)
        texts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(texts)


# ---------------------------------------------------------------------------
# Sentinel — `finish` ends the loop
# ---------------------------------------------------------------------------


class FinishSentinel(Exception):
    def __init__(self, digest: str) -> None:
        self.digest = digest
        super().__init__("finish tool called")


# ---------------------------------------------------------------------------
# Tool schemas (pydantic) + builder that wraps the MCP toolkit
# ---------------------------------------------------------------------------


class SearchPapersArgs(BaseModel):
    query: str = Field(..., description="arxiv search query")
    subtopics: list[str] = Field(
        default_factory=list,
        description="Subtopics to filter by (e.g. ['multi-agent', 'reasoning'])",
    )
    max_results: int = Field(20, description="max arxiv results to fetch")


class GetPaperDetailsArgs(BaseModel):
    arxiv_id: str = Field(..., description="arxiv paper ID (e.g. '2401.12345')")


class SummarizePaperArgs(BaseModel):
    arxiv_id: str = Field(..., description="arxiv paper ID to summarize and store")


class SkipPaperArgs(BaseModel):
    arxiv_id: str = Field(..., description="arxiv paper ID being rejected")
    reason: str = Field(..., description="short reason for rejecting this paper")


class CompareToInterestArgs(BaseModel):
    interest: str = Field(..., description="free-text research interest")
    arxiv_ids: list[str] = Field(..., description="arxiv IDs to compare against")
    top_k: int = Field(5, description="number of top matches to return")


class ListPapersArgs(BaseModel):
    subtopics: list[str] = Field(
        default_factory=list, description="filter by subtopics (empty = all)"
    )


class FinishArgs(BaseModel):
    digest_markdown: str = Field(
        ..., description="final digest content (markdown), <400 words"
    )


def build_langchain_tools(
    toolkit: MCPToolkit,
    state: AgentState,
) -> list[StructuredTool]:
    """Wrap each MCP tool as a langchain StructuredTool bound to `state` for
    budget accounting. The `finish` tool raises FinishSentinel to terminate.
    """

    async def search_papers(
        query: str,
        subtopics: Optional[list[str]] = None,
        max_results: int = 20,
    ) -> str:
        state["arxiv_calls"] = state.get("arxiv_calls", 0) + 1
        return await toolkit.call(
            "search_papers",
            {
                "query": query,
                "subtopics": subtopics or [],
                "max_results": max_results,
            },
        )

    async def get_paper_details(arxiv_id: str) -> str:
        return await toolkit.call("get_paper_details", {"arxiv_id": arxiv_id})

    async def summarize_paper(arxiv_id: str) -> str:
        state["summaries_written"] = state.get("summaries_written", 0) + 1
        if state.get("summaries_written", 0) > SUMMARY_MAX:
            return (
                f"Error: SUMMARY_MAX ({SUMMARY_MAX}) reached; "
                "do not summarize further, call `finish` instead."
            )
        kept = list(state.get("kept_ids", []))
        if arxiv_id not in kept:
            kept.append(arxiv_id)
            state["kept_ids"] = kept
        return await toolkit.call("summarize_paper", {"arxiv_id": arxiv_id})

    async def skip_paper(arxiv_id: str, reason: str) -> str:
        return await toolkit.call(
            "skip_paper", {"arxiv_id": arxiv_id, "reason": reason}
        )

    async def compare_to_interest(
        interest: str, arxiv_ids: list[str], top_k: int = 5
    ) -> str:
        return await toolkit.call(
            "compare_to_interest",
            {"interest": interest, "arxiv_ids": arxiv_ids, "top_k": top_k},
        )

    async def list_papers(subtopics: Optional[list[str]] = None) -> str:
        return await toolkit.call(
            "list_papers", {"subtopics": subtopics or []}
        )

    async def finish(digest_markdown: str) -> str:
        # Ask the MCP server to persist the digest first, then terminate.
        await toolkit.call("finish", {"digest_markdown": digest_markdown})
        raise FinishSentinel(digest_markdown)

    return [
        StructuredTool.from_function(
            search_papers, name="search_papers", description=(
                "Search arxiv for papers matching a query and filter by "
                "subtopics. Returns a filtered candidate list."
            ),
            args_schema=SearchPapersArgs,
            coroutine=search_papers,
        ),
        StructuredTool.from_function(
            get_paper_details, name="get_paper_details", description=(
                "Fetch metadata + abstract for one arxiv paper. Does NOT write "
                "a file or summarize. Use before committing to summarize."
            ),
            args_schema=GetPaperDetailsArgs,
            coroutine=get_paper_details,
        ),
        StructuredTool.from_function(
            summarize_paper, name="summarize_paper", description=(
                "Summarize an arxiv paper by ID, write papers/<id>.md, embed. "
                "Commits the paper to the kept set."
            ),
            args_schema=SummarizePaperArgs,
            coroutine=summarize_paper,
        ),
        StructuredTool.from_function(
            skip_paper, name="skip_paper", description=(
                "Record a rejection (arxiv_id + reason) so future runs do not "
                "re-fetch this paper."
            ),
            args_schema=SkipPaperArgs,
            coroutine=skip_paper,
        ),
        StructuredTool.from_function(
            compare_to_interest, name="compare_to_interest", description=(
                "Rank the given arxiv_ids by cosine similarity to a free-text "
                "research-interest statement. Returns ranked similarities."
            ),
            args_schema=CompareToInterestArgs,
            coroutine=compare_to_interest,
        ),
        StructuredTool.from_function(
            list_papers, name="list_papers", description=(
                "List locally stored paper summaries, optionally filtered by "
                "subtopics. Use to dedupe against already-summarized papers."
            ),
            args_schema=ListPapersArgs,
            coroutine=list_papers,
        ),
        StructuredTool.from_function(
            finish, name="finish", description=(
                "Write the final <400-word markdown digest to papers/digest.md "
                "and terminate the run. Call exactly once when done."
            ),
            args_schema=FinishArgs,
            coroutine=finish,
        ),
    ]


# ---------------------------------------------------------------------------
# LLM factory (overridable for tests)
# ---------------------------------------------------------------------------


_agent_llm: Optional[BaseChatModel] = None


def _get_agent_llm() -> BaseChatModel:
    global _agent_llm
    if _agent_llm is not None:
        return _agent_llm
    logger.info(
        "Agent using Ollama: model=%s base_url=%s", OLLAMA_MODEL, OLLAMA_BASE_URL
    )
    _agent_llm = ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0
    )
    return _agent_llm


# ---------------------------------------------------------------------------
# ReAct node
# ---------------------------------------------------------------------------


async def _invoke_tool(
    tool_name: str,
    tool_args: dict,
    tools: list[StructuredTool],
) -> str:
    for t in tools:
        if t.name == tool_name:
            try:
                out = await t.ainvoke(tool_args)
            except FinishSentinel:
                raise
            except Exception as e:  # pragma: no cover - defensive
                logger.exception("Tool %s raised", tool_name)
                out = f"Error: {e}"
            return str(out)
    return f"Error: unknown tool {tool_name!r}"


async def react_node(
    state: AgentState,
    *,
    llm: BaseChatModel,
    toolkit: MCPToolkit,
) -> dict:
    """One ReAct step: ask the LLM what to do, run its tool calls, append
    ToolMessages. Terminates by raising FinishSentinel from the `finish` tool,
    which we catch here to emit a final message."""
    state["iterations"] = state.get("iterations", 0) + 1

    if not state.get("messages"):
        seed = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["user_query"]),
        ]
    else:
        seed = list(state["messages"])

    tools = build_langchain_tools(toolkit, state)
    bound = llm.bind_tools(tools)

    response = await bound.ainvoke(seed)

    tool_calls = getattr(response, "tool_calls", None) or []
    finished_digest: Optional[str] = None

    if not tool_calls:
        # Model produced a free-text answer with no tool call. Treat it as
        # an implicit finish using its content as the digest.
        content = _extract_text(response)
        logger.info("react_node: no tool_calls; using reply as digest")
        await toolkit.call("finish", {"digest_markdown": content})
        return {
            "messages": seed + [response],
            "user_query": state["user_query"],
            "user_interest": state.get("user_interest"),
            "kept_ids": state.get("kept_ids", []),
            "arxiv_calls": state.get("arxiv_calls", 0),
            "summaries_written": state.get("summaries_written", 0),
            "iterations": state.get("iterations", 0),
        }

    new_messages = [response]
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}
        tc_id = tc.get("id", "")
        try:
            out = await _invoke_tool(name, args, tools)
        except FinishSentinel as fin:
            # `finish` writes digest via the MCP server. Record and stop
            # processing further tool calls this round.
            finished_digest = fin.digest
            out = json.dumps({"finished": True})
        new_messages.append(
            ToolMessage(content=out, tool_call_id=tc_id, name=name)
        )
        if finished_digest is not None:
            break

    return {
        "messages": seed + new_messages,
        "user_query": state["user_query"],
        "user_interest": state.get("user_interest"),
        "kept_ids": state.get("kept_ids", []),
        "arxiv_calls": state.get("arxiv_calls", 0),
        "summaries_written": state.get("summaries_written", 0),
        "iterations": state.get("iterations", 0),
    }


def _extract_text(msg: AIMessage) -> str:
    raw = getattr(msg, "content", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw
        )
    return str(raw)


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


def should_continue(state: AgentState) -> str:
    if state.get("iterations", 0) >= ITER_MAX:
        logger.warning("ITER_MAX (%d) reached", ITER_MAX)
        return END
    if state.get("arxiv_calls", 0) >= ARXIV_MAX and not _recent_finish(state):
        # Allow the run to wind down (summarize/skip/compare/finish) but cap
        # further searches. The LLM-side message handles the negative budget;
        # this just prevents infinite loops if the model ignores it.
        if state.get("arxiv_calls", 0) > ARXIV_MAX + 2:
            logger.warning("ARXIV_MAX (%d) exceeded", ARXIV_MAX)
            return END
    if _recent_finish(state):
        return END
    return "react"


def _recent_finish(state: AgentState) -> bool:
    """True if the most recent ToolMessage was a `finish` call."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, ToolMessage):
            return msg.name == "finish"
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            # implicit finish (free-text answer treated as digest)
            return True
    return False


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph(
    *,
    llm: Optional[BaseChatModel] = None,
    toolkit: Optional[MCPToolkit] = None,
):
    """Compile the react graph. `llm` and `toolkit` are injectable for tests.

    The toolkit's lifecycle must be managed by the caller via `async with`.
    The react node re-uses the toolkit passed in here.
    """

    chosen_llm = llm or _get_agent_llm()
    chosen_toolkit = toolkit
    if chosen_toolkit is None:
        raise RuntimeError(
            "build_graph requires a toolkit (use run_agent / inject for tests)."
        )

    async def _node(state: AgentState) -> dict:
        return await react_node(state, llm=chosen_llm, toolkit=chosen_toolkit)

    graph = StateGraph(AgentState)
    graph.add_node("react", _node)
    graph.add_edge(START, "react")
    graph.add_conditional_edges(
        "react", should_continue, {"react": "react", END: END}
    )
    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point — manages the MCP session lifecycle
# ---------------------------------------------------------------------------


async def run_agent(query: str) -> dict:
    async with MCPToolkit() as toolkit:
        app = build_graph(llm=_get_agent_llm(), toolkit=toolkit)
        initial_state: AgentState = {
            "messages": [],
            "user_query": query,
            "user_interest": None,
            "kept_ids": [],
            "arxiv_calls": 0,
            "summaries_written": 0,
            "iterations": 0,
        }
        result = await app.ainvoke(initial_state)
        logger.info(
            "Agent completed. summaries=%d iterations=%d spawn_count=%d",
            result.get("summaries_written", 0),
            result.get("iterations", 0),
            toolkit.spawn_count,
        )
        return result


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m app.agent <research query>", file=sys.stderr)
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    asyncio.run(run_agent(query))


if __name__ == "__main__":
    main()