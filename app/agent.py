import asyncio
import json
import logging
import sys
from typing import Annotated, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing_extensions import TypedDict

from app.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    query: str
    subtopics: list[str]
    candidates: list[dict]
    fetched: list[str]
    summaries: list[dict]
    paths: list[str]
    user_interest: Optional[str]
    similarities: list[dict]
    error: Optional[str]
    retry_count: int
    messages: Annotated[list, add_messages]


MCP_COMMAND = [sys.executable, "-m", "app.mcp.server"]

_agent_llm: Optional[BaseChatModel] = None


def _get_agent_llm() -> BaseChatModel:
    global _agent_llm
    if _agent_llm is not None:
        return _agent_llm

    if LLM_PROVIDER == "ollama":
        logger.info("Agent using Ollama: model=%s base_url=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
        _agent_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set (LLM_PROVIDER=openai)")
        logger.info("Agent using OpenAI: model=%s", OPENAI_MODEL)
        _agent_llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    return _agent_llm


async def _call_mcp_tool(session: ClientSession, tool_name: str, arguments: dict) -> Any:
    result = await session.call_tool(tool_name, arguments)
    texts = []
    for content in result.content:
        if hasattr(content, "text"):
            texts.append(content.text)
    return "\n".join(texts)


async def plan_node(state: AgentState) -> dict:
    logger.info("plan_node: parsing query")
    llm = _get_agent_llm()

    sys_msg = SystemMessage(content=(
        "You are a research assistant. Given a user's research topic, extract:\n"
        "1. A concise arxiv search query\n"
        "2. Relevant subtopics from this list: agents, multi-agent, planning, reasoning, tool-use, rag, summarization, code-generation, safety, evaluation\n"
        "3. An optional user interest statement (if the user mentions what they're interested in)\n\n"
        "Respond in JSON: {\"query\": \"...\", \"subtopics\": [...], \"user_interest\": \"...\" or null}"
    ))
    user_msg = HumanMessage(content=state["query"])

    response = await llm.ainvoke([sys_msg, user_msg])

    raw = response.content
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        text = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw
        )
    else:
        text = str(raw)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"query": state["query"], "subtopics": [], "user_interest": None}

    return {
        "query": parsed.get("query", state["query"]),
        "subtopics": parsed.get("subtopics", []),
        "user_interest": parsed.get("user_interest"),
        "messages": [sys_msg, user_msg, response],
    }


async def search_node(state: AgentState) -> dict:
    logger.info("search_node: searching arxiv for query=%s subtopics=%s", state["query"], state["subtopics"])

    server_params = StdioServerParameters(command=MCP_COMMAND[0], args=MCP_COMMAND[1:])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result_text = await _call_mcp_tool(session, "search_papers", {
                "query": state["query"],
                "subtopics": state["subtopics"],
                "max_results": 20,
            })

    try:
        candidates = json.loads(result_text)
    except json.JSONDecodeError:
        candidates = []

    logger.info("search_node: found %d candidates", len(candidates))
    return {"candidates": candidates}


async def dedupe_node(state: AgentState) -> dict:
    logger.info("dedupe_node: filtering already-summarized papers")

    server_params = StdioServerParameters(command=MCP_COMMAND[0], args=MCP_COMMAND[1:])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result_text = await _call_mcp_tool(session, "list_papers", {"subtopics": []})

    try:
        existing = json.loads(result_text)
    except json.JSONDecodeError:
        existing = []

    existing_ids = {p["arxiv_id"] for p in existing}
    new_candidates = [c for c in state["candidates"] if c["arxiv_id"] not in existing_ids]

    logger.info("dedupe_node: %d new out of %d candidates", len(new_candidates), len(state["candidates"]))
    return {"candidates": new_candidates}


async def fetch_summaries_node(state: AgentState) -> dict:
    logger.info("fetch_summaries_node: summarizing %d papers", len(state["candidates"]))

    summaries = []
    paths = []
    fetched = []

    server_params = StdioServerParameters(command=MCP_COMMAND[0], args=MCP_COMMAND[1:])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for candidate in state["candidates"]:
                arxiv_id = candidate["arxiv_id"]
                logger.info("Summarizing %s: %s", arxiv_id, candidate.get("title", ""))

                try:
                    result_text = await _call_mcp_tool(session, "summarize_paper", {
                        "arxiv_id": arxiv_id,
                    })
                    result = json.loads(result_text)
                    summaries.append(result)
                    paths.append(result.get("path", ""))
                    fetched.append(arxiv_id)
                except Exception as e:
                    logger.error("Failed to summarize %s: %s", arxiv_id, e)
                    continue

    logger.info("fetch_summaries_node: summarized %d papers", len(summaries))
    return {
        "summaries": summaries,
        "paths": paths,
        "fetched": fetched,
    }


async def compare_node(state: AgentState) -> dict:
    if not state.get("user_interest") and not state.get("fetched"):
        return {"similarities": []}

    similarities = []
    server_params = StdioServerParameters(command=MCP_COMMAND[0], args=MCP_COMMAND[1:])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if state.get("user_interest") and state.get("fetched"):
                logger.info("compare_node: comparing interest to %d papers", len(state["fetched"]))
                result_text = await _call_mcp_tool(session, "compare_to_interest", {
                    "interest": state["user_interest"],
                    "arxiv_ids": state["fetched"],
                    "top_k": 5,
                })
                try:
                    similarities = json.loads(result_text)
                except json.JSONDecodeError:
                    similarities = []

    return {"similarities": similarities}


async def respond_node(state: AgentState) -> dict:
    logger.info("respond_node: synthesizing digest")
    llm = _get_agent_llm()

    summaries_text = ""
    for s in state.get("summaries", []):
        summaries_text += f"\n- **{s.get('summary', 'N/A')}** (file: {s.get('path', 'N/A')})\n"

    similarities_text = ""
    for sim in state.get("similarities", []):
        similarities_text += f"\n- {sim.get('title', sim.get('arxiv_id', '?'))} — similarity: {sim.get('similarity', '?')}\n"

    sys_msg = SystemMessage(content=(
        "You are a research assistant. Synthesize a concise digest of the papers found.\n"
        "Include: paper titles, key findings, file paths where summaries are stored.\n"
        "If similarity rankings are provided, mention the most relevant papers.\n"
        "Keep it brief and actionable."
    ))

    user_content = (
        f"Query: {state['query']}\n"
        f"Subtopics: {state.get('subtopics', [])}\n"
        f"User interest: {state.get('user_interest', 'N/A')}\n"
        f"Summaries:\n{summaries_text}\n"
        f"Similarities:\n{similarities_text}\n"
    )
    user_msg = HumanMessage(content=user_content)

    response = await llm.ainvoke([sys_msg, user_msg])

    raw = response.content
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        text = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw
        )
    else:
        text = str(raw)

    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")

    return {"messages": [sys_msg, user_msg, response]}


def should_compare(state: AgentState) -> str:
    if state.get("user_interest") or state.get("fetched"):
        return "compare"
    return "respond"


def should_retry_search(state: AgentState) -> str:
    if state.get("summaries") and len(state["summaries"]) > 0:
        return "respond"
    if state.get("retry_count", 0) < 2:
        return "retry_search"
    return "respond"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("dedupe", dedupe_node)
    graph.add_node("fetch_summaries", fetch_summaries_node)
    graph.add_node("compare", compare_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "dedupe")
    graph.add_edge("dedupe", "fetch_summaries")
    graph.add_conditional_edges("fetch_summaries", should_retry_search, {
        "respond": "compare",
        "retry_search": "search",
    })
    graph.add_conditional_edges("compare", should_compare, {
        "compare": "compare",
        "respond": "respond",
    })
    graph.add_edge("respond", END)

    return graph


async def run_agent(query: str) -> None:
    graph = build_graph()
    app = graph.compile()

    initial_state: AgentState = {
        "query": query,
        "subtopics": [],
        "candidates": [],
        "fetched": [],
        "summaries": [],
        "paths": [],
        "user_interest": None,
        "similarities": [],
        "error": None,
        "retry_count": 0,
        "messages": [],
    }

    result = await app.ainvoke(initial_state)
    logger.info("Agent completed. Summaries: %d", len(result.get("summaries", [])))


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.agent <research query>", file=sys.stderr)
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    asyncio.run(run_agent(query))


if __name__ == "__main__":
    main()
