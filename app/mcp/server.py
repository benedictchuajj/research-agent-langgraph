import asyncio
import json
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from app.mcp import arxiv as arxiv_client
from app.mcp import embed as embed_module
from app.mcp import store as store_module
from app.mcp import subtopics as subtopics_module
from app.mcp import summarize as summarize_module

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

server = Server("research-agent")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_papers",
            description="Search arxiv for papers matching a query and filter by subtopics. Returns a filtered candidate list.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "arxiv search query"},
                    "subtopics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subtopics to filter by (e.g. ['multi-agent', 'reasoning'])",
                        "default": [],
                    },
                    "max_results": {"type": "integer", "default": 20},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="summarize_paper",
            description="Fetch a paper by arxiv ID, summarize it, and write it to papers/<id>.md. Returns the summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arxiv paper ID (e.g. '2401.12345')"},
                },
                "required": ["arxiv_id"],
            },
        ),
        Tool(
            name="compare_papers",
            description="Compute cosine similarity between two papers by their arxiv IDs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id_a": {"type": "string"},
                    "id_b": {"type": "string"},
                },
                "required": ["id_a", "id_b"],
            },
        ),
        Tool(
            name="compare_to_interest",
            description="Rank papers by cosine similarity to a free-text interest statement.",
            inputSchema={
                "type": "object",
                "properties": {
                    "interest": {"type": "string", "description": "Free-text description of research interest"},
                    "arxiv_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arxiv IDs to compare against",
                    },
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["interest", "arxiv_ids"],
            },
        ),
        Tool(
            name="list_papers",
            description="List stored paper summaries, optionally filtered by subtopics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subtopics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by subtopics (empty = all)",
                        "default": [],
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search_papers":
            return await _handle_search(arguments)
        elif name == "summarize_paper":
            return await _handle_summarize(arguments)
        elif name == "compare_papers":
            return await _handle_compare_papers(arguments)
        elif name == "compare_to_interest":
            return await _handle_compare_interest(arguments)
        elif name == "list_papers":
            return await _handle_list(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return [TextContent(type="text", text=f"Error: {e}")]


async def _handle_search(args: dict) -> list[TextContent]:
    query = args["query"]
    subtopics_filter = args.get("subtopics", [])
    max_results = args.get("max_results", 20)

    papers = await arxiv_client.search_arxiv(query, max_results=max_results)

    if subtopics_filter:
        filtered = []
        for p in papers:
            text = f"{p['title']} {p['abstract']}"
            matched = subtopics_module.match_subtopics(text)
            p["matched_subtopics"] = matched
            if set(matched).intersection(set(subtopics_filter)):
                filtered.append(p)
        papers = filtered
    else:
        for p in papers:
            text = f"{p['title']} {p['abstract']}"
            p["matched_subtopics"] = subtopics_module.match_subtopics(text)

    return [TextContent(type="text", text=json.dumps(papers, indent=2))]


async def _handle_summarize(args: dict) -> list[TextContent]:
    arxiv_id = args["arxiv_id"]

    paper = await arxiv_client.fetch_paper(arxiv_id)
    if not paper:
        return [TextContent(type="text", text=f"Paper {arxiv_id} not found on arxiv")]

    text_for_topics = f"{paper['title']} {paper['abstract']}"
    matched = subtopics_module.match_subtopics(text_for_topics)
    if not matched:
        matched = ["uncategorized"]

    summary = summarize_module.summarize_paper(paper["title"], paper["abstract"])

    path = store_module.write_paper(
        arxiv_id=arxiv_id,
        title=paper["title"],
        authors=paper["authors"],
        published=paper["published"],
        subtopics=matched,
        url=paper["url"],
        abstract=paper["abstract"],
        summary=summary,
    )

    embed_text = f"{paper['title']} {paper['abstract']} {summary}"
    embed_module.get_or_compute_embedding(arxiv_id, embed_text)

    return [TextContent(type="text", text=json.dumps({
        "arxiv_id": arxiv_id,
        "path": str(path),
        "subtopics": matched,
        "summary": summary,
    }, indent=2))]


async def _handle_compare_papers(args: dict) -> list[TextContent]:
    id_a = args["id_a"]
    id_b = args["id_b"]

    paper_a = store_module.read_paper(id_a)
    paper_b = store_module.read_paper(id_b)

    if not paper_a:
        return [TextContent(type="text", text=f"Paper {id_a} not found locally")]
    if not paper_b:
        return [TextContent(type="text", text=f"Paper {id_b} not found locally")]

    text_a = f"{paper_a['title']} {paper_a['abstract']} {paper_a['summary']}"
    text_b = f"{paper_b['title']} {paper_b['abstract']} {paper_b['summary']}"

    vec_a = embed_module.get_or_compute_embedding(id_a, text_a)
    vec_b = embed_module.get_or_compute_embedding(id_b, text_b)

    sim = embed_module.cosine_similarity(vec_a, vec_b)

    return [TextContent(type="text", text=json.dumps({
        "id_a": id_a,
        "id_b": id_b,
        "cosine_similarity": round(sim, 4),
    }, indent=2))]


async def _handle_compare_interest(args: dict) -> list[TextContent]:
    interest = args["interest"]
    arxiv_ids = args["arxiv_ids"]
    top_k = args.get("top_k", 5)

    query_vec = embed_module.embed_text(interest)

    for aid in arxiv_ids:
        paper = store_module.read_paper(aid)
        if paper:
            text = f"{paper['title']} {paper['abstract']} {paper['summary']}"
            embed_module.get_or_compute_embedding(aid, text)

    ranked = embed_module.top_k_similar(query_vec, arxiv_ids, top_k)

    results = []
    for aid, sim in ranked:
        paper = store_module.read_paper(aid)
        results.append({
            "arxiv_id": aid,
            "similarity": round(sim, 4),
            "title": paper["title"] if paper else "unknown",
            "path": paper["path"] if paper else None,
        })

    return [TextContent(type="text", text=json.dumps(results, indent=2))]


async def _handle_list(args: dict) -> list[TextContent]:
    subtopics_filter = args.get("subtopics", [])
    papers = store_module.list_papers(subtopics=subtopics_filter if subtopics_filter else None)

    summaries = []
    for p in papers:
        summaries.append({
            "arxiv_id": p.get("arxiv_id"),
            "title": p.get("title"),
            "subtopics": p.get("subtopics", []),
            "path": p.get("path"),
        })

    return [TextContent(type="text", text=json.dumps(summaries, indent=2))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
