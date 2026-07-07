# Arxiv Research Agent — Plan

## Overview
A LangGraph agent that uses an MCP server (stdio transport, spawned on demand) to search arxiv, filter papers by subtopics, summarize them into markdown files, and compare similarity between papers or against a user's stated interest.

## Stack
- **LangGraph** state machine driving an **MCP** server (stdio transport)
- **LLM**: Ollama `qwen3.5:9b` via `langchain-ollama`, host-run (agent runs with `network_mode: host`, so `OLLAMA_BASE_URL=http://localhost:11434` reaches host Ollama)
- **arxiv**: official `arxiv` PyPI client wrapped with a rate limiter
- **Embeddings**: `sentence-transformers` `all-MiniLM-L6-v2` (local, no API key)
- **Storage**: markdown files in bind-mounted `./papers` volume; sidecar `papers/.index/embeddings.npz` for vectors (no DB)

## Repo Layout
```
research-agent/
├── AGENTS.md
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml    # rename: pyproject.toml listed in AGENTS.md but requirements.txt is the actual install file
├── subtopics.yaml
├── app/
│   ├── __init__.py
│   ├── agent.py               # ReAct loop + MCP toolkit wrapper
│   ├── config.py
│   └── mcp/
│       ├── __init__.py
│       ├── server.py          # MCP server (stdio), registers tools
│       ├── arxiv.py           # search + rate limiter + backoff
│       ├── summarize.py       # Ollama summarization
│       ├── store.py           # markdown read/write + front-matter + rejections + digest
│       ├── embed.py           # sentence-transformers + caching
│       └── subtopics.py       # YAML loader + matcher
├── tests/
│   └── test_react_smoke.py    # stubbed-LLM smoke test
├── papers/                    # bind-mounted volume; also holds digest.md and .index/
└── README.md
```

## LangGraph State Machine
State: `{messages, user_query, user_interest, kept_ids, arxiv_calls, summaries_written, iterations}`

Architecture: **single ReAct loop**. The model is bound to the MCP tool list and
iteratively selects tools based on observed results; no hand-coded node ordering.
One MCP stdio subprocess is held open for the whole run (previously the graph
spawned one per node).

Graph: `START → react → (react | END)` — one node, self-loop, conditional exit
on budget exhaustion (`ITER_MAX`, `ARXIV_MAX`, `SUMMARY_MAX`) or `finish` tool call.

Policy (encoded in `SYSTEM_PROMPT`): search → skim candidates → for borderline
papers call `get_paper_details` before committing → summarize the kept set →
optionally `compare_to_interest` → call `finish` with a <400-word digest saved
to `papers/digest.md`. Up to 2 query rewrites if the first search is thin.

See `.opencode/plans/react-agent-rearchitecture.md` for the full design.

## MCP Tools
| Tool | Inputs | Output |
|---|---|---|
| `search_papers` | `query, subtopics, max_results=20` | filtered candidate list |
| `get_paper_details` | `arxiv_id` | metadata + abstract only (no file) |
| `summarize_paper` | `arxiv_id` | writes `papers/<id>.md`, returns summary |
| `skip_paper` | `arxiv_id, reason` | records rejection to `papers/.index/rejections.json` |
| `compare_to_interest` | `interest, arxiv_ids, top_k=5` | ranked similarities |
| `list_papers` | `subtopics=[]` | stored markdown paths + front-matter |
| `finish` | `digest_markdown` | writes `papers/digest.md`, terminates run |

## Rate Limiting
- Min 3s between arxiv HTTP calls (asyncio lock + timestamp gate)
- `tenacity` retry on 429/503: exponential backoff (base 4s, cap 60s), max 5 attempts
- Single global lock serializes concurrent MCP calls

## Markdown Schema
```yaml
---
arxiv_id: "2401.12345"
title: "..."
authors: ["..."]
published: "2024-01-15"
subtopics: ["multi-agent", "reasoning"]
url: "https://arxiv.org/abs/2401.12345"
---
## Abstract
...
## Summary
...
```

## Embeddings Cache
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Vectors stored in `papers/.index/embeddings.npz` + `meta.json`
- Computed at summarize time; loaded lazily for similarity queries

## Subtopics Taxonomy (`subtopics.yaml`)
```yaml
agents:
  keywords: [agent, autonomous agent, llm agent]
multi-agent:
  keywords: [multi-agent, multiagent, agent collaboration, debate]
planning:
  keywords: [planning, task planning, self-refine]
reasoning:
  keywords: [reasoning, chain-of-thought, cot, self-consistency]
tool-use:
  keywords: [tool use, tool calling, function calling, api]
rag:
  keywords: [retrieval augmented, rag, dense retrieval]
```

## Containerization
- `Dockerfile`: python:3.12-slim, uv-based install, pre-download embedding model
- `docker-compose.yml`: mounts `./papers:/app/papers`, uses `network_mode: host` so the container shares the host network and reaches Ollama on `localhost:11434`
- MCP stdio: agent spawns MCP server as subprocess in same container

## Budgets & Guards
- `ITER_MAX=30` (env) — max tool rounds per run
- `ARXIV_MAX=4` (env) — max `search_papers` calls
- `SUMMARY_MAX=8` (env) — max `summarize_paper` calls
- Hard caps enforced in `should_continue`; model cannot exceed.

## Usage
```bash
ollama pull qwen3.5:9b
docker compose run --rm research-agent \
  "Find recent multi-agent reasoning papers and compare them to my interest: efficient inference"
```
