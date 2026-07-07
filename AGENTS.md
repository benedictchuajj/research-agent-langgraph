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
├── pyproject.toml
├── subtopics.yaml
├── app/
│   ├── __init__.py
│   ├── agent.py               # LangGraph builder + entrypoint
│   ├── config.py
│   └── mcp/
│       ├── __init__.py
│       ├── server.py          # MCP server (stdio), registers tools
│       ├── arxiv.py           # search + rate limiter + backoff
│       ├── summarize.py       # Ollama summarization
│       ├── store.py           # markdown read/write + front-matter
│       ├── embed.py           # sentence-transformers + caching
│       └── subtopics.py       # YAML loader + matcher
├── papers/                    # bind-mounted volume
└── README.md
```

## LangGraph State Machine
State: `{query, subtopics, candidates, fetched, summaries, paths, user_interest, similarities, error}`

Nodes:
1. `plan` — LLM parses topic string into arxiv query + subtopics
2. `search` — MCP `search_papers` with rate-limited arxiv calls
3. `dedupe` — filter already-summarized arxiv IDs
4. `fetch_summaries` — MCP `summarize_paper` for each new ID → writes markdown
5. `compare` (optional) — MCP `compare_to_interest` / `compare_papers`
6. `respond` — LLM synthesizes digest with file paths

Edges: linear `plan -> search -> dedupe -> fetch_summaries -> [compare] -> respond`.
Conditional: `fetch_summaries` loops back to `search` (relaxed subtopics) if zero new summaries found and retry budget remains.

## MCP Tools
| Tool | Inputs | Output |
|---|---|---|
| `search_papers` | `query, subtopics, max_results=20` | filtered candidate list |
| `summarize_paper` | `arxiv_id` | writes `papers/<id>.md`, returns summary |
| `compare_papers` | `id_a, id_b` | cosine similarity |
| `compare_to_interest` | `interest, arxiv_ids, top_k=5` | ranked similarities |
| `list_papers` | `subtopics=[]` | stored markdown paths + front-matter |

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

## Usage
```bash
ollama pull qwen3.5:9b
docker compose run --rm research-agent \
  "Find recent multi-agent reasoning papers and compare them to my interest: efficient inference"
```
