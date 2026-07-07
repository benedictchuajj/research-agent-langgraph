# Arxiv Research Agent

A LangGraph agent that uses an MCP server (stdio transport) to search arxiv, filter papers by subtopics, summarize them into markdown files, and compare similarity between papers or against a user's stated interest.

## Quick Start (Docker)

The agent runs in Docker; Ollama runs on your host. The compose service
uses `network_mode: host`, so the container shares the host's network
namespace and reaches Ollama on `localhost:11434` directly — no
`OLLAMA_HOST=0.0.0.0` bind needed on the host. Pull a model first, then
run:

```bash
ollama pull qwen3.5:9b
docker compose run --rm research-agent \
  "Find recent papers on long horizon planning"
```

## Features

- **Rate-limited arxiv access** (3s between calls, exponential backoff on errors)
- **Subtopic filtering** via `subtopics.yaml` keyword taxonomy
- **Markdown summaries** stored in `./papers/` (bind-mounted volume, no database)
- **Similarity comparison** using local sentence-transformers embeddings
- **Containerized** with pre-downloaded embedding model
- **LLM via Ollama** (local, host-run — no API keys)

## Architecture

- **LangGraph** state machine: `plan → search → dedupe → fetch_summaries → compare → respond`
- **MCP server** (stdio): spawned as subprocess by the agent
- **Embeddings**: `all-MiniLM-L6-v2`, cached in `papers/.index/`

## MCP Tools

| Tool | Description |
|---|---|
| `search_papers` | Search arxiv with subtopic filtering |
| `summarize_paper` | Fetch, summarize, write markdown |
| `compare_papers` | Cosine similarity between two papers |
| `compare_to_interest` | Rank papers by similarity to interest text |
| `list_papers` | List stored summaries, filter by subtopic |

## Configuration

Environment variables:
- `OLLAMA_MODEL` — default: `qwen3.5:9b` (`.env` overrides)
- `OLLAMA_BASE_URL` — default: `http://localhost:11434`. Works for both
  Docker (host-networked container) and local runs, since `localhost`
  inside the container is the host's loopback where Ollama listens.
- `EMBEDDING_MODEL` — default: `all-MiniLM-L6-v2`
- `ARXIV_MIN_INTERVAL` — default: `3.0` seconds

## Subtopics

Edit `subtopics.yaml` to customize the taxonomy:

```yaml
multi-agent:
  keywords: [multi-agent, multiagent, agent collaboration, debate]
reasoning:
  keywords: [reasoning, chain-of-thought, cot, self-consistency]
```

## Local Development (without Docker)

```bash
conda create -n research-agent python=3.12 -y
conda activate research-agent
pip install -r requirements.txt

ollama pull qwen3.5:9b   # Ollama must be running on the host
python -m app.agent "Find papers on multi-agent reasoning"
```

`OLLAMA_BASE_URL` defaults to `http://localhost:11434`, which works the
same way under Docker (host-networked) and for local runs.
