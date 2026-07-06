# Arxiv Research Agent

A LangGraph agent that uses an MCP server (stdio transport) to search arxiv, filter papers by subtopics, summarize them into markdown files, and compare similarity between papers or against a user's stated interest.

## Quick Start (Docker)

```bash
export OPENAI_API_KEY=your-key-here
docker compose run --rm research-agent \
  "Find recent multi-agent reasoning papers and compare them to my interest: efficient inference"
```

### Using Ollama (Docker)

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2
docker compose run --rm research-agent \
  "Find recent multi-agent reasoning papers"
```

## Features

- **Rate-limited arxiv access** (3s between calls, exponential backoff on errors)
- **Subtopic filtering** via `subtopics.yaml` keyword taxonomy
- **Markdown summaries** stored in `./papers/` (bind-mounted volume, no database)
- **Similarity comparison** using local sentence-transformers embeddings
- **Containerized** with pre-downloaded embedding model
- **LLM provider choice**: OpenAI or Ollama (local)

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
- `LLM_PROVIDER` — `openai` (default) or `ollama`
- `OPENAI_API_KEY` — required when `LLM_PROVIDER=openai`
- `OPENAI_MODEL` — default: `gpt-4o-mini`
- `OLLAMA_MODEL` — default: `llama3.2`
- `OLLAMA_BASE_URL` — default: `http://localhost:11434`
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

export OPENAI_API_KEY=your-key-here
python -m app.agent "Find papers on multi-agent reasoning"
```

### Using Ollama locally

```bash
conda activate research-agent
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2
python -m app.agent "Find papers on multi-agent reasoning"
```
