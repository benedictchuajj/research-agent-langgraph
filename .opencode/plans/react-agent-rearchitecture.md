# Plan: ReAct-style Rearchitecture of the Research Agent

## Motivation
The current LangGraph graph in `app/agent.py` is a fixed DAG
(`plan → search → dedupe → fetch_summaries → compare → respond`) where every
node runs deterministic Python calling MCP tools in a fixed order. The LLM only
participates in two nodes (`plan`, `respond`) and never selects tools, inspects
tool output, or rewrites the query. This is a pipeline with two LLM calls glued
on, not an agent. Two latent bugs confirm the routing is not load-bearing:

- `should_compare`/`should_retry_search` map `"respond"→"respond"` and
  `"compare"→"compare"` (`agent.py:288-291`) — a self-loop / wrong target.
- `should_retry_search` loops back to `search` without relaxing subtopics or
  rewriting the query, so the retry is a no-op (`agent.py:262-267`).

## Goal
Collapse the six fixed nodes into a single ReAct loop where the model picks each
action from the MCP tool list, reads the result, and decides the next step.
Keep the MCP server (correct boundary). Drop the hand-coded ordering and the
broken conditional edges.

## Non-Goals
- No changes to `arxiv.py`, `embed.py`, `subtopics.py`, `summarize.py`.
- No Docker / compose / `requirements.txt` changes.
- No new runtime dependencies (`langchain-ollama.bind_tools` already supported).

## Proposed MCP Tool Surface
Split `summarize_paper` so the model has forks; add a terminal tool.

| Tool | Behavior | Status |
|---|---|---|
| `search_papers(query, max_results)` | arxiv search + subtopic-tag, return candidates | unchanged |
| `get_paper_details(arxiv_id)` | fetch metadata+abstract only (no summary / file) | **new** |
| `summarize_paper(arxiv_id)` | summarize + write `.md` + embed | unchanged |
| `skip_paper(arxiv_id, reason)` | record rejection in `papers/.index/rejections.json` | **new** |
| `compare_to_interest(interest, arxiv_ids, top_k)` | unchanged | unchanged |
| `list_papers(subtopics)` | unchanged | unchanged |
| `finish(digest_markdown)` | write `papers/digest.md`, terminate run | **new** |

`compare_papers(id_a, id_b)` is dropped from the active set (unused by the
current graph; can be re-exposed later if needed).

## State
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    user_interest: Optional[str]
    kept_ids: list[str]        # working set curated by the model
    arxiv_calls: int           # budget guard
    summaries_written: int
    iterations: int
```

## Graph
```
START → react → (react | END)
```
Single node, self-loop, conditional edge on `should_continue`.

## Implementation Steps
1. **MCP server: new tools** (`app/mcp/server.py`)
   - Add `get_paper_details` (arxiv fetch only, no write).
   - Add `skip_paper` → write to `papers/.index/rejections.json`.
   - Add `finish` sentinel tool (writes `papers/digest.md`, returns "done").
   - Drop `compare_papers` from `list_tools`.
2. **Store helper** (`app/mcp/store.py`)
   - Add `read_rejections()` / `write_rejection(arxiv_id, reason)`.
   - Add `write_digest(markdown)`.
3. **Agent rewrite** (`app/agent.py`)
   - `MCPToolkit`: one long-lived stdio MCP session for the whole run.
   - `build_langchain_tools`: wrap each MCP tool as a langchain `StructuredTool`
     with pydantic schemas; dispatch via `bind_tools`.
   - `react_node`: invoke LLM, execute each `tool_call`, append `ToolMessage`s,
     catch `FinishTool` sentinel to terminate.
   - `should_continue`: stop on `ITER_MAX`, `ARXIV_MAX`, or `finish` called.
   - `build_graph`: single node + self-loop conditional edge.
   - `SYSTEM_PROMPT`: real scouting policy (search → skim → commit/reject →
     re-search-if-thin → compare-to-interest → finish).
4. **Budgets** (`app/config.py`)
   - `ITER_MAX=30`, `ARXIV_MAX=4`, `SUMMARY_MAX=8` via env vars.
5. **Tests**
   - Add a smoke test that runs the react loop with a stub LLM returning canned
     tool calls; assert one digest written, rejections recorded, single MCP
     subprocess spawned.

## Tradeoffs
- Hand-rolled ReAct loop over prebuilt `create_react_agent`: ~80 LOC, full
  budget control. Worth it because arxiv rate-limits demand hard guards the
  model cannot override.
- `get_paper_details` adds a tool round per borderline paper but avoids a full
  summarize (arxiv fetch + LLM summary + embed + file write) for off-topic hits.
- `skip_paper` adds one JSON file but closes the cross-run dedupe gap (skipped
  papers currently get re-fetched next run).

## Acceptance
- `app/agent.py` is a single react node + self-loop; no `plan`/`search`/`dedupe`
  / `fetch_summaries` / `compare` / `respond` nodes.
- One MCP subprocess per run (not 5).
- `papers/digest.md` written on every successful run.
- `papers/.index/rejections.json` updated on any `skip_paper`.
- `should_compare` / `should_retry_search` removed.
- Existing MCP tools (`search_papers`, `summarize_paper`, `compare_to_interest`,
  `list_papers`) unchanged.
- Smoke test green.