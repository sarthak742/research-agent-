# 🔬 Research Agent

An AI research assistant that plans web searches for a topic, runs them,
**reflects on what's missing and searches again if needed**, then synthesizes
a report where **every claim is cited to a source you can check**. Progress
streams to the browser live over Server-Sent Events.

This is a production-minded rewrite of an earlier prototype. The emphasis is
on the two things that separate a real agent from a GPT wrapper: **a genuine
plan → act → reflect loop**, and **an evaluation harness that measures whether
the output is trustworthy** (are claims grounded in real sources?).

---

## What it does

1. **Plan** — an LLM decides *which* queries to run and *how many* (2–6),
   based on how broad the topic is. Not a hardcoded count.
2. **Search** — queries run concurrently against Tavily; results are
   deduplicated by URL and given stable citation numbers.
3. **Reflect** — the agent looks at what it found and decides whether there's
   a gap worth another round of searching. If coverage is sufficient, it stops.
   The loop is bounded (`max_rounds`, `max_total_searches`) so it can't run away.
4. **Synthesize** — the LLM writes an Executive Summary / Key Findings /
   Limitations report, citing each claim inline as `[n]`, streamed token by
   token to the UI.

## Architecture

```
app.py            FastAPI: routes, real SSE, validation, CORS, lifespan
config.py         all tunables + secrets (pydantic-settings) — no magic numbers
agent/
  models.py       typed events, sessions, sources
  llm.py          async Groq wrapper + retry/backoff  (non-blocking)
  search.py       async Tavily wrapper + retry + TTL cache (non-blocking)
  planner.py      initial_plan (count = topic breadth) + reflect (gap-finding)
  synthesizer.py  report with inline [n] citations, streaming
  store.py        session store behind an interface; in-memory + TTL sweep
  core.py         the plan->search->reflect->synthesize loop, emits events
static/index.html frontend using real EventSource + live token rendering
tests/            41 tests, fully offline (LLM + search are faked)
evals/            planning quality, citation grounding, report quality
```

### Request lifecycle

`POST /research` validates input and creates a session → returns an id.
The browser opens `EventSource("/stream/{id}")`; the agent runs inside that
stream, emitting an event per step. `GET /report/{id}` returns the final
report and sources.

## What changed from the prototype (and why)

| Area | Before | Now | Why it matters |
|------|--------|-----|----------------|
| Async | `groq`/`tavily` sync calls inside `async def` | Groq async client; Tavily via `asyncio.to_thread` | The old code blocked the event loop — one user froze all others |
| "Agent" | plan once, fixed 3/7 searches from a button | plan → reflect → maybe search again; LLM chooses count | Makes it an actual agent, not a pipeline |
| Search count | `3 if basic else 7` (hardcoded) | LLM decides 2–6 by topic breadth, bounded | The résumé claim "decides based on complexity" is now *true* |
| Citations | flat source list at the end | inline `[n]` tied to a numbered reference list | Every claim is checkable — the whole point |
| SSE | `sse_starlette.SSEEvent` (not a real symbol) + manual fetch reader | `EventSourceResponse` + browser `EventSource` | The old streaming was broken |
| Errors | none; a failed search killed the run | retries w/ backoff; per-query failures reported, run continues | Resilience |
| 404s | `return {...}, 404` (served as 200) | `HTTPException(404)` | Correct HTTP semantics |
| Sessions | module-level dict, never cleaned | store interface, in-memory + TTL sweeper, `asyncio.Lock` | Bounded memory; swappable for Redis |
| Validation | none | pydantic `ResearchRequest` | Rejects junk input |
| CORS | `allow_origins=["*"]` | locked to configured origins | Security |
| Config | scattered literals | one `config.py` | Every number has a name and a home |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # add GROQ_API_KEY and TAVILY_API_KEY
python -m uvicorn app:app --reload --port 8000
# open http://localhost:8000
```

Free keys: Groq at console.groq.com, Tavily at tavily.com.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest          # 41 tests, no network needed
```

Every component is tested with the LLM and search faked via the dependency-
injection seams (`LLMClient(client=...)`, `SearchClient(search_fn=...)`,
duck-typed fakes for planner/synthesizer). Coverage: JSON-parsing robustness,
retry/backoff, cache hits, TTL expiry, the loop's dedup + budget enforcement +
failure handling, and the full API surface including SSE.

## Evaluations

The evals answer: *can you trust the output?*

| Eval | Measures | Run |
|------|----------|-----|
| **Grounding** (`eval_grounding.py`) | % of claims cited, % of citations an LLM judge confirms the source supports, count of invalid/fabricated `[n]`, and the list of uncited claims | `python -m evals.eval_grounding --fixture` (offline) or without the flag (live, needs keys) |
| **Planning** (`eval_planning.py`) | per-query relevance (LLM judge), query diversity (lexical), chosen count | `python -m evals.eval_planning` (live) |
| **Report quality** (`eval_report.py`) | 0–1 rubric on coverage/structure/citations | `python -m evals.eval_report` (live) |

The **grounding** scorer is pure logic and unit-tested; the LLM judgements are
injected, so the metric is deterministic given verdicts. An offline fixture run
demonstrates the whole pipeline without API keys.

**Honest limitation:** LLM-as-judge shares blind spots with the model it grades,
so these numbers are a signal, not ground truth. A stronger version adds a small
human-labelled set and checks the judge against it. Also, the claim extractor is
sentence-based, so a meta-sentence like the Limitations line counts as "uncited";
that's a known, harmless false positive worth mentioning rather than hiding.

## Design decisions worth knowing (and defending)

- **Bounded autonomy.** The loop *can* decide to keep searching, but
  `max_rounds` and `max_total_searches` cap cost and latency. An unbounded
  agent is a runaway bill.
- **In-memory sessions.** A research run is transient state, so ephemeral
  storage with a TTL is the right default; the `SessionStore` interface is
  where Redis goes if you scale to multiple workers.
- **`to_thread` for Tavily.** The SDK is synchronous; running it in a worker
  thread keeps the event loop free. If Tavily ships an async client, `search.py`
  is the only file that changes.
- **Temperature split.** Planning uses low temperature (0.2) for consistent,
  parseable output; the report uses 0.4 for readability.
