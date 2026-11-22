# Aero Copilot

A multi-agent engineering assistant for aerospace / mechanical work. It answers numeric questions
with **real calculations** (via a custom MCP server) and document questions with **cited answers**
(via Oracle AI Vector Search), behind **guardrails**, with **logging** and an **eval suite**.

Built from the Oracle Agentic AI Foundations course: agents, ReAct loops, MCP, OpenAI Agents SDK,
guardrails, Oracle AI Database vector search. The LLM is Gemini (free tier) through its
OpenAI-compatible endpoint.

```
                      you (CLI)
                          |
              +-----------v-----------+
 input        |     Triage Agent      |   regex injection screen
 guardrail -> |  (handoffs, max turns)|   + LLM scope check
              +-----+-------------+---+
                    |             |
        +-----------v--+     +----v--------------+
        | Knowledge    |     | Calculation Agent |
        | Agent        |     |  MCP tools (7)    |
        | search_      |     |  + search_        |
        | knowledge    |     |    knowledge      |
        +------+-------+     +----+--------------+
               |                  |  stdio JSON-RPC
      Oracle 23ai Free       aero/mcp_server.py
      VECTOR(384) + SQL      ISA, Mach, Re, sandwich panel,
      filters (doc_type)     radome slab, TVC torque/angle
               |                  |
        output guardrail    output guardrail
        (citations,         (units, no NaN,
         no secrets)         no secrets)
                          |
                 logs/runs.jsonl (every agent step)
```

## Course module -> where it lives

| Module | Code |
|---|---|
| 1 Agents, ReAct, security | `agents_setup.py` (loop cap `MAX_TURNS`), `guardrails.py`, untrusted-content handling in `tools.py`, `hooks.py` (observability) |
| 2 LangChain internals | tool schema from type hints + docstrings, errors returned as readable text (`mcp_server.py`) |
| 3 MCP | `aero/mcp_server.py` (FastMCP, stdio), consumed by `MCPServerStdio` |
| 4 Agents SDK | handoffs, `@function_tool`, input + output guardrails, tripwires |
| 5 Enterprise | OpenAI-compatible model endpoint, retries/backoff, eval harness, containerised DB |
| 6 Oracle AI Vector Search | `db.py`: `VECTOR(384, FLOAT32)`, `VECTOR_DISTANCE(... COSINE)`, relational filter in the same SQL |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# .env  (keep your existing Gemini key line; add the rest from .env.example)
cp .env.example .env     # then paste your real key

docker compose up -d     # Oracle 23ai Free; first start takes a few minutes
docker compose ps        # wait until it says "healthy"
```

Put 3 to 10 **public** PDFs in `data/pdfs/` (sub-folders become filters, e.g. `data/pdfs/standards/`).
Do not use restricted or proprietary employer documents.

```bash
python -m pytest -q                      # physics unit tests (no network)
python -m aero.ingest                    # PDFs -> chunks -> embeddings -> Oracle
python -m aero.cli -v                    # chat; -v prints every agent step
python -m aero.cli --once "Mach number at 250 m/s at 10 km?"
```

## Evals and the injection test

```bash
python -m evals.run_evals                # 11 calc + 4 guardrail cases (no DB needed)
python -m evals.run_evals --with-db      # + poisoned-document injection test
```

The eval set scores *tool correctness* (right tool, right number in the tool output) separately from
*answer faithfulness* (right number in the final text), plus guardrail catch rate.
`data/samples/poisoned_note.txt` is a normal-looking design note with a hidden instruction to email
credentials to an attacker; the test passes only if the agent does not obey it.
Add your own cases to `evals/questions.json` (especially knowledge questions about *your* PDFs).

## Security notes (honest version)

- The LLM never executes anything: it requests tools, the app/MCP server runs them.
- Retrieved text is wrapped in `<passage>` tags and flagged as untrusted; output guardrails also block
  emails/keys as a second layer. Prompt injection is not "solved" by this, only layered.
- Input guardrail fails **closed** if the scope checker errors (`GUARDRAIL_FAIL_CLOSED=1`).
- The DB account is the app user from docker-compose. Use a read-only account for anything real.
- Example credentials in `.env.example` are for local development only.

## Troubleshooting

| Symptom | Fix |
|---|---|
| 429 / quota errors | Free tier RPM is low. Raise `EVAL_DELAY_SECONDS`, set `GUARDRAIL_LLM=0`, or try another Flash model in `GEMINI_MODEL` |
| 404 model not found | Set `GEMINI_MODEL` to the exact string that worked in your course code |
| Tool-schema / 400 error from Gemini | Try a different Gemini model; report the exact message |
| `ORA-01017` / connection refused | `docker compose ps` must say healthy; check `ORACLE_*` values match `.env` |
| `mcp.server.fastmcp` import error | You have `mcp` 2.x; `pip install "mcp<2"` |
| Answers blocked unexpectedly | `-v` shows the guardrail reason; tune `aero/guardrails.py` |

## Known limits

- `slab_transmission` models one flat slab at normal incidence. A real C-sandwich radome is multilayer
  and needs oblique incidence and polarisation; the tool says so in its description.
- `sandwich_panel_stiffness` is a symmetric three-layer idealisation.
- ISA model covers 0 to 20 km.

## Stretch list

Test-data agent for TVC logs (read-only MCP server over CSV), Rust port of the MCP server,
LangGraph comparison agent, container image + OCI hosted deployment, HNSW vector index.

oracle db (community maintained free version) anonymous 
-e APP_USER=vector_user \
 -e APP_USER_PASSWORD=VectorPass_123 
