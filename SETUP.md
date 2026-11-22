# Aero Copilot: Setup and Run Guide

Multi-agent engineering assistant: Gemini (LLM) + custom MCP server (7 engineering tools) + Oracle 23ai
vector search (documents) + guardrails + evals.

## What you need

- Python 3.10+
- A Gemini API key (free tier works)
- Docker **or** Podman, only for the document search (Oracle). The calculation side works without it.
- About 3 GB of disk for the Oracle image and the Python packages

## 1. Python environment

```bash
cd aero-copilot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # large download (PyTorch comes with sentence-transformers)
```

## 2. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Set it to |
|---|---|
| `GEMINI_API_KEY` | Your key (`GOOGLE_API_KEY` also works) |
| `GEMINI_MODEL` | The exact model string that worked in your course code |
| `ORACLE_USER` / `ORACLE_PASSWORD` | Leave the defaults unless you change them in the database command below. They must match. |

## 3. Check the physics (no network needed)

```bash
python -m pytest -q        # expect: 13 passed
```

## 4. First agent test (no database needed)

```bash
python -m aero.cli -v
```

Ask: `What Mach number is 250 m/s at 10,000 m altitude?` (expected about 0.835).
`-v` prints each agent step: handoff, tool call, tool result.

## 5. Start the Oracle database

Pick whichever matches your machine.

**Docker Compose (if `docker compose` works):**

```bash
docker compose up -d
docker compose ps          # wait for "healthy"
```

**Plain Docker (if `docker compose` is unknown):**

```bash
docker run -d --name aero-oracle -p 1521:1521 \
  -e ORACLE_PASSWORD=SysPass_123 \
  -e APP_USER=vector_user \
  -e APP_USER_PASSWORD=VectorPass_123 \
  -v oracle-data:/opt/oracle/oradata \
  gvenzl/oracle-free:23-slim-faststart
```

**Inside distrobox / toolbox (Docker socket not found):**

```bash
distrobox-host-exec podman run -d --name aero-oracle -p 1521:1521 \
  -e ORACLE_PASSWORD=SysPass_123 \
  -e APP_USER=vector_user \
  -e APP_USER_PASSWORD=VectorPass_123 \
  -v oracle-data:/opt/oracle/oradata \
  docker.io/gvenzl/oracle-free:23-slim-faststart
```

Wait for the database to finish initialising (a few minutes on first start):

```bash
docker logs -f aero-oracle          # or: distrobox-host-exec podman logs -f aero-oracle
# Ctrl+C once you see: DATABASE IS READY TO USE!
```

`APP_USER_PASSWORD` must equal `ORACLE_PASSWORD` in `.env`, or ingest fails with `ORA-01017`.

## 6. Add documents

Use **public** documents only. Sub-folders become the `doc_type` filter.

```bash
mkdir -p data/pdfs/tvc data/pdfs/lqr
cd data/pdfs/tvc
curl -L -o ensworth_ntr_tvc_2013.pdf https://ntrs.nasa.gov/api/citations/20140002890/downloads/20140002890.pdf
curl -L -o tvc_vehicle_dynamics_1970.pdf https://ntrs.nasa.gov/api/citations/19700025076/downloads/19700025076.pdf
curl -L -o gimbal_vs_litvc_requirements.pdf https://ntrs.nasa.gov/api/citations/19700003019/downloads/19700003019.pdf
cd ../../..
file data/pdfs/tvc/*       # each line should say "PDF document"
```

Put LQR notes (for example MIT OCW 16.30 lecture 18 and 16.323 lecture notes) in `data/pdfs/lqr/`.
Keep PDFs out of a public repo: add `data/pdfs/**/*.pdf` to `.gitignore`.

## 7. Ingest and chat

```bash
python -m aero.ingest              # PDFs -> chunks -> embeddings -> Oracle
python -m aero.cli -v              # interactive
python -m aero.cli --once "What gimbal angles and rates does the NTR TVC paper consider?"
```

First ingest downloads the embedding model (about 90 MB). If a file prints `no extractable text`, it is
a scanned PDF without a text layer; skip it or use a different source.

## 8. Run the evals

```bash
python -m evals.run_evals              # 11 calculation + 4 guardrail cases, no database
python -m evals.run_evals --with-db    # also runs the poisoned-document injection test
```

Results are printed and saved to `evals/report.json`. Add your own questions to `evals/questions.json`.
Step-by-step logs are in `logs/runs.jsonl`.

## Everyday commands

```bash
docker start aero-oracle          # start the database again later
docker stop aero-oracle           # stop it
docker rm -f aero-oracle && docker volume rm oracle-data   # wipe it completely
```

(Use `distrobox-host-exec podman ...` in place of `docker ...` if that is how you started it.)

## Troubleshooting

| Symptom | Fix |
|---|---|
| `unknown shorthand flag: 'd'` | Compose plugin missing. Use the plain `docker run` command. |
| Cannot connect to `/var/run/docker.sock` | Docker is not reachable from this shell. Use the distrobox/podman command, or start the Docker service on the host. |
| 429 / quota errors | Free-tier rate limit. Wait, raise `EVAL_DELAY_SECONDS`, or set `GUARDRAIL_LLM=0` in `.env`. |
| 404 model not found | Set `GEMINI_MODEL` to a model string that works with your key. |
| `ORA-01017` or connection refused | Database not healthy yet, or passwords in `.env` do not match the container. |
| `mcp.server.fastmcp` import error | `pip install "mcp<2"` |
| Answer blocked unexpectedly | Run with `-v`; the block reason is printed. Tune `aero/guardrails.py`. |

## Project layout

```
aero/
  physics.py          engineering formulas (unit tested)
  mcp_server.py       MCP server, 7 tools, stdio
  agents_setup.py     Gemini model + Triage / Knowledge / Calculation agents
  guardrails.py       input screen + output checks
  db.py, ingest.py    Oracle VECTOR table, chunking, embeddings, search
  tools.py            search_knowledge function tool
  hooks.py            JSONL step logging
  cli.py              chat interface
evals/                questions.json + run_evals.py
tests/                physics unit tests
data/pdfs/            your documents (sub-folder = doc_type)
```

## Security notes

- Passwords in this guide are local development examples. Never reuse them anywhere real.
- Do not put proprietary or restricted employer documents in `data/pdfs/`.
- Do not commit `.env` (it is in `.gitignore`).
