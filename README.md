<div align="center">

# ﷽

---

**lightrag-mcp** — MCP server for LightRAG knowledge graphs with local embeddings

</div>

## Overview

A Model Context Protocol (MCP) server that wraps [LightRAG](https://github.com/HKUDS/LightRAG)
knowledge graphs. Built for a technical-document library (books → skills → knowledge graph)
but generic: any markdown corpus works.

Key properties:
- **Local embeddings only** (HuggingFace models, no cloud embedding APIs)
- **Language-aware model selection**: new graphs use `intfloat/multilingual-e5-small`; graphs built
  earlier keep the model pinned in their `meta.json` (`intfloat/multilingual-e5-base`,
  `heydariAI/persian-embeddings`)
- **Graphs anywhere on disk**: `kg_register` points the server at a graph that already lives
  elsewhere (no symlinks, no data copying)
- **Warm graph cache**: loaded graphs stay in memory for 15 minutes between calls

## Tools

| Tool | Purpose |
|---|---|
| `kg_create(name, language)` | Create a new graph; picks embedding model by language (fa/en) |
| `kg_list()` | List graphs with embedding model + doc count |
| `kg_setup(models)` | Pre-download embedding models, verify lightrag/tiktoken/torch |
| `kg_add_book(graph, pdf_path)` | Book PDF → text extraction → section boundaries from the PDF's own bookmarks (`make_sections.py`, fallback `"Chapter N"` regex) → skill generation (deepseek-chat) → insert into graph |
| `kg_add_repo(graph, repo_path)` | Repo → arc42 doc generation + skill-ify `docs/references/*.pdf` → insert |
| `kg_add_markdown(graph, markdown, md_path, doc_name, replace)` | Insert markdown directly — inline text, a `.md` file, or a directory of them (no book/repo pipeline); `replace=true` refreshes a doc already stored under the same name |
| `kg_add_videos(graph, steps, title)` | Staged ingest of VIDEOS: the `steps` you pass run in order (download audio → ASR → correction → English → one skill per video → insert). Returns a job id at once; poll `kg_jobs`, undo with `kg_rollback` |
| `kg_add_sites(graph, steps, title)` | Staged ingest of WEBSITES: crawl the pages → collect what users say about them → one skill per site → insert. Same contract as `kg_add_videos` |
| `kg_rollback(job_id, dry_run)` | Undo a staged run: delete exactly the graph documents that run inserted, by reading the run's manifest |
| `kg_jobs(limit, include_stale)` | State of background jobs — book runs and staged ingests alike, each in its own shape; jobs whose graph was deleted are hidden unless `include_stale=true` |
| `kg_ask(graph, question, mode)` | Query the graph (`naive` = vector only, `hybrid` = + LLM keywords) |
| `kg_register(graph, root)` | Register an EXISTING graph kept outside `~/lightrag/kg` (writes `meta.json` with `"root": root`; expects `<root>/graph/` + `<root>/inputs/`) |
| `kg_delete(graph, confirm, delete_data)` | Delete a graph (requires `confirm=true`); a registered graph whose data lives elsewhere keeps its data unless `delete_data=true` |

The target graph must be named on every call (e.g. `kg_nav`).

### Graphs outside `~/lightrag/kg`

A graph's data lives in `~/lightrag/kg/<name>/` by default. For a graph already built elsewhere
(e.g. the INS-nav graph at `~/lightrag/ins-nav`), register it in place:

```
kg_register("kg_nav", "/home/oem/lightrag/ins-nav")
```

`meta.json` then carries `"root": "/home/oem/lightrag/ins-nav"`, and every tool (list/ask/add_*/
delete) resolves `<root>/graph` + `<root>/inputs` — no symlink farm, no copying. `meta.json` itself
stays under `~/lightrag/kg/<name>/` as the registry entry, and `kg_delete` never removes data that
lives outside it unless you pass `delete_data=true`.

## Staged ingest: `kg_add_videos` / `kg_add_sites` / `kg_rollback`

The two ingest tools do not contain any download/ASR/crawl machinery. They take the plan and run
it, so an upstream change (yt-dlp, a proxy, a model) never needs a new MCP release:

```
kg_add_videos(graph="kg_tajer", title="pcb supplier reviews", steps='[
  {"name": "collect",  "cmd": "bash ~/scripts/yt_collect.sh ..."},
  {"name": "asr",      "cmd": "python ~/scripts/asr_batch.py ..."},
  {"name": "insert",   "cmd": "... kg_query ...", "inserts": ["corpus__x.md", "skills__x.md"]}
]')
```

- `steps` is a **JSON array run in order** — the order is the contract
  (collect → ASR → correct → translate → skill → insert). Each entry is
  `{"name", "cmd", "inserts"}`; `cmd` is a shell command run with `cwd=workdir`.
- **`inserts` is the manifest.** Every file a step declares there is recorded, so
  `kg_rollback(job_id)` can later delete exactly those documents and nothing else. A run with no
  inserts is a no-op you can undo cleanly.
- `backup` defaults to the `kg_graph_backup.sh` **shipped with this package** (next to the server
  module), run for **this graph only** before and after the job; `backup=""` skips it and any other
  string runs as your own command. The backup's exit code is **checked**: `job.json` records
  `backup_before`/`backup_after` (`ok` / `failed (exit N)` / `skipped`) and a failure is listed
  under `warnings` — a job that reports `done` with no backup is worse than one that says the
  backup failed. Keep a copy of the script wherever you like and pass its path if you prefer.
- **Archive names are unique and say which phase they are:** `<graph>_<YYYYmmdd_HHMMSS>[_before|_after][_N].tar.gz`.
  Second precision + the `KG_BACKUP_LABEL` phase + a collision counter — minute precision used to
  make a fast job's before and after snapshots the same file, so the pre-run state was silently
  overwritten. Retention keeps the **5** newest per graph with their `.sha256` files
  (`LIGHTRAG_KG_BACKUP_KEEP` overrides it, `LIGHTRAG_KG_BACKUP_DIR` moves the target).
- The call returns a **job id immediately** — the work happens in a detached runner
  (`kg_ingest.py`, resolved beside the server module). Per-step state, the manifest and the log
  live in `~/lightrag/jobs/<id>/` (`spec.json`, `job.json`, `job.log`).
- **`kg_jobs` prints the shape that matches the job kind.** Book jobs carry `pdf` + `docs_inserted`;
  staged jobs carry `kind`, `title`, `step`/`step_index`, `steps_total`, `manifest` and `warnings`.
  Jobs whose graph no longer exists are marked `"note": "graph no longer exists ..."` and are
  **skipped unless `include_stale=True`**.
- `kg_delete(graph, delete_data=true)` also removes the job records of that graph (a job whose
  graph is gone can never be rolled back, so keeping it only pollutes `kg_jobs`).
- A step whose command exits non-zero stops the run (`state: failed`, `error: "<step> (exit N)"`).
  Note the manifest is written for a step's inserts even when that step fails, so review
  `kg_rollback(job_id, dry_run=true)` before a real rollback.
- `kg_rollback` resolves documents by **basename** (LightRAG stores basenames, not full paths), so
  declare distinctive `inserts` names — a generic name (`ch04.md`) can collide with an unrelated
  document already in the graph. Both the dry run and a real run report the same **file names**
  (`would_remove` / `removed`); a real run also returns `docs` with the `doc_id` and status of each
  deletion.

## Install (independent procedure)

One venv, one package: the MCP server **and** the pipeline CLI live in the same interpreter.
A second venv without `lightrag` is exactly how the graph tools broke silently before
(`kg_create` worked, `kg_ask` did not) — don't rebuild that shape.

```bash
# both repos are public: HTTPS needs no key (use git@github.com:... if you prefer SSH)
git clone --depth 1 --branch v0.2.2 https://github.com/mah92/lightrag-mcp.git && cd lightrag-mcp
./install.sh                      # venv -> ~/.hermes/lightrag-mcp-venv (LIGHTRAG_MCP_VENV overrides)
```

`install.sh` creates the venv, installs CPU-only torch, installs the package, and then runs the
one check that catches the classic outage:

```
python -c "import lightrag, torch, sentence_transformers, fitz, mcp.server.fastmcp"
```

Manual equivalent:

```bash
python3.11 -m venv ~/.hermes/lightrag-mcp-venv
~/.hermes/lightrag-mcp-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/.hermes/lightrag-mcp-venv/bin/pip install -e .
```

Register it with Hermes (per profile — `-p <name>` for another profile):

```bash
hermes mcp add lightrag-kg --command ~/.hermes/lightrag-mcp-venv/bin/kg-mcp
```

`DEEPSEEK_API_KEY` must be in `~/.hermes/.env` (entity extraction + skill generation only;
embeddings are always local). Models download on first use, or pre-fetch with `kg_setup`.

### MCP server, or skill — or both?

Both, and never duplicated:

| piece | what it is | where it lives |
|---|---|---|
| this repo | behaviour: the MCP tools and the `kg-*` CLI | `mah92/lightrag-mcp` |
| skill `book-to-lightrag-pipeline` | judgement: when to use which tool, graph conventions, pipeline pitfalls, eval/report formats | `mah92/my-hermes-skills` -> `~/.hermes/skills/` |
| persona / profile glue | identity: persona name, persona text, chat ids, which graph — e.g. `askar.py` + `personas/askar.txt` | **outside** both (profile side) |

Install the skill from the collection (knowledge, not code — no copies of the code in it):

```bash
cp -r ~/my-hermes-skills/book-to-lightrag-pipeline ~/.hermes/skills/
```

Model downloads happen automatically on first use, or pre-fetch via `kg_setup`.

**The interpreter that runs the server must have `lightrag-hku` + `openai`.** A separate MCP venv
(e.g. `~/.hermes/mcp-venv`, needed for `mcp<2`) that lost `lightrag` fails every graph operation
with `ModuleNotFoundError` while `kg_create` / `kg_list` keep working — which hides the problem.
Check with: `<venv>/bin/python -c "import lightrag, tiktoken, openai"`.

## Hermes Agent wiring

```yaml
mcp_servers:
  lightrag-kg:
    command: ~/.hermes/lightrag-mcp-venv/bin/kg-mcp     # or <venv>/bin/python + src/.../server.py
    enabled: true
```

### CLI (same venv, no MCP needed)

```bash
kg-add-book kg_nav ./some-book.pdf     # PDF -> skill markdown -> graph (resumable)
kg-query "arrival cost" -g kg_nav      # context only — no answer LLM, no cost beyond embedding
kg-ask   "why MHE?"     -g kg_nav --persona-file p.txt --name "Expert"   # answer + [ref N]
```

## Companion scripts

- `generate_skill.py <extract_dir> <skill_dir> <lang>` — book-to-skill conversion
  (SKILL.md + glossary + patterns + cheatsheet + chapters for large corpora)
- `make_arc42.py <repo_path>` — arc42 documentation with a source-derived
  Mathematical Foundations section

## Data layout

```
~/lightrag/kg/<graph_name>/
├── meta.json        # name, language, embedding model
├── graph/           # LightRAG storage (graphml, vector DBs, caches)
└── inputs/          # inserted markdown (<source>__<file>.md)
```

## Benchmark note

Embedding model choice is backed by a 291-chunk / 24-query benchmark on an
inertial-navigation corpus (RTX 3080, fp16): e5-base beat the Persian-specific
Heidari model on English technical text while indexing 84x faster.

## License

MIT
