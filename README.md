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

## Install (independent procedure)

One venv, one package: the MCP server **and** the pipeline CLI live in the same interpreter.
A second venv without `lightrag` is exactly how the graph tools broke silently before
(`kg_create` worked, `kg_ask` did not) — don't rebuild that shape.

```bash
# both repos are public: HTTPS needs no key (use git@github.com:... if you prefer SSH)
git clone --depth 1 --branch v0.2.0 https://github.com/mah92/lightrag-mcp.git && cd lightrag-mcp
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
