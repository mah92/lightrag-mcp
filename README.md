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
- **Language-aware model selection**: Persian graphs use `heydariAI/persian-embeddings`,
  Graphs created now use `intfloat/multilingual-e5-small`; graphs built earlier keep the
  model pinned in their `meta.json` (`intfloat/multilingual-e5-base`, `heydariAI/persian-embeddings`)
- **Warm graph cache**: loaded graphs stay in memory for 15 minutes between calls

## Tools

| Tool | Purpose |
|---|---|
| `kg_create(name, language)` | Create a new graph; picks embedding model by language (fa/en) |
| `kg_list()` | List graphs with embedding model + doc count |
| `kg_setup(models)` | Pre-download embedding models, verify lightrag/tiktoken/torch |
| `kg_add_book(graph, pdf_path)` | Book PDF → text extraction → skill generation (deepseek-chat) → insert into graph |
| `kg_add_repo(graph, repo_path)` | Repo → arc42 doc generation + skill-ify `docs/references/*.pdf` → insert |
| `kg_add_markdown(graph, markdown, md_path, doc_name, replace)` | Insert markdown directly — inline text, a `.md` file, or a directory of them (no book/repo pipeline); `replace=true` refreshes a doc already stored under the same name |
| `kg_ask(graph, question, mode)` | Query the graph (`naive` = vector only, `hybrid` = + LLM keywords) |
| `kg_delete(graph, confirm)` | Delete a graph completely (requires `confirm=true`) |

The target graph must be named on every call (e.g. `kg_nav`).

## Install

```bash
# python deps (CPU torch is enough)
pip install lightrag-hku==1.5.7 'mcp>=1.9,<2' torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers transformers tiktoken requests

# or pull the models + verify environment from inside the MCP itself:
#   call kg_setup(models="both")
```

Model downloads happen automatically on first use, or pre-fetch via `kg_setup`.

## Hermes Agent wiring

```yaml
mcp_servers:
  lightrag-kg:
    command: <python-with-deps>/bin/python
    args:
      - /path/to/src/lightrag_kg_mcp/server.py
    enabled: true
```

Requires `DEEPSEEK_API_KEY` in `~/.hermes/.env` (used for entity extraction and
skill generation; embeddings never touch an API).

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
