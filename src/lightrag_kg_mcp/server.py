#!/usr/bin/env python3
"""LightRAG MCP server for the INS-nav knowledge library.

Tools:
  kg_create(name, language)     - create a new graph (embedding model by language: fa->Heidari, en->e5)
  kg_add_book(graph, pdf_path)  - book -> skill (book-to-skill pipeline) -> insert into graph
  kg_add_repo(graph, repo_path) - repo -> arc42 doc + skill its reference PDFs -> insert
  kg_ask(graph, question, mode) - query a graph (loads it, keeps warm 15 min)

Graphs live in ~/lightrag/kg_<name>/, embeddings are local models only.
"""
import os, sys, json, time, asyncio, hashlib, shutil, glob, subprocess
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lightrag-kg")

BASE = os.path.expanduser("~/lightrag/kg")
LIBRARY = os.path.expanduser("~/Documents/INS-nav-library")
PY = os.path.expanduser("~/.hermes/hermes-agent/venv/bin/python")

# ---------------- graph registry + warm cache ----------------
_GRAPHS: dict[str, object] = {}
_LAST_USE: dict[str, float] = {}
_TTL = 15 * 60

DEEPSEEK_KEY = ""
for _line in open(os.path.expanduser("~/.hermes/.env")):
    if _line.startswith("DEEPSEEK_API_KEY="):
        DEEPSEEK_KEY = _line.strip().split("=", 1)[1]

def _evict_expired():
    now = time.time()
    for name in list(_GRAPHS):
        if now - _LAST_USE.get(name, 0) > _TTL:
            _GRAPHS.pop(name, None)
            _LAST_USE.pop(name, None)

def _model_for(lang: str) -> str:
    return "heydariAI/persian-embeddings" if lang == "fa" else "intfloat/multilingual-e5-base"

async def _load_graph(name: str):
    _evict_expired()
    if name in _GRAPHS:
        _LAST_USE[name] = time.time()
        return _GRAPHS[name]
    meta_p = f"{BASE}/{name}/meta.json"
    if not os.path.exists(meta_p):
        raise ValueError(f"graph '{name}' does not exist. Use kg_create first.")
    meta = json.load(open(meta_p))
    wd = f"{BASE}/{name}/graph"
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(meta["embedding_model"], device="cpu")
    dim = 768

    async def embed(texts):
        import numpy as np
        prefixed = [("query: " if t.startswith("query:") or len(t) < 300 else "passage: ") + t for t in texts]
        with __import__("torch").no_grad():
            emb = model.encode(prefixed, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    async def llm(text, system_prompt=None, history_messages=[], **kw):
        kw.pop("hashing_kv", None)
        return await openai_complete_if_cache(
            "deepseek-chat", text, system_prompt=system_prompt,
            history_messages=history_messages,
            base_url="https://api.deepseek.com/v1", api_key=DEEPSEEK_KEY, **kw)

    rag = LightRAG(
        working_dir=wd,
        llm_model_func=llm, llm_model_name="deepseek-chat",
        llm_model_max_async=2,
        embedding_func=EmbeddingFunc(embedding_dim=dim, max_token_size=512, func=embed),
        embedding_batch_num=16, embedding_func_max_async=2,
    )
    await rag.initialize_storages()
    _GRAPHS[name] = rag
    _LAST_USE[name] = time.time()
    return rag

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)

# ---------------- tools ----------------
@mcp.tool()
def kg_create(name: str, language: str = "en") -> str:
    """Create a new LightRAG graph. language: 'fa' -> Heidari embeddings, 'en' -> multilingual-e5-base."""
    if language not in ("fa", "en"):
        return json.dumps({"error": "language must be fa or en"})
    os.makedirs(f"{BASE}/{name}/graph", exist_ok=True)
    os.makedirs(f"{BASE}/{name}/inputs", exist_ok=True)
    meta = {"name": name, "language": language, "embedding_model": _model_for(language), "created": time.strftime("%Y-%m-%d")}
    json.dump(meta, open(f"{BASE}/{name}/meta.json", "w"), indent=1)
    return json.dumps({"ok": True, "graph": name, "embedding_model": meta["embedding_model"]})

@mcp.tool()
def kg_list() -> str:
    """List all graphs with their embedding model and doc count."""
    out = []
    for d in sorted(glob.glob(f"{BASE}/*/meta.json")):
        m = json.load(open(d))
        n = len(glob.glob(os.path.dirname(d) + "/inputs/*.md"))
        out.append({"graph": m["name"], "language": m.get("language"), "embedding": m.get("embedding_model"), "docs": n})
    return json.dumps(out, indent=1)

@mcp.tool()
def kg_add_book(graph: str, pdf_path: str, language: str = "en") -> str:
    """Convert a book PDF into a skill (book-to-skill pipeline via subagent delegate script) and insert its markdown into the graph."""
    if not os.path.exists(pdf_path):
        return json.dumps({"error": f"no such pdf: {pdf_path}"})
    slug = os.path.splitext(os.path.basename(pdf_path))[0].lower().replace(" ", "-")[:40]
    extract_dir = f"/tmp/extracted/{slug}"
    os.makedirs(extract_dir, exist_ok=True)
    subprocess.run(["pdftotext", pdf_path, f"{extract_dir}/full_text.txt"], timeout=600)
    # generate a skill via the no-agent batch script (book-to-lightrag-pipeline style), then insert the md files
    skill_dir = f"{LIBRARY}/skills/all-skills/{slug}"
    script = os.path.expanduser("~/lightrag/kg_mcp/generate_skill.py")
    r = subprocess.run([PY, script, extract_dir, skill_dir, language], capture_output=True, text=True, timeout=7200)
    if not os.path.exists(f"{skill_dir}/SKILL.md"):
        return json.dumps({"error": "skill generation failed", "stderr": r.stderr[-400:]})
    os.makedirs(f"{BASE}/{graph}/inputs", exist_ok=True)
    n = 0
    for f in [f"{skill_dir}/SKILL.md"] + glob.glob(f"{skill_dir}/*.md") + glob.glob(f"{skill_dir}/chapters/*.md"):
        bn = os.path.basename(f)
        if bn == "SKILL.md" or True:
            shutil.copy(f, f"{BASE}/{graph}/inputs/{slug}__{bn}"); n += 1
    async def _ins():
        rag = await _load_graph(graph)
        files = sorted(glob.glob(f"{BASE}/{graph}/inputs/{slug}__*.md"))
        await rag.ainsert([open(f, encoding="utf-8").read() for f in files], file_paths=files)
        return len(files)
    inserted = _run(_ins())
    return json.dumps({"ok": True, "skill": slug, "docs_inserted": inserted})

@mcp.tool()
def kg_add_repo(graph: str, repo_path: str, language: str = "en") -> str:
    """For a cloned repo: write arc42 doc (if missing), skill-ify its reference PDFs, insert all into the graph."""
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        return json.dumps({"error": "no such repo dir"})
    repo = os.path.basename(repo_path)
    docs_added = []
    # 1) arc42 if missing -> delegate to arc42 script
    arc = f"{repo_path}/docs/arc42/arc42.md"
    if not os.path.exists(arc):
        script = os.path.expanduser("~/lightrag/kg_mcp/make_arc42.py")
        subprocess.run([PY, script, repo_path], capture_output=True, text=True, timeout=7200)
    if os.path.exists(arc):
        os.makedirs(f"{BASE}/{graph}/inputs", exist_ok=True)
        shutil.copy(arc, f"{BASE}/{graph}/inputs/arc42__{repo}__arc42.md")
        docs_added.append(f"arc42__{repo}__arc42.md")
    # 2) skill-ify reference PDFs
    for pdf in glob.glob(f"{repo_path}/docs/references/*.pdf"):
        slug = os.path.splitext(os.path.basename(pdf))[0].lower()[:40]
        skill_dir = f"{LIBRARY}/skills/all-skills/{slug}"
        if not os.path.exists(f"{skill_dir}/SKILL.md"):
            ed = f"/tmp/extracted/{slug}"; os.makedirs(ed, exist_ok=True)
            subprocess.run(["pdftotext", pdf, f"{ed}/full_text.txt"], timeout=600)
            script = os.path.expanduser("~/lightrag/kg_mcp/generate_skill.py")
            subprocess.run([PY, script, ed, skill_dir, language], capture_output=True, text=True, timeout=7200)
        if os.path.exists(f"{skill_dir}/SKILL.md"):
            for f in [f"{skill_dir}/SKILL.md"] + glob.glob(f"{skill_dir}/*.md") + glob.glob(f"{skill_dir}/chapters/*.md"):
                dst = f"{BASE}/{graph}/inputs/{slug}__{os.path.basename(f)}"
                shutil.copy(f, dst)
                if os.path.basename(dst) not in docs_added: docs_added.append(os.path.basename(dst))
    async def _ins():
        rag = await _load_graph(graph)
        files = [f"{BASE}/{graph}/inputs/{d}" for d in docs_added if os.path.exists(f"{BASE}/{graph}/inputs/{d}")]
        await rag.ainsert([open(f, encoding="utf-8").read() for f in files], file_paths=files)
        return len(files)
    inserted = _run(_ins())
    return json.dumps({"ok": True, "repo": repo, "docs": docs_added, "inserted": inserted})

@mcp.tool()
def kg_ask(graph: str, question: str, mode: str = "naive") -> str:
    """Ask the graph. mode: naive (vector only, cheap) | hybrid (needs LLM keyword extraction). Graph stays warm 15 min."""
    async def _q():
        rag = await _load_graph(graph)
        if mode == "naive":
            res = await rag.aquery(question, param=__import__("lightrag").QueryParam(mode="naive"))
        else:
            res = await rag.aquery(question, param=__import__("lightrag").QueryParam(mode="hybrid"))
        return str(res)
    return _run(_q())


@mcp.tool()
def kg_delete(graph: str, confirm: bool = False) -> str:
    """Delete a graph completely (meta, graph storage, inputs). Requires confirm=True."""
    import shutil as _sh
    gd = f"{BASE}/{graph}"
    if not os.path.isdir(gd):
        return json.dumps({"error": f"graph '{graph}' does not exist"})
    if not confirm:
        return json.dumps({"error": "set confirm=true to actually delete", "would_delete": gd})
    _GRAPHS.pop(graph, None); _LAST_USE.pop(graph, None)
    _sh.rmtree(gd)
    return json.dumps({"ok": True, "deleted": gd})

@mcp.tool()
def kg_setup(models: str = "both") -> str:
    """Pre-download embedding models + verify environment. models: 'both' | 'e5' | 'heidari'. Returns status of each component."""
    out = {"models": {}, "env": {}}
    want = {"both": ["intfloat/multilingual-e5-base", "heydariAI/persian-embeddings"],
            "e5": ["intfloat/multilingual-e5-base"],
            "heidari": ["heydariAI/persian-embeddings"]}[models]
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    from sentence_transformers import SentenceTransformer
    for m in want:
        try:
            st = SentenceTransformer(m, device="cpu")
            out["models"][m] = {"ok": True, "dim": st.get_sentence_embedding_dimension()}
            del st
        except Exception as e:
            out["models"][m] = {"ok": False, "error": str(e)[:200]}
    try:
        import lightrag; out["env"]["lightrag"] = lightrag.__version__
    except Exception as e:
        out["env"]["lightrag"] = f"MISSING: {e}"
    try:
        import tiktoken; tiktoken.get_encoding("o200k_base"); out["env"]["tiktoken"] = "ok"
    except Exception as e:
        out["env"]["tiktoken"] = f"FAIL: {str(e)[:120]}"
    try:
        import torch; out["env"]["torch_cuda"] = torch.cuda.is_available()
    except Exception:
        out["env"]["torch_cuda"] = False
    out["env"]["deepseek_key"] = bool(DEEPSEEK_KEY)
    return json.dumps(out, indent=1)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
