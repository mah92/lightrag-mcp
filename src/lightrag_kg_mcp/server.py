#!/usr/bin/env python3
"""LightRAG MCP server for the INS-nav knowledge library.

Tools:
  kg_create(name, language)     - create a new graph (embedding model by language: fa->Heidari, en->e5)
  kg_register(name, root)       - register an EXISTING graph that lives outside ~/lightrag/kg
  kg_add_book(graph, pdf_path)  - book -> skill (book-to-skill pipeline) -> insert into graph
  kg_add_repo(graph, repo_path) - repo -> arc42 doc + skill its reference PDFs -> insert
  kg_ask(graph, question, mode) - query a graph (loads it, keeps warm 15 min)
  kg_query(graph, question, mode) - retrieve context only (no answer LLM)
  kg_add_markdown(graph, markdown|md_path) - insert markdown text/file straight into a graph
  kg_add_videos(graph, steps)   - staged ingest of videos (download -> ASR -> correct -> translate -> skill -> insert)
  kg_add_sites(graph, steps)    - staged ingest of websites (crawl -> what users say -> one skill per site -> insert)
  kg_rollback(job_id)           - undo a staged run: delete exactly the documents that run inserted

Graphs live in ~/lightrag/kg_<name>/, embeddings are local models only.
"""
import os, re, sys, json, time, asyncio, hashlib, shutil, glob, subprocess

from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lightrag-kg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kg_common import (BASE, HERMES_PY, JOBS, inputs_dir as _inputs_dir, meta_of as _meta_of,  # noqa: E402
                       graph_dir as _graph_dir, script as _script, safe_name as _safe_name,
                       env_value, job_new_id, job_paths)

LIBRARY = os.path.expanduser("~/Documents/INS-nav-library")

# Staged-ingest backup. "default" resolves at call time to the kg_graph_backup.sh that ships with
# the package (so a fresh clone is backed up too) and backs up ONLY the graph being written; the
# empty string skips the backup and any other value is run as the caller's own command.
DEFAULT_BACKUP = "default"


def _backup_cmd(graph: str, backup: str) -> tuple:
    """(shell command, warning) for the staged-run backup. Warning is non-empty when the caller
    asked for the default but the shipped script is missing — the run must say so, not fail quietly."""
    if backup in ("", "none", "-"):
        return "", ""
    if backup == DEFAULT_BACKUP:
        path = _script("kg_graph_backup.sh")
        if not os.path.exists(path):
            return "", f"backup script not found at {path} — the run is NOT backed up"
        return f"bash {path} {graph}", ""
    return backup, ""


def _pipeline_python() -> str:
    """Interpreter for the pipeline helpers: this one when it has lightrag + pymupdf (one-venv install),
    else the hermes venv (older two-venv installs)."""
    import importlib.util
    if importlib.util.find_spec("lightrag") and importlib.util.find_spec("fitz"):
        return sys.executable
    return HERMES_PY

# ---------------- graph registry + warm cache ----------------
_GRAPHS: dict[str, object] = {}
_LAST_USE: dict[str, float] = {}
_TTL = 15 * 60

DEEPSEEK_KEY = env_value("DEEPSEEK_API_KEY", "")

def _evict_expired():
    now = time.time()
    for name in list(_GRAPHS):
        if now - _LAST_USE.get(name, 0) > _TTL:
            _GRAPHS.pop(name, None)
            _LAST_USE.pop(name, None)


_ONNX_CACHE: dict = {}
_ST_CACHE: dict = {}

def _get_st_model(model_name):
    from sentence_transformers import SentenceTransformer
    if model_name not in _ST_CACHE:
        _ST_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _ST_CACHE[model_name]

# Pre-exported ONNX models on HuggingFace (mah92) - downloaded instead of local torch export
HF_ONNX_REPOS = {
    "intfloat/multilingual-e5-small": "mah92/e5-small-onnx",
    "intfloat/multilingual-e5-base": "mah92/e5-base-onnx",
    "heydariAI/persian-embeddings": "mah92/persian-embeddings-onnx",
}

def _onnx_cache_path(model_name: str) -> str:
    safe = model_name.replace("/", "__")
    return os.path.expanduser(f"~/.cache/lightrag-mcp/onnx/{safe}/model.onnx")

def _get_or_export_onnx(model_name: str):
    """Load ONNX session for the model. Order: local cache -> HF repo (mah92) -> local torch export. None = torch fallback."""
    if model_name in _ONNX_CACHE:
        return _ONNX_CACHE[model_name]
    path = _onnx_cache_path(model_name)
    try:
        if not os.path.exists(path):
            hf_repo = HF_ONNX_REPOS.get(model_name)
            if hf_repo:
                from huggingface_hub import hf_hub_download
                os.makedirs(os.path.dirname(path), exist_ok=True)
                hf_hub_download(repo_id=hf_repo, filename="model.onnx",
                                local_dir=os.path.dirname(path))
                from transformers import AutoTokenizer
                try:
                    AutoTokenizer.from_pretrained(os.path.dirname(path))
                except Exception:
                    AutoTokenizer.from_pretrained(model_name)  # tokenizer files may live with base model
            else:
                _export_onnx(model_name, path)
        import onnxruntime as ort
        from transformers import AutoTokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 4
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        tok = AutoTokenizer.from_pretrained(os.path.dirname(path))
        _ONNX_CACHE[model_name] = (sess, tok)
        return _ONNX_CACHE[model_name]
    except Exception:
        return None

def _export_onnx(model_name: str, out_path: str):
    """One-time export: torch -> ONNX with mean-pooling + L2 norm wrapper (same math as sentence-transformers)."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    core = AutoModel.from_pretrained(model_name).eval()

    class _MeanPool(torch.nn.Module):
        def __init__(self, mod):
            super().__init__(); self.mod = mod
        def forward(self, input_ids, attention_mask):
            out = self.mod(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(emb, dim=-1)

    wrapped = _MeanPool(core).eval()
    dummy = tok(["query: warmup"], return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(wrapped, (dummy["input_ids"], dummy["attention_mask"]), out_path,
                          input_names=["input_ids", "attention_mask"],
                          output_names=["embedding"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                        "attention_mask": {0: "batch", 1: "seq"},
                                        "embedding": {0: "batch"}},
                          opset_version=14)
    tok.save_pretrained(os.path.dirname(out_path))

def _model_for(lang: str) -> str:
    # e5-small wins the fa+en benchmark (0.395/0.772 MRR@10, fastest index); heydariAI stays
    # available for graphs already built with it (meta.json pins their model).
    return "intfloat/multilingual-e5-small"

async def _load_graph(name: str):
    _evict_expired()
    if name in _GRAPHS:
        _LAST_USE[name] = time.time()
        return _GRAPHS[name]
    meta, meta_p = _meta_of(name)
    if meta is None:
        raise ValueError(f"graph '{name}' does not exist. Use kg_create or kg_register first.")
    wd = f"{_graph_dir(name, meta)}/graph"
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc
    from sentence_transformers import SentenceTransformer
    model_name = meta["embedding_model"]
    # Probe actual embedding dim at runtime — hardcoding 768 broke 1024-dim models (e.g. heydariAI/persian-embeddings)
    _onnx = _get_or_export_onnx(model_name)
    import numpy as _np
    if _onnx is not None:
        _sess, _tok = _onnx
        _enc = _tok(["dim probe"], padding=True, return_tensors="np")
        dim = int(_sess.run(None, {"input_ids": _enc["input_ids"].astype(_np.int64),
                                   "attention_mask": _enc["attention_mask"].astype(_np.int64)})[0].shape[1])
    else:
        dim = int(_np.asarray(_get_st_model(model_name).encode(["dim probe"])).shape[1])

    async def embed(texts):
        import numpy as np
        prefixed = [("query: " if t.startswith("query:") or len(t) < 300 else "passage: ") + t for t in texts]
        if _onnx is not None:
            sess, tok = _onnx
            enc = tok(prefixed, padding=True, truncation=True, max_length=512, return_tensors="np")
            feed = {"input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64)}
            return sess.run(None, feed)[0].astype(np.float32)
        with __import__("torch").no_grad():
            emb = _get_st_model(model_name).encode(prefixed, batch_size=32,
                normalize_embeddings=True, show_progress_bar=False)
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

async def _insert_files(graph: str, files: list[str]) -> int:
    """Insert markdown files into a graph unchanged, return the file count."""
    rag = await _load_graph(graph)
    texts = [open(f, encoding="utf-8").read() for f in files]
    await rag.ainsert(texts, file_paths=files)
    return len(files)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip()).strip("-.") or "note"


async def _delete_by_file_name(graph: str, names: set) -> list:
    """Delete documents whose stored file name is in `names` (the replace=True path).

    Returns one entry per deleted document — {"file", "doc_id", "status"} — so callers report
    file names instead of raw doc ids the caller never asked with.
    """
    from lightrag.base import DocStatus
    rag = await _load_graph(graph)
    docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    removed = []
    for doc_id, st in docs.items():
        fname = os.path.basename(st.file_path or "")
        if fname in names:
            res = await rag.adelete_by_doc_id(doc_id)
            removed.append({"file": fname, "doc_id": doc_id,
                            "status": str(getattr(res, "status", "unknown"))})
    return removed

# ---------------- tools ----------------
@mcp.tool()
def kg_create(name: str, language: str = "en") -> str:
    """Create a new LightRAG graph. Default embedding is multilingual-e5-small for all languages (benchmark winner); existing graphs keep their pinned model."""
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
        n = len(glob.glob(_inputs_dir(m["name"], m) + "/*.md"))
        out.append({"graph": m["name"], "language": m.get("language"), "embedding": m.get("embedding_model"), "docs": n})
    return json.dumps(out, indent=1)

@mcp.tool()
async def kg_add_book(graph: str, pdf_path: str, language: str = "en", background: bool = False) -> str:
    """Book PDF -> skill markdown -> graph (runs the shared, resumable kg_add_book.py pipeline).

    background=True returns at once with a job id and log path (poll with kg_jobs) — the right
    choice for real books, which take 20-30 min; the foreground call blocks this MCP server.
    """
    pdf_path = os.path.abspath(os.path.expanduser(pdf_path))
    if _meta_of(graph)[0] is None:
        return json.dumps({"error": f"graph '{graph}' does not exist. Use kg_create or kg_register first."})
    if not os.path.exists(pdf_path):
        return json.dumps({"error": f"no such pdf: {pdf_path}"})
    args = [_pipeline_python(), _script("kg_add_book.py"), graph, pdf_path, "--language", language,
            "--skill-root", f"{LIBRARY}/skills/all-skills"]
    if background:
        job_id = job_new_id(_safe_name(os.path.splitext(os.path.basename(pdf_path))[0].lower())[:40])
        jdir, jjson, jlog = job_paths(job_id)
        os.makedirs(jdir, exist_ok=True)
        # seed the job record before spawning: a job whose child has not written yet must not read
        # as "unknown" when polled immediately (the child's first update merges over this).
        json.dump({"job": job_id, "state": "starting", "graph": graph, "pdf": pdf_path,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, open(jjson, "w"), indent=1)
        with open(jlog, "w") as fh:
            p = subprocess.Popen(args + ["--job-json", jjson], stdout=fh, stderr=subprocess.STDOUT,
                                 start_new_session=True, cwd=jdir)
        d0 = json.load(open(jjson)); d0["pid"] = p.pid
        json.dump(d0, open(jjson, "w"), indent=1)
        return json.dumps({"ok": True, "job": job_id, "pid": p.pid, "log": jlog,
                           "hint": "poll with kg_jobs; the job keeps running if this server restarts"})
    r = subprocess.run(args, capture_output=True, text=True, timeout=14400)
    tail = ((r.stdout or "") + (r.stderr or ""))[-400:]
    if "DONE" not in (r.stdout or ""):
        return json.dumps({"error": "book pipeline failed", "rc": r.returncode, "tail": tail})
    return json.dumps({"ok": True, "graph": graph, "pdf": os.path.basename(pdf_path), "tail": tail})

def _start_staged_job(graph: str, steps: str, title: str, kind: str,
                     backup: str, workdir: str) -> str:
    """Shared body of kg_add_videos / kg_add_sites: store the spec, start the runner.

    `steps` is a JSON array run in order — the order is the contract
    (collect -> ASR -> correct -> English -> skill -> graph). The MCP does not know
    about yt-dlp, proxies or ASR models: those live in the scripts each step calls,
    so an upstream change never needs a new MCP release.
    """
    import json as _json
    try:
        parsed = _json.loads(steps)
    except Exception as e:
        return _json.dumps({"error": f"steps must be a JSON array: {e}"})
    if isinstance(parsed, dict):
        parsed = parsed.get("steps", [])
    if not isinstance(parsed, list) or not parsed:
        return _json.dumps({"error": "steps must be a non-empty JSON array"})
    if _meta_of(graph)[0] is None:
        return _json.dumps({"error": f"graph '{graph}' does not exist. Use kg_create or kg_register first."})
    job_id = job_new_id(_safe_name((title or kind).lower())[:32])
    jdir, jjson, jlog = job_paths(job_id)
    os.makedirs(jdir, exist_ok=True)
    backup, backup_warning = _backup_cmd(graph, backup)
    spec = {"job_id": job_id, "graph": graph, "kind": kind, "title": title,
            "steps": parsed, "backup": backup, "workdir": os.path.expanduser(workdir or "~")}
    spec_path = f"{jdir}/spec.json"
    with open(spec_path, "w", encoding="utf-8") as f:
        _json.dump(spec, f, ensure_ascii=False, indent=1)
    # Seed the record in the same shape kg_jobs prints for a staged job, so a poll right after the
    # call shows steps/manifest rather than nulls (the runner merges over this as it progresses).
    _json.dump({"job": job_id, "state": "queued", "kind": kind, "graph": graph, "title": title,
                "steps_total": len(parsed), "step": None, "step_index": 0, "manifest": [],
                "warnings": [backup_warning] if backup_warning else [],
                "backup": backup or None,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")},
               open(jjson, "w"), ensure_ascii=False, indent=1)
    runner = _script("kg_ingest.py")
    if not os.path.exists(runner):
        return _json.dumps({"error": f"runner missing: {runner}"})
    logf = open(jlog, "a")
    proc = subprocess.Popen([_pipeline_python(), runner, spec_path],
                            stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    out = {"job": job_id, "pid": proc.pid, "state": "queued", "kind": kind,
           "steps": [x.get("name") for x in parsed],
           "backup": backup or "none (skipped)",
           "spec": spec_path, "log": jlog,
           "hint": "poll with kg_jobs; undo with kg_rollback(job_id)"}
    if backup_warning:
        out["warning"] = backup_warning
    return _json.dumps(out)


@mcp.tool()
def kg_add_videos(graph: str, steps: str, title: str = "",
                  backup: str = DEFAULT_BACKUP,
                  workdir: str = "") -> str:
    """Add VIDEOS to a graph: download their audio, transcribe, correct, translate to
    English, build the skills, then insert them — in that order.

    `steps` is a JSON array of {"name": "...", "cmd": "shell command",
    "inserts": ["corpus__x.md", "skills__x.md"]}. The command does the work (the
    download/ASR machinery lives in scripts, not in this server); `inserts` names the
    documents the step adds so the run can be undone later with kg_rollback.
    `backup` default = the kg_graph_backup.sh shipped with the package, run for THIS graph
    before and after the job; "" (or "none") skips it; any other string runs as your own
    command. A backup that fails is recorded in the job, never swallowed.
    Returns a job id immediately — poll it with kg_jobs.
    """
    return _start_staged_job(graph, steps, title, "videos", backup, workdir)


@mcp.tool()
def kg_add_sites(graph: str, steps: str, title: str = "",
                 backup: str = DEFAULT_BACKUP,
                 workdir: str = "") -> str:
    """Add WEBSITES to a graph: crawl the pages, collect what users say about them,
    build one skill per site, then insert them — in that order.

    Same contract as kg_add_videos: `steps` is an ordered JSON array of
    {"name", "cmd", "inserts"}; the shell commands do the work, the server only
    tracks the run; `backup` defaults to the shipped kg_graph_backup.sh for this graph
    ("" skips it). Returns a job id immediately — poll it with kg_jobs.
    """
    return _start_staged_job(graph, steps, title, "sites", backup, workdir)


@mcp.tool()
async def kg_rollback(job_id: str, dry_run: bool = False) -> str:
    """Undo a kg_ingest job: delete exactly the graph documents that run inserted.

    Uses the job manifest (the documents declared as `inserts` plus the ids the graph
    actually stored). Nothing else in the graph is touched. `dry_run=true` lists what would go;
    a real run reports the same file names (`would_remove` / `removed`), plus `docs` with the
    doc id and status of each deletion.
    """
    import json as _json
    import glob as _glob
    jdir = f"{JOBS}/{job_id}"
    jjson = f"{jdir}/job.json"
    if not os.path.exists(jjson):
        return _json.dumps({"error": f"no such job: {job_id}"})
    d = _json.load(open(jjson))
    names = set()
    for n in (d.get("manifest") or []):
        names.add(os.path.basename(n))
    # also every stored file of this run: the graph keeps file_name in doc_status
    graph = d.get("graph")
    stored = []
    if graph:
        try:
            root = _graph_dir(graph)
            ds = _json.load(open(os.path.join(root, "graph", "kv_store_doc_status.json")))
            for v in ds.values():
                fn = os.path.basename(v.get("file_path") or "")
                if fn in names:
                    stored.append(fn)
        except Exception:
            pass
    names |= set(stored)
    if not names:
        return _json.dumps({"job": job_id, "removed": 0,
                            "note": "manifest empty — nothing was inserted by this run"})
    if dry_run:
        return _json.dumps({"job": job_id, "would_remove": sorted(names), "count": len(names)})
    removed = await _delete_by_file_name(graph, names)
    return _json.dumps({"job": job_id,
                        "removed": [r["file"] for r in removed],
                        "count": len(removed),
                        "docs": removed})


@mcp.tool()
def kg_jobs(limit: int = 10, include_stale: bool = False) -> str:
    """State of background jobs: book runs (kg_add_book) and staged ingests (kg_add_videos/kg_add_sites).

    Book jobs report `pdf` + `docs_inserted`; staged jobs report `kind`, `step`/`step_index`,
    `steps_total`, `manifest` (the documents the run declared — exactly what kg_rollback deletes)
    and `warnings` (e.g. a backup that failed). Jobs whose graph no longer exists are marked
    "(graph deleted)" and skipped unless include_stale=True.
    """
    import glob as _glob
    out = []
    for jd in sorted(_glob.glob(f"{JOBS}/*"), key=os.path.getmtime, reverse=True):
        jp = f"{jd}/job.json"
        d = json.load(open(jp)) if os.path.exists(jp) else {}
        graph = d.get("graph")
        stale = bool(graph) and _meta_of(graph)[0] is None
        if stale and not include_stale:
            continue
        if len(out) >= limit:
            break
        pid = d.get("pid")
        state = d.get("state", "unknown")
        staged = bool(d.get("steps_total") or d.get("kind"))
        if state in ("running", "queued", "starting") and pid and not os.path.exists(f"/proc/{pid}"):
            state = ("dead (process gone; re-run the staged job or roll it back)"
                     if staged else "dead (process gone; re-run kg_add_book to resume)")
        rec = {"job": os.path.basename(jd), "state": state,
               "kind": d.get("kind") or ("book" if d.get("pdf") else "unknown"),
               "graph": graph, "title": d.get("title"),
               "started": d.get("started"), "finished": d.get("finished"),
               "error": d.get("error"),
               "log": f"{jd}/job.log" if os.path.exists(f"{jd}/job.log") else None}
        if stale:
            rec["note"] = "graph no longer exists — this job can no longer be rolled back"
        if d.get("pdf"):
            rec["pdf"] = os.path.basename(d["pdf"])
            rec["docs_inserted"] = d.get("docs_inserted")
        if staged:
            rec["step"] = d.get("step")
            rec["step_index"] = d.get("step_index")
            rec["steps_total"] = d.get("steps_total")
            rec["manifest"] = [os.path.basename(x) for x in (d.get("manifest") or [])]
            if d.get("warnings"):
                rec["warnings"] = d["warnings"]
        out.append(rec)
    return json.dumps(out, indent=1, ensure_ascii=False)


@mcp.tool()
async def kg_add_repo(graph: str, repo_path: str, language: str = "en") -> str:
    """For a cloned repo: write arc42 doc (if missing), skill-ify its reference PDFs, insert all into the graph."""
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        return json.dumps({"error": "no such repo dir"})
    repo = os.path.basename(repo_path)
    docs_added = []
    # 1) arc42 if missing -> delegate to arc42 script
    arc = f"{repo_path}/docs/arc42/arc42.md"
    if not os.path.exists(arc):
        subprocess.run([_pipeline_python(), _script("make_arc42.py"), repo_path], capture_output=True, text=True, timeout=7200)
    indir = _inputs_dir(graph)
    if os.path.exists(arc):
        os.makedirs(indir, exist_ok=True)
        shutil.copy(arc, f"{indir}/arc42__{repo}__arc42.md")
        docs_added.append(f"arc42__{repo}__arc42.md")
    # 2) skill-ify reference PDFs
    for pdf in glob.glob(f"{repo_path}/docs/references/*.pdf"):
        slug = os.path.splitext(os.path.basename(pdf))[0].lower()[:40]
        skill_dir = f"{LIBRARY}/skills/all-skills/{slug}"
        if not os.path.exists(f"{skill_dir}/SKILL.md"):
            ed = f"/tmp/extracted/{slug}"; os.makedirs(ed, exist_ok=True)
            subprocess.run(["pdftotext", pdf, f"{ed}/full_text.txt"], timeout=600)
            subprocess.run([_pipeline_python(), _script("generate_skill.py"), ed, skill_dir, language], capture_output=True, text=True, timeout=7200)
        if os.path.exists(f"{skill_dir}/SKILL.md"):
            for f in [f"{skill_dir}/SKILL.md"] + glob.glob(f"{skill_dir}/*.md") + glob.glob(f"{skill_dir}/chapters/*.md"):
                dst = f"{indir}/{slug}__{os.path.basename(f)}"
                shutil.copy(f, dst)
                if os.path.basename(dst) not in docs_added: docs_added.append(os.path.basename(dst))
    files = [f"{indir}/{d}" for d in docs_added if os.path.exists(f"{indir}/{d}")]
    inserted = await _insert_files(graph, files)
    return json.dumps({"ok": True, "repo": repo, "docs": docs_added, "inserted": inserted})

@mcp.tool()
async def kg_add_markdown(graph: str, markdown: str = "", md_path: str = "", doc_name: str = "", replace: bool = False) -> str:
    """Insert markdown straight into a graph - no PDF/book/repo pipeline.

    markdown: inline markdown text (notes written directly)
    md_path:  a .md file, or a directory (every *.md inside, recursively)
    doc_name: stored document name (default: the file's own name, or 'note' for inline text)
    replace:  delete a document already stored under the same name first. Without it an
              identical re-insert is a no-op (dedup is by content) and changed content
              leaves the old copy in the graph.
    Returns {"ok": true, "files": [...], "docs_inserted": n, "removed": [...]}.
    """
    if _meta_of(graph)[0] is None:
        return json.dumps({"error": f"graph '{graph}' does not exist. Use kg_create or kg_register first."})
    jobs = []  # (stored file name, markdown text)
    if markdown.strip():
        nm = _safe_name(doc_name or "note")
        jobs.append((nm if nm.endswith(".md") else nm + ".md", markdown))
    elif md_path:
        p = os.path.abspath(os.path.expanduser(md_path))
        if os.path.isdir(p):
            root = _safe_name(os.path.basename(p.rstrip("/")))
            for f in sorted(glob.glob(f"{p}/**/*.md", recursive=True)):
                rel = _safe_name(os.path.relpath(f, p).replace("/", "__"))
                jobs.append((f"{root}__{rel}", open(f, encoding="utf-8").read()))
        elif os.path.isfile(p):
            nm = _safe_name(doc_name) if doc_name else os.path.basename(p)
            jobs.append((nm if nm.endswith(".md") else nm + ".md", open(p, encoding="utf-8").read()))
        else:
            return json.dumps({"error": f"no such md file or dir: {md_path}"})
    if not jobs:
        return json.dumps({"error": "give markdown text or md_path"})
    indir = _inputs_dir(graph)
    os.makedirs(indir, exist_ok=True)
    files = []
    for name, text in jobs:
        dst = f"{indir}/{name}"
        open(dst, "w", encoding="utf-8").write(text)
        files.append(dst)
    removed = await _delete_by_file_name(graph, {os.path.basename(f) for f in files}) if replace else []
    inserted = await _insert_files(graph, files)
    return json.dumps({"ok": True, "files": [os.path.basename(f) for f in files],
                       "docs_inserted": inserted, "removed": [r["file"] for r in removed]})

@mcp.tool()
async def kg_query(graph: str, question: str, mode: str = "naive", max_chars: int = 12000) -> str:
    """Retrieve context from the graph and return it raw — NO answering LLM, so it is cheap and fast.

    Use it to read/ground yourself (or to feed another model), then answer your own way. ``kg_ask``
    is the sibling that also generates a cited answer.
    mode: naive (vector only) | hybrid (keyword extraction + vector, costs one LLM call)
    | mix (entities + relations) | local | global.
    """
    rag = await _load_graph(graph)
    ctx = await rag.aquery(question, param=__import__("lightrag").QueryParam(mode=mode, only_need_context=True))
    ctx = str(ctx or "")
    return json.dumps({"graph": graph, "mode": mode, "context_chars": len(ctx),
                       "context": ctx[:max_chars]}, ensure_ascii=False)


@mcp.tool()
async def kg_ask(graph: str, question: str, mode: str = "naive") -> str:
    """Ask the graph and get an ANSWER (deepseek-chat) with [ref N] citations. Its sibling ``kg_query``
    returns the retrieved context only, with no answer LLM.

    mode: naive (vector only, cheap) | hybrid (needs LLM keyword extraction). Graph stays warm 15 min.
    """
    rag = await _load_graph(graph)
    qp = __import__("lightrag").QueryParam(mode="naive" if mode == "naive" else "hybrid")
    return str(await rag.aquery(question, param=qp))


@mcp.tool()
def kg_delete(graph: str, confirm: bool = False, delete_data: bool = False) -> str:
    """Delete a graph (registry entry + its data). Requires confirm=True.

    For a registered graph whose data lives elsewhere (meta root, e.g. kg_nav -> ~/lightrag/ins-nav),
    only the registry entry is removed unless delete_data=True — the data directory is never deleted
    silently.
    """
    import shutil as _sh
    gd = f"{BASE}/{graph}"
    if not os.path.isdir(gd):
        return json.dumps({"error": f"graph '{graph}' does not exist"})
    meta = _meta_of(graph)[0] or {}
    data = _graph_dir(graph, meta)
    external = os.path.abspath(data) != os.path.abspath(gd)
    if not confirm:
        return json.dumps({"error": "set confirm=true to actually delete", "would_delete": gd,
                           "data_dir": data if external else None,
                           "note": "data lives outside the registry dir; pass delete_data=true to remove it too" if external else None})
    _GRAPHS.pop(graph, None); _LAST_USE.pop(graph, None)
    _sh.rmtree(gd)
    out = {"ok": True, "deleted": gd}
    if external:
        out["data_left_in_place"] = data
        out["note"] = "registry entry removed; pass delete_data=true to also remove the data dir"
        if delete_data:
            _sh.rmtree(data); out["deleted_data"] = data
    if (not external) or delete_data:
        # A job record for a graph that no longer exists is dead weight: its manifest can never be
        # rolled back and kg_jobs would keep listing it. Drop it together with the data.
        removed_jobs = []
        for jd in glob.glob(f"{JOBS}/*"):
            try:
                j = json.load(open(f"{jd}/job.json"))
            except Exception:
                continue
            if j.get("graph") == graph:
                _sh.rmtree(jd, ignore_errors=True)
                removed_jobs.append(os.path.basename(jd))
        if removed_jobs:
            out["removed_jobs"] = removed_jobs
    return json.dumps(out)


@mcp.tool()
def kg_register(graph: str, root: str, language: str = "en", force: bool = False) -> str:
    """Register an EXISTING LightRAG graph kept outside ~/lightrag/kg (e.g. ~/lightrag/ins-nav) so the
    MCP server can query and extend it in place. Expects <root>/graph/ and <root>/inputs/; writes
    <BASE>/<graph>/meta.json with "root": root. No symlinks, no data copying."""
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(f"{root}/graph"):
        return json.dumps({"error": f"no graph storage at {root}/graph"})
    mp = f"{BASE}/{graph}/meta.json"
    if os.path.exists(mp) and not force:
        return json.dumps({"error": f"'{graph}' is already registered (meta.json exists)",
                           "hint": "pass force=true to repoint it", "meta": mp})
    os.makedirs(f"{BASE}/{graph}", exist_ok=True)
    os.makedirs(f"{root}/inputs", exist_ok=True)
    meta = {"name": graph, "language": language, "embedding_model": _model_for(language),
            "created": time.strftime("%Y-%m-%d"), "root": root,
            "note": "registers an existing graph in place (kg_register)"}
    json.dump(meta, open(mp, "w"), indent=1)
    _GRAPHS.pop(graph, None); _LAST_USE.pop(graph, None)
    return json.dumps({"ok": True, "graph": graph, "root": root,
                       "docs": len(glob.glob(f"{root}/inputs/*.md"))})

@mcp.tool()
def kg_setup(models: str = "both") -> str:
    """Pre-download embedding models + verify environment. models: 'both' | 'e5' | 'heidari'. Returns status of each component."""
    out = {"models": {}, "env": {}}
    want = {"both": ["intfloat/multilingual-e5-small", "heydariAI/persian-embeddings"],
            "e5": ["intfloat/multilingual-e5-small"],
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
    out["onnx"] = {}
    for m in want:
        sess = _get_or_export_onnx(m)
        out["onnx"][m] = "ready" if sess else "export failed (torch fallback will be used)"
    return json.dumps(out, indent=1)

def main():
    """Console entry point: ``kg-mcp`` (stdio MCP server)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
