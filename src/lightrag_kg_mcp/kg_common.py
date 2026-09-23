#!/usr/bin/env python3
"""Shared helpers for the lightrag-kg MCP server and its pipeline scripts.

A graph is registered by ``~/lightrag/kg/<name>/meta.json``:

  * default            -> data in ``~/lightrag/kg/<name>/`` (``graph/`` + ``inputs/``)
  * ``"root": "<abs>"`` -> data in ``<root>/graph`` + ``<root>/inputs``   (written by ``kg_register``)

Path helpers here carry no heavy imports, so both the MCP venv (``server.py``) and the
hermes venv (pipeline scripts, which need lightrag + torch) can import this module.
"""
import os
import sys
import json
import glob
import re

# tiktoken downloads the o200k_base BPE file from an Azure blob that is unreachable from this box;
# point it at the local cache so LightRAG can build its tokenizer without network (server.py used to
# do this privately — every entry point needs it, so it lives here now).
if "TIKTOKEN_CACHE_DIR" not in os.environ:
    _tc = os.path.expanduser("~/.cache/tiktoken_cache")
    if os.path.isdir(_tc):
        os.environ["TIKTOKEN_CACHE_DIR"] = _tc

BASE = os.path.expanduser("~/lightrag/kg")
JOBS = os.path.expanduser("~/lightrag/jobs")
# Interpreter that has lightrag + torch + sentence-transformers (pipeline scripts run on it).
HERMES_PY = os.path.expanduser("~/.hermes/hermes-agent/venv/bin/python")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def env_value(key: str, default=None, path: str = "~/.hermes/.env"):
    """First non-empty value of KEY in ~/.hermes/.env (an empty shadowing ``KEY=`` line is skipped)."""
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        return default
    val = default
    for line in open(p, encoding="utf-8", errors="ignore"):
        if line.startswith(key + "="):
            v = line.strip().split("=", 1)[1].strip()
            if v:
                val = v
    return val


def meta_of(name: str):
    """(meta, path) for a registered graph; (None, path) when the registry entry is missing."""
    mp = f"{BASE}/{name}/meta.json"
    if not os.path.exists(mp):
        return None, mp
    try:
        return json.load(open(mp)), mp
    except Exception:
        return {}, mp


def graph_dir(name: str, meta=None) -> str:
    """Where a graph's DATA lives (``root`` from meta.json, else ``BASE/<name>``)."""
    if meta is None:
        meta = meta_of(name)[0] or {}
    root = meta.get("root")
    return os.path.abspath(os.path.expanduser(root)) if root else f"{BASE}/{name}"


def inputs_dir(name: str, meta=None) -> str:
    return f"{graph_dir(name, meta)}/inputs"


def script(name: str) -> str:
    """Pipeline helper by name: beside this module first, then the legacy ~/lightrag/kg_mcp copy."""
    for d in (_HERE, os.path.expanduser("~/lightrag/kg_mcp")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(_HERE, name)


def embedding_model(name: str, meta=None) -> str:
    """The embedding model pinned in meta.json (graphs built with an older model keep working)."""
    if meta is None:
        meta = meta_of(name)[0] or {}
    return meta.get("embedding_model") or "intfloat/multilingual-e5-small"


def e5_prefix(text: str) -> str:
    """e5 models need ``query:``/``passage:`` prefixes; short strings are queries."""
    return ("query: " if text.startswith("query:") or len(text) < 300 else "passage: ") + text


def load_st_model(model_name: str):
    """SentenceTransformer for a graph's embedding model.

    Retries against the local cache: a dropped HF metadata call (this box has no IPv6 route, so an
    AAAA-only endpoint answers "Network is unreachable") must not kill a query when the weights are
    already cached.
    """
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(model_name, device="cpu")
    except Exception:
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)


def embedding_dim(model) -> int:
    fn = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension")
    return int(fn())


def build_rag(name: str = None, root: str = None, meta=None, embedding_model_name: str = None,
              llm_max_async: int = 2, embed_max_async: int = 2):
    """Load a LightRAG instance with the graph's own embedding model (dim probed at runtime).

    Pass ``name`` for a registered graph, or ``root`` for a bare graph directory.
    Returns ``(rag, working_dir)``; call ``await rag.initialize_storages()`` next.
    """
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc
    import numpy as np

    if meta is None:
        meta = meta_of(name)[0] if name else {}
    wd = f"{graph_dir(name, meta)}/graph" if name else f"{os.path.abspath(os.path.expanduser(root))}/graph"
    model_name = embedding_model_name or (meta or {}).get("embedding_model") or "intfloat/multilingual-e5-small"
    model = load_st_model(model_name)
    dim = embedding_dim(model)

    async def embed(texts):
        import torch
        prefixed = [e5_prefix(t) for t in texts]
        with torch.no_grad():
            emb = model.encode(prefixed, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    async def llm(text, system_prompt=None, history_messages=[], **kw):
        kw.pop("hashing_kv", None)
        return await openai_complete_if_cache(
            "deepseek-chat", text, system_prompt=system_prompt, history_messages=history_messages,
            base_url="https://api.deepseek.com/v1", api_key=env_value("DEEPSEEK_API_KEY", ""), **kw)

    rag = LightRAG(working_dir=wd, llm_model_func=llm, llm_model_name="deepseek-chat",
                   llm_model_max_async=llm_max_async,
                   embedding_func=EmbeddingFunc(embedding_dim=dim, max_token_size=512, func=embed),
                   embedding_batch_num=16, embedding_func_max_async=embed_max_async)
    return rag, wd


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip()).strip("-.") or "note"


def job_new_id(prefix: str) -> str:
    import time
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def job_paths(job_id: str):
    d = f"{JOBS}/{job_id}"
    return d, f"{d}/job.json", f"{d}/job.log"
