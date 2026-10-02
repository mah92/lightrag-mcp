#!/usr/bin/env python3
"""Query a LightRAG graph and print the retrieved context (no answering LLM).

  kg-query "question" [-g kg_nav] [--root PATH] [-m naive|hybrid|mix] [--max-chars 14000]

naive = vector search only (cheap, no keyword LLM), hybrid = + LLM keyword extraction,
mix = entities + relations. ``--root`` queries a bare graph directory that was never
registered (kg_register does that properly).
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kg_common import build_rag, graph_dir, meta_of  # noqa: E402


async def fetch(graph, question, mode, root=None):
    from lightrag import QueryParam
    meta = meta_of(graph)[0] if graph else None
    if graph and meta is None:
        raise SystemExit(f"graph '{graph}' is not registered (kg_register / kg_create)")
    rag, wd = build_rag(name=graph, root=root, meta=meta)
    await rag.initialize_storages()
    return wd, await rag.aquery(question, param=QueryParam(mode=mode, only_need_context=True))


def main():
    ap = argparse.ArgumentParser(description="Retrieve context from a LightRAG graph")
    ap.add_argument("question")
    ap.add_argument("-g", "--graph", default="kg_nav")
    ap.add_argument("--root", help="graph data dir (instead of a registered graph name)")
    ap.add_argument("-m", "--mode", default="naive", choices=["naive", "hybrid", "mix", "local", "global"])
    ap.add_argument("--max-chars", type=int, default=14000)
    a = ap.parse_args()
    wd, ctx = asyncio.run(fetch(a.graph if not a.root else None, a.question, a.mode, a.root))
    ctx = ctx or ""
    print(f"graph_dir: {wd}")
    print(f"=== CONTEXT ({len(ctx)} chars) ===")
    print(ctx[: a.max_chars])


if __name__ == "__main__":
    main()
