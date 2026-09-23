#!/usr/bin/env python3
"""Answer a question from a LightRAG graph with a persona prompt — no profile names baked in.

  kg-answer "question" -g kg_nav [--persona-file FILE] [--name "Askar"] [--max-tokens 2000]
                       [--send --chat <id>] [--send-platform bale]

The persona is DATA, not code: pass a text file (a profile keeps its own, e.g.
~/.hermes/profiles/<profile>/persona.txt) or --system "..." inline. Without either, a neutral
"answer from the context only, cite [ref N]" prompt is used.
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kg_common import build_rag, env_value, meta_of  # noqa: E402

NEUTRAL = ("Answer using ONLY the context below. Cite the sources you use as [ref N]. "
           "If the context is insufficient, say so plainly.")


def send_bale(text, chat_id, chunk=3800):
    import json
    import subprocess
    token = env_value("BALE_BOT_TOKEN", "")
    if not token:
        raise SystemExit("no BALE_BOT_TOKEN in ~/.hermes/.env")
    for i in range(0, len(text), chunk):
        cmd = ["curl", "-s", "-X", "POST", f"https://tapi.bale.ai/bot{token}/sendMessage",
               "-F", f"chat_id={chat_id}", "-F", f"text={text[i:i + chunk]}"]
        resp = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout)
        if not resp.get("ok"):
            raise SystemExit(f"bale send failed: {resp}")


async def answer(graph, question, system, max_tokens, context_chars, mode, root=None):
    from lightrag import QueryParam
    meta = meta_of(graph)[0] if graph else None
    if graph and meta is None:
        raise SystemExit(f"graph '{graph}' is not registered (kg_register / kg_create)")
    rag, _ = build_rag(name=graph, root=root, meta=meta)
    await rag.initialize_storages()
    ctx = await rag.aquery(question, param=QueryParam(mode=mode, only_need_context=True)) or ""
    prompt = f"{system}\n\nCONTEXT:\n{ctx[:context_chars]}\n\nQUESTION: {question}"
    from lightrag.llm.openai import openai_complete_if_cache
    return await openai_complete_if_cache(
        "deepseek-chat", prompt, base_url="https://api.deepseek.com/v1",
        api_key=env_value("DEEPSEEK_API_KEY", ""), max_tokens=max_tokens)


def main():
    ap = argparse.ArgumentParser(description="Answer from a LightRAG graph with a persona prompt")
    ap.add_argument("question")
    ap.add_argument("-g", "--graph", default="kg_nav")
    ap.add_argument("--root", help="graph data dir (instead of a registered graph name)")
    ap.add_argument("--persona-file", help="text file holding the persona/system prompt")
    ap.add_argument("--system", help="persona prompt inline (instead of --persona-file)")
    ap.add_argument("--name", default=None, help="header label when sending to a chat")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--context-chars", type=int, default=12000)
    ap.add_argument("-m", "--mode", default="naive", choices=["naive", "hybrid", "mix", "local", "global"])
    ap.add_argument("--send", action="store_true", help="send the answer to a chat")
    ap.add_argument("--chat", help="chat id for --send")
    ap.add_argument("--send-platform", default="bale", choices=["bale"])
    a = ap.parse_args()

    if a.system:
        system = a.system
    elif a.persona_file:
        system = open(os.path.expanduser(a.persona_file), encoding="utf-8").read().strip()
    else:
        system = NEUTRAL
    out = asyncio.run(answer(a.graph if not a.root else None, a.question, system,
                            a.max_tokens, a.context_chars, a.mode, a.root))
    print(out)
    if a.send:
        if not a.chat:
            raise SystemExit("--send needs --chat <id>")
        header = f"{a.name}:\n\n" if a.name else ""
        send_bale(header + (out or "no answer produced"), a.chat)
        print(f"\n[sent to {a.send_platform} chat {a.chat}]")


if __name__ == "__main__":
    main()
