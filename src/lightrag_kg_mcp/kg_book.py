#!/usr/bin/env python3
"""Book/document PDF -> skill markdown -> LightRAG graph, resumable and agent-free.

  kg-book <graph> <pdf> [--language en] [--skill-root DIR] [--extract-dir DIR] [--job-dir DIR]
                        [--skip-insert] [--job-json PATH]

Steps: pdftotext -> make_sections.py (chapter boundaries from the PDF's own bookmarks) ->
generate_skill.py (deepseek-chat) -> copy the markdown into <graph>/inputs/<slug>__*.md ->
insert only the files whose basename has no PROCESSED doc_status row.

Safe to re-run: an existing extraction, an already-generated skill, and already-inserted
documents are all skipped, so a crash costs only the missing pieces. This is also the job body
used by the MCP tool ``kg_add_book(..., background=True)``.
"""
import argparse
import asyncio
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kg_common import HERMES_PY, inputs_dir, meta_of, safe_name, script  # noqa: E402

DEFAULT_SKILL_ROOT = os.path.expanduser("~/Documents/INS-nav-library/skills/all-skills")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def job_update(path, **kw):
    if not path:
        return
    d = json.load(open(path)) if os.path.exists(path) else {}
    d.update(kw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(d, open(path, "w"), indent=1, ensure_ascii=False)


def pipeline_python():
    """The interpreter that can run the helpers: this one when it has lightrag + pymupdf, else the hermes venv."""
    import importlib.util
    if importlib.util.find_spec("lightrag") and importlib.util.find_spec("fitz"):
        return sys.executable
    return HERMES_PY


def extract_text(pdf, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    txt = f"{extract_dir}/full_text.txt"
    if os.path.exists(txt) and os.path.getsize(txt) > 0:
        log(f"extraction reused: {txt} ({os.path.getsize(txt)} bytes)")
    else:
        subprocess.run(["pdftotext", pdf, txt], check=True, timeout=3600)
        log(f"extracted -> {txt} ({os.path.getsize(txt)} bytes)")
    return txt


def make_sections(pdf, extract_dir, py):
    r = subprocess.run([py, script("make_sections.py"), pdf, extract_dir],
                       capture_output=True, text=True, timeout=3600)
    sec_p = f"{extract_dir}/sections.json"
    n = 0
    if os.path.exists(sec_p):
        try:
            n = len(json.load(open(sec_p)).get("sections", []))
        except Exception:
            n = 0
    log(f"make_sections: rc={r.returncode} sections={n}" + (f" | {r.stderr[-200:]}" if r.returncode else ""))
    return n


def generate_skill(extract_dir, skill_dir, language, py):
    sk = f"{skill_dir}/SKILL.md"
    if os.path.exists(sk) and os.path.getsize(sk) > 200:
        log(f"skill reused: {sk} ({os.path.getsize(sk)} bytes)")
        return
    r = subprocess.run([py, script("generate_skill.py"), extract_dir, skill_dir, language],
                       capture_output=True, text=True, timeout=14400)
    tail = (r.stdout or "")[-300:] + (r.stderr or "")[-300:]
    if not os.path.exists(sk) or os.path.getsize(sk) < 200:
        raise RuntimeError(f"skill generation failed (rc={r.returncode}): {tail}")
    log(f"skill generated: {skill_dir}")


def copy_markdown(skill_dir, graph, slug):
    indir = inputs_dir(graph)
    os.makedirs(indir, exist_ok=True)
    src = [f"{skill_dir}/SKILL.md"] + sorted(glob.glob(f"{skill_dir}/*.md")) + sorted(glob.glob(f"{skill_dir}/chapters/*.md"))
    out = []
    for f in src:
        dst = f"{indir}/{slug}__{os.path.basename(f)}"
        shutil.copy(f, dst)
        out.append(dst)
    log(f"copied {len(set(out))} markdown files into {indir}")
    return sorted(set(out))


async def insert_files(graph, files):
    from lightrag.base import DocStatus
    from kg_common import build_rag
    rag, wd = build_rag(graph)
    await rag.initialize_storages()
    docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    done = {os.path.basename(d.file_path or "") for d in docs.values()
            if getattr(d.status, "value", d.status) == "processed"}
    todo = [f for f in files if os.path.basename(f) not in done]
    log(f"insert: {len(todo)} new of {len(files)} files ({len(done)} already processed) in {wd}")
    for i in range(0, len(todo), 4):
        batch = todo[i:i + 4]
        await rag.ainsert([open(f, encoding="utf-8").read() for f in batch], file_paths=batch)
        log(f"inserted batch {i // 4 + 1}/{(len(todo) + 3) // 4}")
    return len(todo)


def main():
    ap = argparse.ArgumentParser(description="PDF -> skill markdown -> LightRAG graph (resumable)")
    ap.add_argument("graph")
    ap.add_argument("pdf")
    ap.add_argument("--language", default="en")
    ap.add_argument("--skill-root", default=DEFAULT_SKILL_ROOT,
                    help="where generated skills go (default: %(default)s)")
    ap.add_argument("--skill-dir", default=None, help="exact skill dir (default: <skill-root>/<slug>)")
    ap.add_argument("--extract-dir", default=None, help="default: /tmp/extracted/<slug>")
    ap.add_argument("--job-dir", default=None, help="job dir whose job.json gets state updates")
    ap.add_argument("--job-json", default=None, help="job.json path for state updates")
    ap.add_argument("--skip-insert", action="store_true", help="stop after copying markdown")
    a = ap.parse_args()

    job_json = a.job_json or (f"{a.job_dir}/job.json" if a.job_dir else None)
    pdf = os.path.abspath(os.path.expanduser(a.pdf))
    if not os.path.exists(pdf):
        raise SystemExit(f"no such pdf: {pdf}")
    if meta_of(a.graph)[0] is None:
        raise SystemExit(f"graph '{a.graph}' is not registered (kg_register / kg_create)")
    slug = safe_name(os.path.splitext(os.path.basename(pdf))[0].lower().replace(" ", "-"))[:40]
    extract_dir = a.extract_dir or f"/tmp/extracted/{slug}"
    skill_dir = a.skill_dir or f"{a.skill_root}/{slug}"
    py = pipeline_python()

    try:
        job_update(job_json, state="running", graph=a.graph, pdf=pdf, slug=slug, pid=os.getpid(),
                   started=time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"graph={a.graph} skill_dir={skill_dir} python={py}")
        extract_text(pdf, extract_dir)
        make_sections(pdf, extract_dir, py)
        generate_skill(extract_dir, skill_dir, a.language, py)
        files = copy_markdown(skill_dir, a.graph, slug)
        n = 0 if a.skip_insert else asyncio.run(insert_files(a.graph, files))
        job_update(job_json, state="done", docs_inserted=n, files=len(files),
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"DONE graph={a.graph} files={len(files)} docs_inserted={n}")
    except Exception as e:
        job_update(job_json, state="failed", error=str(e)[:500], finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
