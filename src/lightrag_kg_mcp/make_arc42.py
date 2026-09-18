#!/usr/bin/env python3
"""Write an arc42 doc skeleton for a repo at <repo>/docs/arc42/arc42.md derived from code scan.
Usage: make_arc42.py <repo_path>  (deepseek-chat enriches the math section)"""
import os, sys, glob, re
import requests

DEEPSEEK_KEY = ""
for line in os.path.expanduser("~/.hermes/.env").readlines():
    if line.startswith("DEEPSEEK_API_KEY="):
        DEEPSEEK_KEY = line.strip().split("=", 1)[1]

repo = os.path.abspath(sys.argv[1])
name = os.path.basename(repo)
out_dir = f"{repo}/docs/arc42"; os.makedirs(out_dir, exist_ok=True)

# scan source files
srcs = []
for ext in ("*.py", "*.cpp", "*.h", "*.m", "*.cc", "*.hpp"):
    srcs += glob.glob(f"{repo}/**/{ext}", recursive=True)
srcs = srcs[:40]
snippets = []
for f in srcs:
    try:
        t = open(f, encoding="utf-8", errors="ignore").read()
        # pull state/update/cost-looking lines
        for kw in ("state", "predict", "update", "measure", "cost", "horizon", "ekf", "kalman", "x_dot", "Q", "R"):
            pass
        snippets.append(f"--- {os.path.relpath(f, repo)} ---\n" + t[:3000])
    except Exception:
        pass
blob = "\n".join(snippets)[:120000]

prompt = f"""Write an arc42 documentation file for the repository "{name}".
Source excerpts (may be truncated):
{blob}

Requirements:
- All 12 arc42 sections: 1 Introduction & Goals, 2 Constraints, 3 Context & Scope, 4 Solution Strategy,
  5 Building Blocks, 6 Runtime View, 7 Deployment, 8 Cross-cutting Concepts, 9 Decisions,
  10 Quality Requirements, 11 Risks, 12 Glossary.
- Inside section 5 add a subsection '## Mathematical Foundations' with the ACTUAL equations used by this
  code (state vector, kinematics, measurement model, update/cost equations) in LaTeX-style plain text,
  derived from the excerpts above. If excerpts are insufficient, say what is missing instead of inventing.
- English, synthesize, never copy code wholesale. Output ONLY the file content."""
r = requests.post("https://api.deepseek.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
    json={"model": "deepseek-chat",
          "messages": [{"role": "user", "content": prompt}],
          "max_tokens": 8000, "temperature": 0.3}, timeout=900)
d = r.json()
c = d["choices"][0]["message"].get("content") or ""
print("finish:", d["choices"][0].get("finish_reason"), "len:", len(c))
if len(c) > 1000:
    open(f"{out_dir}/arc42.md", "w").write(c)
    print("written", f"{out_dir}/arc42.md")
else:
    print("FAILED to produce content"); sys.exit(1)
