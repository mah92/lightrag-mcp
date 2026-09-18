#!/usr/bin/env python3
"""Generate a book-to-skill style skill from an extracted text dir.
Usage: generate_skill.py <extract_dir> <skill_dir> <lang>
Creates SKILL.md + glossary.md + patterns.md + cheatsheet.md via deepseek-chat.
Non-reasoning-safe: logs finish_reason and rejects empty content."""
import os, sys, json, glob, re, time
import requests

DEEPSEEK_KEY = ""
for line in os.path.expanduser("~/.hermes/.env").readlines() if os.path.exists(os.path.expanduser("~/.hermes/.env")) else []:
    if line.startswith("DEEPSEEK_API_KEY="):
        DEEPSEEK_KEY = line.strip().split("=", 1)[1]

extract_dir, skill_dir, lang = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(skill_dir, exist_ok=True)
text = open(f"{extract_dir}/full_text.txt", encoding="utf-8", errors="ignore").read()
words = len(text.split())
print(f"corpus: {words} words")

def ask(prompt, max_tokens=4000):
    r = requests.post("https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": 0.3},
        timeout=600)
    d = r.json()
    c = d["choices"][0]
    content = c["message"].get("content") or ""
    print("finish:", c.get("finish_reason"), "len:", len(content))
    return content if len(content) > 200 else None

# section slices proportional to corpus
title_m = re.search(r"(?m)^\s*(.{10,120})$", text[:2000])
title = title_m.group(1).strip() if title_m else os.path.basename(extract_dir)

# 1) SKILL.md
p = f"""You are converting a technical document into an agent skill file.
Document title: {title}
Write a SKILL.md in YAML-frontmatter + markdown format:
---
name: {os.path.basename(skill_dir)}
description: "Knowledge base from \\"{title}\\". Use when applying its frameworks, methods, or referencing its concepts."
---
Then body (max 2500 chars): ## Core Frameworks & Mental Models (named frameworks, exact formulations, 3-6 bullets),
## Section Index (real sections/chapters found in the text), ## Topic Index (term -> section).
Synthesize; never copy verbatim; keep equations as plain text. Output ONLY the file content."""
c = ask(p)
if c: open(f"{skill_dir}/SKILL.md", "w").write(c)

# 2) supporting files
for kind, instruct in [
    ("glossary", "Alphabetical key terms, one line each: **Term** - definition. Max 40 terms."),
    ("patterns", "## Pattern Name blocks: When to use / How / Trade-offs. Max 8 patterns."),
    ("cheatsheet", "Decision rules, thresholds, trade-off table, tells & smells. Compact, max 1200 chars."),
]:
    p = f"""From this technical document ({title}), produce the {kind} for an agent skill. {instruct}
Synthesize; never copy verbatim. Output ONLY the file content (markdown)."""
    c = ask(p, max_tokens=2500)
    if c: open(f"{skill_dir}/{kind}.md", "w").write(c)

# 3) chapters for big corpora
if words > 15000:
    ch_dir = f"{skill_dir}/chapters"; os.makedirs(ch_dir, exist_ok=True)
    heads = [(m.start(), m.group(0).strip()) for m in re.finditer(r"(?m)^(Chapter\s+\d+.*|CHAPTER\s+\d+.*)$", text)][:24]
    if heads:
        for i, (off, h) in enumerate(heads):
            end = heads[i+1][0] if i+1 < len(heads) else len(text)
            chunk = text[off:min(end, off+120000)]
            p = f"""Summarize section "{h}" of the document {title} as a chapter file (max 1200 tokens):
## Core Idea (1-2 sentences), ## Frameworks Introduced, ## Key Concepts, ## Anti-patterns, ## Key Takeaways (3-5).
Synthesize; equations as text. Output ONLY the file content."""
            c = ask(p)
            if c:
                slug = re.sub(r"[^a-z0-9]+", "-", h.lower())[:50].strip("-")
                open(f"{ch_dir}/ch{i+1:02d}-{slug}.md", "w").write(c)

files = sorted(os.path.relpath(f, skill_dir) for f in glob.glob(f"{skill_dir}/**/*.md", recursive=True))
print("generated:", files)
