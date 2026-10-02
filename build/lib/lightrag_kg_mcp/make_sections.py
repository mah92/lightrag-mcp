#!/usr/bin/env python3
"""Derive section (chapter) boundaries for a book PDF -> <extract_dir>/sections.json

Used by generate_skill.py when present. Strategy, in order:
  1. PDF bookmarks, level 1, with human-readable titles  (e.g. Siouris)
  2. PDF bookmarks, level 2 entries that look like chapters (e.g. Li "Chapter 4\r...")
  3. PDF bookmarks, level 1 with machine titles -> auto-title from body text (e.g. Titterton "13587_04a")
  4. printed ToC lines in the body text ("10. Inertial Navigation System Alignment .... 277")
  5. none -> sections.json = []  (short documents need no chapter files)

Output: [{"n":1,"title":"introduction","page":16,"start":12345,"end":67890,"chars":N}, ...]
Offsets index the pdftotext output of the same PDF (pages separated by \f).

Usage: make_sections.py <pdf> <extract_dir>
"""
import json
import os
import re
import sys

import pymupdf  # pymupdf ships with the hermes venv


def page_offsets(pages):
    offs, acc = [], 0
    for p in pages:
        offs.append(acc)
        acc += len(p) + 1          # +1 for the \f separator
    return offs


def slug(s):
    s = re.sub(r"[^A-Za-z0-9]+", "-", s.lower()).strip("-")
    return (s or "section")[:50]


def clean_title(t):
    t = t.replace("\r", " ").replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def is_machine_title(t):
    return bool(re.fullmatch(r"[\w]*\d{2,}[\w_]*", t)) or bool(re.match(r"^\d+_[a-z]", t))


def first_body_line(text, limit=90):
    for line in text[:2000].splitlines():
        line = clean_title(line)
        if 8 <= len(line) <= limit and not re.match(r"^[\d\s.]+$", line):
            return line
    return "section"


def looks_chapter(t):
    return bool(re.match(r"^(chapter|chapitre|kapitel)\s+\d+", t, re.I)) or bool(
        re.match(r"^\d{1,2}[\.\s]", t)
    )


def main():
    pdf, edir = sys.argv[1], sys.argv[2]
    raw = open(f"{edir}/full_text.txt", encoding="utf-8", errors="ignore").read()
    pages = raw.split("\f")
    offs = page_offsets(pages)
    total = len(raw)
    doc = pymupdf.open(pdf)
    toc = doc.get_toc()

    def start_of(pdf_page):                      # get_toc pages are 1-based
        i = max(0, min(len(offs) - 1, pdf_page - 1))
        return offs[i]

    cands = []
    l1 = [t for t in toc if t[0] == 1 and t[2] >= 1]      # page 0 / negative = front-matter stub
    l2 = [t for t in toc if t[0] == 2 and t[2] >= 1]
    nice_l1 = [t for t in l1 if not is_machine_title(clean_title(t[1]))]
    chap_l2 = [t for t in l2 if looks_chapter(clean_title(t[1]))]

    if len(nice_l1) >= 5:
        cands = [(clean_title(t[1]), t[2]) for t in nice_l1]
        how = "bookmarks-level1"
    elif len(chap_l2) >= 5:
        cands = [(clean_title(t[1]), t[2]) for t in chap_l2]
        how = "bookmarks-level2-chapters"
    elif len(l1) >= 5:
        cands = [(clean_title(t[1]), t[2]) for t in l1]
        how = "bookmarks-level1-machine"

    sections = []
    if cands:
        drop = re.compile(r"^(contents|index|cover|title|preface|foreword|about the author)$", re.I)
        cands = [(t, p) for t, p in cands if not drop.match(t)]
        cands.sort(key=lambda x: x[1])
        for i, (title, page) in enumerate(cands):
            start = start_of(page)
            end = start_of(cands[i + 1][1]) if i + 1 < len(cands) else total
            if end - start < 500:
                continue
            body = raw[start:end]
            if is_machine_title(title) or not title:
                title = first_body_line(body)
            sections.append({"title": title, "page": page, "start": start, "end": end})
    else:
        how = "none"
        # fallback: printed ToC lines like "10. Inertial Navigation System Alignment ... 277"
        for m in re.finditer(r"(?m)^\s*(\d{1,2})\.\s+([A-Z][^.\n]{4,70}?)\s*\.{3,}\s*(\d{1,4})\s*$", raw[:40000]):
            sections.append({"title": f"{m.group(1)}. {clean_title(m.group(2))}", "page": int(m.group(3))})

    # merge runt sections into the previous one
    merged = []
    for s in sections:
        if merged and s["end"] - s["start"] < 3000:
            merged[-1]["end"] = s["end"]
            continue
        merged.append(dict(s))
    for i, s in enumerate(merged):
        s["n"] = i + 1
        s["slug"] = slug(s["title"])
        s["chars"] = s["end"] - s["start"]
        s["raw_title"] = s.pop("title")

    out = merged[:24]
    json.dump({"how": how, "sections": out}, open(f"{edir}/sections.json", "w"), indent=1, ensure_ascii=False)
    print(f"{os.path.basename(pdf)}: {how}, {len(out)} sections")
    for s in out:
        print(f"  ch{s['n']:02d} p{s['page']:>3} {s['chars']:>7} chars  {s['slug']}")


if __name__ == "__main__":
    main()
