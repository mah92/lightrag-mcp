#!/usr/bin/env python3
"""OCR helpers for the lightrag-kg server: PDF/image -> markdown.

Two backends, both local:

  * ``unlimited`` (default) - Baidu Unlimited-OCR, the box's primary OCR engine.
        ~/AI/Unlimited-OCR/venv/bin/python ~/AI/Unlimited-OCR/ocr.py <pdf> -o <dir> --mode base
        override the install root with OCR_UNLIMITED_DIR.
  * ``gemma`` - the local Gemma 4 E2B llama.cpp server (vision/mmproj), OpenAI-compatible API.
        override the endpoint with LIGHTRAG_OCR_GEMMA_URL (default http://127.0.0.1:8081/v1),
        model with LIGHTRAG_OCR_GEMMA_MODEL (default gemma-4-e2b).

CLI:  python ocr_to_md.py input.pdf -o out.md [--backend unlimited|gemma]
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

UNLIMITED_DIR = os.environ.get("OCR_UNLIMITED_DIR", os.path.expanduser("~/AI/Unlimited-OCR"))
GEMMA_URL = os.environ.get("LIGHTRAG_OCR_GEMMA_URL", "http://127.0.0.1:8081/v1")
GEMMA_MODEL = os.environ.get("LIGHTRAG_OCR_GEMMA_MODEL", "gemma-4-e2b")

GEMMA_PROMPT = (
    "Extract ALL text from this document image as clean Markdown. Preserve headings, lists, "
    "tables and numbers exactly. Do not translate, do not summarise, do not comment. "
    "Output only the markdown."
)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _pages(path, out_dir, dpi=150):
    """Render a PDF to PNGs (or normalise a single image); returns page file paths."""
    if path.lower().endswith(".pdf"):
        import fitz  # pymupdf
        doc = fitz.open(path)
        files = []
        for i in range(doc.page_count):
            p = os.path.join(out_dir, f"page_{i + 1:04d}.png")
            doc[i].get_pixmap(dpi=dpi).save(p)
            files.append(p)
        return files
    return [path]


def _gemma_page(png):
    with open(png, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": GEMMA_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        ]}],
        "max_tokens": 4096,
        "temperature": 0,
    }
    req = urllib.request.Request(
        GEMMA_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    return (data["choices"][0]["message"].get("content") or "").strip()


def _unlimited(path):
    ocr_py = os.path.join(UNLIMITED_DIR, "ocr.py")
    venv_py = os.path.join(UNLIMITED_DIR, "venv", "bin", "python")
    if not (os.path.exists(ocr_py) and os.path.exists(venv_py)):
        raise RuntimeError(
            f"Unlimited-OCR not found under {UNLIMITED_DIR} (need ocr.py + venv/bin/python); "
            "set OCR_UNLIMITED_DIR or use backend='gemma'"
        )
    out = tempfile.mkdtemp(prefix="kg_ocr_")
    r = subprocess.run([venv_py, ocr_py, path, "-o", out, "--mode", "base"],
                       capture_output=True, text=True, timeout=7200, cwd=UNLIMITED_DIR)
    md = os.path.join(out, "result.md")
    if not os.path.exists(md):
        raise RuntimeError("Unlimited-OCR produced no result.md: " + (r.stderr or r.stdout)[-500:])
    with open(md, encoding="utf-8") as fh:
        return fh.read()


def ocr_to_markdown(path, backend=None, tmpdir=None):
    """OCR a PDF or image and return markdown text. backend: 'unlimited' | 'gemma'."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    backend = (backend or os.environ.get("OCR_BACKEND") or "unlimited").lower()
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="kg_ocr_pages_")
    if backend == "unlimited":
        return _unlimited(path).strip()
    if backend == "gemma":
        parts = []
        for png in _pages(path, tmpdir):
            txt = _gemma_page(png)
            if txt:
                parts.append(txt)
        if not parts:
            raise RuntimeError("Gemma OCR returned no text (is `gemma start` running?)")
        return "\n\n---\n\n".join(parts)
    raise ValueError(f"unknown OCR backend: {backend}")


def is_document(path):
    p = path.lower()
    return p.endswith(".pdf") or p.endswith(IMAGE_EXT)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    md = ocr_to_markdown(a.input, a.backend)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {a.out} ({len(md)} chars)")


if __name__ == "__main__":
    main()
