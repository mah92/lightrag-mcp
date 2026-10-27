#!/usr/bin/env python3
"""DeepSeek-OCR runner (Persian documents).

pdf_craft drives DeepSeek-OCR page by page: PDF -> markdown (with assets and footnotes).
Run it with an interpreter that has pdf_craft, e.g. the `vits2` conda env:

    <python-with-pdf_craft> deepseek_ocr.py input.pdf output.md [models_cache_dir]

Model weights: DeepSeek-OCR, fetched by pdf_craft on first use (several GB) unless
`models_cache_path` points at an existing cache. Documents in Persian are ONLY handled by this
engine - Unlimited-OCR does not work on Persian text.
"""
import os
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: deepseek_ocr.py INPUT.pdf OUT.md [models_cache_dir]", file=sys.stderr)
        return 2
    pdf, out = sys.argv[1], sys.argv[2]
    cache = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("DEEPSEEK_OCR_CACHE") or None
    from pdf_craft import transform_markdown

    kw = {}
    if cache:
        kw["models_cache_path"] = cache
    meter = transform_markdown(
        pdf_path=pdf,
        markdown_path=out,
        markdown_assets_path=out.rsplit(".", 1)[0] + "_assets",
        includes_footnotes=True,
        ocr_size="gundam",
        **kw,
    )
    print("ocr tokens:", getattr(meter, "total_tokens", "n/a"))
    print("wrote", out, os.path.getsize(out) if os.path.exists(out) else "MISSING")
    return 0 if os.path.exists(out) else 1


if __name__ == "__main__":
    sys.exit(main())
