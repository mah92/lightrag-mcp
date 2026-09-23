#!/usr/bin/env bash
# Independent install / check for lightrag-mcp: one venv, one package, one config snippet.
#
#   ./install.sh                 # install (venv at ~/.hermes/lightrag-mcp-venv)
#   ./install.sh --check         # verify an existing install, change nothing (used by the skill)
#   LIGHTRAG_MCP_VENV=/opt/kg-venv ./install.sh
#
# Installs the MCP server AND the pipeline CLI (kg-query / kg-ask / kg-add-book / kg-mcp) into the
# SAME interpreter — that is the whole point: a second venv without lightrag is how the graph tools
# silently broke before (kg_create worked, kg_ask did not).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${LIGHTRAG_MCP_VENV:-$HOME/.hermes/lightrag-mcp-venv}"
PY_BIN="${PYTHON:-python3.11}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "==> repo:  $REPO_DIR"
echo "==> venv:  $VENV"

verify() {
  "$VENV/bin/python" - <<'EOF'
import importlib.metadata as md
import importlib.util as u
import lightrag, torch, sentence_transformers, mcp.server.fastmcp  # noqa: F401
ok_fitz = bool(u.find_spec("fitz"))
print("  lightrag", lightrag.__version__)
print("  mcp", md.version("mcp"), "| torch", torch.__version__, "| pymupdf", ok_fitz)
for mod in ("kg_query", "kg_ask", "kg_add_book", "kg_common"):
    if not u.find_spec("lightrag_kg_mcp." + mod):
        raise SystemExit(f"  MISSING module lightrag_kg_mcp.{mod} — package not installed (pip install -e .)")
print("  package modules ok")
if not ok_fitz:
    raise SystemExit("  pymupdf missing — `kg-add-book` cannot read PDF bookmarks")
EOF
  for s in kg-mcp kg-query kg-ask kg-add-book; do
    [ -x "$VENV/bin/$s" ] || { echo "  MISSING console script $s"; return 1; }
  done
  echo "  console scripts ok (kg-mcp kg-query kg-ask kg-add-book)"
  echo "  DEEPSEEK_API_KEY: $(grep -c '^DEEPSEEK_API_KEY=..*' "$HOME/.hermes/.env" 2>/dev/null || echo 0) non-empty entry in ~/.hermes/.env"
}

if [ "$CHECK_ONLY" = "1" ]; then
  [ -d "$VENV" ] || { echo "FAIL: no venv at $VENV"; echo "      (run ./install.sh, or point --check at the real one: LIGHTRAG_MCP_VENV=~/.hermes/mcp-venv ./install.sh --check)"; exit 1; }
  echo "==> verify only"
  verify && echo "OK: install looks healthy" || { echo "FAIL: see above"; exit 1; }
  exit 0
fi

command -v "$PY_BIN" >/dev/null || PY_BIN=python3
[ -d "$VENV" ] || "$PY_BIN" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip

# CPU-only torch first: the default PyPI wheel pulls ~2 GB of CUDA packages this box never uses.
if ! "$VENV/bin/python" -c "import torch" 2>/dev/null; then
  echo "==> torch (CPU wheel)"
  "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> lightrag-mcp + dependencies"
"$VENV/bin/pip" install -q -e "$REPO_DIR"

echo "==> verify (the check that would have caught the missing-lightrag outage)"
verify

cat <<EOF

Installed. Two more steps:

1) register the server with Hermes (add --profile <name> for another profile) — this edits
   config.yaml, so ask the user first:

     hermes mcp add lightrag-kg --command $VENV/bin/kg-mcp

   and make sure DEEPSEEK_API_KEY is in ~/.hermes/.env

2) install the operating manual (skill), separately — it is knowledge, not code.
   The skill ships in the collection repo mah92/my-hermes-skills; install it from there
   (hermes skills install <owner>/<repo>/<skill>) or simply copy the folder:

     cp -r ~/my-hermes-skills/book-to-lightrag-pipeline ~/.hermes/skills/

Then build or register a graph:

     $VENV/bin/kg-add-book kg_mine ~/Documents/Books/some.pdf  # PDF -> graph (long; see kg_jobs)
     $VENV/bin/kg-query "question" -g kg_mine                 # context only (no answer LLM)
     $VENV/bin/kg-ask   "question" -g kg_mine                 # answer with [ref N] citations

Re-verify any time with:  ./install.sh --check
EOF
