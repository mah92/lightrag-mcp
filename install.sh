#!/usr/bin/env bash
# Independent install for lightrag-mcp: one venv, one package, one config snippet.
#
#   ./install.sh                # venv at ~/.hermes/lightrag-mcp-venv
#   LIGHTRAG_MCP_VENV=/opt/kg-venv ./install.sh
#
# Installs the MCP server AND the pipeline CLI (kg-query / kg-answer / kg-book) into the SAME
# interpreter — that is the whole point: a second venv without lightrag is how the graph tools
# silently broke before (kg_create worked, kg_ask did not).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${LIGHTRAG_MCP_VENV:-$HOME/.hermes/lightrag-mcp-venv}"
PY_BIN="${PYTHON:-python3.11}"

echo "==> repo:  $REPO_DIR"
echo "==> venv:  $VENV"
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
"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md
import lightrag, torch, sentence_transformers, fitz  # noqa: F401
import mcp.server.fastmcp  # noqa: F401
print("  lightrag", lightrag.__version__)
print("  mcp", md.version("mcp"), "| torch", torch.__version__, "| pymupdf ok")
EOF

cat <<EOF

Installed. Two more steps:

1) register the server with Hermes (add --profile <name> for another profile):

     hermes mcp add lightrag-kg --command $VENV/bin/kg-mcp

   and make sure DEEPSEEK_API_KEY is in ~/.hermes/.env

2) install the operating manual (skill), separately — it is knowledge, not code.
   The skill ships in the collection repo mah92/my-hermes-skills; install it from there
   (hermes skills install <owner>/<repo>/<skill>) or simply copy the folder:

     cp -r ~/my-hermes-skills/book-to-lightrag-pipeline ~/.hermes/skills/

Then build or register a graph:

     $VENV/bin/kg-book kg_mine ~/Documents/Books/some.pdf     # PDF -> graph
     $VENV/bin/kg-query "question" -g kg_mine                 # context only
     $VENV/bin/kg-answer "question" -g kg_mine                # answer with citations
EOF
