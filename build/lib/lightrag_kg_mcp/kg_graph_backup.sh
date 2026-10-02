#!/usr/bin/env bash
# Back up LightRAG knowledge graphs so a bad write can be undone (kg_rollback needs a graph
# that still exists; this is the belt to that suspenders).
#
#   kg_graph_backup.sh                 # every graph under ~/lightrag/kg
#   kg_graph_backup.sh kg_tajer        # only the named graphs (what the staged ingest uses)
#
# Keeps the 3 newest archives + checksums per graph. Prints one line per graph; exits non-zero
# if any graph failed, so the caller (kg_ingest.py) records the failure instead of assuming a
# backup exists.
set -euo pipefail

SRC="${LIGHTRAG_KG_DIR:-$HOME/lightrag/kg}"
DST="${LIGHTRAG_KG_BACKUP_DIR:-$HOME/backups/kg}"
mkdir -p "$DST"
ts=$(date +%Y%m%d_%H%M)
shopt -s nullglob

if [ "$#" -gt 0 ]; then
  graphs=("$@")
else
  graphs=()
  for g in "$SRC"/*/; do graphs+=("$(basename "$g")"); done
fi

if [ "${#graphs[@]}" -eq 0 ]; then
  echo "no graphs under $SRC — nothing to back up"
  exit 0
fi

fail=0
for name in "${graphs[@]}"; do
  if [ ! -d "$SRC/$name" ]; then
    echo "SKIP $name: no such graph dir under $SRC"
    fail=1
    continue
  fi
  tar czf "$DST/${name}_${ts}.tar.gz" -C "$SRC" "$name"
  (cd "$DST" && sha256sum "${name}_${ts}.tar.gz" > "${name}_${ts}.sha256")
  if ! gzip -t "$DST/${name}_${ts}.tar.gz"; then
    echo "CORRUPT archive written for $name"
    fail=1
    continue
  fi
  { ls -1t "$DST/${name}_"*.tar.gz 2>/dev/null || true; } | tail -n +4 | while read -r f; do
    rm -f "$f" "${f%.tar.gz}.sha256"
  done
  echo "backed up $name -> $DST/${name}_${ts}.tar.gz"
done

[ "$fail" -eq 0 ] || exit 1
echo "$(date '+%F %T') graph backup ok -> $DST"
