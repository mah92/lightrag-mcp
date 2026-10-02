#!/usr/bin/env python3
"""Run an ingest job declared by the MCP tool kg_ingest.

The MCP stays thin: it only stores the spec and starts this runner. The spec is an
ordered list of steps (collect -> ASR -> correct -> English -> skill -> graph), so
the volatile machinery (yt-dlp, proxies, ASR models, per-site quirks) lives in the
scripts the steps call, not inside the server.

Every step is recorded in the job JSON as it starts and finishes; a graph backup is taken before
the run and after it (and a backup that FAILS is recorded as a warning — never silently assumed);
the documents each run inserts are written to a manifest so kg_rollback can undo exactly that run.

usage: kg_ingest.py <spec.json>
"""
import json
import os
import subprocess
import sys
import time

SPEC_KEYS = ("graph", "kind", "title", "steps", "backup", "workdir")
JOBS = os.path.expanduser("~/lightrag/jobs")   # same store kg_jobs reads


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def update(job_json, **fields):
    try:
        d = json.load(open(job_json))
    except Exception:
        d = {}
    d.update(fields)
    d["updated"] = now()
    tmp = job_json + ".tmp"
    json.dump(d, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, job_json)


def load_warnings(job_json):
    """Warnings already recorded for this job (the server can seed one before the run)."""
    try:
        return list(json.load(open(job_json)).get("warnings") or [])
    except Exception:
        return []


def main():
    spec_path = os.path.abspath(sys.argv[1])
    spec = json.load(open(spec_path, encoding="utf-8"))
    missing = [k for k in ("graph", "steps") if k not in spec]
    if missing:
        raise SystemExit(f"spec missing {missing}")
    job_id = spec.get("job_id") or os.path.basename(spec_path)[:-5]
    jdir = f"{JOBS}/{job_id}"
    os.makedirs(jdir, exist_ok=True)
    job_json = f"{jdir}/job.json"
    log = f"{jdir}/job.log"
    workdir = spec.get("workdir") or os.path.expanduser("~")
    lf = open(log, "a", encoding="utf-8")

    def run(cmd, step):
        lf.write(f"\n=== [{now()}] {step}: {cmd}\n")
        lf.flush()
        p = subprocess.run(cmd, shell=True, cwd=workdir, stdout=lf,
                           stderr=subprocess.STDOUT, text=True)
        return p.returncode

    update(job_json, state="running", started=now(), steps_total=len(spec["steps"]),
           step=None, manifest=[], graph=spec.get("graph"), kind=spec.get("kind"),
           title=spec.get("title"), pid=os.getpid())

    # Warnings can be seeded by the server (e.g. backup script missing) — keep them.
    warnings = load_warnings(job_json)

    # Backup before touching the graph. The exit code IS checked: a job that reports done with no
    # backup is worse than a job that says the backup failed.
    if spec.get("backup"):
        rc = run(spec["backup"], "backup-before")
        update(job_json, backup_before="ok" if rc == 0 else f"failed (exit {rc})", warnings=warnings)
        if rc != 0:
            warnings.append(f"backup-before failed (exit {rc}) — this run is NOT backed up")
            update(job_json, warnings=warnings)
    else:
        update(job_json, backup_before="skipped", warnings=warnings)

    manifest, failed = [], None
    for i, st in enumerate(spec["steps"], 1):
        name = st.get("name") or f"step{i}"
        update(job_json, step=name, step_index=i)
        rc = run(st["cmd"], name)
        for f in st.get("inserts", []) or []:
            manifest.append(f)
            update(job_json, manifest=manifest)
        if rc != 0:
            failed = f"{name} (exit {rc})"
            break

    if spec.get("backup"):
        rc = run(spec["backup"], "backup-after")
        update(job_json, backup_after="ok" if rc == 0 else f"failed (exit {rc})")
        if rc != 0:
            warnings.append(f"backup-after failed (exit {rc})")

    if failed:
        update(job_json, state="failed", error=failed, finished=now(), warnings=warnings)
    else:
        update(job_json, state="done", finished=now(), warnings=warnings)
    lf.write(f"\n=== job {job_id}: {'FAILED ' + failed if failed else 'done'} at {now()}\n")
    if warnings:
        lf.write("=== warnings: " + "; ".join(warnings) + "\n")
    lf.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
