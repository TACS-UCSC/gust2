"""build_n128_table.py — (re)build plots/scaling_tempopt_n128/n128_table.json.

The per-(cell, T) EMD table behind plot_climate_temp_band.py and
plot_prediction_vs_outcome.py: table[size|sc|arch][T] = {emd, tke, floor}.
Originally assembled ad-hoc (no builder in repo); this script makes it
reproducible from the N=128 sweep's wandb analysis runs. Re-run after new
sweep cells land (e.g. the sc341 hot-edge / warm-sc917 gap runs).

Reads the same projects and run-name convention as plot_scaling_tempopt.py
(whose parse_run_name is imported so the two can't drift). `*-survival` runs
are the collapse-time diagnostic, not climate analysis — skipped explicitly.
Duplicate (cell, T) runs resolve to the most recently created finished run.

Usage:
    ~/llm/bin/python build_n128_table.py            # writes the json
    ~/llm/bin/python build_n128_table.py --diff     # also diff vs existing
"""

import argparse
import json
import os

import wandb

from plot_scaling_tempopt import parse_run_name

ENTITY = "bigpseud-ucsc"
PROJECTS = [f"gust2-scaling-tempopt-n128-{s}" for s in ("small", "medium", "large")]
OUT = "plots/scaling_tempopt_n128/n128_table.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", action="store_true",
                    help="diff against the existing table before overwriting")
    args = ap.parse_args()

    api = wandb.Api()
    latest = {}     # (cell, T) -> (created_at, metrics)
    for project in PROJECTS:
        runs = list(api.runs(f"{ENTITY}/{project}"))
        print(f"{project}: {len(runs)} runs")
        for r in runs:
            if r.state != "finished" or "survival" in r.name:
                continue
            parsed = parse_run_name(r.name)
            if parsed is None:
                continue
            size, sc, arch, temp = parsed
            emd = r.summary.get("emd/nsp")
            if emd is None:
                continue
            key = (f"{size}|{sc}|{arch}", temp)
            rec = (str(r.created_at), {
                "emd": float(emd),
                "tke": float(r.summary.get("tke_rse/nsp", float("nan"))),
                "floor": float(r.summary.get("emd/vqvae", float("nan"))),
            })
            if key not in latest or rec[0] > latest[key][0]:
                latest[key] = rec

    table = {}
    for (cell, temp), (_, metrics) in sorted(latest.items()):
        table.setdefault(cell, {})[str(temp)] = metrics
    n_pts = sum(len(d) for d in table.values())
    print(f"-> {len(table)} cells, {n_pts} (cell, T) points")

    if args.diff and os.path.exists(OUT):
        old = json.load(open(OUT))
        for cell in sorted(set(old) | set(table)):
            o, n = old.get(cell, {}), table.get(cell, {})
            added = sorted(set(n) - set(o), key=float)
            gone = sorted(set(o) - set(n), key=float)
            moved = [t for t in set(o) & set(n)
                     if abs(o[t]["emd"] - n[t]["emd"]) > 1e-9]
            if added or gone or moved:
                print(f"  {cell}: +T{added or ''} -T{gone or ''}"
                      f"{' emd-changed@' + str(moved) if moved else ''}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(table, open(OUT, "w"))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
