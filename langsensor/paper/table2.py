"""Render Table 2 (closed-loop metrics, observed vs never-observed targets) from sweep outputs.

    python -m langsensor.paper.table2 --out-root artifacts/results/closed_loop_2026-05-14 \
        --unobserved-from artifacts/results/closed_loop_2026-05-14/_summary.json \
        --out outputs/paper/table2.tex --csv outputs/paper/table2.csv

Episodes whose target the trajectory never observes are listed in the sweep's
``_summary.json`` ("unobserved_episodes").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from langsensor.paper import style as ps
from langsensor.paper.closed_loop import aggregate, discover

#: (aggregate key, header, format, "higher is better", per-episode key for the
#: standard error, or None for the columns the paper prints without one).
COLUMNS = [
    ("mean_info_gain", r"Mean IG (nats) $\uparrow$", "{:.2f}", True, "mean_info_gain"),
    ("mean_sm", r"Mean prob. mass $\uparrow$", "{:.3f}", True, "terminal_surface_mass"),
    ("mean_argmax_dist", r"Mean argmax dist (m) $\downarrow$", "{:.2f}", False, None),
    ("success_pct", r"Success \% $\uparrow$", "{:.1f}", True, None),
    ("miss_pct", r"Miss \% $\downarrow$", "{:.1f}", False, None),
]


def sem(rows: list[dict], key: str) -> float:
    """Standard error of a per-episode quantity, across episodes."""
    vals = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    return float(vals.std(ddof=1) / np.sqrt(vals.size))


def block(data, methods, unobserved: set[str], want_unobserved: bool,
          success_threshold: float, miss_threshold: float):
    rows, n_eps = {}, 0
    for label in methods:
        recs = [r for r in data[label]
                if (r.get("episode_id") in unobserved) == want_unobserved]
        rows[label] = aggregate(recs, success_threshold=success_threshold,
                                miss_threshold=miss_threshold)
        n_eps = max(n_eps, rows[label]["n_eps"])
    return rows, n_eps


def winners(rows, methods):
    out = {}
    for key, _, _, higher, _ in COLUMNS:
        vals = [(m, rows[m][key]) for m in methods
                if rows[m][key] == rows[m][key]]  # drop NaN
        if vals:
            out[key] = (max if higher else min)(vals, key=lambda kv: kv[1])[0]
    return out


def emit_block(lines, title, rows, methods, ours):
    win = winners(rows, methods)
    lines.append(f"        \\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}} \\\\")
    lines.append("        \\midrule")
    for m in methods:
        if m == ours:
            lines.append("        \\rowcolor{gray!8}")
        cells = []
        for key, _, fmt, _, se_key in COLUMNS:
            v = rows[m][key]
            if v != v:
                body = "--"
            elif se_key is None:
                body = f"${fmt.format(v)}$"
            else:
                e = sem(rows[m]["rows"], se_key)
                body = (f"${fmt.format(v)}$" if e != e
                        else f"${fmt.format(v)} \\pm {fmt.format(e)}$")
            cells.append(f"\\cellcolor{{green!25}}{body}" if win.get(key) == m else body)
        lines.append(f"        {m:<36} & " + " & ".join(cells) + " \\\\")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--unobserved-from", type=Path, default=None,
                    help="A _summary.json whose 'unobserved_episodes' list defines "
                         "the second block (episodes vision never saw).")
    ap.add_argument("--unobserved-episodes", default="",
                    help="Comma-separated ids, as an alternative to --unobserved-from.")
    ap.add_argument("--drop", action="append", default=["lss_reconsider"])
    ap.add_argument("--success-threshold", type=float, default=0.15)
    ap.add_argument("--miss-threshold", type=float, default=0.01)
    args = ap.parse_args()

    print("Table 2")
    data = discover(args.out_root, set(args.drop))
    if not data:
        raise SystemExit("no closed-loop metrics found")
    methods = ps.order_labels(data.keys())
    for m in methods:
        print(f"  {m:<36} {len(data[m]):>3} episodes")

    unobserved = {e.strip() for e in args.unobserved_episodes.split(",") if e.strip()}
    if args.unobserved_from and args.unobserved_from.exists():
        unobserved |= set(json.loads(args.unobserved_from.read_text())
                          .get("unobserved_episodes", []))
    print(f"  unobserved episodes: {len(unobserved)}")

    obs_rows, n_obs = block(data, methods, unobserved, False,
                            args.success_threshold, args.miss_threshold)
    uno_rows, n_uno = block(data, methods, unobserved, True,
                            args.success_threshold, args.miss_threshold)

    header = " & ".join(h for _, h, _, _, _ in COLUMNS)
    lines = [
        "% Generated by langsensor.paper.table2 -- do not hand-edit.",
        "\\begin{table}[!htb]",
        "    \\centering",
        "    \\small",
        "    \\caption{Time-varying belief metrics across the rollout, split by whether",
        "    the target was directly observed by vision during the trajectory. The",
        "    \\emph{Unobserved} regime isolates cases in which the robot never directly",
        "    observes the target.}",
        "    \\label{tab:timevarying}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\renewcommand{\\arraystretch}{0.92}",
        "    \\begin{tabular}{l ccc cc}",
        "        \\toprule",
        f"        Method & {header} \\\\",
        "        \\midrule",
    ]
    emit_block(lines, f"Target observed by vision ($n={n_obs}$)",
               obs_rows, methods, ps.OURS)
    lines.append("        \\midrule")
    emit_block(lines, f"Target never directly observed ($n={n_uno}$)",
               uno_rows, methods, ps.OURS)
    lines += ["        \\bottomrule", "    \\end{tabular}}", "\\end{table}"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {args.out}  (observed n={n_obs}, unobserved n={n_uno})")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w") as f:
            cols = [k for k, _, _, _, _ in COLUMNS]
            f.write("regime,method,n_eps," + ",".join(cols)
                    + ",se_mean_info_gain,se_mean_sm\n")
            for regime, rows in (("observed", obs_rows), ("unobserved", uno_rows)):
                for m in methods:
                    vals = ",".join(f"{rows[m][k]:.6g}" for k in cols)
                    ses = (f"{sem(rows[m]['rows'], 'mean_info_gain'):.6g},"
                           f"{sem(rows[m]['rows'], 'terminal_surface_mass'):.6g}")
                    f.write(f"{regime},\"{m}\",{rows[m]['n_eps']},{vals},{ses}\n")
        print(f"  wrote {args.csv}")


if __name__ == "__main__":
    main()
