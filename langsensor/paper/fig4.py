"""Render Figure 4 (belief evolution in the closed loop) from sweep outputs.

    python -m langsensor.paper.fig4 --out-root artifacts/results/closed_loop_2026-05-14 --fig-dir outputs/paper

Per episode the series is EMA-smoothed (alpha 0.02) and resampled to 200 points
of episode progress; curves are the mean +/- standard error across episodes.
Writes information_gain.{pdf,png} and target_object_probability.{pdf,png}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from langsensor.paper import style as ps
from langsensor.paper.closed_loop import discover

EMA_ALPHA = 0.02
N_RESAMPLE = 200

#: (metric key, y label, output stem) -- the two panels experiments.tex includes.
PANELS = [
    ("info_gain_at_surface", "Information gain (nats)", "information_gain"),
    ("surface_mass", "Target object likelihood", "target_object_probability"),
]

#: 0.48\textwidth of the 5.5in CoRL block, authored at final size.
PANEL_W, PANEL_H = 0.495 * ps.TEXT_W, 2.30   # widest pair that fits on one line


def ema(arr: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """EMA that skips NaNs."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    one_minus = 1.0 - alpha
    state = float("nan")
    for i in range(arr.size):
        v = arr[i]
        if np.isnan(v):
            out[i] = float("nan")
            continue
        state = v if np.isnan(state) else (alpha * v + one_minus * state)
        out[i] = state
    return out


def aggregate(series_list, n_resample: int = N_RESAMPLE):
    """Per-episode EMA -> resample to [0,1] -> NaN-safe mean and SE."""
    valid = [np.asarray(s, dtype=np.float64)
             for s in series_list if s is not None and len(s) > 0]
    if not valid:
        return None

    x = np.linspace(0.0, 1.0, n_resample)

    def align(a):
        if a.size == 1:
            return np.full(n_resample, float(a[0]))
        return np.interp(x, np.linspace(0.0, 1.0, a.size), a)

    smoothed = np.stack([align(ema(a)) for a in valid])
    n_per_t = np.sum(np.isfinite(smoothed), axis=0)
    mean = np.nanmean(smoothed, axis=0)
    if smoothed.shape[0] > 1:
        std = np.nanstd(smoothed, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            se = std / np.sqrt(np.maximum(n_per_t, 1))
    else:
        se = np.zeros_like(mean)
    return x, mean, se, int(np.max(n_per_t)) if n_per_t.size else 0


def panel(data, key: str, ylabel: str, stem: str, fig_dir: Path, methods: list[str],
          *, legend: bool = True):
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    drew = False
    for label in methods:
        agg = aggregate([ep.get(key) for ep in data[label]])
        if agg is None:
            continue
        x, mean, se, _ = agg
        st = ps.style_for(label)
        ax.fill_between(x, mean - se, mean + se, color=st["color"], alpha=0.16, lw=0)
        ax.plot(x, mean, color=st["color"], lw=st["lw"], ls=st["ls"], label=label)
        drew = True
    if not drew:
        plt.close(fig)
        print(f"  WARNING: nothing to draw for '{key}', panel skipped")
        return

    # Information gain is measured against a uniform prior, so zero is the
    # "language told us nothing" line and negative means language actively hurt.
    if key == "info_gain_at_surface":
        ax.axhline(0.0, color="#9ca3af", lw=0.7, ls=":", zorder=0)

    ax.set_xlabel("Episode progress")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 1.0)
    ax.margins(x=0.01)
    if legend:
        # Frameless legend above the data, with headroom so it never sits on a curve.
        lo, hi = ax.get_ylim()
        ticks = [t for t in ax.get_yticks() if lo <= t <= hi]   # ticks for the data only,
        ax.set_ylim(lo, hi + 0.45 * (hi - lo))                   # not for the legend's headroom
        ax.set_yticks(ticks)
        ps.legend_no_frame(ax, loc="upper center", ncol=2, fontsize=ps.BASE_PT - 2.0,
                           handlelength=1.6, columnspacing=1.0, handletextpad=0.4,
                           labelspacing=0.22, borderaxespad=0.1)
    ps.despine(ax)
    fig.tight_layout(pad=0.2)
    for p in ps.save(fig, fig_dir, stem):
        print(f"  wrote {p}")
    plt.close(fig)


def legend_strip(methods: list[str], fig_dir: Path, ncol: int):
    """Shared legend strip, fitted to the text width."""
    handles = [
        plt.Line2D([], [], color=(st := ps.style_for(m))["color"],
                   lw=st["lw"], ls=st["ls"], label=m)
        for m in methods
    ]
    fig = ps.legend_strip(handles, [h.get_label() for h in handles], max_ncol=ncol)
    for p in ps.save(fig, fig_dir, "closedloop_legend", tight=False):
        print(f"  wrote {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", action="append", type=Path, required=True,
                    help="Sweep directory holding <config>/metrics_timeseries.json; "
                         "repeatable, newest wins per method.")
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--legend-mode", choices=["panel", "strip"], default="panel")
    ap.add_argument("--legend-ncol", type=int, default=3)
    ap.add_argument("--drop", action="append", default=["lss_reconsider"],
                    help="Method directory to exclude (lss_reconsider is in the archived sweep but not the paper).")
    args = ap.parse_args()

    print("Figure 4")
    data = discover(args.out_root, set(args.drop))
    if not data:
        raise SystemExit("no closed-loop metrics found")

    methods = ps.order_labels(data.keys())
    print(f"  series ({len(methods)}): {', '.join(methods)}")
    counts = {len(v) for v in data.values()}
    if len(counts) > 1:
        print(f"  WARNING: methods do not share an episode count: "
              f"{ {m: len(data[m]) for m in methods} }")

    with ps.paper_style():
        for key, ylabel, stem in PANELS:
            # One shared legend, in panel (b) only -- the same convention as
            # Figure 3 -- so panel (a) keeps its full height for the data.
            panel(data, key, ylabel, stem, args.fig_dir, methods,
                  legend=(args.legend_mode == "panel" and key == PANELS[-1][0]))
        if args.legend_mode == "strip":
            legend_strip(methods, args.fig_dir, args.legend_ncol)


if __name__ == "__main__":
    main()
