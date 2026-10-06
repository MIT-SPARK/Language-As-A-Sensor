"""Render Figure 3 (ANEES and RMSE vs ambiguity) from ``langsensor.eval.fig3`` CSVs.

    python -m langsensor.paper.fig3 \
        --csv artifacts/results/static/fig3_by_ambiguity.csv --csv outputs/fig3/fig3_by_ambiguity.csv \
        --out-dir outputs/paper

``--csv`` is repeatable (globs allowed); rows are merged on (approach,
ambiguity) with the newest file winning. Writes ambiguity_vs_anees.{pdf,png}
and ambiguity_vs_rmse.{pdf,png}.
"""
from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from langsensor.paper import style as ps

#: Ambiguity buckets, in axis order.  "5+" pools everything >= 5.
ORDER = ["1", "2", "3", "4", "5+"]

#: ANEES is calibrated at 3 for a 3-DoF Gaussian; [2, 4] is the band we accept.
BAND_LO, BAND_HI, BAND_TARGET = 2.0, 4.0, 3.0

#: A bucket is discarded when its n falls below this fraction of the largest n
#: seen -- the signature of a run that died partway through.
MIN_N_FRACTION = 0.9
BAND_COLOR = "#43aa8b"

#: Panel geometry.  A CoRL text block is 5.5in; a 0.49\textwidth subfigure is
#: 2.70in, and the figure is authored at exactly that so nothing is rescaled
#: on the page and the point sizes in paper_style are what the reader gets.
# Two panels side by side at 0.495\textwidth each -- the widest that still sits
# on one line with a sliver of gap -- authored at that size so nothing rescales.
PANEL_W, PANEL_H = 0.495 * ps.TEXT_W, 2.45


def load(csv_args: list[str]) -> pd.DataFrame:
    """Merge every matching CSV, newest file winning per (approach, ambiguity)."""
    paths: list[Path] = []
    for pattern in csv_args:
        hits = [Path(p) for p in globlib.glob(pattern)]
        if not hits:
            raise SystemExit(f"--csv matched nothing: {pattern}")
        paths.extend(hits)
    paths = sorted(set(paths), key=lambda p: p.stat().st_mtime)

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["_source"] = p.name
        frames.append(df)
        print(f"  read {p}  ({len(df)} rows)")
    df = pd.concat(frames, ignore_index=True)

    df["ambiguity"] = df["ambiguity"].astype(str)
    df = df[df["ambiguity"].isin(ORDER)].copy()
    # Later files win.
    df = df.drop_duplicates(subset=["approach", "ambiguity"], keep="last")

    df["method"] = df["approach"].map(ps.APPROACH_LABELS)
    unmapped = sorted(set(df.loc[df["method"].isna(), "approach"]))
    if unmapped:
        print(f"  WARNING: unmapped approach keys dropped: {unmapped}")
        df = df[df["method"].notna()]

    # A run that lost API access partway still writes rows, just with most
    # queries failed, and a short bucket silently distorts a whole series.
    # Drop those loudly rather than plotting them.
    if "n" in df.columns and len(df):
        full = int(df["n"].max())
        short = df[df["n"] < MIN_N_FRACTION * full]
        for _, row in short.iterrows():
            print(f"  DROPPED {row['approach']} @ ambiguity={row['ambiguity']}: "
                  f"n={int(row['n'])} of {full} -- incomplete run")
        df = df[df["n"] >= MIN_N_FRACTION * full]

    df["ambiguity"] = pd.Categorical(df["ambiguity"], categories=ORDER, ordered=True)
    return df.sort_values(["method", "ambiguity"])


def _series(df: pd.DataFrame, method: str):
    sub = df[df["method"] == method].sort_values("ambiguity")
    return sub if not sub.empty else None


def _draw(ax, x, y, lo, hi, st, label):
    ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, lw=0, zorder=1)
    ax.plot(x, y, color=st["color"], lw=st["lw"], ls=st["ls"], zorder=2)
    # A surface-coloured ring keeps overlapping markers separable.
    ax.plot(x, y, color=st["color"], lw=0, marker=st["marker"], ms=st["ms"],
            mec="white", mew=0.7, label=label, zorder=3)


def panel_anees(df: pd.DataFrame, methods: list[str], out_dir: Path, *, scale: str):
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    ax.axhspan(BAND_LO, BAND_HI, color=BAND_COLOR, alpha=0.12, zorder=0, lw=0)
    ax.axhline(BAND_TARGET, ls="--", lw=0.8, color=BAND_COLOR, alpha=0.55, zorder=0)

    top, bot = 0.0, np.inf
    for m in methods:
        sub = _series(df, m)
        if sub is None:
            continue
        x = np.arange(len(sub))
        y = sub["anees_median"].to_numpy(float)
        # SE of a median via the normal approximation: sigma ~ IQR / 1.349.
        n = np.maximum(sub["n"].to_numpy(float), 1.0)
        se = (sub["anees_iqr"].to_numpy(float) / 1.349) / np.sqrt(n)
        lo, hi = np.maximum(y - se, 1e-3), y + se
        top, bot = max(top, hi.max()), min(bot, lo.min())
        _draw(ax, x, y, lo, hi, ps.style_for(m), m)

    ax.set_xticks(np.arange(len(ORDER)))
    ax.set_xticklabels(ORDER)
    ax.set_xlabel("Ambiguity level")
    ax.set_ylabel("ANEES (median)")

    if scale == "log":
        ax.set_yscale("log")
        # Headroom in log space, so the topmost error band is fully inside the
        # axes -- the old linear limit clipped Scaffolded-VLM off the top.
        y_lo, y_hi = max(bot * 0.75, 1e-2), top * 1.6
        ax.set_ylim(y_lo, y_hi)
        # Plain numbers beat 10^0 / 10^1 here: the reader needs to place each
        # curve against 2, 3 and 4, which a decade-only axis hides.
        ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50) if y_lo <= t <= y_hi]
        ax.set_yticks(ticks)
        ax.set_yticklabels([("%g" % t) for t in ticks])
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    else:
        ax.set_ylim(0, top * 1.15)

    # Anchored to the axes' left edge rather than to a data x, and centred in
    # the band, so it survives any change of y-limits.
    # A white pad keeps it readable when a curve runs through the band, which
    # now happens often: the rescaled baselines land right on top of it.
    ax.text(0.015, BAND_TARGET, "CALIBRATED", transform=ax.get_yaxis_transform(),
            color=BAND_COLOR, fontsize=ps.BASE_PT - 2.5, fontweight="bold",
            ha="left", va="center", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.8))

    ps.despine(ax)
    fig.tight_layout(pad=0.2)
    for p in ps.save(fig, out_dir, "ambiguity_vs_anees"):
        print(f"  wrote {p}")
    plt.close(fig)


def _legend_handles(methods: list[str]):
    return [plt.Line2D([], [], color=(st := ps.style_for(m))["color"], lw=st["lw"],
                       ls=st["ls"], marker=st["marker"], ms=st["ms"],
                       mec="white", mew=0.7, label=m) for m in methods]


def panel_rmse(df: pd.DataFrame, methods: list[str], out_dir: Path, *, legend: bool = True):
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    lo_all, hi_all = np.inf, -np.inf
    for m in methods:
        sub = _series(df, m)
        if sub is None:
            continue
        x = np.arange(len(sub))
        y = sub["rmse_mean"].to_numpy(float)
        n = np.maximum(sub["n"].to_numpy(float), 1.0)
        se = sub["rmse_std"].to_numpy(float) / np.sqrt(n)
        lo, hi = np.maximum(y - se, 0.0), y + se
        lo_all, hi_all = min(lo_all, lo.min()), max(hi_all, hi.max())
        _draw(ax, x, y, lo, hi, ps.style_for(m), m)

    ax.set_xticks(np.arange(len(ORDER)))
    ax.set_xticklabels(ORDER)
    ax.set_xlabel("Ambiguity level")
    ax.set_ylabel("RMSE (m)")
    span = hi_all - lo_all
    # Headroom for the in-panel legend (upper left, where RMSE is lowest): the
    # series only climb past it at the 5+ bucket, on the far right.
    ax.set_ylim(max(0.0, lo_all - 0.08 * span), hi_all + (0.12 if legend else 0.10) * span)
    if legend:
        # As in the original Figure 3: the legend lives in panel (b) only and is
        # shared by both panels. Frameless and title-less.
        # Short labels, Table 1 style: the three calibrated variants are listed as
        # "+ ..." under Scaffolded-LLM (they share its hue family), which keeps a
        # single column narrow enough to sit in the empty upper-left of the panel.
        short = [m.replace("Scaffolded-LLM + ", "   + ") for m in methods]
        ps.legend_no_frame(ax, _legend_handles(methods), short, loc="upper left",
                           ncol=1, fontsize=ps.BASE_PT - 2.0, handlelength=1.7,
                           handletextpad=0.4, labelspacing=0.22, borderaxespad=0.1)

    ps.despine(ax)
    fig.tight_layout(pad=0.2)
    for p in ps.save(fig, out_dir, "ambiguity_vs_rmse"):
        print(f"  wrote {p}")
    plt.close(fig)


def panel_legend(methods: list[str], out_dir: Path, *, ncol: int):
    """Standalone legend strip shared by both panels, fitted to the text width."""
    handles = [
        plt.Line2D([], [], color=(st := ps.style_for(m))["color"], lw=st["lw"],
                   ls=st["ls"], marker=st["marker"], ms=st["ms"],
                   mec="white", mew=0.7, label=m)
        for m in methods
    ]
    fig = ps.legend_strip(handles, [h.get_label() for h in handles], max_ncol=ncol)
    for p in ps.save(fig, out_dir, "ambiguity_legend", tight=False):
        print(f"  wrote {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True,
                    help="CSV path or glob; repeatable. Newest file wins per cell.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--anees-scale", choices=["log", "linear"], default="log",
                    help="With the calibrated baselines the medians span ~70x "
                         "(0.4 to 27), which a linear axis renders unreadable at "
                         "the low end. Default log; pass linear for the old look.")
    ap.add_argument("--legend-ncol", type=int, default=3)
    ap.add_argument("--legend-mode", choices=["panel", "strip"], default="panel",
                    help="panel: legend inside the RMSE panel, as the original Figure 3 "
                         "did. strip: a separate full-width legend PDF instead.")
    args = ap.parse_args()

    print("Figure 3")
    df = load(args.csv)
    methods = ps.order_labels(df["method"].unique())
    print(f"  series ({len(methods)}): {', '.join(methods)}")

    # Every series must cover every bucket, or the panels compare unlike things.
    incomplete = []
    for m in list(methods):
        got = sorted(_series(df, m)["ambiguity"].astype(str))
        if got != sorted(ORDER):
            print(f"  DROPPED series {m}: has buckets {got}, expected {ORDER}")
            incomplete.append(m)
    methods = [m for m in methods if m not in incomplete]
    if not methods:
        raise SystemExit("no series survived the completeness check")
    ns = df.groupby("method", observed=True)["n"].unique()
    if len({tuple(sorted(v)) for v in ns}) > 1:
        print(f"  WARNING: per-method sample counts differ:\n{ns}")

    with ps.paper_style():
        panel_anees(df, methods, args.out_dir, scale=args.anees_scale)
        panel_rmse(df, methods, args.out_dir, legend=(args.legend_mode == "panel"))
        if args.legend_mode == "strip":
            panel_legend(methods, args.out_dir, ncol=args.legend_ncol)


if __name__ == "__main__":
    main()
