"""Shared matplotlib style for the paper's results figures (Figures 3 and 4).

* Palatino.  TeX Gyre Pagella is the Palatino clone shipped with TeX Live; it
  lives under /usr/share/texmf/ which matplotlib does not scan, so the four
  faces are registered explicitly at import.  ``P052`` (URW Palladio L) is the
  fallback, and DejaVu Serif the last resort.
* No grid.  ``despine`` drops the top/right spines the way ``sns.despine()``
  does; nothing draws grid lines.
* No legend frame and no legend title -- what the series are belongs in the
  caption, not in a box on the plot.
* Figures are authored at final print size, so a requested point size is the
  size that lands on the page.  Nothing is scaled by ``\\includegraphics``.

The palette was checked for colour-vision deficiency separation.
"""
from __future__ import annotations

import glob
from contextlib import contextmanager
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------

_PAGELLA_GLOB = "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf"


def register_palatino() -> str:
    """Register TeX Gyre Pagella with matplotlib and return the family to use.

    Returns the first Palatino-compatible family actually available, so a
    machine without TeX Live still renders (in P052, or DejaVu Serif).
    """
    for path in sorted(glob.glob(_PAGELLA_GLOB)):
        try:
            fm.fontManager.addfont(path)
        except Exception:  # a broken font file must not take the figure down
            pass
    available = {f.name for f in fm.fontManager.ttflist}
    for family in ("TeX Gyre Pagella", "P052", "DejaVu Serif"):
        if family in available:
            return family
    return "serif"


PALATINO = register_palatino()

#: Width of the CoRL text block, in inches.  Panels are authored as a fraction
#: of this so \includegraphics[width=...] never rescales them -- a rescaled
#: figure is exactly how the legend strip ended up with larger type than the
#: panels it labels.
TEXT_W = 5.5

#: Base point size.  Panels are authored at final size, so this is literally
#: what the reader sees on the page.
BASE_PT = 9.0

PAPER_RC: dict = {
    "font.family": "serif",
    "font.serif": [PALATINO, "P052", "DejaVu Serif"],
    # Pagella has no math face wired into matplotlib's custom fontset, so map
    # the math faces onto it explicitly; digits and axis numbers then match the
    # body text instead of falling back to DejaVu.
    "mathtext.fontset": "custom",
    "mathtext.rm": PALATINO,
    "mathtext.it": f"{PALATINO}:italic",
    "mathtext.bf": f"{PALATINO}:bold",
    # The custom fontset requires every slot to be named, including the ones
    # this figure never uses -- otherwise matplotlib falls back to its default
    # 'cursive' list, fails to resolve it, and warns on every render.
    "mathtext.cal": f"{PALATINO}:italic",
    "mathtext.sf": PALATINO,
    "mathtext.tt": "DejaVu Sans Mono",
    "axes.formatter.use_mathtext": True,

    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT + 1,
    "axes.titlesize": BASE_PT + 1,
    "xtick.labelsize": BASE_PT,
    "ytick.labelsize": BASE_PT,
    "legend.fontsize": BASE_PT,

    # No grid, no top/right spines.
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#374151",
    "axes.labelcolor": "#111827",
    "text.color": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,

    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",

    # Frameless, title-less legends everywhere.
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.2,

    "figure.dpi": 130,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "savefig.facecolor": "white",
    # Keep text as text in the PDF so it scales and can be searched/copied.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextmanager
def paper_style(**overrides):
    """Context manager applying the paper rc, plus any per-figure overrides."""
    rc = dict(PAPER_RC)
    rc.update(overrides)
    with plt.rc_context(rc):
        yield


# --------------------------------------------------------------------------
# Method identity: one colour/marker per method, shared by Figs 3 and 4
# --------------------------------------------------------------------------
#
# Colour follows the entity, not its rank: a figure that drops a series must
# not repaint the survivors, so these are looked up by label, never cycled.
#
# No method is red or green: readers take those as "bad" and "good".
#
# The four Scaffolded-LLM rows are one hue family, because that is what they
# are -- one method with three different post-hoc calibrations.  They are
# lightness steps of Scaffolded-LLM's brown, and each variant also gets its own
# line style, since adjacent steps of a same-hue ramp sit below the
# normal-vision separation floor.

OURS = "LSM (ours)"

METHOD_STYLE: dict[str, dict] = {
    "Vision-only": dict(color="#6b7280", marker="*", ls="-",  lw=1.4, ms=5.5),
    "LLM-E2E": dict(color="#7c3aed", marker="X", ls="-",  lw=1.4, ms=4.6),
    "Scaffolded-LLM": dict(color="#8c4a00", marker="s", ls="-",  lw=1.4, ms=4.0),
    "Scaffolded-LLM + global rescale": dict(color="#4a2600", marker="^", ls="--", lw=1.4, ms=4.4),
    "Scaffolded-LLM + per-axis rescale": dict(color="#b87333", marker="v", ls=":",  lw=1.6, ms=4.4),
    "Scaffolded-LLM + in-context": dict(color="#d4a05a", marker="P", ls="-.", lw=1.4, ms=4.6),
    "Scaffolded-VLM": dict(color="#c9a227", marker="D", ls="-",  lw=1.4, ms=3.8),
    "3D-ViSTA": dict(color="#cc79a7", marker="h", ls="-",  lw=1.4, ms=4.6),
    OURS: dict(color="#0077b6", marker="o", ls="-",  lw=2.1, ms=4.8),
}

#: Legend / drawing order.  Ours last so it is drawn on top of the baselines.
CANONICAL_ORDER: tuple[str, ...] = tuple(METHOD_STYLE)

#: Row / method keys -> display label. Table 1 rows are `gt_*` (ground-truth
#: proposer), Figure 3 rows are bare names, and closed-loop methods use their
#: config names. 3D-ViSTA and Scaffolded-VLM appear only in archived results.
APPROACH_LABELS: dict[str, str] = {
    # Table 1 (langsensor.eval.table1)
    "gt_lss": OURS,
    "gt_llm": "Scaffolded-LLM",
    "gt_llm_scale": "Scaffolded-LLM + global rescale",
    "gt_llm_axis_scale": "Scaffolded-LLM + per-axis rescale",
    "gt_llm_incontext": "Scaffolded-LLM + in-context",
    "gt_vlm": "Scaffolded-VLM",
    # Figure 3 (langsensor.eval.fig3)
    "framework": OURS,
    "straight_llm": "LLM-E2E",
    "scaffolded_llm": "Scaffolded-LLM",
    "scaffolded_llm_scale": "Scaffolded-LLM + global rescale",
    "scaffolded_llm_axis_scale": "Scaffolded-LLM + per-axis rescale",
    "scaffolded_llm_incontext": "Scaffolded-LLM + in-context",
    "scaffolded_vlm": "Scaffolded-VLM",
    "vista_gmm": "3D-ViSTA",
    # Closed loop, Figure 4 / Table 2 (langsensor.vlmap.run)
    "vision_only": "Vision-only",
    "llm_e2e": "LLM-E2E",
    "llm_scaffolded": "Scaffolded-LLM",
    "llm_scaled": "Scaffolded-LLM + global rescale",
    "llm_scaled_axis": "Scaffolded-LLM + per-axis rescale",
    "llm_incontext": "Scaffolded-LLM + in-context",
    "vlm": "Scaffolded-VLM",
    "lss_baseline": OURS,
    # ... and the output names of langsensor.vlmap.run (predictor config names)
    "none": "Vision-only",
    "lsm": OURS,
    "llm": "Scaffolded-LLM",
}


def style_for(label: str) -> dict:
    """Style dict for a display label, with a visible fallback if unmapped."""
    return METHOD_STYLE.get(label, dict(color="#374151", marker=".", ls="-", lw=1.3, ms=4.0))


def order_labels(labels) -> list[str]:
    """Sort labels into CANONICAL_ORDER, appending anything unrecognised."""
    known = [m for m in CANONICAL_ORDER if m in set(labels)]
    extra = sorted(set(labels) - set(known))
    return known + extra


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def despine(ax, *, top: bool = True, right: bool = True) -> None:
    """Drop the top/right spines, as ``sns.despine()`` does by default.

    Uses seaborn when it is installed so the behaviour matches the rest of the
    group's plotting code, and falls back to the two-line matplotlib
    equivalent otherwise -- this module must not hard-depend on seaborn.
    """
    try:
        import seaborn as sns
        sns.despine(ax=ax, top=top, right=right)
    except Exception:
        if top:
            ax.spines["top"].set_visible(False)
        if right:
            ax.spines["right"].set_visible(False)


def legend_no_frame(ax_or_fig, handles=None, labels=None, **kwargs):
    """Legend with no frame and no title."""
    kwargs.setdefault("frameon", False)
    kwargs.pop("title", None)
    if handles is not None:
        return ax_or_fig.legend(handles, labels, **kwargs)
    return ax_or_fig.legend(**kwargs)


def legend_strip(handles, labels, *, max_ncol: int = 4, width: float | None = None):
    """A standalone legend figure exactly ``width`` inches wide (text block by
    default), using the most columns that still fit.

    Returning a fixed-width figure is what keeps its type the same size as the
    panels' once LaTeX places it at ``width=\\textwidth``. Picking ncol by
    measurement is what keeps long labels -- "Scaffolded-LLM + per-axis
    rescale" -- from running off the edge, which a fixed ncol=3 did once the
    rebuttal baselines were added.
    """
    width = TEXT_W if width is None else width
    for ncol in range(min(max_ncol, len(labels)), 0, -1):
        nrows = -(-len(labels) // ncol)
        fig = plt.figure(figsize=(width, 0.235 * nrows + 0.08))
        leg = legend_no_frame(fig, handles, labels, loc="center", ncol=ncol,
                              handlelength=2.0, columnspacing=1.0)
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if bb.width <= width - 0.05 or ncol == 1:
            return fig
        plt.close(fig)


def save(fig, out_dir, stem: str, *, also_png: bool = True,
         tight: bool = True) -> list[Path]:
    """Write ``<out_dir>/<stem>.pdf`` (+ .png) and return the paths written.

    ``tight=False`` keeps the figure's declared size instead of cropping to
    content.  The legend strips need that: cropped, their PDF is narrower than
    the text block, so ``width=\textwidth`` scales them up and their type ends
    up bigger than the axis labels they sit above.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # bbox_inches=None means "use rcParams", which is 'tight' here -- so an
    # explicit full-figure Bbox is the only way to keep the declared size.
    if tight:
        bbox = "tight"
    else:
        w, h = fig.get_size_inches()
        bbox = Bbox([[0.0, 0.0], [w, h]])
    written = [out_dir / f"{stem}.pdf"]
    fig.savefig(written[0], bbox_inches=bbox)
    if also_png:
        written.append(out_dir / f"{stem}.png")
        fig.savefig(written[-1], dpi=400, bbox_inches=bbox)
    return written


__all__ = [
    "PALATINO", "BASE_PT", "TEXT_W", "PAPER_RC", "paper_style",
    "METHOD_STYLE", "CANONICAL_ORDER", "APPROACH_LABELS", "OURS",
    "style_for", "order_labels", "despine", "legend_no_frame", "legend_strip", "save",
]
