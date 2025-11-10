from __future__ import annotations
from typing import Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tg_math import estimate_arrhenius_from_segments, arrhenius_plot_data


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")


def _slice_window(df: pd.DataFrame, time_col: str, t0: float, t1: float) -> pd.DataFrame:
    return df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()


def plot_ln_r_vs_time(
        df: pd.DataFrame,
        *,
        time_window: Tuple[float, float],
        time_col: str = "time_min",
        mass_col: str = "mass_pct",
        label: Optional[str] = None,
        smoothing_window: int = 9,
        overlay_constant_r: Optional[float] = None,
        show: bool = False,
        save_path: Optional[str] = None,
):
    if label is None: label = "segment"
    t0, t1 = time_window
    sel = _slice_window(df, time_col, t0, t1)
    if sel.empty:
        raise ValueError(f"No data in time window {time_window}.")
    t = sel[time_col].to_numpy(float)
    m = sel[mass_col].to_numpy(float)

    win = max(3, int(smoothing_window) | 1)
    m_sm = _moving_average(m, win)
    r_inst = np.gradient(m_sm, t)  # dm/dt
    r_abs = np.abs(r_inst)
    eps = 1e-12
    y = np.log(r_abs + eps)

    plt.figure()
    plt.plot(t, y, marker='o', linestyle='-', label=f"ln r(t) [{label}]")
    if overlay_constant_r is not None and np.isfinite(overlay_constant_r) and overlay_constant_r > 0:
        plt.plot([t.min(), t.max()], [np.log(overlay_constant_r)] * 2, linestyle='--', label="ln r (constant)")
    plt.xlabel("Time")
    plt.ylabel("ln(r)")
    plt.title("ln(r) vs time")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_arrhenius(
        x_invT: np.ndarray,
        y_lnr: np.ndarray,
        *,
        slope: Optional[float] = None,
        intercept: Optional[float] = None,
        label: Optional[str] = None,
        show: bool = False,
        save_path: Optional[str] = None,
):
    if label is None: label = "Arrhenius"
    x = np.asarray(x_invT, dtype=float)
    y = np.asarray(y_lnr, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m];
    y = y[m]

    plt.figure()
    plt.plot(x, y, 'o', label="data")
    if slope is not None and intercept is not None and np.isfinite(slope) and np.isfinite(intercept):
        xx = np.linspace(x.min(), x.max(), 200)
        yy = intercept + slope * xx
        plt.plot(xx, yy, '-', label="fit")
    plt.xlabel("1/T (1/K)")
    plt.ylabel("ln(r)")
    plt.title(label)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_arrhenius_groups(
    groups,
    *,
    show: bool = False,
    save_path: str | None = None,
    title: str | None = None,
    legend_loc: str = "best",
):
    """
    Plot multiple Arrhenius datasets (ln(r) vs 1/T) on the same axes, with one
    regression line per group. Each group is a list of SegmentRate objects for
    the same product/condition (e.g., 3 isotherms per product).

    Parameters
    ----------
    groups : Sequence
        A sequence where each element is one of:
          • {"label": str, "segments": [SegmentRate, ...]}
          • (label: str, segments: [SegmentRate, ...])
        (Anything with attributes/keys 'label' and 'segments' also works.)

    show : bool, default False
        If True, display the figure (matplotlib .show()).

    save_path : str | None, default None
        If provided, save the figure to this path (PNG/PDF/etc. based on extension).

    title : str | None, default None
        Optional plot title.

    legend_loc : str, default "best"
        Matplotlib legend location.

    Returns
    -------
    list of dict
        One entry per group with keys:
          { "label", "Ea_J_per_mol", "Ea_kJ_per_mol", "A", "R2", "n_points" }

    Notes
    -----
    • This function *calls* `tg_math` to compute the fit (keeps math out of plotting).
    • No explicit colors are set; matplotlib will cycle defaults across groups.
    • Units of A are the same time units used to compute r (your pipeline decides).
    """

    # Helper to coerce group item into (label, segments) tuple
    def _coerce_group(g):
        if isinstance(g, dict):
            return g.get("label", "group"), g.get("segments", [])
        if isinstance(g, (list, tuple)) and len(g) == 2:
            return g[0], g[1]
        # Fallback to attribute access
        lab = getattr(g, "label", "group")
        segs = getattr(g, "segments", [])
        return lab, segs

    results = []

    plt.figure()

    for g in groups:
        label, segs = _coerce_group(g)

        # Get x, y data for Arrhenius plot (x=1/T, y=ln r)
        x, y = arrhenius_plot_data(segs)
        if x.size < 2:
            # Not enough points to fit a line; plot what we have and continue
            if x.size > 0:
                plt.plot(x, y, "o", label=f"{label} (insufficient points)")
            results.append(
                {"label": label, "Ea_J_per_mol": np.nan, "Ea_kJ_per_mol": np.nan, "A": np.nan, "R2": np.nan, "n_points": int(x.size)}
            )
            continue

        # Math should stay in tg_math for readability
        fit = estimate_arrhenius_from_segments(segs)
        # Scatter through group
        plt.plot(x, y, "o", label=f"{label} data")

        # Plot the fitted line over this group's x-range
        xx = np.linspace(x.min(), x.max(), 200)
        yy = fit.intercept + fit.slope * xx
        plt.plot(xx, yy, "-", label=f"{label} fit")

        results.append(
            {
                "label": label,
                "Ea_J_per_mol": fit.E_A_J_per_mol,
                "Ea_kJ_per_mol": fit.E_A_J_per_mol / 1000.0 if np.isfinite(fit.E_A_J_per_mol) else np.nan,
                "A": fit.A,
                "R2": fit.r2_ln_r_vs_invT,
                "n_points": fit.n_points,
            }
        )

    plt.xlabel("1/T (1/K)")
    plt.ylabel("ln(r)")
    if title:
        plt.title(title)
    plt.legend(loc=legend_loc)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return results
