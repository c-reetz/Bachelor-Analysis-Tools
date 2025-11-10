from __future__ import annotations
from typing import Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
