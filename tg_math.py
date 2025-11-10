from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Sequence, Dict, Any
import numpy as np
import pandas as pd
import math

R_DEFAULT = 8.314462618  # J/mol/K


@dataclass
class SegmentRate:
    label: str
    T_mean_K: float
    T_span_K: float
    r_abs: float  # |dm/dt|, same time units as input (Minutes for Netzsh data)
    slope_signed: float  # signed dm/dt from linear fit
    intercept: float  # intercept of mass vs time linear fit
    r2_mass_vs_time: float  # R^2 of linear fit for zero-order assumption
    n_points: int
    time_window: Tuple[float, float]


@dataclass
class ArrheniusResult:
    E_A_J_per_mol: float
    A: float
    slope: float
    intercept: float
    r2_ln_r_vs_invT: float
    n_points: int
    x_invT: np.ndarray
    y_lnr: np.ndarray
    extras: Dict[str, Any]


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Ordinary least squares fit of y = intercept + slope * x (with an intercept).
    Returns (slope, intercept, r2).
    """

    # Ensure we're working with 1D float arrays (copies if needed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Keep only finite (non-NaN, non-inf) matched pairs
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    # Need at least 2 points to fit a line
    n = x.size
    if n < 2:
        return (math.nan, math.nan, math.nan)

    # Sample means of x and y: x̄, ȳ
    xm, ym = x.mean(), y.mean()

    # Centered sums used by OLS:
    # Sxx = Σ (xi - x̄)^2   — variation of x
    # Sxy = Σ (xi - x̄)(yi - ȳ) — covariance term between x and y
    Sxx = ((x - xm) ** 2).sum()
    Sxy = ((x - xm) * (y - ym)).sum()

    # If Sxx == 0, all x are identical ⇒ slope undefined
    if Sxx == 0:
        return (math.nan, math.nan, math.nan)

    # OLS closed-form estimators:
    # slope = Sxy / Sxx
    # intercept = ȳ - slope * x̄
    slope = Sxy / Sxx
    intercept = ym - slope * xm

    # Fitted values: ŷi = intercept + slope * xi
    yhat = intercept + slope * x

    # Sum of squares:
    # SS_res = Σ (yi - ŷi)^2      — residual (unexplained) variation
    # SS_tot = Σ (yi - ȳ)^2       — total variation about the mean
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()

    # Coefficient of determination:
    # R^2 = 1 - SS_res / SS_tot  (with an intercept). If SS_tot == 0,
    # y has no variance (all equal), so R^2 is undefined.
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else math.nan

    # Return as plain floats
    return float(slope), float(intercept), float(r2)


def _slice_window(df: pd.DataFrame, time_col: str, t0: float, t1: float) -> pd.DataFrame:
    return df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()


def estimate_segment_rate(
        df: pd.DataFrame,
        *,
        time_window: Tuple[float, float],
        time_col: str = "time_min",
        temp_col: str = "temp_C",
        mass_col: str = "mass_pct",
        label: Optional[str] = None,
) -> SegmentRate:
    if label is None: label = "segment"
    t0, t1 = time_window
    sel = _slice_window(df, time_col, t0, t1)
    if sel.empty:
        raise ValueError(f"No data in time window {time_window}.")
    t = sel[time_col].to_numpy(dtype=float)
    m = sel[mass_col].to_numpy(dtype=float)
    T = sel[temp_col].to_numpy(dtype=float) + 273.15
    slope, intercept, r2 = _linear_fit(t, m)  # mass = intercept + slope * time
    r_abs = abs(slope)
    T_mean = float(np.nanmean(T))
    T_span = float(np.nanmax(T) - np.nanmin(T)) if len(T) > 0 else float("nan")
    return SegmentRate(
        label=str(label),
        T_mean_K=T_mean,
        T_span_K=T_span,
        r_abs=float(r_abs),
        slope_signed=float(slope),
        intercept=float(intercept),
        r2_mass_vs_time=float(r2),
        n_points=int(len(sel)),
        time_window=(float(t0), float(t1)),
    )


def estimate_arrhenius_from_segments(
        segments: Sequence[SegmentRate],
        *,
        R: float = R_DEFAULT,
        min_points: int = 2,
) -> ArrheniusResult:
    T = np.array([s.T_mean_K for s in segments], dtype=float)
    r = np.array([s.r_abs for s in segments], dtype=float)
    mask = np.isfinite(T) & (T > 0) & np.isfinite(r) & (r > 0)
    T = T[mask];
    r = r[mask]
    if T.size < min_points:
        raise ValueError(f"Need at least {min_points} valid segments (got {T.size}).")
    x = 1.0 / T
    y = np.log(r)
    slope, intercept, r2 = _linear_fit(x, y)  # y = intercept + slope*x
    E_A = -slope * R
    A = float(math.exp(intercept)) if np.isfinite(intercept) else float("nan")
    raw = dict(E_A_raw=E_A, A_raw=A, slope_raw=slope, intercept_raw=intercept, r2_raw=r2)
    extras = {
        "segments_used": [s.__dict__ for s in segments],
        "raw_fit": raw,
    }
    return ArrheniusResult(
        E_A_J_per_mol=float(E_A),
        A=float(A),
        slope=float(slope),
        intercept=float(intercept),
        r2_ln_r_vs_invT=float(r2),
        n_points=int(T.size),
        x_invT=x,
        y_lnr=y,
        extras=extras,
    )


def estimate_EA_A_from_dataframes(
        dfs: Sequence[pd.DataFrame],
        windows: Sequence[Tuple[float, float]],
        *,
        labels: Optional[Sequence[str]] = None,
        time_col: str = "time_min",
        temp_col: str = "temp_C",
        mass_col: str = "mass_pct",
        R: float = R_DEFAULT,
        enforce_non_negative: bool = True,
):
    if labels is None:
        labels = [f"df{i + 1}" for i in range(len(dfs))]
    if len(dfs) != len(windows):
        raise ValueError("dfs and windows must have the same length.")
    segments: List[SegmentRate] = []
    for df, win, lab in zip(dfs, windows, labels):
        seg = estimate_segment_rate(df, time_window=win, time_col=time_col,
                                    temp_col=temp_col, mass_col=mass_col, label=lab)
        segments.append(seg)
    result = estimate_arrhenius_from_segments(segments, R=R, enforce_non_negative=enforce_non_negative)
    return result, segments


def arrhenius_plot_data(segments: Sequence[SegmentRate]):
    T = np.array([s.T_mean_K for s in segments], dtype=float)
    r = np.array([s.r_abs for s in segments], dtype=float)
    mask = np.isfinite(T) & (T > 0) & np.isfinite(r) & (r > 0)
    x = 1.0 / T[mask]
    y = np.log(r[mask])
    return x, y
