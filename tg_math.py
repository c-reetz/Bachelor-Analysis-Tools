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


def estimate_segment_rate_zero_order(
        df: pd.DataFrame,
        *,
        time_window: Tuple[float, float],
        time_col: str = "time_min",
        temp_col: str = "temp_C",
        mass_col: str = "mass_pct",
        label: Optional[str] = None,
) -> SegmentRate:
    """
    zero-order isothermal estimation
    uses ln(r) where r=dm/dt = K
    K = A * exp(-E_A / R*T)
    Graph ends up being y = ln(r)
    x = 1/T
    so:
    ln(A) + -E_A/R * 1/T
    """
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

def estimate_segment_rate_first_order(
    df: pd.DataFrame,
    *,
    time_window: Tuple[float, float],
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    label: Optional[str] = None,
) -> SegmentRate:
    """
    FIRST-ORDER (solid) isothermal estimation.
    Computes conversion α from mass%, then fits ln(1−α) vs time inside the window:
        ln(1−α) = ln(1−α0) − k * t   ⇒  slope = −k
    Returns SegmentRate with:
        r_abs        = k  (1/time; positive),
        slope_signed = slope of ln(1−α) vs t  (≈ −k),
        r2_mass_vs_time = R² of ln(1−α) vs t fit (name kept for backward compat).

    α definition (robust to loss/gain):
        If mass decreases:  α = (m0 − m)/(m0 − m∞)
        If mass increases:  α = (m − m0)/(m∞ − m0)
    where m0 is the median of the first 10% of points in the window,
          m∞ is the median of the last 20% of points in the window.
    """
    if label is None:
        label = "segment"

    # --- slice window and extract arrays ---
    t0, t1 = time_window
    sel = df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()
    if sel.empty:
        raise ValueError(f"No data in time window {time_window}.")

    t = sel[time_col].to_numpy(dtype=float)
    m = sel[mass_col].to_numpy(dtype=float)
    T_K = sel[temp_col].to_numpy(dtype=float) + 273.15

    # rebase time to start of window
    t = t - t[0]

    n = len(sel)
    if n < 5:
        raise ValueError("Not enough points in window for a reliable first-order fit (need ≥5).")

    # --- robust m0 and m∞ from window head/tail ---
    import numpy as np
    import math

    k_head = max(3, int(round(0.10 * n)))
    k_tail = max(3, int(round(0.20 * n)))

    m0 = float(np.nanmedian(m[:k_head]))
    minf_tail = float(np.nanmedian(m[-k_tail:]))

    # decide loss vs gain over the window
    loss = (m[-1] < m[0])

    # guard denominator
    eps = 1e-12
    if loss:
        denom = (m0 - minf_tail)
        if not np.isfinite(denom) or abs(denom) < eps:
            # fallback to span if needed
            denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
        alpha = (m0 - m) / denom
    else:
        denom = (minf_tail - m0)
        if not np.isfinite(denom) or abs(denom) < eps:
            denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
        alpha = (m - m0) / denom

    # clip α to (0,1) and build regression vectors
    alpha = np.clip(alpha, 1e-9, 1.0 - 1e-9)
    y = np.log(1.0 - alpha)       # should be linear vs time with slope = -k
    x = t.astype(float)

    # mask finite
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(T_K) & (T_K > 0)
    x = x[mask]; y = y[mask]; T_use = T_K[mask]
    if x.size < 3:
        raise ValueError("Insufficient finite points after masking for ln(1−α) vs t fit.")

    # linear fit: y = a + b x
    slope, intercept, r2 = _linear_fit(x, y)
    k = float(-slope)  # first-order rate constant (should be ≥ 0)

    # enforce k ≥ 0
    if not np.isfinite(k) or k < 0:
        k = 0.0
        slope = 0.0
        # r2 becomes 0 for a flat fallback
        r2 = 0.0

    T_mean = float(np.nanmean(T_use))
    T_span = float(np.nanmax(T_use) - np.nanmin(T_use)) if T_use.size else float("nan")

    return SegmentRate(
        label=str(label),
        T_mean_K=T_mean,
        T_span_K=T_span,
        r_abs=float(k),             # NOTE: now k (1/time), not |dm/dt|
        slope_signed=float(slope),  # slope of ln(1−α) vs t ≈ −k
        intercept=float(intercept), # intercept of ln(1−α) vs t
        r2_mass_vs_time=float(r2),  # R² of ln(1−α) vs t (name kept)
        n_points=int(x.size),
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


def arrhenius_plot_data(segments: Sequence[SegmentRate]):
    T = np.array([s.T_mean_K for s in segments], dtype=float)
    r = np.array([s.r_abs for s in segments], dtype=float)
    mask = np.isfinite(T) & (T > 0) & np.isfinite(r) & (r > 0)
    x = 1.0 / T[mask]
    y = np.log(r[mask])
    return x, y

def estimate_EA_A_nonisothermal_coats_redfern(
    df: pd.DataFrame,
    *,
    time_window: tuple[float, float],
    # solid reaction order in w = 1-α (i.e. dw/dt = -k w^n)
    n_solid: float = 1.0,
    # fit only points where α is in this range (computed within the time_window)
    alpha_range: tuple[float, float] = (0.10, 0.80),
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    R: float = R_DEFAULT if "R_DEFAULT" in globals() else 8.314462618,
    label: str | None = None,
    enforce_non_negative: bool = True,
    # robust m0 / m∞ estimation inside the time window
    head_frac: float = 0.10,
    tail_frac: float = 0.20,
) -> "CoatsRedfernResult":
    """
    Coats–Redfern (integral) estimate of E_A and A on a *selected time segment*
    of a non-isothermal linear heating ramp.

    We use w = 1-α (unreacted fraction), and assume:
        dw/dt = -k(T) * w^n_solid
        k(T) = A * exp(-E_A/(R*T))

    With constant heating rate beta = dT/dt, Coats–Redfern gives (approx.):
        ln( g(w) / T^2 ) = ln( A*R / (beta*E_A) ) - (E_A/R) * (1/T)

    where:
      - if n_solid = 1:  g(w) = -ln(w)
      - if n_solid ≠ 1:  g(w) = (w^(1-n) - 1)/(n-1)

    Parameters
    ----------
    df : DataFrame
        Must contain columns for time, temperature, mass.
    time_window : (t0, t1)
        Segment in the ramp to analyze, consistent with other functions.
        Time units determine A units (e.g., minutes -> A in 1/min).
    n_solid : float
        Reaction order with respect to solid (w).
    alpha_range : (α_low, α_high)
        Points used for regression within the segment.
    enforce_non_negative : bool
        If True, clamp negative E_A and A to 0.

    Returns
    -------
    CoatsRedfernResult
        A small dataclass-like object with attributes:
            E_A_J_per_mol, A, slope, intercept, r2, n_points,
            beta_K_per_time, x_invT, y_ln_g_over_T2,
            used_time_window, used_alpha_range, n_solid, label
    """

    @dataclass
    class CoatsRedfernResult:
        label: str | None
        n_solid: float
        used_time_window: tuple[float, float]
        used_alpha_range: tuple[float, float]
        beta_K_per_time: float
        slope: float
        intercept: float
        r2: float
        n_points: int
        E_A_J_per_mol: float
        A: float
        x_invT: np.ndarray
        y_ln_g_over_T2: np.ndarray

    if time_window is None:
        raise ValueError("time_window must be provided, e.g. (t0, t1).")

    t0, t1 = time_window
    seg = df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()
    if seg.empty:
        raise ValueError(f"No data in time_window={time_window}.")

    # numeric + sort
    seg[time_col] = pd.to_numeric(seg[time_col], errors="coerce")
    seg[temp_col] = pd.to_numeric(seg[temp_col], errors="coerce")
    seg[mass_col] = pd.to_numeric(seg[mass_col], errors="coerce")
    seg = seg.dropna(subset=[time_col, temp_col, mass_col]).sort_values(time_col)
    if seg.shape[0] < 5:
        raise ValueError("Not enough valid points in selected time window (need ≥5).")

    # time (relative) and temperature
    t = seg[time_col].to_numpy(dtype=float)
    t_rel = t - t[0]
    T_K = seg[temp_col].to_numpy(dtype=float) + 273.15
    m = seg[mass_col].to_numpy(dtype=float)

    # robust m0 and m_inf within the segment
    n = m.size
    k_head = max(3, int(round(head_frac * n)))
    k_tail = max(3, int(round(tail_frac * n)))
    m0 = float(np.nanmedian(m[:k_head]))
    m_inf = float(np.nanmedian(m[-k_tail:]))

    # mass loss vs gain
    loss = (m[-1] < m[0])
    eps = 1e-12
    if loss:
        denom = (m0 - m_inf)
        if not np.isfinite(denom) or abs(denom) < eps:
            denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
        alpha = (m0 - m) / denom
    else:
        denom = (m_inf - m0)
        if not np.isfinite(denom) or abs(denom) < eps:
            denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
        alpha = (m - m0) / denom

    alpha = np.clip(alpha, 0.0, 1.0)
    w = np.clip(1.0 - alpha, 1e-12, 1.0)  # avoid log/div by 0

    # estimate heating rate beta = dT/dt (K per time unit of time_col)
    # robust median gradient
    dTdt = np.gradient(T_K, t_rel)
    beta = float(np.nanmedian(dTdt[np.isfinite(dTdt)]))
    if not np.isfinite(beta) or beta <= 0:
        # fallback: endpoint slope
        dt_total = float(t_rel[-1] - t_rel[0])
        beta = float((T_K[-1] - T_K[0]) / dt_total) if dt_total > 0 else float("nan")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Could not estimate a positive heating rate beta from the selected segment.")

    # choose points by alpha_range; if too restrictive, auto-relax inside segment bounds
    a_low, a_high = alpha_range
    mask = (alpha > a_low) & (alpha < a_high) & np.isfinite(T_K) & (T_K > 0) & np.isfinite(t_rel)
    if mask.sum() < 3:
        # relax to interior alpha span of this segment (still avoiding endpoints)
        a_min = float(np.nanmin(alpha))
        a_max = float(np.nanmax(alpha))
        a_low2 = max(a_min + 1e-3, 0.01)
        a_high2 = min(a_max - 1e-3, 0.99)
        mask = (alpha > a_low2) & (alpha < a_high2) & np.isfinite(T_K) & (T_K > 0) & np.isfinite(t_rel)
        used_alpha_range = (a_low2, a_high2)
    else:
        used_alpha_range = (a_low, a_high)

    if mask.sum() < 3:
        raise ValueError(
            f"Not enough points for Coats–Redfern regression in alpha_range={alpha_range} "
            f"(segment alpha spans ~[{alpha.min():.3f}, {alpha.max():.3f}])."
        )

    T_fit = T_K[mask]
    w_fit = w[mask]

    # g(w)
    n_s = float(n_solid)
    if abs(n_s - 1.0) < 1e-12:
        g = -np.log(w_fit)  # g(w) = -ln(w)
    else:
        g = (np.power(w_fit, 1.0 - n_s) - 1.0) / (n_s - 1.0)

    # Coats–Redfern coords: y = ln(g/T^2), x = 1/T
    x = 1.0 / T_fit
    y = np.log(np.clip(g, 1e-300, np.inf) / (T_fit ** 2))

    # linear fit
    slope, intercept, r2 = _linear_fit(x, y)

    # extract EA and A
    E_A = float(-slope * R)  # J/mol
    if enforce_non_negative and (not np.isfinite(E_A) or E_A < 0):
        E_A = 0.0

    # intercept = ln(A*R/(beta*E))  ->  A = (beta*E/R) * exp(intercept)
    if np.isfinite(intercept) and np.isfinite(E_A) and E_A > 0 and beta > 0:
        A = float((beta * E_A / R) * math.exp(intercept))
    else:
        A = float("nan")

    if enforce_non_negative and (not np.isfinite(A) or A < 0):
        A = 0.0

    return CoatsRedfernResult(
        label=label,
        n_solid=n_s,
        used_time_window=(float(t0), float(t1)),
        used_alpha_range=(float(used_alpha_range[0]), float(used_alpha_range[1])),
        beta_K_per_time=float(beta),
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n_points=int(np.sum(np.isfinite(x) & np.isfinite(y))),
        E_A_J_per_mol=float(E_A),
        A=float(A),
        x_invT=np.asarray(x, dtype=float),
        y_ln_g_over_T2=np.asarray(y, dtype=float),
    )

def estimate_EA_A_nonisothermal_coats_redfern_global(
    dfs: list[pd.DataFrame],
    *,
    time_window: tuple[float, float],
    n_solid: float = 1.0,
    alpha_range: tuple[float, float] = (0.10, 0.80),
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    R: float = R_DEFAULT if "R_DEFAULT" in globals() else 8.314462618,
    labels: list[str] | None = None,
    enforce_non_negative: bool = True,
    head_frac: float = 0.10,
    tail_frac: float = 0.20,
    beta_fixed_K_per_time: float | None = 3.0,
):
    """
    Global Coats–Redfern fit across multiple TG ramps.

    Combines all CR points from each dataframe (within the same time_window and alpha_range),
    performs a single linear regression:
        y = ln(g(w)/T^2)  vs  x = 1/T
    and returns ONE slope/intercept -> ONE E_A and ONE A.

    The pre-exponential A depends on heating rate beta:
        intercept = ln(A*R/(beta*E_A))
    If betas differ slightly across datasets, A becomes ambiguous.
    We therefore compute A using beta_ref = median(beta_i) of included datasets.
    """
    import numpy as np
    import math
    from dataclasses import dataclass

    @dataclass
    class CoatsRedfernGlobalResult:
        n_solid: float
        used_time_window: tuple[float, float]
        used_alpha_range: tuple[float, float]
        slope: float
        intercept: float
        r2: float
        n_points: int
        E_A_J_per_mol: float
        A: float
        beta_ref_K_per_time: float
        betas_K_per_time: list[float]
        x_invT: np.ndarray
        y_ln_g_over_T2: np.ndarray
        dataset_point_counts: list[int]
        labels: list[str] | None

    if not dfs:
        raise ValueError("dfs must contain at least one DataFrame.")
    if labels is not None and len(labels) != len(dfs):
        raise ValueError("labels must be None or have same length as dfs.")

    t0, t1 = time_window
    n_s = float(n_solid)
    a_low, a_high = alpha_range

    all_x = []
    all_y = []
    betas = []
    counts = []

    for i, df in enumerate(dfs):
        seg = df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()
        if seg.empty:
            counts.append(0)
            continue

        # numeric + sort
        seg[time_col] = pd.to_numeric(seg[time_col], errors="coerce")
        seg[temp_col] = pd.to_numeric(seg[temp_col], errors="coerce")
        seg[mass_col] = pd.to_numeric(seg[mass_col], errors="coerce")
        seg = seg.dropna(subset=[time_col, temp_col, mass_col]).sort_values(time_col)
        if seg.shape[0] < 5:
            counts.append(0)
            continue

        t = seg[time_col].to_numpy(dtype=float)
        t_rel = t - t[0]
        T_K = seg[temp_col].to_numpy(dtype=float) + 273.15
        m = seg[mass_col].to_numpy(dtype=float)

        # robust m0 / m_inf in window
        npts = m.size
        k_head = max(3, int(round(head_frac * npts)))
        k_tail = max(3, int(round(tail_frac * npts)))
        m0 = float(np.nanmedian(m[:k_head]))
        m_inf = float(np.nanmedian(m[-k_tail:]))

        loss = (m[-1] < m[0])
        eps = 1e-12
        if loss:
            denom = (m0 - m_inf)
            if not np.isfinite(denom) or abs(denom) < eps:
                denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
            alpha = (m0 - m) / denom
        else:
            denom = (m_inf - m0)
            if not np.isfinite(denom) or abs(denom) < eps:
                denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
            alpha = (m - m0) / denom

        alpha = np.clip(alpha, 0.0, 1.0)
        w = np.clip(1.0 - alpha, 1e-12, 1.0)

        # heating rate beta
        dTdt = np.gradient(T_K, t_rel)
        beta_i = float(np.nanmedian(dTdt[np.isfinite(dTdt)]))
        if not np.isfinite(beta_i) or beta_i <= 0:
            dt_total = float(t_rel[-1] - t_rel[0])
            beta_i = float((T_K[-1] - T_K[0]) / dt_total) if dt_total > 0 else float("nan")
        if not np.isfinite(beta_i) or beta_i <= 0:
            counts.append(0)
            continue

        # mask by alpha range
        mask = (
            np.isfinite(T_K) & (T_K > 0) &
            np.isfinite(alpha) &
            (alpha > a_low) & (alpha < a_high)
        )
        if mask.sum() < 3:
            counts.append(0)
            continue

        T_fit = T_K[mask]
        w_fit = w[mask]

        # g(w)
        if abs(n_s - 1.0) < 1e-12:
            g = -np.log(w_fit)
        else:
            g = (np.power(w_fit, 1.0 - n_s) - 1.0) / (n_s - 1.0)

        x = 1.0 / T_fit
        y = np.log(np.clip(g, 1e-300, np.inf) / (T_fit ** 2))

        mm = np.isfinite(x) & np.isfinite(y)
        x = x[mm]
        y = y[mm]

        counts.append(int(x.size))
        if x.size < 3:
            continue

        all_x.append(x)
        all_y.append(y)
        betas.append(beta_i)

    if not all_x:
        raise ValueError("No usable Coats–Redfern points found across datasets for the given window/range.")


    X = np.concatenate(all_x)
    Y = np.concatenate(all_y)

    # One global OLS fit
    slope, intercept, r2 = _linear_fit(X, Y)

    E_A = float(-slope * R)
    if enforce_non_negative and (not np.isfinite(E_A) or E_A < 0):
        E_A = 0.0

    if beta_fixed_K_per_time is not None:
        beta_ref = float(beta_fixed_K_per_time)
    else:
        beta_ref = float(np.nanmedian(np.array(betas, dtype=float))) if betas else float("nan")

    if beta_fixed_K_per_time is not None and betas:
        b = np.array(betas, dtype=float)
        rel_err = np.nanmax(np.abs(b - beta_fixed_K_per_time) / beta_fixed_K_per_time)
        if np.isfinite(rel_err) and rel_err > 0.05:
            raise ValueError(
                f"Estimated beta varies >5% from beta_fixed={beta_fixed_K_per_time}. "
                f"Range: {np.nanmin(b):.3g}..{np.nanmax(b):.3g} (K/time)"
            )

    if np.isfinite(intercept) and np.isfinite(E_A) and E_A > 0 and np.isfinite(beta_ref) and beta_ref > 0:
        A = float((beta_ref * E_A / R) * math.exp(intercept))
    else:
        A = float("nan")

    if enforce_non_negative and (not np.isfinite(A) or A < 0):
        A = 0.0

    return CoatsRedfernGlobalResult(
        n_solid=n_s,
        used_time_window=(float(t0), float(t1)),
        used_alpha_range=(float(a_low), float(a_high)),
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n_points=int(np.sum(np.isfinite(X) & np.isfinite(Y))),
        E_A_J_per_mol=float(E_A),
        A=float(A),
        beta_ref_K_per_time=float(beta_ref),
        betas_K_per_time=[float(b) for b in betas],
        x_invT=np.asarray(X, dtype=float),
        y_ln_g_over_T2=np.asarray(Y, dtype=float),
        dataset_point_counts=counts,
        labels=labels,
    )
