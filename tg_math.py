
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any, Sequence, NamedTuple
import numpy as np
import pandas as pd
import math

R_DEFAULT = 8.314462618  # J/mol/K

@dataclass
class ArrheniusFitResult:
    method: Literal["derivative", "integral", "isothermal_k"]
    E_J_per_mol: Optional[float]
    A: Optional[float]
    A_prime: Optional[float]
    slope: Optional[float]
    intercept: Optional[float]
    r2: Optional[float]
    n_points: int
    T_span_K: float
    heating_rate_K_per_time: Optional[float]
    extras: Dict[str, Any]

def _select_region(
    df: pd.DataFrame,
    segment: Optional[str] = None,
    time_window: Optional[Tuple[float, float]] = None,
    time_col: str = "time_min",
    seg_col: str = "segment",
) -> pd.DataFrame:
    sel = df.copy()
    if segment is not None:
        sel = sel[sel[seg_col].astype(str) == str(segment)]
    if time_window is not None:
        t0, t1 = time_window
        sel = sel[(sel[time_col] >= t0) & (sel[time_col] <= t1)]
    sel = sel.dropna(subset=["temp_C", time_col, "mass_pct"])
    sel = sel.sort_values(by=time_col)
    if sel.empty:
        raise ValueError("No data left after applying segment/time filters.")
    return sel

def _to_kelvin(T_C: np.ndarray) -> np.ndarray:
    return np.asarray(T_C, dtype=float) + 273.15

def _compute_alpha_w(
    mass_pct: np.ndarray,
    m0_pct: Optional[float] = None,
    m_inf_pct: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mass_pct, dtype=float)
    if m0_pct is None:
        m0_pct = float(m[0])
    if m_inf_pct is None:
        m_inf_pct = float(np.nanmin(m))
    denom = (m0_pct - m_inf_pct)
    if denom == 0 or not np.isfinite(denom):
        raise ValueError("Invalid normalization: m0 and m_inf are equal or non-finite.")
    alpha = (m0_pct - m) / denom
    w = 1.0 - alpha
    return alpha, w

def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    ysm = np.convolve(ypad, kernel, mode="valid")
    return ysm

def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = x.size
    if n < 2:
        return (np.nan, np.nan, np.nan)
    x_mean = x.mean(); y_mean = y.mean()
    Sxy = ((x - x_mean) * (y - y_mean)).sum()
    Sxx = ((x - x_mean) ** 2).sum()
    slope = Sxy / Sxx if Sxx != 0 else np.nan
    intercept = y_mean - slope * x_mean
    y_pred = intercept + slope * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return slope, intercept, r2

def _estimate_heating_rate_K_per_time(T_K: np.ndarray, t: np.ndarray) -> float:
    dT = np.gradient(T_K, t)
    med = np.nanmedian(dT[int(0.1*len(dT)):int(0.9*len(dT))]) if len(dT) > 10 else np.nanmedian(dT)
    return float(med)

def _g_alpha(alpha: np.ndarray, n: float) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    if abs(n - 1.0) < 1e-12:
        with np.errstate(divide="ignore", invalid="ignore"):
            g = -np.log(np.clip(1.0 - a, 1e-300, 1.0))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            g = ((1.0 - a) ** (1.0 - n) - 1.0) / (1.0 - n)
    return g

class IsothermK(NamedTuple):
    T_K: float
    k: float
    r2: float
    segment: str | None
    time_window: tuple[float, float] | None
    m_inf_pct: float
    n: float
    points: int

def fit_isothermal_k(
    df: pd.DataFrame,
    *,
    segment: str | None = None,
    time_window: tuple[float, float] | None = None,
    n: float = 1.0,
    m_inf_pct: float | None = None,
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    tail_frac: float = 0.2,
) -> IsothermK:
    sel = _select_region(df, segment=segment, time_window=time_window, time_col=time_col, seg_col="segment")
    t = sel[time_col].to_numpy(dtype=float)
    T_K = _to_kelvin(sel[temp_col].to_numpy(dtype=float))
    m_pct = sel[mass_col].to_numpy(dtype=float)

    # --- estimate k from linearized nth-order model on mass% ---
    # Guess m_inf from tail if not provided
    def _estimate_minf_tail(m_pct: np.ndarray, tail_frac: float = 0.2) -> float:
        n = len(m_pct)
        if n < 5:
            return float(m_pct[-1])
        start = max(0, min(int((1.0 - tail_frac) * n), n-1))
        tail = m_pct[start:]
        return float(np.nanmedian(tail))

    def _monotonic_trend_sign(y: np.ndarray) -> int:
        if len(y) < 3: return 0
        x = np.arange(len(y), dtype=float)
        b, a, _ = _linear_fit(x, y)
        if np.isfinite(b): return 1 if b > 0 else (-1 if b < 0 else 0)
        return 0

    m = np.asarray(m_pct, dtype=float)
    t0 = float(t[0])
    t = t - t0

    if m_inf_pct is None:
        trend = _monotonic_trend_sign(m)
        guess = _estimate_minf_tail(m, tail_frac=tail_frac)
        if trend < 0:
            guess = min(guess, np.nanmin(m))
        elif trend > 0:
            guess = max(guess, np.nanmax(m))
        m_inf_pct = float(guess)

    m0 = float(m[0])
    denom = (m0 - m_inf_pct)
    if abs(denom) < 1e-12:
        span = float(np.nanmax(m) - np.nanmin(m))
        denom = denom if abs(denom) > 0 else (span if span != 0 else 1.0)

    w = (m - m_inf_pct) / denom

    mask = np.isfinite(t) & np.isfinite(w) & (w > 0)
    t = t[mask]; w = w[mask]
    if len(w) < 3:
        return IsothermK(float(np.nanmean(T_K)), float("nan"), float("nan"),
                         segment, time_window, float(m_inf_pct), n, points=len(sel))

    if abs(n - 1.0) < 1e-12:
        y = np.log(w); x = t
        slope, intercept, r2 = _linear_fit(x, y)
        k = -slope
    else:
        y = np.power(w, 1.0 - n); x = t
        slope, intercept, r2 = _linear_fit(x, y)
        k = -slope / (1.0 - n) if (1.0 - n) != 0 else float("nan")

    T_mean = float(np.nanmean(T_K))
    return IsothermK(T_mean, float(k), float(r2), segment, time_window, float(m_inf_pct), n, points=len(sel))

def fit_arrhenius_from_isotherms(
    df: pd.DataFrame,
    regions: Sequence[dict],
    *,
    n: float = 1.0,
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
) -> ArrheniusFitResult:
    iso: list[IsothermK] = []
    for spec in regions:
        seg = spec.get("segment")
        tw  = spec.get("time_window")
        item = fit_isothermal_k(
            df, segment=seg, time_window=tw, n=n,
            time_col=time_col, temp_col=temp_col, mass_col=mass_col
        )
        if np.isfinite(item.k) and item.k > 0 and np.isfinite(item.T_K) and item.T_K > 0:
            iso.append(item)

    if len(iso) < 2:
        raise ValueError("Need at least two valid isothermal regions to determine Ea and A.")

    T = np.array([it.T_K for it in iso], dtype=float)
    k = np.array([it.k   for it in iso], dtype=float)
    x = 1.0 / T
    y = np.log(k)
    slope, intercept, r2 = _linear_fit(x, y)
    E = -slope * R_DEFAULT
    A = float(np.exp(intercept))

    extras = {
        "isotherms": [it._asdict() for it in iso],
        "x": x.tolist(), "y": y.tolist()
    }
    return ArrheniusFitResult(
        method="derivative",
        E_J_per_mol=float(E),
        A=float(A),
        A_prime=float(A),
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n_points=len(x),
        T_span_K=float(np.nanmax(T) - np.nanmin(T)),
        heating_rate_K_per_time=None,
        extras=extras,
    )

# ---- NEW: O2-corrected Arrhenius utilities ----

def fit_arrhenius_from_isotherms_corrected(
    isotherms: Sequence[IsothermK],
    *,
    gas_order: float = 0.0,
    o2_fractions: Optional[Sequence[float]] = None,
    R: float = R_DEFAULT,
) -> ArrheniusFitResult:
    ks = np.array([it.k for it in isotherms], dtype=float)
    Ts = np.array([it.T_K for it in isotherms], dtype=float)

    if o2_fractions is not None and gas_order != 0.0:
        o2 = np.array(o2_fractions, dtype=float)
        if o2.size != ks.size:
            raise ValueError("Length of o2_fractions must match number of isotherms.")
        k_eff = ks / np.clip(o2 ** gas_order, 1e-300, np.inf)  # intrinsic k
        o2_applied = True
    else:
        k_eff = ks.copy()  # apparent k (includes gas dependence)
        o2_applied = False

    x = 1.0 / Ts
    y = np.log(k_eff)
    slope, intercept, r2 = _linear_fit(x, y)
    Ea = -slope * R
    A_intrinsic = float(np.exp(intercept))

    if not o2_applied:
        A_app = A_intrinsic
        A_true = None
    else:
        A_true = A_intrinsic
        A_app = None

    extras = {
        "isotherms": [it._asdict() for it in isotherms],
        "x": x.tolist(), "y": y.tolist(),
        "o2_correction_applied": o2_applied,
        "gas_order": gas_order,
        "o2_fractions": None if o2_fractions is None else list(map(float, o2_fractions)),
    }

    return ArrheniusFitResult(
        method="derivative",
        E_J_per_mol=float(Ea),
        A=A_true if o2_applied else A_app,
        A_prime=A_app if o2_applied else A_intrinsic,
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n_points=len(x),
        T_span_K=float(np.nanmax(Ts) - np.nanmin(Ts)),
        heating_rate_K_per_time=None,
        extras=extras,
    )

def arrhenius_from_three_files(
    specs: Sequence[dict],
    *, n_solid: float = 1.0,
    gas_order: float = 0.0,
    o2_fractions: Optional[Sequence[float]] = None,
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
) -> ArrheniusFitResult:
    iso = []
    for s in specs:
        df = s["df"]
        seg = s.get("segment")
        tw  = s.get("time_window")
        item = fit_isothermal_k(df, segment=seg, time_window=tw, n=n_solid,
                                time_col=time_col, temp_col=temp_col, mass_col=mass_col)
        if np.isfinite(item.k) and item.k > 0:
            iso.append(item)
    return fit_arrhenius_from_isotherms_corrected(iso, gas_order=gas_order, o2_fractions=o2_fractions)
