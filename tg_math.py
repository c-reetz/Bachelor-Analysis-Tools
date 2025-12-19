from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Sequence, Dict, Any, Literal
import numpy as np
import pandas as pd
import math
import warnings


R_DEFAULT = 8.314462618  # J/mol/K


####
## Data classes and helper functions
####
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


@dataclass
class GlobalCR_O2_Result:
    label: str | None
    n_solid: float
    alpha_range: tuple[float, float]
    time_window: tuple[float, float]
    beta_K_per_time: float
    E_A_J_per_mol: float
    A: float
    m_o2: float
    intercept: float
    coef_lnO2: float
    coef_invT: float
    r2: float
    n_points: int
    dataset_point_counts: list[int]
    x_invT_all: np.ndarray
    z_lnO2_all: np.ndarray
    y_all: np.ndarray


@dataclass
class GlobalO2ArrheniusFit:
    """Global fit: ln(k) = ln(A) + n*ln(O2) - Ea/R*(1/T)."""
    label: str | None
    E_A_J_per_mol: float
    A: float
    n_o2: float  # if fixed or fitted
    slope_invT: float  # coefficient on (1/T)  -> should be negative
    intercept_lnA: float  # ln(A)
    r2: float
    n_points: int
    o2_basis: str  # "fraction" or "partial_pressure"
    o2_ref_total_pressure: float | None  # only meaningful if partial_pressure used

    def predict_k(self, T_K: float | np.ndarray, o2: float | np.ndarray) -> np.ndarray:
        """Predict k(T, O2) in the same time units as the input k (e.g. 1/min)."""
        T_K = np.asarray(T_K, dtype=float)
        o2 = np.asarray(o2, dtype=float)
        return self.A * np.power(o2, self.n_o2) * np.exp(-self.E_A_J_per_mol / (R_DEFAULT * T_K))


####
# Core fitting and estimation functions (Privates)
####
def _ols_multi(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ordinary least squares for multiple regressors.
    Returns (beta, r2).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[m]
    y = y[m]
    if X.shape[0] < X.shape[1] + 1:
        raise ValueError("Not enough rows for OLS fit (need > number of parameters).")

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, float(r2)


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


ConversionBasis = Literal["alpha", "carbon"]

ASH_FRACTION_DEFAULTS = {
    "BRF": 0.4008,
    "WS":  0.1728,
    "PW":  0.011463,
}

def _resolve_ash_fraction(feedstock: Optional[str], ash_fraction: Optional[float]) -> float:
    if ash_fraction is not None:
        af = float(ash_fraction)
    else:
        if not feedstock:
            raise ValueError("carbon basis requires ash_fraction or feedstock='BRF'/'WS'/'PW'")
        key = str(feedstock).strip().upper()
        if key not in ASH_FRACTION_DEFAULTS:
            raise ValueError(f"Unknown feedstock '{feedstock}'. Use BRF/WS/PW or pass ash_fraction.")
        af = ASH_FRACTION_DEFAULTS[key]
    if not (0.0 <= af < 1.0):
        raise ValueError(f"ash_fraction must be in [0,1). Got {af}")
    return af

def _compute_Xc_and_w_from_mass(
    mass_pct: np.ndarray,
    *,
    ash_fraction: float,
    m0_pct: Optional[float] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Carbon conversion from raw mass% within the selected window:
      m0 = max(mass%) in the window unless provided
      Xc = (m0 - m) / (m0*(1-ash))
      w  = 1 - Xc
    """
    m = np.asarray(mass_pct, dtype=float)
    if m0_pct is None:
        m0_pct = float(np.nanmax(m))
    denom = m0_pct * (1.0 - float(ash_fraction))
    if not np.isfinite(denom) or denom == 0:
        Xc = np.full_like(m, np.nan, dtype=float)
    else:
        Xc = (m0_pct - m) / denom

    Xc = np.clip(Xc, 0.0, 1.0)
    w = np.clip(1.0 - Xc, eps, 1.0)  # avoid log(0)
    return Xc, w, float(m0_pct)

def _compute_alpha_w(
    mass_pct: np.ndarray,
    *,
    m0_pct: Optional[float] = None,
    m_inf_pct: Optional[float] = None,
    head_frac: float = 0.10,
    tail_frac: float = 0.20,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute classic TG conversion alpha and remaining fraction w = 1 - alpha from a mass% window.

    - If m0_pct/m_inf_pct not provided, estimate from window head/tail medians.
    - Handles both mass-loss and mass-gain windows.
    Returns: (alpha, w, m0_used, m_inf_used)
    """
    m = np.asarray(mass_pct, dtype=float)
    m = m[np.isfinite(m)]
    if m.size < 3:
        alpha = np.full_like(np.asarray(mass_pct, dtype=float), np.nan, dtype=float)
        w = alpha.copy()
        return alpha, w, float("nan"), float("nan")

    # robust m0/m_inf from head/tail if not supplied
    N = m.size
    k_head = max(3, int(round(head_frac * N)))
    k_tail = max(3, int(round(tail_frac * N)))

    m0_used = float(np.nanmedian(m[:k_head])) if m0_pct is None else float(m0_pct)
    m_inf_used = float(np.nanmedian(m[-k_tail:])) if m_inf_pct is None else float(m_inf_pct)

    # decide loss vs gain based on endpoints in the window
    loss = (m[-1] < m[0])

    if loss:
        denom = (m0_used - m_inf_used)
        if not np.isfinite(denom) or abs(denom) < eps:
            denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
        alpha_win = (m0_used - m) / denom
    else:
        denom = (m_inf_used - m0_used)
        if not np.isfinite(denom) or abs(denom) < eps:
            denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
        alpha_win = (m - m0_used) / denom

    alpha_win = np.clip(alpha_win, 0.0, 1.0)
    w_win = np.clip(1.0 - alpha_win, eps, 1.0)

    # map back to original shape (including any NaNs that were removed)
    alpha = np.asarray(mass_pct, dtype=float)
    w = np.asarray(mass_pct, dtype=float)

    finite_mask = np.isfinite(alpha)
    alpha[finite_mask] = alpha_win
    w[finite_mask] = w_win

    alpha[~finite_mask] = np.nan
    w[~finite_mask] = np.nan

    return alpha, w, m0_used, m_inf_used


def _compute_conversion_and_w(
    mass_pct: np.ndarray,
    *,
    conversion_basis: ConversionBasis,
    feedstock: Optional[str] = None,
    ash_fraction: Optional[float] = None,
    m0_pct: Optional[float] = None,
    m_inf_pct: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns:
      X  = alpha  (if basis='alpha') OR Xc (if basis='carbon')
      w  = 1 - X
      meta dict (m0/m_inf/ash)
    """
    if conversion_basis == "alpha":
        alpha, w, m0_used, m_inf_used = _compute_alpha_w(
            mass_pct, m0_pct=m0_pct, m_inf_pct=m_inf_pct
        )
        return alpha, np.clip(w, 1e-12, 1.0), {"m0_pct": m0_used, "m_inf_pct": m_inf_used}

    elif conversion_basis == "carbon":
        af = _resolve_ash_fraction(feedstock, ash_fraction)
        Xc, w, m0_used = _compute_Xc_and_w_from_mass(
            mass_pct, ash_fraction=af, m0_pct=m0_pct
        )
        return Xc, np.clip(w, 1e-12, 1.0), {"m0_pct": m0_used, "ash_fraction": af}

    else:
        raise ValueError(f"Unknown conversion_basis: {conversion_basis}")



####
# Isothermal functions
####
# Isothermal holds rate constant over the segment.
def estimate_segment_rate_first_order(
    df: pd.DataFrame,
    *,
    time_window: Tuple[float, float],
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    label: Optional[str] = None,
    # conversion handling
    conversion_basis: str = "alpha",  # "alpha" or "carbon"
    ash_fraction: Optional[float] = None,  # required for carbon basis unless you hardcode elsewhere
    conversion_range: Optional[Tuple[float, float]] = None,  # overrides alpha_range if provided
    alpha_range: Tuple[float, float] = (0.10, 0.80),
    # robust reference points inside window
    head_frac: float = 0.10,
    tail_frac: float = 0.20,
    # alpha normalization behavior
    normalize_within_window: bool = False,
) -> SegmentRate:
    """
    FIRST-ORDER (solid) isothermal estimation.
    Fits ln(w) vs time inside the window:
        ln(w) = ln(w0) − k * t  => slope = −k

    conversion_basis="alpha":
      - normalize_within_window=True:
          X = (m0 − m)/(m0 − m_inf)  (or gain variant), w = 1 − X
      - normalize_within_window=False:
          w = m/m0_start (capped at 1), X = 1 − w

    conversion_basis="carbon":
      m0 = max(mass%) inside the time window (always)
      Xc = (m0 − m)/(m0*(1-ash_fraction)), w = 1 − Xc

    Safeguards:
      - If requested conversion_range is not covered in the window -> raises ValueError
      - If partially covered -> warns and clips to the overlap
    """
    import warnings

    if label is None:
        label = "segment"

    t0, t1 = time_window
    sel = df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()
    if sel.empty:
        raise ValueError(f"No data in time window {time_window}.")

    # numeric + sort
    sel[time_col] = pd.to_numeric(sel[time_col], errors="coerce")
    sel[temp_col] = pd.to_numeric(sel[temp_col], errors="coerce")
    sel[mass_col] = pd.to_numeric(sel[mass_col], errors="coerce")
    sel = sel.dropna(subset=[time_col, temp_col, mass_col]).sort_values(time_col)
    if sel.shape[0] < 5:
        raise ValueError(f"Too few points ({sel.shape[0]}) in time window {time_window}.")

    t = sel[time_col].to_numpy(dtype=float)
    T_K = sel[temp_col].to_numpy(dtype=float) + 273.15
    m = sel[mass_col].to_numpy(dtype=float)

    # rebase time to start of window
    t_rel = t - t[0]

    eps = 1e-12
    n = m.size
    k_head = max(3, int(round(head_frac * n)))
    k_tail = max(3, int(round(tail_frac * n)))

    # robust start/end masses inside the time window
    m0_start = float(np.nanmedian(m[:k_head]))
    m_inf = float(np.nanmedian(m[-k_tail:]))

    # ---- compute X (conversion) and w (remaining fraction) ----
    conversion_basis = str(conversion_basis).lower().strip()

    if conversion_basis == "alpha":
        if normalize_within_window:
            # old behavior: scale by the span inside the time window
            loss = (m_inf < m0_start)
            if loss:
                denom = (m0_start - m_inf)
                if not np.isfinite(denom) or abs(denom) < eps:
                    denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
                X = (m0_start - m) / denom
            else:
                denom = (m_inf - m0_start)
                if not np.isfinite(denom) or abs(denom) < eps:
                    denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
                X = (m - m0_start) / denom

            X = np.clip(X, 0.0, 1.0)
            w = np.clip(1.0 - X, 1e-12, 1.0)
        else:
            # requested behavior: 100% mass at window start, no forced 100% conversion
            if not np.isfinite(m0_start) or abs(m0_start) < eps:
                raise ValueError("Invalid m0_start in time window.")
            w = m / m0_start
            w = np.clip(w, 1e-12, np.inf)
            w = np.minimum(w, 1.0)  # cap mass gain -> X=0 there
            X = np.clip(1.0 - w, 0.0, 1.0)

    elif conversion_basis == "carbon":
        if ash_fraction is None or (not np.isfinite(float(ash_fraction))):
            raise ValueError("ash_fraction must be provided for conversion_basis='carbon'.")

        af = float(ash_fraction)

        # IMPORTANT: m0 is highest point inside THIS time window
        m0_top = float(np.nanmax(m))
        if not np.isfinite(m0_top) or abs(m0_top) < eps:
            raise ValueError("Invalid m0_top in time window for carbon conversion.")

        denom = m0_top * (1.0 - af)
        if not np.isfinite(denom) or abs(denom) < eps:
            raise ValueError("Invalid denominator for carbon conversion (check ash_fraction).")

        X = (m0_top - m) / denom
        X = np.clip(X, 0.0, 1.0)
        w = np.clip(1.0 - X, 1e-12, 1.0)

    else:
        raise ValueError(f"Unknown conversion_basis: {conversion_basis!r} (use 'alpha' or 'carbon').")

    # ---- apply conversion-range safeguards ----
    lo_req, hi_req = (conversion_range if conversion_range is not None else alpha_range)

    X_min = float(np.nanmin(X))
    X_max = float(np.nanmax(X))

    lo_eff = max(float(lo_req), X_min)
    hi_eff = min(float(hi_req), X_max)

    if hi_eff <= lo_eff:
        raise ValueError(
            f"{label}: requested conversion range [{lo_req:.2f},{hi_req:.2f}] not covered in time_window "
            f"(available [{X_min:.2f},{X_max:.2f}])."
        )

    if (lo_eff > float(lo_req) + 1e-12) or (hi_eff < float(hi_req) - 1e-12):
        warnings.warn(
            f"{label}: requested conversion range [{lo_req:.2f},{hi_req:.2f}] only partially covered "
            f"(available [{X_min:.2f},{X_max:.2f}]); using [{lo_eff:.2f},{hi_eff:.2f}] for this fit."
        )

    mask = np.isfinite(t_rel) & np.isfinite(w) & np.isfinite(T_K) & (T_K > 0) & (X > lo_eff) & (X < hi_eff)
    if int(np.sum(mask)) < 3:
        raise ValueError(f"{label}: insufficient points after conversion filtering.")

    x = t_rel[mask].astype(float)
    y = np.log(w[mask].astype(float))  # ln(w) = ln(w0) - k*t
    T_use = T_K[mask].astype(float)

    slope, intercept, r2 = _linear_fit(x, y)
    k = float(-slope)  # k >= 0 ideally

    if not np.isfinite(k):
        raise ValueError(f"{label}: could not estimate k (non-finite slope).")
    if k < 0:
        # keep it non-negative to avoid nonsense downstream
        k = 0.0

    return SegmentRate(
        label=str(label),
        T_mean_K=float(np.nanmean(T_use)),
        T_span_K=float(np.nanmax(T_use) - np.nanmin(T_use)),
        r_abs=float(k),                 # for first-order: r_abs stores k (1/time)
        slope_signed=float(slope),      # slope of ln(w) vs t (should be ~ -k)
        intercept=float(intercept),     # ln(w0)
        r2_mass_vs_time=float(r2),      # name kept for backward compat
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


#Global fitting. Multiple datasets
def estimate_global_arrhenius_with_o2_from_segments(
        segments: list,
        o2_values: list[float],
        *,
        # set this if you want to FORCE n=1 (or any fixed n)
        n_o2_fixed: float | None = None,
        # "fraction" uses y_O2 (0.05/0.10/0.20); "partial_pressure" uses p_O2 (e.g. bar)
        o2_basis: str = "fraction",
        total_pressure: float | None = None,  # only used if you pass fractions but want p_O2
        R: float = R_DEFAULT if "R_DEFAULT" in globals() else 8.314462618,
        label: str | None = None,
        enforce_non_negative: bool = True,
) -> GlobalO2ArrheniusFit:
    """
    Fit one global (Ea, A, n) across many isothermal segments:
        ln(k) = ln(A) + n*ln(O2) - Ea/R*(1/T)

    `segments` should contain objects with:
        - s.T_mean_K
        - s.r_abs   (k, must be >0)

    If n_o2_fixed is None -> n is fitted (requires >=2 unique O2 values).
    If n_o2_fixed is given -> fit only Ea and A (n fixed).
    """
    if len(segments) != len(o2_values):
        raise ValueError("segments and o2_values must have the same length.")
    if len(segments) < 3:
        raise ValueError("Need at least 3 segments for a stable global fit.")

    T = np.array([float(s.T_mean_K) for s in segments], dtype=float)
    k = np.array([float(s.r_abs) for s in segments], dtype=float)

    if np.any(~np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("Invalid T_mean_K in segments.")
    if np.any(~np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("All segment rates (r_abs) must be finite and >0 to take ln(k).")

    o2 = np.array([float(v) for v in o2_values], dtype=float)
    if np.any(~np.isfinite(o2)) or np.any(o2 <= 0):
        raise ValueError("All O2 values must be finite and >0.")

    # Choose O2 representation for the regression
    if o2_basis not in ("fraction", "partial_pressure"):
        raise ValueError("o2_basis must be 'fraction' or 'partial_pressure'.")

    if o2_basis == "fraction":
        ln_o2 = np.log(o2)  # dimensionless
        p_ref = None
    else:
        # if user supplied fractions but wants pO2: need total_pressure
        # here we assume o2_values already are pO2 in chosen units
        ln_o2 = np.log(o2)
        p_ref = total_pressure

    invT = 1.0 / T
    y = np.log(k)

    # If n fixed: subtract its contribution and fit y = ln(A) + (-Ea/R)*invT
    if n_o2_fixed is not None:
        n_val = float(n_o2_fixed)
        y_adj = y - n_val * ln_o2
        X = np.column_stack([np.ones_like(invT), invT])
        beta, r2 = _ols_multi(X, y_adj)
        lnA = float(beta[0])
        b_invT = float(beta[1])  # equals -Ea/R
    else:
        # Need at least 2 distinct O2 levels to estimate n
        if np.unique(np.round(ln_o2, 12)).size < 2:
            raise ValueError("Cannot fit n_o2: only one O2 level present. Provide n_o2_fixed.")
        X = np.column_stack([np.ones_like(invT), ln_o2, invT])
        beta, r2 = _ols_multi(X, y)
        lnA = float(beta[0])
        n_val = float(beta[1])
        b_invT = float(beta[2])

    E_A = float(-b_invT * R)
    A = float(math.exp(lnA)) if np.isfinite(lnA) else float("nan")

    if enforce_non_negative:
        if not np.isfinite(E_A) or E_A < 0:
            E_A = 0.0
        if not np.isfinite(A) or A < 0:
            A = 0.0

    return GlobalO2ArrheniusFit(
        label=label,
        E_A_J_per_mol=E_A,
        A=A,
        n_o2=n_val,
        slope_invT=b_invT,
        intercept_lnA=lnA,
        r2=r2,
        n_points=int(np.sum(np.isfinite(y) & np.isfinite(invT) & np.isfinite(ln_o2))),
        o2_basis=o2_basis,
        o2_ref_total_pressure=p_ref,
    )


#Convenience wrapper for above function
def estimate_global_arrhenius_with_o2_from_isothermal_datasets(
    dfs: list[pd.DataFrame],
    time_windows: list[tuple[float, float]],
    o2_fractions: list[float],
    *,
    # Conversion filtering (applies inside estimate_segment_rate_first_order)
    alpha_range: tuple[float, float] = (0.10, 0.60),                  # legacy name
    conversion_range: Optional[Tuple[float, float]] = None,           # preferred
    conversion_basis: ConversionBasis = "alpha",                      # "alpha" or "carbon"
    feedstock: Optional[str] = None,
    ash_fraction: Optional[float] = None,

    # columns
    time_col: str = "time_min",
    mass_col: str = "mass_pct",
    temp_col: str = "temp_C",

    # oxygen order handling
    n_o2_fixed: float | None = None,                                  # fix m (gas order) if desired
    label: str | None = None,
) -> GlobalO2ArrheniusFit:
    """
    Global fit from multiple ISOTHERMAL datasets.

    Workflow:
      1) For each dataset i (each at a different temperature T_i), extract k_i from a chosen time window
         using first-order solid kinetics on the chosen conversion basis:
             ln(w) = ln(w0) - k t
         where w = 1 - alpha   (conversion_basis="alpha")
               w = 1 - X_C     (conversion_basis="carbon")

      2) Global regression across datasets:
             ln(k) = ln(A) + n*ln(O2) - Ea/R * (1/T)

         If all O2 fractions are the same, the n*ln(O2) term is constant and folds into ln(A)
         (you still get Ea correctly; A becomes an apparent A').

    Parameters:
      - dfs, time_windows, o2_fractions must have same length.
      - time_windows can differ per dataset (common for different holds).
      - conversion_range is preferred; if None, alpha_range is used.

    Returns:
      GlobalO2ArrheniusFit with Ea, A, n_o2, r2, etc.
    """
    if len(dfs) != len(time_windows) or len(dfs) != len(o2_fractions):
        raise ValueError("dfs, time_windows, and o2_fractions must have the same length.")
    if len(dfs) < 2:
        raise ValueError("Need at least 2 isothermal datasets (3 recommended) to estimate Ea reliably.")

    # Basic validation of O2 inputs
    for y in o2_fractions:
        if not (np.isfinite(y) and y > 0):
            raise ValueError("All o2_fractions must be finite and > 0.")

    segments: list[SegmentRate] = []

    for i, (df, tw, yO2) in enumerate(zip(dfs, time_windows, o2_fractions), start=1):
        seg_label = f"{label or 'iso'}_{i}_O2_{yO2:g}"

        seg = estimate_segment_rate_first_order(
            df,
            time_window=tw,
            time_col=time_col,
            temp_col=temp_col,
            mass_col=mass_col,
            label=seg_label,

            # NEW: pass conversion controls through
            conversion_basis=conversion_basis,
            feedstock=feedstock,
            ash_fraction=ash_fraction,
            conversion_range=conversion_range,
            alpha_range=alpha_range,   # keep for backwards compatibility
        )
        segments.append(seg)

    return estimate_global_arrhenius_with_o2_from_segments(
        segments,
        o2_values=o2_fractions,
        n_o2_fixed=n_o2_fixed,
        o2_basis="fraction",
        label=label,
    )



####
# Non-isothermal Coats-Redfern fits
####
def estimate_global_coats_redfern_with_o2(
    dfs: list[pd.DataFrame],
    o2_fractions: list[float],
    *,
    time_window: tuple[float, float],
    n_solid: float = 1.0,  # solid reaction order used in g(w)
    alpha_range: tuple[float, float] = (0.10, 0.80),  # legacy name (used if conversion_range=None)
    beta_fixed_K_per_time: float = 3.0,  # 3 K/min if time is minutes
    # columns
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    # alpha normalization parameters (within time_window)
    head_frac: float = 0.10,
    tail_frac: float = 0.20,
    # NEW: alpha normalization behavior
    # True  -> old behavior: alpha scaled by (m0 - m_inf) within the time_window (can force 0→1 inside window)
    # False -> window-start mass is 100%: w = m/m0_start, X = 1-w (does NOT force 100% conversion)
    normalize_within_window: bool = False,
    # oxygen order handling
    m_o2_fixed: float | None = None,  # set to 1.0 if you want to force O2 order
    # fit options
    equal_weight_per_dataset: bool = True,  # avoids runs with more points dominating
    R: float = R_DEFAULT if "R_DEFAULT" in globals() else 8.314462618,
    label: str | None = None,
    enforce_non_negative: bool = True,
    conversion_basis: ConversionBasis = "alpha",  # "alpha" or "carbon"
    feedstock: Optional[str] = None,
    ash_fraction: Optional[float] = None,
    conversion_range: Optional[Tuple[float, float]] = None,  # preferred over alpha_range
):
    """
    Global Coats–Redfern fit across multiple linear-heating ramps at different O2 fractions.

    Model:
        y = ln(g(w)/T^2) = ln(A*R/(beta*Ea)) + m*ln(yO2) - Ea/R * (1/T)

    Conversion handling:
      - basis="alpha":
          * normalize_within_window=True:
              X = (m0 - m)/(m0 - m_inf)   (or gain variant)
          * normalize_within_window=False:
              w = m/m0_start (capped at 1); X = 1 - w
      - basis="carbon":
          * m0 is ALWAYS the highest mass% inside the time window:
              Xc = (m0_top - m)/(m0_top*(1-ash_fraction))
              w  = 1 - Xc
    Safeguards:
      - If requested conversion_range (or alpha_range) is not present in the time_window, warn and skip that dataset.
      - If partially present, warn and fit only the overlapping range.
      - Requires >=2 usable datasets after filtering.
    """

    if len(dfs) != len(o2_fractions):
        raise ValueError("dfs and o2_fractions must have the same length.")
    if len(dfs) < 2:
        raise ValueError("Need at least 2 datasets (preferably 3: 5%,10%,20%).")

    t0, t1 = time_window
    lo_req, hi_req = (conversion_range if conversion_range is not None else alpha_range)
    n_s = float(n_solid)

    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    counts: list[int] = []

    # --------------------------
    # FIT LOOP (with warnings)
    # --------------------------
    for idx, (df, yO2) in enumerate(zip(dfs, o2_fractions), start=1):
        ds_id = f"{label or 'CR'} dataset #{idx} (O2={float(yO2):g})"

        if not (np.isfinite(yO2) and yO2 > 0):
            raise ValueError("All o2_fractions must be finite and > 0.")

        seg = df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()
        if seg.empty:
            warnings.warn(f"{ds_id}: empty time_window, skipping.")
            counts.append(0)
            continue

        seg[time_col] = pd.to_numeric(seg[time_col], errors="coerce")
        seg[temp_col] = pd.to_numeric(seg[temp_col], errors="coerce")
        seg[mass_col] = pd.to_numeric(seg[mass_col], errors="coerce")
        seg = seg.dropna(subset=[time_col, temp_col, mass_col]).sort_values(time_col)

        if seg.shape[0] < 5:
            warnings.warn(f"{ds_id}: <5 points in time_window, skipping.")
            counts.append(0)
            continue

        T_K = seg[temp_col].to_numpy(dtype=float) + 273.15
        m = seg[mass_col].to_numpy(dtype=float)

        eps = 1e-12
        N = m.size
        k_head = max(3, int(round(head_frac * N)))
        k_tail = max(3, int(round(tail_frac * N)))

        # robust start / end masses (within time_window)
        m0_start = float(np.nanmedian(m[:k_head]))
        m_inf = float(np.nanmedian(m[-k_tail:]))

        # --- compute conversion X_conv and remaining fraction w = 1 - X_conv ---
        if conversion_basis == "alpha":
            if normalize_within_window:
                # old behavior: scale by window span (can force X→1 by definition)
                loss = (m_inf < m0_start)
                if loss:
                    denom = (m0_start - m_inf)
                    if not np.isfinite(denom) or abs(denom) < eps:
                        denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
                    X_conv = (m0_start - m) / denom
                else:
                    denom = (m_inf - m0_start)
                    if not np.isfinite(denom) or abs(denom) < eps:
                        denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
                    X_conv = (m - m0_start) / denom

                X_conv = np.clip(X_conv, 0.0, 1.0)
                w = np.clip(1.0 - X_conv, 1e-12, 1.0)

            else:
                # requested behavior: 100% mass at window start, no scaling to 1 at window end
                if not np.isfinite(m0_start) or abs(m0_start) < eps:
                    warnings.warn(f"{ds_id}: invalid m0_start, skipping.")
                    counts.append(0)
                    continue
                w = m / m0_start
                # if there is minor mass gain, cap at 1 -> X=0 there
                w = np.clip(w, 1e-12, np.inf)
                w = np.minimum(w, 1.0)
                X_conv = np.clip(1.0 - w, 0.0, 1.0)

        elif conversion_basis == "carbon":
            # m0 for carbon conversion = highest mass% inside THIS time window (always)
            af = _resolve_ash_fraction(feedstock, ash_fraction)
            m0_top = float(np.nanmax(m))
            if not np.isfinite(m0_top) or abs(m0_top) < eps:
                warnings.warn(f"{ds_id}: invalid m0_top for carbon basis, skipping.")
                counts.append(0)
                continue

            denom = m0_top * (1.0 - af)
            if not np.isfinite(denom) or abs(denom) < eps:
                warnings.warn(f"{ds_id}: invalid X_C denominator (ash_fraction?), skipping.")
                counts.append(0)
                continue

            X_conv = (m0_top - m) / denom
            X_conv = np.clip(X_conv, 0.0, 1.0)
            w = np.clip(1.0 - X_conv, 1e-12, 1.0)

        else:
            raise ValueError(f"Unknown conversion_basis: {conversion_basis}")

        # --- safeguard: requested conversion range vs available conversion in this window ---
        X_min = float(np.nanmin(X_conv))
        X_max = float(np.nanmax(X_conv))

        lo_eff = max(float(lo_req), X_min)
        hi_eff = min(float(hi_req), X_max)

        if hi_eff <= lo_eff:
            warnings.warn(
                f"{ds_id}: requested conversion range [{lo_req:.2f},{hi_req:.2f}] not covered in time_window "
                f"(available [{X_min:.2f},{X_max:.2f}]). Skipping this dataset."
            )
            counts.append(0)
            continue

        if (lo_eff > float(lo_req) + 1e-12) or (hi_eff < float(hi_req) - 1e-12):
            warnings.warn(
                f"{ds_id}: requested conversion range [{lo_req:.2f},{hi_req:.2f}] only partially covered "
                f"(available [{X_min:.2f},{X_max:.2f}]); using [{lo_eff:.2f},{hi_eff:.2f}] for this dataset."
            )

        mask = (
            np.isfinite(T_K) & (T_K > 0) &
            np.isfinite(X_conv) & (X_conv > lo_eff) & (X_conv < hi_eff) &
            np.isfinite(w)
        )
        if int(np.sum(mask)) < 3:
            warnings.warn(f"{ds_id}: <3 points after conversion filtering, skipping.")
            counts.append(0)
            continue

        T_fit = T_K[mask]
        w_fit = w[mask]

        # g(w) for chosen solid order
        if abs(n_s - 1.0) < 1e-12:
            g = -np.log(w_fit)
        else:
            g = (np.power(w_fit, 1.0 - n_s) - 1.0) / (n_s - 1.0)

        x_invT = 1.0 / T_fit
        y_cr = np.log(np.clip(g, 1e-300, np.inf) / (T_fit ** 2))
        z_lnO2 = np.log(float(yO2)) * np.ones_like(x_invT)

        mm = np.isfinite(x_invT) & np.isfinite(y_cr) & np.isfinite(z_lnO2)
        x_invT = x_invT[mm]
        y_cr = y_cr[mm]
        z_lnO2 = z_lnO2[mm]

        npts = int(x_invT.size)
        counts.append(npts)
        if npts < 3:
            warnings.warn(f"{ds_id}: <3 finite points after final masking, skipping.")
            continue

        # Design matrix: y = b0 + b1*lnO2 + b2*(1/T)
        if m_o2_fixed is not None:
            y_adj = y_cr - float(m_o2_fixed) * z_lnO2
            Xmat = np.column_stack([np.ones_like(x_invT), x_invT])
        else:
            Xmat = np.column_stack([np.ones_like(x_invT), z_lnO2, x_invT])
            y_adj = y_cr

        if equal_weight_per_dataset:
            wgt = 1.0 / max(1, npts)
            sw = math.sqrt(wgt)
            Xmat = Xmat * sw
            y_adj = y_adj * sw

        X_blocks.append(Xmat)
        y_blocks.append(y_adj)

    n_used = sum(c > 0 for c in counts)
    if n_used < 2 or (not X_blocks):
        raise ValueError(
            f"Global CR fit needs >=2 usable datasets, but only {n_used} were usable after filtering. "
            f"Try widening conversion_range/alpha_range or adjusting time_window."
        )

    # --------------------------
    # OLS fit (global)
    # --------------------------
    X_all = np.vstack(X_blocks)
    y_all_fit = np.concatenate(y_blocks)

    beta, *_ = np.linalg.lstsq(X_all, y_all_fit, rcond=None)
    yhat = X_all @ beta
    ss_res = float(np.sum((y_all_fit - yhat) ** 2))
    ss_tot = float(np.sum((y_all_fit - float(np.mean(y_all_fit))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if m_o2_fixed is not None:
        b0 = float(beta[0])
        b2 = float(beta[1])
        b1 = float(m_o2_fixed)
    else:
        b0 = float(beta[0])
        b1 = float(beta[1])  # oxygen order m
        b2 = float(beta[2])  # coefficient on 1/T

    E_A = float(-b2 * R)
    if enforce_non_negative and (not np.isfinite(E_A) or E_A < 0):
        E_A = 0.0

    # b0 = ln(A*R/(beta*Ea)) -> A = (beta*Ea/R)*exp(b0)
    beta_used = float(beta_fixed_K_per_time)
    if np.isfinite(b0) and np.isfinite(E_A) and E_A > 0 and beta_used > 0:
        A = float((beta_used * E_A / R) * math.exp(b0))
    else:
        A = float("nan")
    if enforce_non_negative and (not np.isfinite(A) or A < 0):
        A = 0.0

    # --------------------------
    # Rebuild unweighted points for plotting (NO warnings)
    # --------------------------
    x_plot_list: list[np.ndarray] = []
    y_plot_list: list[np.ndarray] = []
    z_plot_list: list[np.ndarray] = []

    for df, yO2 in zip(dfs, o2_fractions):
        seg = df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()
        if seg.empty:
            continue

        seg[time_col] = pd.to_numeric(seg[time_col], errors="coerce")
        seg[temp_col] = pd.to_numeric(seg[temp_col], errors="coerce")
        seg[mass_col] = pd.to_numeric(seg[mass_col], errors="coerce")
        seg = seg.dropna(subset=[time_col, temp_col, mass_col]).sort_values(time_col)
        if seg.shape[0] < 5:
            continue

        T_K = seg[temp_col].to_numpy(dtype=float) + 273.15
        m = seg[mass_col].to_numpy(dtype=float)

        eps = 1e-12
        N = m.size
        k_head = max(3, int(round(head_frac * N)))
        k_tail = max(3, int(round(tail_frac * N)))

        m0_start = float(np.nanmedian(m[:k_head]))
        m_inf = float(np.nanmedian(m[-k_tail:]))

        if conversion_basis == "alpha":
            if normalize_within_window:
                loss = (m_inf < m0_start)
                if loss:
                    denom = (m0_start - m_inf)
                    if not np.isfinite(denom) or abs(denom) < eps:
                        denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
                    X_conv = (m0_start - m) / denom
                else:
                    denom = (m_inf - m0_start)
                    if not np.isfinite(denom) or abs(denom) < eps:
                        denom = float(np.nanmax(m) - np.nanmin(m)) or 1.0
                    X_conv = (m - m0_start) / denom
                X_conv = np.clip(X_conv, 0.0, 1.0)
                w = np.clip(1.0 - X_conv, 1e-12, 1.0)
            else:
                if not np.isfinite(m0_start) or abs(m0_start) < eps:
                    continue
                w = m / m0_start
                w = np.clip(w, 1e-12, np.inf)
                w = np.minimum(w, 1.0)
                X_conv = np.clip(1.0 - w, 0.0, 1.0)

        elif conversion_basis == "carbon":
            af = _resolve_ash_fraction(feedstock, ash_fraction)
            m0_top = float(np.nanmax(m))
            if not np.isfinite(m0_top) or abs(m0_top) < eps:
                continue
            denom = m0_top * (1.0 - af)
            if not np.isfinite(denom) or abs(denom) < eps:
                continue
            X_conv = (m0_top - m) / denom
            X_conv = np.clip(X_conv, 0.0, 1.0)
            w = np.clip(1.0 - X_conv, 1e-12, 1.0)

        else:
            continue

        X_min = float(np.nanmin(X_conv))
        X_max = float(np.nanmax(X_conv))
        lo_eff = max(float(lo_req), X_min)
        hi_eff = min(float(hi_req), X_max)
        if hi_eff <= lo_eff:
            continue

        mask = (
            np.isfinite(T_K) & (T_K > 0) &
            np.isfinite(X_conv) & (X_conv > lo_eff) & (X_conv < hi_eff) &
            np.isfinite(w)
        )
        if int(np.sum(mask)) < 3:
            continue

        T_fit = T_K[mask]
        w_fit = w[mask]

        if abs(n_s - 1.0) < 1e-12:
            g = -np.log(w_fit)
        else:
            g = (np.power(w_fit, 1.0 - n_s) - 1.0) / (n_s - 1.0)

        x = 1.0 / T_fit
        yv = np.log(np.clip(g, 1e-300, np.inf) / (T_fit ** 2))
        zv = np.log(float(yO2)) * np.ones_like(x)

        mm = np.isfinite(x) & np.isfinite(yv) & np.isfinite(zv)
        if int(np.sum(mm)) < 1:
            continue

        x_plot_list.append(x[mm])
        y_plot_list.append(yv[mm])
        z_plot_list.append(zv[mm])

    x_plot = np.concatenate(x_plot_list) if x_plot_list else np.array([], dtype=float)
    y_plot = np.concatenate(y_plot_list) if y_plot_list else np.array([], dtype=float)
    z_plot = np.concatenate(z_plot_list) if z_plot_list else np.array([], dtype=float)

    return GlobalCR_O2_Result(
        label=label,
        n_solid=n_s,
        alpha_range=(float(lo_req), float(hi_req)),  # kept for backwards compatibility
        time_window=(float(t0), float(t1)),
        beta_K_per_time=float(beta_used),
        E_A_J_per_mol=float(E_A),
        A=float(A),
        m_o2=float(b1),
        intercept=float(b0),
        coef_lnO2=float(b1),
        coef_invT=float(b2),
        r2=float(r2),
        n_points=int(x_plot.size),
        dataset_point_counts=counts,
        x_invT_all=x_plot,
        z_lnO2_all=z_plot,
        y_all=y_plot,
    )





def simulate_alpha_ramp(
        *,
        time_min: np.ndarray,
        temp_C: np.ndarray,
        yO2: float,
        E_A_J_per_mol: float,
        A: float,
        m_o2: float,
        solid_order: int = 1,
        alpha0: float = 0.0,
        R: float = R_DEFAULT if "R_DEFAULT" in globals() else 8.314462618,
) -> np.ndarray:
    """
    Simulate conversion alpha(t) for a non-isothermal ramp using:
        dα/dt = k(T,yO2) * (1-α)^(solid_order)
        k = A * yO2^m_o2 * exp(-Ea/(R*T))

    time_min must be increasing; A will be in 1/min if time is minutes.
    Returns alpha array, clipped to [0,1].
    """
    t = np.asarray(time_min, dtype=float)
    T_K = np.asarray(temp_C, dtype=float) + 273.15

    if t.ndim != 1 or T_K.ndim != 1 or t.size != T_K.size:
        raise ValueError("time_min and temp_C must be 1D arrays of same length.")
    if t.size < 2:
        raise ValueError("Need at least 2 time points.")
    if not (np.isfinite(yO2) and yO2 > 0):
        raise ValueError("yO2 must be > 0.")
    if not (np.isfinite(A) and A > 0):
        raise ValueError("A must be > 0.")
    if not (np.isfinite(E_A_J_per_mol) and E_A_J_per_mol >= 0):
        raise ValueError("Ea must be finite and >= 0.")

    # Ensure increasing time
    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)
        t = t[idx]
        T_K = T_K[idx]

    alpha = np.empty_like(t)
    a = float(alpha0)
    a = min(max(a, 0.0), 1.0)
    alpha[0] = a

    yO2_term = float(yO2) ** float(m_o2)

    for i in range(1, t.size):
        dt = float(t[i] - t[i - 1])
        Ti = float(T_K[i - 1])
        if not np.isfinite(Ti) or Ti <= 0:
            alpha[i] = a
            continue

        k = float(A * yO2_term * np.exp(-float(E_A_J_per_mol) / (R * Ti)))

        if solid_order == 0:
            f = 1.0
        elif solid_order == 1:
            f = max(0.0, 1.0 - a)
        elif solid_order == 2:
            f = max(0.0, (1.0 - a) ** 2)
        else:
            f = max(0.0, (1.0 - a) ** float(solid_order))

        da = k * f * dt  # Increment a based on timestep, k is rate, f is fraction unreacted, dt is time step in min
        a = a + da  # update conversion for this time
        a = min(max(a, 0.0), 1.0)  #make sure it is <1 and >0
        alpha[i] = a  #add to list for time point i

    return alpha


def alpha_to_mass_pct(alpha: np.ndarray, m0: float, m_inf: float, *, loss: bool = True) -> np.ndarray:
    """
    Convert alpha(t) back to mass%.
    If loss=True:  alpha=(m0-m)/(m0-m_inf) -> m = m0 - alpha*(m0-m_inf)
    If loss=False (gain): alpha=(m-m0)/(m_inf-m0) -> m = m0 + alpha*(m_inf-m0)
    """
    a = np.asarray(alpha, dtype=float)
    if loss:
        return m0 - a * (m0 - m_inf)
    else:
        return m0 + a * (m_inf - m0)

def compute_dtg_curve(
    df: pd.DataFrame,
    *,
    time_window: tuple[float, float] | None = None,
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    smooth_window: int = 0,
    beta_min: float = 1e-6,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """
    Compute DTG (derivative thermogravimetry) for a TG dataset.

    Returns a DataFrame (time-ordered) with columns:
      - time_min
      - temp_C
      - mass_pct
      - dm_dt              [mass% / time-unit]
      - beta_dT_dt         [°C / time-unit]
      - dm_dT              [mass% / °C]
      - dtg_loss           [-dm/dT, mass% / °C]  (positive for mass loss)

    Notes:
      - DTG is only meaningful when temperature is changing (beta_dT_dt not ~0).
        Points with |beta_dT_dt| < beta_min are set to NaN in dm_dT/dtg_loss.
      - If you pass a time_window, normalization/slicing is strictly inside that window.
      - smooth_window (odd int recommended, e.g. 7, 11) applies a moving-average
        to mass and temperature *before* differentiation to reduce derivative noise.
    """
    import numpy as np
    import pandas as pd

    if time_window is not None:
        t0, t1 = time_window
        d = df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()
    else:
        d = df.copy()

    # numeric + drop NaNs
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[temp_col] = pd.to_numeric(d[temp_col], errors="coerce")
    d[mass_col] = pd.to_numeric(d[mass_col], errors="coerce")
    d = d.dropna(subset=[time_col, temp_col, mass_col])

    # ensure strictly increasing time (average duplicates)
    d = (
        d.groupby(time_col, as_index=False)[[temp_col, mass_col]]
        .mean()
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    if d.shape[0] < 5:
        raise ValueError("Too few points to compute DTG.")

    t = d[time_col].to_numpy(dtype=float)
    T = d[temp_col].to_numpy(dtype=float)
    m = d[mass_col].to_numpy(dtype=float)

    # optional moving-average smoothing
    if smooth_window and smooth_window >= 3:
        w = int(smooth_window)
        if w % 2 == 0:
            w += 1  # enforce odd
        kernel = np.ones(w, dtype=float) / float(w)
        # pad edges to avoid shrinking
        pad = w // 2
        m_pad = np.pad(m, (pad, pad), mode="edge")
        T_pad = np.pad(T, (pad, pad), mode="edge")
        m = np.convolve(m_pad, kernel, mode="valid")
        T = np.convolve(T_pad, kernel, mode="valid")

    # derivatives vs time
    dm_dt = np.gradient(m, t)       # mass% / time
    dT_dt = np.gradient(T, t)       # °C / time (heating rate)

    # dm/dT via chain rule
    dm_dT = np.full_like(dm_dt, np.nan, dtype=float)
    ok = np.isfinite(dm_dt) & np.isfinite(dT_dt) & (np.abs(dT_dt) >= float(beta_min))
    dm_dT[ok] = dm_dt[ok] / dT_dt[ok]

    # common DTG convention: positive peaks for mass loss
    dtg_loss = -dm_dT

    out = pd.DataFrame(
        {
            time_col: t,
            temp_col: T,
            mass_col: m,
            "dm_dt": dm_dt,
            "beta_dT_dt": dT_dt,
            "dm_dT": dm_dT,
            "dtg_loss": dtg_loss,
        }
    )

    if drop_invalid:
        out = out.dropna(subset=["dm_dT", "dtg_loss"]).reset_index(drop=True)

    return out



####
# Old, unused code, kept for reference
####
"""
# For isothermal holds. Not used, assuming first-order.
def estimate_segment_rate_zero_order(
        df: pd.DataFrame,
        *,
        time_window: Tuple[float, float],
        time_col: str = "time_min",
        temp_col: str = "temp_C",
        mass_col: str = "mass_pct",
        label: Optional[str] = None,
) -> SegmentRate:
    " " "
    zero-order isothermal estimation
    uses ln(r) where r=dm/dt = K
    K = A * exp(-E_A / R*T)
    Graph ends up being y = ln(r)
    x = 1/T
    so:
    ln(A) + -E_A/R * 1/T
    " " "
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


# This is not really used, because we do global fits with O2 Dependence, this does NOT include the O2 dependence!!!!
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
    " " "
    Coats–Redfern (integral) estimate of E_A and A on a selected time segment
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
    " " "
    
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


#pretty sure this is not what Hao meant, not really any use now
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
    " " "
    Global Coats–Redfern fit across multiple TG ramps.

    Combines all CR points from each dataframe (within the same time_window and alpha_range),
    performs a single linear regression:
        y = ln(g(w)/T^2)  vs  x = 1/T
    and returns ONE slope/intercept -> ONE E_A and ONE A.

    The pre-exponential A depends on heating rate beta:
        intercept = ln(A*R/(beta*E_A))
    If betas differ slightly across datasets, A becomes ambiguous.
    We therefore compute A using beta_ref = median(beta_i) of included datasets.
    " " "
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
"""
