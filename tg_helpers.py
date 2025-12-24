import math
import numpy as np
from pprint import pprint
import re
import pandas as pd
from tg_math import estimate_segment_rate_first_order, ASH_FRACTION_DEFAULTS, GlobalO2ArrheniusFit, \
    estimate_global_arrhenius_with_o2_from_segments
from tg_math import GlobalCR_O2_Result


R_GAS = 8.314462618


def print_global_cr_o2_result(res):
    """
    Pretty-print a tg_math.GlobalCR_O2_Result without dumping huge arrays.
    """

    # Scalars / metadata
    summary = {
        "label": res.label,
        "n_solid": res.n_solid,                 # your model order (you set this to 1)
        "alpha_range_used": res.alpha_range,    # conversion window actually used (may be clipped)
        "time_window_used": res.time_window,
        "beta_K_per_time": res.beta_K_per_time, # heating rate in K / (time unit)

        # Main fit results
        "E_A_kJ_per_mol": res.E_A_J_per_mol / 1000.0,
        "A": res.A,
        "lnA": (math.log(res.A) if res.A > 0 else float("nan")),
        "m_o2": res.m_o2,
        "r2": res.r2,

        # Regression internals (useful for debugging / appendix)
        "intercept": res.intercept,
        "coef_lnO2": res.coef_lnO2,
        "coef_invT": res.coef_invT,

        # Point counts
        "N_pts_total": res.n_points,                 # total points used across all datasets
        "N_pts_per_dataset": res.dataset_point_counts,
    }

    pprint(summary)

    # Array summaries (so you can verify content without printing all values)
    def _arr_stats(name, a):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            print(f"{name}: EMPTY")
            return
        print(
            f"{name}: shape={a.shape}, "
            f"min={np.nanmin(a):.6g}, max={np.nanmax(a):.6g}, "
            f"finite={np.isfinite(a).sum()}/{a.size}"
        )

  #  _arr_stats("x_invT_all (1/T)", res.x_invT_all)
  #  _arr_stats("z_lnO2_all (ln O2)", res.z_lnO2_all)
  #  _arr_stats("y_all (ln(g/T^2))", res.y_all)

def _parse_temp_from_regime_key(k: str) -> int | None:
    m = re.search(r"(\d{3})", str(k))
    return int(m.group(1)) if m else None


def _parse_o2_fraction(o2_label: str) -> float:
    s = str(o2_label).strip().lower().replace("o2", "")
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)


def predict_k_from_cr(cr: GlobalCR_O2_Result, *, T_K: float, yO2: float) -> float:
    return float(cr.A * (yO2 ** cr.m_o2) * math.exp(-cr.E_A_J_per_mol / (R_GAS * T_K)))


def _valid_df(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, str):
        return obj.strip() != ""
    if not hasattr(obj, "columns"):
        return False
    if getattr(obj, "empty", False):
        return False
    return True


def _carbon_Xmax_in_window(df: pd.DataFrame, time_window: tuple[float, float], ash_fraction: float) -> float:
    t0, t1 = time_window
    sel = df[(df["time_min"] >= t0) & (df["time_min"] <= t1)].copy()
    if sel.empty:
        return float("nan")
    m = pd.to_numeric(sel["mass_pct"], errors="coerce").to_numpy(float)
    m = m[np.isfinite(m)]
    if m.size < 5:
        return float("nan")
    m0_top = float(np.nanmax(m))
    denom = m0_top * (1.0 - float(ash_fraction))
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    X = (m0_top - m) / denom
    X = np.clip(X, 0.0, 1.0)
    return float(np.nanmax(X))


def compare_cr_to_char_isothermals(
    cr: GlobalCR_O2_Result,
    char_data: dict,
    *,
    char_name: str = "BRF",
    temps_C: tuple[int, ...] = (225, 250),
    o2_labels: tuple[str, ...] = ("5%", "10%", "20%"),
    conversion_basis: str = "carbon",
    ash_fraction: float | None = None,
    # --- NEW: common conversion window controls ---
    enforce_common_conversion: bool = True,
    common_hi: float | None = None,        # if None, auto = min(Xmax)*common_hi_frac
    common_hi_frac: float = 0.90,
    min_common_hi: float = 0.01,           # don't go below this unless you force common_hi
    # fit behavior
    alpha_range: tuple[float, float] = (0.0, 1.0),  # only used if enforce_common_conversion=False
    tol_C: float = 2.0,
    trim_start_min: float = 0.2,
    trim_end_min: float = 0.2,
):
    """
    Uses data["BRF"] and compares extracted isothermal k to CR-predicted k.
    If enforce_common_conversion=True, fits ALL runs on the same carbon-conversion window [0, common_hi].
    """

    conv = str(conversion_basis).lower().strip()
    if conv == "carbon" and ash_fraction is None:
        ash_fraction = ASH_FRACTION_DEFAULTS[str(char_name).upper()]

    # ---- collect candidate (df, T_C, yO2, time_window, label) ----
    candidates = []
    for regime_key, o2_map in char_data.items():
        rk = str(regime_key).lower()
        if "isotherm" not in rk:
            continue
        T_C = _parse_temp_from_regime_key(regime_key)
        if T_C is None or T_C not in temps_C:
            continue
        if not isinstance(o2_map, dict):
            continue

        for o2_lab in o2_labels:
            df = o2_map.get(o2_lab, None)
            if not _valid_df(df):
                continue

            tw = infer_isothermal_time_window(
                df,
                target_temp_C=float(T_C),
                tol_C=tol_C,
                trim_start_min=trim_start_min,
                trim_end_min=trim_end_min,
            )

            yO2 = _parse_o2_fraction(o2_lab)
            candidates.append((regime_key, int(T_C), float(yO2), o2_lab, df, tw))

    if not candidates:
        return pd.DataFrame()

    # ---- first pass: find common_hi from Xmax overlap ----
    if enforce_common_conversion and conv == "carbon":
        xmaxs = []
        for _, _, _, _, df, tw in candidates:
            xmaxs.append(_carbon_Xmax_in_window(df, tw, ash_fraction=float(ash_fraction)))
        xmaxs = np.asarray(xmaxs, float)
        xmaxs = xmaxs[np.isfinite(xmaxs)]

        if xmaxs.size == 0:
            raise ValueError("Could not compute Xmax for common conversion window (check columns/time windows).")

        auto_hi = float(np.min(xmaxs) * common_hi_frac)
        if common_hi is None:
            common_hi = max(auto_hi, min_common_hi)
        # If user explicitly set common_hi, respect it even if tiny.

    rows = []
    for regime_key, T_C, yO2, o2_lab, df, tw in candidates:
        # use a consistent conversion window if requested
        if enforce_common_conversion and conv == "carbon":
            seg = estimate_segment_rate_first_order(
                df,
                time_window=tw,
                label=f"{char_name}_iso_{T_C}C_{o2_lab}",
                conversion_basis="carbon",
                ash_fraction=float(ash_fraction),
                conversion_range=(0.0, float(common_hi)),
                normalize_within_window=False,
            )
        else:
            seg = estimate_segment_rate_first_order(
                df,
                time_window=tw,
                label=f"{char_name}_iso_{T_C}C_{o2_lab}",
                conversion_basis=conversion_basis,
                ash_fraction=float(ash_fraction) if conv == "carbon" else None,
                alpha_range=alpha_range,
                normalize_within_window=False,
            )

        k_iso = float(seg.r_abs)
        T_mean_K = float(seg.T_mean_K)
        k_cr = predict_k_from_cr(cr, T_K=T_mean_K, yO2=yO2)
        ratio = (k_cr / k_iso) if k_iso > 0 else float("nan")

        rows.append({
            "T_C": float(T_C),
            "yO2": float(yO2),
            "k_iso_1_per_min": k_iso,
            "k_CR_pred_1_per_min": k_cr,
            "CR/ISO_ratio": ratio,
            "percent_error_%": (ratio - 1.0) * 100.0 if np.isfinite(ratio) else float("nan"),
            "iso_r2": float(seg.r2_mass_vs_time),
            "iso_n_points": int(seg.n_points),
            "time_window": tw,
            "regime": str(regime_key),
            "common_hi_used": float(common_hi) if (enforce_common_conversion and conv == "carbon") else float("nan"),
        })

    return pd.DataFrame(rows).sort_values(["T_C", "yO2"]).reset_index(drop=True)


def infer_isothermal_time_window(
    df: pd.DataFrame,
    target_temp_C: float,
    *,
    tol_C: float = 2.0,
    min_points: int = 30,
    trim_start_min: float = 0.2,
    trim_end_min: float = 0.2,
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    seg_col: str = "segment",
) -> tuple[float, float]:
    """
    Pick the segment whose median temperature is closest to target_temp_C.
    Returns (t0, t1) trimmed.
    """
    if seg_col in df.columns:
        best_seg = None
        best_score = None
        for seg_id, sdf in df.groupby(seg_col, dropna=True):
            if len(sdf) < min_points:
                continue
            Tmed = float(np.nanmedian(pd.to_numeric(sdf[temp_col], errors="coerce").to_numpy()))
            if not np.isfinite(Tmed):
                continue
            score = abs(Tmed - target_temp_C)
            if score <= tol_C and (best_score is None or score < best_score):
                best_score = score
                best_seg = seg_id

        if best_seg is not None:
            sdf = df[df[seg_col] == best_seg]
            t0 = float(np.nanmin(pd.to_numeric(sdf[time_col], errors="coerce").to_numpy())) + trim_start_min
            t1 = float(np.nanmax(pd.to_numeric(sdf[time_col], errors="coerce").to_numpy())) - trim_end_min
            if t1 <= t0:
                raise ValueError("Invalid trimmed time window (t1<=t0).")
            return (t0, t1)

    # Fallback: temperature mask
    T = pd.to_numeric(df[temp_col], errors="coerce").to_numpy(float)
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    mask = np.isfinite(T) & np.isfinite(t) & (np.abs(T - target_temp_C) <= tol_C)
    idx = np.where(mask)[0]
    if idx.size < min_points:
        raise ValueError(f"Could not infer isothermal region near {target_temp_C}°C.")
    t0 = float(t[idx[0]]) + trim_start_min
    t1 = float(t[idx[-1]]) - trim_end_min
    if t1 <= t0:
        raise ValueError("Invalid trimmed time window (t1<=t0).")
    return (t0, t1)


def extract_isothermal_segments_from_char_data(
    char_data: dict,
    *,
    char_name: str,
    temps_C: tuple[int, ...] = (225, 250),
    o2_labels: tuple[str, ...] = ("5%", "10%", "20%"),
    conversion_basis: str = "carbon",
    ash_fraction: float | None = None,
    alpha_range: tuple[float, float] = (0.0, 1.0),
    tol_C: float = 2.0,
    trim_start_min: float = 0.2,
    trim_end_min: float = 0.2,
) -> tuple[list, list[float], pd.DataFrame]:
    """
    Collect isothermal SegmentRate objects + O2 values from data["BRF"]-style dict.
    Returns: (segments, o2_values, table_of_extracted_k)
    """
    conv = str(conversion_basis).lower().strip()
    if conv == "carbon" and ash_fraction is None:
        ash_fraction = ASH_FRACTION_DEFAULTS[str(char_name).upper()]

    segments = []
    o2_values: list[float] = []
    rows = []

    for regime_key, o2_map in char_data.items():
        rk = str(regime_key).lower()

        # be tolerant: "isothermal_225", "isotherm_225", even typos like "isotherml_250"
        if "isotherm" not in rk:
            continue

        T_C = _parse_temp_from_regime_key(regime_key)
        if T_C is None or T_C not in temps_C:
            continue

        if not isinstance(o2_map, dict):
            continue

        for o2_lab in o2_labels:
            df = o2_map.get(o2_lab, None)
            if df is None:
                continue
            if isinstance(df, str):
                # skip placeholders like "" (your main currently sets some to "")
                if df.strip() == "":
                    continue
                # if it is a non-empty string, it's likely a bug in wiring
                raise TypeError(f"{char_name}/{regime_key}/{o2_lab}: expected DataFrame, got str: {df!r}")

            if not _valid_df(df):
                continue

            yO2 = _parse_o2_fraction(o2_lab)
            tw = infer_isothermal_time_window(
                df,
                target_temp_C=float(T_C),
                tol_C=tol_C,
                trim_start_min=trim_start_min,
                trim_end_min=trim_end_min,
            )

            seg = estimate_segment_rate_first_order(
                df,
                time_window=tw,
                label=f"{char_name}_iso_{T_C}C_{o2_lab}",
                conversion_basis=conversion_basis,
                ash_fraction=ash_fraction,   # used only if carbon basis
                alpha_range=alpha_range,
                normalize_within_window=False,
            )

            segments.append(seg)
            o2_values.append(float(yO2))

            rows.append({
                "char": char_name,
                "regime": str(regime_key),
                "T_C": float(T_C),
                "yO2": float(yO2),
                "time_window": tw,
                "T_mean_C": float(seg.T_mean_K - 273.15),
                "T_mean_K": float(seg.T_mean_K),
                "k_iso_1_per_min": float(seg.r_abs),
                "iso_r2": float(seg.r2_mass_vs_time),
                "iso_n_points": int(seg.n_points),
            })

    tbl = pd.DataFrame(rows).sort_values(["T_C", "yO2"]).reset_index(drop=True)
    return segments, o2_values, tbl


# -------------------------
# (a) Fit isothermal global law from data["BRF"]
# -------------------------
def fit_isothermal_global_from_char_data(
    char_data: dict,
    *,
    char_name: str,
    temps_C: tuple[int, ...] = (225, 250),
    o2_labels: tuple[str, ...] = ("5%", "10%", "20%"),
    conversion_basis: str = "carbon",
    ash_fraction: float | None = None,
    alpha_range: tuple[float, float] = (0.0, 1.0),
    tol_C: float = 2.0,
    trim_start_min: float = 0.2,
    trim_end_min: float = 0.2,
    o2_basis: str = "fraction",
    n_o2_fixed: float | None = None,
) -> tuple[GlobalO2ArrheniusFit, pd.DataFrame]:
    """
    Returns (iso_global_fit, tbl_iso_extracted) where:
      iso_global_fit: GlobalO2ArrheniusFit  (A, Ea, n)
      tbl_iso_extracted: per-run extracted k table (k_iso from ln(w) vs t)
    """
    segments, o2_values, tbl = extract_isothermal_segments_from_char_data(
        char_data,
        char_name=char_name,
        temps_C=temps_C,
        o2_labels=o2_labels,
        conversion_basis=conversion_basis,
        ash_fraction=ash_fraction,
        alpha_range=alpha_range,
        tol_C=tol_C,
        trim_start_min=trim_start_min,
        trim_end_min=trim_end_min,
    )

    if len(segments) < 3:
        raise ValueError(f"{char_name}: need >=3 isothermal segments to fit global law (got {len(segments)}).")

    fit = estimate_global_arrhenius_with_o2_from_segments(
        segments=segments,
        o2_values=o2_values,
        n_o2_fixed=n_o2_fixed,
        o2_basis=o2_basis,
        label=f"{char_name} isothermal global fit",
    )
    return fit, tbl


# -------------------------
# (b) Compare CR vs isothermal-global on same (T, O2)
# -------------------------
def compare_cr_vs_isothermal_global_on_isothermals(
    cr: GlobalCR_O2_Result,
    iso_fit: GlobalO2ArrheniusFit,
    tbl_iso_extracted: pd.DataFrame,
) -> pd.DataFrame:
    """
    Input tbl_iso_extracted from fit_isothermal_global_from_char_data (includes T_mean_K, yO2, k_iso).
    Output table adds:
      - k_CR_pred
      - k_isoGlobal_pred
      - CR/ISO, ISOglob/ISO, CR/ISOglob ratios
    """
    if tbl_iso_extracted is None or tbl_iso_extracted.empty:
        raise ValueError("tbl_iso_extracted is empty.")

    out = tbl_iso_extracted.copy()

    T_K = out["T_mean_K"].to_numpy(float)
    yO2 = out["yO2"].to_numpy(float)

    k_cr = np.array([predict_k_from_cr(cr, T_K=float(T), yO2=float(y)) for T, y in zip(T_K, yO2)], dtype=float)
    k_iso_glob = iso_fit.predict_k(T_K=T_K, o2=yO2).astype(float)

    out["k_CR_pred_1_per_min"] = k_cr
    out["k_isoGlobal_pred_1_per_min"] = k_iso_glob

    k_iso = out["k_iso_1_per_min"].to_numpy(float)
    eps = 1e-30

    out["CR/ISO_ratio"] = (k_cr + eps) / (k_iso + eps)
    out["ISOglob/ISO_ratio"] = (k_iso_glob + eps) / (k_iso + eps)
    out["CR/ISOglob_ratio"] = (k_cr + eps) / (k_iso_glob + eps)

    out["CR_vs_ISO_error_%"] = (out["CR/ISO_ratio"] - 1.0) * 100.0
    out["ISOglob_vs_ISO_error_%"] = (out["ISOglob/ISO_ratio"] - 1.0) * 100.0
    out["CR_vs_ISOglob_diff_%"] = (out["CR/ISOglob_ratio"] - 1.0) * 100.0

    return out.sort_values(["T_C", "yO2"]).reset_index(drop=True)