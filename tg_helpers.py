import math
import numpy as np
from pprint import pprint, pformat
import re
import pandas as pd
from tg_math import (
    estimate_segment_rate_first_order,
    ASH_FRACTION_DEFAULTS,
    GlobalO2ArrheniusFit,
    estimate_global_arrhenius_with_o2_from_segments,
    _resolve_ash_fraction,
    _compute_Xc_and_w_from_mass,
    _compute_alpha_w,
    simulate_alpha_ramp,
    alpha_to_mass_pct,
)
from tg_math import GlobalCR_O2_Result
from tg_loader import SPEC #debug
from pathlib import Path
import matplotlib.pyplot as plt

R_GAS = 8.314462618


def format_global_cr_o2_result(res) -> str:
    """
    Return a formatted summary string for a tg_math.GlobalCR_O2_Result.
    (Safe for writing to file.)
    """
    summary = {
        "label": res.label,
        "n_solid": res.n_solid,
        "alpha_range_used": res.alpha_range,
        "time_window_used": res.time_window,
        "beta_K_per_time": res.beta_K_per_time,

        "E_A_kJ_per_mol": res.E_A_J_per_mol / 1000.0,
        "A": res.A,
        "m_o2": res.m_o2,
        "r2": res.r2,
    }
    return pformat(summary, sort_dicts=False)


def print_global_cr_o2_result(res):
    """
    Pretty-print AND return a formatted summary string.
    """
    txt = format_global_cr_o2_result(res)
    print(txt)
    return txt


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
    # common conversion window controls
    enforce_common_conversion: bool = True,
    common_hi: float | dict[int, float] | None = None,
    common_hi_frac: float = 0.90,
    min_common_hi: float = 0.01,
    common_per_temperature: bool = True,
    # time-window inference
    tol_C: float = 2.0,
    trim_start_min: float = 0.2,
    trim_end_min: float = 0.2,
    start_at_mass_peak: bool = True,
    peak_extra_start_min: float = 0.0,
    # segment-rate fit options
    alpha_range: tuple[float, float] = (0.0, 1.0),
    # debug / robustness
    debug: bool = False,
    min_points_for_fit: int = 10,
    skip_on_error: bool = True,
) -> pd.DataFrame:
    """
    Compare CR-predicted k(T,yO2) against extracted isothermal k from each hold.

    debug=True prints:
      - file path (SPEC)
      - inferred time window + points inside it
      - skips/errors per dataset (so you see exactly which file breaks)

    skip_on_error=True prevents the first bad run from killing the whole report.
    """
    conv = str(conversion_basis).lower().strip()
    if conv == "carbon" and ash_fraction is None:
        ash_fraction = ASH_FRACTION_DEFAULTS[str(char_name).upper()]

    def _refine_window_to_mass_peak(df: pd.DataFrame, tw: tuple[float, float]) -> tuple[float, float]:
        t0, t1 = tw
        sel = df[(df["time_min"] >= t0) & (df["time_min"] <= t1)].copy()
        if sel.empty:
            return tw
        t = pd.to_numeric(sel["time_min"], errors="coerce").to_numpy(float)
        m = pd.to_numeric(sel["mass_pct"], errors="coerce").to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(m)
        t = t[mask]
        m = m[mask]
        if t.size < 5:
            return tw
        t_peak = float(t[int(np.nanargmax(m))])
        t0_new = max(float(t0), t_peak + float(peak_extra_start_min))
        if float(t1) <= t0_new:
            return tw
        return (t0_new, float(t1))

    if debug:
        print(f"\n[CR↔ISO] {char_name} | CR='{getattr(cr, 'label', '')}'")

    # ---- collect candidates: (regime_key, T_C, yO2, o2_lab, df, tw) ----
    candidates: list[tuple[str, int, float, str, pd.DataFrame, tuple[float, float]]] = []

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
            if df is None or not _valid_df(df):
                continue

            src = SPEC.get(str(char_name).upper(), {}).get(str(regime_key), {}).get(str(o2_lab), None)

            # infer time window
            try:
                # If your infer function supports debug args, use them; otherwise fall back
                try:
                    tw = infer_isothermal_time_window(
                        df,
                        target_temp_C=float(T_C),
                        tol_C=tol_C,
                        trim_start_min=trim_start_min,
                        trim_end_min=trim_end_min,
                        debug=bool(debug),
                        debug_prefix=f"[CR↔ISO] {char_name} {regime_key} {o2_lab} ",
                    )
                except TypeError:
                    tw = infer_isothermal_time_window(
                        df,
                        target_temp_C=float(T_C),
                        tol_C=tol_C,
                        trim_start_min=trim_start_min,
                        trim_end_min=trim_end_min,
                    )

                if start_at_mass_peak:
                    tw = _refine_window_to_mass_peak(df, tw)

            except Exception as e:
                msg = f"[CR↔ISO][SKIP] window inference failed | {char_name} | {regime_key} | {o2_lab} | src={src} | err={e}"
                if debug:
                    print(msg)
                if skip_on_error:
                    continue
                raise

            yO2 = _parse_o2_fraction(o2_lab)

            if debug:
                sel = df[(df["time_min"] >= tw[0]) & (df["time_min"] <= tw[1])]
                dur = float(tw[1] - tw[0])
                tmed = float(pd.to_numeric(sel["temp_C"], errors="coerce").median())
                print(
                    f"[CR↔ISO] cand | {regime_key} | {T_C}C | {o2_lab} | yO2={yO2:.3f} | "
                    f"tw={tw} (dur={dur:.2f} min) | sel_n={len(sel)} | src={src} | Tmed={tmed:.2f}"
                )

            candidates.append((str(regime_key), int(T_C), float(yO2), str(o2_lab), df, tw))

    if not candidates:
        if debug:
            print(f"[CR↔ISO] No isothermal candidates found for {char_name}. Keys={list(char_data.keys())}")
        return pd.DataFrame([])

    # ---- determine common conversion window(s) if requested ----
    common_hi_by_T: dict[int, float] = {}

    if enforce_common_conversion and conv == "carbon":
        if isinstance(common_hi, dict):
            common_hi_by_T = {int(k): float(v) for k, v in common_hi.items()}
        elif isinstance(common_hi, (int, float)):
            for T_C in temps_C:
                common_hi_by_T[int(T_C)] = float(common_hi)
        else:
            if common_per_temperature:
                for T_C in temps_C:
                    xmaxs = []
                    for _, TT, _, _, df, tw in candidates:
                        if int(TT) != int(T_C):
                            continue
                        xmaxs.append(_carbon_Xmax_in_window(df, tw, ash_fraction=float(ash_fraction)))
                    xmaxs = np.asarray(xmaxs, float)
                    xmaxs = xmaxs[np.isfinite(xmaxs)]
                    if xmaxs.size == 0:
                        continue
                    min_x = float(np.min(xmaxs))
                    auto_hi = float(min_x * common_hi_frac)
                    hi = max(auto_hi, min_common_hi)
                    hi = min(hi, min_x)
                    common_hi_by_T[int(T_C)] = float(hi)
            else:
                xmaxs = []
                for _, _, _, _, df, tw in candidates:
                    xmaxs.append(_carbon_Xmax_in_window(df, tw, ash_fraction=float(ash_fraction)))
                xmaxs = np.asarray(xmaxs, float)
                xmaxs = xmaxs[np.isfinite(xmaxs)]
                if xmaxs.size == 0:
                    raise ValueError("Could not compute Xmax for common conversion window.")
                min_x = float(np.min(xmaxs))
                auto_hi = float(min_x * common_hi_frac)
                hi = max(auto_hi, min_common_hi)
                hi = min(hi, min_x)
                for T_C in temps_C:
                    common_hi_by_T[int(T_C)] = float(hi)

        if debug:
            print(f"[CR↔ISO] common_hi_by_T({char_name}) = {common_hi_by_T}")

        if not common_hi_by_T:
            raise ValueError("enforce_common_conversion=True but could not determine common_hi_by_T.")

    # ---- fit each run and compare to CR ----
    rows = []

    for regime_key, T_C, yO2, o2_lab, df, tw in candidates:
        src = SPEC.get(str(char_name).upper(), {}).get(str(regime_key), {}).get(str(o2_lab), None)
        sel = df[(df["time_min"] >= tw[0]) & (df["time_min"] <= tw[1])]
        sel_n = int(sel.shape[0])

        if sel_n < int(min_points_for_fit):
            msg = (
                f"[CR↔ISO][SKIP] Too few points | {char_name} | {regime_key} | {T_C}C | {o2_lab} | "
                f"tw={tw} | sel_n={sel_n} | src={src}"
            )
            if debug:
                print(msg)
            if skip_on_error:
                continue
            raise ValueError(msg)

        if enforce_common_conversion and conv == "carbon":
            hi = float(common_hi_by_T[int(T_C)])
        else:
            hi = float(alpha_range[1])

        try:
            if enforce_common_conversion and conv == "carbon":
                seg = estimate_segment_rate_first_order(
                    df,
                    time_window=tw,
                    label=f"{char_name}_iso_{T_C}C_{o2_lab}",
                    conversion_basis="carbon",
                    ash_fraction=float(ash_fraction),
                    conversion_range=(0.0, float(hi)),
                    normalize_within_window=False,
                )
                common_hi_used = float(hi)
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
                common_hi_used = float("nan")

        except Exception as e:
            msg = (
                f"[CR↔ISO][SKIP] rate fit failed | {char_name} | {regime_key} | {T_C}C | {o2_lab} | "
                f"tw={tw} | sel_n={sel_n} | src={src} | err={e}"
            )
            if debug:
                print(msg)
            if skip_on_error:
                continue
            raise

        k_iso = float(seg.r_abs)               # first-order k (1/min)
        T_mean_K = float(seg.T_mean_K)
        k_cr = predict_k_from_cr(cr, T_K=T_mean_K, yO2=float(yO2))
        ratio = (k_cr / k_iso) if (np.isfinite(k_iso) and k_iso > 0) else float("nan")

        if debug:
            print(
                f"[CR↔ISO] fit  | {char_name} | {regime_key} | {T_C}C | {o2_lab} | "
                f"k_iso={k_iso:.3e} | k_CR={k_cr:.3e} | err%={(ratio-1)*100 if np.isfinite(ratio) else float('nan'):.2f} | src={src}"
            )

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
            "common_hi_used": common_hi_used,
            "src": src,
        })

    out = pd.DataFrame(rows)
    if debug:
        print(f"[CR↔ISO] Done {char_name}: kept {len(out)} / {len(candidates)} candidates.\n")

    if out.empty:
        return out

    return out.sort_values(["T_C", "yO2"]).reset_index(drop=True)



def infer_isothermal_time_window(
    df: pd.DataFrame,
    target_temp_C: float,
    *,
    tol_C: float = 2.0,
    min_points: int = 30,
    min_duration_min: float = 5.0,   # refuse tiny windows
    trim_start_min: float = 0.2,
    trim_end_min: float = 0.2,
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    seg_col: str = "segment",
) -> tuple[float, float]:
    """
    Pick the segment whose median temperature is closest to target_temp_C,
    but reject segments that are too short or have too few valid points.

    Falls back to a temperature mask if no usable segment exists.
    Returns (t0, t1) trimmed.
    """

    # numeric copies
    t_all = pd.to_numeric(df.get(time_col, np.nan), errors="coerce").to_numpy(float)
    T_all = pd.to_numeric(df.get(temp_col, np.nan), errors="coerce").to_numpy(float)
    m_all = pd.to_numeric(df.get(mass_col, np.nan), errors="coerce").to_numpy(float)

    def _valid_counts(mask: np.ndarray) -> int:
        return int(np.sum(mask & np.isfinite(t_all) & np.isfinite(T_all) & np.isfinite(m_all)))

    def _duration(mask: np.ndarray) -> float:
        tt = t_all[mask & np.isfinite(t_all)]
        if tt.size == 0:
            return float("nan")
        return float(np.nanmax(tt) - np.nanmin(tt))

    # ---- Segment-based selection (preferred if segment column exists) ----
    if seg_col in df.columns:
        best_seg = None
        best_score = None
        best_t0 = None
        best_t1 = None

        # groupby on original df, but evaluate using numeric arrays by index
        for seg_id, sdf in df.groupby(seg_col, dropna=True):
            idx = sdf.index.to_numpy()

            # build mask for this segment indices
            mask_seg = np.zeros(len(df), dtype=bool)
            # indexes might not be 0..N-1; use position mask safely
            # easiest: use .iloc positions by converting to positional indices
            # If df has default RangeIndex, idx == positions; otherwise, map:
            try:
                pos = df.index.get_indexer(idx)
                mask_seg[pos[pos >= 0]] = True
            except Exception:
                # fallback: if index mapping fails, skip
                continue

            # require enough valid points for rate fit
            n_valid = _valid_counts(mask_seg)
            if n_valid < min_points:
                continue

            # require enough duration
            dur = _duration(mask_seg)
            if not np.isfinite(dur) or dur < min_duration_min:
                continue

            Tmed = float(np.nanmedian(T_all[mask_seg & np.isfinite(T_all)]))
            if not np.isfinite(Tmed):
                continue

            score = abs(Tmed - target_temp_C)
            if score > tol_C:
                continue

            # candidate time bounds using min/max time (robust)
            tt = t_all[mask_seg & np.isfinite(t_all)]
            tmin = float(np.nanmin(tt))
            tmax = float(np.nanmax(tt))

            t0 = tmin + trim_start_min
            t1 = tmax - trim_end_min
            if t1 <= t0:
                continue

            if best_score is None or score < best_score:
                best_score = score
                best_seg = seg_id
                best_t0, best_t1 = t0, t1

        if best_seg is not None:
            return (float(best_t0), float(best_t1))

    # ---- Fallback: temperature mask over entire df ----
    mask = (
        np.isfinite(t_all) & np.isfinite(T_all) & np.isfinite(m_all)
        & (np.abs(T_all - target_temp_C) <= tol_C)
    )

    n_valid = int(np.sum(mask))
    if n_valid < min_points:
        raise ValueError(f"Could not infer isothermal region near {target_temp_C}°C (valid points={n_valid}).")

    # IMPORTANT: use min/max time (not first/last index)
    tmin = float(np.nanmin(t_all[mask]))
    tmax = float(np.nanmax(t_all[mask]))
    dur = tmax - tmin
    if not np.isfinite(dur) or dur < min_duration_min:
        raise ValueError(
            f"Inferred region near {target_temp_C}°C is too short (duration={dur:.3g} min). "
            f"Try increasing tol_C or fixing segment labels."
        )

    t0 = tmin + trim_start_min
    t1 = tmax - trim_end_min
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
    # NEW
    start_at_mass_peak: bool = True,
    peak_extra_start_min: float = 0.0,
    min_points_for_fit: int = 10,
    min_duration_min: float = 3.0,
    skip_on_error: bool = True,
    debug: bool = False,
) -> tuple[list, list[float], pd.DataFrame]:
    """
    Collect isothermal SegmentRate objects + O2 values from data["BRF"]-style dict.
    Returns: (segments, o2_values, table_of_extracted_k)

    Robust behavior:
      - Any dataset that fails window inference or segment fitting is SKIPPED (if skip_on_error=True)
      - A row is still emitted to tbl with status='SKIP' and err message
    """
    if conversion_basis.lower().strip() == "carbon" and ash_fraction is None:
        ash_fraction = ASH_FRACTION_DEFAULTS[str(char_name).upper()]

    segments: list = []
    o2_values: list[float] = []
    rows = []

    for regime_key, o2_map in char_data.items():
        rk = str(regime_key).lower()

        # Only isothermals here (avoid ramp files entirely)
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
                if df.strip() == "":
                    continue
                raise TypeError(f"{char_name}/{regime_key}/{o2_lab}: expected DataFrame, got str: {df!r}")
            if not _valid_df(df):
                continue

            src = SPEC.get(str(char_name).upper(), {}).get(str(regime_key), {}).get(str(o2_lab), None)
            yO2 = _parse_o2_fraction(o2_lab)

            # -------- infer time window --------
            try:
                tw = infer_isothermal_time_window(
                    df,
                    target_temp_C=float(T_C),
                    tol_C=tol_C,
                    trim_start_min=trim_start_min,
                    trim_end_min=trim_end_min,
                )
            except Exception as e:
                msg = f"window_inference_failed: {e}"
                if debug:
                    print(f"[ISO-EXTRACT][SKIP] {char_name} | {regime_key} | {o2_lab} | src={src} | {msg}")
                rows.append({
                    "char": char_name,
                    "regime": str(regime_key),
                    "T_C": float(T_C),
                    "yO2": float(yO2),
                    "o2_label": str(o2_lab),
                    "src": src,
                    "time_window": None,
                    "status": "SKIP",
                    "err": msg,
                })
                if skip_on_error:
                    continue
                raise

            # -------- optional refine start at mass peak (SAFE) --------
            tw_before = tw
            if start_at_mass_peak:
                try:
                    tw = refine_window_to_mass_peak(df, tw, extra_start_min=float(peak_extra_start_min))
                except Exception:
                    tw = tw_before

                # If refinement collapses the window, ignore it
                sel = df[(df["time_min"] >= tw[0]) & (df["time_min"] <= tw[1])]
                dur = float(tw[1] - tw[0])
                if sel.shape[0] < min_points_for_fit or dur < min_duration_min:
                    if debug:
                        print(
                            f"[ISO-EXTRACT][PEAKREF-IGNORE] {char_name} | {regime_key} | {o2_lab} | "
                            f"tw_refined={tw} (n={sel.shape[0]}, dur={dur:.2f} min) -> using tw={tw_before} | src={src}"
                        )
                    tw = tw_before

            # -------- fit segment rate --------
            sel = df[(df["time_min"] >= tw[0]) & (df["time_min"] <= tw[1])]
            if debug:
                tmed = float(pd.to_numeric(sel["temp_C"], errors="coerce").median())
                print(
                    f"[ISO-EXTRACT] {char_name} | {regime_key} | {o2_lab} | T={T_C}C | yO2={yO2:.3f} | "
                    f"tw={tw} (dur={tw[1]-tw[0]:.2f} min) | sel_n={sel.shape[0]} | Tmed={tmed:.2f} | src={src}"
                )

            if sel.shape[0] < min_points_for_fit:
                msg = f"too_few_points: {sel.shape[0]} in tw={tw}"
                if debug:
                    print(f"[ISO-EXTRACT][SKIP] {char_name} | {regime_key} | {o2_lab} | src={src} | {msg}")
                rows.append({
                    "char": char_name,
                    "regime": str(regime_key),
                    "T_C": float(T_C),
                    "yO2": float(yO2),
                    "o2_label": str(o2_lab),
                    "src": src,
                    "time_window": tw,
                    "status": "SKIP",
                    "err": msg,
                })
                if skip_on_error:
                    continue
                raise ValueError(msg)

            try:
                seg = estimate_segment_rate_first_order(
                    df,
                    time_window=tw,
                    label=f"{char_name}_iso_{T_C}C_{o2_lab}",
                    conversion_basis=conversion_basis,
                    ash_fraction=ash_fraction,
                    alpha_range=alpha_range,
                    normalize_within_window=False,
                )
            except Exception as e:
                msg = f"segment_fit_failed: {e}"
                if debug:
                    print(f"[ISO-EXTRACT][SKIP] {char_name} | {regime_key} | {o2_lab} | src={src} | {msg}")
                rows.append({
                    "char": char_name,
                    "regime": str(regime_key),
                    "T_C": float(T_C),
                    "yO2": float(yO2),
                    "o2_label": str(o2_lab),
                    "src": src,
                    "time_window": tw,
                    "status": "SKIP",
                    "err": msg,
                })
                if skip_on_error:
                    continue
                raise

            # OK
            segments.append(seg)
            o2_values.append(float(yO2))
            rows.append({
                "char": char_name,
                "regime": str(regime_key),
                "T_C": float(T_C),
                "yO2": float(yO2),
                "o2_label": str(o2_lab),
                "src": src,
                "time_window": tw,
                "status": "OK",
                "k_iso_1_per_min": float(seg.r_abs),
                "iso_r2": float(getattr(seg, "r2_mass_vs_time", float("nan"))),
                "n_points": int(getattr(seg, "n_points", sel.shape[0])),
                "T_mean_K": float(getattr(seg, "T_mean_K", (T_C + 273.15))),
            })

    tbl = pd.DataFrame(rows)
    return segments, o2_values, tbl




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
    debug: bool = False,
    skip_on_error: bool = True,
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
        debug=debug,
        skip_on_error=skip_on_error,
    )

    # --- Filter out unusable segments (k must be finite and > 0) ---
    good_idx = []
    bad_rows = []

    for i, seg in enumerate(segments):
        k = float(getattr(seg, "r_abs", float("nan")))
        if np.isfinite(k) and k > 0.0:
            good_idx.append(i)
        else:
            bad_rows.append((i, k))

    if bad_rows and debug:
        for i, k in bad_rows:
            print(f"[ISO-GLOBAL][DROP] {char_name}: segment #{i} has invalid k={k} (must be >0)")

    segments = [segments[i] for i in good_idx]
    o2_values = [o2_values[i] for i in good_idx]

    # Also mark dropped segments in the table (if present)
    if isinstance(tbl, pd.DataFrame) and not tbl.empty and "status" in tbl.columns:
        # mark any OK rows with nonpositive k as DROP
        if "k_iso_1_per_min" in tbl.columns:
            mask_drop = (tbl["status"] == "OK") & (~np.isfinite(tbl["k_iso_1_per_min"]) | (tbl["k_iso_1_per_min"] <= 0))
            tbl.loc[mask_drop, "status"] = "DROP"
            tbl.loc[mask_drop, "err"] = "k<=0 or non-finite; excluded from global Arrhenius fit"

    if len(segments) < 2:
        msg = f"{char_name}: Not enough valid isothermal segments for global fit after filtering (n={len(segments)})."
        if debug:
            print("[ISO-GLOBAL][STOP]", msg)
        if skip_on_error:
            return None, tbl
        raise ValueError(msg)

  #  if len(segments) < 3:
  #      raise ValueError(f"{char_name}: need >=3 isothermal segments to fit global law (got {len(segments)}).")

    fit = estimate_global_arrhenius_with_o2_from_segments(
        segments=segments,
        o2_values=o2_values,
        n_o2_fixed=n_o2_fixed,
        o2_basis=o2_basis,
        label=f"{char_name} isothermal global fit",
    )
    return fit, tbl



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


def _linear_fit_with_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Fit y = slope*x + intercept and return (slope, intercept, r2).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def fit_isothermal_matrix_for_char_loaded(
    char_data: dict,
    *,
    char_label: str = "CHAR",
    temps_C: tuple[int, ...] = (225, 250),
    o2_labels: tuple[str, ...] = ("5%", "10%", "20%"),
    conversion_basis: str = "carbon",
    feedstock: str | None = None,
    ash_fraction: float | None = None,
    # conversion filtering inside estimate_segment_rate_first_order:
    alpha_range: tuple[float, float] = (0.0, 1.0),
    conversion_range: tuple[float, float] | None = None,
    # optionally force same conversion window for all 6 runs:
    enforce_common_conversion: bool = False,
    common_hi: float | None = None,
    common_hi_frac: float = 0.90,
    min_common_hi: float = 0.01,
    # time-window inference
    tol_C: float = 2.0,
    trim_start_min: float = 0.2,
    trim_end_min: float = 0.2,
) -> dict:
    """
    Fit isothermal matrix (225/250 x 5/10/20% O2) for one char, using already-loaded DataFrames:
        char_data["isothermal_225"]["5%"] -> df

    Returns dict:
      {
        "char": ...,
        "per_run": <pd.DataFrame>,
        "per_temp_o2_fit": {225: {...}, 250: {...}},
        "global_fit": <GlobalO2ArrheniusFit>,
        "conversion_range_used": (lo,hi) or None
      }
    """
    conv = str(conversion_basis).lower().strip()

    # resolve ash fraction if needed
    if conv == "carbon" and ash_fraction is None:
        key = (feedstock or char_label).upper()
        ash_fraction = ASH_FRACTION_DEFAULTS[key]

    runs = []
    # tolerate minor regime-key typos: match any key containing "isotherm"
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
            if df is None:
                continue
            if isinstance(df, str):
                if df.strip() == "":
                    continue
                raise TypeError(f"{char_label}/{regime_key}/{o2_lab}: expected DataFrame, got str: {df!r}")
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
            runs.append((str(regime_key), int(T_C), float(yO2), str(o2_lab), df, tw))

    if not runs:
        raise ValueError(f"{char_label}: no isothermal runs found in char_data for temps={temps_C} and o2={o2_labels}")

    # optionally compute an overlap conversion window (carbon-basis)
    conversion_range_used = conversion_range
    if enforce_common_conversion and conv == "carbon" and conversion_range_used is None:
        xmaxs = []
        for _, _, _, _, df, tw in runs:
            xmaxs.append(_carbon_Xmax_in_window(df, tw, ash_fraction=float(ash_fraction)))
        xmaxs = np.asarray(xmaxs, float)
        xmaxs = xmaxs[np.isfinite(xmaxs)]
        if xmaxs.size == 0:
            raise ValueError(f"{char_label}: could not compute Xmax for common conversion window")
        auto_hi = float(np.min(xmaxs) * common_hi_frac)
        hi = float(common_hi) if common_hi is not None else max(auto_hi, min_common_hi)
        conversion_range_used = (0.0, hi)

    segments = []
    o2_values = []
    rows = []

    for regime_key, T_C, yO2, o2_lab, df, tw in runs:
        seg = estimate_segment_rate_first_order(
            df,
            time_window=tw,
            label=f"{char_label}_iso_{T_C}C_{o2_lab}",
            conversion_basis=conversion_basis,
            feedstock=feedstock,
            ash_fraction=ash_fraction,
            alpha_range=alpha_range,
            conversion_range=conversion_range_used,
            normalize_within_window=False,
        )

        segments.append(seg)
        o2_values.append(yO2)

        k_iso = float(seg.r_abs)
        T_K = float(seg.T_mean_K)

        rows.append({
            "char": char_label,
            "regime": regime_key,
            "T_C": float(T_C),
            "yO2": float(yO2),
            "T_mean_C": float(T_K - 273.15),
            "T_mean_K": float(T_K),
            "k_iso_1_per_min": float(k_iso),
            "iso_r2": float(seg.r2_mass_vs_time),
            "iso_n_points": int(seg.n_points),
            "time_window": tw,
        })

    per_run = pd.DataFrame(rows).sort_values(["T_C", "yO2"]).reset_index(drop=True)

    # per-temperature oxygen-order fits: ln(k)=b_T + n_T ln(yO2)
    per_temp_o2_fit = {}
    for T_C in sorted(set(per_run["T_C"].astype(int).tolist())):
        sub = per_run[per_run["T_C"].astype(int) == int(T_C)]
        if sub.shape[0] < 2:
            per_temp_o2_fit[int(T_C)] = {"n_o2": None, "lnk_at_yO2_1": None, "r2": None, "N": int(sub.shape[0])}
            continue
        x = np.log(sub["yO2"].to_numpy(float))
        y = np.log(sub["k_iso_1_per_min"].to_numpy(float))
        n_T, b_T, r2_T = _linear_fit_with_r2(x, y)
        per_temp_o2_fit[int(T_C)] = {"n_o2": n_T, "lnk_at_yO2_1": b_T, "r2": r2_T, "N": int(sub.shape[0])}

    # global Arrhenius+O2 fit
    global_fit = estimate_global_arrhenius_with_o2_from_segments(
        segments=segments,
        o2_values=o2_values,
        label=f"{char_label} isothermal global fit",
    )

    return {
        "char": char_label,
        "per_run": per_run,
        "per_temp_o2_fit": per_temp_o2_fit,
        "global_fit": global_fit,
        "conversion_range_used": conversion_range_used,
    }


def fit_isothermal_matrix_for_char(*args, **kwargs) -> dict:
    """
    Backward-compatible alias (your main.py calls this for BRF).
    """
    return fit_isothermal_matrix_for_char_loaded(*args, **kwargs)

def refine_window_to_mass_peak(
    df: pd.DataFrame,
    time_window: tuple[float, float],
    *,
    time_col: str = "time_min",
    mass_col: str = "mass_pct",
    extra_start_min: float = 0.0,
) -> tuple[float, float]:
    t0, t1 = time_window
    sel = df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()
    if sel.empty:
        return time_window

    t = pd.to_numeric(sel[time_col], errors="coerce").to_numpy(float)
    m = pd.to_numeric(sel[mass_col], errors="coerce").to_numpy(float)
    mask = np.isfinite(t) & np.isfinite(m)
    t = t[mask]
    m = m[mask]
    if t.size < 5:
        return time_window

    t_peak = float(t[int(np.nanargmax(m))])
    t0_new = max(float(t0), t_peak + float(extra_start_min))
    if t1 <= t0_new:
        return time_window
    return (t0_new, t1)


def simulate_isothermal_holds_from_cr(
    cr: GlobalCR_O2_Result,
    char_data: dict,
    *,
    char_name: str,
    out_dir: str | Path,
    temps_C: tuple[int, ...] = (225, 250),
    o2_labels: tuple[str, ...] = ("5%", "10%", "20%"),
    conversion_basis: str = "carbon",   # "carbon" -> X_C, "alpha" -> alpha
    ash_fraction: float | None = None,
    # window inference + selection
    tol_C: float = 2.0,
    trim_start_min: float = 0.2,
    trim_end_min: float = 0.2,
    start_at_mass_peak: bool = True,
    peak_extra_start_min: float = 0.0,
    min_points: int = 20,
    # common conversion window (carbon only)
    enforce_common_conversion: bool = True,
    common_hi: float | dict[int, float] | None = None,
    common_hi_frac: float = 0.90,
    min_common_hi: float = 0.01,
    common_per_temperature: bool = True,
    use_common_window_for_curves: bool = False,  # keep False for full-hold overlays
    # outputs
    make_plots: bool = True,
    export_csv: bool = True,
    debug: bool = False,
):
    """
    Simulate and overlay isothermal holds: measured TG vs CR-predicted (mass + conversion).
    Adds R^2_mass and R^2_conv as extra lines in the TOP legend (same location/box).

    Writes:
      - sim_hold_<...>_<basis>.csv
      - sim_hold_<...>_<basis>.png
    Returns:
      - summary DataFrame with per-run R^2 fields
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    conv = str(conversion_basis).lower().strip()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- local helpers ----------
    def r2_score_safe(y_true, y_pred, *, min_n: int = 5, eps: float = 1e-16) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < min_n:
            return float("nan")
        yt = y_true[mask]
        yp = y_pred[mask]
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
        if ss_tot < eps:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    def fmt_r2(x: float) -> str:
        return "n/a" if (x is None) or (not np.isfinite(x)) else f"{x:.3f}"

    # Resolve ash fraction default if needed
    if conv == "carbon" and ash_fraction is None:
        ash_fraction = ASH_FRACTION_DEFAULTS[str(char_name).upper()]
    if conv == "carbon":
        ash_fraction = float(_resolve_ash_fraction(str(char_name), float(ash_fraction)))

    # ---- collect isothermal candidates (for common_hi computation) ----
    candidates: list[dict] = []
    for regime_key, o2_map in char_data.items():
        if "isotherm" not in str(regime_key).lower():
            continue

        T_C = _parse_temp_from_regime_key(regime_key)
        if T_C is None or int(T_C) not in temps_C:
            continue
        if not isinstance(o2_map, dict):
            continue

        for o2_lab in o2_labels:
            df = o2_map.get(o2_lab, None)
            if df is None or not _valid_df(df):
                continue

            src = SPEC.get(str(char_name).upper(), {}).get(str(regime_key), {}).get(str(o2_lab), None)

            try:
                tw = infer_isothermal_time_window(
                    df,
                    target_temp_C=float(T_C),
                    tol_C=tol_C,
                    trim_start_min=trim_start_min,
                    trim_end_min=trim_end_min,
                )
                if start_at_mass_peak:
                    tw = refine_window_to_mass_peak(df, tw, extra_start_min=float(peak_extra_start_min))
            except Exception as e:
                if debug:
                    print(f"[SIM-HOLD][SKIP] window inference failed | {char_name} | {regime_key} | {o2_lab} | src={src} | err={e}")
                continue

            candidates.append(dict(
                char=str(char_name),
                regime=str(regime_key),
                T_C=int(T_C),
                o2_label=str(o2_lab),
                yO2=float(_parse_o2_fraction(o2_lab)),
                src=src,
                df=df,
                time_window=tw,
            ))

    if not candidates:
        if debug:
            print(f"[SIM-HOLD] No candidates found for {char_name}.")
        return pd.DataFrame([])

    # ---- compute common_hi_by_T if requested (carbon only) ----
    common_hi_by_T: dict[int, float] = {}
    if (conv == "carbon") and enforce_common_conversion:
        if isinstance(common_hi, dict):
            common_hi_by_T = {int(k): float(v) for k, v in common_hi.items()}
        elif isinstance(common_hi, (int, float)):
            for T in temps_C:
                common_hi_by_T[int(T)] = float(common_hi)
        else:
            if common_per_temperature:
                for T in temps_C:
                    xmaxs = []
                    for c in candidates:
                        if int(c["T_C"]) != int(T):
                            continue
                        xmaxs.append(_carbon_Xmax_in_window(c["df"], c["time_window"], ash_fraction=float(ash_fraction)))
                    xmaxs = np.asarray(xmaxs, float)
                    xmaxs = xmaxs[np.isfinite(xmaxs)]
                    if xmaxs.size == 0:
                        continue
                    min_x = float(np.min(xmaxs))
                    hi = max(float(min_x * common_hi_frac), float(min_common_hi))
                    hi = min(hi, min_x)
                    common_hi_by_T[int(T)] = float(hi)
            else:
                xmaxs = []
                for c in candidates:
                    xmaxs.append(_carbon_Xmax_in_window(c["df"], c["time_window"], ash_fraction=float(ash_fraction)))
                xmaxs = np.asarray(xmaxs, float)
                xmaxs = xmaxs[np.isfinite(xmaxs)]
                if xmaxs.size == 0:
                    raise ValueError("simulate_isothermal_holds_from_cr: could not compute Xmax for common conversion window.")
                min_x = float(np.min(xmaxs))
                hi = max(float(min_x * common_hi_frac), float(min_common_hi))
                hi = min(hi, min_x)
                for T in temps_C:
                    common_hi_by_T[int(T)] = float(hi)

        if debug:
            print(f"[SIM-HOLD] common_hi_by_T({char_name}) = {common_hi_by_T}")

    rows = []
    for c in candidates:
        df = c["df"]
        t0, t1 = map(float, c["time_window"])

        sel = df[(df["time_min"] >= t0) & (df["time_min"] <= t1)].copy()
        sel["time_min"] = pd.to_numeric(sel["time_min"], errors="coerce")
        sel["temp_C"] = pd.to_numeric(sel.get("temp_C", np.nan), errors="coerce")
        sel["mass_pct"] = pd.to_numeric(sel.get("mass_pct", np.nan), errors="coerce")
        sel = sel.dropna(subset=["time_min", "temp_C", "mass_pct"]).sort_values("time_min")

        if sel.shape[0] < int(min_points):
            if debug:
                print(f"[SIM-HOLD][SKIP] too few points | {c['char']} | {c['regime']} | {c['o2_label']} | n={sel.shape[0]} | tw={c['time_window']}")
            continue

        t_abs = sel["time_min"].to_numpy(float)
        t_rel = t_abs - float(t_abs[0])
        T_C_trace = sel["temp_C"].to_numpy(float)
        m_meas = sel["mass_pct"].to_numpy(float)
        yO2 = float(c["yO2"])

        common_hi_used = float("nan")
        if (conv == "carbon") and enforce_common_conversion and common_hi_by_T:
            common_hi_used = float(common_hi_by_T.get(int(c["T_C"]), float("nan")))

        # ---------- measured + predicted ----------
        if conv == "carbon":
            X_meas, _, m0_top = _compute_Xc_and_w_from_mass(m_meas, ash_fraction=float(ash_fraction))
            X0 = float(X_meas[0]) if np.isfinite(X_meas[0]) else 0.0

            # simulate (treat alpha as X_C)
            X_pred = simulate_alpha_ramp(
                time_min=t_rel,
                temp_C=T_C_trace,
                yO2=yO2,
                E_A_J_per_mol=float(cr.E_A_J_per_mol),
                A=float(cr.A),
                m_o2=float(cr.m_o2),
                solid_order=int(round(getattr(cr, "n_solid", 1))),
                alpha0=X0,
            )

            m_inf = float(m0_top) * float(ash_fraction)
            m_pred = alpha_to_mass_pct(X_pred, m0=float(m0_top), m_inf=float(m_inf), loss=True)

            if use_common_window_for_curves and np.isfinite(common_hi_used):
                mask = np.isfinite(X_meas) & (X_meas >= 0.0) & (X_meas <= common_hi_used)
                if np.sum(mask) >= int(min_points):
                    t_abs, t_rel = t_abs[mask], t_rel[mask]
                    T_C_trace = T_C_trace[mask]
                    m_meas = m_meas[mask]
                    X_meas = X_meas[mask]
                    X_pred = X_pred[mask]
                    m_pred = m_pred[mask]

            conv_ylabel = r"Carbon conversion, $X_C$ (-)"
            key_meas, key_pred = "Xc_meas", "Xc_pred"

        elif conv == "alpha":
            alpha_meas, _, m0_used, m_inf_used = _compute_alpha_w(m_meas)
            X0 = float(alpha_meas[0]) if np.isfinite(alpha_meas[0]) else 0.0

            X_pred = simulate_alpha_ramp(
                time_min=t_rel,
                temp_C=T_C_trace,
                yO2=yO2,
                E_A_J_per_mol=float(cr.E_A_J_per_mol),
                A=float(cr.A),
                m_o2=float(cr.m_o2),
                solid_order=int(round(getattr(cr, "n_solid", 1))),
                alpha0=X0,
            )
            m_pred = alpha_to_mass_pct(X_pred, m0=float(m0_used), m_inf=float(m_inf_used), loss=True)
            X_meas = alpha_meas

            conv_ylabel = r"Conversion, $\alpha$ (-)"
            key_meas, key_pred = "alpha_meas", "alpha_pred"

        else:
            raise ValueError("conversion_basis must be 'carbon' or 'alpha'")

        # ---------- R^2 computed on what we actually plot ----------
        r2_mass = r2_score_safe(m_meas, m_pred)
        r2_conv = r2_score_safe(X_meas, X_pred)

        # ---------- export ----------
        stem = f"{c['char']}_{c['T_C']}C_{c['o2_label'].replace('%','pct')}"
        csv_path = out_dir / f"sim_hold_{stem}_{conv}.csv"
        fig_path = out_dir / f"sim_hold_{stem}_{conv}.png"

        out = pd.DataFrame({
            "time_min": t_abs,
            "time_rel_min": t_rel,
            "temp_C": T_C_trace,
            "mass_pct": m_meas,
            "mass_pred_pct": m_pred,
            key_meas: X_meas,
            key_pred: X_pred,
            "k_CR_pred_at_Tmean_1_per_min": predict_k_from_cr(
                cr, T_K=float(np.nanmean(T_C_trace) + 273.15), yO2=float(yO2)
            ),
            "common_hi_used": float(common_hi_used),
            "r2_mass": float(r2_mass) if np.isfinite(r2_mass) else np.nan,
            "r2_conv": float(r2_conv) if np.isfinite(r2_conv) else np.nan,
            "src": c["src"],
        })
        if export_csv:
            out.to_csv(csv_path, index=False)

        if make_plots:
            from matplotlib.lines import Line2D

            fig, (axm, axx) = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)

            axm.plot(t_rel, m_meas, "-", label="measured")
            axm.plot(t_rel, m_pred, "--", label="CR-predicted")
            axm.set_ylabel("Mass (%)")
            axm.set_title(
                f"{c['char']} {c['T_C']}°C {c['o2_label']} | CR='{getattr(cr, 'label', '')}'"
            )

            # Legend + R2 in same box (upper right)
            handles, labels = axm.get_legend_handles_labels()
            handles += [
                Line2D([], [], linestyle="none", label=rf"$R^2_{{mass}}$ = {fmt_r2(r2_mass)}"),
                Line2D([], [], linestyle="none", label=rf"$R^2_{{conv}}$ = {fmt_r2(r2_conv)}"),
            ]
            axm.legend(
                handles=handles,
                loc="upper right",
                frameon=True,
                handlelength=2.0,
                handletextpad=0.6
,
            )

            axx.plot(t_rel, X_meas, "-", label="measured")
            axx.plot(t_rel, X_pred, "--", label="CR-predicted")
            axx.set_xlabel("Time (min)")
            axx.set_ylabel(conv_ylabel)
            axx.legend(loc="best")

            fig.tight_layout()
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        rows.append({
            "char": c["char"],
            "regime": c["regime"],
            "T_C": float(c["T_C"]),
            "o2_label": c["o2_label"],
            "yO2": float(yO2),
            "time_window": c["time_window"],
            "common_hi_used": float(common_hi_used),
            "r2_mass": float(r2_mass) if np.isfinite(r2_mass) else np.nan,
            "r2_conv": float(r2_conv) if np.isfinite(r2_conv) else np.nan,
            "csv_path": str(csv_path),
            "fig_path": str(fig_path) if make_plots else "",
        })

    return pd.DataFrame(rows).sort_values(["T_C", "yO2"]).reset_index(drop=True)


def plot_linear_ramp_overlays_from_cr(
    cr: GlobalCR_O2_Result,
    char_data: dict,
    *,
    char_name: str,
    out_dir: str | Path,
    template_o2_label: str = "10%",
    yO2_targets: tuple[float, ...] = (0.05, 0.10, 0.20),
    conversion_basis: str = "carbon",  # "carbon" recommended for X_C
    ash_fraction: float | None = None,
    make_plots: bool = True,
    export_csv: bool = True,
    debug: bool = False,
):
    """
    Simulate full linear ramps using CR parameters and overlay measured vs predicted
    (mass + conversion vs temperature). Adds R^2 lines inside the TOP legend.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    conv = str(conversion_basis).lower().strip()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- local helpers ----------
    def r2_score_safe(y_true, y_pred, *, min_n: int = 5, eps: float = 1e-16) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < min_n:
            return float("nan")
        yt = y_true[mask]
        yp = y_pred[mask]
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
        if ss_tot < eps:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    def fmt_r2(x: float) -> str:
        return "n/a" if (x is None) or (not np.isfinite(x)) else f"{x:.3f}"

    if conv not in ("carbon", "alpha"):
        raise ValueError("conversion_basis must be 'carbon' or 'alpha'")

    # locate linear ramp map
    linear_map = None
    if "linear" in char_data and isinstance(char_data["linear"], dict):
        linear_map = char_data["linear"]
    else:
        for k, v in char_data.items():
            if "linear" in str(k).lower() and isinstance(v, dict):
                linear_map = v
                break
    if linear_map is None:
        raise KeyError(f"No linear ramp datasets found for {char_name} (expected key 'linear').")

    # template ramp
    df_template = linear_map.get(template_o2_label, None)
    if df_template is None:
        for _, df_any in linear_map.items():
            if _valid_df(df_any):
                df_template = df_any
                template_o2_label = "available"
                break
    if df_template is None or not _valid_df(df_template):
        raise ValueError(f"No valid ramp dataframe found for {char_name}.")

    # resolve ash
    if conv == "carbon" and ash_fraction is None:
        ash_fraction = ASH_FRACTION_DEFAULTS[str(char_name).upper()]
    if conv == "carbon":
        ash_fraction = float(_resolve_ash_fraction(str(char_name), float(ash_fraction)))

    rows = []
    for yO2 in yO2_targets:
        o2_label = f"{int(round(100*yO2))}%"
        df_meas = linear_map.get(o2_label, None)

        # base for simulation grid (use measured if available, else template)
        df_base = df_meas if _valid_df(df_meas) else df_template

        base = df_base.copy()
        base["time_min"] = pd.to_numeric(base["time_min"], errors="coerce")
        base["temp_C"] = pd.to_numeric(base["temp_C"], errors="coerce")
        base["mass_pct"] = pd.to_numeric(base["mass_pct"], errors="coerce")
        base = base.dropna(subset=["time_min", "temp_C", "mass_pct"]).sort_values("time_min")
        if base.shape[0] < 10:
            if debug:
                print(f"[SIM-RAMP][SKIP] too few base points | {char_name} | yO2={yO2:.3f}")
            continue

        t_abs = base["time_min"].to_numpy(float)
        t_rel = t_abs - float(t_abs[0])
        T_pred = base["temp_C"].to_numpy(float)

        # measured series if available
        has_meas = _valid_df(df_meas)
        if has_meas:
            meas = df_meas.copy()
            meas["temp_C"] = pd.to_numeric(meas["temp_C"], errors="coerce")
            meas["mass_pct"] = pd.to_numeric(meas["mass_pct"], errors="coerce")
            meas = meas.dropna(subset=["temp_C", "mass_pct"]).sort_values("temp_C")
            T_meas = meas["temp_C"].to_numpy(float)
            m_meas = meas["mass_pct"].to_numpy(float)
        else:
            T_meas = None
            m_meas = None

        # ---------- simulate + map ----------
        if conv == "carbon":
            # choose m0 from measured if available else from base
            if has_meas and m_meas is not None and m_meas.size:
                _, _, m0_top = _compute_Xc_and_w_from_mass(m_meas, ash_fraction=float(ash_fraction))
            else:
                _, _, m0_top = _compute_Xc_and_w_from_mass(base["mass_pct"].to_numpy(float), ash_fraction=float(ash_fraction))

            X_pred = simulate_alpha_ramp(
                time_min=t_rel,
                temp_C=T_pred,
                yO2=float(yO2),
                E_A_J_per_mol=float(cr.E_A_J_per_mol),
                A=float(cr.A),
                m_o2=float(cr.m_o2),
                solid_order=int(round(getattr(cr, "n_solid", 1))),
                alpha0=0.0,
            )

            m_inf = float(m0_top) * float(ash_fraction)
            m_pred = alpha_to_mass_pct(X_pred, m0=float(m0_top), m_inf=float(m_inf), loss=True)

            if has_meas and m_meas is not None and m_meas.size:
                X_meas, _, _ = _compute_Xc_and_w_from_mass(m_meas, ash_fraction=float(ash_fraction), m0_pct=float(m0_top))
            else:
                X_meas = None

            conv_ylabel = r"Carbon conversion, $X_C$ (-)"
            conv_pred_arr = X_pred
            conv_meas_arr = X_meas

        else:  # alpha
            if has_meas and m_meas is not None and m_meas.size:
                alpha_meas, _, m0_used, m_inf_used = _compute_alpha_w(m_meas)
            else:
                alpha_meas, _, m0_used, m_inf_used = _compute_alpha_w(base["mass_pct"].to_numpy(float))

            X_pred = simulate_alpha_ramp(
                time_min=t_rel,
                temp_C=T_pred,
                yO2=float(yO2),
                E_A_J_per_mol=float(cr.E_A_J_per_mol),
                A=float(cr.A),
                m_o2=float(cr.m_o2),
                solid_order=int(round(getattr(cr, "n_solid", 1))),
                alpha0=0.0,
            )
            m_pred = alpha_to_mass_pct(X_pred, m0=float(m0_used), m_inf=float(m_inf_used), loss=True)

            conv_pred_arr = X_pred
            conv_meas_arr = alpha_meas if (has_meas and m_meas is not None and m_meas.size) else None
            conv_ylabel = r"Conversion, $\alpha$ (-)"

        # ---------- R^2 (interpolate pred onto meas grid if available) ----------
        r2_mass = float("nan")
        r2_conv = float("nan")

        if has_meas and T_meas is not None and m_meas is not None and T_meas.size >= 5:
            # sort pred for interpolation
            op = np.argsort(T_pred)
            Tp = T_pred[op]
            mp = m_pred[op]
            xp = conv_pred_arr[op]

            m_pred_on_meas = np.interp(T_meas, Tp, mp)
            x_pred_on_meas = np.interp(T_meas, Tp, xp)

            r2_mass = r2_score_safe(m_meas, m_pred_on_meas)
            if conv_meas_arr is not None:
                r2_conv = r2_score_safe(conv_meas_arr, x_pred_on_meas)

        # ---------- export ----------
        stem = f"{str(char_name)}_linear_sim_yO2_{int(round(100*yO2))}pct_{conv}"
        csv_path = out_dir / f"{stem}.csv"
        fig_path = out_dir / f"{stem}.png"

        out = pd.DataFrame({
            "time_min": t_abs,
            "time_rel_min": t_rel,
            "temp_C": T_pred,
            "mass_pred_pct": m_pred,
            "conv_pred": conv_pred_arr,
            "yO2_sim": float(yO2),
            "r2_mass": float(r2_mass) if np.isfinite(r2_mass) else np.nan,
            "r2_conv": float(r2_conv) if np.isfinite(r2_conv) else np.nan,
        })
        if export_csv:
            out.to_csv(csv_path, index=False)

        if make_plots:
            from matplotlib.lines import Line2D

            fig, (axm, axx) = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)

            if has_meas and T_meas is not None:
                axm.plot(T_meas, m_meas, "-", label=f"measured ({o2_label})")
            axm.plot(T_pred, m_pred, "--", label=f"CR-predicted (yO2={yO2:.2f})")
            axm.set_ylabel("Mass (%)")
            axm.set_title(f"{char_name} linear ramp | CR='{getattr(cr, 'label', '')}'")

            # Legend + R2 in same box (upper right)
            handles, labels = axm.get_legend_handles_labels()
            handles += [
                Line2D([], [], linestyle="none", label=rf"$R^2_{{mass}}$ = {fmt_r2(r2_mass)}"),
                Line2D([], [], linestyle="none", label=rf"$R^2_{{conv}}$ = {fmt_r2(r2_conv)}"),
            ]
            axm.legend(
                handles=handles,
                loc="upper right",
                frameon=True,
                handlelength=2.0,
                handletextpad=0.6,
            )

            if has_meas and T_meas is not None and conv_meas_arr is not None:
                axx.plot(T_meas, conv_meas_arr, "-", label=f"measured ({o2_label})")
            axx.plot(T_pred, conv_pred_arr, "--", label=f"CR-predicted (yO2={yO2:.2f})")
            axx.set_xlabel(r"Temperature ($^\circ$C)")
            axx.set_ylabel(conv_ylabel)
            axx.legend(loc="best")

            fig.tight_layout()
            fig.savefig(fig_path, dpi=160, bbox_inches="tight")
            plt.close(fig)

        rows.append({
            "char": str(char_name),
            "yO2_sim": float(yO2),
            "o2_label_meas_used": o2_label if has_meas else "",
            "template_o2_label": str(template_o2_label),
            "r2_mass": float(r2_mass) if np.isfinite(r2_mass) else np.nan,
            "r2_conv": float(r2_conv) if np.isfinite(r2_conv) else np.nan,
            "csv_path": str(csv_path),
            "fig_path": str(fig_path) if make_plots else "",
        })

    return pd.DataFrame(rows)

