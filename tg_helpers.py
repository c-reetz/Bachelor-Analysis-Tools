import math
import numpy as np
from pprint import pprint, pformat
import re
import pandas as pd
from tg_math import estimate_segment_rate_first_order, ASH_FRACTION_DEFAULTS, GlobalO2ArrheniusFit, \
    estimate_global_arrhenius_with_o2_from_segments
from tg_math import GlobalCR_O2_Result
from tg_loader import SPEC #debug


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
            tw = refine_window_to_mass_peak(df, tw, extra_start_min=0.0)

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
