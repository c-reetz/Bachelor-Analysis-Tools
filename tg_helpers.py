import math
import numpy as np
from pprint import pprint
from typing import Dict, Tuple, Optional, Any
import re
from tg_math import (
    estimate_segment_rate_first_order,
    estimate_global_arrhenius_with_o2_from_segments,
    _linear_fit,  # <- reuse existing tg_math OLS fitter (slope, intercept, r2)
)


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

def _parse_o2_label(o2_label: str) -> float:
    """
    Accepts "5%", "10%", "20%", "0.05", "0.1", etc.
    Returns fraction, e.g. 0.05.
    """
    s = str(o2_label).strip().lower().replace("o2", "")
    s = s.replace(",", ".")
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        raise ValueError(f"Could not parse O2 label: {o2_label!r}")
    val = float(m.group(1))
    return val / 100.0 if "%" in s or val > 1.0 else val


def infer_isothermal_time_window(
    df,
    target_temp_C: float,
    *,
    tol_C: float = 2.0,
    min_points: int = 30,
    trim_start_min: float = 0.0,
    trim_end_min: float = 0.0,
):
    import numpy as np
    import pandas as pd

    d = df.copy()

    # Make sure numeric cols are numeric
    d["time_min"] = pd.to_numeric(d["time_min"], errors="coerce")
    d["temp_C"] = pd.to_numeric(d["temp_C"], errors="coerce")

    # Convert segment to integer (nullable Int64)
    if "segment" in d.columns:
        d["segment"] = pd.to_numeric(d["segment"], errors="coerce").astype("Int64")

    best_seg = None
    best_score = None

    if "segment" in d.columns and d["segment"].notna().any():
        for seg_id in sorted(d["segment"].dropna().unique()):
            sdf = d[d["segment"] == seg_id].dropna(subset=["time_min", "temp_C"])
            if len(sdf) < min_points:
                continue
            Tmed = float(np.nanmedian(sdf["temp_C"].to_numpy(float)))
            score = abs(Tmed - target_temp_C)
            if score <= tol_C and (best_score is None or score < best_score):
                best_score = score
                best_seg = int(seg_id)

        if best_seg is not None:
            sdf = d[d["segment"] == best_seg].dropna(subset=["time_min"])
            t0 = float(np.nanmin(sdf["time_min"].to_numpy(float))) + float(trim_start_min)
            t1 = float(np.nanmax(sdf["time_min"].to_numpy(float))) - float(trim_end_min)
            if t1 <= t0:
                raise ValueError("Invalid trimmed time window.")
            return (t0, t1), best_seg

    # Fallback: temperature mask
    mask = np.abs(d["temp_C"].to_numpy(float) - float(target_temp_C)) <= float(tol_C)
    idx = np.where(mask)[0]
    if idx.size < min_points:
        raise ValueError(f"Could not infer isothermal region near {target_temp_C}°C (tol={tol_C}°C).")
    t0 = float(d["time_min"].iloc[idx[0]]) + float(trim_start_min)
    t1 = float(d["time_min"].iloc[idx[-1]]) - float(trim_end_min)
    if t1 <= t0:
        raise ValueError("Invalid trimmed time window.")
    return (t0, t1), -1



def fit_isothermal_matrix_for_char_loaded(
    sample_block: Dict[str, Dict[str, Optional[Any]]],
    *,
    char_label: str,
    temps_C: Tuple[int, ...] = (225, 250),
    o2_labels: Tuple[str, ...] = ("5%", "10%", "20%"),
    alpha_range: Tuple[float, float] = (0.0, 1.0),
    conversion_basis: str = "carbon",
    feedstock: Optional[str] = None,
    ash_fraction: Optional[float] = None,      # override if desired
    tol_C: float = 2.0,
    trim_start_min: float = 0.0,
    trim_end_min: float = 0.0,
    n_similarity_tol: float = 0.20,
) -> dict:
    """
    sample_block is: data[sample] from load_all_thermogravimetric_data
      sample_block[regime][o2_label] -> DataFrame | None
    where regime keys include "isothermal_225", "isothermal_250", etc.
    """
    per_run: Dict[Tuple[int, float], dict] = {}
    segments = []
    o2s = []

    for T_C in temps_C:
        regime = f"isothermal_{T_C}"
        if regime not in sample_block:
            raise KeyError(f"{char_label}: missing regime {regime!r} in loaded data.")
        for o2_label in o2_labels:
            df = sample_block[regime].get(o2_label, None)
            if df is None:
                raise ValueError(f"{char_label}: missing dataset for {regime} / {o2_label}.")

            yO2 = _parse_o2_label(o2_label)

            (t0, t1), seg_id = infer_isothermal_time_window(
                df,
                target_temp_C=float(T_C),
                tol_C=tol_C,
                trim_start_min=trim_start_min,
                trim_end_min=trim_end_min,
            )

            seg = estimate_segment_rate_first_order(
                df,
                time_window=(t0, t1),
                time_col="time_min",
                temp_col="temp_C",
                mass_col="mass_pct",
                label=f"{char_label}_{T_C}C_{o2_label}",
                conversion_basis=conversion_basis,
                feedstock=feedstock,
                ash_fraction=ash_fraction,
                alpha_range=alpha_range,
                normalize_within_window=False,
            )

            per_run[(int(T_C), float(yO2))] = {
                "regime": regime,
                "o2_label": o2_label,
                "segment": seg_id,
                "time_window": (t0, t1),
                "T_mean_K": seg.T_mean_K,
                "k_1_per_min": seg.r_abs,
                "r2_ln_w_vs_t": seg.r2_mass_vs_time,
                "n_points": seg.n_points,
            }

            segments.append(seg)
            o2s.append(float(yO2))

    # ---- per-temperature oxygen-order fits: ln(k) = b_T + n_T ln(yO2) ----
    per_temp_fit = {}
    for T_C in temps_C:
        ks = []
        ys = []
        for o2_label in o2_labels:
            yO2 = _parse_o2_label(o2_label)
            key = (int(T_C), float(yO2))
            ks.append(per_run[key]["k_1_per_min"])
            ys.append(yO2)

        x = np.log(np.asarray(ys, float))
        y = np.log(np.asarray(ks, float))
        n_T, b_T, r2_T = _linear_fit(x, y)
        per_temp_fit[int(T_C)] = {
            "n_o2": float(n_T),
            "lnk_at_yO2_1": float(b_T),
            "r2": float(r2_T),
            "N": int(len(ks)),
        }

    bad = []
    for (T_C, yO2), info in per_run.items():
        k = info["k_1_per_min"]
        if (k is None) or (not np.isfinite(k)) or (k <= 0.0):
            bad.append((T_C, yO2, info["regime"], info["o2_label"], k, info["time_window"], info["segment"]))

    if bad:
        msg = "\n".join(
            f"T={T_C}C, yO2={yO2:.3f} ({reg}/{o2lbl}): k={k} window={tw} seg={seg}"
            for (T_C, yO2, reg, o2lbl, k, tw, seg) in bad
        )
        raise ValueError("Non-positive/invalid k in isothermal matrix:\n" + msg)

    # ---- full global char fit across all runs: ln k = ln A + n ln(O2) - Ea/RT ----
    global_fit = estimate_global_arrhenius_with_o2_from_segments(
        segments=segments,
        o2_values=o2s,
        label=f"{char_label} isothermal global fit",
    )

    n_225 = per_temp_fit.get(225, {}).get("n_o2", None)
    n_250 = per_temp_fit.get(250, {}).get("n_o2", None)
    n_similar = None
    if (n_225 is not None) and (n_250 is not None):
        n_similar = abs(float(n_225) - float(n_250)) <= float(n_similarity_tol)

    return {
        "char": char_label,
        "per_run": per_run,
        "per_temp_o2_fit": per_temp_fit,
        "n_similar_225_250": n_similar,
        "global_fit": global_fit,
    }