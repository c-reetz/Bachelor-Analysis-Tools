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

def plot_coats_redfern_overlays(
    results,
    *,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = False,
):
    """
    Overlay multiple Coats–Redfern datasets + fitted lines in one plot.

    `results` is a list of objects returned by
    tg_math.estimate_EA_A_nonisothermal_coats_redfern(...).
    Each result must expose:
        - x_invT
        - y_ln_g_over_T2
        - slope
        - intercept
        - r2
        - E_A_J_per_mol
        - A
        - label (optional)
        - n_solid (optional)
    """

    fig, ax = plt.subplots(figsize=(7, 5))

    for res in results:
        x = np.asarray(res.x_invT, dtype=float)
        y = np.asarray(res.y_ln_g_over_T2, dtype=float)

        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            continue

        # sort for nice lines
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        y_hat = res.intercept + res.slope * x

        lab = getattr(res, "label", None) or "segment"
        n_solid = getattr(res, "n_solid", None)
        if n_solid is not None:
            lab = f"{lab} (n={n_solid:g})"

        Ea_kJ = float(res.E_A_J_per_mol) / 1000.0 if np.isfinite(res.E_A_J_per_mol) else float("nan")
        A = float(res.A) if np.isfinite(res.A) else float("nan")
        r2 = float(res.r2) if np.isfinite(res.r2) else float("nan")

        ax.plot(x, y, "o", label=f"{lab} data")
        ax.plot(x, y_hat, "-", label=f"{lab} fit: Ea={Ea_kJ:.1f} kJ/mol, A={A:.3g}, R²={r2:.3f}")

    ax.set_xlabel("1/T [1/K]")
    ax.set_ylabel("ln(g(w)/T²)")
    ax.set_title(title or "Coats–Redfern overlays")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_coats_redfern_global(
    res_global,
    *,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = False,
):
    """
    Plot the combined Coats–Redfern points (all datasets concatenated) and the single
    global fitted line.

    Expects `res_global` returned by estimate_EA_A_nonisothermal_coats_redfern_global(...)
    and that it exposes:
      - x_invT (1/K)
      - y_ln_g_over_T2
      - slope, intercept, r2
      - E_A_J_per_mol, A
      - beta_ref_K_per_time (optional)
      - labels / dataset_point_counts (optional)

    Saves to `save_path` if provided.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.asarray(res_global.x_invT, dtype=float)
    y = np.asarray(res_global.y_ln_g_over_T2, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        raise ValueError("Not enough finite points to plot.")

    # sort by x for a clean line
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    y_hat = res_global.intercept + res_global.slope * x

    Ea_kJ = float(res_global.E_A_J_per_mol) / 1000.0
    A = float(res_global.A)
    r2 = float(res_global.r2)

    beta_txt = ""
    if hasattr(res_global, "beta_ref_K_per_time"):
        beta_txt = f", β={float(res_global.beta_ref_K_per_time):g}"

    n_txt = ""
    if hasattr(res_global, "n_solid"):
        n_txt = f" (n={float(res_global.n_solid):g})"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, "o", label="All data (combined)")
    ax.plot(x, y_hat, "-", label=f"Global fit: Ea={Ea_kJ:.2f} kJ/mol, A={A:.3g}, R²={r2:.4f}{beta_txt}")

    ax.set_xlabel("1/T [1/K]")
    ax.set_ylabel("ln(g(w)/T²)")
    ax.set_title(title or f"Coats–Redfern global fit{n_txt}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_global_coats_redfern_o2_fit(
    res,
    *,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = False,
    make_corrected_plot: bool = True,
    make_raw_plot: bool = True,
    group_by_o2: bool = True,
    o2_label_fmt: str = "{:.0f}% O2",
):
    """
    Plot global Coats–Redfern O2 fit with consistent colours between raw and corrected plots.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.asarray(res.x_invT_all, dtype=float)
    y = np.asarray(res.y_all, dtype=float)
    z = np.asarray(res.z_lnO2_all, dtype=float)

    msk = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[msk], y[msk], z[msk]
    if x.size < 3:
        raise ValueError("Not enough finite points to plot.")

    base_title = title or (getattr(res, "label", None) or "Global Coats–Redfern O2 fit")
    n_txt = f", n_solid={float(res.n_solid):g}" if hasattr(res, "n_solid") else ""

    Ea_kJ = float(res.E_A_J_per_mol) / 1000.0
    A_val = float(res.A)
    r2 = float(res.r2)

    # --- group by O2 (z = ln(yO2) constant per dataset) ---
    if group_by_o2:
        z_round = np.round(z, 12)
        z_groups = np.unique(z_round)
    else:
        z_round = np.round(z, 12)
        z_groups = np.array([float(np.nanmedian(z_round))], dtype=float)

    def _label_for_z(z_val: float) -> str:
        yO2 = float(np.exp(z_val))
        return o2_label_fmt.format(100.0 * yO2)

    # --- consistent color mapping across plots ---
    # Sort groups by actual O2 fraction (low -> high) so colours are stable & intuitive
    z_groups_sorted = sorted(z_groups, key=lambda zz: float(np.exp(zz)))
    cmap = plt.get_cmap("tab10")
    color_map = {zg: cmap(i % 10) for i, zg in enumerate(z_groups_sorted)}

    # ---------- RAW PLOT ----------
    if make_raw_plot:
        fig, ax = plt.subplots(figsize=(7, 5))

        if group_by_o2:
            for zg in z_groups_sorted:
                gmask = (z_round == zg)
                xg = x[gmask]
                yg = y[gmask]
                if xg.size < 2:
                    continue

                idx = np.argsort(xg)
                xg = xg[idx]
                yg = yg[idx]

                # model line for this O2 group
                yhat_g = res.intercept + res.coef_invT * xg + res.m_o2 * float(zg)

                col = color_map[zg]
                lab = _label_for_z(float(zg))

                ax.plot(xg, yg, "o", color=col, label=f"data ({lab})")
                ax.plot(xg, yhat_g, "-", color=col, label=f"model ({lab})")
        else:
            # no grouping: avoid connecting across mixed O2 values
            yhat = res.intercept + res.coef_invT * x + res.m_o2 * z
            ax.plot(x, y, "o", label="data (all runs)")
            ax.plot(x, yhat, ".", label="model (per-point)")

        ax.set_xlabel("1/T [1/K]")
        ax.set_ylabel("ln(g(w)/T²)")
        ax.set_title(base_title + " — raw" + n_txt)
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(f"{save_path}_raw.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ---------- O2-CORRECTED PLOT ----------
    if make_corrected_plot:
        fig, ax = plt.subplots(figsize=(7, 5))

        y_corr = y - res.m_o2 * z

        if group_by_o2:
            for zg in z_groups_sorted:
                gmask = (z_round == zg)
                xg = x[gmask]
                yc = y_corr[gmask]
                if xg.size < 2:
                    continue

                idx = np.argsort(xg)
                xg = xg[idx]
                yc = yc[idx]

                col = color_map[zg]
                lab = _label_for_z(float(zg))
                ax.plot(xg, yc, "o", color=col, label=f"data ({lab})")
        else:
            ax.plot(x, y_corr, "o", label="data (O₂-corrected)")

        # single collapsed fit line
        x_line = np.linspace(np.min(x), np.max(x), 200)
        yhat_corr = res.intercept + res.coef_invT * x_line
        ax.plot(
            x_line, yhat_corr, "-",
            label=f"fit (Ea={Ea_kJ:.1f} kJ/mol, A={A_val:.3g}, m_O2={res.m_o2:.2f}, R²={r2:.3f})"
        )

        ax.set_xlabel("1/T [1/K]")
        ax.set_ylabel("ln(g(w)/T²) − m_O2·ln(yO2)")
        ax.set_title(base_title + " — O₂-corrected" + n_txt)
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(f"{save_path}_corrected.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def plot_dtg_curve(
    df: pd.DataFrame,
    *,
    time_window: Tuple[float, float] | None = None,
    x_axis: str = "temp",  # "temp" or "time"
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    smooth_window: int = 11,
    beta_min: float = 1e-6,
    label: Optional[str] = None,
    show: bool = False,
    save_path: Optional[str] = None,
):
    """
    Plot DTG for a TG dataset (math is done in tg_math.compute_dtg_curve).

    Plots:
      y = dtg_loss = -dm/dT  [mass% / °C]  (positive peaks = mass loss)
      x = temp_C (default) or time_min

    Returns the DTG dataframe (so you can reuse it / debug).
    """
    from tg_math import compute_dtg_curve  # local import to avoid changing top-level imports

    if label is None:
        label = "DTG"

    dtg = compute_dtg_curve(
        df,
        time_window=time_window,
        time_col=time_col,
        temp_col=temp_col,
        mass_col=mass_col,
        smooth_window=smooth_window,
        beta_min=beta_min,
        drop_invalid=True,
    )
    if dtg.empty:
        raise ValueError("DTG dataframe is empty (often happens if dataset is isothermal or beta_min too high).")

    x_axis_l = str(x_axis).lower().strip()
    if x_axis_l in ("temp", "temperature", "t"):
        x = dtg[temp_col].to_numpy(dtype=float)
        xlabel = "Temperature [°C]"
    elif x_axis_l in ("time", "tmin", "minutes"):
        x = dtg[time_col].to_numpy(dtype=float)
        xlabel = "Time [min]"
    else:
        raise ValueError("x_axis must be 'temp' or 'time'.")

    y = dtg["dtg_loss"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, "-", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("DTG = -dm/dT [mass% / °C]")
    ax.set_title("DTG curve")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return dtg

def _subset_df_by_segment_or_window(
    df: pd.DataFrame,
    *,
    segment: int | str | None = None,
    time_window: tuple[float, float] | None = None,
    time_col: str = "time_min",
    seg_col: str = "segment",
) -> pd.DataFrame:
    """
    Helper: choose data either by segment (preferred) or by time_window.
    If segment is given, time_window is ignored.
    """
    d = df.copy()

    # Coerce time
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")

    if segment is not None:
        if seg_col not in d.columns:
            raise KeyError(f"Missing '{seg_col}' column; cannot subset by segment.")
        # Coerce segment column to numeric to avoid dtype mismatches
        d[seg_col] = pd.to_numeric(d[seg_col], errors="coerce")
        seg_val = float(pd.to_numeric(str(segment), errors="coerce"))
        if not np.isfinite(seg_val):
            raise ValueError(f"Could not parse segment value: {segment!r}")
        d = d[d[seg_col] == seg_val].copy()
    elif time_window is not None:
        t0, t1 = map(float, time_window)
        d = d[(d[time_col] >= t0) & (d[time_col] <= t1)].copy()

    # Drop invalid time rows
    d = d.dropna(subset=[time_col])
    return d


def plot_tg_curve_time(
    df: pd.DataFrame,
    *,
    segment: int | str | None = None,
    time_window: tuple[float, float] | None = None,
    time_col: str = "time_min",
    mass_col: str = "mass_pct",
    seg_col: str = "segment",
    # choose m0 and time-zero independently
    m0_mode: str = "start",      # "start" (shows mass gain >100%) or "max" (caps at 100%)
    t0_mode: str = "start",      # "start" or "m0"
    drop_before_t0: bool = False,
    label: str | None = None,
    title: str | None = None,
    show: bool = False,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Plot TG (mass vs time) normalized to m0, with time re-zeroing.
    Provide either segment=... or time_window=... (segment wins if both).
    """
    d = _subset_df_by_segment_or_window(
        df,
        segment=segment,
        time_window=time_window,
        time_col=time_col,
        seg_col=seg_col,
    )

    d[mass_col] = pd.to_numeric(d[mass_col], errors="coerce")
    d = d.dropna(subset=[time_col, mass_col]).sort_values(time_col)

    if d.empty:
        raise ValueError("plot_tg_curve_time: no data after segment/window selection and cleaning.")

    t = d[time_col].to_numpy(float)
    m = d[mass_col].to_numpy(float)

    i_max = int(np.nanargmax(m))

    if m0_mode not in {"start", "max"}:
        raise ValueError("plot_tg_curve_time: m0_mode must be 'start' or 'max'.")
    m0 = float(m[0]) if m0_mode == "start" else float(m[i_max])
    if not np.isfinite(m0) or m0 == 0.0:
        raise ValueError("plot_tg_curve_time: invalid m0 for normalization.")

    if t0_mode not in {"start", "m0"}:
        raise ValueError("plot_tg_curve_time: t0_mode must be 'start' or 'm0'.")
    i_t0 = 0 if t0_mode == "start" else i_max

    if drop_before_t0 and i_t0 > 0:
        t = t[i_t0:]
        m = m[i_t0:]
        t_rel = t - float(t[0])
    else:
        t_rel = t - float(t[i_t0])

    mass_norm = 100.0 * (m / m0)

    out = pd.DataFrame(
        {"time_rel_min": t_rel, "mass_norm_pct": mass_norm, "mass_raw_pct": m}
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(out["time_rel_min"], out["mass_norm_pct"], "-", label=label)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mass (% of m0)")
    ax.set_title(title or "TG curve (mass vs time)")
    if label:
        ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return out


def plot_Xc_curve_time(
    df: pd.DataFrame,
    *,
    segment: int | str | None = None,
    time_window: tuple[float, float] | None = None,
    time_col: str = "time_min",
    mass_col: str = "mass_pct",
    seg_col: str = "segment",
    feedstock: str | None = None,
    ash_fraction: float | None = None,
    as_percent: bool = True,
    # for Xc you usually want m0 at max and t0 at m0:
    m0_mode: str = "max",        # "max" or "start"
    t0_mode: str = "m0",         # "m0" or "start"
    drop_before_t0: bool = True, # if True, Xc starts at 0
    clip: bool = False,
    label: str | None = None,
    title: str | None = None,
    show: bool = False,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Plot carbon conversion X_C vs time.
    Provide either segment=... or time_window=... (segment wins if both).
    """
    from tg_math import _resolve_ash_fraction

    d = _subset_df_by_segment_or_window(
        df,
        segment=segment,
        time_window=time_window,
        time_col=time_col,
        seg_col=seg_col,
    )

    d[mass_col] = pd.to_numeric(d[mass_col], errors="coerce")
    d = d.dropna(subset=[time_col, mass_col]).sort_values(time_col)

    if d.empty:
        raise ValueError("plot_Xc_curve_time: no data after segment/window selection and cleaning.")

    t = d[time_col].to_numpy(float)
    m = d[mass_col].to_numpy(float)

    i_max = int(np.nanargmax(m))

    if m0_mode not in {"max", "start"}:
        raise ValueError("plot_Xc_curve_time: m0_mode must be 'max' or 'start'.")
    m0 = float(m[i_max]) if m0_mode == "max" else float(m[0])
    if not np.isfinite(m0) or m0 == 0.0:
        raise ValueError("plot_Xc_curve_time: invalid m0 for conversion.")

    if t0_mode not in {"start", "m0"}:
        raise ValueError("plot_Xc_curve_time: t0_mode must be 'start' or 'm0'.")
    i_t0 = 0 if t0_mode == "start" else i_max

    if drop_before_t0 and i_t0 > 0:
        t = t[i_t0:]
        m = m[i_t0:]
        t_rel = t - float(t[0])
    else:
        t_rel = t - float(t[i_t0])

    af = _resolve_ash_fraction(feedstock, ash_fraction)
    denom = m0 * (1.0 - af)
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("plot_Xc_curve_time: invalid denominator (check ash fraction/feedstock).")

    Xc = (m0 - m) / denom
    if clip:
        Xc = np.clip(Xc, 0.0, 1.0)

    out = pd.DataFrame(
        {"time_rel_min": t_rel, "Xc": Xc, "Xc_pct": 100.0 * Xc}
    )

    y = out["Xc_pct"] if as_percent else out["Xc"]
    ylabel = "Carbon conversion, $X_C$ (%)" if as_percent else "Carbon conversion, $X_C$ (-)"

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(out["time_rel_min"], y, "-", label=label)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Carbon conversion vs time")
    if label:
        ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return out

