# main.py
from __future__ import annotations

from pathlib import Path
import json
import math
import inspect

import numpy as np
import pandas as pd

# Make matplotlib non-interactive (no pop-up windows)
import matplotlib

from main import OUT_ROOT, CR_WINDOWS, RAMP_TIME_WINDOW, N_SOLID, BETA_K_PER_MIN, DO_CR_TO_ISOTHERMAL_TABLES_AND_PLOTS, \
    COMPARE_CFG, DO_ISOTHERMAL_GLOBAL_BENCHMARK, DO_CR_WINDOW_SENSITIVITY, DO_CR_FITS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tg_loader import load_all_thermogravimetric_data, SPEC
from tg_math import estimate_global_coats_redfern_with_o2
from tg_helpers import (
    print_global_cr_o2_result,
    compare_cr_to_char_isothermals,
    fit_isothermal_global_from_char_data,
    compare_cr_vs_isothermal_global_on_isothermals,
)

from tg_plotting import plot_global_coats_redfern_o2_fit


def _ensure_dirs(char: str) -> dict[str, Path]:
    base = OUT_ROOT / char
    d = {
        "base": base,
        "tables": base / "tables",
        "figures": base / "figures",
        "fits": base / "fits",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _is_df(x) -> bool:
    return hasattr(x, "columns") and getattr(x, "empty", False) is False


def _filter_kwargs(func, kwargs: dict) -> dict:
    """Pass only kwargs that the function actually accepts (keeps main.py compatible during refactors)."""
    try:
        sig = inspect.signature(func)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _export_table(df: pd.DataFrame, csv_path: Path, tex_path: Path | None = None) -> None:
    df.to_csv(csv_path, index=False)
    if tex_path is not None:
        df.to_latex(tex_path, index=False, float_format=lambda x: f"{x:.6g}")


def _dump_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


# -------------------------
# Local plotting helper (so main.py works even if tg_plotting isn't updated)
# -------------------------
def plot_cr_vs_isothermal_k_table_local(
    tbl: pd.DataFrame,
    *,
    title: str,
    save_prefix: Path,
    log_scale: bool = True,
    annotate_points: bool = True,
) -> None:
    """
    Creates:
      - scatter: k_CR_pred vs k_iso with 1:1 line
      - per-temperature plots: k vs O2 (iso extracted + CR predicted)
    """
    if tbl is None or tbl.empty:
        return

    required = {"T_C", "yO2", "k_iso_1_per_min", "k_CR_pred_1_per_min"}
    if not required.issubset(set(tbl.columns)):
        return

    df = tbl.copy()

    # --- Scatter ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    x = df["k_iso_1_per_min"].to_numpy(float)
    y = df["k_CR_pred_1_per_min"].to_numpy(float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    df_sc = df.loc[df.index[mask]]

    if log_scale:
        mask2 = (x > 0) & (y > 0)
        x = x[mask2]
        y = y[mask2]
        df_sc = df_sc.loc[df_sc.index[mask2]]
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.plot(x, y, "o")

    if x.size and y.size:
        lo = float(np.min([np.min(x), np.min(y)]))
        hi = float(np.max([np.max(x), np.max(y)]))
        if hi > lo > 0:
            ax.plot([lo, hi], [lo, hi], "-")

    if annotate_points:
        for _, r in df_sc.iterrows():
            lab = f"{int(round(float(r['T_C'])))}C, {100*float(r['yO2']):g}%"
            ax.annotate(lab, (float(r["k_iso_1_per_min"]), float(r["k_CR_pred_1_per_min"])),
                        textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("k_iso extracted [1/min]")
    ax.set_ylabel("k_CR predicted [1/min]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(save_prefix) + "_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- k vs O2 per temperature ---
    for Tval in sorted(df["T_C"].unique()):
        sub = df[df["T_C"] == Tval].sort_values("yO2")
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        o2 = sub["yO2"].to_numpy(float) * 100.0
        k_iso = sub["k_iso_1_per_min"].to_numpy(float)
        k_cr = sub["k_CR_pred_1_per_min"].to_numpy(float)

        ax.plot(o2, k_iso, "o-", label="isothermal extracted")
        ax.plot(o2, k_cr, "o-", label="CR predicted")
        if log_scale and np.all((k_iso > 0) & (k_cr > 0)):
            ax.set_yscale("log")

        ax.set_xlabel("O$_2$ [%]")
        ax.set_ylabel("k [1/min]")
        ax.set_title(f"{title} — {int(round(float(Tval)))}°C")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(save_prefix) + f"_vsO2_{int(round(float(Tval)))}C.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# -------------------------
# Pipeline for one char
# -------------------------
def run_char(char: str, char_data: dict) -> dict:
    dirs = _ensure_dirs(char)

    # ---- collect linear (ramp) datasets available ----
    linear = char_data.get("linear", {})
    o2_order = ["5%", "10%", "20%"]
    ramp_dfs = []
    ramp_o2 = []
    ramp_labels = []

    for lab in o2_order:
        df = linear.get(lab, None)
        if _is_df(df):
            ramp_dfs.append(df)
            ramp_o2.append(float(lab.strip("%")) / 100.0)
            ramp_labels.append(lab)

    if len(ramp_dfs) < 2:
        raise RuntimeError(f"{char}: need >=2 ramp datasets to fit global CR (found {len(ramp_dfs)}).")

    # ---- run CR fits for multiple conversion windows ----
    cr_fits = {}
    cr_rows = []

    if DO_CR_FITS:
        for win_name, conv_rng in CR_WINDOWS:
            res = estimate_global_coats_redfern_with_o2(
                ramp_dfs,
                o2_fractions=ramp_o2,
                time_window=RAMP_TIME_WINDOW,
                n_solid=N_SOLID,
                beta_fixed_K_per_time=BETA_K_PER_MIN,
                label=f"{char} ramps global O2 fit ({win_name})",
                conversion_basis="carbon",
                conversion_range=conv_rng,
                feedstock=char,
            )
            cr_fits[win_name] = res

            # params row
            cr_rows.append({
                "char": char,
                "window": win_name,
                "conv_lo": conv_rng[0],
                "conv_hi": conv_rng[1],
                "Ea_kJ_per_mol": res.E_A_J_per_mol / 1000.0,
                "A_1_per_min": res.A,
                "m_o2": res.m_o2,
                "r2": res.r2,
                "n_runs": len(ramp_dfs),
                "runs": ",".join(ramp_labels),
            })

            # save fit details
            (dirs["fits"] / f"{win_name}.txt").write_text(print_global_cr_o2_result(res) + "\n")
            _dump_json(
                {
                    "char": char,
                    "window": win_name,
                    "conversion_range": conv_rng,
                    "Ea_kJ_per_mol": res.E_A_J_per_mol / 1000.0,
                    "A_1_per_min": res.A,
                    "m_o2": res.m_o2,
                    "r2": res.r2,
                    "time_window": list(RAMP_TIME_WINDOW),
                    "beta_K_per_min": BETA_K_PER_MIN,
                    "ramp_runs": ramp_labels,
                },
                dirs["fits"] / f"{win_name}.json",
            )

            # plot CR fit
            plot_global_coats_redfern_o2_fit(
                res,
                save_path=str(dirs["figures"] / f"{win_name}_global_cr"),
                title=f"{char} — {win_name}",
            )

        df_cr_params = pd.DataFrame(cr_rows).sort_values(["char", "window"]).reset_index(drop=True)
        _export_table(df_cr_params, dirs["tables"] / "cr_fit_params.csv", dirs["tables"] / "cr_fit_params.tex")
    else:
        df_cr_params = pd.DataFrame()

    # choose "standard" CR fit as the main one for prediction/benchmark
    std_name = "CR_std_0p10_0p80"
    if std_name not in cr_fits:
        # fallback to first available
        std_name = next(iter(cr_fits.keys()))
    cr_std = cr_fits[std_name]

    # ---- compare CR predictions to isothermal extracted k ----
    tbl_cr_vs_iso = pd.DataFrame()
    if DO_CR_TO_ISOTHERMAL_TABLES_AND_PLOTS:
        kw = _filter_kwargs(compare_cr_to_char_isothermals, dict(char_name=char, **COMPARE_CFG))
        tbl_cr_vs_iso = compare_cr_to_char_isothermals(cr_std, char_data, **kw)

        _export_table(tbl_cr_vs_iso, dirs["tables"] / "cr_vs_isothermal.csv", dirs["tables"] / "cr_vs_isothermal.tex")

        # plot (local helper)
        plot_cr_vs_isothermal_k_table_local(
            tbl_cr_vs_iso,
            title=f"{char}: CR({std_name}) predicted vs isothermal extracted",
            save_prefix=dirs["figures"] / "cr_vs_isothermal",
            log_scale=True,
            annotate_points=True,
        )

    # ---- window sensitivity summary (optional) ----
    df_win_sens = pd.DataFrame()
    if DO_CR_WINDOW_SENSITIVITY and DO_CR_TO_ISOTHERMAL_TABLES_AND_PLOTS and not tbl_cr_vs_iso.empty:
        sens_rows = []
        for win_name, res in cr_fits.items():
            kw = _filter_kwargs(compare_cr_to_char_isothermals, dict(char_name=char, **COMPARE_CFG))
            tbl = compare_cr_to_char_isothermals(res, char_data, **kw)
            if tbl.empty or "percent_error_%" not in tbl.columns:
                continue

            sens_rows.append({
                "char": char,
                "window": win_name,
                "Ea_kJ_per_mol": res.E_A_J_per_mol / 1000.0,
                "m_o2": res.m_o2,
                "r2": res.r2,
                "median_abs_error_%": float(tbl["percent_error_%"].abs().median()),
                "median_abs_error_225_%": float(tbl.loc[tbl["T_C"] == 225.0, "percent_error_%"].abs().median())
                    if (tbl["T_C"] == 225.0).any() else float("nan"),
                "median_abs_error_250_%": float(tbl.loc[tbl["T_C"] == 250.0, "percent_error_%"].abs().median())
                    if (tbl["T_C"] == 250.0).any() else float("nan"),
            })

        df_win_sens = pd.DataFrame(sens_rows).sort_values("median_abs_error_%").reset_index(drop=True)
        _export_table(df_win_sens, dirs["tables"] / "cr_window_sensitivity.csv", dirs["tables"] / "cr_window_sensitivity.tex")

    # ---- isothermal-global benchmark + compare to CR ----
    iso_fit = None
    tbl_iso_extracted = pd.DataFrame()
    tbl_cr_vs_isoGlobal = pd.DataFrame()

    if DO_ISOTHERMAL_GLOBAL_BENCHMARK:
        iso_fit, tbl_iso_extracted = fit_isothermal_global_from_char_data(
            char_data,
            char_name=char,
            conversion_basis="carbon",
            alpha_range=(0.0, 1.0),
            trim_start_min=COMPARE_CFG.get("trim_start_min", 0.2),
            trim_end_min=COMPARE_CFG.get("trim_end_min", 0.2),
        )
        _export_table(tbl_iso_extracted, dirs["tables"] / "isothermal_extracted_k.csv", dirs["tables"] / "isothermal_extracted_k.tex")

        # save iso-fit params
        _dump_json(
            {
                "char": char,
                "Ea_kJ_per_mol": iso_fit.E_A_J_per_mol / 1000.0,
                "A_1_per_min": iso_fit.A,
                "n_o2": iso_fit.n_o2,
                "r2": iso_fit.r2,
            },
            dirs["fits"] / "iso_global_fit.json",
        )

        tbl_cr_vs_isoGlobal = compare_cr_vs_isothermal_global_on_isothermals(cr_std, iso_fit, tbl_iso_extracted)
        _export_table(tbl_cr_vs_isoGlobal, dirs["tables"] / "cr_vs_isothermal_global.csv", dirs["tables"] / "cr_vs_isothermal_global.tex")

    return {
        "char": char,
        "cr_params_table": df_cr_params,
        "cr_fits": cr_fits,
        "cr_std_name": std_name,
        "tbl_cr_vs_iso": tbl_cr_vs_iso,
        "win_sens": df_win_sens,
        "iso_fit": iso_fit,
        "tbl_iso_extracted": tbl_iso_extracted,
        "tbl_cr_vs_isoGlobal": tbl_cr_vs_isoGlobal,
    }