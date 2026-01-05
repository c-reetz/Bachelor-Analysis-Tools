from __future__ import annotations

import re
from pathlib import Path
import json
import math
import inspect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Make matplotlib non-interactive (no pop-up windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tg_math import estimate_global_coats_redfern_with_o2
from tg_helpers import (
    print_global_cr_o2_result,
    compare_cr_to_char_isothermals,
    fit_isothermal_global_from_char_data,
    compare_cr_vs_isothermal_global_on_isothermals, format_global_cr_o2_result,
)
from tg_plotting import plot_global_coats_redfern_o2_fit, plot_tg_curve_time


@dataclass
class ReportConfig:
    out_root: Path = Path("out")

    # Coats–Redfern (ramp) conversion windows
    cr_windows: list[tuple[str, tuple[float, float]]] = field(
        default_factory=lambda: [
            ("CR_std_0p10_0p80", (0.10, 0.80)),
            ("CR_mid_0p05_0p20", (0.05, 0.20)),
            ("CR_early_0p00_0p06", (0.00, 0.06)),
        ]
    )

    # Ramp program metadata
    ramp_time_window: tuple[float, float] = (32.0, 195.0)  # min
    n_solid: float = 1.0
    beta_k_per_min: float = 3.0  # K/min

    # Isothermal extraction / comparison defaults
    compare_cfg: dict[str, Any] = field(
        default_factory=lambda: dict(
            conversion_basis="carbon",
            enforce_common_conversion=True,
            common_per_temperature=True,
            start_at_mass_peak=True,
            common_hi_frac=0.90,
            min_common_hi=0.01,
            trim_start_min=0.2,
            trim_end_min=0.2,
        )
    )

    # Section toggles
    do_cr_fits: bool = True
    do_cr_window_sensitivity: bool = True
    do_cr_to_isothermal_tables_and_plots: bool = True
    do_isothermal_global_benchmark: bool = True


def _ensure_dirs(char: str, out_root: Path) -> dict[str, Path]:
    base = out_root / char
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
            ax.annotate(
                lab,
                (float(r["k_iso_1_per_min"]), float(r["k_CR_pred_1_per_min"])),
                textcoords="offset points",
                xytext=(6, 4),
            )

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
def run_char(char: str, char_data: dict, *, cfg: ReportConfig | None = None) -> dict:
    if cfg is None:
        cfg = ReportConfig()

    dirs = _ensure_dirs(char, cfg.out_root)

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

    if cfg.do_cr_fits:
        for win_name, conv_rng in cfg.cr_windows:
            res = estimate_global_coats_redfern_with_o2(
                ramp_dfs,
                o2_fractions=ramp_o2,
                time_window=cfg.ramp_time_window,
                n_solid=cfg.n_solid,
                beta_fixed_K_per_time=cfg.beta_k_per_min,
                label=f"{char} ramps global O2 fit ({win_name})",
                conversion_basis="carbon",
                conversion_range=conv_rng,
                feedstock=char,
            )
            cr_fits[win_name] = res

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

            (dirs["fits"] / f"{win_name}.txt").write_text(format_global_cr_o2_result(res) + "\n")
            _dump_json(
                {
                    "char": char,
                    "window": win_name,
                    "conversion_range": conv_rng,
                    "Ea_kJ_per_mol": res.E_A_J_per_mol / 1000.0,
                    "A_1_per_min": res.A,
                    "m_o2": res.m_o2,
                    "r2": res.r2,
                    "time_window": list(cfg.ramp_time_window),
                    "beta_K_per_min": cfg.beta_k_per_min,
                    "ramp_runs": ramp_labels,
                },
                dirs["fits"] / f"{win_name}.json",
            )

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
    if std_name not in cr_fits and cr_fits:
        std_name = next(iter(cr_fits.keys()))
    if not cr_fits:
        raise RuntimeError(f"{char}: no CR fits available (cfg.do_cr_fits is False?).")
    cr_std = cr_fits[std_name]

    # ---- compare CR predictions to isothermal extracted k ----
    tbl_cr_vs_iso = pd.DataFrame()
    if cfg.do_cr_to_isothermal_tables_and_plots:
        kw = _filter_kwargs(compare_cr_to_char_isothermals, dict(char_name=char, **cfg.compare_cfg))
        tbl_cr_vs_iso = compare_cr_to_char_isothermals(cr_std, char_data, **kw)

        _export_table(tbl_cr_vs_iso, dirs["tables"] / "cr_vs_isothermal.csv", dirs["tables"] / "cr_vs_isothermal.tex")

        plot_cr_vs_isothermal_k_table_local(
            tbl_cr_vs_iso,
            title=f"{char}: CR({std_name}) predicted vs isothermal extracted",
            save_prefix=dirs["figures"] / "cr_vs_isothermal",
            log_scale=True,
            annotate_points=True,
        )

    # ---- window sensitivity summary (optional) ----
    df_win_sens = pd.DataFrame()
    if cfg.do_cr_window_sensitivity and cfg.do_cr_to_isothermal_tables_and_plots and not tbl_cr_vs_iso.empty:
        sens_rows = []
        for win_name, res in cr_fits.items():
            kw = _filter_kwargs(compare_cr_to_char_isothermals, dict(char_name=char, **cfg.compare_cfg))
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

    if cfg.do_isothermal_global_benchmark:
        iso_fit, tbl_iso_extracted = fit_isothermal_global_from_char_data(
            char_data,
            char_name=char,
            conversion_basis="carbon",
            alpha_range=(0.0, 1.0),
            trim_start_min=float(cfg.compare_cfg.get("trim_start_min", 0.2)),
            trim_end_min=float(cfg.compare_cfg.get("trim_end_min", 0.2)),
            debug=True,
            skip_on_error=True
        )
        _export_table(tbl_iso_extracted, dirs["tables"] / "isothermal_extracted_k.csv", dirs["tables"] / "isothermal_extracted_k.tex")

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

        if iso_fit is not None:
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
        else:
            print(f"[ISO-GLOBAL] {char}: iso_fit=None (not enough valid segments).")
            tbl_cr_vs_isoGlobal = pd.DataFrame()

    df_hold = build_hold_usability_table(
        char_name=char,
        tbl_cr_vs_iso=tbl_cr_vs_iso,
        raw_char_data=char_data,
        delta_alpha_min=0.01,  # your “nothing happened” threshold
    )
    outp = dirs["tables"] / "optionB_isothermal_summary.tex"
    print("[OptionB] writing:", outp.resolve())
    write_optionB_tex(
        df_hold=df_hold,
        out_tex_path=dirs["tables"] / "optionB_isothermal_summary.tex",
        example_figure_relpath=f"{char}/figures/<PUT_YOUR_EXAMPLE_FIGURE_NAME_HERE>.png",
    )
    print("[OptionB] exists:", outp.exists(), "size:", outp.stat().st_size if outp.exists() else None)

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


def _parse_feedstock_and_charT(char_name: str) -> tuple[str, int | None]:
    """
    Best-effort parse, e.g. 'BRF500' -> ('BRF', 500), 'PW_650' -> ('PW', 650).
    If it can't parse T, returns (CHAR_NAME, None).
    """
    s = str(char_name).strip()
    m = re.match(r"^\s*([A-Za-z]+)[\s_\-]*([0-9]{3,4})\s*$", s)
    if not m:
        return (s, None)
    return (m.group(1).upper(), int(m.group(2)))

def _parse_holdT_from_regime(regime: str) -> int | None:
    m = re.search(r"(\d{3})", str(regime))
    return int(m.group(1)) if m else None

def _safe_float(x) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")

def build_hold_usability_table(
    *,
    char_name: str,
    tbl_cr_vs_iso: pd.DataFrame,
    raw_char_data: dict,
    delta_alpha_min: float = 0.01,
) -> pd.DataFrame:
    """
    Creates the compact table you described:
    Feedstock | Char T | Hold T | O2(%) | Δα in hold | Fit window | k | R² | Note

    Uses tbl_cr_vs_iso for k, R² and time_window, and raw_char_data to compute Δα
    (from mass change in the hold window).
    """
    feedstock, charT = _parse_feedstock_and_charT(char_name)

    rows = []
    for _, r in tbl_cr_vs_iso.iterrows():
        regime = r.get("regime", None)
        o2_lab = r.get("o2_lab", None)
        T_C = r.get("T_C", None)
        yO2 = r.get("yO2", None)
        k_iso = r.get("k_iso_1_per_min", None)
        r2 = r.get("iso_r2", None)
        tw = r.get("time_window", None)  # expected (t0, t1) as tuple/list

        holdT = _parse_holdT_from_regime(regime) or (int(T_C) if pd.notna(T_C) else None)

        # --- compute Δα in hold from raw mass data (robust median-of-ends) ---
        # raw_char_data structure is assumed like raw_char_data[regime][o2_lab] -> df
        delta_alpha = float("nan")
        try:
            df = raw_char_data.get(str(regime), {}).get(str(o2_lab), None)
            if isinstance(df, pd.DataFrame) and tw is not None and len(tw) == 2:
                t0, t1 = float(tw[0]), float(tw[1])
                sel = df[(df["time_min"] >= t0) & (df["time_min"] <= t1)].copy()
                if len(sel) >= 4:
                    m0 = pd.to_numeric(sel["mass_pct"], errors="coerce").iloc[:5].median()
                    m1 = pd.to_numeric(sel["mass_pct"], errors="coerce").iloc[-5:].median()
                    if np.isfinite(m0) and np.isfinite(m1) and m0 > 0:
                        # “fractional mass loss during hold” as a practical Δα proxy
                        delta_alpha = max(0.0, (m0 - m1) / m0)
        except Exception:
            pass

        # --- note / usability logic ---
        used = np.isfinite(delta_alpha) and (delta_alpha >= float(delta_alpha_min))
        note = "Used" if used else "No measurable oxidation"

        # --- formatting fields (keep numeric in df; format later for LaTeX) ---
        o2_pct = float(yO2) * 100.0 if pd.notna(yO2) else (
            float(str(o2_lab).replace("%", "")) if o2_lab is not None and "%" in str(o2_lab) else float("nan")
        )

        rows.append({
            "Feedstock": feedstock,
            "Char T (°C)": charT if charT is not None else "",
            "Hold T (°C)": holdT if holdT is not None else "",
            "O2 (%)": o2_pct if np.isfinite(o2_pct) else "",
            "Δα in hold": delta_alpha if np.isfinite(delta_alpha) else np.nan,
            "Fit window (min)": (tw if (tw is not None and len(tw) == 2) else None),
            "k (min$^{-1}$)": _safe_float(k_iso) if used else np.nan,
            "R$^2$": _safe_float(r2) if used else np.nan,
            "Note": note,
            "regime": regime,
            "o2_lab": o2_lab,
        })

    out = pd.DataFrame(rows)

    # Convert Fit window to a nice "t0--t1" string (or em dash)
    def _tw_str(x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            return r"\textemdash"
        return f"{float(x[0]):.0f}--{float(x[1]):.0f}"

    out["Fit window (min)"] = out["Fit window (min)"].apply(_tw_str)

    return out[[
        "Feedstock", "Char T (°C)", "Hold T (°C)", "O2 (%)",
        "Δα in hold", "Fit window (min)", "k (min$^{-1}$)", "R$^2$", "Note"
    ]].sort_values(["Feedstock", "Char T (°C)", "Hold T (°C)", "O2 (%)"], kind="stable")

def write_optionB_tex(
    *,
    df_hold: pd.DataFrame,
    out_tex_path: Path,
    example_figure_relpath: str | None = None,
    caption: str = "Isothermal hold usability and extracted rate constants.",
    label: str = "tab:iso-usability",
    fig_caption: str = "Example first-order fit on an isothermal hold.",
    fig_label: str = "fig:iso-example-fit",
) -> None:
    """
    Writes ONE .tex file containing the table + (optional) one example figure.
    Requires LaTeX packages: booktabs, siunitx, float, graphicx.
    """
    df = df_hold.copy()

    # LaTeX-friendly formatting (uses siunitx \num{} for scientific notation)
    def fmt_num(x, nd=2):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return r"\textemdash"
        return f"{float(x):.{nd}f}"

    def fmt_sci(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return r"\textemdash"
        return r"\num{" + f"{float(x):.2e}" + "}"

    df["O2 (%)"] = df["O2 (%)"].apply(lambda v: fmt_num(v, nd=0) if str(v) != "" else r"\textemdash")
    df["Δα in hold"] = df["Δα in hold"].apply(lambda v: fmt_num(v, nd=2))
    df["k (min$^{-1}$)"] = df["k (min$^{-1}$)"].apply(fmt_sci)
    df["R$^2$"] = df["R$^2$"].apply(lambda v: fmt_num(v, nd=2))

    tabular = df.to_latex(
        index=False,
        escape=False,
        na_rep=r"\textemdash",
        column_format="llrrrrrll",
        longtable=False,
        caption=None,
        label=None,
    )

    parts = []
    parts.append(r"\begin{table}[H]")
    parts.append(r"\centering")
    parts.append(r"\small")
    parts.append(r"\caption{" + caption + r"}")
    parts.append(r"\label{" + label + r"}")
    parts.append(r"\sisetup{detect-all=true}")
    parts.append(tabular)
    parts.append(r"\end{table}")
    parts.append("")

    if example_figure_relpath:
        parts.append(r"\begin{figure}[H]")
        parts.append(r"\centering")
        parts.append(r"\includegraphics[width=0.90\linewidth]{" + example_figure_relpath + r"}")
        parts.append(r"\caption{" + fig_caption + r"}")
        parts.append(r"\label{" + fig_label + r"}")
        parts.append(r"\end{figure}")
        parts.append("")

    out_tex_path.write_text("\n".join(parts), encoding="utf-8")

def _find_isothermal_regime_key(char_data: dict, hold_temp_C: int) -> str | None:
    target = f"isothermal_{int(hold_temp_C)}"
    if target in char_data:
        return target
    # fallback: any key containing both 'isothermal' and the temperature number
    cands = [k for k in char_data.keys()
             if isinstance(k, str) and ("isothermal" in k.lower()) and (str(int(hold_temp_C)) in k)]
    return cands[0] if cands else None


def _norm_mass_curve(df: pd.DataFrame, *, trim_start_min: float = 0.0, trim_end_min: float = 0.0):
    """
    Returns time_rel (min) and m_norm = m/m0 for an isothermal segment DF with columns:
    'time_min' and 'mass_pct'.
    """
    if df is None or df.empty:
        return None, None

    d = df.copy()
    d["time_min"] = pd.to_numeric(d["time_min"], errors="coerce")
    d["mass_pct"] = pd.to_numeric(d["mass_pct"], errors="coerce")
    d = d.dropna(subset=["time_min", "mass_pct"])
    if d.empty:
        return None, None

    tmin = float(d["time_min"].min())
    tmax = float(d["time_min"].max())
    lo = tmin + float(trim_start_min)
    hi = tmax - float(trim_end_min)
    if hi <= lo:
        lo, hi = tmin, tmax

    d = d[(d["time_min"] >= lo) & (d["time_min"] <= hi)].copy()
    if len(d) < 3:
        return None, None

    d = d.sort_values("time_min")
    t = d["time_min"].to_numpy(float)
    m = d["mass_pct"].to_numpy(float)

    # robust m0: median of first few points
    m0 = float(np.median(m[: min(5, len(m))]))
    if not np.isfinite(m0) or m0 == 0:
        return None, None

    t_rel = t - t[0]
    m_norm = m / m0
    return t_rel, m_norm


def plot_isothermal_matrix_feedstock_o2(
    data: dict,
    *,
    hold_temp_C: int = 225,
    charT: int | None = 500,
    feedstocks: list[str] = ("BRF", "WS", "PW"),
    o2_order: list[str] = ("5%", "10%", "20%"),
    out_dir: Path,
    filename_stem: str | None = None,
    trim_start_min: float = 0.0,
    trim_end_min: float = 0.0,
    save_tex_snippet: bool = True,
) -> dict:
    """
    Builds a grid:
      rows = feedstocks
      cols = O2 levels

    Each cell: m/m0 vs time for isothermal hold at hold_temp_C.
    Saves PNG+PDF (and optional .tex snippet) into out_dir.

    Returns a dict with paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pick one char per feedstock (matching charT if provided)
    chosen = {}
    keys = list(data.keys())

    for fs in feedstocks:
        fsU = fs.upper()
        cands = []
        for k in keys:
            feed, ct = _parse_feedstock_and_charT(k)
            if feed == fsU and (charT is None or ct == int(charT)):
                cands.append(k)
        chosen[fsU] = cands[0] if cands else None

    # figure naming
    if filename_stem is None:
        if charT is None:
            filename_stem = f"iso_matrix_{int(hold_temp_C)}C"
        else:
            filename_stem = f"iso_matrix_{int(hold_temp_C)}C_charT{int(charT)}"

    png_path = out_dir / f"{filename_stem}.png"
    pdf_path = out_dir / f"{filename_stem}.pdf"
    tex_path = out_dir / f"{filename_stem}.tex"

    nrows = len(feedstocks)
    ncols = len(o2_order)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.6*ncols, 2.6*nrows), sharex=True, sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    # gather global limits for consistent axes
    all_tmax = []
    all_ymin = []

    curves = {}  # (i,j) -> (t, y) or None
    for i, fs in enumerate(feedstocks):
        fsU = fs.upper()
        char_key = chosen.get(fsU)
        for j, o2 in enumerate(o2_order):
            ax = axes[i, j]
            if not char_key:
                curves[(i, j)] = None
                continue

            char_data = data.get(char_key, {})
            reg = _find_isothermal_regime_key(char_data, int(hold_temp_C))
            if not reg:
                curves[(i, j)] = None
                continue

            df = char_data.get(reg, {}).get(o2, None)
            if not _is_df(df):
                curves[(i, j)] = None
                continue

            t_rel, m_norm = _norm_mass_curve(df, trim_start_min=trim_start_min, trim_end_min=trim_end_min)
            if t_rel is None:
                curves[(i, j)] = None
                continue

            curves[(i, j)] = (t_rel, m_norm)
            all_tmax.append(float(np.nanmax(t_rel)))
            all_ymin.append(float(np.nanmin(m_norm)))

    x_max = max(all_tmax) if all_tmax else 1.0
    y_min = min(all_ymin) if all_ymin else 0.95
    y_min = min(y_min, 0.999)  # ensure room

    # plot
    for i, fs in enumerate(feedstocks):
        fsU = fs.upper()
        char_key = chosen.get(fsU)

        for j, o2 in enumerate(o2_order):
            ax = axes[i, j]
            item = curves.get((i, j), None)

            if item is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0, x_max)
                ax.set_ylim(y_min, 1.01)
            else:
                t_rel, m_norm = item
                ax.plot(t_rel, m_norm)

                # small annotation: Δm/m0 (%)
                dm_pct = 100.0 * (1.0 - float(m_norm[-1]))
                ax.text(0.02, 0.95, f"$\\Delta m/m_0$={dm_pct:.1f}\\%",
                        transform=ax.transAxes, ha="left", va="top", fontsize=9)

                ax.set_xlim(0, x_max)
                ax.set_ylim(y_min, 1.01)

            # column titles
            if i == 0:
                ax.set_title(f"O$_2$ = {o2}")

            # row labels (feedstock + char key)
            if j == 0:
                if char_key:
                    ax.set_ylabel(f"{fsU}\n({char_key})")
                else:
                    ax.set_ylabel(f"{fsU}\n(no char)")

            # x-labels on bottom row only
            if i == nrows - 1:
                ax.set_xlabel("Time in hold (min)")

    fig.suptitle(f"Isothermal hold at {int(hold_temp_C)} $^\\circ$C: normalized mass $m/m_0$ vs time", y=1.02)
    fig.tight_layout()

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    if save_tex_snippet:
        # Use relative path if you \input from OUT_ROOT; adjust as you prefer
        cap = (f"Normalized mass $m/m_0$ vs time during isothermal holds at {int(hold_temp_C)} "
               f"$^\\circ$C for multiple feedstocks and oxygen levels. Each panel shows $m/m_0$ "
               f"with time shifted to the start of the hold.")
        lab = f"fig:iso-matrix-{int(hold_temp_C)}C" + (f"-charT{int(charT)}" if charT is not None else "")
        tex = "\n".join([
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.98\linewidth]{{{png_path.as_posix()}}}",
            rf"\caption{{{cap}}}",
            rf"\label{{{lab}}}",
            r"\end{figure}",
            ""
        ])
        tex_path.write_text(tex, encoding="utf-8")

    return {
        "png": png_path,
        "pdf": pdf_path,
        "tex": tex_path if save_tex_snippet else None,
        "chosen_chars": chosen,
    }

# -------------------------
# Basics: TG conversion curves
# -------------------------

def _ensure_basics_dirs(char: str, out_root: Path) -> dict[str, Path]:
    """Create out/{char}/basics/{figures,fits}."""
    base = Path(out_root) / str(char) / "basics"
    d = {
        "base": base,
        "figures": base / "figures",
        "fits": base / "fits",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _safe_stem(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "plot"


def _ash_fraction_for_char(char: str, default: float = 0.0) -> float:
    """Best-effort ash fraction lookup used for carbon conversion."""
    try:
        from tg_math import ASH_FRACTION_DEFAULTS  # type: ignore
        return float(ASH_FRACTION_DEFAULTS.get(str(char).upper(), default))
    except Exception:
        return float(default)


def _resolve_ash_fraction_like_tg_math(feedstock: str, fallback: float = 0.0) -> float:
    """
    Prefer tg_math._resolve_ash_fraction if present (keeps consistent with your pipeline),
    otherwise fall back to local defaults.
    """
    try:
        from tg_math import _resolve_ash_fraction  # type: ignore
        af = float(_resolve_ash_fraction(feedstock, None))
        if np.isfinite(af):
            return af
    except Exception:
        pass
    return _ash_fraction_for_char(feedstock, default=fallback)


def _compute_alpha_pct_and_x_c(
    df: pd.DataFrame,
    *,
    ash_fraction: float,
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    drop_before_mass_max: bool = True,
) -> pd.DataFrame | None:
    """
    Compute alpha_pct and x_c using YOUR definitions:

    alpha_pct:
      - baseline is the FIRST point of the used slice (after optional drop to mass max):
          m0 = mass_pct(first)
          alpha_pct(t) = 100 - (m0 - mass_pct(t))

      => alpha_pct starts at 100 and goes down as mass decreases.

    x_c:
      - baseline mass is max mass in the segment:
          mmax = max(mass_pct in segment)
          x_c(t) = (mmax - mass_pct(t)) / (mmax*(1-ash_fraction))

    Returns a time-series DataFrame with:
      time_min, time_rel_min, temp_C, mass_pct, alpha_pct, x_c
    """
    if df is None or df.empty:
        return None

    d = df.copy()
    if time_col not in d.columns or mass_col not in d.columns:
        return None

    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[mass_col] = pd.to_numeric(d[mass_col], errors="coerce")
    if temp_col in d.columns:
        d[temp_col] = pd.to_numeric(d[temp_col], errors="coerce")

    d = d.dropna(subset=[time_col, mass_col]).sort_values(time_col)
    if d.empty or len(d) < 3:
        return None

    t = d[time_col].to_numpy(float)
    m = d[mass_col].to_numpy(float)

    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(m)):
        mask = np.isfinite(t) & np.isfinite(m)
        t = t[mask]
        m = m[mask]
        if temp_col in d.columns:
            TT = d[temp_col].to_numpy(float)[mask]
        else:
            TT = None
        if len(t) < 3:
            return None
    else:
        TT = d[temp_col].to_numpy(float) if temp_col in d.columns else None

    # --- Use max mass within segment as the reference point for x_c,
    # and (optionally) drop everything before it so x_c starts at 0 like your tg_math plotting typically does.
    i_max = int(np.nanargmax(m))
    if drop_before_mass_max and i_max > 0:
        t = t[i_max:]
        m = m[i_max:]
        if TT is not None:
            TT = TT[i_max:]

    # time relative to first point of the used slice (which is at mass max if drop_before_mass_max=True)
    t_rel = t - float(t[0])

    # --- alpha_pct: first point becomes 100, then decreases by absolute mass loss (%-points)
    m0 = float(m[0])
    if not np.isfinite(m0):
        return None
    alpha_pct = 100.0 - (m0 - m)  # == 100 + m - m0

    # --- x_c: uses max mass in *the used slice*
    mmax = float(np.nanmax(m))
    denom = mmax * (1.0 - float(ash_fraction))
    if not np.isfinite(denom) or denom <= 0.0:
        # if ash is invalid, just avoid crash and return NaNs for x_c
        x_c = np.full_like(m, np.nan, dtype=float)
    else:
        x_c = (mmax - m) / denom

    out = pd.DataFrame(
        {
            "time_min": t,
            "time_rel_min": t_rel,
            "temp_C": TT if TT is not None else np.nan,
            "mass_pct": m,
            "alpha_pct": alpha_pct,
            "x_c": x_c,
        }
    )
    return out


def _longest_true_run(t: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    """Return (start_idx, end_idx) of the longest consecutive True-run in `mask` (by time duration)."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None

    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, splits + 1]
    ends = np.r_[splits, idx.size - 1]

    best = None
    best_dur = -np.inf
    for s, e in zip(starts, ends):
        i0 = int(idx[s])
        i1 = int(idx[e])
        dur = float(t[i1] - t[i0])
        if dur > best_dur:
            best_dur = dur
            best = (i0, i1)
    return best


def _slice_isothermal_hold(
    df: pd.DataFrame,
    *,
    target_temp_C: float,
    tol_C: float = 2.0,
    trim_start_min: float = 0.2,
    trim_end_min: float = 0.2,
    start_at_mass_peak: bool = True,
    peak_extra_start_min: float = 0.0,
    time_col: str = "time_min",
    temp_col: str = "temp_C",
    mass_col: str = "mass_pct",
    # --- NEW controls ---
    prefer_last_segment_first: bool = True,
    last_segment_temp_guard_C: float = 3.0,     # if |Tmed - target| > this => do NOT use last segment, try infer
    min_duration_min: float = 5.0,              # reject any candidate window shorter than this (after trimming/refine)
    min_points: int = 30,                       # reject any candidate window with fewer points
    debug: bool = False,
    ctx: str = "",
) -> tuple[pd.DataFrame | None, tuple[float, float] | None]:
    """
    Infer and slice the hold window of an isothermal experiment.

    Behavior (as requested):
      1) Try LAST segment first (if available), but ONLY accept if:
           - duration and points pass thresholds
           - and |T_median - target| <= last_segment_temp_guard_C
      2) If last segment is too short OR T_median is off, run infer_isothermal_time_window (strict then loose)
      3) If still none, fallback to longest temp-run mask near target

    Notes:
      - After picking a window, optional refine_window_to_mass_peak is applied.
      - If refine collapses the window, it falls back to the unrefined window.
    """
    from tg_helpers import infer_isothermal_time_window, refine_window_to_mass_peak  # local import

    if df is None or df.empty or time_col not in df.columns or temp_col not in df.columns:
        return (None, None)

    dsort = df.copy()
    dsort[time_col] = pd.to_numeric(dsort[time_col], errors="coerce")
    dsort[temp_col] = pd.to_numeric(dsort[temp_col], errors="coerce")
    if mass_col in dsort.columns:
        dsort[mass_col] = pd.to_numeric(dsort[mass_col], errors="coerce")

    dsort = dsort.dropna(subset=[time_col, temp_col]).sort_values(time_col)
    if dsort.empty or len(dsort) < 3:
        return (None, None)

    def _select_by_tw(tw: tuple[float, float]) -> pd.DataFrame:
        return dsort[(dsort[time_col] >= float(tw[0])) & (dsort[time_col] <= float(tw[1]))].copy()

    def _is_window_valid(tw: tuple[float, float]) -> tuple[bool, str]:
        if tw is None:
            return (False, "tw=None")
        if not (np.isfinite(tw[0]) and np.isfinite(tw[1])) or tw[1] <= tw[0]:
            return (False, "invalid tw bounds")
        sel = _select_by_tw(tw)
        if sel.shape[0] < int(min_points):
            return (False, f"too few points ({sel.shape[0]} < {min_points})")
        dur = float(sel[time_col].max() - sel[time_col].min())
        if dur < float(min_duration_min):
            return (False, f"too short duration ({dur:.3f} < {min_duration_min})")
        return (True, "ok")

    def _tw_from_df(seg_df: pd.DataFrame) -> tuple[float, float] | None:
        if seg_df is None or seg_df.empty or seg_df.shape[0] < 3:
            return None
        tmin = float(seg_df[time_col].min())
        tmax = float(seg_df[time_col].max())
        t0 = tmin + float(trim_start_min)
        t1 = tmax - float(trim_end_min)
        if t1 <= t0:
            return None
        return (t0, t1)

    tw: tuple[float, float] | None = None
    method: str | None = None

    # ------------------------------------------------------------------
    # 1) Prefer LAST segment first (but only accept if it's really the hold)
    # ------------------------------------------------------------------
    if prefer_last_segment_first and "segment" in dsort.columns:
        seg_series = dsort["segment"].dropna()
        if len(seg_series) > 0:
            last_seg = seg_series.iloc[-1]
            seg_df = dsort[dsort["segment"] == last_seg].copy()

            # compute median temperature of the segment
            tmed = float(pd.to_numeric(seg_df[temp_col], errors="coerce").median())

            # only accept segment if temp looks like the target
            if np.isfinite(tmed) and abs(tmed - float(target_temp_C)) <= float(last_segment_temp_guard_C):
                tw_cand = _tw_from_df(seg_df)
                if tw_cand is not None:
                    ok, why = _is_window_valid(tw_cand)
                    if ok:
                        tw = tw_cand
                        method = f"segment(last={last_seg},Tmed={tmed:.2f})"
                    elif debug:
                        print(f"[TG-BASICS][LAST-SEG-REJECT] {ctx} | Tmed={tmed:.2f} | {why} | tw={tw_cand}")
            else:
                if debug:
                    print(
                        f"[TG-BASICS][LAST-SEG-TMED-OFF] {ctx} | last_seg={last_seg} "
                        f"| Tmed={tmed:.2f} | target={target_temp_C} | guard={last_segment_temp_guard_C}"
                    )

    # ------------------------------------------------------------------
    # 2) If last-segment wasn't accepted, do infer(strict) then infer(loose)
    # ------------------------------------------------------------------
    if tw is None:
        # strict
        try:
            tw_cand = infer_isothermal_time_window(
                dsort,
                target_temp_C=float(target_temp_C),
                tol_C=float(tol_C),
                min_points=int(min_points),
                min_duration_min=float(min_duration_min),
                trim_start_min=float(trim_start_min),
                trim_end_min=float(trim_end_min),
                time_col=time_col,
                temp_col=temp_col,
                mass_col=mass_col,
            )
            ok, why = _is_window_valid(tw_cand)
            if ok:
                tw = tw_cand
                method = "infer(strict)"
            elif debug:
                print(f"[TG-BASICS][INFER-STRICT-REJECT] {ctx} | {why} | tw={tw_cand}")
        except Exception as e:
            if debug:
                print(f"[TG-BASICS][INFER-STRICT-FAIL] {ctx} | {repr(e)}")

    if tw is None:
        # loose
        try:
            tw_cand = infer_isothermal_time_window(
                dsort,
                target_temp_C=float(target_temp_C),
                tol_C=float(max(tol_C, 6.0)),
                min_points=max(10, int(min_points // 3)),
                min_duration_min=max(2.0, float(min_duration_min) / 2.0),
                trim_start_min=float(trim_start_min),
                trim_end_min=float(trim_end_min),
                time_col=time_col,
                temp_col=temp_col,
                mass_col=mass_col,
            )
            ok, why = _is_window_valid(tw_cand)
            if ok:
                tw = tw_cand
                method = "infer(loose)"
            elif debug:
                print(f"[TG-BASICS][INFER-LOOSE-REJECT] {ctx} | {why} | tw={tw_cand}")
        except Exception as e:
            if debug:
                print(f"[TG-BASICS][INFER-LOOSE-FAIL] {ctx} | {repr(e)}")

    # ------------------------------------------------------------------
    # 3) Fallback: longest contiguous run near target
    # ------------------------------------------------------------------
    if tw is None:
        t = dsort[time_col].to_numpy(float)
        T = dsort[temp_col].to_numpy(float)

        for thr in (max(float(tol_C), 10.0), 20.0):
            mask = np.isfinite(t) & np.isfinite(T) & (np.abs(T - float(target_temp_C)) <= float(thr))
            run = _longest_true_run(t, mask)
            if run is None:
                continue
            i0, i1 = run
            tmin = float(t[i0])
            tmax = float(t[i1])
            if (tmax - tmin) < 2.0:
                continue
            tw_cand = (tmin + float(trim_start_min), tmax - float(trim_end_min))
            ok, why = _is_window_valid(tw_cand)
            if ok:
                tw = tw_cand
                method = f"runmask(thr=±{thr:g}C)"
                break
            elif debug:
                print(f"[TG-BASICS][RUNMASK-REJECT] {ctx} | thr={thr} | {why} | tw={tw_cand}")

    if tw is None:
        if debug:
            print(f"[TG-BASICS][ISO-WINDOW-FAIL] {ctx} | target={target_temp_C}C")
        return (None, None)

    # ------------------------------------------------------------------
    # Optional refine start to mass peak (but don't allow it to collapse window)
    # ------------------------------------------------------------------
    tw_before_refine = tw
    if start_at_mass_peak:
        try:
            tw2 = refine_window_to_mass_peak(dsort, tw, extra_start_min=float(peak_extra_start_min))
            if tw2[1] > tw2[0]:
                ok, why = _is_window_valid(tw2)
                if ok:
                    tw = tw2
                else:
                    # keep old tw if refine makes it too short
                    tw = tw_before_refine
                    if debug:
                        print(f"[TG-BASICS][PEAK-REFINE-REJECT] {ctx} | {why} | tw2={tw2} | keep={tw_before_refine}")
        except Exception as e:
            if debug:
                print(f"[TG-BASICS][PEAK-REFINE-FAIL] {ctx} | {repr(e)}")

    sel = _select_by_tw(tw)
    if sel.empty or len(sel) < 3:
        if debug:
            print(f"[TG-BASICS][ISO-EMPTY] {ctx} | method={method} | tw={tw}")
        return (None, tw)

    if debug:
        tmed = float(pd.to_numeric(sel[temp_col], errors="coerce").median())
        dur = float(sel[time_col].max() - sel[time_col].min())
        print(f"[TG-BASICS][ISO-OK] {ctx} | method={method} | tw={tw} | n={sel.shape[0]} | dur={dur:.2f} | Tmed={tmed:.2f}")

    return (sel, tw)




def create_tg_graphs_for_char(
    char: str,
    char_data: dict,
    *,
    out_root: Path = Path("out"),
    conversion_basis: str = "both",
    figure_mode: str = "overlay",  # 'overlay' | 'separate' | 'both'
    o2_order: tuple[str, ...] = ("5%", "10%", "20%"),
    tol_C: float = 2.0,
    trim_start_min: float = 0.2,
    trim_end_min: float = 0.2,
    start_at_mass_peak: bool = True,
    peak_extra_start_min: float = 0.0,
    ramp_time_window: tuple[float, float] | None = None,
    debug: bool = False,
) -> dict:
    """
    Exports:
      - out/{char}/basics/fits/tg_conversions.csv              (SUMMARY in your requested format)
      - out/{char}/basics/fits/tg_conversions_timeseries.csv   (FULL timeseries)
      - out/{char}/basics/figures/*.png

    Summary columns (exactly):
      char,regime,o2,time_window,kind,total_alpha_conversion,total_Xc_conversion

    Definitions:
      total_alpha_conversion = max mass loss in the window (in %-points) = 100 - min(alpha_pct)
      total_Xc_conversion    = max(x_c) in the window (fraction)
    """
    dirs = _ensure_basics_dirs(char, Path(out_root))
    ash = _resolve_ash_fraction_like_tg_math(str(char), fallback=0.0)

    conv_choice = str(conversion_basis).strip().lower()
    if conv_choice not in {"alpha", "carbon", "both"}:
        conv_choice = "both"

    fig_mode = str(figure_mode).strip().lower()
    if fig_mode not in {"overlay", "separate", "both"}:
        fig_mode = "overlay"

    rows_ts: list[pd.DataFrame] = []
    rows_sum: list[dict] = {}
    rows_sum_list: list[dict] = []
    regime_series: dict[str, dict[str, pd.DataFrame]] = {}

    for regime, o2_map in (char_data or {}).items():
        if not isinstance(o2_map, dict):
            continue

        regime_key = str(regime)
        is_iso = "isotherm" in regime_key.lower()

        target_T = None
        if is_iso:
            mT = re.search(r"(\d{3})", regime_key)
            if mT:
                try:
                    target_T = float(int(mT.group(1)))
                except Exception:
                    target_T = None

        present = list(o2_map.keys())
        ordered = [k for k in o2_order if k in present] + [k for k in present if k not in o2_order]

        for o2_lab in ordered:
            df = o2_map.get(o2_lab, None)
            if not _is_df(df):
                continue

            df_use = None
            tw = None

            if is_iso and target_T is not None:
                ctx = f"{char} | {regime_key} | {o2_lab}"
                df_use, tw = _slice_isothermal_hold(
                    df,
                    target_temp_C=target_T,
                    tol_C=tol_C,
                    trim_start_min=trim_start_min,
                    trim_end_min=trim_end_min,
                    start_at_mass_peak=start_at_mass_peak,
                    peak_extra_start_min=peak_extra_start_min,
                    debug=debug,
                    ctx=ctx,
                )
                if df_use is None:
                    if debug:
                        print(f"[TG-BASICS][SKIP] {ctx}: could not infer hold window")
                    continue
            else:
                df_use = df.copy()
                if ramp_time_window is not None and len(ramp_time_window) == 2:
                    t0, t1 = float(ramp_time_window[0]), float(ramp_time_window[1])
                    df_use = df_use[(df_use["time_min"] >= t0) & (df_use["time_min"] <= t1)].copy()

            conv = _compute_alpha_pct_and_x_c(df_use, ash_fraction=ash, drop_before_mass_max=True)
            if conv is None or conv.empty:
                continue

            # annotate timeseries
            conv.insert(0, "char", str(char))
            conv.insert(1, "regime", regime_key)
            conv.insert(2, "o2", str(o2_lab))
            conv.insert(3, "ash_fraction", float(ash))
            conv.insert(4, "time_window", f"{tw[0]:.3f},{tw[1]:.3f}" if tw is not None else "")
            conv.insert(5, "kind", "isothermal" if is_iso else "linear")

            rows_ts.append(conv)
            regime_series.setdefault(regime_key, {})[str(o2_lab)] = conv

            # summary totals (per your required CSV)
            alpha_pct_min = float(np.nanmin(conv["alpha_pct"].to_numpy(float))) if "alpha_pct" in conv.columns else np.nan
            total_alpha_conv = 100.0 - alpha_pct_min if np.isfinite(alpha_pct_min) else np.nan

            total_xc = float(np.nanmax(conv["x_c"].to_numpy(float))) if "x_c" in conv.columns else np.nan

            rows_sum_list.append(
                {
                    "char": str(char),
                    "regime": regime_key,
                    "o2": str(o2_lab),
                    "time_window": f"{tw[0]:.3f},{tw[1]:.3f}" if tw is not None else "",
                    "kind": "isothermal" if is_iso else "linear",
                    "total_alpha_conversion": total_alpha_conv,
                    "total_Xc_conversion": total_xc,
                }
            )

    # --- export SUMMARY CSV (exact format requested) ---
    csv_summary_path = dirs["fits"] / "tg_conversions.csv"
    df_sum = pd.DataFrame(
        rows_sum_list,
        columns=[
            "char",
            "regime",
            "o2",
            "time_window",
            "kind",
            "total_alpha_conversion",
            "total_Xc_conversion",
        ],
    )
    df_sum.to_csv(csv_summary_path, index=False)

    # --- export FULL timeseries (audit + plot inputs) ---
    csv_ts_path = dirs["fits"] / "tg_conversions_timeseries.csv"
    if rows_ts:
        df_ts = pd.concat(rows_ts, ignore_index=True)
        df_ts.to_csv(csv_ts_path, index=False)
    else:
        pd.DataFrame().to_csv(csv_ts_path, index=False)

    # --- plots ---
    for regime_key, o2_map in regime_series.items():
        is_iso = "isotherm" in regime_key.lower()

        # x axis: time in hold (after drop_before_mass_max => starts at 0 at mass max)
        x_col = "time_rel_min"
        x_lab = "Time (min)"

        o2_keys = [k for k in o2_order if k in o2_map] + [k for k in o2_map.keys() if k not in o2_order]

        def _plot_overlay(y_col: str, y_lab: str, suffix: str):
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            drew = False
            for o2_lab in o2_keys:
                dfc = o2_map.get(o2_lab)
                if dfc is None or dfc.empty:
                    continue
                xx = pd.to_numeric(dfc.get(x_col, np.nan), errors="coerce").to_numpy(float)
                yy = pd.to_numeric(dfc.get(y_col, np.nan), errors="coerce").to_numpy(float)
                mask = np.isfinite(xx) & np.isfinite(yy)
                if mask.sum() < 3:
                    continue
                order = np.argsort(xx[mask])
                ax.plot(xx[mask][order], yy[mask][order], label=str(o2_lab))
                drew = True
            if not drew:
                plt.close(fig)
                return
            ax.set_xlabel(x_lab)
            ax.set_ylabel(y_lab)
            ax.set_title(f"{char} — {regime_key} ({suffix})")
            ax.legend(title="O$_2$", loc="best")
            fig.tight_layout()
            out = dirs["figures"] / f"{_safe_stem(regime_key)}_{suffix}.png"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)

        def _plot_separate(y_col: str, y_lab: str, suffix: str):
            for o2_lab in o2_keys:
                dfc = o2_map.get(o2_lab)
                if dfc is None or dfc.empty:
                    continue
                xx = pd.to_numeric(dfc.get(x_col, np.nan), errors="coerce").to_numpy(float)
                yy = pd.to_numeric(dfc.get(y_col, np.nan), errors="coerce").to_numpy(float)
                mask = np.isfinite(xx) & np.isfinite(yy)
                if mask.sum() < 3:
                    continue
                order = np.argsort(xx[mask])
                fig, ax = plt.subplots(figsize=(6.8, 4.8))
                ax.plot(xx[mask][order], yy[mask][order])
                ax.set_xlabel(x_lab)
                ax.set_ylabel(y_lab)
                ax.set_title(f"{char} — {regime_key} — {o2_lab} ({suffix})")
                fig.tight_layout()
                out = dirs["figures"] / f"{_safe_stem(regime_key)}__{_safe_stem(o2_lab)}_{suffix}.png"
                fig.savefig(out, dpi=200, bbox_inches="tight")
                plt.close(fig)

        # alpha basis = alpha_pct
        if conv_choice in {"alpha", "both"}:
            if fig_mode in {"overlay", "both"}:
                _plot_overlay("alpha_pct", r"$\alpha$ (%)", "alpha")
            if fig_mode in {"separate", "both"}:
                _plot_separate("alpha_pct", r"$\alpha$ (%)", "alpha")

        # carbon basis = x_c
        if conv_choice in {"carbon", "both"}:
            if fig_mode in {"overlay", "both"}:
                _plot_overlay("x_c", r"$X_C$ (-)", "carbon")
            if fig_mode in {"separate", "both"}:
                _plot_separate("x_c", r"$X_C$ (-)", "carbon")

    return {
        "char": char,
        "csv_summary": csv_summary_path,
        "csv_timeseries": csv_ts_path,
        "figures_dir": dirs["figures"],
        "fits_dir": dirs["fits"],
    }


def create_tg_graphs_all(
    data: dict,
    *,
    out_root: Path = Path("out"),
    conversion_basis: str = "both",
    figure_mode: str = "overlay",
    debug: bool = False,
) -> dict[str, dict]:
    """Run create_tg_graphs_for_char(...) for every char in the loaded `data` dict."""
    out: dict[str, dict] = {}
    for char, char_data in (data or {}).items():
        try:
            if debug:
                print(f"[TG-BASICS] {char}: exporting TG conversion curves")
            out[str(char)] = create_tg_graphs_for_char(
                str(char),
                char_data,
                out_root=Path(out_root),
                conversion_basis=conversion_basis,
                figure_mode=figure_mode,
                debug=debug,
            )
        except Exception as e:
            if debug:
                print(f"[TG-BASICS][ERROR] {char}: {e}")
            out[str(char)] = {"error": str(e)}
    return out

def _safe_name(s: str) -> str:
    s = s.replace("%", "pct")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")


def _last_segment_value(df: pd.DataFrame, seg_col: str = "segment") -> int | None:
    if seg_col not in df.columns:
        return None
    seg = pd.to_numeric(df[seg_col], errors="coerce").dropna()
    if seg.empty:
        return None
    return int(seg.max())


def create_tg_plots_last_segment_all(
    data: dict,
    *,
    out_root: str | Path,
    m0_mode: str = "start",  # "start" keeps initial point at 100%; "max" caps at 100%
    t0_mode: str = "start",  # "start" makes time_rel=0 at segment start
):
    out_root = Path(out_root) / "tg_plots_last_segment"
    out_root.mkdir(parents=True, exist_ok=True)

    for char, char_data in (data or {}).items():
        if not isinstance(char_data, dict):
            continue

        for regime, o2_map in (char_data or {}).items():
            if not isinstance(o2_map, dict):
                continue

            for o2_lab, df in (o2_map or {}).items():
                if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                    continue

                last_seg = _last_segment_value(df)
                if last_seg is None:
                    continue

                save_dir = out_root / _safe_name(str(char)) / _safe_name(str(regime))
                save_dir.mkdir(parents=True, exist_ok=True)

                fname = f"{char}__{regime}__{o2_lab}__seg{last_seg}.png"
                save_path = save_dir / _safe_name(fname)

                plot_tg_curve_time(
                    df,
                    segment=last_seg,          # <-- THIS is the “full last segment”
                    m0_mode=m0_mode,
                    t0_mode=t0_mode,
                    label=f"{o2_lab} (seg {last_seg})",
                    title=f"{char} | {regime} | {o2_lab} | segment {last_seg}",
                    show=False,
                    save_path=str(save_path),
                )

    print(f"[TG] Saved last-segment TG plots to: {out_root.resolve()}")