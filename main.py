from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib

from report_data_helper import run_char, _export_table, ReportConfig, plot_isothermal_matrix_feedstock_o2

matplotlib.use("Agg")
from tg_loader import load_all_thermogravimetric_data, SPEC


# -------------------------
# Paths
# -------------------------
ODIN_PATH = Path("./TG_Data/Odin Data/")
SIF_PATH = Path("./TG_Data/Sif Data/")
BASE_DIR = SIF_PATH  # <- change to ODIN_PATH if needed

OUT_ROOT = Path("out")


# -------------------------
# Global analysis config
# --------------------- ----
RAMP_TIME_WINDOW = (32.0, 195.0)   # min
BETA_K_PER_MIN = 3.0               # K/min (since your time axis is time_min)
N_SOLID = 1.0                      # 1st order in solid assumption

CR_WINDOWS = [
    ("CR_std_0p10_0p80", (0.10, 0.80)),
    ("CR_mid_0p05_0p20", (0.05, 0.20)),
    ("CR_early_0p00_0p06", (0.00, 0.06)),
]

COMPARE_CFG = dict(
    conversion_basis="carbon",
    enforce_common_conversion=True,
    common_per_temperature=True,
    start_at_mass_peak=True,
    common_hi_frac=0.90,
    min_common_hi=0.01,
    trim_start_min=0.2,
    trim_end_min=0.2,
)
COMPARE_CFG.update({
    "debug": True,
    "skip_on_error": True,
    "min_points_for_fit": 20,  # bump if you want stricter
})


DO_CR_FITS = True
DO_CR_WINDOW_SENSITIVITY = True
DO_CR_TO_ISOTHERMAL_TABLES_AND_PLOTS = True
DO_ISOTHERMAL_GLOBAL_BENCHMARK = True


def main():
    OUT_ROOT.mkdir(exist_ok=True, parents=True)

    data = load_all_thermogravimetric_data(BASE_DIR, SPEC)

    # Bundle report settings to avoid circular imports
    cfg = ReportConfig(
        out_root=OUT_ROOT,
        cr_windows=CR_WINDOWS,
        ramp_time_window=RAMP_TIME_WINDOW,
        n_solid=N_SOLID,
        beta_k_per_min=BETA_K_PER_MIN,
        compare_cfg=COMPARE_CFG,
        do_cr_fits=DO_CR_FITS,
        do_cr_window_sensitivity=DO_CR_WINDOW_SENSITIVITY,
        do_cr_to_isothermal_tables_and_plots=DO_CR_TO_ISOTHERMAL_TABLES_AND_PLOTS,
        do_isothermal_global_benchmark=DO_ISOTHERMAL_GLOBAL_BENCHMARK,
    )

    summary_rows = []

    for char in SPEC.keys():
        if char not in data:
            continue

        print(f"\n=== Running {char} ===")
        res = run_char(char, data[char], cfg=cfg)

        cr_std = res["cr_fits"][res["cr_std_name"]]
        row = {
            "char": char,
            "CR_window": res["cr_std_name"],
            "CR_Ea_kJ_per_mol": cr_std.E_A_J_per_mol / 1000.0,
            "CR_A_1_per_min": cr_std.A,
            "CR_m_o2": cr_std.m_o2,
            "CR_r2": cr_std.r2,
        }
        if res["iso_fit"] is not None:
            iso = res["iso_fit"]
            row.update({
                "ISO_Ea_kJ_per_mol": iso.E_A_J_per_mol / 1000.0,
                "ISO_A_1_per_min": iso.A,
                "ISO_n_o2": iso.n_o2,
                "ISO_r2": iso.r2,
            })
        summary_rows.append(row)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).sort_values("char").reset_index(drop=True)
        _export_table(df_summary, OUT_ROOT / "summary_fit_params.csv", OUT_ROOT / "summary_fit_params.tex")

    print("\nDone. Results written to:", OUT_ROOT.resolve())

    matrix_out = plot_isothermal_matrix_feedstock_o2(
        data,
        hold_temp_C=225,
        charT=500,  # set None if you want "best available per feedstock"
        feedstocks=["BRF", "WS", "PW"],
        o2_order=["5%", "10%", "20%"],
        out_dir=OUT_ROOT / "figures",
        trim_start_min=0.2,
        trim_end_min=0.2,
        save_tex_snippet=True,
    )
    print("[Matrix] saved:", matrix_out["png"])
    print("[Matrix] chosen chars:", matrix_out["chosen_chars"])


if __name__ == "__main__":
    main()
