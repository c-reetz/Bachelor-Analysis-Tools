from tg_loader import load_thermogravimetric_data
from tg_math import estimate_segment_rate, estimate_arrhenius_from_segments, arrhenius_plot_data
from tg_plotting import plot_ln_r_vs_time, plot_arrhenius, plot_arrhenius_groups

ODIN_PATH = "./TG_Data/Odin Data/"
SIF_PATH = "./TG_Data/Sif Data/"

#########
# DATA LOADING todo: find better data loading scheme
#########

# BRF
bfr200 = load_thermogravimetric_data(f"{ODIN_PATH}200C, 20% O2, Isothermal 4H, Odin/ExpDat_BRF 500C.xlsx")
bfr225 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 4 - ISOTHERMAL 225C 20% O2/ExpDat_BRF.xlsx")
bfr250 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 3 - ISOTHERMAL 250C 20% O2/ExpDat_BRF 500.xlsx")

# WS
ws200 = load_thermogravimetric_data(f"{ODIN_PATH}200C, 20% O2, Isothermal 4H, Odin/ExpDat_WS 500C.xlsx")
ws225 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 4 - ISOTHERMAL 225C 20% O2/ExpDat_WS.xlsx")
ws250 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 3 - ISOTHERMAL 250C 20% O2/ExpDat_WS500.xlsx")

# PW
pw200 = load_thermogravimetric_data(f"{ODIN_PATH}200C, 20% O2, Isothermal 4H, Odin/ExpDat_PW 500C.xlsx")
pw225 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 4 - ISOTHERMAL 225C 20% O2/ExpDat_PW.xlsx")
pw250 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 3 - ISOTHERMAL 250C 20% O2/ExpDat_PW500.xlsx")

#####
# Parse time segments todo: better scheme
#####

# BRF
seg_bfr_200 = estimate_segment_rate(bfr200, time_window=(63, 290), label="BRF 200C")
seg_bfr_225 = estimate_segment_rate(bfr225, time_window=(88, 290), label="BRF 225C")
seg_bfr_250 = estimate_segment_rate(bfr250, time_window=(90, 290), label="BRF 250C")

# WS
seg_ws_200 = estimate_segment_rate(ws200, time_window=(63, 290), label="WS 200C")
seg_ws_225 = estimate_segment_rate(ws225, time_window=(88, 290), label="WS 225C")
seg_ws_250 = estimate_segment_rate(ws250, time_window=(90, 290), label="WS 250C")

# PW
seg_pw_200 = estimate_segment_rate(pw200, time_window=(63, 290), label="PW 200C")
seg_pw_225 = estimate_segment_rate(pw225, time_window=(88, 290), label="PW 225C")
seg_pw_250 = estimate_segment_rate(pw250, time_window=(90, 154), label="PW 250C")

####
# Eh, can do, 1st. deriv. - annoying to look at
####
#plot_ln_r_vs_time(df200, time_window=(63, 290), label="200C", overlay_constant_r=seg200.r_abs, save_path="lnr_vs_t_200C.png")

groups = [
    {"label": "BRF Biochar 500C", "segments": [seg_bfr_200, seg_bfr_225, seg_bfr_250]},
    {"label": "WS Biochar 500C", "segments": [seg_ws_200, seg_ws_225, seg_ws_250]},
    {"label": "PW Biochar 500C", "segments": [seg_pw_200, seg_pw_225, seg_pw_250]},
]

group_results = plot_arrhenius_groups(
    groups,
    save_path="arrhenius_groups.png",
    title="Arrhenius (ln r vs 1/T) – by product",
)


res = estimate_arrhenius_from_segments([seg_pw_200, seg_pw_225, seg_pw_250])
x, y = arrhenius_plot_data([seg_pw_200, seg_pw_225, seg_pw_250])
plot_arrhenius(x, y, slope=res.slope, intercept=res.intercept, save_path="pw_arr.png")

for r in group_results:
    print(
        f"{r['label']}: Ea = {r['Ea_kJ_per_mol']:.2f} kJ/mol, "
        f"A = {r['A']:.3g} [1/time], R² = {r['R2']:.3f}, n = {r['n_points']}"
    )