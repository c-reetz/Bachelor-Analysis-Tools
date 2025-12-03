from tg_loader import load_thermogravimetric_data
from tg_math import estimate_segment_rate_zero_order, estimate_segment_rate_first_order, \
    estimate_arrhenius_from_segments, arrhenius_plot_data, estimate_EA_A_nonisothermal_coats_redfern, \
    estimate_EA_A_nonisothermal_coats_redfern_global
from tg_plotting import plot_ln_r_vs_time, plot_arrhenius, plot_arrhenius_groups, plot_coats_redfern_overlays, \
    plot_coats_redfern_global

ODIN_PATH = "./TG_Data/Odin Data/"
SIF_PATH = "./TG_Data/Sif Data/"

#########
# DATA LOADING todo: find better data loading scheme
#########

# BRF
bfr200 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 6 - OXIDATION 600C 10% O2/ExpDat_BRF500.xlsx")
bfr225 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 7 . OXIDATION 600C 5% O2/ExpDat_BRF500.xlsx") #useless
bfr250 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 8 - OXIDATION 600C 20% O2/ExpDat_BRF500.xlsx")

# WS
ws200 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 6 - OXIDATION 600C 10% O2/ExpDat_WS500.xlsx")
ws225 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 7 . OXIDATION 600C 5% O2/ExpDat_WS500.xlsx")
ws250 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 8 - OXIDATION 600C 20% O2/ExpDat_WS500.xlsx")

# PW
pw200 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 6 - OXIDATION 600C 10% O2/ExpDat_PW500.xlsx")
pw225 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 7 . OXIDATION 600C 5% O2/ExpDat_PW500.xlsx")
pw250 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 8 - OXIDATION 600C 20% O2/ExpDat_PW500.xlsx")

#####
# Parse time segments todo: better scheme
#####

# BRF
seg_bfr_200 = estimate_EA_A_nonisothermal_coats_redfern(bfr200, time_window=(32, 195.5), label="BRF 10%")
#seg_bfr_225 = estimate_EA_A_nonisothermal_coats_redfern(bfr225, time_window=(88, 290), label="BRF 5%")
seg_bfr_250 = estimate_EA_A_nonisothermal_coats_redfern(bfr250, time_window=(32, 195), label="BRF 20%")

# WS
seg_ws_200 = estimate_EA_A_nonisothermal_coats_redfern(ws200, time_window=(32, 195), label="WS 10%")
seg_ws_225 = estimate_EA_A_nonisothermal_coats_redfern(ws225, time_window=(32, 195), label="WS 5%")
seg_ws_250 = estimate_EA_A_nonisothermal_coats_redfern(ws250, time_window=(32, 195), label="WS 20%")

# PW
seg_pw_200 = estimate_EA_A_nonisothermal_coats_redfern(pw200, time_window=(32, 195), label="PW 10%")
seg_pw_225 = estimate_EA_A_nonisothermal_coats_redfern(pw225, time_window=(32, 195), label="PW 5%")
seg_pw_250 = estimate_EA_A_nonisothermal_coats_redfern(pw250, time_window=(32, 195), label="PW 20%")

res_global = estimate_EA_A_nonisothermal_coats_redfern_global(
    [pw200, pw225, pw250],
    time_window=(32, 195),
    n_solid=1.0,
    alpha_range=(0.10, 0.80),
    beta_fixed_K_per_time=3.0,   # 3 K/min
    labels=["PW 10% O2","PW 5% O2","PW 20% O2"],
)


#plot_coats_redfern_overlays([seg_bfr_200, seg_bfr_250], save_path="cr_brf_overlay.png")
#plot_coats_redfern_overlays([seg_ws_200, seg_ws_225, seg_ws_250], save_path="cr_ws_overlay.png")
#plot_coats_redfern_overlays([seg_pw_200, seg_pw_225, seg_pw_250], save_path="cr_pw_overlay.png")

plot_coats_redfern_global(
    res_global,
    title="Global Coats–Redfern fit (3 °C/min)",
    save_path="cr_global.png",
)