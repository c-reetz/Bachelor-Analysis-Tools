from pathlib import Path

from tg_helpers import print_global_cr_o2_result, compare_cr_to_char_isothermals
from tg_loader import load_all_thermogravimetric_data, SPEC
from tg_math import estimate_global_coats_redfern_with_o2
from tg_plotting import plot_global_coats_redfern_o2_fit

ODIN_PATH = "./TG_Data/Odin Data/"
SIF_PATH = "./TG_Data/Sif Data/"

#########
# DATA LOADING todo: find better data loading scheme
#########

data = load_all_thermogravimetric_data(Path(SIF_PATH), SPEC)

#######
# LOAD BRF ISOTHERMAL DATA
#######
# Isothermal, 225C
brf_5o2_225C = data["BRF"]["isothermal_225"]["5%"]
brf_10o2_225C = data["BRF"]["isothermal_225"]["10%"]
brf_20o2_225c = data["BRF"]["isothermal_225"]["20%"]

# Isothermal, 250C
brf_5o2_250C = ""
brf_10o2_250C = ""
brf_20o2_250C = data["BRF"]["isothermal_250"]["20%"]

#Linear Heating Ramps
brf_5o2_linear = data["BRF"]["linear"]["5%"]
brf_10o2_linear = data["BRF"]["linear"]["10%"]
brf_20o2_linear = data["BRF"]["linear"]["20%"]

#######
# LOAD WS DATA
#######
# Isothermal, 225C
ws_5o2_225C = data["WS"]["isothermal_225"]["5%"]
ws_10o2_225C = data["WS"]["isothermal_225"]["10%"]
ws_20o2_225c = data["WS"]["isothermal_225"]["20%"]

# Isothermal, 250C
ws_5o2_250C = ""
ws_10o2_250C = ""
ws_20o2_250C = data["WS"]["isothermal_250"]["20%"]

#Linear Heating Ramps
ws_5o2_linear = data["WS"]["linear"]["5%"]
ws_10o2_linear = data["WS"]["linear"]["10%"]
ws_20o2_linear = data["WS"]["linear"]["20%"]

#######
# LOAD PW DATA
#######
# Isothermal, 225C
pw_5o2_225C = data["PW"]["isothermal_225"]["5%"]
pw_10o2_225C = data["PW"]["isothermal_225"]["10%"]
pw_20o2_225c = data["PW"]["isothermal_225"]["20%"]

# Isothermal, 250C
pw_5o2_250C = ""
pw_10o2_250C = ""
pw_20o2_250C = data["PW"]["isothermal_250"]["20%"]

#Linear Heating Ramps
pw_5o2_linear = data["PW"]["linear"]["5%"]
pw_10o2_linear = data["PW"]["linear"]["10%"]
pw_20o2_linear = data["PW"]["linear"]["20%"]

#####
# Parse Segments
#####

# global fit BRF
res_global_fit_brf = estimate_global_coats_redfern_with_o2(
    [brf_10o2_linear, brf_20o2_linear],
    o2_fractions=[0.10, 0.20],
    time_window=(32.0, 195.0),      # ramp region
    n_solid=1.0,                   # 1st order in solid assumption
    conversion_basis="carbon",
    conversion_range=(0.10, 0.80),
    feedstock="BRF",
    #alpha_range=(0.20, 0.50),
    beta_fixed_K_per_time=3.0,     # 3 K/min (since time_min)
    label="BRF linear heating ramps global O2 fit",
)
print_global_cr_o2_result(res_global_fit_brf)
plot_global_coats_redfern_o2_fit(res_global_fit_brf, save_path="brf_global_cr", title="BRF global CR fit")


# global fit WS
res_global_fit_ws = estimate_global_coats_redfern_with_o2(
    [ws_5o2_linear, ws_10o2_linear],
    o2_fractions=[0.05, 0.10],
    time_window=(32.0, 195.0),      # ramp region
    n_solid=1.0,                   # 1st order in solid assumption
    #alpha_range=(0.20, 0.50),
    beta_fixed_K_per_time=3.0,     # 3 K/min (since time_min)
    label="WS linear heating ramps global O2 fit",
    conversion_basis="carbon",
    conversion_range=(0.10, 0.80),
    feedstock="WS",
)
print_global_cr_o2_result(res_global_fit_ws)
plot_global_coats_redfern_o2_fit(res_global_fit_ws, save_path="ws_global_cr", title="WS global CR fit")


# global fit PW
res_global_fit_pw = estimate_global_coats_redfern_with_o2(
    [pw_5o2_linear, pw_10o2_linear, pw_20o2_linear],
    o2_fractions=[0.05, 0.10, 0.20],
    time_window=(32.0, 195.0),      # ramp region
    n_solid=1.0,                   # 1st order in solid assumption
    #alpha_range=(0.20, 0.50),
    beta_fixed_K_per_time=3.0,     # 3 K/min (since time_min)
    label="PW linear heating ramps global O2 fit",
    conversion_basis="carbon",
    conversion_range=(0.10, 0.80),
    feedstock="PW",
)
print_global_cr_o2_result(res_global_fit_pw)
plot_global_coats_redfern_o2_fit(res_global_fit_pw, save_path="pw_global_cr", title="PW global CR fit")

# Example usage:
# res = estimate_global_coats_redfern_with_o2(...)
# print_global_cr_o2_result(res)

tbl_brf = compare_cr_to_char_isothermals(
    res_global_fit_brf,
    data["BRF"],
    char_name="BRF",
    conversion_basis="carbon",
    alpha_range=(0.0, 1.0),
    trim_start_min=0.2,
    trim_end_min=0.2,
)
print(tbl_brf[["T_C","yO2","k_iso_1_per_min","k_CR_pred_1_per_min","CR/ISO_ratio","percent_error_%"]])



# BRF
"""
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

res = estimate_global_coats_redfern_with_o2(
    [pw225, pw200, pw250],
    o2_fractions=[0.05, 0.10, 0.20],
    time_window=(32.0, 195.0),      # ramp region
    n_solid=1.0,                   # 1st order in solid assumption
    alpha_range=(0.10, 0.20),
    beta_fixed_K_per_time=3.0,     # 3 K/min (since time_min)
    label="PW ramps global O2 fit",
)

print("Ea [kJ/mol]:", res.E_A_J_per_mol/1000)
print("A  [1/min]: ", res.A)
print("m_O2 order:", res.m_o2)
print("R²:", res.r2)

plot_global_coats_redfern_o2_fit(res, save_path="pw_global_cr", title="PW global CR fit")

df = pw250
t0 = 32.0
t1 = 195.0 # same time window
seg = df[(df["time_min"] >= t0) & (df["time_min"] <= t1)].copy()

t = seg["time_min"].to_numpy(dtype=float)
T = seg["temp_C"].to_numpy(dtype=float)
m_exp = seg["mass_pct"].to_numpy(dtype=float)

# estimate m0 and m_inf in THIS window (same as you do elsewhere)
N = len(m_exp)
k_head = max(3, int(round(0.10 * N)))
k_tail = max(3, int(round(0.20 * N)))
m0 = float(np.nanmedian(m_exp[:k_head]))
m_inf = float(np.nanmedian(m_exp[-k_tail:]))

# simulate alpha(t) using your global fitted parameters (res = global fit result)
alpha_sim = simulate_alpha_ramp(
    time_min=t,
    temp_C=T,
    yO2=0.2,
    E_A_J_per_mol=res.E_A_J_per_mol,
    A=res.A,
    m_o2=res.m_o2,
    solid_order=1,          # match n_solid you assumed
    alpha0=0.0,
)

# convert simulated alpha -> mass%
m_sim = alpha_to_mass_pct(alpha_sim, m0=m0, m_inf=m_inf, loss=True)

# overlay plot
plt.figure()
plt.plot(t, m_exp, "o", label="exp mass%")
plt.plot(t, m_sim, "-", label="sim mass%")
plt.xlabel("time [min]")
plt.ylabel("mass [%]")
plt.legend()
plt.tight_layout()
#plt.savefig("pw_mass_overlay.png", dpi=150)
plt.close()
"""