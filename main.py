from pathlib import Path

from tg_helpers import print_global_cr_o2_result
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
brf_5o2_250C = data["BRF"]["isothermal_250"]["5%"]
brf_10o2_250C = data["BRF"]["isothermal_250"]["10%"]
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
ws_5o2_250C = data["WS"]["isothermal_250"]["5%"]
ws_10o2_250C = data["WS"]["isothermal_250"]["10%"]
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
pw_5o2_250C = data["PW"]["isothermal_250"]["5%"]
pw_10o2_250C = data["PW"]["isothermal_250"]["10%"]
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
    [brf_5o2_linear, brf_10o2_linear, brf_20o2_linear],
    o2_fractions=[0.05, 0.10, 0.20],
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
    [ws_5o2_linear, ws_10o2_linear, ws_20o2_linear],
    o2_fractions=[0.05, 0.10, 0.2],
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


# BRF
"""
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